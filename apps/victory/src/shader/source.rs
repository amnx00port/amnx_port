//! Scan without keeping the whole archive or a file-wide index in memory.

use std::borrow::Cow;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use victory_game::sgd5;
use victory_shader::Container;

const CHUNK: usize = 1 << 20;
const BATCH_BYTES: usize = 16 << 20;
const BATCH_COUNT: usize = 1024;
// Keep the old limit on a single encoded shader, independent of file size.
const MAX_SOURCE: usize = 2 << 30;
// A one-byte name length, up to 254 name bytes, NUL, kind and size.
const RECORD_PREFIX: u64 = 261;

pub(crate) enum Source {
    Raw {
        offset: u64,
        data: Vec<u8>,
    },
    Packed {
        offset: u64,
        data: Vec<u8>,
        marker: usize,
    },
}

impl Source {
    pub(crate) fn bytes(&self) -> Option<Cow<'_, [u8]>> {
        match self {
            Self::Raw { data, .. } => Some(Cow::Borrowed(data)),
            Self::Packed { data, marker, .. } => {
                sgd5::record(data, *marker)?.unpack().map(Cow::Owned)
            }
        }
    }

    pub(crate) fn label(&self) -> String {
        match self {
            Self::Raw { offset, .. } => format!("@{offset:#x}"),
            Self::Packed {
                offset,
                data,
                marker,
            } => {
                let name = sgd5::record(data, *marker).and_then(|r| r.name);
                format!("@{offset:#x} {}", name.unwrap_or("?"))
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Raw { data, .. } | Self::Packed { data, .. } => data.len(),
        }
    }
}

pub(crate) fn sources(path: &Path, visit: impl FnMut(&[Source])) -> io::Result<()> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    scan_sources(&mut file, len, CHUNK, visit)
}

fn scan_sources(
    file: &mut (impl Read + Seek),
    len: u64,
    chunk: usize,
    mut visit: impl FnMut(&[Source]),
) -> io::Result<()> {
    let mut scan = vec![0; (chunk as u64 + 3).min(len) as usize];
    let mut sgd = false;
    scan_magic(file, len, &mut scan, chunk, sgd5::MAGIC, |_, _| {
        sgd = true;
        Ok(false)
    })?;
    let mut batch = Vec::new();
    let mut bytes = 0;
    let mut push = |source: Source| {
        if !batch.is_empty()
            && (source.len() > BATCH_BYTES.saturating_sub(bytes) || batch.len() == BATCH_COUNT)
        {
            visit(&batch);
            batch.clear();
            bytes = 0;
        }
        bytes += source.len();
        batch.push(source);
    };
    if sgd {
        scan_magic(
            file,
            len,
            &mut scan,
            chunk,
            &sgd5::MARKER,
            |file, offset| {
                if let Some(source) = packed(file, len, offset)? {
                    push(source);
                }
                Ok(true)
            },
        )?;
    }
    scan_magic(file, len, &mut scan, chunk, b"DXBC", |file, offset| {
        if let Some(source) = raw(file, len, offset)? {
            push(source);
        }
        Ok(true)
    })?;
    if !batch.is_empty() {
        visit(&batch);
    }
    Ok(())
}

/// Read three lookahead bytes so a four-byte marker can start at the end
/// of a chunk. Report only positions owned by that chunk.
fn scan_magic<F: Read + Seek>(
    file: &mut F,
    len: u64,
    scan: &mut [u8],
    chunk: usize,
    magic: &[u8; 4],
    mut visit: impl FnMut(&mut F, u64) -> io::Result<bool>,
) -> io::Result<()> {
    let finder = memchr::memmem::Finder::new(magic);
    for start in (0..len).step_by(chunk) {
        let size = (len - start).min(scan.len() as u64) as usize;
        read_at(file, start, &mut scan[..size])?;
        for local in finder.find_iter(&scan[..size]).take_while(|&at| at < chunk) {
            if !visit(file, start + local as u64)? {
                return Ok(());
            }
        }
    }
    Ok(())
}

fn raw(file: &mut (impl Read + Seek), len: u64, offset: u64) -> io::Result<Option<Source>> {
    let mut header = [0; 32];
    if len - offset < header.len() as u64 {
        return Ok(None);
    }
    read_at(file, offset, &mut header)?;
    let Ok(size) = Container::declared_len(&header) else {
        return Ok(None);
    };
    if size > MAX_SOURCE || size as u64 > len - offset {
        return Ok(None);
    }
    let mut data = vec![0; size];
    read_at(file, offset, &mut data)?;
    Ok(Container::parse(&data)
        .is_ok()
        .then_some(Source::Raw { offset, data }))
}

fn packed(file: &mut (impl Read + Seek), len: u64, offset: u64) -> io::Result<Option<Source>> {
    let mut header = [0; 16];
    if offset < 6 || len - offset < 12 {
        return Ok(None);
    }
    read_at(file, offset - 4, &mut header)?;
    let Some(size) = sgd5::encoded_len(&header) else {
        return Ok(None);
    };
    if size > MAX_SOURCE || size as u64 > len - offset {
        return Ok(None);
    }
    let start = offset.saturating_sub(RECORD_PREFIX);
    let marker = (offset - start) as usize;
    let mut data = vec![0; marker + size];
    read_at(file, start, &mut data)?;
    Ok(sgd5::record(&data, marker)
        .is_some()
        .then_some(Source::Packed {
            offset,
            data,
            marker,
        }))
}

fn read_at(file: &mut (impl Read + Seek), offset: u64, data: &mut [u8]) -> io::Result<()> {
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(data)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn container() -> [u8; 32] {
        let mut bytes = [0; 32];
        bytes[..4].copy_from_slice(b"DXBC");
        bytes[24..28].copy_from_slice(&32u32.to_le_bytes());
        bytes
    }

    fn compressed_record(name: &str, data: &[u8]) -> Vec<u8> {
        // A single final, uncompressed DEFLATE block. The record still
        // runs through the same inflater as a normally compressed block.
        let length = data.len() as u16;
        let size_from_marker = 12 + 5 + u32::from(length);
        let mut record = vec![(name.len() + 1) as u8];
        record.extend_from_slice(name.as_bytes());
        record.extend_from_slice(&[0, 0x3c]);
        record.extend_from_slice(&size_from_marker.to_le_bytes());
        record.extend_from_slice(&sgd5::MARKER);
        record.extend_from_slice(&size_from_marker.to_le_bytes());
        record.extend_from_slice(&u32::from(length).to_le_bytes());
        record.push(1);
        record.extend_from_slice(&length.to_le_bytes());
        record.extend_from_slice(&(!length).to_le_bytes());
        record.extend_from_slice(data);
        record
    }

    #[test]
    fn finds_each_marker_once_across_chunk_boundaries() {
        const CHUNK: usize = 19;
        let data = container();
        let name = "A".repeat(254);
        let record = compressed_record(&name, &[1, 2, 3]);
        // Move every marker through every position in a chunk.
        for prefix in 0..CHUNK {
            let mut archive = vec![0; prefix];
            let raw_at = archive.len();
            archive.extend_from_slice(&data);
            let packed_at = archive.len() + name.len() + 7;
            archive.extend_from_slice(&record);
            // The global marker can be anywhere, including after records.
            archive.extend_from_slice(sgd5::MAGIC);
            archive.extend_from_slice(b"DXBC"); // incomplete raw header
            archive.extend_from_slice(&sgd5::MARKER); // incomplete record
            let mut found = Vec::new();
            let len = archive.len() as u64;
            scan_sources(&mut Cursor::new(archive), len, CHUNK, |batch| {
                found.extend(
                    batch
                        .iter()
                        .map(|s| (s.label(), s.bytes().unwrap().into_owned())),
                );
            })
            .unwrap();
            assert_eq!(
                found,
                [
                    (format!("@{packed_at:#x} {name}"), vec![1, 2, 3]),
                    (format!("@{raw_at:#x}"), data.to_vec()),
                ]
            );
        }
    }

    #[test]
    fn visits_large_source_sets_in_bounded_batches() {
        let data = container().repeat(2 * BATCH_COUNT + 1);
        let len = data.len() as u64;
        let mut counts = Vec::new();
        scan_sources(&mut Cursor::new(data), len, CHUNK, |batch| {
            counts.push(batch.len())
        })
        .unwrap();
        assert_eq!(counts, [BATCH_COUNT, BATCH_COUNT, 1]);
    }
}
