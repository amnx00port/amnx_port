//! `SGD5`: the engine's shader file. Each record holds one shader container,
//! deflate-packed. The header tables are not decoded yet, so records are
//! found by their marker and cross-checked by their sizes:
//!
//! ```text
//! u8 name length (with NUL), name, NUL, u8 kind,
//! u32 n, u32 0x000DDCE7, u32 n, u32 unpacked size, raw deflate (n - 12 bytes)
//! ```

pub const MAGIC: &[u8; 4] = b"SGD5";
const MARKER: [u8; 4] = 0x000D_DCE7_u32.to_le_bytes();
/// Containers are at most a few MiB; refuse to allocate for absurd sizes.
const MAX_UNPACKED: usize = 64 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record<'a> {
    /// Offset of the marker in the file.
    pub offset: usize,
    /// Like `PS_<name>_<hash>_Wrapped`; `None` if the bytes before the
    /// record do not read as one.
    pub name: Option<&'a str>,
    /// Record type, meaning unknown (0x32 most shaders, 0x41 mesh shaders
    /// and some material families, 0x3c compute, 0x40, 0x42).
    pub kind: u8,
    packed: &'a [u8],
    unpacked_size: usize,
}

impl Record<'_> {
    /// The container, or `None` if the stream is corrupt.
    pub fn unpack(&self) -> Option<Vec<u8>> {
        miniz_oxide::inflate::decompress_to_vec_with_limit(self.packed, self.unpacked_size)
            .ok()
            .filter(|out| out.len() == self.unpacked_size)
    }

    /// The file bytes from the marker to the end of the packed stream.
    pub fn span(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + 12 + self.packed.len()
    }
}

/// Whether `offset` lies inside one of `records`, taken in the order
/// [`records`] yields them (file order, disjoint). Bytes there belong to a
/// record: a container stored as is inside its packed stream (deflate stored
/// blocks) is that record's container, not another one.
pub fn in_record(records: &[Record<'_>], offset: usize) -> bool {
    let i = records.partition_point(|r| r.span().end <= offset);
    records.get(i).is_some_and(|r| r.span().contains(&offset))
}

/// An SGD5 file, or a file holding one (an archive: records are found by
/// their marker, so no index is needed).
pub fn is_sgd5(file: &[u8]) -> bool {
    memchr::memmem::find(file, MAGIC).is_some()
}

/// Every record in an SGD5 file, in file order. The search starts at the
/// first magic: bytes before it (an archive's other files) belong to no SGD5
/// file, whatever markers they happen to hold. It resumes after each
/// record's packed bytes: a marker inside them (stored deflate blocks carry a
/// container's bytes as they are) is part of that record, and so is any part
/// of the next record's header or name, which must follow them.
pub fn records(file: &[u8]) -> impl Iterator<Item = Record<'_>> {
    let finder = memchr::memmem::Finder::new(&MARKER);
    // Where the next record may start: after the magic, then after each
    // record's packed bytes. No magic: past the end, so nothing is found.
    let mut floor = memchr::memmem::find(file, MAGIC).map_or(file.len(), |m| m + MAGIC.len());
    // A marker needs its name's NUL, the kind and `n` (6 bytes) before it.
    let mut from = floor + 6;
    std::iter::from_fn(move || {
        while let Some(found) = finder.find(file.get(from..)?) {
            let at = from + found;
            if let Some(r) = record(&file[floor..], at - floor) {
                let r = Record { offset: at, ..r };
                floor = r.span().end;
                from = floor + 6;
                return Some(r);
            }
            from = at + 1;
        }
        None
    })
}

/// The record whose marker is at `at`, reading nothing before `file`.
fn record(file: &[u8], at: usize) -> Option<Record<'_>> {
    let u32_at = |o: usize| {
        file.get(o..o + 4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize)
    };
    let n = u32_at(at.checked_sub(4)?)?;
    let unpacked_size = u32_at(at + 8)?;
    if u32_at(at + 4)? != n || n < 12 || unpacked_size > MAX_UNPACKED {
        return None;
    }
    let nul = at.checked_sub(6)?;
    Some(Record {
        offset: at,
        name: (file[nul] == 0).then(|| name_before(file, nul)).flatten(),
        kind: file[at - 5],
        packed: file.get(at + 12..at + n)?,
        unpacked_size,
    })
}

/// The printable name that ends at `nul` and follows its length byte.
fn name_before(file: &[u8], nul: usize) -> Option<&str> {
    let run = file[..nul]
        .iter()
        .rposition(|b| !b.is_ascii_graphic())
        .map_or(0, |p| p + 1);
    // From 32 characters on, the length byte is printable too.
    (run.saturating_sub(1)..nul)
        .find(|&l| usize::from(file[l]) == nul - l)
        .and_then(|l| std::str::from_utf8(&file[l + 1..nul]).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record_bytes(name: &str, kind: u8, payload: &[u8]) -> Vec<u8> {
        let packed = miniz_oxide::deflate::compress_to_vec(payload, 6);
        let n = (packed.len() + 12) as u32;
        let mut out = vec![0xF4, 0x01, 0, 0, name.len() as u8 + 1];
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&[0, kind]);
        for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&packed);
        out
    }

    #[test]
    fn reads_records() {
        let long = format!("PS_{}_Wrapped", "x".repeat(40));
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(&[1, 2, 3, 4]);
        file.extend(record_bytes(
            "VS_Blit_0011aabb_Wrapped",
            0x32,
            b"first container",
        ));
        // A marker whose sizes disagree is not a record.
        file.extend_from_slice(&[9, 0, 0, 0, 0xE7, 0xDC, 0x0D, 0, 8, 0, 0, 0]);
        file.extend(record_bytes(&long, 0x41, &[7; 300]));

        assert!(is_sgd5(&file));
        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].name, Some("VS_Blit_0011aabb_Wrapped"));
        assert_eq!(found[0].kind, 0x32);
        assert_eq!(found[0].unpack().as_deref(), Some(&b"first container"[..]));
        assert_eq!(found[1].name, Some(long.as_str()));
        assert_eq!(found[1].unpack(), Some(vec![7; 300]));

        // The same file inside an archive.
        let mut archive = vec![0x5A; 0x1000];
        archive.extend_from_slice(&file);
        assert!(is_sgd5(&archive));
        assert_eq!(records(&archive).count(), 2);
        assert!(!is_sgd5(&archive[..0x1000]));
    }

    #[test]
    fn marker_inside_a_record_is_not_another_record() {
        let inner = record_bytes("PS_inner", 0x32, b"inner container");
        // Level 0: stored blocks, so the inner record's bytes appear as they are.
        let packed = miniz_oxide::deflate::compress_to_vec(&inner, 0);
        assert!(memchr::memmem::find(&packed, &MARKER).is_some());
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(&[0xF4, 0x01, 0, 0, 9]);
        file.extend_from_slice(b"PS_outer\0\x32");
        for v in [n, 0x000D_DCE7, n, inner.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        file.extend(record_bytes("PS_after", 0x32, b"next"));

        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_outer"), Some("PS_after")]);
        assert_eq!(found[0].unpack(), Some(inner));
    }

    #[test]
    fn offsets_inside_records() {
        // A container stored as is (level 0) inside a record's packed bytes.
        let container = b"DXBC stored as is";
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x05PS_a\0\x32");
        let packed = miniz_oxide::deflate::compress_to_vec(container, 0);
        let n = (packed.len() + 12) as u32;
        for v in [n, 0x000D_DCE7, n, container.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        let between = file.len();
        file.extend_from_slice(b"DXBC between records");
        file.extend(record_bytes("PS_b", 0x32, b"second"));
        let after = file.len();
        file.extend_from_slice(b"DXBC after");

        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), 2);
        let stored = memchr::memmem::find(&file, b"DXBC").unwrap();
        assert!(stored < between);
        assert!(in_record(&found, stored));
        assert!(!in_record(&found, 0), "the magic");
        assert!(!in_record(&found, between));
        assert!(!in_record(&found, after));
        for r in &found {
            let span = r.span();
            assert!(in_record(&found, span.start));
            assert!(in_record(&found, span.end - 1));
            assert!(!in_record(&found, span.start - 1), "the header");
        }
        assert!(!in_record(&[], stored));
    }

    #[test]
    fn header_inside_a_record_is_not_another_record() {
        // The record's packed bytes end (after the deflate stream, which
        // ignores them) with what would be the next header: NUL, kind, `n`.
        let mut packed = miniz_oxide::deflate::compress_to_vec(b"first", 6);
        let tail = miniz_oxide::deflate::compress_to_vec(b"phantom", 6);
        let tail_n = (tail.len() + 12) as u32;
        packed.extend_from_slice(&[0, 0x32]);
        packed.extend_from_slice(&tail_n.to_le_bytes());
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x09PS_first\0\x32");
        for v in [n, 0x000D_DCE7, n, 5] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        // Right after it, a marker with no header of its own.
        for v in [0x000D_DCE7, tail_n, 7] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&tail);

        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_first")]);
        assert_eq!(found[0].unpack().as_deref(), Some(&b"first"[..]));
    }

    #[test]
    fn records_before_the_magic_are_not_read() {
        // An archive whose other file happens to hold a well-formed record.
        let mut archive = record_bytes("PS_elsewhere", 0x32, b"not ours");
        assert_eq!(records(&archive).count(), 0, "no magic, no records");
        archive.extend_from_slice(MAGIC);
        archive.extend(record_bytes("PS_ours", 0x32, b"ours"));
        let names: Vec<_> = records(&archive).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_ours")]);
    }

    #[test]
    fn truncated_file_yields_only_whole_records() {
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_a", 0x32, &[5; 200]));
        let first_end = file.len();
        file.extend(record_bytes("PS_b", 0x41, b"second"));
        for cut in 0..=file.len() {
            let part = &file[..cut];
            let found: Vec<_> = records(part).collect();
            let expected = usize::from(cut >= first_end) + usize::from(cut == file.len());
            assert_eq!(found.len(), expected, "cut at {cut}");
            for r in found {
                assert!(
                    r.unpack().is_some(),
                    "cut at {cut}: {:?} does not unpack",
                    r.name
                );
            }
        }
    }

    #[test]
    fn empty_container_unpacks() {
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_empty", 0x32, b""));
        let r = records(&file).next().unwrap();
        assert_eq!(r.unpack(), Some(Vec::new()));
    }

    /// Every single-byte change to a small file: no panic, and the records
    /// found are in file order, inside the file and do not overlap.
    #[test]
    fn damaged_file_yields_disjoint_records() {
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_a", 0x32, &[5; 40]));
        file.extend(record_bytes("VS_b", 0x41, b"second"));
        for at in 0..file.len() {
            for value in [0x00, 0x01, 0x0C, 0x0D, 0x7F, 0xDC, 0xE7, 0xFF] {
                let mut damaged = file.clone();
                damaged[at] = value;
                let mut end = 0;
                for r in records(&damaged) {
                    let start = r.offset - 6;
                    assert!(start >= end, "byte {at} = {value:#x}: records overlap");
                    end = r.span().end;
                    assert!(end <= damaged.len());
                    let _ = r.unpack();
                }
            }
        }
    }

    #[test]
    fn corrupt_stream_does_not_unpack() {
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("CS_x", 0x3c, &[1; 64]));
        file[31] = 0xFF; // first packed byte: reserved block type
        let r = records(&file).next().unwrap();
        assert_eq!(r.unpack(), None);
    }
}
