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
}

/// An SGD5 file, or a file holding one (an archive: records are found by
/// their marker, so no index is needed).
pub fn is_sgd5(file: &[u8]) -> bool {
    memchr::memmem::find(file, MAGIC).is_some()
}

/// Every record in an SGD5 file, in file order.
pub fn records(file: &[u8]) -> impl Iterator<Item = Record<'_>> {
    memchr::memmem::find_iter(file, &MARKER).filter_map(|at| record(file, at))
}

fn record(file: &[u8], at: usize) -> Option<Record<'_>> {
    let u32_at = |o: usize| {
        file.get(o..o.checked_add(4)?)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize)
    };
    let n = u32_at(at.checked_sub(4)?)?;
    let unpacked_size = u32_at(at.checked_add(8)?)?;
    if u32_at(at.checked_add(4)?)? != n || n < 12 || unpacked_size > MAX_UNPACKED {
        return None;
    }
    let nul = at.checked_sub(6)?;
    let kind = *file.get(at.checked_sub(5)?)?;
    let start = at.checked_add(12)?;
    let end = at.checked_add(n)?;
    Some(Record {
        offset: at,
        name: (file.get(nul) == Some(&0))
            .then(|| name_before(file, nul))
            .flatten(),
        kind,
        packed: file.get(start..end)?,
        unpacked_size,
    })
}

/// The printable name that ends at `nul` and follows its length byte.
fn name_before(file: &[u8], nul: usize) -> Option<&str> {
    let prefix = file.get(..nul)?;
    let run = prefix
        .iter()
        .rposition(|b| !b.is_ascii_graphic())
        .map_or(0, |p| p + 1);
    // From 32 characters on, the length byte is printable too.
    (run.saturating_sub(1)..nul)
        .find(|&l| {
            prefix
                .get(l)
                .is_some_and(|&len| nul.checked_sub(l) == Some(usize::from(len)))
        })
        .and_then(|l| prefix.get(l.checked_add(1)?..nul))
        .and_then(|s| std::str::from_utf8(s).ok())
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
    fn corrupt_stream_does_not_unpack() {
        let mut file = record_bytes("CS_x", 0x3c, &[1; 64]);
        file[27] = 0xFF; // first packed byte: reserved block type
        let r = records(&file).next().unwrap();
        assert_eq!(r.unpack(), None);
    }

    #[test]
    fn rejects_boundary_and_malformed_records() {
        fn header(prefix: &[u8], n1: u32, n2: u32, unpacked: u32, body: &[u8]) -> Vec<u8> {
            let mut out = prefix.to_vec();
            for v in [n1, 0x000D_DCE7, n2, unpacked] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(body);
            out
        }

        // Marker at offsets 0..6 lacks room for `nul`, `kind`, and leading `n`.
        assert_eq!(records(&MARKER).count(), 0);
        for pad in 0..2 {
            let near_start = header(&vec![0; pad], 12, 12, 0, &[]);
            assert_eq!(records(&near_start).count(), 0);
        }

        // Truncated right after the marker.
        let mut cut = 12u32.to_le_bytes().to_vec();
        cut.extend_from_slice(&MARKER);
        assert_eq!(records(&cut).count(), 0);

        // `n < 12`, mismatched `n`, oversized `unpacked_size`, and truncated payload.
        assert_eq!(records(&header(&[0, 0x32], 11, 11, 0, &[])).count(), 0);
        assert_eq!(records(&header(&[0, 0x32], 16, 20, 4, &[0; 8])).count(), 0);
        let too_big = (MAX_UNPACKED as u32) + 1;
        assert_eq!(
            records(&header(&[0, 0x32], 12, 12, too_big, &[])).count(),
            0
        );
        assert_eq!(records(&header(&[0, 0x32], 24, 24, 4, &[0; 4])).count(), 0);
        assert_eq!(
            records(&header(&[0, 0x32], u32::MAX, u32::MAX, 4, &[0; 4])).count(),
            0
        );

        // Extreme synthetic offsets on internal helpers do not panic or overflow.
        assert_eq!(record(&MARKER, usize::MAX), None);
        assert_eq!(record(&MARKER, usize::MAX - 2), None);
        assert_eq!(name_before(&[0], 5), None);

        // Missing NUL or mismatched name length leaves `name` as `None` while keeping the record.
        let unnamed = header(&[0x7f, 0x32], 12, 12, 0, &[]);
        let parsed: Vec<_> = records(&unnamed).collect();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, None);
        assert_eq!(parsed[0].kind, 0x32);

        let bad_len = header(&[99, b'P', b'S', 0, 0x40], 12, 12, 0, &[]);
        let parsed: Vec<_> = records(&bad_len).collect();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, None);
        assert_eq!(parsed[0].kind, 0x40);
    }
}
