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
/// Bytes a record's packed stream may hold beyond what deflate needs (see
/// [`max_packed`]): the stream's end is not checked, so allow some padding.
const PACKED_SLACK: usize = 4096;
/// Packed bytes read, and unpacked bytes made, to tell whether a record's
/// bytes are a deflate stream (see [`starts_stream`]).
const PROBE: usize = 4096;

/// The most packed bytes a container of `unpacked` bytes can take. Deflate
/// never spends more than 9 bits a byte (fixed codes), and stored blocks only
/// 5 bytes per 65535, so a size beyond this is not a record's: it is bytes
/// that look like a marker (in a container stored between records, say).
fn max_packed(unpacked: usize) -> usize {
    unpacked + unpacked / 8 + PACKED_SLACK
}

/// The fewest packed bytes a container of `unpacked` bytes can take. The
/// smallest deflate stream (one empty fixed-code block) is 2 bytes, and no
/// stream does better than one 258-byte match per 2 bits (1-bit length and
/// distance codes, no extra bits): 1032 bytes a byte. Fewer packed bytes is
/// not a record's, and taking it for one would make its span (12 bytes and
/// up) swallow the header of a real record right after it.
fn min_packed(unpacked: usize) -> usize {
    (unpacked / 1032).max(2)
}

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

    /// Whether the packed bytes are a deflate stream of the stated size (or
    /// its start: the stream's end is not checked).
    fn is_stream(&self) -> bool {
        starts_stream(self.packed, self.unpacked_size)
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
/// of the next record's header or name, which must follow them. A record
/// whose packed bytes run past the end of the file (a truncated file) is not
/// yielded, and ends the search: every byte after its marker is its own.
/// Sizes that no deflate stream of the stated unpacked size could have (too
/// many packed bytes, or too few) do not form a record, and neither do bytes
/// running past the end that do not start a deflate stream, nor bytes that
/// are no deflate stream laid over a record that is one (whole or cut off),
/// so such bytes neither end the search nor hide records. A corrupt record
/// that covers no other record is still yielded (its [`Record::unpack`] is
/// `None`).
pub fn records(file: &[u8]) -> impl Iterator<Item = Record<'_>> {
    let finder = memchr::memmem::Finder::new(&MARKER);
    // Where the next record may start: after the magic, then after each
    // record's packed bytes. No magic: past the end, so nothing is found.
    let mut floor = memchr::memmem::find(file, MAGIC).map_or(file.len(), |m| m + MAGIC.len());
    // A marker needs its name's NUL, the kind and `n` (6 bytes) before it.
    let mut from = floor + 6;
    let mut streams = Streams::default();
    std::iter::from_fn(move || {
        while let Some(found) = finder.find(file.get(from..)?) {
            let at = from + found;
            match record(&file[floor..], at - floor) {
                Found::Record(r) => {
                    let r = Record { offset: at, ..r };
                    if covers_a_record(file, &r, &mut streams) && !r.is_stream() {
                        // Bytes that only look like a marker, laid over a
                        // real record: the real one wins.
                        from = at + 1;
                        continue;
                    }
                    floor = r.span().end;
                    from = floor + 6;
                    return Some(r);
                }
                Found::Truncated => {
                    from = file.len() + 1;
                    return None;
                }
                Found::Nothing => from = at + 1,
            }
        }
        None
    })
}

enum Found<'a> {
    Record(Record<'a>),
    /// A record header whose packed bytes run past the end of the file.
    Truncated,
    Nothing,
}

/// The record whose marker is at `at`, reading nothing before `file`.
fn record(file: &[u8], at: usize) -> Found<'_> {
    let u32_at = |o: usize| {
        file.get(o..o.checked_add(4)?)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize)
    };
    let Some(nul) = at.checked_sub(6) else {
        return Found::Nothing;
    };
    let (Some(n), Some(n2), Some(unpacked_size)) = (u32_at(at - 4), u32_at(at + 4), u32_at(at + 8))
    else {
        return Found::Nothing;
    };
    if n2 != n
        || n < 12
        || unpacked_size > MAX_UNPACKED
        || n - 12 > max_packed(unpacked_size)
        || n - 12 < min_packed(unpacked_size)
    {
        return Found::Nothing;
    }
    let Some(packed) = at.checked_add(n).and_then(|end| file.get(at + 12..end)) else {
        // Bytes that only look like a marker (in a container stored between
        // records, say) may have sizes that fit and still run past the end.
        // Only a record whose bytes so far are a deflate stream is one.
        return if starts_stream(file.get(at + 12..).unwrap_or_default(), unpacked_size) {
            Found::Truncated
        } else {
            Found::Nothing
        };
    };
    Found::Record(Record {
        offset: at,
        name: (file[nul] == 0).then(|| name_before(file, nul)).flatten(),
        kind: file[at - 5],
        packed,
        unpacked_size,
    })
}

/// Whether a record whose packed bytes are a deflate stream (or, cut off by
/// the end of the file, start one) starts (its header, 6 bytes before its
/// marker) inside `r`'s span. Such a record inside a real one is that
/// record's (stored deflate blocks); but if `r`'s own bytes are no deflate
/// stream, `r` is bytes that look like a marker (in a container stored
/// between records, say) and the record it covers is real. A cut-off record
/// counts too: were `r` taken for a record, the search would go on inside
/// the cut-off one and find the records it stores.
fn covers_a_record(file: &[u8], r: &Record<'_>, streams: &mut Streams) -> bool {
    let span = r.span();
    // Its header must be after `r`'s marker, and a marker from
    // `span.end + 6` on has its header after the span.
    let hi = file.len().min(span.end + 6 + MARKER.len() - 1);
    streams.any_in(file, r.offset + 7, hi)
}

/// Which markers start a record whose packed bytes are (or, cut off, start)
/// a deflate stream, as
/// far as [`records`] has asked. Its questions move forward through the
/// file, and markers in bytes that only look like records may each cover
/// all the rest of it: asking each of them anew rescans and reinflates the
/// same bytes, so a crafted file of `n` such markers takes `n²` work and
/// more. Each marker is looked at once instead.
#[derive(Default)]
struct Streams {
    /// No marker at an offset in `from..to` starts such a record.
    from: usize,
    to: usize,
    /// The marker at `to` does (and was the first found from `from` on).
    hit: bool,
}

impl Streams {
    /// Whether a marker that lies wholly in `lo..hi` starts such a record
    /// (its header read from `lo` on). Bytes past the end of the file hold
    /// no marker: a `hi` beyond it asks only up to the end, and is not taken
    /// to have looked at markers it never saw.
    fn any_in(&mut self, file: &[u8], lo: usize, hi: usize) -> bool {
        let hi = hi.min(file.len());
        if lo < self.from || lo > self.to {
            *self = Self {
                from: lo,
                to: lo,
                hit: false,
            };
        }
        if !self.hit {
            let window = file.get(self.to..hi).unwrap_or_default();
            match memchr::memmem::find_iter(window, &MARKER)
                .map(|m| self.to + m)
                // A marker too near the start of the file has no header.
                .find(
                    |&at| match at.checked_sub(6).map(|h| record(&file[h..], 6)) {
                        Some(Found::Record(inner)) => inner.is_stream(),
                        // Cut off, its bytes so far a deflate stream.
                        Some(Found::Truncated) => true,
                        Some(Found::Nothing) | None => false,
                    },
                ) {
                Some(at) => (self.to, self.hit) = (at, true),
                // Markers at offsets up to `hi - 4` have been looked at.
                None => self.to = self.to.max(hi.saturating_sub(MARKER.len() - 1)),
            }
        }
        self.hit && self.to + MARKER.len() <= hi
    }
}

/// Whether `bytes` are the start of a deflate stream of `unpacked` bytes:
/// they run out before the stream ends, or the stream ends having produced
/// exactly that many bytes.
///
/// Only the first [`PROBE`] bytes are read, and at most [`PROBE`] bytes
/// inflated: [`records`] asks this of every marker that might be a record,
/// and in bytes that only look like records each such marker's bytes may run
/// on through all the ones after it (stored blocks that each end where the
/// next marker's begin), so reading them all makes a file of `n` markers
/// take `n²` work. Real records are told from noise long before that.
fn starts_stream(bytes: &[u8], unpacked: usize) -> bool {
    use miniz_oxide::inflate::{TINFLStatus, decompress_to_vec_with_limit};
    let limit = unpacked.min(PROBE);
    let probe = &bytes[..bytes.len().min(PROBE)];
    #[cfg(test)]
    tests::WORK.with(|w| {
        let (checks, read) = w.get();
        w.set((checks + 1, read + probe.len()));
    });
    match decompress_to_vec_with_limit(probe, limit) {
        Ok(out) => out.len() == unpacked,
        Err(e) => {
            e.status == TINFLStatus::FailedCannotMakeProgress
                // Inflated as far as the probe goes, with more to come.
                || (e.status == TINFLStatus::HasMoreOutput && limit < unpacked)
        }
    }
}

/// The printable name that ends at `nul` and follows its length byte. A
/// length byte of 1 (no characters) is not a name: any `01 00` before the
/// kind would read as one.
fn name_before(file: &[u8], nul: usize) -> Option<&str> {
    let run = file[..nul]
        .iter()
        .rposition(|b| !b.is_ascii_graphic())
        .map_or(0, |p| p + 1);
    // From 32 characters on, the length byte is printable too, and so may be
    // the bytes before it (the previous record's packed tail, say) and the
    // name's own characters: any of them that equals its distance to the NUL
    // reads as a length byte as well. Shader names are identifiers (letters,
    // digits, `_`, not starting with a digit), and a length byte from 33 to
    // 47 (names of 32 to 46 characters) is punctuation and from 48 to 57 a
    // digit, so the earliest length byte whose name is an identifier wins
    // over a stray byte before it; failing that, the earliest.
    let mut lengths = (run.saturating_sub(1)..nul).filter(|&l| usize::from(file[l]) == nul - l);
    let first = lengths.clone().next();
    lengths
        .find(|&l| {
            let name = &file[l + 1..nul];
            !name.first().is_some_and(u8::is_ascii_digit)
                && name.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'_')
        })
        .or(first)
        .and_then(|l| std::str::from_utf8(&file[l + 1..nul]).ok())
        .filter(|name| !name.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        /// Stream checks made, and packed bytes they read, on this thread
        /// (each test runs on its own): the work [`records`] does, counted
        /// rather than timed, so a slow or busy machine cannot fail a test
        /// and a fast one cannot hide a regression.
        pub(super) static WORK: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    }

    /// The records in `file`, and the stream checks and packed bytes read
    /// to find them.
    fn records_counting(file: &[u8]) -> (Vec<Record<'_>>, usize, usize) {
        WORK.with(|w| w.set((0, 0)));
        let found = records(file).collect();
        let (checks, read) = WORK.with(Cell::get);
        (found, checks, read)
    }

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
    fn truncated_record_hides_the_records_it_stores() {
        // A record whose packed bytes (stored blocks) hold a whole record,
        // cut off after that inner record but before its own end.
        let inner = record_bytes("PS_inner", 0x32, b"inner container");
        let mut payload = inner.clone();
        payload.extend_from_slice(&[0x55; 64]);
        let packed = miniz_oxide::deflate::compress_to_vec(&payload, 0);
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_first", 0x32, b"first"));
        file.extend_from_slice(b"\x09PS_outer\0\x32");
        for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        let stored = file.len();
        file.extend_from_slice(&packed);
        let inner_end = stored + memchr::memmem::find(&packed, &inner).unwrap() + inner.len();
        let names: Vec<_> = records(&file).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_first"), Some("PS_outer")]);

        // Before: [PS_first, PS_inner].
        let names: Vec<_> = records(&file[..inner_end]).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_first")]);
    }

    #[test]
    fn marker_without_a_stream_over_a_cut_off_record_hides_nothing() {
        // Bytes that read as a marker with sizes that fit, then bytes that
        // are no deflate stream (block type 3 is reserved), laid over the
        // header of a record whose packed bytes (stored blocks) hold a whole
        // record; the file then cut off after that inner record.
        let inner = record_bytes("PS_inner", 0x32, b"inner container");
        let mut payload = inner.clone();
        payload.extend_from_slice(&[0x55; 64]);
        let packed = miniz_oxide::deflate::compress_to_vec(&payload, 0);
        let n = (packed.len() + 12) as u32;
        let mut outer = b"\x09PS_outer\0\x32".to_vec();
        for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
            outer.extend_from_slice(&v.to_le_bytes());
        }
        let stored = outer.len();
        outer.extend_from_slice(&packed);

        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_first", 0x32, b"first"));
        file.extend_from_slice(b"DXBC");
        // 16 bytes of no stream, then the outer record's header and the
        // start of its packed bytes.
        let phantom = 16 + stored + 8;
        let m = (12 + phantom) as u32;
        for v in [m, 0x000D_DCE7, m, 64] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&[0x07; 16]);
        let outer_at = file.len();
        file.extend_from_slice(&outer);
        let inner_end =
            outer_at + stored + memchr::memmem::find(&packed, &inner).unwrap() + inner.len();
        let names: Vec<_> = records(&file).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_first"), Some("PS_outer")]);

        // Cut off: the phantom covers a record whose bytes so far are a
        // deflate stream, so it is not the record. Before: [PS_first, None,
        // PS_inner] (the phantom was yielded and the search went on inside
        // the outer record, exposing the record it stores).
        for cut in [inner_end, inner_end + 10] {
            let names: Vec<_> = records(&file[..cut]).map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_first")], "cut at {cut}");
        }
    }

    #[test]
    fn marker_with_impossible_sizes_does_not_end_the_search() {
        // Bytes between records that read as a marker whose packed size runs
        // past the end of the file, but is far more than a deflate stream of
        // 16 bytes could take.
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"DXBC");
        for v in [u32::MAX, 0x000D_DCE7, u32::MAX, 16] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend(record_bytes("PS_after", 0x32, b"after"));
        // Before: [] (the phantom was a truncated record and ended the search).
        let names: Vec<_> = records(&file).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_after")]);
    }

    #[test]
    fn marker_running_past_the_end_without_a_stream_does_not_end_the_search() {
        // Bytes between records that read as a marker with sizes that fit
        // (64 KiB both ways) and run past the end of the file, followed by
        // bytes that are no deflate stream (block type 3 is reserved).
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_first", 0x32, b"first"));
        file.extend_from_slice(b"DXBC");
        for v in [0x1_0000_u32, 0x000D_DCE7, 0x1_0000, 0x1_0000] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&[0x07; 16]);
        file.extend(record_bytes("PS_after", 0x32, b"after"));
        // Before: [PS_first] (the phantom was a truncated record and ended
        // the search).
        let names: Vec<_> = records(&file).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_first"), Some("PS_after")]);
    }

    #[test]
    fn truncated_stream_is_told_from_noise() {
        let payload: Vec<u8> = (0..4000u32).map(|i| (i * 7 % 251) as u8).collect();
        for level in [0, 1, 6, 10] {
            let packed = miniz_oxide::deflate::compress_to_vec(&payload, level);
            for cut in 0..packed.len() {
                assert!(
                    starts_stream(&packed[..cut], payload.len()),
                    "level {level}, cut at {cut}"
                );
            }
            assert!(starts_stream(&packed, payload.len()), "level {level}");
            // A whole stream of another size is not this record's.
            assert!(!starts_stream(&packed, payload.len() + 1), "level {level}");
            assert!(!starts_stream(&packed, payload.len() - 1), "level {level}");
        }
        // Reserved block type, and a stored block whose length check fails.
        assert!(!starts_stream(&[0x07, 0, 0], 100));
        assert!(!starts_stream(&[0x01, 0x05, 0x00, 0x00, 0x00, 1, 2, 3], 5));
    }

    #[test]
    fn packed_size_bound_holds_for_every_level() {
        // Incompressible bytes, at every level miniz writes: the bound is
        // never reached, so no real record is refused.
        let mut x = 0x1234_5678_u32;
        let noise: Vec<u8> = (0..200_000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect();
        for len in [0, 1, 100, 65_535, 65_536, noise.len()] {
            for level in 0..=10 {
                let packed = miniz_oxide::deflate::compress_to_vec(&noise[..len], level);
                assert!(packed.len() <= max_packed(len), "len {len} level {level}");
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

    /// Seeded random files: gaps of noise and of near-miss markers between
    /// records, some records stored as is (level 0) and holding records of
    /// their own. Exactly the planted records are found, each with its
    /// container, and [`in_record`] agrees with a scan of their spans.
    #[test]
    fn random_files_yield_exactly_the_planted_records() {
        let mut x = 0x9E37_79B9_u32;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for seed in 0..200 {
            let mut file = MAGIC.to_vec();
            // (offset of the marker, name, container)
            let mut planted: Vec<(usize, String, Vec<u8>)> = Vec::new();
            for i in 0..1 + next() % 6 {
                // A gap: noise, then sometimes a marker whose sizes disagree
                // or that no deflate stream could have.
                for _ in 0..next() % 40 {
                    file.push(next() as u8);
                }
                match next() % 3 {
                    0 => {
                        for v in [20_u32, 0x000D_DCE7, 21, 8] {
                            file.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    1 => {
                        for v in [u32::MAX, 0x000D_DCE7, u32::MAX, 16] {
                            file.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    _ => {}
                }
                // A name's length byte follows the gap; a long name's is
                // printable, so noise right before it is kept non-printable.
                file.push(0x01);
                let name = format!("PS_{seed}_{i}_{}", "w".repeat(next() as usize % 50));
                let mut payload: Vec<u8> = (0..next() % 300).map(|_| next() as u8).collect();
                let level = if next() % 2 == 0 {
                    // Stored as is, holding a record and a container's magic.
                    payload.extend(record_bytes("PS_stored", 0x32, b"stored"));
                    payload.extend_from_slice(b"DXBC");
                    0
                } else {
                    6
                };
                let packed = miniz_oxide::deflate::compress_to_vec(&payload, level);
                let n = (packed.len() + 12) as u32;
                file.push(name.len() as u8 + 1);
                file.extend_from_slice(name.as_bytes());
                file.extend_from_slice(&[0, 0x32]);
                planted.push((file.len() + 4, name, payload.clone()));
                for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
                    file.extend_from_slice(&v.to_le_bytes());
                }
                file.extend_from_slice(&packed);
            }
            for _ in 0..next() % 40 {
                file.push(next() as u8);
            }

            let found: Vec<_> = records(&file).collect();
            let got: Vec<_> = found
                .iter()
                .map(|r| (r.offset, r.name.map(str::to_owned), r.unpack()))
                .collect();
            let want: Vec<_> = planted
                .iter()
                .map(|(at, name, payload)| (*at, Some(name.clone()), Some(payload.clone())))
                .collect();
            assert_eq!(got, want, "seed {seed}");
            for offset in 0..=file.len() {
                let linear = found.iter().any(|r| r.span().contains(&offset));
                assert_eq!(
                    in_record(&found, offset),
                    linear,
                    "seed {seed}, offset {offset}"
                );
            }
        }
    }

    #[test]
    fn empty_name_is_no_name() {
        // A record whose name would be empty: `01 00` before the kind.
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("", 0x32, b"nameless"));
        file.extend(record_bytes("PS_named", 0x32, b"named"));
        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [None, Some("PS_named")]);
        assert_eq!(found[0].unpack().as_deref(), Some(&b"nameless"[..]));
    }

    #[test]
    fn printable_bytes_before_a_long_name_are_not_part_of_it() {
        // From 32 characters on, a name's length byte is printable, so the
        // bytes before it (the previous record's packed tail, say) run on
        // into the name. One that equals its own distance to the NUL reads
        // as a length byte too, and must not win over the real one.
        let name = "PS_DeferredLightingComposite_0011aabb_Wrapped";
        let len = name.len() as u8 + 1;
        for before in [vec![len + 1], vec![len + 2, b'A'], vec![b'x', len + 1]] {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(&[0, 0]);
            file.extend_from_slice(&before);
            let header = record_bytes(name, 0x32, b"container");
            // Drop the helper's 4 bytes before the length byte.
            file.extend_from_slice(&header[4..]);
            let names: Vec<_> = records(&file).map(|r| r.name).collect();
            assert_eq!(names, [Some(name)], "bytes before: {before:?}");
        }
        // A character of the name that equals its distance to the NUL (the
        // `4`, 52) is not its length byte either.
        for long in [
            format!("PS_{}_Wrapped", "x".repeat(100)),
            format!("PS_4_0_{}", "w".repeat(48)),
        ] {
            let mut file = MAGIC.to_vec();
            file.extend(record_bytes(&long, 0x32, b"long"));
            let names: Vec<_> = records(&file).map(|r| r.name).collect();
            assert_eq!(names, [Some(&long[..])]);
        }
    }

    #[test]
    fn a_name_does_not_start_with_its_length_byte() {
        // A name of 47 to 56 characters has a digit for its length byte, so a
        // stray byte right before it that equals its distance to the NUL
        // reads as a length byte whose name (the digit, then the name) is all
        // letters, digits and `_`. Names are identifiers, which do not start
        // with a digit.
        for name in [
            format!("PS_{}", "w".repeat(44)),
            format!(
                "PS_DeferredLightingComposite_0011aabb_{}_Wrapped",
                "x".repeat(3)
            ),
            format!("PS_{}", "w".repeat(53)),
        ] {
            let len = name.len() as u8 + 1;
            assert!(len.is_ascii_digit(), "{name}");
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(&[0, 0, len + 1]);
            let header = record_bytes(&name, 0x32, b"container");
            file.extend_from_slice(&header[4..]);
            let names: Vec<_> = records(&file).map(|r| r.name).collect();
            assert_eq!(names, [Some(&name[..])]);
        }
    }

    #[test]
    fn archive_with_two_sgd5_files_yields_both() {
        // Two SGD5 files one after the other in an archive, with another
        // file's bytes between them: the second magic sits in a gap.
        let mut archive = b"other file".to_vec();
        archive.extend_from_slice(MAGIC);
        archive.extend(record_bytes("PS_one", 0x32, b"first file"));
        archive.extend_from_slice(b"another file between");
        archive.extend_from_slice(MAGIC);
        archive.extend(record_bytes("PS_two", 0x41, b"second file"));
        let found: Vec<_> = records(&archive).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_one"), Some("PS_two")]);
        assert_eq!(found[1].kind, 0x41);
        assert_eq!(found[1].unpack().as_deref(), Some(&b"second file"[..]));
    }

    #[test]
    fn packed_size_lower_bound_holds_for_every_level() {
        // The most compressible bytes, at every level miniz writes: no real
        // record has fewer packed bytes than the bound.
        let zeros = vec![0_u8; 4 << 20];
        for len in [0, 1, 2, 258, 259, 1032, 65_536, 1 << 20, zeros.len()] {
            for level in 0..=10 {
                let packed = miniz_oxide::deflate::compress_to_vec(&zeros[..len], level);
                assert!(packed.len() >= min_packed(len), "len {len} level {level}");
            }
        }
    }

    #[test]
    fn marker_with_too_few_packed_bytes_does_not_hide_the_next_record() {
        // Bytes between records that read as a marker with no packed bytes
        // (or one) for a 1 MiB container, right before a real record: taken
        // for a record, its span would cover the real record's header.
        for packed_len in [0_u32, 1, 2, 3] {
            let mut file = MAGIC.to_vec();
            file.extend(record_bytes("PS_first", 0x32, b"first"));
            file.extend_from_slice(b"DXBC");
            let n = 12 + packed_len;
            for v in [n, 0x000D_DCE7, n, 1 << 20] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            // The next record from its length byte on: with packed bytes,
            // the phantom's span would start over it.
            let mut next = record_bytes("PS_after", 0x32, b"after");
            next.drain(..4);
            file.extend(next);
            // Before: [PS_first, None, None] for a phantom of 1 to 3 packed
            // bytes (its span swallowed PS_after's length byte, so PS_after
            // lost its name), and [PS_first, None, PS_after] for none (a
            // record that never unpacks).
            let names: Vec<_> = records(&file).map(|r| r.name).collect();
            assert_eq!(
                names,
                [Some("PS_first"), Some("PS_after")],
                "{packed_len} packed bytes"
            );
        }
    }

    #[test]
    fn smallest_containers_are_records() {
        // The fewest packed bytes deflate writes: every small container at
        // every level is still found.
        for len in [0_usize, 1, 2, 3] {
            for level in 0..=10 {
                let payload = vec![0x41; len];
                let packed = miniz_oxide::deflate::compress_to_vec(&payload, level);
                let n = (packed.len() + 12) as u32;
                let mut file = MAGIC.to_vec();
                file.extend_from_slice(b"\x05PS_s\0\x32");
                for v in [n, 0x000D_DCE7, n, len as u32] {
                    file.extend_from_slice(&v.to_le_bytes());
                }
                file.extend_from_slice(&packed);
                let found: Vec<_> = records(&file).collect();
                assert_eq!(found.len(), 1, "len {len} level {level}");
                assert_eq!(found[0].unpack(), Some(payload), "len {len} level {level}");
            }
        }
    }

    #[test]
    fn marker_without_a_stream_does_not_hide_the_record_it_covers() {
        // Bytes between records that read as a marker with sizes that fit
        // and stay inside the file, followed by bytes that are no deflate
        // stream (block type 3 is reserved) and then a real record: taken
        // for a record, its span covers the real record's header.
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_first", 0x32, b"first"));
        let after = record_bytes("PS_after", 0x32, b"after");
        // PS_after's header (6 bytes before its marker) starts 16 + 13 bytes
        // into the phantom's packed bytes: from 30 on they cover it.
        for span in [30, 35, after.len(), after.len() + 31] {
            let mut file = file.clone();
            file.extend_from_slice(b"DXBC");
            let n = (12 + span) as u32;
            for v in [n, 0x000D_DCE7, n, 64] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&[0x07; 16]);
            file.extend(&after);
            file.extend_from_slice(&[0x5A; 16]);
            // Before: [PS_first, None] (the phantom hid PS_after).
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_first"), Some("PS_after")], "span {span}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        }
    }

    #[test]
    fn marker_without_a_stream_ending_before_a_header_is_a_record_too() {
        // The phantom of the test above, its span ending just short of
        // PS_after's header (29 packed bytes: 16 of noise, then PS_after's
        // 13 bytes before its NUL) or of its name (20: noise and the 4 bytes
        // before the length byte). It covers no record, so it is yielded
        // (and does not unpack), and PS_after follows it; a name the phantom
        // swallowed, wholly or in part, is not read, as the name lies before
        // where PS_after may start.
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_first", 0x32, b"first"));
        let after = record_bytes("PS_after", 0x32, b"after");
        for (span, name) in [(20, Some("PS_after")), (21, None), (25, None), (29, None)] {
            let mut file = file.clone();
            file.extend_from_slice(b"DXBC");
            let phantom = file.len() + 4;
            let n = (12 + span) as u32;
            for v in [n, 0x000D_DCE7, n, 64] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&[0x07; 16]);
            let start = file.len();
            file.extend(&after);
            file.extend_from_slice(&[0x5A; 16]);
            let found: Vec<_> = records(&file).collect();
            assert_eq!(found.len(), 3, "span {span}");
            assert_eq!(found[1].offset, phantom, "span {span}");
            assert_eq!(found[1].name, None, "span {span}");
            assert_eq!(found[1].unpack(), None, "span {span}");
            assert_eq!(found[2].offset, start + 19, "span {span}");
            assert_eq!(found[2].name, name, "span {span}");
            assert_eq!(found[2].unpack().as_deref(), Some(&b"after"[..]));
            assert!(found[1].span().end <= found[2].offset - 6, "span {span}");
        }
    }

    #[test]
    fn corrupt_record_is_still_a_record() {
        // A real record whose stream is damaged covers no other record: it
        // is yielded (and does not unpack), and so is the record after it.
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("CS_x", 0x3c, &[1; 64]));
        file[31] = 0xFF;
        file.extend(record_bytes("PS_after", 0x32, b"after"));
        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("CS_x"), Some("PS_after")]);
        assert_eq!(found[0].unpack(), None);
        assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
    }

    #[test]
    fn stream_of_another_size_does_not_unpack() {
        // A record whose stream makes one byte more, or one fewer, than its
        // stated size covers no other record: it is yielded, does not
        // unpack, and the record after it is found.
        let packed = miniz_oxide::deflate::compress_to_vec(b"twelve bytes", 6);
        let n = (packed.len() + 12) as u32;
        for stated in [11_u32, 13] {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x05PS_a\0\x32");
            for v in [n, 0x000D_DCE7, n, stated] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&packed);
            file.extend(record_bytes("PS_after", 0x32, b"after"));
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_a"), Some("PS_after")], "stated {stated}");
            assert_eq!(found[0].unpack(), None, "stated {stated}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        }
    }

    #[test]
    fn packed_bytes_up_to_the_bound_are_a_record() {
        // A real stream padded out (its end is not checked) to exactly the
        // most packed bytes the stated size allows is a record and unpacks;
        // one byte more is not a record (it never hides the next one).
        for payload in [Vec::new(), vec![0x41; 64], (0..=255).collect()] {
            let packed = miniz_oxide::deflate::compress_to_vec(&payload, 6);
            let bound = max_packed(payload.len());
            for (len, want) in [
                (bound, vec![Some("PS_a"), Some("PS_after")]),
                (bound + 1, vec![Some("PS_after")]),
            ] {
                let mut padded = packed.clone();
                padded.resize(len, 0);
                let n = (padded.len() + 12) as u32;
                let mut file = MAGIC.to_vec();
                file.extend_from_slice(b"\x05PS_a\0\x32");
                for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
                    file.extend_from_slice(&v.to_le_bytes());
                }
                file.extend_from_slice(&padded);
                file.extend(record_bytes("PS_after", 0x32, b"after"));
                let found: Vec<_> = records(&file).collect();
                let names: Vec<_> = found.iter().map(|r| r.name).collect();
                let at = format!("{} bytes, {len} packed", payload.len());
                assert_eq!(names, want, "{at}");
                if len == bound {
                    assert_eq!(found[0].unpack(), Some(payload.clone()), "{at}");
                }
                let last = found.last().expect("PS_after");
                assert_eq!(last.unpack().as_deref(), Some(&b"after"[..]), "{at}");
            }
        }
    }

    #[test]
    fn marker_in_padding_belongs_to_the_record_even_if_padding_is_cut_short() {
        let payload = b"outer";
        let packed = miniz_oxide::deflate::compress_to_vec(payload, 6);
        let inner = record_bytes("PS_inner", 0x41, b"inner");
        let n = (12 + packed.len() + inner.len() + 1) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x09PS_outer\0\x32");
        file.extend_from_slice(&n.to_le_bytes());
        let marker = file.len();
        for v in [0x000D_DCE7, n, payload.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        file.extend_from_slice(&inner);
        let cut = file.len();
        file.push(0);
        let end = file.len();
        file.extend(record_bytes("PS_after", 0x32, b"after"));

        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].span(), marker..end);
        assert_eq!(found[0].unpack().as_deref(), Some(payload.as_slice()));
        assert_eq!(found[1].name, Some("PS_after"));
        assert_eq!(records(&file[..cut]).count(), 0);

        // The first stream ended after five bytes. If the outer header
        // claims six, the complete record in its padding wins instead.
        file[marker + 8..marker + 12].copy_from_slice(&6u32.to_le_bytes());
        for (bytes, names) in [
            (&file[..], vec![Some("PS_inner"), Some("PS_after")]),
            (&file[..cut], vec![Some("PS_inner")]),
        ] {
            let found: Vec<_> = records(bytes).map(|r| r.name).collect();
            assert_eq!(found, names);
        }
    }

    #[test]
    fn marker_stating_more_than_the_cap_is_not_a_record() {
        // A record stored as is (level 0) inside bytes whose marker states
        // an unpacked size: up to the cap they are a record, which holds
        // the stored one; past it they are not, and the stored record is
        // found in their place. Enough packed bytes that the cap's lower
        // bound on them is met.
        let inner = record_bytes("PS_inner", 0x32, b"inner");
        let mut payload = vec![0x5A; 100];
        payload.extend_from_slice(&inner);
        payload.resize(min_packed(MAX_UNPACKED + 1) + 64, 0x5A);
        let packed = miniz_oxide::deflate::compress_to_vec(&payload, 0);
        let n = (packed.len() + 12) as u32;
        for (stated, want) in [
            (MAX_UNPACKED, Some("PS_outer")),
            (MAX_UNPACKED + 1, Some("PS_inner")),
        ] {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x09PS_outer\0\x32");
            for v in [n, 0x000D_DCE7, n, stated as u32] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&packed);
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [want], "stated {stated}");
        }
    }

    #[test]
    fn nested_markers_without_streams_take_linear_work() {
        // Bytes that each read as a marker whose span runs to the end of the
        // file and covers every one after it, then a real record: each is
        // skipped in favour of the record it covers. Before, every one of
        // them rescanned (and reinflated) the rest of the file: these 8000
        // (150 KB) took 43 s in a test build, and now take milliseconds.
        let real = record_bytes("PS_real", 0x32, b"real");
        let count = 8000;
        let end = MAGIC.len() + count * 19 + real.len();
        let mut file = MAGIC.to_vec();
        for _ in 0..count {
            let n = (end - (file.len() + 6)) as u32;
            file.extend_from_slice(&[0, 0x32]);
            for v in [n, 0x000D_DCE7, n, 1 << 20] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.push(0x07); // reserved block type: no deflate stream
        }
        file.extend(&real);
        assert_eq!(file.len(), end);
        let (found, checks, _) = records_counting(&file);
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_real")]);
        assert_eq!(found[0].unpack().as_deref(), Some(&b"real"[..]));
        // Each marker is checked for a stream at most twice (as a record,
        // and as one another covers), not once for every marker before it.
        assert!(checks <= 2 * (count + 1), "{checks} stream checks");
    }

    #[test]
    fn corrupt_stream_does_not_unpack() {
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("CS_x", 0x3c, &[1; 64]));
        file[31] = 0xFF; // first packed byte: reserved block type
        let r = records(&file).next().unwrap();
        assert_eq!(r.unpack(), None);
    }

    /// `count` markers between the magic and a real record, each followed by
    /// a stored block that ends where the next marker's stored block starts,
    /// so the deflate stream after every marker runs through all the ones
    /// after it, to a reserved block (no deflate stream) just before the
    /// real record. `n(at)` is each marker's `n`, given its offset.
    fn chained_markers(count: usize, n: impl Fn(usize) -> u32, real: &[u8]) -> Vec<u8> {
        const PAD: usize = 41;
        let len = (PAD + 18) as u16;
        let mut file = MAGIC.to_vec();
        for _ in 0..count {
            file.extend_from_slice(&[0, 0x32]);
            let n = n(file.len() + 4);
            for v in [n, 0x000D_DCE7, n, n] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.push(0x00); // stored, not final
            file.extend_from_slice(&len.to_le_bytes());
            file.extend_from_slice(&(!len).to_le_bytes());
            file.extend_from_slice(&[0x5A; PAD]);
        }
        file.extend_from_slice(&[0x5A; 18]);
        file.push(0x07); // reserved block type
        file.extend_from_slice(real);
        file
    }

    #[test]
    fn chained_markers_past_the_end_take_linear_work() {
        // Each marker's packed size runs past the end of the file, so its
        // bytes are checked for the start of a deflate stream. Before, each
        // check inflated the rest of the chain: these 32000 (2 MB) took 9 s
        // in a test build. The chain is a valid deflate stream for far more
        // than the probe reads, so the first marker is now taken for a
        // truncated record, as a cut-off file's last record would be.
        let real = record_bytes("PS_real", 0x32, b"real");
        let count = 32_000;
        let file = chained_markers(count, |_| MAX_UNPACKED as u32, &real);
        let (found, checks, read) = records_counting(&file);
        assert!(found.is_empty());
        // Each check reads a probe, not the rest of the chain.
        assert!(checks <= 2 * count, "{checks} stream checks");
        assert!(read <= checks * PROBE, "{read} bytes read");
    }

    #[test]
    fn chained_markers_inside_the_file_take_linear_work() {
        // Each marker's packed bytes end at the end of the file, so the
        // first covers all the others and each of them is checked for a
        // deflate stream. Before, each check inflated the rest of the chain
        // (9 s for these 32000 in a test build). The first is yielded as a
        // corrupt record whose span is all of them.
        let count = 32_000;
        let end = chained_markers(count, |_| 0, &[]).len();
        let file = chained_markers(count, |at| (end - at) as u32, &[]);
        let (found, checks, read) = records_counting(&file);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].span(), MAGIC.len() + 6..file.len());
        assert_eq!(found[0].unpack(), None);
        assert!(checks <= 2 * count, "{checks} stream checks");
        assert!(read <= checks * PROBE, "{read} bytes read");
    }

    /// Seeded files, some records stored as is (level 0, holding a record of
    /// their own), the bytes then damaged (changed, inserted, removed, or a
    /// marker or a small `n` put in), and every prefix of each: a file cut
    /// short (still being written or copied, say) yields no container the
    /// whole file does not, each from the same record in every field, so
    /// reading it early never reports a container that later turns out to be
    /// bytes of another record. Records only appear as the file grows:
    /// nothing it adds can hide one that unpacks.
    #[test]
    fn cut_off_files_yield_only_records_of_the_whole_file() {
        let mut x = 0x2545_F491_u32;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for seed in 0..300 {
            let mut file = MAGIC.to_vec();
            for i in 0..1 + next() % 4 {
                let mut payload: Vec<u8> = (0..next() % 120).map(|_| (next() % 4) as u8).collect();
                let level = [0, 1, 6, 10][next() as usize % 4];
                if level == 0 {
                    // Stored as is, holding a record: cut off after it, the
                    // outer record must not expose it.
                    payload.extend(record_bytes("PS_stored", 0x32, b"stored"));
                    payload.extend((0..next() % 40).map(|_| next() as u8));
                }
                let packed = miniz_oxide::deflate::compress_to_vec(&payload, level);
                let n = (packed.len() + 12) as u32;
                let name = format!("PS_{i}");
                file.extend_from_slice(&[0x01, name.len() as u8 + 1]);
                file.extend_from_slice(name.as_bytes());
                file.extend_from_slice(&[0, 0x32]);
                for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
                    file.extend_from_slice(&v.to_le_bytes());
                }
                file.extend_from_slice(&packed);
            }
            for _ in 0..1 + next() % 5 {
                let at = next() as usize % (file.len() + 1);
                match next() % 5 {
                    0 if at < file.len() => file[at] = next() as u8,
                    1 => file.insert(at, next() as u8),
                    2 if at < file.len() => _ = file.remove(at),
                    3 => _ = file.splice(at..at, MARKER),
                    _ => _ = file.splice(at..at, (next() % 600).to_le_bytes()),
                }
            }

            let whole: Vec<_> = records(&file).collect();
            for cut in 0..file.len() {
                let part = &file[..cut];
                for r in records(part) {
                    assert!(r.span().end <= cut, "seed {seed}, cut at {cut}");
                    if r.unpack().is_none() {
                        // Corrupt bytes over corrupt bytes: which is the
                        // record is not known, and neither yields a container.
                        continue;
                    }
                    assert!(
                        whole.contains(&r),
                        "seed {seed}, cut at {cut}: {r:?} is not a record of the whole file"
                    );
                }
            }
        }
    }

    /// A bare record header (NUL, kind, `n`, no name) with its marker.
    fn bare_record(nul: u8, kind: u8, payload: &[u8]) -> Vec<u8> {
        let packed = miniz_oxide::deflate::compress_to_vec(payload, 6);
        let n = (packed.len() + 12) as u32;
        let mut out = vec![nul, kind];
        out.extend_from_slice(&n.to_le_bytes());
        for v in [0x000D_DCE7, n, payload.len() as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&packed);
        out
    }

    #[test]
    fn header_right_after_the_magic_is_a_record() {
        // The header's 6 bytes start at the magic's end: the earliest
        // marker the search reads. Nothing before them reads as a name.
        let mut file = MAGIC.to_vec();
        file.extend(bare_record(0, 0x3c, b"compute"));
        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].offset, MAGIC.len() + 6);
        assert_eq!(found[0].name, None);
        assert_eq!(found[0].kind, 0x3c);
        assert_eq!(found[0].unpack().as_deref(), Some(&b"compute"[..]));
    }

    #[test]
    fn header_reaching_into_the_magic_is_not_a_record() {
        // One header byte short: its NUL would be the magic's last byte, so
        // the marker (one byte earlier than the earliest) is not looked at,
        // whatever the sizes say. A real record after it is still found.
        let mut file = MAGIC.to_vec();
        let mut cut = bare_record(0, 0x32, b"inside the magic");
        cut.remove(0);
        file.extend(&cut);
        assert_eq!(records(&file).count(), 0);
        file.extend(record_bytes("PS_after", 0x32, b"after"));
        let names: Vec<_> = records(&file).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_after")]);
    }

    #[test]
    fn record_without_a_nul_before_its_kind_has_no_name() {
        // The NUL is not checked to find a record, only to read its name.
        for nul in [0x01, 0x41, 0xFF] {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x05PS_x");
            file.extend(bare_record(nul, 0x32, b"no name"));
            let found: Vec<_> = records(&file).collect();
            assert_eq!(found.len(), 1, "nul {nul:#x}");
            assert_eq!(found[0].name, None, "nul {nul:#x}");
            assert_eq!(found[0].unpack().as_deref(), Some(&b"no name"[..]));
        }
    }

    #[test]
    fn header_right_after_a_record_is_a_record_and_one_byte_earlier_is_not() {
        // Back to back: the next header starts where the packed bytes end.
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_first", 0x32, b"first"));
        let end = file.len();
        let mut back_to_back = file.clone();
        back_to_back.extend(bare_record(0, 0x41, b"second"));
        let found: Vec<_> = records(&back_to_back).collect();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].span().end, end);
        assert_eq!(found[1].offset, end + 6);
        assert_eq!(found[1].kind, 0x41);
        assert_eq!(found[1].unpack().as_deref(), Some(&b"second"[..]));

        // Its NUL laid over the first record's last packed byte: that byte
        // is the first record's, so the second is not read.
        let mut overlap = file.clone();
        let mut second = bare_record(0, 0x41, b"second");
        second.remove(0);
        overlap.extend(&second);
        let names: Vec<_> = records(&overlap).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_first")]);
    }

    #[test]
    fn record_stating_no_packed_bytes_is_not_a_record() {
        // `n` of 12 (header only) and 13 hold fewer packed bytes than the
        // smallest deflate stream, even for an empty container.
        for n in [12_u32, 13] {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x05PS_e\0\x32");
            for v in [n, 0x000D_DCE7, n, 0] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&[0x03, 0x00][..(n - 12) as usize]);
            file.extend(record_bytes("PS_after", 0x32, b"after"));
            let names: Vec<_> = records(&file).map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_after")], "n {n}");
        }
    }

    #[test]
    fn packed_bytes_at_the_lower_bound_are_a_record_and_one_fewer_is_not() {
        // A 10320-byte container needs at least 10 packed bytes: 10 bytes
        // that are no deflate stream (block type 3 is reserved) and cover no
        // record are a corrupt record; 9 are not a record at all. Either
        // way the real record right after them is found.
        let unpacked = 1032 * 10;
        assert_eq!(min_packed(unpacked), 10);
        for (packed_len, corrupt) in [(10_u32, true), (9, false)] {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x05PS_l\0\x32");
            let n = 12 + packed_len;
            for v in [n, 0x000D_DCE7, n, unpacked as u32] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&vec![0x07; packed_len as usize]);
            file.extend(record_bytes("PS_after", 0x32, b"after"));
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            if corrupt {
                assert_eq!(names, [Some("PS_l"), Some("PS_after")]);
                assert_eq!(found[0].unpack(), None);
            } else {
                assert_eq!(names, [Some("PS_after")]);
            }
            let last = found.last().expect("PS_after");
            assert_eq!(
                last.unpack().as_deref(),
                Some(&b"after"[..]),
                "{packed_len}"
            );
        }
    }

    #[test]
    fn marker_at_the_end_without_its_sizes_is_not_a_record() {
        // A marker in the file's last bytes, with its header before it but
        // none, some or all of its second `n` and unpacked size cut off: not
        // a record, not a truncated one, and no out-of-bounds read. The
        // record before it is still yielded.
        let mut whole = MAGIC.to_vec();
        whole.extend(record_bytes("PS_first", 0x32, b"first"));
        let first_end = whole.len();
        let mut tail = vec![0, 0x32];
        for v in [64_u32, 0x000D_DCE7, 64, 64] {
            tail.extend_from_slice(&v.to_le_bytes());
        }
        // Header and marker (10 bytes), then up to 7 of the 8 size bytes.
        for keep in 10..tail.len() {
            let mut file = whole.clone();
            file.extend_from_slice(&tail[..keep]);
            let found: Vec<_> = records(&file).collect();
            assert_eq!(found.len(), 1, "{keep} bytes");
            assert_eq!(found[0].span().end, first_end, "{keep} bytes");
            assert_eq!(found[0].unpack().as_deref(), Some(&b"first"[..]));
        }
    }

    #[test]
    fn sizes_that_disagree_are_not_a_record() {
        // The two copies of `n` must match: one more or one less in either
        // copy is bytes that only look like a record, whatever stream
        // follows, and the real record after them is still found.
        let packed = miniz_oxide::deflate::compress_to_vec(b"phantom", 6);
        let n = (packed.len() + 12) as u32;
        for (first, second) in [(n, n + 1), (n, n - 1), (n + 1, n), (n - 1, n)] {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x05PS_d\0\x32");
            for v in [first, 0x000D_DCE7, second, 7] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&packed);
            file.extend(record_bytes("PS_after", 0x32, b"after"));
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_after")], "n {first} and {second}");
            assert_eq!(found[0].unpack().as_deref(), Some(&b"after"[..]));
        }
    }

    #[test]
    fn record_whose_size_reads_as_a_marker_is_one_record() {
        // `n` of 0x000DDCE7 (908519): both copies of it, before and after
        // the real marker, are marker bytes too. Neither is another record
        // (the first's sizes are its name and the real marker, the second's
        // lie inside its own header), and the record unpacks.
        let n = 0x000D_DCE7_u32;
        let packed_len = n as usize - 12;
        // Stored blocks: a few bytes per block on top of the container.
        let mut payload = vec![0x5A_u8; packed_len];
        let mut packed = miniz_oxide::deflate::compress_to_vec(&payload, 0);
        payload.truncate(2 * packed_len - packed.len());
        packed = miniz_oxide::deflate::compress_to_vec(&payload, 0);
        assert!(packed.len() <= packed_len);
        // The stream's end is not checked: pad it out to `n`.
        packed.resize(packed_len, 0);
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x05PS_m\0\x32");
        for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        file.extend(record_bytes("PS_after", 0x32, b"after"));
        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_m"), Some("PS_after")]);
        assert_eq!(found[0].offset, MAGIC.len() + 11);
        assert_eq!(found[0].span().len(), n as usize);
        assert_eq!(found[0].unpack().as_deref(), Some(&payload[..]));
        assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
    }

    #[test]
    fn files_too_short_for_a_header_yield_nothing() {
        // Nothing, part of the magic, the magic alone, and the magic with
        // fewer bytes than a header and marker: no records, no truncated
        // one, and no out-of-bounds read, whatever those bytes are.
        assert_eq!(records(b"").count(), 0);
        for cut in 1..MAGIC.len() {
            assert!(!is_sgd5(&MAGIC[..cut]));
            assert_eq!(records(&MAGIC[..cut]).count(), 0, "{cut} bytes");
        }
        let record = bare_record(0, 0x32, b"cut");
        for keep in 0..10 {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(&record[..keep]);
            assert!(is_sgd5(&file));
            assert_eq!(records(&file).count(), 0, "{keep} bytes");
        }
        // Marker bytes right after the magic have no header before them.
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(&MARKER);
        file.extend_from_slice(&record[6..]);
        assert_eq!(records(&file).count(), 0);
    }

    #[test]
    fn magic_inside_a_record_does_not_restart_the_search() {
        // A whole SGD5 file stored as is (level 0) in a record's packed
        // bytes: its magic and records are that record's container, and the
        // search goes on after the packed bytes, not from the inner magic.
        let mut inner = MAGIC.to_vec();
        inner.extend(record_bytes("PS_inner", 0x32, b"inner container"));
        let packed = miniz_oxide::deflate::compress_to_vec(&inner, 0);
        assert!(memchr::memmem::find(&packed, MAGIC).is_some());
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x09PS_outer\0\x32");
        for v in [n, 0x000D_DCE7, n, inner.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        file.extend(record_bytes("PS_after", 0x32, b"after"));

        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_outer"), Some("PS_after")]);
        assert_eq!(found[0].unpack(), Some(inner.clone()));
        assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        // The stored file, read on its own, is one record.
        let nested: Vec<_> = records(&inner).map(|r| r.name).collect();
        assert_eq!(nested, [Some("PS_inner")]);
    }

    #[test]
    fn container_longer_than_the_probe_is_one_record_whole_or_cut_off() {
        // Only the first PROBE packed bytes are read to tell a stream from
        // noise. A container stored as is (level 0), several probes long,
        // holding a record past the probe: whole, it is one record and the
        // one it stores is its own; cut off past the probe, it is a cut-off
        // record, so the search ends and the stored record is not yielded.
        let mut container: Vec<u8> = (0..3 * PROBE as u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        let stored_at = container.len();
        container.extend(record_bytes("PS_inner", 0x32, b"inner container"));
        let packed = miniz_oxide::deflate::compress_to_vec(&container, 0);
        let marker = memchr::memmem::find(&packed, &MARKER).unwrap();
        assert!(marker > 2 * PROBE, "stored past the probe: {marker}");
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_before", 0x32, b"before"));
        let outer = file.len() + 15;
        file.extend_from_slice(b"\x09PS_outer\0\x32");
        for v in [n, 0x000D_DCE7, n, container.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        file.extend(record_bytes("PS_after", 0x32, b"after"));

        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(
            names,
            [Some("PS_before"), Some("PS_outer"), Some("PS_after")]
        );
        assert_eq!(found[1].offset, outer);
        assert_eq!(found[1].span().len(), n as usize);
        assert_eq!(found[1].unpack(), Some(container.clone()));
        assert!(in_record(&found, outer + 12 + marker));
        assert!(stored_at > PROBE);

        // Cut off anywhere past the probe, up to the stored record's end.
        let packed_at = outer + 12;
        for cut in [PROBE + 1, 2 * PROBE, marker, marker + 12, packed.len() - 1] {
            let names: Vec<_> = records(&file[..packed_at + cut]).map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_before")], "cut {cut} packed bytes in");
        }
    }

    #[test]
    fn search_stays_ended_once_it_ends() {
        // Past the last record, or at a cut-off one, the iterator keeps
        // returning `None` (no rescan from the start, no out-of-bounds
        // read), and a new search over the same bytes finds the same.
        let mut whole = MAGIC.to_vec();
        whole.extend(record_bytes("PS_a", 0x32, b"first"));
        whole.extend(record_bytes("PS_b", 0x41, b"second"));
        let cut = whole.len() - 1;
        for file in [&whole[..], &whole[..cut]] {
            let want = if file.len() == whole.len() { 2 } else { 1 };
            let mut it = records(file);
            for _ in 0..want {
                assert!(it.next().is_some(), "{} bytes", file.len());
            }
            for _ in 0..3 {
                assert_eq!(it.next(), None, "{} bytes", file.len());
            }
            assert_eq!(records(file).count(), want, "{} bytes", file.len());
        }
    }

    #[test]
    fn stream_is_checked_only_as_far_as_the_probe() {
        // At most PROBE bytes are inflated, from at most PROBE packed bytes.
        // Within both, a stream's size is checked exactly; past either,
        // only as far as the probe goes, and such a record is still one
        // (it covers no other), with `unpack` saying it is the wrong size.
        let payload =
            |size: usize| -> Vec<u8> { (0..size as u32).map(|i| (i * 13 % 251) as u8).collect() };
        let deflated = |size: usize| miniz_oxide::deflate::compress_to_vec(&payload(size), 6);

        // Exactly PROBE bytes, packed well within the probe: exact.
        let packed = deflated(PROBE);
        assert!(packed.len() < PROBE);
        assert!(starts_stream(&packed, PROBE));
        assert!(!starts_stream(&packed, PROBE - 1));
        assert!(!starts_stream(&packed, PROBE + 1));

        // One byte more than PROBE: a stated size past it is not checked.
        let packed = deflated(PROBE + 1);
        assert!(starts_stream(&packed, PROBE + 1));
        assert!(!starts_stream(&packed, PROBE), "the limit is reached");
        assert!(starts_stream(&packed, PROBE + 2), "past the probe");

        // Stored as is (level 0), fewer bytes than PROBE take more packed
        // bytes than it: the probe ends inside the stored block.
        let stored = miniz_oxide::deflate::compress_to_vec(&payload(PROBE - 1), 0);
        assert!(stored.len() > PROBE);
        assert!(starts_stream(&stored, PROBE - 1));
        assert!(starts_stream(&stored, PROBE - 2), "cut by the probe");
        // Within the probe, stored bytes past the stated size still show.
        assert!(!starts_stream(&stored[..PROBE / 2], PROBE / 4));

        for (packed, stated) in [(deflated(PROBE + 1), PROBE + 2), (stored, PROBE - 2)] {
            let n = (packed.len() + 12) as u32;
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x05PS_p\0\x32");
            for v in [n, 0x000D_DCE7, n, stated as u32] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&packed);
            file.extend(record_bytes("PS_after", 0x32, b"after"));
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_p"), Some("PS_after")], "stated {stated}");
            assert!(found[0].is_stream(), "stated {stated}");
            assert_eq!(found[0].unpack(), None, "stated {stated}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        }
    }

    #[test]
    fn marker_straddling_a_records_end_is_neither_a_record_nor_hides_the_next() {
        // A record's packed bytes (past the end of its deflate stream, which
        // ignores them) end with the first bytes of a marker, and the next
        // record's header supplies the rest: 3 bytes and its NUL, or 2 bytes
        // and its NUL and kind. With all 4 in the packed bytes, the marker
        // reads its sizes from the next record's header. Whichever, the
        // marker lies in the first record, and both records are found.
        let first = miniz_oxide::deflate::compress_to_vec(b"first", 6);
        for (tail, nul, kind) in [
            (&MARKER[..3], 0x00, 0x32),
            (&MARKER[..2], MARKER[2], MARKER[3]),
            (&MARKER[..], 0x00, 0x41),
        ] {
            let mut packed = first.clone();
            packed.extend_from_slice(tail);
            let n = (packed.len() + 12) as u32;
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x09PS_first\0\x32");
            for v in [n, 0x000D_DCE7, n, 5] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&packed);
            let end = file.len();
            file.extend(bare_record(nul, kind, b"second"));
            let straddling = end - tail.len();
            assert_eq!(
                file[straddling..straddling + 4],
                MARKER,
                "{} bytes",
                tail.len()
            );

            let found: Vec<_> = records(&file).collect();
            assert_eq!(found.len(), 2, "{} bytes", tail.len());
            assert_eq!(found[0].name, Some("PS_first"));
            assert_eq!(found[0].span().end, end);
            assert_eq!(found[0].unpack().as_deref(), Some(&b"first"[..]));
            assert!(in_record(&found, straddling));
            assert_eq!(found[1].offset, end + 6);
            assert_eq!(found[1].kind, kind);
            assert_eq!(found[1].name, None, "{} bytes", tail.len());
            assert_eq!(found[1].unpack().as_deref(), Some(&b"second"[..]));
        }
    }

    #[test]
    fn size_shorter_than_its_own_header_is_not_a_record() {
        // `n` counts the 12 bytes from the marker on, so 0 to 11 cannot be a
        // record's, whatever follows (here a real stream of the stated
        // size): no record, no underflow, and the record after is found.
        let packed = miniz_oxide::deflate::compress_to_vec(b"short", 6);
        for n in 0_u32..12 {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x05PS_s\0\x32");
            for v in [n, 0x000D_DCE7, n, 5] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&packed);
            file.extend(record_bytes("PS_after", 0x32, b"after"));
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_after")], "n {n}");
            assert_eq!(found[0].unpack().as_deref(), Some(&b"after"[..]));
        }
    }

    #[test]
    fn longest_name_a_length_byte_states_is_read_whole() {
        // A length byte of 0xFF: 254 characters and the NUL, the most one
        // byte can state. Read whole, with no byte before it taken for part
        // of the name, whether the record follows the magic or another
        // record's packed bytes.
        let name = format!("PS_{}", "n".repeat(251));
        assert_eq!(name.len() + 1, 0xFF);
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes(&name, 0x32, b"long"));
        file.extend(record_bytes(&name, 0x41, b"again"));
        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), 2);
        for (r, kind, body) in [(&found[0], 0x32, &b"long"[..]), (&found[1], 0x41, b"again")] {
            assert_eq!(r.name, Some(name.as_str()));
            assert_eq!(r.kind, kind);
            assert_eq!(r.unpack().as_deref(), Some(body));
        }
    }

    #[test]
    fn length_byte_that_misstates_the_name_gives_no_name() {
        // `PS_wrong` and its NUL take 9 bytes. A length byte of 0, one off
        // either way, or the most a byte can state is not its distance to
        // the NUL, and no character of the name is either: no name. The
        // record is still found, and so is a named one after it.
        for len in [0_u8, 8, 10, 0xFF] {
            let mut header = record_bytes("PS_wrong", 0x32, b"misnamed");
            header[4] = len;
            let mut file = MAGIC.to_vec();
            file.extend(&header);
            file.extend(record_bytes("PS_after", 0x41, b"after"));
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [None, Some("PS_after")], "length byte {len}");
            assert_eq!(found[0].kind, 0x32);
            assert_eq!(found[0].unpack().as_deref(), Some(&b"misnamed"[..]));
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        }
    }

    #[test]
    fn printable_name_that_is_no_identifier_is_still_read() {
        // Punctuation is printable: with no identifier among the length
        // bytes, the earliest one wins, so the whole name is read.
        for name in ["PS-x.y@1", "PS_a+b", "!"] {
            let mut file = MAGIC.to_vec();
            file.extend(record_bytes(name, 0x32, b"odd"));
            let found: Vec<_> = records(&file).collect();
            assert_eq!(found.len(), 1, "{name}");
            assert_eq!(found[0].name, Some(name));
            assert_eq!(found[0].unpack().as_deref(), Some(&b"odd"[..]));
        }
    }

    #[test]
    fn name_with_unprintable_bytes_is_no_name() {
        // Bytes that are not printable ASCII (UTF-8 past ASCII, a space, a
        // control byte) cut the name off from its length byte: no name,
        // rather than the tail after them. The record is still found.
        for name in ["PS_\u{e9}t\u{e9}", "PS a", "PS_\tb"] {
            let mut file = MAGIC.to_vec();
            file.extend(record_bytes(name, 0x32, b"unprintable"));
            file.extend(record_bytes("PS_after", 0x32, b"after"));
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [None, Some("PS_after")], "{name:?}");
            assert_eq!(found[0].unpack().as_deref(), Some(&b"unprintable"[..]));
        }
    }

    #[test]
    fn a_name_is_not_read_from_before_the_magic_or_from_the_previous_record() {
        // A record whose name has no length byte of its own. The byte that
        // would state it lies before the magic, or in the previous record's
        // padding past its stream: both belong to something else, so the
        // name is not read across them. Read whole, it would be
        // `SGD5PS_x` or `ABPS_x`.
        let tail = |name: &[u8], payload: &[u8]| {
            let packed = miniz_oxide::deflate::compress_to_vec(payload, 6);
            let n = (packed.len() + 12) as u32;
            let mut out = name.to_vec();
            out.extend_from_slice(&[0, 0x32]);
            for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&packed);
            out
        };
        let mut file = vec![0x09];
        file.extend_from_slice(MAGIC);
        file.extend(tail(b"PS_x", b"after the magic"));
        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, None);
        assert_eq!(found[0].unpack().as_deref(), Some(&b"after the magic"[..]));

        // The previous record, padded past its stream with `07 'A' 'B'`.
        let payload = b"padded";
        let mut packed = miniz_oxide::deflate::compress_to_vec(payload, 6);
        packed.extend_from_slice(b"\x07AB");
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x0aPS_padded\0\x32");
        for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        file.extend(tail(b"PS_x", b"after a record"));
        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_padded"), None]);
        assert_eq!(found[0].unpack().as_deref(), Some(&payload[..]));
        assert_eq!(found[1].unpack().as_deref(), Some(&b"after a record"[..]));
    }

    #[test]
    fn every_kind_byte_is_reported_as_it_is() {
        // The kind is not checked to find a record: any byte, even one that
        // reads as a NUL, a length byte or part of the marker, is reported
        // as it is, and the name before it is still read.
        let kinds = [0x00, 0x01, 0x09, 0x0D, 0x32, 0x7F, 0xDC, 0xE7, 0xFF];
        let mut file = MAGIC.to_vec();
        for &kind in &kinds {
            file.extend(record_bytes("PS_kind", kind, &[kind; 40]));
        }
        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), kinds.len());
        for (r, &kind) in found.iter().zip(&kinds) {
            assert_eq!(r.kind, kind);
            assert_eq!(r.name, Some("PS_kind"), "kind {kind:#x}");
            assert_eq!(r.unpack(), Some(vec![kind; 40]), "kind {kind:#x}");
        }
    }

    #[test]
    fn every_name_length_a_byte_can_state_is_read_whole() {
        // Names of 1 to 254 characters: the length byte is a control byte,
        // then punctuation, a digit, a letter, and past 0x7F unprintable.
        // Whichever it is, the name it states is read whole, and the record
        // after it is found with its own name.
        let mut file = MAGIC.to_vec();
        let names: Vec<String> = (1..=254)
            .map(|len| {
                "PS_"
                    .chars()
                    .chain(std::iter::repeat('n'))
                    .take(len)
                    .collect()
            })
            .collect();
        for (i, name) in names.iter().enumerate() {
            file.extend(record_bytes(name, 0x32, &[i as u8; 3]));
        }
        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), names.len());
        for (i, (r, name)) in found.iter().zip(&names).enumerate() {
            assert_eq!(r.name, Some(name.as_str()), "{} characters", name.len());
            assert_eq!(r.unpack(), Some(vec![i as u8; 3]));
        }
    }

    #[test]
    fn magic_in_a_name_is_part_of_the_name() {
        // `SGD5` in a record's name, after the file's own magic: the search
        // started at the first magic, so the name is read whole and the
        // records on both sides of it are found.
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_before", 0x32, b"before"));
        file.extend(record_bytes("PS_SGD5_Wrapped", 0x41, b"named SGD5"));
        file.extend(record_bytes("SGD5", 0x32, b"just the magic"));
        file.extend(record_bytes("PS_after", 0x32, b"after"));
        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(
            names,
            [
                Some("PS_before"),
                Some("PS_SGD5_Wrapped"),
                Some("SGD5"),
                Some("PS_after")
            ]
        );
        assert_eq!(found[1].kind, 0x41);
        assert_eq!(found[1].unpack().as_deref(), Some(&b"named SGD5"[..]));
        assert_eq!(found[2].unpack().as_deref(), Some(&b"just the magic"[..]));
    }

    #[test]
    fn records_in_an_archive_are_the_bare_files_shifted() {
        // The same SGD5 file after 0 to 40 bytes of another file (zeros,
        // printable bytes that read as length bytes, bytes of the marker):
        // the same records, with every offset and span moved by as many
        // bytes, and the same names, kinds and containers.
        let mut bare = MAGIC.to_vec();
        bare.extend(record_bytes("VS_Blit_0011aabb_Wrapped", 0x32, b"first"));
        bare.extend(bare_record(0, 0x3c, b"nameless"));
        bare.extend(record_bytes(
            &format!("PS_{}", "x".repeat(40)),
            0x41,
            &[7; 300],
        ));
        let expected: Vec<_> = records(&bare).collect();
        assert_eq!(expected.len(), 3);
        let fills: [&[u8]; 4] = [&[0], b"\x1aPS_x\0", &MARKER, &[0xE7, 0x2A, 0x0D]];
        for fill in fills {
            for skip in 0..=40 {
                let mut archive: Vec<u8> = fill.iter().copied().cycle().take(skip).collect();
                archive.extend_from_slice(&bare);
                let found: Vec<_> = records(&archive).collect();
                assert_eq!(found.len(), expected.len(), "{skip} bytes of {fill:?}");
                for (r, e) in found.iter().zip(&expected) {
                    assert_eq!(r.offset, e.offset + skip);
                    assert_eq!(r.span(), e.span().start + skip..e.span().end + skip);
                    assert_eq!((r.name, r.kind), (e.name, e.kind));
                    assert_eq!(r.unpack(), e.unpack());
                }
            }
        }
    }

    #[test]
    fn containers_packed_at_every_level_are_records() {
        // The same containers packed at every level (0, stored blocks, to
        // 10), sized around the probe and a stored block's 65535 bytes: all
        // are records, with their names, and unpack whole.
        let sizes = [0, 1, PROBE - 1, PROBE, PROBE + 1, 65_535, 65_536, 70_000];
        let mut file = MAGIC.to_vec();
        let mut want = Vec::new();
        for (i, &size) in sizes.iter().enumerate() {
            let container: Vec<u8> = (0..size as u32)
                .map(|j| (j.wrapping_mul(2_654_435_761) >> (i + 9)) as u8)
                .collect();
            for level in 0..=10u8 {
                let packed = miniz_oxide::deflate::compress_to_vec(&container, level);
                let n = (packed.len() + 12) as u32;
                let name = format!("PS_{size}_{level}");
                file.push(name.len() as u8 + 1);
                file.extend_from_slice(name.as_bytes());
                file.extend_from_slice(&[0, 0x32]);
                for v in [n, 0x000D_DCE7, n, size as u32] {
                    file.extend_from_slice(&v.to_le_bytes());
                }
                file.extend_from_slice(&packed);
                want.push((name, container.clone()));
            }
        }
        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), want.len());
        for (r, (name, container)) in found.iter().zip(&want) {
            assert_eq!(r.name, Some(name.as_str()));
            assert_eq!(r.unpack().as_ref(), Some(container), "{name}");
        }
    }

    #[test]
    fn whole_sgd5_file_stored_in_a_record_is_that_records() {
        // A record whose container is, stored as is (level 0), a whole SGD5
        // file: its magic and records are the outer record's bytes, so the
        // search neither restarts at that magic nor yields the records; each
        // lies inside the outer record, whose container holds them all.
        let mut inner = MAGIC.to_vec();
        inner.extend(record_bytes("PS_inner_a", 0x32, b"first inside"));
        inner.extend(bare_record(0, 0x3c, b"nameless inside"));
        inner.extend(record_bytes("PS_inner_c", 0x41, &[9; 200]));
        let inside: Vec<_> = records(&inner).map(|r| (r.offset, r.name)).collect();
        assert_eq!(inside.len(), 3);

        let packed = miniz_oxide::deflate::compress_to_vec(&inner, 0);
        let stored = memchr::memmem::find(&packed, &inner).expect("stored as is");
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_before", 0x32, b"before"));
        // The marker follows the name, NUL, kind and `n`.
        let outer = file.len() + 13 + 4;
        file.extend_from_slice(b"\x0bPS_archive\0\x32");
        for v in [n, 0x000D_DCE7, n, inner.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        file.extend(record_bytes("PS_after", 0x32, b"after"));

        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(
            names,
            [Some("PS_before"), Some("PS_archive"), Some("PS_after")]
        );
        assert_eq!(found[1].offset, outer);
        let container = found[1].unpack().expect("the stored file");
        assert_eq!(container, inner);
        let base = outer + 12 + stored;
        assert!(in_record(&found, base), "the stored magic");
        for &(offset, _) in &inside {
            assert!(in_record(&found, base + offset));
        }
        let again: Vec<_> = records(&container).map(|r| (r.offset, r.name)).collect();
        assert_eq!(again, inside);
    }

    #[test]
    fn each_record_is_yielded_before_the_bytes_after_it_are_read() {
        // Two real records, then 32000 chained markers whose stated sizes
        // run past the end. The search is lazy: the first record is yielded
        // having checked no stream beyond its own span's end, and the second
        // likewise, however much of the file is left; only asking for a
        // third walks the chain.
        let mut head = MAGIC.to_vec();
        head.extend(record_bytes("PS_first", 0x32, b"first"));
        head.extend(record_bytes("PS_second", 0x41, &[3; 500]));
        let chain = chained_markers(32_000, |_| MAX_UNPACKED as u32, &[]);
        let mut file = head.clone();
        file.extend_from_slice(&chain[MAGIC.len()..]);

        WORK.with(|w| w.set((0, 0)));
        let mut it = records(&file);
        let first = it.next().expect("PS_first");
        assert_eq!(first.name, Some("PS_first"));
        let second = it.next().expect("PS_second");
        assert_eq!(second.name, Some("PS_second"));
        assert_eq!(second.span().end, head.len());
        let (checks, read) = WORK.with(Cell::get);
        assert!(checks <= 4, "{checks} stream checks for two records");
        assert!(read <= 4 * PROBE, "{read} bytes read for two records");
        assert_eq!(first.unpack().as_deref(), Some(&b"first"[..]));
        assert_eq!(second.unpack(), Some(vec![3; 500]));

        // The chain itself is a truncated record: the search ends there,
        // having now checked its first marker for a stream.
        assert_eq!(it.next(), None);
        let (after, _) = WORK.with(Cell::get);
        assert!(after > checks, "the chain was looked at");
    }

    #[test]
    fn zlib_wrapped_container_is_a_record_that_does_not_unpack() {
        // Packed bytes are raw deflate. The same container with a zlib
        // header and checksum around it is not a stream of the stated size;
        // covering no other record, it is still yielded (with its name and
        // span over all its bytes) and does not unpack, and the raw record
        // after it is found and unpacks.
        let container = b"DXBC wrapped in zlib".repeat(8);
        let wrapped = miniz_oxide::deflate::compress_to_vec_zlib(&container, 6);
        assert_eq!(wrapped[0], 0x78, "zlib header");
        let n = (wrapped.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x08PS_zlib\0\x32");
        let marker = file.len() + 4;
        for v in [n, 0x000D_DCE7, n, container.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&wrapped);
        let raw = file.len();
        file.extend(record_bytes("PS_raw", 0x32, &container));

        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_zlib"), Some("PS_raw")]);
        assert_eq!(found[0].span(), marker..raw);
        assert_eq!(found[0].unpack(), None);
        assert_eq!(found[1].unpack(), Some(container));
    }

    #[test]
    fn stream_split_into_blocks_with_empty_ones_between_is_a_record() {
        // Deflate allows any number of blocks, stored ones of length 0
        // among them (a sync flush writes `00 00 00 FF FF`). A stream of
        // stored blocks built by hand, empty ones between them, ended by a
        // fixed-code block from miniz, is a record: it starts a stream
        // however it is cut, unpacks whole, and the record after it is
        // found.
        fn stored(data: &[u8]) -> Vec<u8> {
            let len = data.len() as u16;
            let mut out = vec![0]; // BFINAL 0, stored
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&(!len).to_le_bytes());
            out.extend_from_slice(data);
            out
        }
        let container = b"DXBC in blocks, then a fixed-code tail".to_vec();
        let (head, tail) = container.split_at(13);
        let mut packed = Vec::new();
        packed.extend(stored(&[]));
        packed.extend(stored(&head[..5]));
        packed.extend(stored(&[]));
        packed.extend(stored(&head[5..]));
        packed.extend(stored(&[]));
        // miniz's stream for the tail ends it: a block after a fixed-code
        // one starts mid-byte, so only stored blocks are put together here.
        let fixed = miniz_oxide::deflate::compress_to_vec(tail, 1);
        assert_eq!(fixed[0] & 0b111, 0b011, "one final fixed-code block");
        packed.extend(fixed);
        assert_eq!(
            miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
            Some(container.clone())
        );
        for cut in 0..packed.len() {
            assert!(
                starts_stream(&packed[..cut], container.len()),
                "cut at {cut}"
            );
        }
        assert!(!starts_stream(&packed, container.len() + 1));

        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x0aPS_blocks\0\x32");
        let marker = file.len() + 4;
        for v in [n, 0x000D_DCE7, n, container.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        let after = file.len();
        file.extend(record_bytes("PS_after", 0x32, b"after"));

        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_blocks"), Some("PS_after")]);
        assert_eq!(found[0].span(), marker..after);
        assert_eq!(found[0].unpack(), Some(container));
        assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
    }

    #[test]
    fn stream_broken_after_a_valid_block_is_noise() {
        // Packed bytes that start a valid stored block and then break (a
        // stored block whose NLEN is not LEN's complement, or a block of
        // the reserved type 3) are no deflate stream: alone, such a record
        // covers no other, so it is yielded and does not unpack; laid over
        // a real record stored in its first block, the real one wins. The
        // record after it is found either way.
        fn stored(data: &[u8]) -> Vec<u8> {
            let len = data.len() as u16;
            let mut out = vec![0]; // BFINAL 0, stored
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&(!len).to_le_bytes());
            out.extend_from_slice(data);
            out
        }
        let bad_nlen = [1, 4, 0, 4, 0, b'x', b'y', b'z', b'w'];
        let reserved = [0b111, 0, 0, 0];
        let inner = record_bytes("PS_inner", 0x41, b"stored inside");
        for (what, tail) in [
            ("bad NLEN", &bad_nlen[..]),
            ("reserved type", &reserved[..]),
        ] {
            for (covering, first) in [(false, &b"DXBC head"[..]), (true, &inner[..])] {
                let mut packed = stored(first);
                packed.extend_from_slice(tail);
                let stated = first.len() + 4;
                assert!(!starts_stream(&packed, stated), "{what}");
                assert!(
                    starts_stream(&packed[..1 + 4 + first.len()], stated),
                    "{what}"
                );

                let n = (packed.len() + 12) as u32;
                let mut file = MAGIC.to_vec();
                file.extend_from_slice(b"\x0aPS_broken\0\x32");
                let marker = file.len() + 4;
                for v in [n, 0x000D_DCE7, n, stated as u32] {
                    file.extend_from_slice(&v.to_le_bytes());
                }
                file.extend_from_slice(&packed);
                let after = file.len();
                file.extend(record_bytes("PS_after", 0x32, b"after"));

                let found: Vec<_> = records(&file).collect();
                let names: Vec<_> = found.iter().map(|r| r.name).collect();
                if covering {
                    assert_eq!(names, [Some("PS_inner"), Some("PS_after")], "{what}");
                    let at = marker + 12 + 5 + 4 + 15;
                    assert_eq!(found[0].offset, at, "{what}");
                    assert_eq!(found[0].kind, 0x41, "{what}");
                    assert_eq!(found[0].unpack().as_deref(), Some(&b"stored inside"[..]));
                    assert!(!in_record(&found, marker), "{what}");
                } else {
                    assert_eq!(names, [Some("PS_broken"), Some("PS_after")], "{what}");
                    assert_eq!(found[0].span(), marker..after, "{what}");
                    assert_eq!(found[0].unpack(), None, "{what}");
                }
                assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            }
        }
    }

    /// A non-final stored deflate block holding `data` as is.
    fn stored_block(data: &[u8]) -> Vec<u8> {
        let len = u16::try_from(data.len()).expect("one stored block");
        let mut out = vec![0]; // BFINAL 0, stored
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    /// A file of the magic, a record named `PS_outer` with these packed
    /// bytes stating `unpacked` bytes, and `PS_after`; and the outer
    /// record's marker offset.
    fn outer_then_after(packed: &[u8], unpacked: usize) -> (Vec<u8>, usize) {
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x09PS_outer\0\x32");
        let marker = file.len() + 4;
        for v in [n, 0x000D_DCE7, n, unpacked as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(packed);
        file.extend(record_bytes("PS_after", 0x32, b"after"));
        (file, marker)
    }

    #[test]
    fn record_split_across_stored_blocks_is_inside_the_one_storing_it() {
        // A container holding a record, stored as is in two stored blocks
        // split at every byte of that record: a block header (5 bytes) may
        // cut its header, its marker, its sizes or its packed bytes. Where
        // the marker survives whole, the record it starts is no stream
        // (its packed bytes are cut) or is the outer one's own: either way
        // the outer record is the one yielded, it unpacks to the container
        // whole, and the record after it is found. Offsets past every
        // record, the end of the file among them, are in none.
        let inner = record_bytes("PS_inner", 0x41, &[9; 64]);
        let mut container = b"DXBC head ".to_vec();
        let inner_at = container.len();
        container.extend_from_slice(&inner);
        container.extend_from_slice(b" DXBC tail");
        for split in inner_at..=inner_at + inner.len() {
            let (head, tail) = container.split_at(split);
            let mut packed = stored_block(head);
            packed.extend(miniz_oxide::deflate::compress_to_vec(tail, 0));
            assert!(starts_stream(&packed, container.len()), "split {split}");

            let (file, marker) = outer_then_after(&packed, container.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "split {split}");
            assert_eq!(found[0].offset, marker, "split {split}");
            assert_eq!(found[0].unpack(), Some(container.clone()), "split {split}");
            assert_eq!(found[1].span().end, file.len(), "split {split}");
            if let Some(at) = memchr::memmem::find(&file[marker + 1..], &MARKER) {
                let at = marker + 1 + at;
                if at < found[0].span().end {
                    assert!(in_record(&found, at), "split {split}");
                }
            }
            for past in [file.len(), file.len() + 1, usize::MAX] {
                assert!(!in_record(&found, past), "split {split}, offset {past}");
            }
        }
    }

    #[test]
    fn stream_broken_past_the_probe_still_hides_the_record_it_stores() {
        // Only the first PROBE packed bytes tell a stream from noise. A
        // record whose stored block (more than a probe long, holding a
        // record past the probe) is followed by a broken block reads as a
        // stream: it wins over the record it stores, is yielded with its
        // whole span, and does not unpack. Broken within the probe, the
        // same bytes are noise and the stored record wins instead.
        let inner = record_bytes("PS_inner", 0x32, b"stored past the probe");
        let bad_nlen = [1, 4, 0, 4, 0, b'x', b'y', b'z', b'w'];
        for lead in [PROBE + 64, PROBE / 4] {
            let mut first: Vec<u8> = (0..lead as u32)
                .map(|i| (i.wrapping_mul(2_654_435_761) >> 11) as u8)
                .collect();
            first.extend_from_slice(&inner);
            assert!(memchr::memmem::find(&first[..lead], &MARKER).is_none());
            let mut packed = stored_block(&first);
            packed.extend_from_slice(&bad_nlen);
            let stated = first.len() + 4;
            let broken_at = packed.len() - bad_nlen.len();
            assert_eq!(
                starts_stream(&packed, stated),
                broken_at >= PROBE,
                "lead {lead}"
            );

            let (file, marker) = outer_then_after(&packed, stated);
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            if broken_at >= PROBE {
                assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "lead {lead}");
                assert_eq!(found[0].span().end, marker + 12 + packed.len());
                assert_eq!(found[0].unpack(), None, "lead {lead}");
                let stored = marker + 12 + 5 + lead + 19;
                assert_eq!(&file[stored..stored + 4], &MARKER);
                assert!(in_record(&found, stored), "lead {lead}");
            } else {
                assert_eq!(names, [Some("PS_inner"), Some("PS_after")], "lead {lead}");
                assert_eq!(
                    found[0].unpack().as_deref(),
                    Some(&b"stored past the probe"[..])
                );
                assert!(!in_record(&found, marker), "lead {lead}");
            }
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        }
    }

    #[test]
    fn stated_size_cutting_a_stream_short_ends_the_record_there() {
        // A record's stated size, not its stream, says where it ends. Cut
        // short by that size at every byte, a stream packed at any level is
        // still the start of one: the record is yielded with the span its
        // size states, does not unpack, and the record after it is found in
        // the bytes the stream would have gone on through.
        let container: Vec<u8> = (0..200_u32).map(|i| (i * 7 % 61) as u8).collect();
        for level in [0, 1, 6, 10] {
            let whole = miniz_oxide::deflate::compress_to_vec(&container, level);
            for keep in min_packed(container.len())..=whole.len() {
                let (mut file, marker) = outer_then_after(&whole[..keep], container.len());
                file.splice(
                    marker + 12 + keep..marker + 12 + keep,
                    whole[keep..].to_vec(),
                );
                let what = format!("level {level}, {keep} of {} bytes", whole.len());
                let found: Vec<_> = records(&file).collect();
                let names: Vec<_> = found.iter().map(|r| r.name).collect();
                assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
                assert_eq!(found[0].span(), marker..marker + 12 + keep, "{what}");
                assert_eq!(
                    found[0].unpack(),
                    (keep == whole.len()).then(|| container.clone()),
                    "{what}"
                );
                assert_eq!(found[1].span().end, file.len(), "{what}");
                assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            }
        }
    }

    #[test]
    fn record_stored_past_where_a_stated_size_cuts_its_block_is_the_next_record() {
        // A record stored as is in a container whose record states fewer
        // packed bytes than its stored block holds: where the stated end
        // falls at or before the stored record's header, the stored record
        // is the next one, read whole from the bytes after the cut; where it
        // falls inside that header or later, the stored record starts inside
        // the outer one, which is a stream (cut short) and wins.
        let inner = record_bytes("PS_inner", 0x41, b"stored inside");
        let mut container = b"DXBC head ".to_vec();
        let inner_at = container.len();
        container.extend_from_slice(&inner);
        container.extend_from_slice(b" DXBC tail");
        let whole = stored_block(&container);
        // The inner header (NUL, kind, n) is the 6 bytes before its marker.
        let inner_marker = 5 + inner_at + memchr::memmem::find(&inner, &MARKER).unwrap();
        for keep in min_packed(container.len())..whole.len() {
            let (mut file, marker) = outer_then_after(&whole[..keep], container.len());
            let end = marker + 12 + keep;
            file.splice(end..end, whole[keep..].to_vec());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(found[0].name, Some("PS_outer"), "keep {keep}");
            assert_eq!(found[0].span(), marker..end, "keep {keep}");
            assert_eq!(found[0].unpack(), None, "keep {keep}");
            let stored = marker + 12 + inner_marker;
            assert_eq!(&file[stored..stored + 4], &MARKER, "keep {keep}");
            if keep <= inner_marker - 6 {
                assert_eq!(names.len(), 3, "keep {keep}");
                assert_eq!(found[1].offset, stored, "keep {keep}");
                assert_eq!(found[1].kind, 0x41, "keep {keep}");
                assert_eq!(found[1].unpack().as_deref(), Some(&b"stored inside"[..]));
                assert!(!in_record(&found, stored - 1), "keep {keep}");
                if keep < inner_marker - 6 - 13 {
                    // Its length byte and name are past the cut too.
                    assert_eq!(found[1].name, Some("PS_inner"), "keep {keep}");
                }
            } else {
                assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "keep {keep}");
                assert!(!found.iter().any(|r| r.offset == stored), "keep {keep}");
            }
            assert_eq!(found.last().unwrap().name, Some("PS_after"), "keep {keep}");
        }
    }

    #[test]
    fn stream_broken_where_the_probe_ends_is_noise_only_if_the_probe_sees_the_break() {
        // A stored block's header is 5 bytes; a broken one (its NLEN not
        // the complement of its LEN) is told only once all 5 are read. Put
        // just inside the probe, the break makes the outer record noise and
        // the record it stores wins; one byte later, the probe ends inside
        // the broken header, which reads as a stream cut off, and the outer
        // record wins over the one it stores.
        let inner = record_bytes("PS_inner", 0x32, b"stored early");
        let bad_header = [1, 4, 0, 4, 0];
        for broken_at in [PROBE - 6, PROBE - 5, PROBE - 4, PROBE - 1, PROBE] {
            let mut first = b"DXBC head ".to_vec();
            let inner_at = first.len();
            first.extend_from_slice(&inner);
            first.resize(broken_at - 5, b'.');
            let mut packed = stored_block(&first);
            assert_eq!(packed.len(), broken_at);
            packed.extend_from_slice(&bad_header);
            packed.extend_from_slice(b"wxyz");
            let stated = first.len() + 4;
            let seen = broken_at + bad_header.len() <= PROBE;
            assert_eq!(
                starts_stream(&packed, stated),
                !seen,
                "break at {broken_at}"
            );

            let (file, marker) = outer_then_after(&packed, stated);
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            let stored =
                marker + 12 + 5 + inner_at + memchr::memmem::find(&inner, &MARKER).unwrap();
            assert_eq!(&file[stored..stored + 4], &MARKER);
            if seen {
                assert_eq!(
                    names,
                    [Some("PS_inner"), Some("PS_after")],
                    "break at {broken_at}"
                );
                assert_eq!(found[0].offset, stored, "break at {broken_at}");
                assert_eq!(found[0].unpack().as_deref(), Some(&b"stored early"[..]));
                assert!(!in_record(&found, marker), "break at {broken_at}");
            } else {
                assert_eq!(
                    names,
                    [Some("PS_outer"), Some("PS_after")],
                    "break at {broken_at}"
                );
                assert_eq!(found[0].span(), marker..marker + 12 + packed.len());
                assert_eq!(found[0].unpack(), None, "break at {broken_at}");
                assert!(in_record(&found, stored), "break at {broken_at}");
            }
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "break at {broken_at}");
        }
    }

    #[test]
    fn stream_of_another_size_over_a_stored_record_is_noise() {
        // A container holding a record, stored as is in one final stored
        // block, under a record stating one byte more or one fewer than the
        // container: its bytes are no stream of that size, so it is noise
        // and the record it stores wins. Stating the container's size
        // exactly, it is a stream and wins over the record it stores.
        let inner = record_bytes("PS_inner", 0x41, b"stored inside");
        let mut container = b"DXBC head ".to_vec();
        let inner_at = container.len();
        container.extend_from_slice(&inner);
        container.extend_from_slice(b" DXBC tail");
        let packed = miniz_oxide::deflate::compress_to_vec(&container, 0);
        for stated in [container.len() - 1, container.len(), container.len() + 1] {
            let (file, marker) = outer_then_after(&packed, stated);
            let stored =
                marker + 12 + 5 + inner_at + memchr::memmem::find(&inner, &MARKER).unwrap();
            assert_eq!(&file[stored..stored + 4], &MARKER, "stated {stated}");
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            if stated == container.len() {
                assert_eq!(
                    names,
                    [Some("PS_outer"), Some("PS_after")],
                    "stated {stated}"
                );
                assert_eq!(found[0].unpack(), Some(container.clone()));
                assert!(in_record(&found, stored), "stated {stated}");
            } else {
                assert_eq!(
                    names,
                    [Some("PS_inner"), Some("PS_after")],
                    "stated {stated}"
                );
                assert_eq!(found[0].offset, stored, "stated {stated}");
                assert_eq!(found[0].kind, 0x41, "stated {stated}");
                assert_eq!(found[0].unpack().as_deref(), Some(&b"stored inside"[..]));
                assert!(!in_record(&found, marker), "stated {stated}");
            }
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "stated {stated}");
        }
    }

    #[test]
    fn two_records_stored_in_one_container_are_both_its_or_both_yielded() {
        // A container holding two records back to back, stored as is in one
        // final stored block. Stating the container's size, the outer record
        // is a stream and wins over both: neither is yielded, both markers
        // are inside it. Stating one byte more or fewer, it is noise laid
        // over the first stored record, which wins; the search goes on
        // after it and finds the second, then the record after the outer.
        let first = record_bytes("PS_first", 0x41, b"stored first");
        let second = record_bytes("PS_second", 0x3c, b"stored second");
        let mut container = b"DXBC head ".to_vec();
        let first_at = container.len();
        container.extend_from_slice(&first);
        let second_at = container.len();
        container.extend_from_slice(&second);
        container.extend_from_slice(b" DXBC tail");
        let packed = miniz_oxide::deflate::compress_to_vec(&container, 0);
        let marker_in = |r: &[u8]| memchr::memmem::find(r, &MARKER).unwrap();
        for stated in [container.len() - 1, container.len(), container.len() + 1] {
            let (file, marker) = outer_then_after(&packed, stated);
            let stored = [
                marker + 12 + 5 + first_at + marker_in(&first),
                marker + 12 + 5 + second_at + marker_in(&second),
            ];
            for at in stored {
                assert_eq!(&file[at..at + 4], &MARKER, "stated {stated}");
            }
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            if stated == container.len() {
                assert_eq!(
                    names,
                    [Some("PS_outer"), Some("PS_after")],
                    "stated {stated}"
                );
                assert_eq!(found[0].unpack(), Some(container.clone()));
                for at in stored {
                    assert!(in_record(&found, at), "stated {stated}");
                }
            } else {
                assert_eq!(
                    names,
                    [Some("PS_first"), Some("PS_second"), Some("PS_after")],
                    "stated {stated}"
                );
                assert_eq!(found[0].offset, stored[0], "stated {stated}");
                assert_eq!(found[1].offset, stored[1], "stated {stated}");
                assert_eq!(found[0].kind, 0x41, "stated {stated}");
                assert_eq!(found[1].kind, 0x3c, "stated {stated}");
                // The second starts right where the first one's span ends.
                assert_eq!(found[0].span().end + marker_in(&second), found[1].offset);
                assert_eq!(found[0].unpack().as_deref(), Some(&b"stored first"[..]));
                assert_eq!(found[1].unpack().as_deref(), Some(&b"stored second"[..]));
                assert!(!in_record(&found, marker), "stated {stated}");
            }
            let last = found.last().unwrap();
            assert_eq!(last.unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(last.span().end, file.len(), "stated {stated}");
        }
    }

    /// Bits written least significant first, as deflate packs them.
    #[derive(Default)]
    struct Bits {
        out: Vec<u8>,
        used: u32,
    }

    impl Bits {
        /// `count` bits of `value`, its low bit first (block headers,
        /// extra bits).
        fn put(&mut self, value: u32, count: u32) {
            for i in 0..count {
                if self.used.is_multiple_of(8) {
                    self.out.push(0);
                }
                let bit = (value >> i) & 1;
                *self.out.last_mut().unwrap() |= (bit as u8) << (self.used % 8);
                self.used += 1;
            }
        }

        /// A Huffman code of `count` bits, its high bit first.
        fn code(&mut self, code: u32, count: u32) {
            for i in (0..count).rev() {
                self.put((code >> i) & 1, 1);
            }
        }
    }

    /// One final fixed-code block: the literal `A`, then a copy of 3 bytes
    /// from `distance` (1 to 4) bytes back, then the end of the block.
    fn literal_then_copy(distance: u32) -> Vec<u8> {
        let mut bits = Bits::default();
        bits.put(1, 1); // BFINAL
        bits.put(1, 2); // fixed codes
        bits.code(0x30 + u32::from(b'A'), 8); // literals 0-143: 8 bits from 0x30
        bits.code(1, 7); // length code 257: 3 bytes
        bits.code(distance - 1, 5); // distance codes 0-3: 1-4, no extra bits
        bits.code(0, 7); // end of block (256)
        bits.out
    }

    #[test]
    fn copy_from_before_the_start_of_the_container_is_noise() {
        // A copy may only reach back over bytes already unpacked. One byte
        // back from the first literal is a stream of `AAAA`; two bytes back
        // reaches before the container's start, and the stream breaks
        // there: such a record covers no other, so it is yielded and does
        // not unpack, and the record after it is found either way.
        let good = literal_then_copy(1);
        assert_eq!(
            miniz_oxide::inflate::decompress_to_vec(&good)
                .ok()
                .as_deref(),
            Some(&b"AAAA"[..])
        );
        for distance in [2, 3, 4] {
            let bad = literal_then_copy(distance);
            assert_eq!(bad.len(), good.len());
            assert!(miniz_oxide::inflate::decompress_to_vec(&bad).is_err());
            assert!(!starts_stream(&bad, 4), "distance {distance}");
        }
        assert!(starts_stream(&good, 4));
        assert!(!starts_stream(&good, 3));
        assert!(!starts_stream(&good, 5));

        for distance in [1, 2, 4] {
            let (file, marker) = outer_then_after(&literal_then_copy(distance), 4);
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(
                names,
                [Some("PS_outer"), Some("PS_after")],
                "distance {distance}"
            );
            assert_eq!(found[0].offset, marker, "distance {distance}");
            let want = (distance == 1).then(|| b"AAAA".to_vec());
            assert_eq!(found[0].unpack(), want, "distance {distance}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "distance {distance}");
        }
    }

    #[test]
    fn stream_without_a_final_block_is_a_record_that_does_not_unpack() {
        // Packed bytes of non-final stored blocks only (a stream left after
        // a sync flush, never finished) hold the whole container but no
        // end: inflating them runs out of bytes, as a cut-off stream does,
        // so they are a record, but one that does not unpack. Its span is
        // what its sizes state, and the record after it is found.
        let container = b"DXBC never finished".to_vec();
        let (head, tail) = container.split_at(7);
        for packed in [
            stored_block(&container),
            [stored_block(head), stored_block(tail)].concat(),
            [stored_block(&container), stored_block(&[])].concat(),
        ] {
            assert!(miniz_oxide::inflate::decompress_to_vec(&packed).is_err());
            assert!(starts_stream(&packed, container.len()));

            let (file, marker) = outer_then_after(&packed, container.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")]);
            assert_eq!(found[0].span(), marker..marker + 12 + packed.len());
            assert_eq!(found[0].unpack(), None);
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));

            // Ended by an empty final stored block, the same bytes unpack.
            let mut ended = packed.clone();
            ended.extend_from_slice(&[1, 0, 0, 0xFF, 0xFF]);
            let (file, _) = outer_then_after(&ended, container.len());
            let found: Vec<_> = records(&file).collect();
            assert_eq!(found.len(), 2);
            assert_eq!(found[0].unpack(), Some(container.clone()));
        }
    }

    /// One final fixed-code block: the literal `A`, then a copy coded by
    /// length symbol `length` (257 to 287) and distance code `distance` (0
    /// to 31), neither with extra bits read (285 is 258 bytes; the copy
    /// takes them from 1 byte back), then the end of the block.
    fn literal_then_coded_copy(length: u32, distance: u32) -> Vec<u8> {
        let mut bits = Bits::default();
        bits.put(1, 1); // BFINAL
        bits.put(1, 2); // fixed codes
        bits.code(0x30 + u32::from(b'A'), 8); // literals 0-143: 8 bits from 0x30
        match length {
            257..=279 => bits.code(length - 256, 7), // 256-279: 7 bits from 0
            _ => bits.code(0xC0 + length - 280, 8),  // 280-287: 8 bits from 0xC0
        }
        bits.code(distance, 5);
        bits.code(0, 7); // end of block (256)
        bits.out
    }

    #[test]
    fn copy_coded_with_symbols_deflate_never_uses_is_noise() {
        // Fixed codes have room for length symbols 286 and 287 and distance
        // codes 30 and 31, but deflate gives them no meaning: a stream using
        // one breaks there, though its bits are laid out as a real one's.
        // Such a record covers no other, so it is yielded and does not
        // unpack, and the record after it is found either way.
        let good = literal_then_coded_copy(285, 0);
        let want = vec![b'A'; 259];
        assert_eq!(
            miniz_oxide::inflate::decompress_to_vec(&good).ok(),
            Some(want.clone())
        );
        assert!(starts_stream(&good, want.len()));
        let bad = [(286, 0), (287, 0), (285, 30), (285, 31), (287, 31)];
        for (length, distance) in bad {
            let what = format!("length {length}, distance {distance}");
            let packed = literal_then_coded_copy(length, distance);
            assert_eq!(packed.len(), good.len(), "{what}");
            assert!(
                miniz_oxide::inflate::decompress_to_vec(&packed).is_err(),
                "{what}"
            );
            assert!(!starts_stream(&packed, want.len()), "{what}");
        }

        for (length, distance) in [(285, 0)].into_iter().chain(bad) {
            let what = format!("length {length}, distance {distance}");
            let packed = literal_then_coded_copy(length, distance);
            let (file, marker) = outer_then_after(&packed, want.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            let unpacked = (length, distance) == (285, 0);
            assert_eq!(found[0].unpack(), unpacked.then(|| want.clone()), "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
    }

    /// Canonical Huffman codes for these code lengths (0: no code), as
    /// deflate assigns them: shorter codes first, then by symbol. Lengths
    /// that are no code (oversubscribed) still get codes, which collide.
    fn canonical(lengths: &[u32]) -> Vec<u32> {
        let mut next = 0;
        let mut codes = vec![0; lengths.len()];
        for len in 1..=15 {
            for (sym, _) in lengths.iter().enumerate().filter(|&(_, &l)| l == len) {
                codes[sym] = next;
                next += 1;
            }
            next <<= 1;
        }
        codes
    }

    /// Code lengths as code-length symbols with their extra bits' values:
    /// runs of 11 zeros and more as 18, of 3 to 10 as 17, the rest as is.
    fn length_symbols(lengths: &[u32]) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < lengths.len() {
            let run = lengths[i..].iter().take_while(|&&l| l == 0).count();
            match run {
                11.. => {
                    let run = run.min(138);
                    out.push((18, run as u32 - 11));
                    i += run;
                }
                3..=10 => {
                    out.push((17, run as u32 - 3));
                    i += run;
                }
                _ => {
                    out.push((lengths[i], 0));
                    i += 1;
                }
            }
        }
        out
    }

    /// Code-length symbols in the order a dynamic block's header sends
    /// their own code lengths.
    const LENGTH_ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];

    /// One final dynamic-code block of 257 literal/length codes and one
    /// distance code: the code-length code of lengths `clen` (by symbol),
    /// the code-length symbols `seq`, then `data` coded with literal/length
    /// code lengths `lit` (literals, then 256 for the end of the block).
    fn dynamic_block(clen: &[u32; 19], seq: &[(u32, u32)], lit: &[u32], data: &[u32]) -> Vec<u8> {
        dynamic_block_of(257, 1, clen, seq, lit, data)
    }

    /// [`dynamic_block`] whose header states `hlit` literal/length codes
    /// (257 to 288) and `hdist` distance codes (1 to 32).
    fn dynamic_block_of(
        hlit: u32,
        hdist: u32,
        clen: &[u32; 19],
        seq: &[(u32, u32)],
        lit: &[u32],
        data: &[u32],
    ) -> Vec<u8> {
        let mut bits = dynamic_header(hlit, hdist, clen, seq);
        let lit_codes = canonical(lit);
        for &sym in data {
            bits.code(lit_codes[sym as usize], lit[sym as usize]);
        }
        bits.out
    }

    /// The header of a final dynamic-code block, as [`dynamic_block_of`]
    /// sends it, ready for the block's data.
    fn dynamic_header(hlit: u32, hdist: u32, clen: &[u32; 19], seq: &[(u32, u32)]) -> Bits {
        dynamic_header_sending(19, hlit, hdist, clen, seq)
    }

    /// [`dynamic_header`] whose HCLEN sends only the first `hclen` (4 to
    /// 19) code-length code lengths, in [`LENGTH_ORDER`]; `seq` is still
    /// coded with the code of all of `clen`.
    fn dynamic_header_sending(
        hclen: usize,
        hlit: u32,
        hdist: u32,
        clen: &[u32; 19],
        seq: &[(u32, u32)],
    ) -> Bits {
        let mut bits = Bits::default();
        bits.put(1, 1); // BFINAL
        bits.put(2, 2); // dynamic codes
        bits.put(hlit - 257, 5); // HLIT
        bits.put(hdist - 1, 5); // HDIST
        bits.put(hclen as u32 - 4, 4); // HCLEN
        for sym in &LENGTH_ORDER[..hclen] {
            bits.put(clen[*sym], 3);
        }
        let clen_codes = canonical(clen);
        for &(sym, extra) in seq {
            bits.code(clen_codes[sym as usize], clen[sym as usize]);
            match sym {
                16 => bits.put(extra, 2),
                17 => bits.put(extra, 3),
                18 => bits.put(extra, 7),
                _ => {}
            }
        }
        bits
    }

    #[test]
    fn dynamic_block_whose_code_lengths_form_no_code_is_noise() {
        // A dynamic block sends its own codes, as code lengths: a set of
        // lengths that is no prefix code (more codes of a length than fit,
        // or a literal/length code without the end of the block), or a
        // code-length sequence that repeats a length before there is one or
        // runs past the codes the header states, breaks the stream, though
        // its bits are laid out as a real one's. Such a record covers no
        // other, so it is yielded and does not unpack, and the record after
        // it is found either way.
        let a = u32::from(b'A');
        // Code-length code: 18 in 1 bit, 0 and 1 in 2 (a complete code).
        let mut clen = [0; 19];
        (clen[18], clen[0], clen[1]) = (1, 2, 2);
        // `A` and the end of the block in 1 bit each; one distance code,
        // unused, of length 0.
        let mut lit = vec![0; 258];
        (lit[a as usize], lit[256]) = (1, 1);
        let data = [a, a, a, a, 256];
        let want = b"AAAA".to_vec();

        let good = dynamic_block(&clen, &length_symbols(&lit), &lit[..257], &data);
        assert_eq!(
            miniz_oxide::inflate::decompress_to_vec(&good).ok(),
            Some(want.clone())
        );
        assert!(starts_stream(&good, want.len()));

        let mut bad = Vec::new();
        // Code-length code of three 1-bit codes.
        let mut over = clen;
        over[0] = 1;
        over[1] = 1;
        bad.push((
            "oversubscribed code-length code",
            dynamic_block(&over, &length_symbols(&lit), &lit[..257], &data),
        ));
        // Literal/length code of three 1-bit codes.
        let mut three = lit.clone();
        three[usize::from(b'B')] = 1;
        bad.push((
            "oversubscribed literal/length code",
            dynamic_block(&clen, &length_symbols(&three), &three[..257], &data),
        ));
        // No code for the end of the block.
        let mut no_end = lit.clone();
        no_end[256] = 0;
        bad.push((
            "no end of block",
            dynamic_block(&clen, &length_symbols(&no_end), &lit[..257], &data),
        ));
        // Repeat of the previous length (16) as the first symbol.
        let mut rep_clen = clen;
        rep_clen[16] = 2;
        rep_clen[1] = 3;
        rep_clen[0] = 3;
        let mut rep = vec![(16, 0)];
        rep.extend(length_symbols(&lit));
        bad.push((
            "repeat with nothing before",
            dynamic_block(&rep_clen, &rep, &lit[..257], &data),
        ));
        // A run of zeros past the 258 lengths the header states.
        let mut past = length_symbols(&lit);
        let last = past.pop().unwrap();
        assert_eq!(last, (0, 0), "the distance code's length");
        past.push((18, 0));
        bad.push((
            "zeros past the stated codes",
            dynamic_block(&clen, &past, &lit[..257], &data),
        ));

        for (what, packed) in &bad {
            assert!(
                miniz_oxide::inflate::decompress_to_vec(packed).is_err(),
                "{what}"
            );
            assert!(!starts_stream(packed, want.len()), "{what}");
        }

        for (what, packed) in [("good", &good)]
            .into_iter()
            .chain(bad.iter().map(|(w, p)| (*w, p)))
        {
            let (file, marker) = outer_then_after(packed, want.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            let unpacked = what == "good";
            assert_eq!(found[0].unpack(), unpacked.then(|| want.clone()), "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
    }

    #[test]
    fn dynamic_block_stating_codes_deflate_does_not_have_is_noise() {
        // A dynamic block's header has room for 288 literal/length codes
        // and 32 distance codes, but deflate has only 286 and 30: a header
        // stating more breaks the stream, though every code it sends is a
        // valid length and the data never uses the extra symbols. Such a
        // record covers no other, so it is yielded and does not unpack, and
        // the record after it is found either way.
        let a = u32::from(b'A');
        let mut clen = [0; 19];
        (clen[18], clen[0], clen[1]) = (1, 2, 2);
        let data = [a, a, a, a, 256];
        let want = b"AAAA".to_vec();
        let block = |hlit: u32, hdist: u32| {
            let mut lengths = vec![0; (hlit + hdist) as usize];
            (lengths[a as usize], lengths[256]) = (1, 1);
            let lit = &lengths[..hlit as usize];
            dynamic_block_of(hlit, hdist, &clen, &length_symbols(&lengths), lit, &data)
        };
        let mut cases = Vec::new();
        for hlit in [257, 285, 286, 287, 288] {
            for hdist in [1, 29, 30, 31, 32] {
                cases.push((hlit, hdist, hlit <= 286 && hdist <= 30));
            }
        }
        for &(hlit, hdist, valid) in &cases {
            let what = format!("{hlit} literal/length codes, {hdist} distance codes");
            let packed = block(hlit, hdist);
            assert_eq!(
                miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                valid.then(|| want.clone()),
                "{what}"
            );
            assert_eq!(starts_stream(&packed, want.len()), valid, "{what}");

            let (file, marker) = outer_then_after(&packed, want.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            assert_eq!(found[0].unpack(), valid.then(|| want.clone()), "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
    }

    #[test]
    fn code_length_run_crossing_into_the_distance_codes_is_a_stream() {
        // A dynamic block sends its literal/length and distance code lengths
        // as one sequence, and a run (16: the previous length again) may
        // carry on from the last literal/length codes into the distance
        // codes. Such a block is a stream; the same run one longer runs
        // past the codes the header states and breaks it. Either way the
        // record covers no other, so it is yielded, unpacking only if it is
        // a stream, and the record after it is found.
        let a = u32::from(b'A');
        // Literal/length code: `A` in 1 bit, 255 and the end of the block
        // in 2; four distance codes in 2 bits each, all complete codes.
        let mut lengths = vec![0; 257 + 4];
        lengths[a as usize] = 1;
        for l in &mut lengths[255..] {
            *l = 2;
        }
        // Code-length code: 18 in 1 bit, 16 in 2, 1 and 2 in 3.
        let mut clen = [0; 19];
        (clen[18], clen[16], clen[1], clen[2]) = (1, 2, 3, 3);
        let head = length_symbols(&lengths[..256]);
        assert_eq!(head.last(), Some(&(2, 0)), "255's length, sent as is");
        assert!(head.iter().all(|&(sym, _)| matches!(sym, 1 | 2 | 18)));
        let data = [a, a, 255, a, 256];
        let want = vec![b'A', b'A', 255, b'A'];
        // 16's 2 extra bits state 3 to 6 repeats: 256 and the four distance
        // codes are 5; 6 runs one past them.
        let block = |repeats: u32| {
            let mut seq = head.clone();
            seq.push((16, repeats - 3));
            dynamic_block_of(257, 4, &clen, &seq, &lengths[..257], &data)
        };
        for (repeats, valid) in [(5, true), (6, false)] {
            let what = format!("{repeats} repeats");
            let packed = block(repeats);
            assert_eq!(
                miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                valid.then(|| want.clone()),
                "{what}"
            );
            assert_eq!(starts_stream(&packed, want.len()), valid, "{what}");

            let (file, marker) = outer_then_after(&packed, want.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            assert_eq!(found[0].unpack(), valid.then(|| want.clone()), "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
    }

    /// A non-final stored block of `before`, then one final fixed-code
    /// block: a copy of 3 bytes from `distance` (1 to 4) bytes back, then
    /// the end of the block.
    fn stored_then_copy(before: &[u8], distance: u32) -> Vec<u8> {
        let mut bits = Bits::default();
        bits.put(1, 1); // BFINAL
        bits.put(1, 2); // fixed codes
        bits.code(1, 7); // length code 257: 3 bytes
        bits.code(distance - 1, 5); // distance codes 0-3: 1-4, no extra bits
        bits.code(0, 7); // end of block (256)
        [stored_block(before), bits.out].concat()
    }

    #[test]
    fn copy_reaching_back_into_an_earlier_block_is_a_stream() {
        // A copy may reach back over every byte unpacked so far, earlier
        // blocks' included: a final block that opens with a copy from a
        // stored block before it is a stream, as far back as that block
        // goes. One byte further reaches before the container's start, and
        // the stream breaks there: such a record covers no other, so it is
        // yielded and does not unpack, and the record after it is found
        // either way.
        let before = b"xyz";
        for distance in 1..=4 {
            let what = format!("distance {distance}");
            let packed = stored_then_copy(before, distance);
            let valid = distance as usize <= before.len();
            let want = valid.then(|| {
                let mut out = before.to_vec();
                for _ in 0..3 {
                    out.push(out[out.len() - distance as usize]);
                }
                out
            });
            assert_eq!(
                miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                want,
                "{what}"
            );
            assert_eq!(starts_stream(&packed, 6), valid, "{what}");

            let (file, marker) = outer_then_after(&packed, 6);
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(found[0].offset, marker, "{what}");
            assert_eq!(found[0].unpack(), want, "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
        // With nothing stored before it, even one byte back is too far.
        assert!(!starts_stream(&stored_then_copy(b"", 1), 3));
    }

    /// A non-final stored block of `before`, then one final fixed-code
    /// block: a copy of 3 bytes coded by distance code `code` (4 to 29) and
    /// `extra`, the value of its extra bits, then the end of the block.
    fn stored_then_far_copy(before: &[u8], code: u32, extra: u32) -> Vec<u8> {
        let mut bits = Bits::default();
        bits.put(1, 1); // BFINAL
        bits.put(1, 2); // fixed codes
        bits.code(1, 7); // length code 257: 3 bytes, no extra bits
        bits.code(code, 5);
        bits.put(extra, code / 2 - 1); // codes 4 and up: (code / 2 - 1) extra bits
        bits.code(0, 7); // end of block (256)
        [stored_block(before), bits.out].concat()
    }

    /// The distance distance code `code` (4 to 29) and the value `extra` of
    /// its extra bits state, as deflate defines them.
    fn far_distance(code: u32, extra: u32) -> u32 {
        let bits = code / 2 - 1;
        ((2 + (code & 1)) << bits) + 1 + extra
    }

    #[test]
    fn copy_reaching_back_by_its_extra_bits_is_told_by_the_whole_distance() {
        // Distance codes from 4 on state a range of distances, the extra
        // bits after the code picking one: code 7 is 13 to 16 bytes back.
        // Whether a copy reaches before the container's start depends on
        // the whole distance, not on the code alone: after 14 stored bytes,
        // code 7 with extra bits 0 or 1 (13, 14 back) is a stream and with
        // 2 or 3 (15, 16 back) breaks it. Such a broken record covers no
        // other, so it is yielded and does not unpack, and the record after
        // it is found either way.
        assert_eq!(far_distance(4, 0), 5);
        assert_eq!(far_distance(6, 3), 12);
        assert_eq!(far_distance(7, 3), 16);
        assert_eq!(far_distance(29, 8191), 32_768);
        let before = b"DXBC fourteen.";
        assert_eq!(before.len(), 14);
        for code in 4..=8 {
            for extra in 0..1 << (code / 2 - 1) {
                let distance = far_distance(code, extra) as usize;
                let what = format!("code {code}, extra {extra}: {distance} back");
                let packed = stored_then_far_copy(before, code, extra);
                let valid = distance <= before.len();
                let want = valid.then(|| {
                    let mut out = before.to_vec();
                    for _ in 0..3 {
                        out.push(out[out.len() - distance]);
                    }
                    out
                });
                assert_eq!(
                    miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                    want,
                    "{what}"
                );
                assert_eq!(starts_stream(&packed, before.len() + 3), valid, "{what}");

                let (file, marker) = outer_then_after(&packed, before.len() + 3);
                let found: Vec<_> = records(&file).collect();
                let names: Vec<_> = found.iter().map(|r| r.name).collect();
                assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
                assert_eq!(
                    found[0].span(),
                    marker..marker + 12 + packed.len(),
                    "{what}"
                );
                assert_eq!(found[0].unpack(), want, "{what}");
                assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
                assert_eq!(found[1].span().end, file.len(), "{what}");
            }
        }
    }

    #[test]
    fn stream_at_deflates_best_ratio_is_a_record_just_above_the_bound() {
        // The packed-size lower bound holds for any deflate stream, not
        // just the ones miniz writes: a dynamic block whose length code 285
        // (258 bytes, no extra bits) and only distance code (1 back) take
        // one bit each makes 258 bytes from every 2 bits, deflate's best,
        // and is still at or above the bound, which it misses only by the
        // block's header. Such a record is found and unpacks, and so does
        // the record after it.
        let a = u32::from(b'A');
        // Literal/length codes: 285 in 1 bit, `A` and the end of the block
        // in 2; one distance code (distance 1) in 1 bit.
        let mut lengths = vec![0; 286 + 1];
        (
            lengths[285],
            lengths[a as usize],
            lengths[256],
            lengths[286],
        ) = (1, 2, 2, 1);
        // Code-length code: 18 in 1 bit, 2 in 2, 1 in 3, 0 and 17 in 4.
        let mut clen = [0; 19];
        (clen[18], clen[2], clen[1], clen[0], clen[17]) = (1, 2, 3, 4, 4);
        let lit = &lengths[..286];
        let lit_codes = canonical(lit);
        let copies = 4 * 1024;
        let mut bits = dynamic_header(286, 1, &clen, &length_symbols(&lengths));
        bits.code(lit_codes[a as usize], 2);
        for _ in 0..copies {
            bits.code(lit_codes[285], 1);
            bits.code(0, 1); // the one distance code
        }
        bits.code(lit_codes[256], 2);
        let packed = bits.out;
        let want = vec![b'A'; 1 + 258 * copies];
        assert_eq!(
            miniz_oxide::inflate::decompress_to_vec(&packed)
                .ok()
                .as_ref(),
            Some(&want)
        );
        assert!(starts_stream(&packed, want.len()));

        let bound = min_packed(want.len());
        assert_eq!(bound, copies / 4, "2 bits a copy");
        assert!(packed.len() >= bound);
        let header = packed.len() - bound;
        assert!(header < 32, "the bound misses by {header} bytes");

        let (file, marker) = outer_then_after(&packed, want.len());
        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_outer"), Some("PS_after")]);
        assert_eq!(found[0].span(), marker..marker + 12 + packed.len());
        assert_eq!(found[0].unpack().as_ref(), Some(&want));
        assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        assert_eq!(found[1].span().end, file.len());
    }

    #[test]
    fn lone_distance_code_is_a_stream_and_its_missing_half_is_noise() {
        // A dynamic block's codes must be complete, with two exceptions
        // deflate allows for the distance code: one code of 1 bit, the
        // other 1-bit code left unassigned, and no code at all (one length
        // of 0) for data of literals only. A copy coded with the one code
        // is a stream; a copy coded with the unassigned half, or any copy
        // when there is no distance code, breaks it. Such a broken record
        // covers no other, so it is yielded and does not unpack, and the
        // record after it is found either way.
        let a = u32::from(b'A');
        // Code-length code: 18 in 1 bit, 2 in 2, 1 in 3, 0 and 17 in 4.
        let mut clen = [0; 19];
        (clen[18], clen[2], clen[1], clen[0], clen[17]) = (1, 2, 3, 4, 4);
        // Literal/length codes: `A` in 1 bit, the end of the block and
        // length code 257 (3 bytes) in 2; one distance code (distance 1)
        // of `dist` bits.
        let block = |dist: u32, copy: Option<u32>| {
            let mut lengths = vec![0; 257 + 1 + 1];
            (lengths[a as usize], lengths[256], lengths[257]) = (1, 2, 2);
            lengths[258] = dist;
            let lit = &lengths[..258];
            let lit_codes = canonical(lit);
            let mut bits = dynamic_header(258, 1, &clen, &length_symbols(&lengths));
            bits.code(lit_codes[a as usize], 1);
            match copy {
                Some(code) => {
                    bits.code(lit_codes[257], 2);
                    bits.code(code, 1);
                }
                None => {
                    for _ in 0..3 {
                        bits.code(lit_codes[a as usize], 1);
                    }
                }
            }
            bits.code(lit_codes[256], 2);
            bits.out
        };
        let want = b"AAAA".to_vec();
        for (what, dist, copy, valid) in [
            ("copy by the lone 1-bit code", 1, Some(0), true),
            ("copy by its unassigned half", 1, Some(1), false),
            ("literals only, no distance code", 0, None, true),
            ("copy with no distance code", 0, Some(0), false),
        ] {
            let packed = block(dist, copy);
            assert_eq!(
                miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                valid.then(|| want.clone()),
                "{what}"
            );
            assert_eq!(starts_stream(&packed, want.len()), valid, "{what}");

            let (file, marker) = outer_then_after(&packed, want.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            assert_eq!(found[0].unpack(), valid.then(|| want.clone()), "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
    }

    #[test]
    fn code_length_lengths_left_unsent_are_zeros() {
        // HCLEN states how many of the 19 code-length code lengths a dynamic
        // block sends, in their fixed order; the rest are 0. A header that
        // stops once every nonzero one is sent is the same code in fewer
        // bits, and a stream. One that stops before a length its sequence
        // uses (here symbol 1, 18th in the order) leaves that symbol no
        // code, and the stream breaks where it is first sent. Such a broken
        // record covers no other, so it is yielded and does not unpack, and
        // the record after it is found either way.
        let a = u32::from(b'A');
        // Code-length code: 18 in 1 bit, 0 and 1 in 2 (a complete code).
        let mut clen = [0; 19];
        (clen[18], clen[0], clen[1]) = (1, 2, 2);
        assert_eq!(LENGTH_ORDER.iter().position(|&s| s == 1), Some(17));
        let mut lengths = vec![0; 258];
        (lengths[a as usize], lengths[256]) = (1, 1);
        let lit = &lengths[..257];
        let lit_codes = canonical(lit);
        let block = |hclen: usize| {
            let mut bits = dynamic_header_sending(hclen, 257, 1, &clen, &length_symbols(&lengths));
            for sym in [a, a, a, a, 256] {
                bits.code(lit_codes[sym as usize], lit[sym as usize]);
            }
            bits.out
        };
        let want = b"AAAA".to_vec();
        assert_eq!(
            block(19),
            dynamic_block(&clen, &length_symbols(&lengths), lit, &[a, a, a, a, 256])
        );
        for (hclen, valid) in [(19, true), (18, true), (17, false), (4, false)] {
            let what = format!("{hclen} code-length code lengths sent");
            let packed = block(hclen);
            assert_eq!(
                miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                valid.then(|| want.clone()),
                "{what}"
            );
            assert_eq!(starts_stream(&packed, want.len()), valid, "{what}");

            let (file, marker) = outer_then_after(&packed, want.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            assert_eq!(found[0].unpack(), valid.then(|| want.clone()), "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
    }

    #[test]
    fn incomplete_literal_length_code_is_noise_but_a_lone_one_bit_code_is_a_stream() {
        // A dynamic block's literal/length code must be complete: lengths
        // that leave some codes unassigned (`A` in 1 bit and the end of the
        // block in 2, or three codes of 2 bits) break the stream, though
        // every symbol the data sends has a code. The one exception is a
        // lone code of 1 bit, the other half unassigned: a block holding
        // only the end of the block that way is a stream of no bytes. Such a
        // broken record covers no other, so it is yielded and does not
        // unpack, and the record after it is found either way.
        let a = u32::from(b'A');
        let b = u32::from(b'B');
        // Code-length code: 18 in 1 bit, 0 in 2, 1 and 2 in 3.
        let mut clen = [0; 19];
        (clen[18], clen[0], clen[1], clen[2]) = (1, 2, 3, 3);
        let block = |codes: &[(u32, u32)], data: &[u32]| {
            // 257 literal/length codes, one distance code of length 0.
            let mut lengths = vec![0; 258];
            for &(sym, len) in codes {
                lengths[sym as usize] = len;
            }
            dynamic_block(&clen, &length_symbols(&lengths), &lengths[..257], data)
        };
        let aaaa = [a, a, a, a, 256];
        for (what, packed, want, valid) in [
            (
                "complete: A and end in 1 bit",
                block(&[(a, 1), (256, 1)], &aaaa),
                &b"AAAA"[..],
                true,
            ),
            (
                "A in 1 bit, end in 2",
                block(&[(a, 1), (256, 2)], &aaaa),
                b"AAAA",
                false,
            ),
            (
                "three 2-bit codes",
                block(&[(a, 2), (b, 2), (256, 2)], &aaaa),
                b"AAAA",
                false,
            ),
            (
                "lone 1-bit end of block",
                block(&[(256, 1)], &[256]),
                b"",
                true,
            ),
            (
                "lone 2-bit end of block",
                block(&[(256, 2)], &[256]),
                b"",
                false,
            ),
        ] {
            assert_eq!(
                miniz_oxide::inflate::decompress_to_vec(&packed)
                    .ok()
                    .as_deref(),
                valid.then_some(want),
                "{what}"
            );
            assert_eq!(starts_stream(&packed, want.len()), valid, "{what}");

            let (file, marker) = outer_then_after(&packed, want.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            assert_eq!(
                found[0].unpack().as_deref(),
                valid.then_some(want),
                "{what}"
            );
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
    }

    /// One final fixed-code block: the literal `A`, then a copy coded by
    /// length symbol 284 (227 bytes and up, 5 extra bits) with `extra` as
    /// its extra bits, from 1 byte back, then the end of the block.
    fn literal_then_copy_284(extra: u32) -> Vec<u8> {
        let mut bits = Bits::default();
        bits.put(1, 1); // BFINAL
        bits.put(1, 2); // fixed codes
        bits.code(0x30 + u32::from(b'A'), 8); // literals 0-143: 8 bits from 0x30
        bits.code(0xC0 + 284 - 280, 8); // 280-287: 8 bits from 0xC0
        bits.put(extra, 5);
        bits.code(0, 5); // distance code 0: 1 back
        bits.code(0, 7); // end of block (256)
        bits.out
    }

    #[test]
    fn length_284_with_every_extra_bit_set_is_a_258_byte_copy() {
        // Length symbol 284 states 227 to 258 bytes by its 5 extra bits;
        // deflate writers send 258 as symbol 285 instead, but 284 with all
        // extra bits set states 258 too, and inflaters take it as that
        // copy, unlike symbols 286 and 287, which mean nothing. Such a
        // record is a stream at every extra value, unpacks, and covers no
        // other; the record after it is found.
        for extra in [0, 1, 30, 31] {
            let what = format!("extra {extra}");
            let packed = literal_then_copy_284(extra);
            let want = vec![b'A'; 1 + 227 + extra as usize];
            assert_eq!(
                miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                Some(want.clone()),
                "{what}"
            );
            assert!(starts_stream(&packed, want.len()), "{what}");
            assert!(!starts_stream(&packed, want.len() + 1), "{what}");

            let (file, marker) = outer_then_after(&packed, want.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            assert_eq!(found[0].unpack(), Some(want), "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
        // 284 with every extra bit set and 285 make the same container.
        assert_eq!(
            miniz_oxide::inflate::decompress_to_vec(&literal_then_copy_284(31)).ok(),
            miniz_oxide::inflate::decompress_to_vec(&literal_then_coded_copy(285, 0)).ok()
        );
    }

    /// A non-final fixed-code block of the literal `A`, then a final stored
    /// block of `BC`. Its header ends mid-byte: `aligned` pads the rest of
    /// that byte with `pad` (its bits from low to high) before LEN, as
    /// deflate does; otherwise LEN follows the header's last bit at once.
    fn coded_then_stored(aligned: bool, pad: u32) -> Vec<u8> {
        let mut bits = Bits::default();
        bits.put(0, 1); // BFINAL 0
        bits.put(1, 2); // fixed codes
        bits.code(0x30 + u32::from(b'A'), 8); // literals 0-143: 8 bits from 0x30
        bits.code(0, 7); // end of block (256)
        bits.put(1, 1); // BFINAL
        bits.put(0, 2); // stored
        assert_eq!(bits.used, 21);
        if aligned {
            bits.put(pad, 3);
        }
        for v in [2, !2] {
            bits.put(u32::from(v as u16), 16);
        }
        for b in *b"BC" {
            bits.put(u32::from(b), 8);
        }
        bits.out
    }

    #[test]
    fn stored_block_after_a_coded_one_starts_at_the_next_byte_whatever_the_padding() {
        // A stored block's LEN starts at a byte: the bits left in the byte
        // its header ends in are skipped, whatever they hold. A stored block
        // after a fixed-code one is a stream with that padding zeros or
        // ones; the same LEN, NLEN and bytes written with no padding are
        // read shifted, NLEN no longer LEN's complement, and the stream
        // breaks. Such a broken record covers no other, so it is yielded
        // and does not unpack, and the record after it is found either way.
        let want = b"ABC".to_vec();
        for (aligned, pad) in [(true, 0b000), (true, 0b101), (true, 0b111), (false, 0)] {
            let what = format!("aligned {aligned}, padding {pad:03b}");
            let packed = coded_then_stored(aligned, pad);
            assert_eq!(packed.len(), 3 + 4 + 2, "{what}");
            assert_eq!(
                miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                aligned.then(|| want.clone()),
                "{what}"
            );
            assert_eq!(starts_stream(&packed, want.len()), aligned, "{what}");

            let (file, marker) = outer_then_after(&packed, want.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            assert_eq!(found[0].unpack(), aligned.then(|| want.clone()), "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
    }

    #[test]
    fn copy_from_the_farthest_distance_is_told_only_by_unpacking() {
        // Distance code 29 with every extra bit set reaches 32768 bytes
        // back, the farthest deflate states. After 32768 stored bytes that
        // copy is a stream; after one byte fewer it reaches before the
        // container's start and breaks it. Both breaks lie far past the
        // probe, which reads only stored bytes either way: both records are
        // taken as streams, so each is yielded with the span its sizes
        // state, the broken one does not unpack, and the record after it is
        // found.
        let far = far_distance(29, 8191) as usize;
        assert_eq!(far, 32_768);
        for stored in [far, far - 1] {
            let what = format!("{stored} stored bytes");
            let before: Vec<u8> = (0..stored).map(|i| (i % 251) as u8).collect();
            let packed = stored_then_far_copy(&before, 29, 8191);
            assert!(packed.len() > PROBE, "{what}");
            assert!(memchr::memmem::find(&packed, &MARKER).is_none(), "{what}");
            let valid = stored == far;
            let want = valid.then(|| {
                let mut out = before.clone();
                for _ in 0..3 {
                    out.push(out[out.len() - far]);
                }
                out
            });
            assert_eq!(
                miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                want,
                "{what}"
            );
            assert!(starts_stream(&packed, stored + 3), "{what}");

            let (file, marker) = outer_then_after(&packed, stored + 3);
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            assert_eq!(found[0].unpack(), want, "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
    }

    #[test]
    fn codes_of_fifteen_bits_are_a_stream_only_when_the_code_is_complete() {
        // Code lengths go up to 15 bits. A code as deep as deflate allows,
        // one symbol at each length from 1 to 14 and two at 15, is complete,
        // and data sent with every one of its codes (the 15-bit ones
        // crossing byte boundaries) is a stream. Moving one symbol a bit
        // deeper leaves a 15-bit code unassigned, and a bit shallower
        // oversubscribes the code: both break the stream, though every
        // symbol the data sends has a code. Such a broken record covers no
        // other, so it is yielded and does not unpack, and the record after
        // it is found either way.
        let a = u32::from(b'A');
        let n = u32::from(b'n');
        // Code-length code: symbols 0 to 12 in 4 bits, 13 to 18 in 5
        // (complete: 13/16 + 6/32).
        let mut clen = [0; 19];
        for (sym, len) in clen.iter_mut().enumerate() {
            *len = if sym <= 12 { 4 } else { 5 };
        }
        // `a` to `n` in 1 to 14 bits, `A` and the end of the block in 15.
        let block = |nudge: &[(u32, u32)]| {
            let mut lengths = vec![0; 258];
            for (i, sym) in (u32::from(b'a')..=n).enumerate() {
                lengths[sym as usize] = i as u32 + 1;
            }
            (lengths[a as usize], lengths[256]) = (15, 15);
            for &(sym, len) in nudge {
                lengths[sym as usize] = len;
            }
            let mut data: Vec<u32> = (u32::from(b'a')..=n).collect();
            data.extend([a, 256]);
            dynamic_block(&clen, &length_symbols(&lengths), &lengths[..257], &data)
        };
        let want = b"abcdefghijklmnA".to_vec();
        for (what, nudge, valid) in [
            ("complete, 15 bits deep", &[][..], true),
            ("`n` in 15 bits: one code unassigned", &[(n, 15)][..], false),
            ("`A` in 14 bits: oversubscribed", &[(a, 14)][..], false),
        ] {
            let packed = block(nudge);
            assert_eq!(
                miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                valid.then(|| want.clone()),
                "{what}"
            );
            assert_eq!(starts_stream(&packed, want.len()), valid, "{what}");

            let (file, marker) = outer_then_after(&packed, want.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            assert_eq!(found[0].unpack(), valid.then(|| want.clone()), "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
    }

    #[test]
    fn code_length_code_of_seven_bits_is_a_stream_only_when_complete() {
        // A dynamic block's code-length code has lengths of up to 7 bits
        // (3 bits each in the header). One as deep as that, a symbol at
        // each length from 1 to 6 and two at 7, is complete, and code
        // lengths sent with both 7-bit codes (crossing byte boundaries) are
        // a stream. Moving another symbol to 7 bits leaves a 7-bit code
        // unassigned, and moving a 7-bit one up to 6 oversubscribes the
        // code: both break the stream, though every symbol the sequence
        // sends has a code. Such a broken record covers no other, so it is
        // yielded and does not unpack, and the record after it is found
        // either way.
        let (a, b) = (u32::from(b'A'), u32::from(b'B'));
        // `A` in 1 bit, `B` and the end of the block in 2; one distance
        // code, unused, of length 0. The sequence sends 18, 1, 2 and 0.
        let mut lengths = vec![0; 258];
        (lengths[a as usize], lengths[b as usize], lengths[256]) = (1, 2, 2);
        let seq = length_symbols(&lengths);
        let used: std::collections::BTreeSet<_> = seq.iter().map(|&(sym, _)| sym).collect();
        assert_eq!(used.into_iter().collect::<Vec<_>>(), [0, 1, 2, 18]);
        // Code-length code: 18 in 1 bit, 0 in 2, 3 to 6 in 3 to 6, and the
        // two lengths the literal codes use, 1 and 2, in 7.
        let mut clen = [0; 19];
        (clen[18], clen[0], clen[3], clen[4], clen[5], clen[6]) = (1, 2, 3, 4, 5, 6);
        (clen[1], clen[2]) = (7, 7);
        let data = [a, a, a, b, 256];
        let want = b"AAAB".to_vec();
        for (what, nudge, valid) in [
            ("complete, 7 bits deep", &[][..], true),
            ("6 in 7 bits: one code unassigned", &[(6, 7)][..], false),
            ("2 in 6 bits: oversubscribed", &[(2, 6)][..], false),
        ] {
            let mut clen = clen;
            for &(sym, len) in nudge {
                clen[sym] = len;
            }
            let packed = dynamic_block(&clen, &seq, &lengths[..257], &data);
            assert_eq!(
                miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                valid.then(|| want.clone()),
                "{what}"
            );
            assert_eq!(starts_stream(&packed, want.len()), valid, "{what}");

            let (file, marker) = outer_then_after(&packed, want.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            assert_eq!(found[0].unpack(), valid.then(|| want.clone()), "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
    }

    #[test]
    fn bytes_after_the_final_block_are_never_read() {
        // A stream ends with the block whose BFINAL bit is set: the rest of
        // its last byte and every byte after it (up to the size the record
        // states) are padding and never read, whatever they hold, a reserved
        // block type, a broken stored header or a marker among them. With
        // BFINAL clear, the same bits are read as the next block's header
        // (here all ones: the reserved type 3) and the stream breaks; with
        // no bytes after it, the stream runs out instead, which is the start
        // of one that does not unpack. Either way the record covers no
        // other, so it is yielded with the span its size states, unpacking
        // only with BFINAL set, and the record after it is found.
        let block = literal_then_copy(1);
        assert_eq!(block.len(), 4, "30 bits: two left over in the last byte");
        let want = b"AAAA".to_vec();
        let mut marker = MARKER.to_vec();
        marker.extend_from_slice(&[16, 0, 0, 0, 4, 0, 0, 0]);
        for (what, tail) in [
            ("nothing", &[][..]),
            ("ones", &[0xFF; 8][..]),
            ("a reserved block", &[0x07, 0, 0, 0][..]),
            ("a broken stored header", &[1, 4, 0, 4, 0, b'x', b'y'][..]),
            ("a marker", &marker[..]),
        ] {
            for last in [true, false] {
                let what = format!("BFINAL {last}, then {what}");
                let mut packed = block.clone();
                packed[0] = (packed[0] & !1) | u8::from(last);
                *packed.last_mut().unwrap() |= 0b1100_0000;
                packed.extend_from_slice(tail);
                assert_eq!(
                    miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                    last.then(|| want.clone()),
                    "{what}"
                );
                let stream = last || tail.is_empty();
                assert_eq!(starts_stream(&packed, want.len()), stream, "{what}");

                let (file, at) = outer_then_after(&packed, want.len());
                let found: Vec<_> = records(&file).collect();
                let names: Vec<_> = found.iter().map(|r| r.name).collect();
                assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
                assert_eq!(found[0].span(), at..at + 12 + packed.len(), "{what}");
                assert_eq!(found[0].unpack(), last.then(|| want.clone()), "{what}");
                assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
                assert_eq!(found[1].span().end, file.len(), "{what}");
            }
        }
    }

    #[test]
    fn incomplete_code_length_or_distance_code_is_noise() {
        // Every code a dynamic block sends must be complete, the code-length
        // code and a distance code of two codes or more included: lengths
        // leaving some codes unassigned break the stream, though every
        // symbol the block sends has a code (the lone 1-bit distance code is
        // the one exception; see
        // `lone_distance_code_is_a_stream_and_its_missing_half_is_noise`).
        // Such a broken record covers no other, so it is yielded and does
        // not unpack, and the record after it is found either way.
        let a = u32::from(b'A');
        // Literal/length codes: `A` in 1 bit, the end of the block and
        // length code 257 (3 bytes) in 2; two distance codes (distances 1
        // and 2) of `dist` bits each.
        let block = |clen: &[u32; 19], dist: u32| {
            let mut lengths = vec![0; 258 + 2];
            (lengths[a as usize], lengths[256], lengths[257]) = (1, 2, 2);
            (lengths[258], lengths[259]) = (dist, dist);
            let lit = &lengths[..258];
            let lit_codes = canonical(lit);
            let dist_codes = canonical(&lengths[258..]);
            let mut bits = dynamic_header(258, 2, clen, &length_symbols(&lengths));
            bits.code(lit_codes[a as usize], 1);
            bits.code(lit_codes[257], 2);
            bits.code(dist_codes[0], dist);
            bits.code(lit_codes[256], 2);
            bits.out
        };
        // Code-length code: 18 in 1 bit, 0 in 2, 1 and 2 in 3 (complete).
        let mut clen = [0; 19];
        (clen[18], clen[0], clen[1], clen[2]) = (1, 2, 3, 3);
        // The same with 2 in 4 bits: one 4-bit code left unassigned.
        let mut short = clen;
        short[2] = 4;
        let want = b"AAAA".to_vec();
        for (what, packed, valid) in [
            ("complete codes", block(&clen, 1), true),
            ("incomplete code-length code", block(&short, 1), false),
            ("two distance codes of 2 bits", block(&clen, 2), false),
        ] {
            assert_eq!(
                miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                valid.then(|| want.clone()),
                "{what}"
            );
            assert_eq!(starts_stream(&packed, want.len()), valid, "{what}");

            let (file, marker) = outer_then_after(&packed, want.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            assert_eq!(found[0].unpack(), valid.then(|| want.clone()), "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
    }

    /// The fixed literal/length code of `sym` and its length in bits.
    fn fixed_code(sym: u32) -> (u32, u32) {
        match sym {
            0..=143 => (0x30 + sym, 8),
            144..=255 => (0x190 + sym - 144, 9),
            256..=279 => (sym - 256, 7),
            280..=287 => (0xC0 + sym - 280, 8),
            _ => unreachable!("no such symbol"),
        }
    }

    #[test]
    fn symbols_at_every_edge_of_the_fixed_code_ranges_are_a_stream() {
        // The fixed literal/length code is four ranges of code lengths:
        // 0-143 in 8 bits, 144-255 in 9, 256-279 in 7 and 280-287 in 8.
        // A block sending the symbols on both sides of each edge (literals
        // 0, 143, 144 and 255; copies coded 279, 280 and 285; the end of
        // the block, 256) is a stream: it unpacks, it is a stream of
        // exactly its size and of no size one off, and the record covers
        // no other, so the record after it is found.
        let mut bits = Bits::default();
        bits.put(1, 1); // BFINAL
        bits.put(1, 2); // fixed codes
        let mut want = Vec::new();
        for lit in [0u8, 143, 144, 255] {
            let (code, len) = fixed_code(lit.into());
            bits.code(code, len);
            want.push(lit);
        }
        // (length symbol, extra bits' value, their count, length, distance)
        for (sym, extra, count, length, distance) in [
            (279, 0, 4, 99, 1),
            (280, 15, 4, 130, 4),
            (285, 0, 0, 258, 2),
        ] {
            let (code, len) = fixed_code(sym);
            bits.code(code, len);
            bits.put(extra, count);
            bits.code(distance - 1, 5); // distance codes 0-3: 1-4, no extra bits
            for _ in 0..length {
                want.push(want[want.len() - distance as usize]);
            }
        }
        let (code, len) = fixed_code(256);
        bits.code(code, len);
        let packed = bits.out;
        assert_eq!(want.len(), 4 + 99 + 130 + 258);

        assert_eq!(
            miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
            Some(want.clone())
        );
        assert!(starts_stream(&packed, want.len()));
        assert!(!starts_stream(&packed, want.len() - 1));
        assert!(!starts_stream(&packed, want.len() + 1));

        let (file, marker) = outer_then_after(&packed, want.len());
        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_outer"), Some("PS_after")]);
        assert_eq!(found[0].span(), marker..marker + 12 + packed.len());
        assert_eq!(found[0].unpack(), Some(want));
        assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        assert_eq!(found[1].span().end, file.len());
    }

    #[test]
    fn zero_run_crossing_into_the_distance_codes_is_a_stream() {
        // Runs of zeros (17: 3 to 10, 18: 11 to 138) may carry on from the
        // last literal/length codes into the distance codes, as a run of
        // the previous length (16) may. A header stating more literal/length
        // codes than the block uses, its unused ones and all distance codes
        // but the last sent as one run of zeros, is a stream. The same run
        // two longer (one longer only ends the header early) runs past the
        // codes the header states and breaks it. Either way the record
        // covers no other, so it is yielded, unpacking only if it is a
        // stream, and the record after it is found.
        let a = u32::from(b'A');
        // Code-length code: 0, 1, 17 and 18 in 2 bits each (complete).
        let mut clen = [0; 19];
        (clen[0], clen[1], clen[17], clen[18]) = (2, 2, 2, 2);
        let data = [a, a, a, a, 256];
        let want = b"AAAA".to_vec();
        // `A` and the end of the block in 1 bit each; the last distance
        // code, unused, in 1 bit (a lone distance code).
        let block = |hlit: u32, hdist: u32, run: u32| {
            let mut lit = vec![0; hlit as usize];
            (lit[a as usize], lit[256]) = (1, 1);
            let mut seq = length_symbols(&lit[..257]);
            assert!(seq.iter().all(|&(sym, _)| matches!(sym, 1 | 18)));
            seq.push(match run {
                3..=10 => (17, run - 3),
                _ => (18, run - 11),
            });
            seq.push((1, 0));
            dynamic_block_of(hlit, hdist, &clen, &seq, &lit, &data)
        };
        // (literal/length codes, distance codes): the zeros between the
        // end of the block and the last distance code fill a 17 run (6),
        // the longest a 17 still breaks with (8), and an 18 run (58, the
        // most the header can state).
        for (hlit, hdist) in [(260, 4), (261, 5), (286, 30)] {
            let zeros = (hlit - 257) + (hdist - 1);
            assert!(hlit > 257 && hdist > 1, "the run crosses");
            for (run, valid) in [(zeros, true), (zeros + 2, false)] {
                let what = format!("{hlit}/{hdist} codes, run of {run} zeros");
                let packed = block(hlit, hdist, run);
                assert_eq!(
                    miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                    valid.then(|| want.clone()),
                    "{what}"
                );
                assert_eq!(starts_stream(&packed, want.len()), valid, "{what}");

                let (file, marker) = outer_then_after(&packed, want.len());
                let found: Vec<_> = records(&file).collect();
                let names: Vec<_> = found.iter().map(|r| r.name).collect();
                assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
                assert_eq!(
                    found[0].span(),
                    marker..marker + 12 + packed.len(),
                    "{what}"
                );
                assert_eq!(found[0].unpack(), valid.then(|| want.clone()), "{what}");
                assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
                assert_eq!(found[1].span().end, file.len(), "{what}");
            }
        }
    }

    /// One final fixed-code block: the literals `ABC`, then a copy coded by
    /// length symbol `length` (257 to 264: 3 to 10 bytes, or 285: 258)
    /// from `distance` (1 to 4) bytes back, then the end of the block.
    fn abc_then_copy(length: u32, distance: u32) -> Vec<u8> {
        let mut bits = Bits::default();
        bits.put(1, 1); // BFINAL
        bits.put(1, 2); // fixed codes
        for b in *b"ABC" {
            bits.code(0x30 + u32::from(b), 8); // literals 0-143: 8 bits from 0x30
        }
        match length {
            257..=264 => bits.code(length - 256, 7), // 256-279: 7 bits from 0
            285 => bits.code(0xC0 + 285 - 280, 8),   // 280-287: 8 bits from 0xC0
            _ => unreachable!("a length symbol without extra bits"),
        }
        bits.code(distance - 1, 5); // distance codes 0-3: 1-4, no extra bits
        bits.code(0, 7); // end of block (256)
        bits.out
    }

    #[test]
    fn copy_longer_than_its_distance_repeats_what_it_makes() {
        // A copy may be longer than its distance: it then reads bytes it has
        // just made, repeating the last `distance` bytes. After `ABC`, every
        // length from 3 bytes back reaches the first byte and no further,
        // and is a stream; from 4 bytes back it reaches before the start
        // and breaks, however long. A copy 1 or 2 back repeats a shorter
        // tail. Either way the record covers no other, so it is yielded,
        // unpacking only if it is a stream, and the record after it is
        // found.
        let repeat = |length: usize, distance: usize| -> Vec<u8> {
            let mut out = b"ABC".to_vec();
            for _ in 0..length {
                out.push(out[out.len() - distance]);
            }
            out
        };
        assert_eq!(repeat(4, 3), b"ABCABCA");
        assert_eq!(repeat(4, 2), b"ABCBCBC");
        for (symbol, length) in [(257, 3), (258, 4), (264, 10), (285, 258)] {
            for distance in 1..=4 {
                let what = format!("{length} bytes from {distance} back");
                let packed = abc_then_copy(symbol, distance);
                let valid = distance <= 3;
                let want = repeat(length, distance.min(3) as usize);
                assert_eq!(want.len(), 3 + length, "{what}");
                assert_eq!(
                    miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                    valid.then(|| want.clone()),
                    "{what}"
                );
                assert_eq!(starts_stream(&packed, want.len()), valid, "{what}");
                if valid {
                    assert!(!starts_stream(&packed, want.len() - 1), "{what}");
                    assert!(!starts_stream(&packed, want.len() + 1), "{what}");
                }

                let (file, marker) = outer_then_after(&packed, want.len());
                let found: Vec<_> = records(&file).collect();
                let names: Vec<_> = found.iter().map(|r| r.name).collect();
                assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
                assert_eq!(
                    found[0].span(),
                    marker..marker + 12 + packed.len(),
                    "{what}"
                );
                assert_eq!(found[0].unpack(), valid.then(|| want.clone()), "{what}");
                assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
                assert_eq!(found[1].span().end, file.len(), "{what}");
            }
        }
    }

    #[test]
    fn stated_size_of_zero_is_a_stream_only_if_nothing_is_made() {
        // A record may state an empty container. Its packed bytes are then a
        // stream only if they make nothing: an empty fixed-code block, an
        // empty stored one, or empty blocks one after another. Bytes that
        // make even one byte are noise, as no byte may be made at all; the
        // same bytes stating one byte are a stream. Alone, such a record
        // covers no other, so it is yielded either way, unpacking only if it
        // is a stream, and the record after it is found.
        let empty: [(&str, &[u8]); 3] = [
            ("empty fixed block", &[0x03, 0x00]),
            ("empty stored block", &[0x01, 0, 0, 0xFF, 0xFF]),
            (
                "empty stored then empty fixed",
                &[0x00, 0, 0, 0xFF, 0xFF, 0x03, 0x00],
            ),
        ];
        for (what, packed) in empty {
            assert_eq!(
                miniz_oxide::inflate::decompress_to_vec(packed).ok(),
                Some(Vec::new()),
                "{what}"
            );
            assert!(starts_stream(packed, 0), "{what}");
            assert!(!starts_stream(packed, 1), "{what}");

            let (file, marker) = outer_then_after(packed, 0);
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            assert_eq!(found[0].unpack(), Some(Vec::new()), "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        }
        for level in [0, 1, 6, 10] {
            let what = format!("one byte at level {level}");
            let packed = miniz_oxide::deflate::compress_to_vec(b"A", level);
            assert!(!starts_stream(&packed, 0), "{what}");
            assert!(starts_stream(&packed, 1), "{what}");

            let (file, marker) = outer_then_after(&packed, 0);
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            assert_eq!(found[0].unpack(), None, "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        }
    }

    #[test]
    fn marker_stating_zero_over_a_stream_making_bytes_does_not_hide_the_record() {
        // Bytes that read as a marker stating an empty container, whose
        // packed bytes make one byte and then run on over a real record:
        // they are no stream of that size, so the record they cover is
        // found. Stating one byte instead, the same bytes are a stream (the
        // bytes after its final block are never read) and the record they
        // cover is theirs.
        let after = record_bytes("PS_after", 0x32, b"after");
        for level in [0, 1, 6] {
            let stream = miniz_oxide::deflate::compress_to_vec(b"A", level);
            let mut packed = stream.clone();
            packed.extend(&after);
            packed.extend_from_slice(&[0x5A; 8]);
            for stated in [0, 1] {
                let what = format!("level {level}, stating {stated}");
                let n = (12 + packed.len()) as u32;
                let mut file = MAGIC.to_vec();
                file.extend_from_slice(b"\x09PS_outer\0\x32");
                let marker = file.len() + 4;
                for v in [n, 0x000D_DCE7, n, stated] {
                    file.extend_from_slice(&v.to_le_bytes());
                }
                let inner = file.len() + stream.len() + 19;
                file.extend_from_slice(&packed);
                let found: Vec<_> = records(&file).collect();
                let names: Vec<_> = found.iter().map(|r| r.name).collect();
                if stated == 0 {
                    assert_eq!(names, [Some("PS_after")], "{what}");
                    assert_eq!(found[0].offset, inner, "{what}");
                    assert_eq!(found[0].unpack().as_deref(), Some(&b"after"[..]));
                } else {
                    assert_eq!(names, [Some("PS_outer")], "{what}");
                    assert_eq!(found[0].span(), marker..file.len(), "{what}");
                    assert_eq!(found[0].unpack().as_deref(), Some(&b"A"[..]), "{what}");
                }
            }
        }
    }

    #[test]
    fn remembered_stream_markers_answer_as_a_fresh_scan_would() {
        // `Streams` remembers how far it has looked and the first marker it
        // found starting a stream; a question from anywhere, in any order,
        // must get the answer a fresh scan of `lo..hi` gives: whether a
        // marker wholly inside it starts a record whose packed bytes are (or,
        // cut off, start) a deflate stream (`hi` within the file, as
        // [`covers_a_record`] asks). The file mixes real records,
        // markers that are noise (sizes that disagree, a reserved block) and
        // a record cut off by the end of the file.
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_a", 0x32, b"first"));
        for v in [20_u32, 0x000D_DCE7, 21, 8] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&[0x5A; 7]);
        file.extend(record_bytes("PS_b", 0x41, &[7; 300]));
        file.extend_from_slice(b"\x05PS_n\0\x32");
        for v in [16_u32, 0x000D_DCE7, 16, 4] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&[0x07, 0, 0, 0]); // reserved block type
        file.extend(record_bytes("PS_c", 0x3c, b"third"));
        let cut = record_bytes("PS_cut", 0x32, &[3; 2000]);
        file.extend_from_slice(&cut[..cut.len() - 3]);

        let fresh = |lo: usize, hi: usize| {
            memchr::memmem::find_iter(&file, &MARKER)
                .filter(|&m| m >= lo && m + MARKER.len() <= hi)
                .any(|m| match record(&file[m - 6..], 6) {
                    Found::Record(r) => r.is_stream(),
                    Found::Truncated => true,
                    Found::Nothing => false,
                })
        };
        let markers: Vec<_> = memchr::memmem::find_iter(&file, &MARKER).collect();
        assert_eq!(markers.len(), 6, "planted markers");
        let streams = markers.iter().filter(|&&m| fresh(m, m + 4)).count();
        assert_eq!(streams, 4, "three whole records and the cut-off one");

        // Every edge: a marker's first byte and one past, its last byte.
        let mut edges = vec![6, file.len()];
        for &m in &markers {
            edges.extend([m - 1, m, m + 1, m + 3, m + 4, m + 5]);
        }
        // Its caller never asks past the end of the file.
        edges.retain(|&e| e <= file.len());
        let mut x = 0x2545_F491_u32;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x as usize
        };
        for round in 0..400 {
            let mut cache = Streams::default();
            for q in 0..12 {
                let lo = edges[next() % edges.len()].max(6);
                let hi = match next() % 3 {
                    0 => (lo + next() % 64).min(file.len()),
                    1 => edges[next() % edges.len()].max(lo),
                    _ => file.len(),
                };
                assert_eq!(
                    cache.any_in(&file, lo, hi),
                    fresh(lo, hi),
                    "round {round}, question {q}: {lo}..{hi}"
                );
            }
        }

        // The patterns `records` asks in: moving forward over the same
        // window, a hit just past `hi` remembered for the next question,
        // and a question from before what was remembered.
        let first = markers[0];
        let mut cache = Streams::default();
        assert!(!cache.any_in(&file, 6, first + 3), "marker one byte short");
        assert!(cache.any_in(&file, 6, first + 4), "now wholly inside");
        assert!(cache.any_in(&file, first, first + 4), "same hit, later lo");
        assert!(
            !cache.any_in(&file, first + 1, markers[1] + 4),
            "noise only"
        );
        assert!(cache.any_in(&file, first + 1, markers[2] + 4));
        assert!(cache.any_in(&file, 6, first + 4), "back before: rescanned");
    }

    #[test]
    fn stream_markers_asked_past_the_end_of_the_file_are_not_skipped() {
        // A question whose `hi` runs past the end of the file must not count
        // markers it never looked at as noise: one that stops short of a
        // stream's marker, then one reaching past the end, must still find
        // it, and so must questions from before what was remembered.
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_a", 0x32, b"first"));
        let marker = memchr::memmem::find(&file, &MARKER).unwrap();
        let mut cache = Streams::default();
        assert!(!cache.any_in(&file, 6, marker + 3), "one byte short");
        assert!(cache.any_in(&file, 6, file.len() + 7), "past the end");

        let mut cache = Streams::default();
        // Nothing wholly in `marker + 1..`, however far past the end.
        assert!(!cache.any_in(&file, marker + 1, file.len() + 1000));
        assert!(!cache.any_in(&file, marker + 1, usize::MAX));
        // Back to the marker (before what was remembered): rescanned.
        assert!(cache.any_in(&file, marker, usize::MAX));
        assert!(cache.any_in(&file, 0, marker + MARKER.len()));

        // A cut-off record (a deflate stream's start) asked past the end.
        let cut = record_bytes("PS_cut", 0x32, &[3; 2000]);
        let mut cut_file = MAGIC.to_vec();
        cut_file.extend_from_slice(&cut[..cut.len() - 3]);
        let mut cache = Streams::default();
        assert!(!cache.any_in(&cut_file, 6, 7));
        assert!(cache.any_in(&cut_file, 6, cut_file.len() + 64));
    }

    #[test]
    fn a_marker_too_near_the_start_for_a_header_is_not_a_stream() {
        // A marker whose header would start before the file has none: it is
        // no record, and asking from offset 0 must not underflow.
        for at in 0..6 {
            let mut file = vec![0_u8; at];
            file.extend_from_slice(&MARKER);
            for v in [16_u32, 16, 4] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&[0x03, 0, 0, 0]);
            let mut cache = Streams::default();
            assert!(!cache.any_in(&file, 0, file.len()), "marker at {at}");
            assert!(!cache.any_in(&file, 0, usize::MAX), "marker at {at}");
        }
    }

    #[test]
    fn repeat_after_a_zero_run_repeats_zero() {
        // A dynamic block's 16 repeats the previous code length, and after
        // a run of zeros (17 or 18, or a 0 sent as is) that length is 0:
        // zeros sent that way are a stream whichever way the run before
        // them was sent. After a length of 1 the same 16 repeats 1, giving
        // more 1-bit codes than fit, and breaks the stream. Either way the record covers no other, so it is yielded,
        // unpacking only if it is a stream, and the record after it is found.
        let a = u32::from(b'A');
        // 257 literal/length codes and one distance code: `A` and the end
        // of the block in 1 bit, every other length (the distance code's
        // too) 0.
        let mut lit = vec![0; 258];
        (lit[a as usize], lit[256]) = (1, 1);
        let data = [a, a, a, a, 256];
        let want = b"AAAA".to_vec();
        // Code-length code: 0, 16 and 18 in 2 bits, 1 and 17 in 3.
        let mut clen = [0; 19];
        (clen[0], clen[16], clen[18], clen[1], clen[17]) = (2, 2, 2, 3, 3);
        // `n` (3 and up) more of the previous length, as 16s of 3 to 6.
        let again = |mut n: usize| {
            let mut out = Vec::new();
            while n > 6 {
                let take = if n - 6 >= 3 { 6 } else { n - 3 };
                out.push((16, take as u32 - 3));
                n -= take;
            }
            out.push((16, n as u32 - 3));
            out
        };
        // The 65 zeros before `A`, started by a 0, a 17 or an 18, the rest
        // repeated; then `A`; the 190 zeros after it as an 18 and repeats;
        // the end of the block and the distance code's 0.
        let starts: [(&str, (u32, u32), usize); 3] = [
            ("a 0", (0, 0), 1),
            ("a 17", (17, 7), 10),
            ("an 18", (18, 0), 11),
        ];
        let mut cases = Vec::new();
        for (what, start, run) in starts {
            let mut seq = vec![start];
            seq.extend(again(65 - run));
            seq.extend([(1, 0), (18, 127)]);
            seq.extend(again(190 - 138));
            seq.extend([(1, 0), (0, 0)]);
            cases.push((format!("{what} first"), seq, true));
        }
        // A 16 right after `A`'s 1 repeats 1, not the zeros before it.
        let mut after_one = vec![(18, 54), (1, 0)];
        after_one.extend(again(3));
        after_one.push((18, 127));
        after_one.extend(again(190 - 138 - 3));
        after_one.extend([(1, 0), (0, 0)]);
        cases.push(("a repeat right after A".to_string(), after_one, false));

        for (what, seq, valid) in &cases {
            let packed = dynamic_block(&clen, seq, &lit[..257], &data);
            assert_eq!(
                miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
                valid.then(|| want.clone()),
                "{what}"
            );
            assert_eq!(starts_stream(&packed, want.len()), *valid, "{what}");

            let (file, marker) = outer_then_after(&packed, want.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            assert_eq!(found[0].unpack(), valid.then(|| want.clone()), "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
    }

    #[test]
    fn length_and_distance_codes_where_extra_bits_begin_are_a_stream() {
        // Length symbols 257-264 are 3 to 10 bytes with no extra bits and
        // 265 is 11 or 12 by one extra bit; distance codes 0-3 are 1 to 4
        // back with none and 4 is 5 or 6 by one. A fixed-code block sending
        // each of 264, 265 (both extra values), 3 and 4 (both) is a stream
        // of exactly its size, and its record covers no other, so the record
        // after it is found. The same block with 265's extra bit left out
        // reads every later code shifted and breaks.
        let block = |with_extra: bool| {
            let mut bits = Bits::default();
            bits.put(1, 1); // BFINAL
            bits.put(1, 2); // fixed codes
            let mut want = Vec::new();
            for lit in *b"ABCDEF" {
                let (code, len) = fixed_code(lit.into());
                bits.code(code, len);
                want.push(lit);
            }
            // (length symbol, its extra bit, length, distance code, its
            // extra bit, distance)
            for (sym, lx, length, dcode, dx, distance) in [
                (264, None, 10, 3, None, 4),
                (265, Some(0), 11, 4, Some(0), 5),
                (265, Some(1), 12, 4, Some(1), 6),
                (257, None, 3, 3, None, 4),
            ] {
                let (code, len) = fixed_code(sym);
                bits.code(code, len);
                if let Some(x) = lx.filter(|_| with_extra) {
                    bits.put(x, 1);
                }
                bits.code(dcode, 5);
                if let Some(x) = dx {
                    bits.put(x, 1);
                }
                for _ in 0..length {
                    want.push(want[want.len() - distance]);
                }
            }
            let (code, len) = fixed_code(256);
            bits.code(code, len);
            (bits.out, want)
        };

        let (packed, want) = block(true);
        assert_eq!(want.len(), 6 + 10 + 11 + 12 + 3);
        assert_eq!(
            miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
            Some(want.clone())
        );
        assert!(starts_stream(&packed, want.len()));
        assert!(!starts_stream(&packed, want.len() - 1));
        assert!(!starts_stream(&packed, want.len() + 1));

        let (shifted, _) = block(false);
        assert_ne!(
            miniz_oxide::inflate::decompress_to_vec(&shifted).ok(),
            Some(want.clone())
        );
        assert!(!starts_stream(&shifted, want.len()));

        for (what, packed, valid) in [("whole", &packed, true), ("shifted", &shifted, false)] {
            let (file, marker) = outer_then_after(packed, want.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
            assert_eq!(
                found[0].span(),
                marker..marker + 12 + packed.len(),
                "{what}"
            );
            assert_eq!(found[0].unpack(), valid.then(|| want.clone()), "{what}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "{what}");
        }
    }
}
