//! XMA2 packets are transparent to the decoder: frames laid across the
//! packets of one stream, with other streams' packets between them, then
//! gathered back with `read_frame` and `next`, decode to exactly the
//! samples and results of decoding the same frames directly, from the first
//! frame or from any frame a seek lands on.
//!
//! Synthetic input only: frames of sparse xorshift bits behind their length
//! field, each picked so that it decodes to sound on its own, and packet
//! headers computed here from where each frame lands.

use victory_formats::xma::{self, Decoder, Error, PACKET_BYTES};

const PACKET_BITS: usize = PACKET_BYTES * 8;
const HEADER_BITS: usize = 32;

fn put_bit(data: &mut [u8], pos: usize, bit: bool) {
    let mask = 0x80 >> (pos % 8);
    if bit {
        data[pos / 8] |= mask;
    } else {
        data[pos / 8] &= !mask;
    }
}

/// A frame's bits: its 15-bit length, then xorshift noise with about one
/// bit in four set (sparse payloads decode far more often than dense ones).
fn frame_bits(len: usize, seed: u32) -> Vec<bool> {
    let mut state = seed | 1;
    (0..len)
        .map(|i| {
            if i < 15 {
                (len >> (14 - i)) & 1 == 1
            } else {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state.is_multiple_of(4)
            }
        })
        .collect()
}

fn pack(bits: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; bits.len().div_ceil(8)];
    for (i, &b) in bits.iter().enumerate() {
        put_bit(&mut out, i, b);
    }
    out
}

/// Lays `frames` end to end across the packets `own` (ascending) of a
/// buffer of `total` packets whose other packets are all ones, and writes
/// each own packet's header. Returns the buffer and each frame's start.
fn packetize(total: usize, own: &[usize], frames: &[Vec<bool>]) -> (Vec<u8>, Vec<usize>) {
    let mut data = vec![0xffu8; total * PACKET_BYTES];
    for &p in own {
        data[p * PACKET_BYTES..][..PACKET_BYTES].fill(0);
    }
    let mut counts = vec![(0u32, None::<usize>); own.len()];
    let mut pos = own[0] * PACKET_BITS + HEADER_BITS;
    let align = |pos: &mut usize| {
        if pos.is_multiple_of(PACKET_BITS) {
            let i = own
                .iter()
                .position(|&p| p + 1 == *pos / PACKET_BITS)
                .unwrap();
            *pos = own[i + 1] * PACKET_BITS + HEADER_BITS;
        }
    };
    let mut starts = vec![];
    for bits in frames {
        align(&mut pos);
        let i = own.iter().position(|&p| p == pos / PACKET_BITS).unwrap();
        counts[i].0 += 1;
        counts[i].1.get_or_insert(pos);
        starts.push(pos);
        for &bit in bits {
            align(&mut pos);
            put_bit(&mut data, pos, bit);
            pos += 1;
        }
    }
    for (i, &p) in own.iter().enumerate() {
        let skip = own.get(i + 1).map_or(0, |&n| n - p - 1) as u32;
        let (count, first) = counts[i];
        let first = first.map_or(0x7fff, |s| s - p * PACKET_BITS - HEADER_BITS) as u32;
        let h = count << 26 | first << 11 | skip;
        data[p * PACKET_BYTES..][..4].copy_from_slice(&h.to_be_bytes());
    }
    (data, starts)
}

type Decoded = (Vec<i16>, Vec<Result<(), Error>>);

fn decode_direct(channels: usize, frames: &[Vec<bool>]) -> Decoded {
    let mut d = Decoder::new(channels, 48_000);
    let mut out = vec![];
    let results = frames
        .iter()
        .map(|bits| d.decode(&pack(bits), bits.len(), &mut out))
        .collect();
    d.flush(&mut out);
    (out, results)
}

fn decode_packets(channels: usize, data: &[u8], start: usize) -> Decoded {
    let mut d = Decoder::new(channels, 48_000);
    let mut out = vec![];
    let mut results = vec![];
    // A last frame that ends past the packet it began in points `next` at
    // the padding after it, where no frame reads.
    let mut at = Some(start);
    while let Some(f) = at.and_then(|pos| xma::read_frame(data, pos)) {
        results.push(d.decode(&f.data, f.bits, &mut out));
        at = f.next;
    }
    d.flush(&mut out);
    (out, results)
}

/// Whether a fresh decoder of `channels` decodes `bits` without an error
/// and with some non-zero samples.
fn sounds(channels: usize, bits: &[bool]) -> bool {
    let mut d = Decoder::new(channels, 48_000);
    let mut out = vec![];
    let data = pack(bits);
    d.decode(&data, bits.len(), &mut out).is_ok()
        && d.decode(&data, bits.len(), &mut out).is_ok()
        && out.iter().any(|&s| s != 0)
}

/// Short frames, frames that straddle a packet header, one longer than a
/// whole packet; for each length the first seed that sounds in mono and
/// stereo.
fn frames() -> Vec<Vec<bool>> {
    [
        700,
        3100,
        PACKET_BITS - 200,
        1500,
        PACKET_BITS + 2500,
        900,
        4000,
        9000,
        1333,
        2048,
    ]
    .iter()
    .enumerate()
    .map(|(i, &len)| {
        (0..1000)
            .map(|seed| frame_bits(len, 7 + 31 * i as u32 + 1000 * seed))
            .find(|bits| sounds(1, bits) && sounds(2, bits))
            .unwrap_or_else(|| panic!("no sparse frame of {len} bits decodes"))
    })
    .collect()
}

#[test]
fn packetized_frames_decode_like_the_frames_themselves() {
    let frames = frames();
    let layouts: [(usize, &[usize]); 3] = [
        (6, &[0, 1, 2, 3, 4, 5]),
        (11, &[0, 2, 3, 6, 7, 10]),
        (9, &[1, 4, 5, 8]),
    ];
    for (total, own) in layouts {
        let (data, starts) = packetize(total, own, &frames);
        for channels in [1, 2] {
            let direct = decode_direct(channels, &frames);
            // The comparison is only worth something if the stream decodes.
            assert!(direct.1.iter().all(Result::is_ok), "{channels}ch");
            assert!(direct.0.iter().any(|&s| s != 0), "{channels}ch");
            let packets = decode_packets(channels, &data, starts[0]);
            assert_eq!(packets.1.len(), frames.len(), "{own:?} {channels}ch");
            assert_eq!(packets, direct, "{own:?} {channels}ch");
        }
    }
}

#[test]
fn seeking_to_any_frame_decodes_like_the_rest_of_the_frames() {
    let frames = frames();
    let (data, starts) = packetize(11, &[0, 2, 3, 6, 7, 10], &frames);
    for channels in [1, 2] {
        for (i, &start) in starts.iter().enumerate() {
            assert_eq!(
                decode_packets(channels, &data, start),
                decode_direct(channels, &frames[i..]),
                "seek to frame {i}, {channels}ch"
            );
        }
    }
}
