//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, and turn-range reduction for `v_sin_*`/`v_cos_*`.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
fn build_valu_cs(body: &[u32]) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        options: super::TranslateOptions::default(),
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// One GPU test at a time. With these tests on the hardware adapter and WARP
/// from several test threads at once, now and then none of their dispatches
/// ever completes and the test binary hangs; and two GPU tests at once on
/// WARP can end the test process with an access violation. Other GPU test
/// modules take this lock too.
pub(super) static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(shader: &HostShader, cases: &[[u32; 4]], check: impl FnMut(&str, usize, [u32; 4], u32)) {
    run_cases_on(shader, cases, &[("hardware", false), ("warp", true)], check);
}

/// `run_cases` on the given backends only (WARP cannot pin a wave size, which
/// the lane-aware programs ask for).
fn run_cases_on(
    shader: &HostShader,
    cases: &[[u32; 4]],
    backends: &[(&str, bool)],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for &(label, warp) in backends {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

/// MTBUF `op vdata, off, s[srsrc:srsrc+3], 0 format:format` (at byte 0).
const fn mtbuf(op: u8, vdata: u8, srsrc: u8, format: u8) -> [u32; 2] {
    [
        0x3A << 26 | (format as u32 & 0x7F) << 19 | (op as u32 & 7) << 16,
        0x80 << 24
            | (op as u32 >> 3 & 1) << 21
            | (srsrc as u32 >> 2 & 0x1F) << 16
            | (vdata as u32) << 8,
    ]
}

/// Stores `v1` to the buffer, then reads it back into `v0` with the tbuffer
/// load `op` in buffer format `format`.
fn tbuffer_load_of_v1(op: u8, format: u8) -> HostShader {
    let mut body = buffer_store_dword(1, 8, 0).to_vec();
    body.extend(mtbuf(op, 0, 8, format));
    build_valu_cs(&body)
}

#[test]
fn tbuffer_d16_loads_pack_two_channels_per_dword() {
    // tbuffer_load_format_d16_x and tbuffer_load_format_d16_xy.
    let (x, xy) = (8, 9);
    // (opcode, format, dword stored, dword loaded)
    let cases: &[(u8, u8, u32, u32)] = &[
        // 16_16_FLOAT: the f16 pair (1.0, 2.0) as it was stored.
        (xy, 29, 0x4000_3C00, 0x4000_3C00),
        // 16_16_UINT: the 16-bit integers as they were stored.
        (xy, 27, 0xBEEF_1234, 0xBEEF_1234),
        // 16_16_UNORM: 0.0 and 1.0, as f16.
        (xy, 23, 0xFFFF_0000, 0x3C00_0000),
        // 16_FLOAT, one channel: the high half is 0.
        (x, 13, 0x4000_3C00, 0x0000_3C00),
    ];
    for &(op, format, stored, want) in cases {
        let shader = tbuffer_load_of_v1(op, format);
        run_cases(&shader, &[[stored, 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(
                got, want,
                "{backend}: d16 opcode {op} format {format} of {stored:#010x} = {got:#010x}, want {want:#010x}",
            );
        });
    }
}

#[test]
fn normalized_buffer_formats_load_their_ends_exactly() {
    // tbuffer_load_format_x.
    let x = 0;
    // (format, dword stored, value loaded)
    let cases: &[(u8, u32, f32)] = &[
        (1, 0xFF, 1.0),   // 8_UNORM
        (7, 0xFFFF, 1.0), // 16_UNORM
        (2, 0x7F, 1.0),   // 8_SNORM: the top, the one above the lowest, the lowest
        (2, 0x81, -1.0),
        (2, 0x80, -1.0),
        (8, 0x7FFF, 1.0), // 16_SNORM
        (8, 0x8001, -1.0),
        (8, 0x8000, -1.0),
    ];
    for &(format, stored, want) in cases {
        let shader = tbuffer_load_of_v1(x, format);
        run_cases(&shader, &[[stored, 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(
                got,
                want.to_bits(),
                "{backend}: format {format} of {stored:#x} = {got:#010x}, want {want}",
            );
        });
    }
}

#[test]
fn packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves.
    // The same GFX10 source field (248) has a different bit pattern for a
    // packed floating-point and a packed integer instruction.
    let cases = [
        // v_pk_add_f16 v0, 0x3118, v1
        ([0xcc0f4000, 0x1a0202f8], 0x0000_3118),
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        // v_pk_add_u16 v0, 0x3e22f983, v1
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ];
    for (words, expected) in cases {
        let shader = build_valu_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: {words:08x?}");
        });
    }
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

/// RDNA2 orders -0 below +0 in its maximums and minimums; DXIL's FMax and
/// FMin compare and select, which can give either zero.
#[test]
fn max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    // Each case has a -0 and a +0 among its first two operands.
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f32", 0x110, pz),
        ("v_min_f32", 0x10F, nz),
        ("v_max3_f32", 0x154, pz),
        ("v_min3_f32", 0x151, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    }
}

/// The ISA defines `v_med3_f32` over `v_min_f32`/`v_max_f32`, which order -0
/// below +0 (the same order as [`max_and_min_order_negative_zero_below_positive_zero`]),
/// so of three zeros the median is the sign two of them share. DXIL's plain
/// FMin/FMax chain gives either zero.
#[test]
fn med3_orders_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    let f = f32::to_bits;
    let cases: &[([u32; 4], u32)] = &[
        ([nz, pz, nz, 0], nz),      // -0, +0, -0
        ([pz, nz, pz, 0], pz),      // +0, -0, +0
        ([nz, pz, pz, 0], pz),      // -0, +0, +0
        ([pz, nz, nz, 0], nz),      // +0, -0, -0
        ([nz, nz, nz, 0], nz),      // -0, -0, -0
        ([pz, pz, pz, 0], pz),      // +0, +0, +0
        ([nz, pz, f(1.0), 0], pz),  // -0 < +0 < 1
        ([nz, f(-1.0), pz, 0], nz), // -1 < -0 < +0
        ([pz, f(1.0), nz, 0], pz),  // -0 < +0 < 1
    ];
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_med3_f32({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });

    // The same in halves: -0 is 0x8000, +0 is 0x0000 (VOP3 opcode 0x357).
    let (hpz, hnz) = (0x0000_u32, 0x8000_u32);
    let (h1, hn1) = (0x3C00_u32, 0xBC00_u32); // 1.0, -1.0
    let cases16: &[([u32; 4], u32)] = &[
        ([hnz, hpz, hnz, 0], hnz),
        ([hpz, hnz, hpz, 0], hpz),
        ([hnz, hpz, hpz, 0], hpz),
        ([hpz, hnz, hnz, 0], hnz),
        ([hnz, hpz, h1, 0], hpz),
        ([hnz, hn1, hpz, 0], hnz),
        ([hpz, h1, hnz, 0], hpz),
    ];
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let inputs: Vec<[u32; 4]> = cases16.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases16[i].1 as u16;
        assert_eq!(
            got as u16, want,
            "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got:#06x}, want {want:#06x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_min3_max3_med3_integer_forms_match_rdna2() {
    // v_{min3,max3,med3}_{i32,u32} v0, v1, v2, v3 (VOP3 0x152..0x159).
    type Pick = fn([u32; 3]) -> u32;
    let ops: [(&str, u16, Pick); 6] = [
        ("v_min3_i32", 0x152, |t| {
            t.into_iter().min_by_key(|&x| x as i32).unwrap()
        }),
        ("v_min3_u32", 0x153, |t| t.into_iter().min().unwrap()),
        ("v_max3_i32", 0x155, |t| {
            t.into_iter().max_by_key(|&x| x as i32).unwrap()
        }),
        ("v_max3_u32", 0x156, |t| t.into_iter().max().unwrap()),
        ("v_med3_i32", 0x158, |mut t| {
            t.sort_by_key(|&x| x as i32);
            t[1]
        }),
        ("v_med3_u32", 0x159, |mut t| {
            t.sort_unstable();
            t[1]
        }),
    ];
    // Every ordering of three values, with the sign bit set on some so that
    // the signed and unsigned answers differ.
    let values = [0u32, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF];
    let mut cases = Vec::new();
    for a in values {
        for b in values {
            for c in values {
                cases.push([a, b, c, 0]);
            }
        }
    }
    for (name, op, pick) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let want = pick([inp[0], inp[1], inp[2]]);
            assert_eq!(
                got, want,
                "{backend}: {name}({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1], inp[2],
            );
        });
    }
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

/// `v_cmp_lg_*` is "less or greater": ordered and not equal, so false when
/// either operand is a NaN (RDNA2 ISA; the same on gfx1036 hardware).
#[test]
fn v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    let nan = f32::NAN.to_bits();
    let (inf, ninf) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([inf, ninf, 0, 0], 1),
        ([inf, inf, 0, 0], 0),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([nan, nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0), // a signalling NaN
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x4000, 0x3C00, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x3C00, 0x7E00, 0, 0], 0),
        ([0x7E00, 0x7E00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),           // a signalling NaN
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0), // only the low halves are read
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

/// `v_div_fixup_f16` as the RDNA2 ISA lists it; `None` for a NaN.
fn div_fixup_f16_reference(q: u16, den: u16, num: u16) -> Option<u16> {
    let nan = |x: u16| is_f16_nan(x);
    let abs = |x: u16| x & 0x7FFF;
    let sign = (den ^ num) & 0x8000;
    if nan(num) || nan(den) {
        return None;
    }
    if (abs(den) == 0 && abs(num) == 0) || (abs(den) == 0x7C00 && abs(num) == 0x7C00) {
        return None;
    }
    if abs(den) == 0 || abs(num) == 0x7C00 {
        return Some(sign | 0x7C00);
    }
    if abs(den) == 0x7C00 || abs(num) == 0 {
        return Some(sign);
    }
    // An all-ones exponent overflows, so a NaN quotient gives infinity
    // (gfx1036 hardware).
    if abs(q) >= 0x7C00 {
        return Some(sign | 0x7C00);
    }
    Some(sign | abs(q))
}

#[test]
fn v_div_fixup_f16_follows_the_isa_special_cases() {
    // v_mov_b32 v0, 0; v_div_fixup_f16 v0, v1, v2, v3 (VOP3 opcode 0x35f):
    // v1 the quotient, v2 the denominator, v3 the numerator.
    let mut body = vec![vop1(0x001, 0, 128)];
    body.extend_from_slice(&vop3(0x35f, 0, vgpr(1), vgpr(2), vgpr(3)));
    let shader = build_valu_cs(&body);
    // 0, -0, 1, -1.5, 2^-24, max finite, inf, -inf, NaN, -NaN.
    let values = [
        0x0000u16, 0x8000, 0x3C00, 0xBE00, 0x0001, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE01,
    ];
    let mut cases = Vec::new();
    for &q in &[0x4248u16, 0xC248, 0x7E00, 0x7C01] {
        for &den in &values {
            for &num in &values {
                cases.push([u32::from(q), u32::from(den), u32::from(num), 0]);
            }
        }
    }
    run_cases(&shader, &cases, |backend, _, inp, got| {
        let (q, den, num) = (inp[0] as u16, inp[1] as u16, inp[2] as u16);
        let got_h = got as u16;
        assert_eq!(got >> 16, 0, "{backend}: the high half changed");
        match div_fixup_f16_reference(q, den, num) {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_div_fixup_f16({q:#06x}, {den:#06x}, {num:#06x}) = {got_h:#06x}, \
                 want {want:#06x}",
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_div_fixup_f16({q:#06x}, {den:#06x}, {num:#06x}) = {got_h:#06x}, \
                 want NaN",
            ),
        }
    });
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// RDNA2's flt16_to_snorm16 / flt16_to_unorm16: clamp, scale by 32767 or
/// 65535, round the exact product to nearest even; NaN gives 0.
fn cvt_norm16_reference(bits: u16, signed: bool) -> u16 {
    let x = f16_to_f32(bits);
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    // f64 holds the product of an f16 and a 16-bit scale exactly.
    let p = f64::from(x.clamp(lo, 1.0)) * scale;
    p.round_ties_even() as i32 as u16
}

#[test]
fn v_cvt_norm_i16_and_u16_f16_round_the_exact_product() {
    // Every 251st f16 pattern, plus the values where an f32 product would
    // round before the rounding to an integer, ties, and the edges.
    let mut bits: Vec<u16> = (0..=u16::MAX).step_by(251).collect();
    bits.extend([
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3801, 0xB801, 0x37FF,
        0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01,
        0x3A9B, 0xBA9B, 0x2E66, 0x1400,
    ]);
    for (op, signed, what) in [
        (0x063u16, true, "v_cvt_norm_i16_f16"),
        (0x064, false, "v_cvt_norm_u16_f16"),
    ] {
        // v_cvt_norm_*16_f16 v0, v1 (VOP1); v0's high half starts at 0.
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = bits.iter().map(|&b| [u32::from(b), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = cvt_norm16_reference(bits[i], signed);
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                bits[i],
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_cvt_u16_and_i16_f16_saturate() {
    // v_mov_b32 v0, 0; v_cvt_u16_f16 / v_cvt_i16_f16 v0, v1 (VOP1 0x052 / 0x053).
    let cases: &[(u16, u16, u16)] = &[
        // (f16 input, u16 result, i16 result)
        (0x0000, 0, 0),          // 0
        (0x8000, 0, 0),          // -0
        (0x3E00, 1, 1),          // 1.5 -> 1 (toward zero)
        (0xBE00, 0, 0xFFFF),     // -1.5 -> 0 / -1
        (0x57D0, 125, 125),      // 125.0
        (0x7800, 32768, 32767),  // 32768 saturates i16
        (0x77FF, 32752, 32752),  // 32752, the largest f16 below 32768
        (0xF800, 0, 0x8000),     // -32768 fits i16
        (0xF801, 0, 0x8000),     // -32800 saturates i16
        (0x7BFF, 65504, 32767),  // max finite f16
        (0xFBFF, 0, 0x8000),     // -65504
        (0x7C00, 0xFFFF, 32767), // +inf
        (0xFC00, 0, 0x8000),     // -inf
        (0x7E00, 0, 0),          // NaN
        (0xFE00, 0, 0),          // -NaN
        (0x0001, 0, 0),          // smallest subnormal
    ];
    for (op, what, pick) in [
        (0x052u16, "v_cvt_u16_f16", 0usize),
        (0x053, "v_cvt_i16_f16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_cvt_f16_u16_and_i16_widen_the_integer_first() {
    // v_mov_b32 v0, 0; v_cvt_f16_u16 / v_cvt_f16_i16 v0, v1 (VOP1 0x050 / 0x051).
    // Only integers f16 holds exactly: how the others round is the narrowing's
    // business, not this conversion's. `None`: the result depends on rounding.
    let cases: &[(u16, Option<u16>, Option<u16>)] = &[
        // (u16 / i16 input bits, f16 of the unsigned value, f16 of the signed value)
        (0x0000, Some(0x0000), Some(0x0000)), // 0
        (0x0001, Some(0x3C00), Some(0x3C00)), // 1.0
        (0x07FF, Some(0x67FF), Some(0x67FF)), // 2047, the last integer below 2048 that f16 holds
        (0x8000, Some(0x7800), Some(0xF800)), // 32768 / -32768
        (0xFC00, Some(0x7BE0), Some(0xE400)), // 64512 / -1024
        (0xFFFE, None, Some(0xC000)),         // 65534 does not fit f16 / -2
        (0xFFFF, None, Some(0xBC00)),         // 65535 does not fit f16 / -1
    ];
    for (op, what, pick) in [
        (0x050u16, "v_cvt_f16_u16", 0usize),
        (0x051, "v_cvt_f16_i16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            let Some(want) = want else { return };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_i32 width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    run_cases(
        &v_signed,
        &[[0x8000_0000, 4, 32, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: v_bfe_i32 width 32 should mask to 0");
        },
    );

    // s_bfe_i32 width is 7-bit (src1[22:16]), so width=32 keeps all 32 bits.
    let mut s_cases = signed_cases.to_vec();
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    let s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });

    // s_bfe_u32 s6, s0, s1 (SOP2 opcode 0x27): the same 7-bit width as the signed form,
    // so width=32 keeps all 32 bits and nothing is sign-extended.
    let s_unsigned = build_valu_cs(&[sop2(0x27, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);
    let mut su_cases = unsigned_cases.to_vec();
    su_cases.push(([0x8000_0000, 4, 32, 0], 0x0800_0000));
    let su_in: Vec<[u32; 4]> = su_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_unsigned, &su_in, |backend, i, inp, got| {
        let want = su_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_u32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn v_cmp_class_f32_matches_rdna2() {
    // v_cmp_class_f32_e64 s6, v1, v2 (VOP3 opcode 0x088); v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp = build_valu_cs(&[
        vop3(0x088, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x088, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary32 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7F80_0001, 0), // +sNaN (quiet bit 0x0040_0000 clear)
        (0xFFBF_FFFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7FC0_0000, 1), // +qNaN
        (0xFFC0_0001, 1), // -qNaN
        (0xFF80_0000, 2), // -inf
        (0xBF80_0000, 3), // -normal (-1.0)
        (0x8000_0001, 4), // -denormal
        (0x8000_0000, 5), // -0.0
        (0x0000_0000, 6), // +0.0
        (0x0000_0001, 7), // +denormal
        (0x3F80_0000, 8), // +normal (+1.0)
        (0x7F80_0000, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f32({:#010x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn v_cmp_class_f16_matches_rdna2() {
    // v_cmp_class_f16_e64 s6, v1, v2 (VOP3 opcode 0x08f); v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp_lo = build_valu_cs(&[
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);
    // Same with op_sel:[1,0] (high half of v1) + neg on src0 (bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x08f, 6, vgpr(1), vgpr(2), 0);
    w0 |= 1 << 11; // opsel[0] = 1 (high half)
    w1 |= 1 << 29; // neg[0] = 1
    let cmp_hi_neg = build_valu_cs(&[
        w0,
        w1,
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary16 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7C01, 0), // sNaN (+sNaN, quiet bit 0x0200 clear)
        (0xFDFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7E00, 1), // +qNaN
        (0xFE01, 1), // -qNaN
        (0xFC00, 2), // -inf
        (0xBC00, 3), // -normal (-1.0)
        (0x8001, 4), // -denormal
        (0x8000, 5), // -0.0
        (0x0000, 6), // +0.0
        (0x0001, 7), // +denormal
        (0x3C00, 8), // +normal (+1.0)
        (0x7C00, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_lo, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16({:#06x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // High half with neg: high half 0x3C00 (+1.0) negated -> -normal (class 3);
    // high half 0x7C01 (+sNaN) negated -> still sNaN (class 0, not quieted).
    let hi_cases: &[([u32; 4], u32)] = &[
        ([0x3C00_0000, 1 << 3, 0, 0], 1),
        ([0x3C00_0000, 1 << 8, 0, 0], 0),
        ([0x7C01_FFFF, 1 << 0, 0, 0], 1),
        ([0x7C01_FFFF, 1 << 1, 0, 0], 0),
    ];
    let hi_in: Vec<[u32; 4]> = hi_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_hi_neg, &hi_in, |backend, i, inp, got| {
        let want = hi_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16(-hi({:#010x}), {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

/// A select whose mask is a plain SGPR (here a root constant), so the
/// structure pass does not follow it: this lane takes its own bit of the
/// number. Inline 0 and -1 masks pick src0 and src1 in every lane.
#[test]
fn v_cndmask_on_an_untracked_mask_tests_this_lanes_bit() {
    // v_cndmask_b32_e64 v0, v1, v2, <mask> (VOP3 0x101); s3 holds the mask.
    let from_sgpr = build_valu_cs(&vop3(0x101, 0, vgpr(1), vgpr(2), sgpr(3)));
    let hw = &[("hardware", false)];
    run_cases_on(
        &from_sgpr,
        &[[10, 20, 0, 0], [10, 20, 0, 1], [10, 20, 0, 2], [10, 20, 0, !0]],
        hw,
        |label, i, inp, got| {
            let want = if inp[3] & 1 != 0 { inp[1] } else { inp[0] };
            assert_eq!(got, want, "{label} case {i}");
        },
    );
    for (mask, pick) in [(128u16, 0usize), (193, 1)] {
        let k = build_valu_cs(&vop3(0x101, 0, vgpr(1), vgpr(2), mask));
        run_cases_on(&k, &[[10, 20, 0, 0]], hw, |label, _, inp, got| {
            assert_eq!(got, inp[pick], "{label} inline {mask}");
        });
    }
}

#[test]
fn scalar_buffer_loads_read_a_uav_through_its_table() {
    // The UAV is the one `build_valu_cs` stores to: write root constant 0
    // to its dword 0, read it back with a scalar buffer load, and store
    // that.
    let body = [
        0xE070_0000,
        0x8002_0100, // buffer_store_dword v1, off, s[8:11], 0
        0xF420_0304,
        0xFA00_0000, // s_buffer_load_dword s12, s[8:11], null
        0x7E00_020C, // v_mov_b32 v0, s12
    ];
    let shader = build_valu_cs(&body);
    let cases = [[0x1234_5678, 0, 0, 0], [0xDEAD_BEEF, 0, 0, 0]];
    run_cases(&shader, &cases, |backend, _, inp, got| {
        assert_eq!(got, inp[0], "{backend}: read back {got:#010x}");
    });
}
/// SCC after the SOP2 `op s12, s0, s1`, read back into `v0` through
/// `s_cselect_b32 s13, 1, 0`.
fn scc_after(op: u8) -> HostShader {
    build_valu_cs(&[
        sop2(op, 12, 0, 1),
        sop2(0x0A, 13, 129, 128), // s_cselect_b32 s13, 1, 0
        vop1(0x001, 0, sgpr(13)), // v_mov_b32 v0, s13
    ])
}

/// Checks the SCC that `op` leaves for each `(s0, s1, scc)`.
fn check_scc(name: &str, op: u8, cases: &[(u32, u32, u32)]) {
    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(a, b, _)| [a, b, 0, 0]).collect();
    run_cases(&scc_after(op), &inputs, |backend, i, inp, got| {
        let want = cases[i].2;
        assert_eq!(
            got, want,
            "{backend}: SCC after {name}({:#010x}, {:#010x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn s_min_and_s_max_set_scc_when_src0_is_chosen() {
    let m1 = (-1i32) as u32;
    check_scc("s_min_i32", 0x06, &[(m1, 2, 1), (2, m1, 0), (4, 4, 0)]);
    check_scc("s_min_u32", 0x07, &[(3, 5, 1), (5, 3, 0), (4, 4, 0)]);
    check_scc("s_max_i32", 0x08, &[(2, m1, 1), (m1, 2, 0), (4, 4, 0)]);
    check_scc("s_max_u32", 0x09, &[(5, 3, 1), (3, 5, 0), (4, 4, 0)]);
}

#[test]
fn s_lshl_add_sets_scc_on_unsigned_overflow() {
    // (s0 << n) + s1 past 32 bits, through the shift or through the add.
    check_scc(
        "s_lshl1_add_u32",
        0x2E,
        &[
            (1, 2, 0),
            (1, 0xFFFF_FFFD, 0),
            (1, 0xFFFF_FFFE, 1),
            (0x8000_0000, 0, 1),
        ],
    );
    check_scc(
        "s_lshl4_add_u32",
        0x31,
        &[
            (0x0FFF_FFFF, 0xF, 0),
            (0x0FFF_FFFF, 0x10, 1),
            (0x1000_0000, 0, 1),
        ],
    );
}
/// SOPK `op sdst, simm16`.
const fn sopk(op: u8, sdst: u8, simm16: i16) -> u32 {
    0xB000_0000 | (op as u32 & 0x1F) << 23 | (sdst as u32 & 0x7F) << 16 | (simm16 as u16) as u32
}

#[test]
fn signed_scalar_add_sub_scc_matches_rdna2() {
    // s_add_i32 s6, s0, s1 (SOP2 0x02); s_cselect_b32 s7, 1, 0 (SOP2 0x0A); v_mov_b32 v0, s7
    let add_scc = build_valu_cs(&[
        sop2(0x02, 6, 0, 1),
        sop2(0x0A, 7, 0x81, 0x80),
        vop1(0x001, 0, sgpr(7)),
    ]);
    // s_sub_i32 s6, s0, s1 (SOP2 0x03); s_cselect_b32 s7, 1, 0; v_mov_b32 v0, s7
    let sub_scc = build_valu_cs(&[
        sop2(0x03, 6, 0, 1),
        sop2(0x0A, 7, 0x81, 0x80),
        vop1(0x001, 0, sgpr(7)),
    ]);
    // s_mov_b32 s6, s0; s_addk_i32 s6, +1 (SOPK 0x0F); s_cselect_b32 s7, 1, 0; v_mov_b32 v0, s7
    let addk_pos1 = build_valu_cs(&[
        s_mov_b32(6, 0),
        sopk(0x0F, 6, 1),
        sop2(0x0A, 7, 0x81, 0x80),
        vop1(0x001, 0, sgpr(7)),
    ]);
    // s_mov_b32 s6, s0; s_addk_i32 s6, -32768 (SOPK 0x0F); s_cselect_b32 s7, 1, 0; v_mov_b32 v0, s7
    let addk_neg_min = build_valu_cs(&[
        s_mov_b32(6, 0),
        sopk(0x0F, 6, i16::MIN),
        sop2(0x0A, 7, 0x81, 0x80),
        vop1(0x001, 0, sgpr(7)),
    ]);

    let add_cases: &[(i32, i32)] = &[
        (0, 0),
        (10, 20),
        (-1, 1), // unsigned carry, no signed overflow
        (i32::MAX, 0),
        (i32::MAX, 1),        // positive overflow
        (i32::MAX, i32::MAX), // positive overflow
        (i32::MIN, 0),
        (i32::MIN, -1),       // negative overflow
        (i32::MIN, i32::MIN), // negative overflow (result 0)
        (i32::MIN, i32::MAX), // -1, no overflow
    ];
    let add_in: Vec<[u32; 4]> = add_cases
        .iter()
        .map(|&(a, b)| [a as u32, b as u32, 0, 0])
        .collect();
    run_cases(&add_scc, &add_in, |backend, i, _, got| {
        let (a, b) = add_cases[i];
        let want = u32::from(a.overflowing_add(b).1);
        assert_eq!(
            got, want,
            "{backend}: s_add_i32({a}, {b}) SCC = {got}, want {want}",
        );
    });

    let sub_cases: &[(i32, i32)] = &[
        (0, 0),
        (0, 1),        // unsigned borrow, no signed overflow
        (0, i32::MIN), // 0 - (-2^31) overflows i32
        (i32::MIN, 0),
        (i32::MIN, 1),        // negative overflow
        (i32::MIN, i32::MIN), // 0, no overflow
        (i32::MAX, -1),       // positive overflow
        (i32::MAX, i32::MIN), // positive overflow
        (-5, -3),             // -2, no overflow
        (-1, i32::MAX),       // -2^31 (i32::MIN), no overflow
    ];
    let sub_in: Vec<[u32; 4]> = sub_cases
        .iter()
        .map(|&(a, b)| [a as u32, b as u32, 0, 0])
        .collect();
    run_cases(&sub_scc, &sub_in, |backend, i, _, got| {
        let (a, b) = sub_cases[i];
        let want = u32::from(a.overflowing_sub(b).1);
        assert_eq!(
            got, want,
            "{backend}: s_sub_i32({a}, {b}) SCC = {got}, want {want}",
        );
    });

    let addk_pos_cases: &[i32] = &[0, -1, i32::MAX - 1, i32::MAX, i32::MIN];
    let addk_pos_in: Vec<[u32; 4]> = addk_pos_cases
        .iter()
        .map(|&a| [a as u32, 0, 0, 0])
        .collect();
    run_cases(&addk_pos1, &addk_pos_in, |backend, i, _, got| {
        let a = addk_pos_cases[i];
        let want = u32::from(a.overflowing_add(1).1);
        assert_eq!(
            got, want,
            "{backend}: s_addk_i32({a}, +1) SCC = {got}, want {want}",
        );
    });

    let addk_neg_cases: &[i32] = &[0, i32::MAX, i32::MIN + 32768, i32::MIN + 32767, i32::MIN];
    let addk_neg_in: Vec<[u32; 4]> = addk_neg_cases
        .iter()
        .map(|&a| [a as u32, 0, 0, 0])
        .collect();
    run_cases(&addk_neg_min, &addk_neg_in, |backend, i, _, got| {
        let a = addk_neg_cases[i];
        let want = u32::from(a.overflowing_add(i32::from(i16::MIN)).1);
        assert_eq!(
            got, want,
            "{backend}: s_addk_i32({a}, -32768) SCC = {got}, want {want}",
        );
    });
}
/// VOP3P `op vdst, src0, src1, src2` with standard packed lane selects
/// (`opsel = 0b000`, `opsel_hi = 0b111`) and optional `clamp`.
const fn vop3p(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16, clamp: bool) -> [u32; 2] {
    [
        0x33 << 26
            | (op as u32 & 0x7F) << 16
            | (clamp as u32) << 15
            | (1 << 14) // opsel_hi[2] = 1
            | vdst as u32,
        (0b11 << 27) // opsel_hi[1:0] = 0b11
            | (src0 as u32 & 0x1FF)
            | (src1 as u32 & 0x1FF) << 9
            | (src2 as u32 & 0x1FF) << 18,
    ]
}

#[test]
fn v_pk_int16_clamp_saturates_as_rdna2() {
    let pack_u = |lo: u16, hi: u16| u32::from(lo) | (u32::from(hi) << 16);
    let pack_i = |lo: i16, hi: i16| pack_u(lo as u16, hi as u16);

    // VOP3P opcodes: 0x00=mad_i16, 0x02=add_i16, 0x03=sub_i16, 0x09=mad_u16, 0x0a=add_u16, 0x0b=sub_u16.
    let add_u16 = build_valu_cs(&vop3p(0x0a, 0, vgpr(1), vgpr(2), 0, true));
    let sub_u16 = build_valu_cs(&vop3p(0x0b, 0, vgpr(1), vgpr(2), 0, true));
    let mad_u16 = build_valu_cs(&vop3p(0x09, 0, vgpr(1), vgpr(2), vgpr(3), true));
    let add_i16 = build_valu_cs(&vop3p(0x02, 0, vgpr(1), vgpr(2), 0, true));
    let sub_i16 = build_valu_cs(&vop3p(0x03, 0, vgpr(1), vgpr(2), 0, true));
    let mad_i16 = build_valu_cs(&vop3p(0x00, 0, vgpr(1), vgpr(2), vgpr(3), true));

    // add_u16 clamp: lo overflows (60000+10000 -> 65535), hi fits (100+200 -> 300)
    run_cases(
        &add_u16,
        &[
            [pack_u(60000, 100), pack_u(10000, 200), 0, 0],
            [pack_u(u16::MAX, 0), pack_u(u16::MAX, 0), 0, 0],
        ],
        |backend, i, _, got| {
            let want = [pack_u(u16::MAX, 300), pack_u(u16::MAX, 0)][i];
            assert_eq!(got, want, "{backend}: v_pk_add_u16 clamp case {i}");
        },
    );

    // sub_u16 clamp: lo underflows (5-10 -> 0), hi fits (500-200 -> 300)
    run_cases(
        &sub_u16,
        &[
            [pack_u(5, 500), pack_u(10, 200), 0, 0],
            [pack_u(0, u16::MAX), pack_u(u16::MAX, 0), 0, 0],
        ],
        |backend, i, _, got| {
            let want = [pack_u(0, 300), pack_u(0, u16::MAX)][i];
            assert_eq!(got, want, "{backend}: v_pk_sub_u16 clamp case {i}");
        },
    );

    // mad_u16 clamp: lo max*max+max -> 65535 (tests no u32 wrap!), hi 20*30+40 -> 640
    run_cases(
        &mad_u16,
        &[[
            pack_u(u16::MAX, 20),
            pack_u(u16::MAX, 30),
            pack_u(u16::MAX, 40),
            0,
        ]],
        |backend, _, _, got| {
            let want = pack_u(u16::MAX, 640);
            assert_eq!(got, want, "{backend}: v_pk_mad_u16 clamp");
        },
    );

    // add_i16 clamp: lo +overflow (30000+10000 -> 32767), hi -overflow (-30000-10000 -> -32768)
    run_cases(
        &add_i16,
        &[
            [pack_i(30000, -30000), pack_i(10000, -10000), 0, 0],
            [pack_i(100, -50), pack_i(-25, 20), 0, 0],
        ],
        |backend, i, _, got| {
            let want = [pack_i(i16::MAX, i16::MIN), pack_i(75, -30)][i];
            assert_eq!(got, want, "{backend}: v_pk_add_i16 clamp case {i}");
        },
    );

    // sub_i16 clamp: lo -overflow (-30000-10000 -> -32768), hi +overflow (30000-(-10000) -> 32767)
    run_cases(
        &sub_i16,
        &[
            [pack_i(-30000, 30000), pack_i(10000, -10000), 0, 0],
            [pack_i(0, -100), pack_i(i16::MIN, -200), 0, 0],
        ],
        |backend, i, _, got| {
            let want = [pack_i(i16::MIN, i16::MAX), pack_i(i16::MAX, 100)][i];
            assert_eq!(got, want, "{backend}: v_pk_sub_i16 clamp case {i}");
        },
    );

    // mad_i16 clamp: lo (-32768)*(-32768)+32767 -> 32767, hi (-32768)*(32767)-32768 -> -32768
    run_cases(
        &mad_i16,
        &[
            [
                pack_i(i16::MIN, i16::MIN),
                pack_i(i16::MIN, i16::MAX),
                pack_i(i16::MAX, i16::MIN),
                0,
            ],
            [pack_i(-10, 5), pack_i(20, -4), pack_i(50, -3), 0],
        ],
        |backend, i, _, got| {
            let want = [pack_i(i16::MAX, i16::MIN), pack_i(-150, -23)][i];
            assert_eq!(got, want, "{backend}: v_pk_mad_i16 clamp case {i}");
        },
    );
}
#[test]
fn v_frexp_f32_and_f16_match_rdna2() {
    // VOP1 opcodes: 0x3f=v_frexp_exp_i32_f32, 0x40=v_frexp_mant_f32,
    //               0x59=v_frexp_mant_f16,    0x5a=v_frexp_exp_i16_f16.
    let exp_f32 = build_valu_cs(&[vop1(0x3f, 0, vgpr(1))]);
    let mant_f32 = build_valu_cs(&[vop1(0x40, 0, vgpr(1))]);
    // For f16 tests, initialize v0 = 0xDEAD_0000 so we also verify high-half preservation.
    let mant_f16 = build_valu_cs(&[
        vop1(0x001, 0, vgpr(2)), // v_mov_b32 v0, v2 (sentinel upper half)
        vop1(0x59, 0, vgpr(1)),
    ]);
    let exp_f16 = build_valu_cs(&[
        vop1(0x001, 0, vgpr(2)), // v_mov_b32 v0, v2 (sentinel upper half)
        vop1(0x5a, 0, vgpr(1)),
    ]);

    // (input f32 bits, expected mantissa f32 bits, expected i32 exponent)
    let f32_cases: &[(u32, u32, i32)] = &[
        (0.0f32.to_bits(), 0.0f32.to_bits(), 0),
        ((-0.0f32).to_bits(), (-0.0f32).to_bits(), 0),
        (0.5f32.to_bits(), 0.5f32.to_bits(), 0),
        (1.0f32.to_bits(), 0.5f32.to_bits(), 1),
        ((-6.0f32).to_bits(), (-0.75f32).to_bits(), 3),
        (0.125f32.to_bits(), 0.5f32.to_bits(), -2),
        (f32::INFINITY.to_bits(), f32::INFINITY.to_bits(), 0),
        (f32::NEG_INFINITY.to_bits(), f32::NEG_INFINITY.to_bits(), 0),
        // Subnormals: 2^-127 = 0.5 * 2^-126; 3 * 2^-149 = 0.75 * 2^-147; -2^-149 = -0.5 * 2^-148
        (0x0040_0000, 0.5f32.to_bits(), -126),
        (0x0000_0003, 0.75f32.to_bits(), -147),
        (0x8000_0001, (-0.5f32).to_bits(), -148),
    ];
    let f32_in: Vec<[u32; 4]> = f32_cases.iter().map(|&(x, _, _)| [x, 0, 0, 0]).collect();
    run_cases(&mant_f32, &f32_in, |backend, i, inp, got| {
        let want = f32_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_frexp_mant_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    run_cases(&exp_f32, &f32_in, |backend, i, inp, got| {
        let want = f32_cases[i].2 as u32;
        assert_eq!(
            got, want,
            "{backend}: v_frexp_exp_i32_f32({:#010x}) = {} ({got:#010x}), want {}",
            inp[0], got as i32, f32_cases[i].2,
        );
    });

    // NaN: mantissa stays NaN, exponent is 0.
    let qnan = 0x7FC0_0000u32;
    run_cases(&mant_f32, &[[qnan, 0, 0, 0]], |backend, _, _, got| {
        assert!(
            f32::from_bits(got).is_nan(),
            "{backend}: v_frexp_mant_f32(NaN) = {got:#010x}, want NaN",
        );
    });
    run_cases(&exp_f32, &[[qnan, 0, 0, 0]], |backend, _, _, got| {
        assert_eq!(got, 0, "{backend}: v_frexp_exp_i32_f32(NaN) = {got}, want 0");
    });

    // f16 cases: (input f16, expected mantissa f16, expected i16 exponent)
    // 0x0000=+0, 0x8000=-0, 0x3800=0.5, 0x3C00=1.0, 0xC600=-6.0 (-0.75 * 2^3, -0.75=0xBA00),
    // 0x7BFF=65504 (max f16 -> mant=0x3BFF, exp=16), 0x0001=2^-24 (min subnormal -> 0.5 * 2^-23),
    // 0x8003=-3*2^-24 (-0.75 * 2^-22), 0x7C00=+inf, 0xFC00=-inf.
    let sentinel = 0xDEAD_0000u32;
    let f16_cases: &[(u16, u16, i16)] = &[
        (0x0000, 0x0000, 0),
        (0x8000, 0x8000, 0),
        (0x3800, 0x3800, 0),
        (0x3C00, 0x3800, 1),
        (0xC600, 0xBA00, 3),
        (0x7BFF, 0x3BFF, 16),
        (0x0001, 0x3800, -23),
        (0x8003, 0xBA00, -22),
        (0x7C00, 0x7C00, 0),
        (0xFC00, 0xFC00, 0),
    ];
    let f16_in: Vec<[u32; 4]> = f16_cases
        .iter()
        .map(|&(x, _, _)| [u32::from(x), sentinel, 0, 0])
        .collect();
    run_cases(&mant_f16, &f16_in, |backend, i, inp, got| {
        let want = sentinel | u32::from(f16_cases[i].1);
        assert_eq!(
            got, want,
            "{backend}: v_frexp_mant_f16({:#06x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    run_cases(&exp_f16, &f16_in, |backend, i, inp, got| {
        let want = sentinel | u32::from(f16_cases[i].2 as u16);
        assert_eq!(
            got, want,
            "{backend}: v_frexp_exp_i16_f16({:#06x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });

    // f16 NaN: mantissa is f16 NaN (upper half preserved), exponent is 0.
    run_cases(&mant_f16, &[[0x7E00, sentinel, 0, 0]], |backend, _, _, got| {
        assert_eq!(got & 0xFFFF_0000, sentinel);
        let lo = (got & 0x7FFF) as u16;
        assert!(lo > 0x7C00, "{backend}: v_frexp_mant_f16(NaN) lo={lo:#06x}, want NaN");
    });
    run_cases(&exp_f16, &[[0x7E00, sentinel, 0, 0]], |backend, _, _, got| {
        assert_eq!(got, sentinel, "{backend}: v_frexp_exp_i16_f16(NaN) = {got:#010x}, want 0");
    });
}
#[test]
fn v_mad_16_and_int16_clamp_match_rdna2() {
    let vop3_clamp = |op: u16, s0: u16, s1: u16, s2: u16, clamp: bool| {
        let [mut w0, w1] = vop3(op, 0, s0, s1, s2);
        if clamp {
            w0 |= 1 << 15;
        }
        [w0, w1]
    };
    let with_sentinel = |words: [u32; 2]| {
        build_valu_cs(&[
            vop1(0x001, 0, vgpr(4)), // v_mov_b32 v0, v4 (sentinel upper half)
            words[0],
            words[1],
        ])
    };

    // VOP3 opcodes: 0x303=v_add_nc_u16, 0x304=v_sub_nc_u16, 0x340=v_mad_u16, 0x35e=v_mad_i16.
    let mad_u16_wrap = with_sentinel(vop3_clamp(0x340, vgpr(1), vgpr(2), vgpr(3), false));
    let mad_u16_sat = with_sentinel(vop3_clamp(0x340, vgpr(1), vgpr(2), vgpr(3), true));
    let mad_i16_wrap = with_sentinel(vop3_clamp(0x35e, vgpr(1), vgpr(2), vgpr(3), false));
    let mad_i16_sat = with_sentinel(vop3_clamp(0x35e, vgpr(1), vgpr(2), vgpr(3), true));
    let add_u16_sat = with_sentinel(vop3_clamp(0x303, vgpr(1), vgpr(2), 0, true));
    let sub_u16_sat = with_sentinel(vop3_clamp(0x304, vgpr(1), vgpr(2), 0, true));

    let sentinel = 0xBEEF_0000u32;

    // v_mad_u16 wrapping & saturating (including u16::MAX * u16::MAX + u16::MAX without u32 wrap)
    let u_cases: &[(u16, u16, u16, u16, u16)] = &[
        (20, 30, 40, 640, 640),
        (300, 300, 100, (90_100u32 & 0xFFFF) as u16, u16::MAX),
        (u16::MAX, u16::MAX, u16::MAX, 0, u16::MAX),
    ];
    let u_in: Vec<[u32; 4]> = u_cases
        .iter()
        .map(|&(a, b, c, _, _)| [u32::from(a), u32::from(b), u32::from(c), sentinel])
        .collect();
    run_cases(&mad_u16_wrap, &u_in, |backend, i, _, got| {
        let want = sentinel | u32::from(u_cases[i].3);
        assert_eq!(got, want, "{backend}: v_mad_u16 wrap case {i}");
    });
    run_cases(&mad_u16_sat, &u_in, |backend, i, _, got| {
        let want = sentinel | u32::from(u_cases[i].4);
        assert_eq!(got, want, "{backend}: v_mad_u16 clamp case {i}");
    });

    // v_mad_i16 wrapping & saturating (positive & negative overflow extremes)
    let i_cases: &[(i16, i16, i16, i16, i16)] = &[
        (-10, 20, 50, -150, -150),
        (i16::MIN, i16::MIN, i16::MAX, i16::MAX, i16::MAX),
        (i16::MIN, i16::MAX, i16::MIN, 0, i16::MIN),
        (200, 200, 100, (40_100i32 as i16), i16::MAX),
        (-200, 200, -100, (-40_100i32 as i16), i16::MIN),
    ];
    let i_in: Vec<[u32; 4]> = i_cases
        .iter()
        .map(|&(a, b, c, _, _)| [u32::from(a as u16), u32::from(b as u16), u32::from(c as u16), sentinel])
        .collect();
    run_cases(&mad_i16_wrap, &i_in, |backend, i, _, got| {
        let want = sentinel | u32::from(i_cases[i].3 as u16);
        assert_eq!(got, want, "{backend}: v_mad_i16 wrap case {i}");
    });
    run_cases(&mad_i16_sat, &i_in, |backend, i, _, got| {
        let want = sentinel | u32::from(i_cases[i].4 as u16);
        assert_eq!(got, want, "{backend}: v_mad_i16 clamp case {i}");
    });

    // v_add_nc_u16 clamp & v_sub_nc_u16 clamp
    run_cases(
        &add_u16_sat,
        &[
            [60000, 10000, 0, sentinel],
            [123, 456, 0, sentinel],
        ],
        |backend, i, _, got| {
            let want = sentinel | [u32::from(u16::MAX), 579][i];
            assert_eq!(got, want, "{backend}: v_add_nc_u16 clamp case {i}");
        },
    );
    run_cases(
        &sub_u16_sat,
        &[
            [5, 10, 0, sentinel],
            [500, 200, 0, sentinel],
        ],
        |backend, i, _, got| {
            let want = sentinel | [0, 300][i];
            assert_eq!(got, want, "{backend}: v_sub_nc_u16 clamp case {i}");
        },
    );
}
/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

#[test]
fn v_ldexp_f16_matches_rdna2() {
    // VOP2 `v_ldexp_f16 v0, v1, v2` (opcode 0x3b), preserving upper half from v3.
    let vop2_cs = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)),   // v_mov_b32 v0, v3 (sentinel upper half)
        vop2(0x3b, 0, vgpr(1), 2), // v_ldexp_f16 v0, v1, v2
    ]);
    // VOP3 `v_ldexp_f16 v0, |v1|, v2 clamp` (opcode 0x13b) with op_sel:[1, 0, 0, 1]
    // (reads high half of v1, low half of v2, writes high half of v0).
    let [mut w0, w1] = vop3(0x13b, 0, vgpr(1), vgpr(2), 0);
    w0 |= (1 << 8) | (1 << 11) | (1 << 14) | (1 << 15); // abs[0], opsel[0], opsel[3], clamp
    let vop3_cs = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)), // v_mov_b32 v0, v3 (sentinel lower half)
        w0,
        w1,
    ]);

    let sentinel = 0xCAFE_0000u32;
    // (f16 input, i16 exponent, expected f16 output)
    let cases: &[(u16, i16, u16)] = &[
        (0x3C00, 0, 0x3C00),         // 1.0 * 2^0 = 1.0
        (0x3E00, 3, 0x4A00),         // 1.5 * 2^3 = 12.0
        (0xC200, -2, 0xBA00),        // -3.0 * 2^-2 = -0.75
        (0x3C00, -15, 0x0200),       // 1.0 * 2^-15 = f16 subnormal 0x0200
        (0x0001, 24, 0x3C00),        // min f16 subnormal (2^-24) * 2^24 = 1.0
        (0x7BFF, 0, 0x7BFF),         // max finite f16 * 2^0 stays 65504 (no premature inf)
        (0x3C00, 16, 0x7C00),        // 1.0 * 2^16 = 65536 -> +inf (RTE overflow)
        (0xBC00, 1000, 0xFC00),      // -1.0 * 2^1000 -> -inf
        (0x0001, i16::MAX, 0x7C00),  // min subnormal * 2^32767 -> +inf
        (0x7BFF, i16::MIN, 0x0000),  // max finite * 2^-32768 -> +0.0
        (0xFBFF, i16::MIN, 0x8000),  // -max finite * 2^-32768 -> -0.0
        (0x0000, i16::MAX, 0x0000),  // +0.0 * 2^32767 = +0.0 (not NaN)
        (0x8000, i16::MAX, 0x8000),  // -0.0 * 2^32767 = -0.0 (not NaN)
        (0x7C00, i16::MIN, 0x7C00),  // +inf * 2^-32768 = +inf (not NaN)
        (0xFC00, i16::MIN, 0xFC00),  // -inf * 2^-32768 = -inf (not NaN)
    ];
    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(a, e, _)| [u32::from(a), u32::from(e as u16), sentinel, 0])
        .collect();
    run_cases(&vop2_cs, &inputs, |backend, i, _, got| {
        let (a, e, want_h) = cases[i];
        let want = sentinel | u32::from(want_h);
        assert_eq!(
            got, want,
            "{backend}: v_ldexp_f16({a:#06x}, {e}) = {got:#010x}, want {want:#010x}",
        );
    });

    // NaN stays NaN (upper half preserved).
    run_cases(&vop2_cs, &[[0x7E00, 4, sentinel, 0]], |backend, _, _, got| {
        assert_eq!(got & 0xFFFF_0000, sentinel);
        let lo = (got & 0x7FFF) as u16;
        assert!(lo > 0x7C00, "{backend}: v_ldexp_f16(NaN, 4) lo={lo:#06x}, want NaN");
    });

    // VOP3 abs[0] + opsel:[1,0,0,1] + clamp:
    // |-0.25(=0xB400)| * 2^1 = 0.5 (0x3800) into high half; |-1.0(=0xBC00)| * 2^2 = 4.0 -> clamped 1.0 (0x3C00)
    let lo_sentinel = 0x0000_1234u32;
    let vop3_cases: &[([u32; 4], u32)] = &[
        ([0xB400_0000, 1, lo_sentinel, 0], (0x3800u32 << 16) | lo_sentinel),
        ([0xBC00_0000, 2, lo_sentinel, 0], (0x3C00u32 << 16) | lo_sentinel),
    ];
    let vop3_in: Vec<[u32; 4]> = vop3_cases.iter().map(|c| c.0).collect();
    run_cases(&vop3_cs, &vop3_in, |backend, i, _, got| {
        let want = vop3_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_ldexp_f16_e64 case {i} = {got:#010x}, want {want:#010x}",
        );
    });
}
