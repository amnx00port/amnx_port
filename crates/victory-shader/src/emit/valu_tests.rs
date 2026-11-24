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
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
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

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
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
}

fn validate_dxil(dxil: &[u8]) {
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = dxil.to_vec();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }
}

/// Builds a synthetic compute shader with explicit thread-group size, optional
/// 512-byte LDS allocation, custom `loop_limit`, and an `out_dwords` raw UAV
/// at heap index 0 (`s[8:11]` loaded from table slot 1).
fn build_custom_cs(threads: u32, use_lds: bool, loop_limit: u32, body: &[u32]) -> HostShader {
    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80));
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.extend_from_slice(body);
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let rsrc2 = (5 << 1) | if use_lds { 1 << 15 } else { 0 };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
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
            rsrc2,
            threads: [threads, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower_with_limit(
        &lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        },
        loop_limit,
    )
    .expect("lower_with_limit");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };
    validate_dxil(&shader.dxil);
    shader
}

const fn vopc(op: u8, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | (op as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn sopp(op: u8, imm: i16) -> u32 {
    0xBF80_0000 | (op as u32) << 16 | (imm as u16 as u32)
}

const fn ds_op(op: u8, vdst: u8, vaddr: u8, vdata0: u8, offset: u16) -> [u32; 2] {
    [
        0x36 << 26 | (op as u32) << 18 | (offset as u32),
        (vdst as u32) << 24 | (vdata0 as u32) << 8 | (vaddr as u32),
    ]
}

const fn buffer_store_dword_offen(vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (1 << 12),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | (vaddr as u32),
    ]
}

/// A minimal full-screen-triangle vertex shader (`float2 POSITION0 -> SV_Position`)
/// for running pixel shaders with [`Host::draw`].
fn fullscreen_vs() -> HostShader {
    use super::bitcode::{Attr, Term};
    use super::container::SigElement;
    use super::shader::Shader;

    let mut s = Shader::new(false);
    s.inputs.push(SigElement {
        name: "POSITION".into(),
        index: 0,
        kind: 0,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 2,
        usage: 0b0011,
        half: false,
    });
    s.outputs.push(SigElement {
        name: "SV_Position".into(),
        index: 0,
        kind: 3,
        comp: 9,
        interp: 4,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0b1111,
        half: false,
    });
    let b = s.entry;
    let undef = s.m.undef(s.t.i32);
    let (z0, c0, c1) = (s.m.i32(0), s.m.i8(0), s.m.i8(1));
    let x = s.op(
        b,
        "loadInput.f32",
        Attr::ReadNone,
        s.t.f32,
        4,
        &[z0, z0, c0, undef],
    );
    let y = s.op(
        b,
        "loadInput.f32",
        Attr::ReadNone,
        s.t.f32,
        4,
        &[z0, z0, c1, undef],
    );
    let z = s.m.f32(0.5f32.to_bits());
    let w = s.m.f32(1.0f32.to_bits());
    for (col, val) in [x, y, z, w].into_iter().enumerate() {
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col as u8), val];
        s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
    }
    s.m.terminate(b, Term::Ret);
    let dxil = s.finish();
    validate_dxil(&dxil);
    HostShader {
        stage: HwStage::Vs,
        dxil,
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
    }
}

#[test]
fn s_wqm_pixel_loop_and_skip_run_per_quad() {
    use super::container::SigElement;
    use super::host::Draw;

    // Pixel shader over a 4x2 target (`v0 = SV_Position.x`):
    // - Only `x == 0` starts with counter `v3 = 3` (all other pixels start at `0`).
    // - `s_wqm_b32 exec_lo, vcc_lo` + `s_cbranch_execz` skips quads where no
    //   lane has `x == 0` (the right 2x2 quad `x = 2..3` must output `0.0`).
    // - Inside the left 2x2 quad (`x = 0..1`), all 4 pixels stay in the loop
    //   via `s_wqm_b32` + `s_cbranch_execnz` while `x == 0` counts down `3..0`,
    //   exiting cleanly at `3.0` instead of spinning to `loop_limit = 16`.
    let words: &[u32] = &[
        vop1(0x007, 1, vgpr(0)), //  0: v_cvt_u32_f32 v1, v0
        vop1(0x001, 2, 0x80),    //  4: v_mov_b32 v2, 0
        vopc(0xC2, 0x80, 1),     //  8: v_cmp_eq_u32 vcc_lo, 0, v1
        vop1(0x001, 3, 0x83),    // 12: v_mov_b32 v3, 3
        vop2(0x01, 3, 0x80, 3),  // 16: v_cndmask_b32 v3, 0, v3, vcc_lo
        0xBE82_037E,             // 20: s_mov_b32 s2, exec_lo
        0xBEFE_096A,             // 24: s_wqm_b32 exec_lo, vcc_lo
        sopp(0x08, 7),           // 28: s_cbranch_execz -> 60
        // loop (offset 32):
        vop2(0x25, 2, 0x81, 2), // 32: v_add_nc_u32 v2, 1, v2
        vopc(0xC1, 0x80, 3),    // 36: v_cmp_lt_u32 vcc_lo, 0, v3
        vop2(0x25, 4, 0xC1, 3), // 40: v_add_nc_u32 v4, -1, v3
        vop2(0x01, 3, 0x80, 4), // 44: v_cndmask_b32 v3, 0, v4, vcc_lo
        vopc(0xC1, 0x80, 3),    // 48: v_cmp_lt_u32 vcc_lo, 0, v3
        0xBEFE_096A,            // 52: s_wqm_b32 exec_lo, vcc_lo
        sopp(0x09, -7),         // 56: s_cbranch_execnz -> 32
        // done (offset 60):
        0xBEFE_0302,             // 60: s_mov_b32 exec_lo, s2
        vop1(0x006, 2, vgpr(2)), // 64: v_cvt_f32_u32 v2, v2
        0xF800_180F,             // 68: exp mrt0 v2, v2, v2, v2 done vm
        0x0202_0202,
        S_ENDPGM, // 76: s_endpgm
    ];

    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 1 << 8, // Position.x in v0
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower_with_limit(
        &lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![SigElement {
                name: "SV_Position".into(),
                index: 0,
                kind: 3,
                comp: 9,
                interp: 4,
                row: Some(0),
                col: 0,
                cols: 4,
                usage: 0,
                half: false,
            }],
            outputs: vec![SigElement {
                name: "SV_Target".into(),
                index: 0,
                kind: 16,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage: 0,
                half: false,
            }],
            prims: Vec::new(),
            mesh_stride: 0,
        },
        16,
    )
    .expect("lower_with_limit");
    validate_dxil(&lowered.dxil);

    let ps = HostShader {
        stage: HwStage::Ps,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
    };
    let vs = fullscreen_vs();
    let vertices = [-1.0f32, -1.0, 3.0, -1.0, -1.0, 3.0];

    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host.pipeline(&vs, &ps).expect("graphics pipeline");
        let pixels = host
            .draw(
                &root,
                &pso,
                &Draw {
                    roots: vec![(0, Root::Constants(vec![0])), (1, Root::Constants(vec![0]))],
                    textures: &[],
                    samplers: &[],
                    vertices: &vertices,
                    stride: 8,
                    size: (4, 2),
                    mesh_groups: 0,
                },
            )
            .expect("draw");
        assert_eq!(pixels.len(), 8);
        for y in 0..2usize {
            for x in 0..4usize {
                let want = if x < 2 { 3.0 } else { 0.0 };
                let got = pixels[y * 4 + x][0];
                assert_eq!(got, want, "{label}: pixel ({x}, {y}) = {got}, want {want}");
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

#[test]
fn oob_lds_reads_writes_and_atomics_clamp_and_drop() {
    // 64 threads, 512 bytes (128 dwords) of LDS (`s.wave_size == None`, so both
    // hardware and WARP run it):
    // 1. Write `lds[tid] = tid + 100` for `tid < 64`, then `s_barrier`.
    // 2. Under a `tid == 0` guard (`v_cmpx`), lanes `1..63` hold an OOB byte
    //    address (`1024`, dword `256 >= 128`) while lane 0 reads `lds[0]`:
    //    inactive lanes keep their sentinel `77`, lane 0 reads `100`.
    // 3. Every lane performs an active OOB `ds_write_b32`, OOB `ds_add_rtn_u32`,
    //    and OOB `ds_read_b32` at byte offset `512` (dword `128 >= 128`):
    //    both OOB reads/atomics return `0`, and `lds[0..64]` stays intact.
    let mut body = vec![
        vop2(0x1A, 1, 0x82, 0),    // v1 = tid * 4 (in-bounds byte offset 0..252)
        vop2(0x25, 2, sgpr(0), 0), // v2 = tid + 100 (`s0 = 100` from root constants)
    ];
    body.extend_from_slice(&ds_op(0x0D, 0, 1, 2, 0)); // ds_write_b32 v1, v2
    body.extend([
        sopp(0x0A, 0),             // s_barrier
        vop1(0x001, 3, sgpr(1)),   // v3 = 77 (`s1 = 77`)
        vop1(0x001, 4, sgpr(2)),   // v4 = 1024 (`s2 = 1024`, OOB)
        vopc(0xC2, 0x80, 0),       // v_cmp_eq_u32 vcc_lo, 0, v0
        vop1(0x001, 5, 0x80),      // v_mov_b32 v5, 0
        vop2(0x01, 4, sgpr(2), 5), // v_cndmask_b32 v4, s2, v5, vcc_lo (`tid == 0 ? 0 : 1024`)
        0xBE86_3C6A,               // s_and_saveexec_b32 s6, vcc_lo
    ]);
    body.extend_from_slice(&ds_op(0x36, 3, 4, 0, 0)); // ds_read_b32 v3, v4
    body.extend([
        0x887E_067E,             // s_or_b32 exec_lo, exec_lo, s6
        vop1(0x001, 6, sgpr(3)), // v6 = s3 (999)
    ]);
    // Active OOB write, OOB atomic_add_rtn, and OOB read at offset 512 (dword 128 == lds_dwords).
    body.extend_from_slice(&ds_op(0x0D, 0, 5, 6, 512)); // ds_write_b32 v5(0), v6 offset:512
    body.extend_from_slice(&ds_op(0x20, 7, 5, 6, 516)); // ds_add_rtn_u32 v7, v5(0), v6 offset:516
    body.extend_from_slice(&ds_op(0x36, 8, 5, 0, 512)); // ds_read_b32 v8, v5(0) offset:512
    body.push(sopp(0x0A, 0)); // s_barrier

    // Verify in-bounds `lds[tid]` was not corrupted: `v9 = ds_read_b32(v1)`.
    body.extend_from_slice(&ds_op(0x36, 9, 1, 0, 0));
    // Pack `(v3 << 16) | v9` + `v7` + `v8` into `uav[tid]`:
    body.extend([
        vop2(0x1A, 3, 0x90, 3),    // v_lshlrev_b32 v3, 16, v3
        vop2(0x1C, 9, vgpr(3), 9), // v_or_b32 v9, v3, v9
        vop2(0x25, 9, vgpr(7), 9), // v_add_nc_u32 v9, v7, v9
        vop2(0x25, 9, vgpr(8), 9), // v_add_nc_u32 v9, v8, v9
    ]);
    body.extend_from_slice(&buffer_store_dword_offen(9, 1, 8));

    let shader = build_custom_cs(64, true, 1024, &body);
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host.compute_pipeline(&shader).expect("compute_pipeline");
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let inputs = [100u32, 77, 1024, 999];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Constants, 0) => inputs[u.dword as usize],
                (RootType::Table, 1) => 0,
                other => panic!("unexpected {other:?}"),
            };
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch");
        for tid in 0..64u32 {
            let want_guarded = if tid == 0 { 100 } else { 77 };
            let want = (want_guarded << 16) | (100 + tid);
            assert_eq!(
                out[0][tid as usize], want,
                "{label}: lane {tid} got {:#010x}, want {want:#010x}",
                out[0][tid as usize]
            );
        }
        ran += 1;
    }
    assert!(ran > 0);
}

#[test]
fn nested_loops_share_invocation_step_budget() {
    // Two nested loops whose conditions always hold (`s0 == 0` -> `scc1` taken):
    // with per-loop reset counters and `limit = 25`, the inner body would run
    // `25 * 25 = 625` times; with a function-wide step budget of `25`, total
    // inner-body executions are capped at `25`.
    let body: &[u32] = &[
        vop1(0x001, 0, 0x80), //  0: v_mov_b32 v0, 0
        // outer (offset 4):
        // inner (offset 4):
        vop2(0x25, 0, 0x81, 0), //  4: v_add_nc_u32 v0, 1, v0
        0xBF06_8000,            //  8: s_cmp_eq_u32 s0, 0
        sopp(0x05, -3),         // 12: s_cbranch_scc1 -> 4 (inner back-edge)
        0xBF06_8000,            // 16: s_cmp_eq_u32 s0, 0
        sopp(0x05, -5),         // 20: s_cbranch_scc1 -> 4 (outer back-edge)
        buffer_store_dword(0, 8, 0)[0],
        buffer_store_dword(0, 8, 0)[1],
    ];
    let shader = build_custom_cs(1, false, 25, body);
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host.compute_pipeline(&shader).expect("compute_pipeline");
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = 0;
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
            .expect("dispatch");
        assert!(
            out[0][0] <= 25 && out[0][0] >= 24,
            "{label}: nested loops ran {} iterations, expected ~25 (not 625)",
            out[0][0]
        );
        ran += 1;
    }
    assert!(ran > 0);
}

#[test]
fn divergent_loop_and_barrier_branches_do_not_deadlock() {
    // Part 1 (64 threads -> multi-wave on both Wave32 hardware and Wave4 WARP,
    // `loop_limit = 40`):
    // - First, a per-lane divergent loop runs `tid / 2` steps (`0..31` steps),
    //   leaving every lane/wave with a different `steps[0]` count.
    // - Next, an infinite loop containing `s_barrier` runs until `loop_limit`:
    //   because barrier loops use their own group-uniform `steps[1]` budget,
    //   all waves exit on the exact same iteration (`40`) without deadlocking!
    let mut multi_wave = vec![
        vop2(0x16, 1, 0x81, 0), // v1 = tid >> 1 (0..31)
        vop1(0x001, 2, 0x80),   // v2 = 0 (divergent counter)
        0xBE86_037E,            // s6 = saved full exec_lo
        vopc(0xC1, vgpr(2), 1), // v_cmp_lt_u32 vcc_lo, v2, v1
        0xBEFE_036A,            // s_mov_b32 exec_lo, vcc_lo
        sopp(0x08, 4),          // s_cbranch_execz -> after_div
        // div_loop:
        vop2(0x25, 2, 0x81, 2), // v2 += 1
        vopc(0xC1, vgpr(2), 1), // v_cmp_lt_u32 vcc_lo, v2, v1
        0x877E_6A7E,            // s_and_b32 exec_lo, exec_lo, vcc_lo
        sopp(0x09, -4),         // s_cbranch_execnz -> div_loop
        // after_div:
        0xBEFE_0306,          // s_mov_b32 exec_lo, s6
        vop1(0x001, 3, 0x80), // v3 = 0 (barrier loop counter)
        // bar_loop:
        vop2(0x25, 3, 0x81, 3), // v3 += 1
        sopp(0x0A, 0),          // s_barrier
        0xBF06_8000,            // s_cmp_eq_u32 s0, 0 (always true)
        sopp(0x05, -4),         // s_cbranch_scc1 -> bar_loop
        // Write `(v2 << 16) | v3` to `uav[tid]`:
        vop2(0x1A, 4, 0x82, 0),    // v4 = tid * 4
        vop2(0x1A, 2, 0x90, 2),    // v2 <<= 16
        vop2(0x1C, 3, vgpr(2), 3), // v3 |= v2
    ];
    multi_wave.extend_from_slice(&buffer_store_dword_offen(3, 4, 8));

    let mw_shader = build_custom_cs(64, true, 40, &multi_wave);
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host.compute_pipeline(&mw_shader).expect("compute_pipeline");
        let ud = vec![0u32; mw_shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch");
        for tid in 0..64u32 {
            let want = ((tid >> 1) << 16) | 40;
            assert_eq!(
                out[0][tid as usize], want,
                "{label}: lane {tid} got {:#010x}, want {want:#010x}",
                out[0][tid as usize]
            );
        }
        ran += 1;
    }
    assert!(ran > 0);

    // Part 2 (single 32-lane wave): `if (tid == 0)` via `s_and_saveexec_b32` +
    // `s_cbranch_execz` enclosing both a `ds_write_b32` and an `s_barrier`.
    // Validates with `dxil.dll` and executes on Wave32-capable hardware without
    // deadlocking `dx.op.barrier` or letting lanes 1..31 clobber `lds[0]`.
    let mut single_wave = vec![
        vop1(0x001, 1, 0x80),      // v1 = 0 (byte offset 0)
        vop2(0x25, 2, sgpr(0), 0), // v2 = 777 + tid (lane 0 has 777, others != 777)
        vopc(0xC2, 0x80, 0),       // v_cmp_eq_u32 vcc_lo, 0, v0
        0xBE86_3C6A,               // s_and_saveexec_b32 s6, vcc_lo
        sopp(0x08, 3),             // s_cbranch_execz -> join (ds_write[2] + s_barrier[1])
    ];
    single_wave.extend_from_slice(&ds_op(0x0D, 0, 1, 2, 0)); // ds_write_b32 v1, v2
    single_wave.extend([
        sopp(0x0A, 0), // s_barrier
        0x887E_067E,   // join: s_or_b32 exec_lo, exec_lo, s6
    ]);
    single_wave.extend_from_slice(&ds_op(0x36, 3, 1, 0, 0)); // v3 = ds_read_b32(0)
    single_wave.push(vop2(0x1A, 4, 0x82, 0)); // v4 = tid * 4
    single_wave.extend_from_slice(&buffer_store_dword_offen(3, 4, 8));

    let sw_shader = build_custom_cs(32, true, 40, &single_wave);
    if let Ok(host) = Host::new(false)
        && let Ok((root, pso)) = host.compute_pipeline(&sw_shader)
    {
        let mut ud = vec![0u32; sw_shader.user_data_dwords().max(1) as usize];
        for u in &sw_shader.user_data {
            if (u.root, u.slot) == (RootType::Constants, 0) {
                ud[u.index as usize] = 777;
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(32)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("single-wave barrier dispatch");
        for (tid, &got) in out[0][..32].iter().enumerate() {
            assert_eq!(got, 777, "lane {tid} read {got} instead of lane 0's 777");
        }
    }
}
