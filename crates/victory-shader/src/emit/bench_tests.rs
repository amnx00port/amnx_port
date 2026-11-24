//! Timing of the translation stages on synthetic programs.
//!
//! No game data: the programs are built here, word by word, like the ones in
//! `valu_tests`. Run with
//!
//! ```text
//! cargo test --release -p victory-shader --lib bench_tests -- --ignored --nocapture
//! ```
//!
//! It prints nanoseconds per instruction for each stage at several program
//! sizes, so a change can be measured against the same program rather than
//! against a wall clock on somebody else's corpus.

use super::lower;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};
use std::time::{Duration, Instant};

const S_ENDPGM: u32 = 0xBF81_0000;

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

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

/// A straight-line compute program with `body` VALU instructions, each one
/// reading the two VGPRs the previous one wrote, so nothing folds away.
fn synthetic(body: usize) -> NativeProgram {
    let mut words = Vec::with_capacity(body * 2 + 16);
    words.push(s_mov_b32(5, 0x80));
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i)));
    }
    for i in 0..body {
        // v_add_f32_e64 v[3], v1, v2 - VOP3 0x103.
        words.extend_from_slice(&vop3(0x103, 3, vgpr(1 + (i % 2) as u8), vgpr(2), 0));
    }
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    NativeProgram {
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
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code,
    }
}

fn best_of<T>(rounds: usize, mut f: impl FnMut() -> T) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..rounds {
        let t = Instant::now();
        let out = f();
        best = best.min(t.elapsed());
        drop(out);
    }
    best
}

#[test]
#[ignore = "timing; run with --ignored --nocapture"]
fn translation_stages_by_size() {
    println!(
        "\n{:>7} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "body", "ir.build", "structure", "resource", "io", "lower", "total"
    );
    for body in [256usize, 1024, 4096, 16384] {
        let native = synthetic(body);
        let ninst = body as f64 + 8.0;
        let rounds = if body >= 8192 { 5 } else { 11 };

        let t_ir = best_of(rounds, || {
            ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build")
        });
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let t_st = best_of(rounds, || structure::structure(&p).expect("structure"));
        let s = structure::structure(&p).expect("structure");
        let t_res = best_of(rounds, || {
            resource::track(&p, &native.layout, native.stage, Some(&s)).expect("resource::track")
        });
        let r = resource::track(&p, &native.layout, native.stage, Some(&s)).expect("resource");
        let t_io = best_of(rounds, || io::raise(&p, &native).expect("io::raise"));
        let i = io::raise(&p, &native).expect("io::raise");
        let t_low = best_of(rounds, || {
            lower::lower(&lower::Parts {
                native: &native,
                p: &p,
                st: &s,
                io: &i,
                res: &r,
                inputs: Vec::new(),
                outputs: Vec::new(),
                prims: Vec::new(),
                mesh_stride: 0,
            })
            .expect("lower")
        });

        let ns = |d: Duration| d.as_nanos() as f64 / ninst;
        let total = t_ir + t_st + t_res + t_io + t_low;
        let lowered = lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &s,
            io: &i,
            res: &r,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .expect("lower");
        let mut fnv: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in &lowered.dxil {
            fnv ^= u64::from(b);
            fnv = fnv.wrapping_mul(0x1000_0000_01b3);
        }
        println!(
            "{:>7} {:>10.0} {:>10.0} {:>10.0} {:>10.0} {:>10.0} {:>10.0}   dxil={} bytes fnv={:016x}",
            body,
            ns(t_ir),
            ns(t_st),
            ns(t_res),
            ns(t_io),
            ns(t_low),
            ns(total),
            lowered.dxil.len(),
            fnv
        );
    }
    println!("(nanoseconds per instruction; best of N rounds)\n");
}
