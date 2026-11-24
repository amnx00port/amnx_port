//! I/O raising (decision 0003): what a program's input registers, its
//! interpolation and its exports mean.
//!
//! Pixel shaders get the VGPRs `SPI_PS_INPUT_ADDR` enables, in a fixed
//! order (barycentrics, then position, front face...). `v_interp_p1` and
//! `v_interp_p2` (and their f16 forms) evaluate an attribute from a pair of
//! barycentrics; `v_interp_mov` reads one vertex's value. After the 32-bit
//! export patch ([`NativeProgram::patched_code`]) exports carry plain
//! 32-bit values: `mrt<n>` is render target n, `mrtz` depth, stencil and
//! sample mask, `null` nothing. A lane whose EXEC bit is clear at the
//! export that ends the program (`done`) is discarded: that export's guard
//! after structurisation.
//!
//! Vertex shaders that run as the hardware VS get the vertex id in v0, the
//! primitive id in v2 and the instance id in v3; the fetch shader loads
//! channel k of each attribute into its VGPR + k. `exp pos0` is the
//! position, `exp param<n>` the output with parameter n.
//!
//! Attributes and outputs are named by their row: a register of the
//! container's signature part.
//!
//! Compute programs get the thread id in the group in v0-v2 and the group
//! id in the SGPRs after the user data, as `COMPUTE_PGM_RSRC2` enables them.
//!
//! Mesh programs run as one NGG stage. The program computes its thread
//! index from the wave index (s3 bits 24-27) and the lane (`v_mbcnt`), and
//! its group id from v5 of the first lane (`v_readfirstlane`: the group
//! times the group size). Lane n exports vertex n (`exp pos`, `exp param`)
//! and primitive n (`exp prim`: 10-bit vertex indices at bits 0, 10, 20,
//! bit 31 culls); `param` entries with bit 7 are per-primitive outputs. One
//! lane sends the output counts (`s_sendmsg` GS_ALLOC_REQ, m0 = vertices |
//! primitives << 12).

use crate::ShaderError;
use crate::gfx10::{Fields, Operand};
use crate::ir::{EXEC, Inst, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, PsInput, StageIo, Usage};
use crate::resource::Site;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    VertexId,
    PrimitiveId,
    InstanceId,
    /// A vertex attribute the fetch shader loads.
    Attribute {
        row: u8,
        channel: u8,
    },
    /// `index` 0 is i, 1 is j (2 the pull model's w).
    Barycentric {
        kind: Barycentric,
        index: u8,
    },
    /// Pixel position channel (0-3: x, y, z, w).
    Position(u8),
    FrontFace,
    /// Sample id and other bits.
    Ancillary,
    SampleCoverage,
    FixedPosition,
    LineStipple,
    /// Compute: thread-group id component (an SGPR).
    GroupId(u8),
    /// Compute: thread id within the group, component (a VGPR).
    ThreadIdInGroup(u8),
    /// Compute: the thread-group info SGPR (wave index and count).
    GroupInfo,
    /// Mesh: v5, the group id times the group size (read as uniform).
    MeshGroup,
    /// Mesh: s3, the wave's index in the group in bits 24-27.
    MergedWaveInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Barycentric {
    PerspectiveSample,
    PerspectiveCenter,
    PerspectiveCentroid,
    PerspectivePullModel,
    LinearSample,
    LinearCenter,
    LinearCentroid,
}

/// The pixel input VGPRs in hardware order: one entry per
/// `SPI_PS_INPUT_ADDR` bit.
const PS_VGPRS: [(u8, Option<Barycentric>); 16] = [
    (2, Some(Barycentric::PerspectiveSample)),
    (2, Some(Barycentric::PerspectiveCenter)),
    (2, Some(Barycentric::PerspectiveCentroid)),
    (3, Some(Barycentric::PerspectivePullModel)),
    (2, Some(Barycentric::LinearSample)),
    (2, Some(Barycentric::LinearCenter)),
    (2, Some(Barycentric::LinearCentroid)),
    (1, None), // line stipple
    (1, None), // position x
    (1, None), // y
    (1, None), // z
    (1, None), // w
    (1, None), // front face
    (1, None), // ancillary
    (1, None), // sample coverage
    (1, None), // fixed-point position
];

fn ps_system(bit: usize) -> Input {
    match bit {
        7 => Input::LineStipple,
        8..=11 => Input::Position(bit as u8 - 8),
        12 => Input::FrontFace,
        13 => Input::Ancillary,
        14 => Input::SampleCoverage,
        _ => Input::FixedPosition,
    }
}

/// An attribute channel of the input signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attribute {
    pub usage: Usage,
    pub row: u8,
    pub channel: u8,
    /// The upper f16 half of the channel (packed fp16 inputs).
    pub high: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Interp {
    /// This site's result (`v_interp_p2*`) is the attribute interpolated
    /// with `barycentric`; `first` are the `v_interp_p1*` halves it
    /// completes (one per path where the compiler shares a first half).
    Interpolated {
        attribute: Attribute,
        barycentric: Barycentric,
        first: Vec<Site>,
    },
    /// One vertex's value (`v_interp_mov_f32`), as encoded: 0 for P10,
    /// 1 for P20, 2 for P0 (the provoking vertex).
    Vertex { attribute: Attribute, vertex: u8 },
    /// A half of an interpolation whose i or j is computed rather than
    /// loaded: each half evaluates as the hardware does, from the
    /// attribute's vertex values (`v_interp_p1*`: P0 + i P10, with P10 =
    /// P1 - P0; `v_interp_p2*`: its old dst + j P20, with P20 = P2 - P0).
    Weighted { attribute: Attribute },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Color(u8),
    /// Depth, stencil, sample mask (channels as `DB_SHADER_CONTROL` enables).
    Depth,
    Null,
    Position(u8),
    Param {
        usage: Usage,
        row: u8,
    },
    /// `param0` of a vertex shader without outputs: the hardware wants one
    /// parameter export regardless.
    Unread,
    /// The primitive export of a merged or mesh stage.
    Primitive,
    /// A mesh program's per-primitive output (a row of `PSG1`).
    PrimitiveParam {
        usage: Usage,
        row: u8,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Export {
    pub site: Site,
    pub target: Target,
    /// Per channel, as the instruction enables them.
    pub values: [Option<Value>; 4],
    /// Two f16 values per dword.
    pub compressed: bool,
    /// The program's last export (for pixels: kills the lanes not in EXEC).
    pub done: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Io {
    /// Input values the hardware loads, with their meaning; other VGPRs
    /// read before written hold undefined values.
    pub inputs: Vec<(Value, Input)>,
    pub interps: Vec<(Site, Interp)>,
    pub exports: Vec<Export>,
    /// Compute: how many SGPRs the hardware initializes
    /// ([`ComputeIo::initialized_sgprs`](crate::native::ComputeIo::initialized_sgprs)).
    pub initialized_sgprs: Option<u8>,
}

/// Raises the I/O of `p`, built from `native`'s patched code. Fails for
/// stages whose stage data is not read and on constructs not covered yet.
pub fn raise(p: &Program, native: &NativeProgram) -> Result<Io, ShaderError> {
    let unsupported = |offset, what| ShaderError::Unsupported { offset, what };
    let (loaded, ps_inputs, params) = match (&native.io, native.stage) {
        (
            StageIo::Ps {
                inputs, input_addr, ..
            },
            HwStage::Ps,
        ) => {
            // A pixel shader exports no parameters; `[]` never reaches `Unread`.
            let loaded = ps_vgprs(*input_addr)
                .into_iter()
                .map(|(r, i)| (Reg::V(r), i))
                .collect();
            (loaded, &inputs[..], &[][..])
        }
        (StageIo::Vs { inputs, outputs }, HwStage::Vs) => {
            let mut loaded = vec![
                (Reg::V(0), Input::VertexId),
                (Reg::V(2), Input::PrimitiveId),
                (Reg::V(3), Input::InstanceId),
            ];
            for i in inputs {
                for channel in (0..4).filter(|c| i.mask >> c & 1 != 0) {
                    let input = Input::Attribute {
                        row: i.row,
                        channel,
                    };
                    loaded.push((Reg::V(i.vgpr + channel), input));
                }
            }
            (loaded, &[][..], &outputs[..])
        }
        // Compute: thread ids in v0-v2, group ids and the group info in the
        // SGPRs after the user data.
        (StageIo::Cs(cs), HwStage::Cs) => {
            let (ids, info) = cs.system_sgprs();
            let mut loaded: Vec<(Reg, Input)> = (0..cs.thread_id_vgprs())
                .map(|c| (Reg::V(c), Input::ThreadIdInGroup(c)))
                .collect();
            for (c, s) in ids.into_iter().enumerate() {
                if let Some(s) = s {
                    loaded.push((Reg::S(s), Input::GroupId(c as u8)));
                }
            }
            loaded.extend(info.map(|s| (Reg::S(s), Input::GroupInfo)));
            (loaded, &[][..], &[][..])
        }
        (StageIo::Ms(ms), HwStage::Ms) => {
            let group = p.inputs.iter().find(|i| i.0 == Reg::V(5)).map(|i| i.1);
            // The group input holds the same value in every lane only as the
            // first lane reads it; a VGPR write reads its old value too.
            let per_lane = p.blocks.iter().flat_map(|b| &b.insts).find(|i| {
                group.is_some_and(|g| i.reads.contains(&(Reg::V(5), g)))
                    && i.name() != "v_readfirstlane_b32"
                    && i.write(Reg::V(5)).is_none()
            });
            if let Some(i) = per_lane {
                return Err(unsupported(
                    i.machine.offset,
                    "the mesh group input read per lane",
                ));
            }
            let loaded = vec![
                (Reg::V(5), Input::MeshGroup),
                (Reg::S(3), Input::MergedWaveInfo),
            ];
            (loaded, &[][..], &ms.outputs[..])
        }
        _ => return Err(unsupported(0, "I/O of this stage")),
    };
    let inputs: Vec<(Value, Input)> = p
        .inputs
        .iter()
        .filter_map(|&(reg, v)| loaded.iter().find(|l| l.0 == reg).map(|l| (v, l.1)))
        .collect();
    let attribute = |inst: &Inst, attr: &Operand| match *attr {
        Operand::Attr { index, chan, high } => ps_inputs
            .iter()
            .find(|i: &&PsInput| i.attr == index)
            .map(|i| Attribute {
                usage: i.usage,
                row: i.row,
                channel: chan,
                high,
            })
            .ok_or(unsupported(
                inst.machine.offset,
                "an interpolated attribute the stage data does not list",
            )),
        _ => Err(unsupported(inst.machine.offset, "an interpolation operand")),
    };

    let mut io = Io {
        inputs: inputs.clone(),
        initialized_sgprs: match &native.io {
            StageIo::Cs(cs) => Some(cs.initialized_sgprs()),
            _ => None,
        },
        ..Io::default()
    };
    let defs: std::collections::HashMap<Value, Site> = p
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, block)| {
            block
                .insts
                .iter()
                .enumerate()
                .flat_map(move |(i, inst)| inst.writes.iter().map(move |w| (w.1, (b, i))))
        })
        .collect();
    let phis: std::collections::HashMap<Value, (usize, usize)> = p
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, block)| {
            block
                .phis
                .iter()
                .enumerate()
                .map(move |(i, phi)| (phi.value, (b, i)))
        })
        .collect();
    let halves = Halves {
        p,
        inputs: &inputs,
        defs: &defs,
        phis: &phis,
    };
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let site = (b, i);
            let at = inst.machine.offset;
            let name = inst.name();
            match &inst.fields {
                Fields::Vintrp { dst, src, attr } => {
                    let attribute = attribute(inst, attr)?;
                    match (name, *src) {
                        ("v_interp_mov_f32", Operand::Const(vertex)) => {
                            io.interps
                                .push((site, Interp::Vertex { attribute, vertex }));
                        }
                        ("v_interp_p1_f32", _) => {}
                        ("v_interp_p2_f32", Operand::Vector { reg, .. }) => {
                            let Operand::Vector { reg: d, .. } = *dst else {
                                return Err(unsupported(at, "an interpolation operand"));
                            };
                            // The first half's result is this one's old dst.
                            let interp = halves
                                .pair(attr, inst.read(Reg::V(d)), inst.read(Reg::V(reg)), inst)
                                .map_err(|what| unsupported(at, what))?;
                            push_pair(&mut io, site, interp, attribute);
                        }
                        _ => return Err(unsupported(at, "an interpolation form")),
                    }
                }
                Fields::Valu(v) if name.starts_with("v_interp_") => match name {
                    // First halves: `v_interp_p2_f16` pairs with them
                    // below. `v_interp_p1lv_f16`'s P0 comes from a lane
                    // VGPR (`v.src[2]`) instead of the local data share, but
                    // the pair only needs its site, attribute and I.
                    "v_interp_p1ll_f16" | "v_interp_p1lv_f16" => {}
                    "v_interp_p2_f16" => {
                        let attribute = attribute(inst, &v.src[0])?;
                        let read = |k: usize| match v.src.get(k) {
                            Some(&Operand::Vector { reg, .. }) => inst.read(Reg::V(reg)),
                            _ => None,
                        };
                        let interp = halves
                            .pair(&v.src[0], read(2), read(1), inst)
                            .map_err(|what| unsupported(at, what))?;
                        push_pair(&mut io, site, interp, attribute);
                    }
                    _ => return Err(unsupported(at, "an interpolation form")),
                },
                Fields::Exp {
                    target,
                    src,
                    compr,
                    done,
                    ..
                } => {
                    let target = match *target {
                        t @ 0..=7 => Target::Color(t),
                        8 => Target::Depth,
                        9 => Target::Null,
                        t @ 12..=16 => Target::Position(t - 12),
                        20 => Target::Primitive,
                        32 if params.is_empty() => Target::Unread,
                        t @ 32..=63 => {
                            // Bit 7: a mesh program's per-primitive output.
                            let o = params.iter().find(|o| o.param & 0x7F == t - 32).ok_or(
                                unsupported(at, "a parameter export the stage data does not list"),
                            )?;
                            let (usage, row) = (o.usage, o.row);
                            if o.param & 0x80 != 0 {
                                Target::PrimitiveParam { usage, row }
                            } else {
                                Target::Param { usage, row }
                            }
                        }
                        _ => return Err(unsupported(at, "an export target")),
                    };
                    let values = src.map(|s| match s {
                        Some(Operand::Vector { reg, .. }) => inst.read(Reg::V(reg)),
                        _ => None,
                    });
                    io.exports.push(Export {
                        site,
                        target,
                        values,
                        compressed: *compr,
                        done: *done,
                    });
                }
                _ => {}
            }
        }
    }
    Ok(io)
}

/// The pixel input VGPRs that `addr` enables, with their meaning.
fn ps_vgprs(addr: u32) -> Vec<(u8, Input)> {
    let mut out = Vec::new();
    let mut next = 0u8;
    for (bit, &(count, kind)) in PS_VGPRS.iter().enumerate() {
        if addr >> bit & 1 == 0 {
            continue;
        }
        for index in 0..count {
            let input = match kind {
                Some(kind) => Input::Barycentric { kind, index },
                None => ps_system(bit),
            };
            out.push((next + index, input));
        }
        next += count;
    }
    out
}

/// An interpolation before its attribute is known: `barycentric` none
/// when i or j is computed.
struct Pair {
    barycentric: Option<Barycentric>,
    first: Vec<Site>,
}

impl Pair {
    fn with(self, attribute: Attribute) -> Interp {
        match self.barycentric {
            Some(barycentric) => Interp::Interpolated {
                attribute,
                barycentric,
                first: self.first,
            },
            None => Interp::Weighted { attribute },
        }
    }
}

/// The source value of a plain VGPR-to-VGPR `v_mov_b32` copy (no modifiers,
/// SDWA, or DPP).
fn vgpr_copy(inst: &Inst) -> Option<Value> {
    if inst.name() != "v_mov_b32" {
        return None;
    }
    let Fields::Valu(v) = &inst.fields else {
        return None;
    };
    if v.neg != 0
        || v.abs != 0
        || v.neg_hi != 0
        || v.opsel != 0
        || v.opsel_hi != 0
        || v.clamp
        || v.omod != 0
        || v.sdwa.is_some()
        || v.dpp.is_some()
    {
        return None;
    }
    let [Operand::Vector { reg, .. }] = v.src.as_slice() else {
        return None;
    };
    inst.read(Reg::V(*reg))
}

/// Records interpolation `pair` at `site`, and its first halves too when
/// they evaluate on their own (computed barycentrics).
fn push_pair(io: &mut Io, site: Site, pair: Pair, attribute: Attribute) {
    if pair.barycentric.is_none() {
        for &f in &pair.first {
            if !io.interps.iter().any(|(s, _)| *s == f) {
                io.interps.push((f, Interp::Weighted { attribute }));
            }
        }
    }
    io.interps.push((site, pair.with(attribute)));
}

/// Finds the first halves of interpolations.
struct Halves<'a> {
    p: &'a Program,
    inputs: &'a [(Value, Input)],
    defs: &'a std::collections::HashMap<Value, Site>,
    phis: &'a std::collections::HashMap<Value, (usize, usize)>,
}

impl Halves<'_> {
    /// Copy tracing is limited to the entry EXEC mask. Equal SSA names for
    /// a loop's EXEC phi need not mean equal masks in different iterations.
    /// Keeping the entry mask also ensures inactive lanes cannot rejoin
    /// with the destination's old value after a partial copy.
    fn entry_exec(&self, inst: &Inst) -> bool {
        (0..1 + u8::from(self.p.wave64)).all(|half| {
            let reg = Reg::S(EXEC + half);
            self.p
                .inputs
                .iter()
                .find(|input| input.0 == reg)
                .is_some_and(|input| inst.read(reg) == Some(input.1))
        })
    }

    fn input_of(&self, v: Value) -> Option<Input> {
        self.inputs.iter().find(|i| i.0 == v).map(|i| i.1)
    }

    /// A VGPR value's barycentric kind and index, following plain `v_mov_b32`
    /// copies and phis whose incoming values agree.
    fn barycentric(&self, v: Option<Value>, copied: &mut bool) -> Option<(Barycentric, u8)> {
        let mut seen = Vec::new();
        let mut found = None;
        self.barycentric_from(v?, &mut seen, &mut found, copied)
            .then_some(found)
            .flatten()
    }

    fn barycentric_from(
        &self,
        v: Value,
        seen: &mut Vec<Value>,
        found: &mut Option<(Barycentric, u8)>,
        copied: &mut bool,
    ) -> bool {
        if seen.contains(&v) {
            return true;
        }
        if let Some(Input::Barycentric { kind, index }) = self.input_of(v) {
            return match *found {
                Some(prev) => prev == (kind, index),
                None => {
                    *found = Some((kind, index));
                    true
                }
            };
        }
        if let Some(&site) = self.defs.get(&v) {
            let inst = &self.p.blocks[site.0].insts[site.1];
            let Some(src) = vgpr_copy(inst) else {
                return false;
            };
            if !self.entry_exec(inst) {
                return false;
            }
            *copied = true;
            seen.push(v);
            return self.barycentric_from(src, seen, found, copied);
        }
        let Some(&(b, i)) = self.phis.get(&v) else {
            return false;
        };
        seen.push(v);
        self.p.blocks[b].phis[i]
            .args
            .iter()
            .all(|&a| self.barycentric_from(a, seen, found, copied))
    }

    /// The interpolation whose second half reads the first half's result
    /// `acc`, the barycentric `j`, and attribute `attr`: every first half
    /// must read the same attribute channel. With j and every first half's
    /// i loaded as one kind's, the pair interpolates with that kind;
    /// otherwise each half evaluates on its own (`Interp::Weighted`).
    fn pair(
        &self,
        attr: &Operand,
        acc: Option<Value>,
        j: Option<Value>,
        second: &Inst,
    ) -> Result<Pair, &'static str> {
        let mut first = Vec::new();
        let mut copied = false;
        self.sites(
            acc.ok_or("an interpolation without its first half")?,
            &mut Vec::new(),
            &mut first,
            &mut copied,
        )?;
        if first.is_empty() {
            return Err("an interpolation without its first half");
        }
        let mut kind = match self.barycentric(j, &mut copied) {
            Some((kind, 1)) => Some(kind),
            _ => None,
        };
        for &(b, i) in &first {
            let inst = &self.p.blocks[b].insts[i];
            let (a, i) = match &inst.fields {
                Fields::Vintrp {
                    src: Operand::Vector { reg, .. },
                    attr,
                    ..
                } if inst.name() == "v_interp_p1_f32" => (*attr, inst.read(Reg::V(*reg))),
                Fields::Valu(v)
                    if inst.name() == "v_interp_p1ll_f16" || inst.name() == "v_interp_p1lv_f16" =>
                {
                    match v.src.get(1) {
                        Some(&Operand::Vector { reg, .. }) => (v.src[0], inst.read(Reg::V(reg))),
                        _ => return Err("an interpolation without its first half"),
                    }
                }
                _ => return Err("an interpolation without its first half"),
            };
            if a != *attr {
                return Err("interpolation halves of different attributes");
            }
            if kind.is_some_and(|k| self.barycentric(i, &mut copied) != Some((k, 0))) {
                kind = None;
            }
        }
        // A full-mask copy can still carry a partially computed p1 result.
        // Require both interpolation halves to use the same entry mask too.
        if copied
            && (!self.entry_exec(second)
                || first
                    .iter()
                    .any(|&(b, i)| !self.entry_exec(&self.p.blocks[b].insts[i])))
        {
            return Err("an interpolation copy with a changed execution mask");
        }
        Ok(Pair {
            barycentric: kind,
            first,
        })
    }

    /// The instructions `v` comes from, through plain copies and phis (`seen`:
    /// values on the way; a loop back to one adds nothing).
    fn sites(
        &self,
        v: Value,
        seen: &mut Vec<Value>,
        out: &mut Vec<Site>,
        copied: &mut bool,
    ) -> Result<(), &'static str> {
        if seen.contains(&v) {
            return Ok(());
        }
        if let Some(&site) = self.defs.get(&v) {
            let inst = &self.p.blocks[site.0].insts[site.1];
            if let Some(src) = vgpr_copy(inst) {
                if !self.entry_exec(inst) {
                    return Err("an interpolation copy with a changed execution mask");
                }
                *copied = true;
                seen.push(v);
                return self.sites(src, seen, out, copied);
            }
            if !out.contains(&site) {
                out.push(site);
            }
            return Ok(());
        }
        let &(b, i) = self
            .phis
            .get(&v)
            .ok_or("an interpolation without its first half")?;
        seen.push(v);
        for &a in &self.p.blocks[b].phis[i].args {
            self.sites(a, seen, out, copied)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::ExportPatch;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn program(words: &[u32]) -> Program {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::ir::build(&code, false, 256).unwrap()
    }

    fn pixel(inputs: Vec<PsInput>, input_addr: u32) -> NativeProgram {
        NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs,
                input_addr,
                db_shader_control: 0,
                patches: Vec::<ExportPatch>::new(),
            },
            code: vec![],
        }
    }

    const TEXCOORD: PsInput = PsInput {
        usage: Usage::Generic,
        row: 1,
        mask: 3,
        attr: 0,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 1,
    };

    #[test]
    fn interpolation_pairs_and_color_exports() {
        // Blit's pixel shader: TEXCOORD.xy from the perspective-center i/j
        // in v0/v1, then exported unconverted to MRT0.
        let p = program(&[
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0302_0102, // exp mrt0, v2, v1, v2, v3 done vm
            S_ENDPGM,
        ]);
        let io = raise(&p, &pixel(vec![TEXCOORD], 2)).unwrap();
        let input = |reg| p.inputs.iter().find(|i| i.0 == Reg::V(reg)).unwrap().1;
        let center = |index| Input::Barycentric {
            kind: Barycentric::PerspectiveCenter,
            index,
        };
        assert!(io.inputs.contains(&(input(0), center(0))));
        assert!(io.inputs.contains(&(input(1), center(1))));
        // v2 and v3 are read before written but hold nothing.
        assert!(!io.inputs.iter().any(|i| i.0 == input(3)));
        assert_eq!(
            io.interps,
            [(
                (0, 1),
                Interp::Interpolated {
                    attribute: Attribute {
                        usage: Usage::Generic,
                        row: 1,
                        channel: 0,
                        high: false
                    },
                    barycentric: Barycentric::PerspectiveCenter,
                    first: vec![(0, 0)],
                }
            )]
        );
        let [e] = &io.exports[..] else {
            panic!("{:?}", io.exports)
        };
        assert_eq!(
            (e.target, e.done, e.compressed),
            (Target::Color(0), true, false)
        );
        assert_eq!(e.values[0], p.blocks[0].insts[1].write(Reg::V(2)));
    }

    #[test]
    fn lane_value_first_half_pairs_with_f16_second() {
        // v_interp_p1lv_f16 v4, v0, attr0.x, v3 — the I coordinate in v0
        // (the perspective-center i), the lane's P0 in v3 — then
        // v_interp_p2_f16 v5, v1, attr0.x, v4 — the j in v1, the first
        // half's result in v4.
        let p = program(&[
            0xD743_0004,
            0x040E_0000, // v_interp_p1lv_f16 v4, v0, attr0.x, v3
            0xD75A_0005,
            0x0412_0200, // v_interp_p2_f16 v5, v1, attr0.x, v4
            S_ENDPGM,
        ]);
        let io = raise(&p, &pixel(vec![TEXCOORD], 2)).unwrap();
        assert_eq!(
            io.interps,
            [(
                (0, 1),
                Interp::Interpolated {
                    attribute: Attribute {
                        usage: Usage::Generic,
                        row: 1,
                        channel: 0,
                        high: false
                    },
                    barycentric: Barycentric::PerspectiveCenter,
                    first: vec![(0, 0)],
                }
            )]
        );
    }

    #[test]
    fn unpaired_or_unknown_interpolation_fails() {
        // p2 alone: its old dst is no p1's result.
        let lone = program(&[0xC809_0001, S_ENDPGM]); // v_interp_p2_f32 v2, v1, attr0.x
        assert!(raise(&lone, &pixel(vec![TEXCOORD], 2)).is_err());
        // attr1 is not in the stage data.
        let other = program(&[0xC808_0400, S_ENDPGM]); // v_interp_p1_f32 v2, v0, attr1.x
        assert!(raise(&other, &pixel(vec![TEXCOORD], 2)).is_err());
    }

    #[test]
    fn interpolation_follows_vgpr_copies() {
        let p = program(&[
            0x7E08_0300, // v_mov_b32 v4, v0 (i copy)
            0x7E0A_0301, // v_mov_b32 v5, v1 (j copy)
            0xC808_0004, // v_interp_p1_f32 v2, v4, attr0.x
            0x7E06_0302, // v_mov_b32 v3, v2 (acc copy)
            0xC80D_0005, // v_interp_p2_f32 v3, v5, attr0.x
            0xF800_180F,
            0x0303_0303, // exp mrt0, v3, v3, v3, v3 done vm
            S_ENDPGM,
        ]);
        let io = raise(&p, &pixel(vec![TEXCOORD], 2)).unwrap();
        assert_eq!(
            io.interps,
            [(
                (0, 4),
                Interp::Interpolated {
                    attribute: Attribute {
                        usage: Usage::Generic,
                        row: 1,
                        channel: 0,
                        high: false,
                    },
                    barycentric: Barycentric::PerspectiveCenter,
                    first: vec![(0, 2)],
                }
            )]
        );
    }

    #[test]
    fn computed_barycentrics_evaluate_each_half() {
        // i scaled before the first half: both halves stand on their own.
        let p = program(&[
            0x1006_00F0, // v_mul_f32 v3, 0.5, v0
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            S_ENDPGM,
        ]);
        let io = raise(&p, &pixel(vec![TEXCOORD], 2)).unwrap();
        let attribute = Attribute {
            usage: Usage::Generic,
            row: 1,
            channel: 0,
            high: false,
        };
        let mut interps = io.interps.clone();
        interps.sort_by_key(|i| i.0);
        assert_eq!(
            interps,
            [
                ((0, 1), Interp::Weighted { attribute }),
                ((0, 2), Interp::Weighted { attribute }),
            ]
        );
    }

    #[test]
    fn interpolation_follows_agreeing_phis_and_rejects_mismatched_or_modified_copies() {
        // Both arms copy i/j, compute p1 into v2, join, copy v2 -> v3, then finish with p2.
        let agreeing = program(&[
            0xBF06_0100, // s_cmp_eq_u32 s0, s1
            0xBF85_0004, // s_cbranch_scc1 -> 24
            0x7E08_0300, // v_mov_b32 v4, v0
            0x7E0A_0301, // v_mov_b32 v5, v1
            0xC808_0004, // v_interp_p1_f32 v2, v4, attr0.x
            0xBF82_0003, // s_branch -> 36
            0x7E08_0300, // 24: v_mov_b32 v4, v0
            0x7E0A_0301, // v_mov_b32 v5, v1
            0xC808_0004, // v_interp_p1_f32 v2, v4, attr0.x
            0x7E06_0302, // 36: v_mov_b32 v3, v2
            0xC80D_0005, // v_interp_p2_f32 v3, v5, attr0.x
            0xF800_180F,
            0x0303_0303, // exp mrt0, v3, v3, v3, v3 done vm
            S_ENDPGM,
        ]);
        let io = raise(&agreeing, &pixel(vec![TEXCOORD], 2)).unwrap();
        assert_eq!(io.interps.len(), 1);
        let (
            site,
            Interp::Interpolated {
                barycentric, first, ..
            },
        ) = &io.interps[0]
        else {
            panic!("{:?}", io.interps);
        };
        assert_eq!(*site, (3, 1));
        assert_eq!(*barycentric, Barycentric::PerspectiveCenter);
        assert_eq!(first, &[(1, 2), (2, 2)]);

        // One arm passes i and the other passes j through a phi into p1.
        let mismatched_phi = program(&[
            0xBF06_0100, // s_cmp_eq_u32 s0, s1
            0xBF85_0002, // s_cbranch_scc1 -> 16
            0x7E08_0300, // v_mov_b32 v4, v0
            0xBF82_0001, // s_branch -> 20
            0x7E08_0301, // 16: v_mov_b32 v4, v1
            0xC808_0004, // v_interp_p1_f32 v2, v4, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            S_ENDPGM,
        ]);
        // That is a computed i: the halves evaluate on their own
        // (`Interp::Weighted`) rather than as one barycentric pair.
        let io = raise(&mismatched_phi, &pixel(vec![TEXCOORD], 2)).unwrap();
        assert!(!io.interps.is_empty());
        assert!(
            io.interps
                .iter()
                .all(|(_, i)| matches!(i, Interp::Weighted { .. })),
            "{:?}",
            io.interps
        );

        // A modified v_mov_b32_e64 (neg on src0) is not treated as a plain barycentric copy.
        let neg_bary = program(&[
            0xD581_0004,
            0x2000_0100, // v_mov_b32_e64 v4, -v0
            0xC808_0004, // v_interp_p1_f32 v2, v4, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            S_ENDPGM,
        ]);
        let mov = &neg_bary.blocks[0].insts[0];
        assert_eq!(mov.name(), "v_mov_b32");
        assert!(matches!(&mov.fields, Fields::Valu(v) if v.neg != 0));
        let io = raise(&neg_bary, &pixel(vec![TEXCOORD], 2)).unwrap();
        assert!(!io.interps.is_empty());
        assert!(
            io.interps
                .iter()
                .all(|(_, i)| matches!(i, Interp::Weighted { .. })),
            "a negated copy is a computed i: {:?}",
            io.interps
        );

        // A modified v_mov_b32_e64 (neg on src0) breaks the accumulator chain.
        let neg_acc = program(&[
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xD581_0003,
            0x2000_0102, // v_mov_b32_e64 v3, -v2
            0xC80D_0001, // v_interp_p2_f32 v3, v1, attr0.x
            S_ENDPGM,
        ]);
        let acc_mov = &neg_acc.blocks[0].insts[1];
        assert_eq!(acc_mov.name(), "v_mov_b32");
        assert!(matches!(&acc_mov.fields, Fields::Valu(v) if v.neg != 0));
        assert!(matches!(
            raise(&neg_acc, &pixel(vec![TEXCOORD], 2)),
            Err(ShaderError::Unsupported {
                what: "an interpolation without its first half",
                ..
            })
        ));
    }

    #[test]
    fn interpolation_copies_require_the_entry_execution_mask() {
        for fp16 in [false, true] {
            let first = if fp16 {
                vec![0xD742_0002, 0x104 << 9] // v_interp_p1ll_f16 v2, attr0.x, v4
            } else {
                vec![0xC808_0004] // v_interp_p1_f32 v2, v4, attr0.x
            };
            let second = if fp16 {
                vec![0xD75A_0003, (0x105 << 9) | (0x103 << 18)] // p2 v3, attr0.x, v5, v3
            } else {
                vec![0xC80D_0005] // v_interp_p2_f32 v3, v5, attr0.x
            };
            let instructions = [
                vec![0x7E08_0300], // v_mov_b32 v4, v0 (i)
                vec![0x7E0A_0301], // v_mov_b32 v5, v1 (j)
                first,
                vec![0x7E06_0302], // v_mov_b32 v3, v2 (acc)
                second,
                vec![S_ENDPGM],
            ];
            for wave64 in [false, true] {
                let mut native = pixel(vec![PsInput { fp16, ..TEXCOORD }], 2);
                native.wave64 = wave64;
                let build = |words: &[u32]| {
                    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                    crate::ir::build(&code, wave64, 256).unwrap()
                };
                let words: Vec<u32> = instructions.iter().flatten().copied().collect();
                assert_eq!(raise(&build(&words), &native).unwrap().interps.len(), 1);

                // Independently mask i, j, p1, acc, or p2. Inactive lanes
                // retain old values; restoring EXEC must not hide that.
                for half in 0..=u32::from(wave64) {
                    let narrow = 0xBEFE_0380 | (half << 16); // s_mov_b32 exec_{lo,hi}, 0
                    let restore = 0xBEFE_03C1 | (half << 16); // s_mov_b32 exec_{lo,hi}, -1
                    for masked in 0..5 {
                        let mut changed = Vec::new();
                        for (i, words) in instructions.iter().enumerate() {
                            if i == masked {
                                changed.push(narrow);
                            }
                            changed.extend(words);
                            if i == masked {
                                changed.push(restore);
                            }
                        }
                        assert!(
                            matches!(
                                raise(&build(&changed), &native),
                                Err(ShaderError::Unsupported { .. })
                            ),
                            "fp16={fp16}, wave64={wave64}, half={half}, masked={masked}"
                        );
                    }
                }

                // Even matching mask SSA names are not enough if they are
                // not the entry mask (a loop phi can vary per iteration).
                let mut changed = vec![0xBEFE_0381]; // s_mov_b32 exec_lo, 1
                changed.extend(&words);
                assert!(matches!(
                    raise(&build(&changed), &native),
                    Err(ShaderError::Unsupported { .. })
                ));

                if !wave64 {
                    // EXEC_HI does not control any lane in a wave32 program.
                    let mut changed = vec![0xBEFF_0380];
                    changed.extend(&words);
                    assert_eq!(raise(&build(&changed), &native).unwrap().interps.len(), 1);
                }
            }
        }
    }

    #[test]
    fn mesh_inputs_and_primitive_export() {
        let mesh = |words: &[u32]| {
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let p = crate::ir::build(&code, true, 256).unwrap();
            let native = NativeProgram {
                stage: HwStage::Ms,
                wave64: true,
                vgprs: 256,
                layout: vec![],
                used: vec![],
                io: StageIo::Ms(crate::native::MeshIo {
                    outputs: vec![],
                    clip_cntl: 0,
                    threads: 128,
                    max_vertices: 128,
                    max_primitives: 128,
                    topology: 2,
                    lds_bytes: 0,
                }),
                code,
            };
            raise(&p, &native)
        };
        let io = mesh(&[
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0xF800_0941,
            0x0000_0009, // exp prim, v9, off, off, off done
            S_ENDPGM,
        ])
        .unwrap();
        assert!(io.inputs.iter().any(|i| i.1 == Input::MeshGroup));
        assert_eq!(io.exports[0].target, Target::Primitive);
        // The group input is uniform only as the first lane reads it.
        assert!(mesh(&[0x7E02_0305, S_ENDPGM]).is_err()); // v_mov_b32 v1, v5
    }

    #[test]
    fn pixel_vgprs_follow_the_input_address() {
        // Linear center (bit 5), then position x and y (bits 8, 9).
        let v = ps_vgprs(1 << 5 | 1 << 8 | 1 << 9);
        assert_eq!(
            v,
            [
                (
                    0,
                    Input::Barycentric {
                        kind: Barycentric::LinearCenter,
                        index: 0
                    }
                ),
                (
                    1,
                    Input::Barycentric {
                        kind: Barycentric::LinearCenter,
                        index: 1
                    }
                ),
                (2, Input::Position(0)),
                (3, Input::Position(1)),
            ]
        );
    }
}
