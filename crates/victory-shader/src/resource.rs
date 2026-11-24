//! Resource tracking (decision 0003): where each descriptor a program uses
//! comes from, in terms of the root signature.
//!
//! Root parameters arrive in user SGPRs ([`RootParam`]): a descriptor table
//! as the 32-bit heap address of its range (the code sets the high half to
//! 0), a root descriptor as a 64-bit buffer address, root constants as
//! themselves. A root signature with dynamic resources (SM 6.6) adds the
//! 32-bit base addresses of both heaps; a descriptor is then loaded from
//! one of them at an index the code computes (from a root constant, say). The code loads table descriptors from heap memory with
//! `s_load_dword*`, and builds a buffer descriptor around a root
//! descriptor's address itself. Each descriptor operand and each scalar
//! memory address is traced through the SSA values back to those
//! parameters, by data flow: copies, phis that agree, and additions.

use std::cell::OnceCell;
use std::collections::HashMap;

use crate::ShaderError;
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::native::{HwStage, RootParam, RootType};
use crate::structure::{Atom, Bool, LaneExit, Structured};

/// An instruction: block and index in [`crate::ir::Block::insts`].
pub type Site = (usize, usize);

/// A byte address: a root parameter's, plus a constant and at most one
/// value the program computes (an index times a stride, say).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Address {
    /// Index into the layout, or [`CODE`].
    pub param: usize,
    /// Added modulo 2^32 (with carry into the high half of a root
    /// descriptor's address).
    pub offset: u32,
    pub dynamic: Option<Value>,
}

/// [`Address::param`] of the program's own code: a constant table the
/// compiler appends after the code and reads through `s_getpc_b64`; the
/// offset is then the byte offset in the code.
pub const CODE: usize = usize::MAX;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// Dwords `first..` of what the scalar load at `load` reads (see
    /// [`Resources::scalar`] for its address).
    Loaded { load: Site, first: u8 },
    /// A buffer descriptor built around an address: `bits` OR-ed into
    /// dword 1 (stride, swizzle), dwords 2 and 3 as given.
    Built {
        address: Address,
        bits: u32,
        words: [u32; 2],
    },
    /// Fixed in the code (an immediate sampler, say).
    Constant(Vec<u32>),
    /// Chosen by control flow: one source per predecessor of `block`, in
    /// [`crate::ir::Block::preds`] order.
    Phi { block: usize, args: Vec<Source> },
    /// Chosen by SCC (`scc`, the value an `s_cselect` or `s_cmov` of each
    /// dword reads): the source when it is set, then when it is clear.
    Select { scc: Value, args: Vec<Source> },
    /// Dwords `first..` of what the vector load at `load` (`global_load_*`)
    /// reads, each lane at its own address (see [`Resources::scalar`]); the
    /// dwords past that load come from loads of the bytes that follow. A
    /// descriptor that differs between lanes.
    Fetched { load: Site, first: u8 },
    /// Picked per lane by `v_cndmask_b32` reading the lane mask whose low
    /// dword is `mask`: the source where the lane's bit is set, then where it
    /// is clear.
    Lanes { mask: Value, args: Vec<Source> },
    /// Per-lane dwords chosen by control flow (VGPRs joined where lanes
    /// that took different paths meet): one source per predecessor of
    /// `block`, in [`crate::ir::Block::preds`] order, `None` on an edge no
    /// lane that accesses the descriptor takes. A descriptor that differs
    /// between lanes.
    LanePhi {
        block: usize,
        args: Vec<Option<Source>>,
    },
}

impl Source {
    /// The sources this one picks from (itself unless a phi).
    pub fn leaves(&self) -> Vec<&Source> {
        match self {
            Source::Phi { args, .. } | Source::Select { args, .. } | Source::Lanes { args, .. } => {
                args.iter().flat_map(Source::leaves).collect()
            }
            Source::LanePhi { args, .. } => {
                args.iter().flatten().flat_map(Source::leaves).collect()
            }
            s => vec![s],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Image, buffer or `s_buffer_load` resource.
    Resource,
    Sampler,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Descriptor {
    pub site: Site,
    pub role: Role,
    pub source: Source,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Resources {
    /// Every scalar memory access by address (`s_load_*`, `s_store_*`), and
    /// every global memory access (`global_*`, the dynamic term per lane).
    pub scalar: Vec<(Site, Address)>,
    pub descriptors: Vec<Descriptor>,
    /// User SGPRs holding root parameters: value on entry, layout index,
    /// dword within the parameter.
    pub params: Vec<(Value, usize, u32)>,
    pub waterfall: Waterfall,
}

/// Waterfall loops over descriptors that differ between lanes: each pass
/// reads the first active lane's descriptor with `v_readfirstlane_b32`,
/// compares it with every lane's own (`v_cmp_eq`), and runs the access with
/// the lanes that match. Their accesses take each lane's own descriptor
/// ([`Source::Fetched`], [`Source::Lanes`]), so the first lane's dwords are
/// each lane's own too: the compares hold in every lane and one pass serves
/// all lanes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Waterfall {
    /// The compares, true in every lane.
    pub compares: Vec<Site>,
    /// The `v_readfirstlane_b32` results they compare, each the lane's own
    /// source dword.
    pub reads: Vec<Value>,
    /// Descriptor loads at a table offset among `reads` (the console
    /// compiler's form): each lane loads its own entry.
    pub loads: Vec<Site>,
}

/// Traces every descriptor and scalar address of `p`, which runs as
/// `stage`, to the root parameters in `layout`. Fails on the first one it
/// cannot trace. `lanes`, `p` as one lane runs it, tells which paths the
/// lanes that access a per-lane descriptor took and which writes they ran;
/// without it, per-lane descriptors chosen by control flow are refused and
/// per-lane writes are taken to run in every lane.
pub fn track(
    p: &Program,
    layout: &[RootParam],
    stage: HwStage,
    lanes: Option<&Structured>,
) -> Result<Resources, ShaderError> {
    let t = Tracer::new(p, layout, stage, lanes)?;
    let mut out = Resources {
        params: p
            .inputs
            .iter()
            .filter_map(|&(reg, v)| {
                let (param, dword) = t.param_of(reg)?;
                Some((v, param, dword))
            })
            .collect(),
        ..Resources::default()
    };
    // Group accesses in the same loop so a shared mask can select several
    // descriptors together.
    let mut first_lane_at = std::collections::BTreeMap::<usize, (Vec<usize>, Vec<Value>)>::new();
    // The console compiler's waterfall reads the first lane's table offset
    // and loads that entry: accesses, offset reads and loads by block.
    let mut offset_at =
        std::collections::BTreeMap::<usize, (Vec<usize>, Vec<Value>, Vec<Site>)>::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let site = (b, i);
            let fail = |what| ShaderError::Unsupported {
                offset: inst.machine.offset,
                what,
            };
            let mut first_lane = Vec::new();
            let mut offset = Vec::new();
            let mut descriptor = |op: &Operand, role| -> Result<(), ShaderError> {
                let source = t.descriptor(inst, op, site).map_err(fail)?;
                first_lane.extend(t.first_lane_words(inst, op).unwrap_or_default());
                if let Source::Loaded { load, .. } = source
                    && let Some((_, a)) = out.scalar.iter().find(|s| s.0 == load)
                    && let Some(Term::Value(v)) = a.dynamic.map(|d| t.origin(d))
                    && t.first_lane_source(v).is_some()
                {
                    offset.push((v, load));
                }
                out.descriptors.push(Descriptor { site, role, source });
                Ok(())
            };
            match &inst.fields {
                Fields::Image { rsrc, sampler, .. } => {
                    descriptor(rsrc, Role::Resource)?;
                    if let Some(s) = sampler {
                        descriptor(s, Role::Sampler)?;
                    }
                }
                Fields::Buffer { rsrc, .. } => descriptor(rsrc, Role::Resource)?,
                Fields::Smem { base, .. } if inst.name().starts_with("s_buffer_") => {
                    descriptor(base, Role::Resource)?;
                }
                Fields::Smem {
                    base,
                    offset,
                    soffset,
                    ..
                } if inst.name().contains("_load_") || inst.name().contains("_store_") => {
                    let address = t.smem_address(inst, base, *offset, soffset).map_err(fail)?;
                    out.scalar.push((site, address));
                }
                Fields::Flat {
                    addr,
                    saddr,
                    offset,
                    ..
                } if inst.name().starts_with("global_") => {
                    let address = t
                        .flat_address(inst, addr.as_ref(), saddr.as_ref(), *offset)
                        .map_err(fail)?;
                    out.scalar.push((site, address));
                }
                _ => {}
            }
            if !first_lane.is_empty() {
                let (accesses, reads) = first_lane_at.entry(b).or_default();
                accesses.push(i);
                reads.extend(first_lane);
            }
            if !offset.is_empty() {
                let (accesses, reads, loads) = offset_at.entry(b).or_default();
                accesses.push(i);
                for (v, load) in offset {
                    reads.push(v);
                    loads.push(load);
                }
            }
        }
    }
    // The proof needs the per-lane form. Without it nothing is emitted, and
    // its failure is the one to report, so the reads stay unproven here.
    if t.lanes.is_none() {
        return Ok(out);
    }
    for (block, (accesses, mut reads)) in first_lane_at {
        reads.sort_unstable();
        reads.dedup();
        let proof = t
            .waterfall(block, &accesses, &reads)
            .map_err(|what| ShaderError::Unsupported {
                offset: t.inst((block, accesses[0])).machine.offset,
                what,
            })?;
        out.waterfall.compares.extend(proof.compares);
        out.waterfall.reads.extend(proof.reads);
    }
    // Unproven, an offset read stays the first lane's (one uniform entry),
    // which is what the hardware reads outside a waterfall loop.
    for (block, (accesses, mut reads, loads)) in offset_at {
        reads.sort_unstable();
        reads.dedup();
        if let Ok(proof) = t.waterfall(block, &accesses, &reads) {
            out.waterfall.compares.extend(proof.compares);
            out.waterfall.reads.extend(proof.reads);
            out.waterfall.loads.extend(loads);
        }
    }
    Ok(out)
}

/// A dword after copy propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Term {
    Const(u32),
    Value(Value),
}

/// A 32-bit sum: optional root parameter (as its first dword), constant,
/// optional computed term.
struct Sum {
    base: Option<Value>,
    offset: u32,
    dynamic: Option<Value>,
}

#[derive(Clone, Copy)]
enum UseSite {
    Phi {
        block: usize,
        pred_idx: usize,
        phi_idx: usize,
    },
    Inst {
        block: usize,
        inst_idx: usize,
    },
}

struct Tracer<'a> {
    p: &'a Program,
    layout: &'a [RootParam],
    stage: HwStage,
    /// Defining instruction per value (phis and inputs have none).
    defs: Vec<Option<Site>>,
    /// Block and index of each phi value.
    phis: Vec<Option<(usize, usize)>>,
    /// Register per input value.
    inputs: Vec<Option<Reg>>,
    origin: Vec<Term>,
    lanes: Option<&'a Structured>,
    /// Reverse-postorder index per block.
    rpo: Vec<usize>,
    /// SSA uses per value, built lazily when a partial per-lane join checks deadness.
    uses: OnceCell<Vec<Vec<UseSite>>>,
}

/// What lanes are known to satisfy: each [`Bool`] where its flag is set,
/// its negation where not.
type Known = Vec<(Bool, bool)>;

const MAX_ROUNDS: usize = 64;
const MAX_JOINS: usize = 8;
const MAX_TAINT_STATES: usize = 512;

impl<'a> Tracer<'a> {
    fn new(
        p: &'a Program,
        layout: &'a [RootParam],
        stage: HwStage,
        lanes: Option<&'a Structured>,
    ) -> Result<Self, ShaderError> {
        let all = || {
            p.inputs
                .iter()
                .map(|i| i.1)
                .chain(p.blocks.iter().flat_map(|b| {
                    b.phis
                        .iter()
                        .map(|phi| phi.value)
                        .chain(b.insts.iter().flat_map(|i| i.writes.iter().map(|w| w.1)))
                }))
        };
        let n = all().map(|v| v.0 as usize + 1).max().unwrap_or(0);
        let mut defs = vec![None; n];
        let mut phis = vec![None; n];
        let mut inputs = vec![None; n];
        for &(reg, v) in &p.inputs {
            inputs[v.0 as usize] = Some(reg);
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phis[phi.value.0 as usize] = Some((b, i));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, v) in &inst.writes {
                    defs[v.0 as usize] = Some((b, i));
                }
            }
        }
        let mut rpo = vec![usize::MAX; p.blocks.len()];
        for (n, b) in crate::ir::reverse_postorder(&p.blocks)
            .into_iter()
            .enumerate()
        {
            rpo[b] = n;
        }
        let mut t = Tracer {
            p,
            layout,
            stage,
            defs,
            phis,
            inputs,
            origin: Vec::new(),
            lanes,
            rpo,
            uses: OnceCell::new(),
        };
        t.origin = t.propagate_copies(n)?;
        Ok(t)
    }

    /// Optimistic copy propagation: a move's result is its source's
    /// origin, a phi's the one origin its arguments agree on.
    fn propagate_copies(&self, n: usize) -> Result<Vec<Term>, ShaderError> {
        let mut o: Vec<Option<Term>> = vec![None; n];
        for &(_, v) in &self.p.inputs {
            o[v.0 as usize] = Some(Term::Value(v));
        }
        let order = crate::ir::reverse_postorder(&self.p.blocks);
        for _ in 0..MAX_ROUNDS {
            let mut changed = false;
            for &b in &order {
                let block = &self.p.blocks[b];
                for phi in &block.phis {
                    let own = Term::Value(phi.value);
                    let mut args = phi
                        .args
                        .iter()
                        .filter_map(|a| o[a.0 as usize])
                        .filter(|&t| t != own);
                    let new = args
                        .next()
                        .map(|first| if args.all(|t| t == first) { first } else { own });
                    if new.is_some() && new != o[phi.value.0 as usize] {
                        o[phi.value.0 as usize] = new;
                        changed = true;
                    }
                }
                for inst in &block.insts {
                    for (k, &(_, v)) in inst.writes.iter().enumerate() {
                        let new = match copy_source(inst, k).or_else(|| self.restored(inst)) {
                            Some(Moved::Const(c)) => Some(Term::Const(c)),
                            Some(Moved::Value(s)) => o[s.0 as usize],
                            None => Some(Term::Value(v)),
                        };
                        if new.is_some() && new != o[v.0 as usize] {
                            o[v.0 as usize] = new;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                return Ok(o
                    .into_iter()
                    .enumerate()
                    .map(|(v, t)| t.unwrap_or(Term::Value(Value(v as u32))))
                    .collect());
            }
        }
        Err(ShaderError::Unsupported {
            offset: 0,
            what: "copies that do not settle",
        })
    }

    fn inst(&self, (b, i): Site) -> &'a Inst {
        &self.p.blocks[b].insts[i]
    }

    fn def(&self, v: Value) -> Option<&'a Inst> {
        self.defs[v.0 as usize].map(|s| self.inst(s))
    }

    fn origin(&self, v: Value) -> Term {
        self.origin[v.0 as usize]
    }

    /// The origin of a one-dword operand as `inst` reads it.
    fn term(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Some(Term::Const(0)),
            Operand::Scalar { code, .. } => Some(self.origin(inst.read(Reg::S(code))?)),
            Operand::Const(c) => inline(c).map(Term::Const),
            Operand::Literal(x) => Some(Term::Const(x)),
            _ => None,
        }
    }

    /// Layout index and dword of the root parameter `reg` holds on entry.
    fn param_of(&self, reg: Reg) -> Option<(usize, u32)> {
        let Reg::S(code) = reg else { return None };
        self.layout.iter().enumerate().find_map(|(i, p)| {
            let first = self.stage.user_sgpr(p.user_data);
            let dword = u32::from(code).checked_sub(first.into())?;
            (dword < p.sgprs).then_some((i, dword))
        })
    }

    /// The SOP2 operands of `v` if an `s_add` with one of `names` wrote it.
    fn sop2(&self, v: Value, names: &[&str]) -> Option<(&'a Inst, [Term; 2])> {
        let inst = self.def(v)?;
        if !names.contains(&inst.name()) || inst.writes.first()?.1 != v {
            return None;
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        Some((
            inst,
            [
                self.term(inst, src.first()?)?,
                self.term(inst, src.get(1)?)?,
            ],
        ))
    }

    fn sum(&self, t: Term) -> Sum {
        let v = match t {
            Term::Const(c) => {
                return Sum {
                    base: None,
                    offset: c,
                    dynamic: None,
                };
            }
            Term::Value(v) => v,
        };
        let dynamic = Sum {
            base: None,
            offset: 0,
            dynamic: Some(v),
        };
        if let Some(reg) = self.inputs[v.0 as usize] {
            return match self.param_of(reg) {
                Some((_, 0)) => Sum {
                    base: Some(v),
                    offset: 0,
                    dynamic: None,
                },
                _ => dynamic,
            };
        }
        if self.getpc(v).is_some() {
            return Sum {
                base: Some(v),
                offset: 0,
                dynamic: None,
            };
        }
        let Some([a, b]) = self
            .sop2(v, &["s_add_u32", "s_add_i32"])
            .map(|(_, t)| t)
            .or_else(|| self.vadd(v))
        else {
            return dynamic;
        };
        let (x, y) = (self.sum(a), self.sum(b));
        if x.base.is_some() && y.base.is_some() || x.dynamic.is_some() && y.dynamic.is_some() {
            return dynamic;
        }
        Sum {
            base: x.base.or(y.base),
            offset: x.offset.wrapping_add(y.offset),
            dynamic: x.dynamic.or(y.dynamic),
        }
    }

    /// The `s_getpc_b64` writing `lo` as its low dword.
    fn getpc(&self, lo: Value) -> Option<&'a Inst> {
        let inst = self.def(lo)?;
        (inst.name() == "s_getpc_b64" && inst.writes.first().is_some_and(|w| w.1 == lo))
            .then_some(inst)
    }

    /// The operands of a VALU addition (`v_add_co_u32`, `v_add_nc_u32`)
    /// writing `v`, through VGPR copies of scalars.
    fn vadd(&self, v: Value) -> Option<[Term; 2]> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if !matches!(inst.name(), "v_add_co_u32" | "v_add_nc_u32")
            || f.dst.is_none_or(|d| !matches!(d, Operand::Vector { .. }))
            || inst.write(Reg::V(match f.dst? {
                Operand::Vector { reg, .. } => reg,
                _ => return None,
            })) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        Some([
            self.vterm(inst, f.src.first()?)?,
            self.vterm(inst, f.src.get(1)?)?,
        ])
    }

    /// The origin of a VALU operand.
    fn vterm(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Vector { reg, .. } => Some(self.vorigin(inst.read(Reg::V(reg))?)),
            op => self.term(inst, &op),
        }
    }

    /// A VGPR value's origin through `v_mov_b32` copies.
    fn vorigin(&self, v: Value) -> Term {
        if let Some(inst) = self.def(v)
            && inst.name() == "v_mov_b32"
            && let Fields::Valu(f) = &inst.fields
            && f.abs == 0
            && f.neg == 0
            && !f.clamp
            && f.omod == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
            && let Some(t) = f.src.first().and_then(|s| self.vterm(inst, s))
        {
            return t;
        }
        self.origin(v)
    }

    /// A global memory address: an SGPR pair plus a VGPR offset, or a VGPR
    /// pair.
    fn flat_address(
        &self,
        inst: &Inst,
        addr: Option<&Operand>,
        saddr: Option<&Operand>,
        offset: i16,
    ) -> Result<Address, &'static str> {
        let Some(&Operand::Vector { reg, .. }) = addr else {
            return Err("a global address not in VGPRs");
        };
        let off = offset as i32 as u32;
        match saddr {
            Some(&Operand::Scalar { code, .. }) if code != NULL => {
                let word = |k| Some(self.origin(inst.read(Reg::S(code + k))?));
                let (Some(lo), Some(hi)) = (word(0), word(1)) else {
                    return Err("an address not in SGPRs");
                };
                let v = inst.read(Reg::V(reg)).ok_or("a global address")?;
                let mut extra = self.sum(self.vorigin(v));
                if extra.base.is_some() {
                    return Err("an offset from a root parameter");
                }
                extra.offset = extra.offset.wrapping_add(off);
                self.address(lo, hi, extra)
            }
            _ => {
                let word = |k| inst.read(Reg::V(reg + k)).map(|v| self.vorigin(v));
                let (Some(lo), Some(hi)) = (word(0), word(1)) else {
                    return Err("a global address");
                };
                self.address(
                    lo,
                    hi,
                    Sum {
                        base: None,
                        offset: off,
                        dynamic: None,
                    },
                )
            }
        }
    }

    /// The address `lo` and `hi` hold, plus `extra` (an instruction's
    /// immediate and offset register).
    fn address(&self, lo: Term, hi: Term, extra: Sum) -> Result<Address, &'static str> {
        let s = self.sum(lo);
        if s.dynamic.is_some() && extra.dynamic.is_some() {
            return Err("an address with two computed terms");
        }
        let base = s.base.ok_or_else(|| self.why_no_base(s.dynamic))?;
        if let Some(pc) = self.getpc(base) {
            // A table in the code: the base is the next instruction's offset.
            let high = pc.writes.get(1).map(|w| w.1);
            if !self.high_half(hi, high) && !high.is_some_and(|h| self.carried(hi, h)) {
                return Err("an address with an untracked high half");
            }
            return Ok(Address {
                param: CODE,
                offset: (pc.machine.offset as u32 + 4)
                    .wrapping_add(s.offset)
                    .wrapping_add(extra.offset),
                dynamic: s.dynamic.or(extra.dynamic),
            });
        }
        let reg = self.inputs[base.0 as usize].ok_or("an address not from a root parameter")?;
        let (param, _) = self
            .param_of(reg)
            .ok_or("an address not from a root parameter")?;
        let high = match self.layout[param].root {
            RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap => None,
            RootType::Descriptor => {
                let high = Reg::S(self.stage.user_sgpr(self.layout[param].user_data) + 1);
                let input = self.p.inputs.iter().find(|i| i.0 == high);
                Some(input.ok_or("a root descriptor without its high half")?.1)
            }
            _ => return Err("an address from a root parameter that is no address"),
        };
        // shortcut: with a computed term, a high half that adds anything to the
        // base's is taken as the carry of that term (offsets below 4 GB).
        let dynamic = s.dynamic.or(extra.dynamic);
        if !self.high_half(hi, high)
            && !(dynamic.is_some() && high.is_some_and(|h| self.carried(hi, h)))
        {
            return Err("an address with an untracked high half");
        }
        Ok(Address {
            param,
            offset: s.offset.wrapping_add(extra.offset),
            dynamic: s.dynamic.or(extra.dynamic),
        })
    }

    /// `hi` is `high` (0 for a table), possibly plus a carry.
    fn high_half(&self, hi: Term, high: Option<Value>) -> bool {
        let want = high.map_or(Term::Const(0), Term::Value);
        hi == want
            || matches!(hi, Term::Value(v) if self.sop2(v, &["s_addc_u32"]).is_some_and(
                |(_, [a, b])| a == want && b == Term::Const(0) || b == want && a == Term::Const(0)
            ))
    }

    /// `hi` adds something (with carry) to `high`: `s_addc_u32` or
    /// `v_add_co_ci_u32` with `high` as an operand.
    fn carried(&self, hi: Term, high: Value) -> bool {
        let Term::Value(v) = hi else { return false };
        if v == high {
            return true;
        }
        if let Some((_, [a, b])) = self.sop2(v, &["s_addc_u32"]) {
            return self.carried(a, high) || self.carried(b, high);
        }
        let Some(inst) = self.def(v) else {
            return false;
        };
        let Fields::Valu(f) = &inst.fields else {
            return false;
        };
        inst.name() == "v_add_co_ci_u32"
            && f.src[..2.min(f.src.len())]
                .iter()
                .any(|o| self.vterm(inst, o).is_some_and(|t| self.carried(t, high)))
    }

    fn why_no_base(&self, dynamic: Option<Value>) -> &'static str {
        let Some(v) = dynamic else {
            return "a constant address";
        };
        if self.inputs[v.0 as usize].is_some() {
            return "an address in an SGPR that is no root parameter";
        }
        if self
            .leaves(v)
            .any(|l| self.def(l).is_some_and(|i| i.name() == "s_getpc_b64"))
        {
            return "an address relative to the program counter";
        }
        match self.def(v).map(|i| i.name()) {
            Some(n) if n.starts_with("s_load_") => "an address loaded from memory",
            None => "an address from a phi",
            _ => "an address not from a root parameter",
        }
    }

    /// The terms of the additions that compute `v`.
    fn leaves(&self, v: Value) -> Box<dyn Iterator<Item = Value> + '_> {
        match self.sop2(v, &["s_add_u32", "s_add_i32"]) {
            Some((_, terms)) => Box::new(terms.into_iter().flat_map(move |t| match t {
                Term::Value(x) => self.leaves(x),
                Term::Const(_) => Box::new(std::iter::empty()),
            })),
            None => Box::new(std::iter::once(v)),
        }
    }

    fn smem_address(
        &self,
        inst: &Inst,
        base: &Operand,
        offset: i32,
        soffset: &Operand,
    ) -> Result<Address, &'static str> {
        let Operand::Scalar { code, .. } = *base else {
            return Err("an address not in SGPRs");
        };
        let word = |k| Some(self.origin(inst.read(Reg::S(code + k))?));
        let (Some(lo), Some(hi)) = (word(0), word(1)) else {
            return Err("an address not in SGPRs");
        };
        let mut extra = self.sum(
            self.term(inst, soffset)
                .ok_or("an unknown offset operand")?,
        );
        if extra.base.is_some() {
            return Err("an offset from a root parameter");
        }
        extra.offset = extra.offset.wrapping_add(offset as u32);
        self.address(lo, hi, extra)
    }

    /// The source of the descriptor operand `op` of `inst`, at `site`.
    fn descriptor(&self, inst: &Inst, op: &Operand, site: Site) -> Result<Source, &'static str> {
        let Operand::Scalar { code, dwords } = *op else {
            return Err("a descriptor not in SGPRs");
        };
        let words = (0..dwords)
            .map(|k| inst.read(Reg::S(code + k)).map(|v| self.origin(v)))
            .collect::<Option<Vec<_>>>()
            .ok_or("a descriptor not in SGPRs")?;
        // Only the lanes that reach the access read the descriptor.
        let mut known = Vec::from_iter(self.lanes.map(|l| (l.reach[site.0], true)));
        self.classify(site, &words, &mut Vec::new(), &mut known)
    }

    /// The source of a descriptor's words; `joins` are the phi blocks
    /// being resolved, `known` holds in every lane that reads them.
    fn classify(
        &self,
        site: Site,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Source, &'static str> {
        if let Some(source) = self.phi(site, words, joins, known)? {
            return Ok(source);
        }
        if let Some(source) = self.select(site, words, joins, known)? {
            return Ok(source);
        }
        let read = |w: &Term| match *w {
            Term::Value(v) => self.first_lane_source(v),
            Term::Const(_) => None,
        };
        if let Some(sources) = words.iter().map(read).collect::<Option<Vec<_>>>() {
            // Read from the first active lane, as a waterfall loop does: the
            // dwords each lane holds.
            let lanes: Vec<Term> = sources.into_iter().map(Term::Value).collect();
            return self.per_lane(site, &lanes, joins, known);
        }
        if words.iter().any(|w| read(w).is_some()) {
            return Err("a descriptor partly read from the first lane");
        }
        if let Some(consts) = words
            .iter()
            .map(|w| match *w {
                Term::Const(c) => Some(c),
                Term::Value(_) => None,
            })
            .collect::<Option<Vec<_>>>()
        {
            return Ok(Source::Constant(consts));
        }
        if let Some(source) = self.loaded(words) {
            return Ok(source);
        }
        if let [lo, hi, Term::Const(w2), Term::Const(w3)] = *words {
            let (mut hi, mut bits) = (hi, 0);
            while let Term::Value(v) = hi
                && let Some((x, more)) = self.set_bits(v)
            {
                (hi, bits) = (x, bits | more);
            }
            let address = self.address(
                lo,
                hi,
                Sum {
                    base: None,
                    offset: 0,
                    dynamic: None,
                },
            )?;
            return Ok(Source::Built {
                address,
                bits,
                words: [w2, w3],
            });
        }
        let loaded = |w: &Term| {
            matches!(*w, Term::Value(v)
                if self.def(v).is_some_and(|i| i.name().starts_with("s_load_")))
        };
        Err(if words.iter().any(loaded) {
            "a descriptor changed after loading"
        } else {
            "a descriptor neither loaded nor built"
        })
    }

    /// Words that phis of one block join, traced per predecessor (words
    /// defined before the join hold on every edge).
    fn phi(
        &self,
        site: Site,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let phi_of = |w: &Term| match *w {
            Term::Value(v) => self.phis[v.0 as usize],
            Term::Const(_) => None,
        };
        let Some((block, _)) = words.iter().find_map(phi_of) else {
            return Ok(None);
        };
        if words.iter().filter_map(phi_of).any(|(b, _)| b != block) {
            return Err("a descriptor joined in several blocks");
        }
        if joins.contains(&block) {
            return Err("a descriptor that changes in a loop");
        }
        // Each level multiplies the sources by the predecessor count.
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(block);
        let phis = &self.p.blocks[block].phis;
        let args = (0..self.p.blocks[block].preds.len())
            .map(|j| {
                let edge: Vec<Term> = words
                    .iter()
                    .map(|w| match phi_of(w) {
                        Some((_, i)) => self.origin(phis[i].args[j]),
                        None => *w,
                    })
                    .collect();
                self.classify(site, &edge, joins, known)
            })
            .collect::<Result<_, _>>();
        joins.pop();
        Ok(Some(Source::Phi { block, args: args? }))
    }

    /// Words that `s_cselect` (or `s_cmov`) picks by one SCC value, traced
    /// per outcome (other words are the same for both).
    fn select(
        &self,
        site: Site,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let picks: Vec<_> = words
            .iter()
            .map(|w| match *w {
                Term::Value(v) => self.picked(v),
                Term::Const(_) => None,
            })
            .collect();
        let Some(scc) = picks.iter().flatten().map(|p| p.0).next() else {
            return Ok(None);
        };
        if picks.iter().flatten().any(|p| p.0 != scc) {
            return Err("a descriptor chosen by several conditions");
        }
        // Each level doubles the sources; selects count like joins (a
        // marker no block index reaches).
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(usize::MAX);
        let side = |set: bool| -> Vec<Term> {
            words
                .iter()
                .zip(&picks)
                .map(|(w, p)| match p {
                    Some((_, t, f)) => {
                        if set {
                            *t
                        } else {
                            *f
                        }
                    }
                    None => *w,
                })
                .collect()
        };
        let args = [true, false]
            .into_iter()
            .map(|set| self.classify(site, &side(set), joins, known))
            .collect::<Result<_, _>>();
        joins.pop();
        Ok(Some(Source::Select { scc, args: args? }))
    }

    /// For `v` written by `s_cselect_b32`/`_b64` or `s_cmov_b32`/`_b64`:
    /// the SCC it reads and the dword it is when SCC is set, then clear.
    fn picked(&self, v: Value) -> Option<(Value, Term, Term)> {
        let inst = self.def(v)?;
        let Fields::Sop { dst, src, .. } = &inst.fields else {
            return None;
        };
        let k = inst.writes.iter().position(|w| w.1 == v)? as u8;
        let wide = match inst.name() {
            "s_cselect_b32" | "s_cmov_b32" => false,
            "s_cselect_b64" | "s_cmov_b64" => true,
            _ => return None,
        };
        // A dword of a 32- or 64-bit operand: 64-bit inline integers are
        // sign-extended; 64-bit literals and float constants are not
        // followed.
        let dword = |op: &Operand| match *op {
            Operand::Scalar { code: NULL, .. } => Some(Term::Const(0)),
            Operand::Scalar { code, .. } => Some(self.origin(inst.read(Reg::S(code + k))?)),
            Operand::Const(c) if !wide => inline(c).map(Term::Const),
            Operand::Const(c @ 128..=208) => {
                let low = inline(c)?;
                Some(Term::Const(if k == 0 {
                    low
                } else {
                    ((low as i32) >> 31) as u32
                }))
            }
            Operand::Literal(x) if !wide => Some(Term::Const(x)),
            _ => None,
        };
        let set = dword(src.first()?)?;
        let clear = if inst.name().starts_with("s_cmov") {
            dword(dst.as_ref()?)?
        } else {
            dword(src.get(1)?)?
        };
        Some((inst.read(Reg::Scc)?, set, clear))
    }

    /// `v` as some other dword with constant bits set.
    fn set_bits(&self, v: Value) -> Option<(Term, u32)> {
        if let Some((_, [x, Term::Const(bits)] | [Term::Const(bits), x])) =
            self.sop2(v, &["s_or_b32"])
        {
            return Some((x, bits));
        }
        let inst = self.def(v)?;
        let Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            src,
            ..
        } = &inst.fields
        else {
            return None;
        };
        match (inst.name(), self.term(inst, src.first()?)?) {
            ("s_bitset1_b32", Term::Const(bit)) => {
                Some((self.origin(inst.read(Reg::S(*code))?), 1 << (bit & 31)))
            }
            _ => None,
        }
    }

    /// For a `v_readfirstlane_b32` result: the VGPR value it reads.
    fn first_lane_source(&self, v: Value) -> Option<Value> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if inst.name() != "v_readfirstlane_b32" || f.sdwa.is_some() || f.dpp.is_some() {
            return None;
        }
        let Some(&Operand::Vector { reg, .. }) = f.src.first() else {
            return None;
        };
        inst.read(Reg::V(reg))
    }

    /// The `v_readfirstlane_b32` results the descriptor operand `op` of
    /// `inst` is, when every dword is one.
    fn first_lane_words(&self, inst: &Inst, op: &Operand) -> Option<Vec<Value>> {
        let Operand::Scalar { code, dwords } = *op else {
            return None;
        };
        (0..dwords)
            .map(|k| match self.origin(inst.read(Reg::S(code + k))?) {
                Term::Value(v) => self.first_lane_source(v).map(|_| v),
                Term::Const(_) => None,
            })
            .collect()
    }

    /// Whether `t` is a value each lane holds its own of (one in VGPRs).
    fn varies(&self, t: Term) -> bool {
        let Term::Value(v) = t else { return false };
        let reg = match (self.inputs[v.0 as usize], self.phis[v.0 as usize]) {
            (Some(reg), _) => Some(reg),
            (None, Some((b, i))) => Some(self.p.blocks[b].phis[i].reg),
            (None, None) => self
                .def(v)
                .and_then(|i| i.writes.iter().find(|w| w.1 == v))
                .map(|w| w.0),
        };
        matches!(reg, Some(Reg::V(_)))
    }

    /// `t` through VGPR copies and lane-preserving moves/extracts where
    /// every lane in `known` sees the same result (including guarded writes
    /// that provably rewrite what active lanes already held).
    fn lane_value(&self, t: Term, known: &[(Bool, bool)]) -> Result<Term, &'static str> {
        self.lane_value_in(t, known, 0)
    }

    fn lane_value_in(
        &self,
        mut t: Term,
        known: &[(Bool, bool)],
        depth: usize,
    ) -> Result<Term, &'static str> {
        for _ in 0..MAX_JOINS * 2 {
            let Term::Value(v) = t else { break };
            if !self.varies(t) {
                let next = self.origin(v);
                if next == t {
                    break;
                }
                t = next;
                continue;
            }
            let Some(site) = self.defs[v.0 as usize] else {
                break;
            };
            if let Some((dst_reg, next)) = self.lane_copy(v) {
                match self.runs(site, known) {
                    Ok(()) => {
                        if next == t {
                            break;
                        }
                        t = next;
                    }
                    Err(err) => {
                        t = self
                            .redundant_write(site, dst_reg, next, known, depth)
                            .ok_or(err)?
                    }
                }
            } else if let Some((dst_reg, _, _, _)) = self.lane_picked_with_reg(v)
                && let Err(err) = self.runs(site, known)
            {
                // A guarded v_cndmask_b32 whose active lanes already hold the
                // selected value is also a no-op across all lanes in `known`.
                t = self
                    .redundant_write(site, dst_reg, Term::Value(v), known, depth)
                    .ok_or(err)?;
            } else {
                break;
            }
        }
        Ok(t)
    }

    /// When a guarded write to `dst_reg` at `site` writes `rhs` equal to the
    /// prior value on every active lane in `known`, returns the prior value.
    fn redundant_write(
        &self,
        site: Site,
        dst_reg: Reg,
        rhs: Term,
        known: &[(Bool, bool)],
        depth: usize,
    ) -> Option<Term> {
        if depth >= MAX_JOINS {
            return None;
        }
        let old = self.inst(site).read(dst_reg)?;
        let lanes = self.lanes?;
        let guard = lanes.guards[site.0][site.1]?;
        let mut active = known.to_vec();
        active.extend([(lanes.reach[site.0], true), (guard, true)]);
        self.lane_value_equals(Term::Value(old), rhs, &active, depth + 1)
            .then_some(Term::Value(old))
    }

    /// A single lane-preserving VGPR copy, identity ALU operation, or
    /// 32-bit word extract writing `v`: its destination register and the
    /// value written when active.
    fn lane_copy(&self, v: Value) -> Option<(Reg, Term)> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if f.abs != 0 || f.neg != 0 || f.clamp || f.omod != 0 || f.dpp.is_some() {
            return None;
        }
        let name = inst.name();
        if name == "v_mov_b32" {
            let Some(Operand::Vector { reg, .. }) = f.dst else {
                return None;
            };
            if inst.write(Reg::V(reg)) != Some(v) {
                return None;
            }
            if !f
                .sdwa
                .is_none_or(|s| s.dst_sel == 6 && s.src_sel[0] == 6 && s.sext & 1 == 0)
            {
                return None;
            }
            return Some((Reg::V(reg), self.raw(inst, f.src.first()?)?));
        }
        if f.sdwa.is_some() {
            return None;
        }
        if name == "v_swap_b32" {
            let (Some(Operand::Vector { reg: dreg, .. }), Some(&Operand::Vector { reg: sreg, .. })) =
                (f.dst, f.src.first())
            else {
                return None;
            };
            if inst.write(Reg::V(dreg)) == Some(v) {
                return Some((Reg::V(dreg), Term::Value(inst.read(Reg::V(sreg))?)));
            }
            if inst.write(Reg::V(sreg)) == Some(v) {
                return Some((Reg::V(sreg), Term::Value(inst.read(Reg::V(dreg))?)));
            }
            return None;
        }
        let Some(Operand::Vector { reg: dreg, dwords }) = f.dst else {
            return None;
        };
        if dwords == 2 && matches!(name, "v_lshrrev_b64" | "v_ashrrev_i64" | "v_lshlrev_b64") {
            let [sh_op, src_op] = &f.src[..] else {
                return None;
            };
            let Term::Const(sh) = self.raw(inst, sh_op)? else {
                return None;
            };
            let words = self.raw64(inst, src_op)?;
            let sh = sh & 63;
            let k = inst
                .writes
                .iter()
                .filter(|w| matches!(w.0, Reg::V(_)))
                .position(|w| w.1 == v)?;
            let reg = Reg::V(dreg + k as u8);
            return match (name, sh, k) {
                (_, 0, 0) => Some((reg, words[0])),
                (_, 0, 1) => Some((reg, words[1])),
                ("v_lshrrev_b64" | "v_ashrrev_i64", 32, 0) => Some((reg, words[1])),
                ("v_lshrrev_b64", 32, 1) => Some((reg, Term::Const(0))),
                ("v_lshlrev_b64", 32, 0) => Some((reg, Term::Const(0))),
                ("v_lshlrev_b64", 32, 1) => Some((reg, words[0])),
                _ => None,
            };
        }
        if inst.write(Reg::V(dreg)) != Some(v) {
            return None;
        }
        let dst_reg = Reg::V(dreg);
        let op = |k: usize| self.raw(inst, f.src.get(k)?);
        let term = match name {
            "v_or_b32" => match (op(0)?, op(1)?) {
                (a, Term::Const(0)) | (Term::Const(0), a) => Some(a),
                (a, b) if a == b => Some(a),
                _ => None,
            },
            "v_and_b32" => match (op(0)?, op(1)?) {
                (a, Term::Const(u32::MAX)) | (Term::Const(u32::MAX), a) => Some(a),
                (a, b) if a == b => Some(a),
                _ => None,
            },
            "v_xor_b32" => match (op(0)?, op(1)?) {
                (a, Term::Const(0)) | (Term::Const(0), a) => Some(a),
                _ => None,
            },
            "v_add_nc_u32" | "v_add_co_u32" => match (op(0)?, op(1)?) {
                (a, Term::Const(0)) | (Term::Const(0), a) => Some(a),
                _ => None,
            },
            "v_sub_nc_u32" | "v_sub_co_u32" => match (op(0)?, op(1)?) {
                (a, Term::Const(0)) => Some(a),
                _ => None,
            },
            "v_subrev_nc_u32" | "v_subrev_co_u32" => match (op(0)?, op(1)?) {
                (Term::Const(0), a) => Some(a),
                _ => None,
            },
            "v_lshlrev_b32" | "v_lshrrev_b32" | "v_ashrrev_i32" => match (op(0)?, op(1)?) {
                (Term::Const(sh), a) if sh & 31 == 0 => Some(a),
                _ => None,
            },
            "v_mul_lo_u32" => match (op(0)?, op(1)?) {
                (a, Term::Const(1)) | (Term::Const(1), a) => Some(a),
                _ => None,
            },
            "v_min_u32" => match (op(0)?, op(1)?) {
                (a, Term::Const(u32::MAX)) | (Term::Const(u32::MAX), a) => Some(a),
                (a, b) if a == b => Some(a),
                _ => None,
            },
            "v_max_u32" => match (op(0)?, op(1)?) {
                (a, Term::Const(0)) | (Term::Const(0), a) => Some(a),
                (a, b) if a == b => Some(a),
                _ => None,
            },
            "v_min_i32" | "v_max_i32" => match (op(0)?, op(1)?) {
                (a, b) if a == b => Some(a),
                _ => None,
            },
            "v_bfi_b32" => match (op(0)?, op(1)?, op(2)?) {
                (Term::Const(u32::MAX), a, _) | (Term::Const(0), _, a) => Some(a),
                (_, a, b) if a == b => Some(a),
                _ => None,
            },
            "v_perm_b32" => match (op(0)?, op(1)?, op(2)?) {
                (_, src1, Term::Const(0x0302_0100)) => Some(src1),
                (src0, _, Term::Const(0x0706_0504)) => Some(src0),
                _ => None,
            },
            "v_alignbit_b32" => match (op(1)?, op(2)?) {
                (src1, Term::Const(sh)) if sh & 31 == 0 => Some(src1),
                _ => None,
            },
            "v_alignbyte_b32" => match (op(1)?, op(2)?) {
                (src1, Term::Const(sh)) if sh & 3 == 0 => Some(src1),
                _ => None,
            },
            "v_cndmask_b32" => {
                let (clear, set, mask) = (op(0)?, op(1)?, f.src.get(2)?);
                match *mask {
                    Operand::Scalar { code: NULL, .. }
                    | Operand::Const(128)
                    | Operand::Literal(0) => Some(clear),
                    Operand::Const(193) => Some(set),
                    Operand::Literal(u32::MAX) if !self.p.wave64 => Some(set),
                    _ if clear == set => Some(clear),
                    _ => None,
                }
            }
            _ => None,
        }?;
        Some((dst_reg, term))
    }

    /// Resolves a 64-bit VALU operand to its `[low, high]` dwords.
    fn raw64(&self, inst: &Inst, op: &Operand) -> Option<[Term; 2]> {
        match *op {
            Operand::Vector { reg, .. } => Some([
                Term::Value(inst.read(Reg::V(reg))?),
                Term::Value(inst.read(Reg::V(reg + 1))?),
            ]),
            Operand::Scalar { code: NULL, .. } => Some([Term::Const(0), Term::Const(0)]),
            Operand::Scalar { code, .. } => Some([
                self.origin(inst.read(Reg::S(code))?),
                self.origin(inst.read(Reg::S(code + 1))?),
            ]),
            Operand::Const(c @ 128..=208) => {
                let low = inline(c)?;
                Some([Term::Const(low), Term::Const(((low as i32) >> 31) as u32)])
            }
            Operand::Literal(x) => Some([Term::Const(x), Term::Const(0)]),
            _ => None,
        }
    }

    /// A VALU operand as `inst` reads it: a VGPR's value itself, other
    /// operands' origins.
    fn raw(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Vector { reg, .. } => Some(Term::Value(inst.read(Reg::V(reg))?)),
            op => self.term(inst, &op),
        }
    }

    /// Proves that `a` and `b` evaluate to the same 32-bit word in every lane
    /// satisfying `known`.
    fn lane_value_equals(&self, a: Term, b: Term, known: &[(Bool, bool)], depth: usize) -> bool {
        if !self.feasible(known) {
            return true;
        }
        if depth > MAX_JOINS {
            return false;
        }
        let Ok(a) = self.lane_value_in(a, known, depth) else {
            return false;
        };
        let Ok(b) = self.lane_value_in(b, known, depth) else {
            return false;
        };
        if a == b {
            return true;
        }
        if depth == MAX_JOINS {
            return false;
        }
        if let Some(branches) = self.split_term(a, known) {
            let mut next_known = known.to_vec();
            return branches.into_iter().all(|(ext, arg)| {
                let n = next_known.len();
                next_known.extend(ext);
                let ok = self.lane_value_equals(arg, b, &next_known, depth + 1);
                next_known.truncate(n);
                ok
            });
        }
        if let Some(branches) = self.split_term(b, known) {
            let mut next_known = known.to_vec();
            return branches.into_iter().all(|(ext, arg)| {
                let n = next_known.len();
                next_known.extend(ext);
                let ok = self.lane_value_equals(a, arg, &next_known, depth + 1);
                next_known.truncate(n);
                ok
            });
        }
        false
    }

    /// Splits a forward phi or select `t` into `(edge_conditions, branch_term)`
    /// cases under `known`.
    fn split_term(&self, t: Term, known: &[(Bool, bool)]) -> Option<Vec<(Known, Term)>> {
        let Term::Value(v) = t else { return None };
        if let Some((block, phi_idx)) = self.phis[v.0 as usize] {
            let preds = &self.p.blocks[block].preds;
            if preds.iter().any(|&q| self.rpo[q] >= self.rpo[block]) {
                return None;
            }
            let lanes = self.lanes?;
            let args = &self.p.blocks[block].phis[phi_idx].args;
            return Some(
                preds
                    .iter()
                    .enumerate()
                    .map(|(j, &q)| {
                        let mut ext = vec![(lanes.reach[q], true)];
                        if let LaneExit::Branch {
                            when,
                            taken,
                            not_taken,
                        } = lanes.exits[q]
                            && taken != not_taken
                        {
                            ext.push((when, taken == block));
                        }
                        (ext, Term::Value(args[j]))
                    })
                    .collect(),
            );
        }
        if let Some((_, mask, set_val, clear_val)) = self.lane_picked_with_reg(v) {
            let site = self.defs[v.0 as usize]?;
            self.runs(site, known).ok()?;
            let bit = self.lanes?.masks.get(&mask).copied()?;
            return Some(vec![
                (vec![(bit, true)], set_val),
                (vec![(bit, false)], clear_val),
            ]);
        }
        if let Some((scc, set_val, clear_val)) = self.picked(v) {
            if let Some(lanes) = self.lanes
                && let Some(ub) = lanes.bools.var_bool(Atom::Uniform(scc))
            {
                return Some(vec![
                    (vec![(ub, true)], set_val),
                    (vec![(ub, false)], clear_val),
                ]);
            }
            return Some(vec![(vec![], set_val), (vec![], clear_val)]);
        }
        None
    }

    /// Whether some lane can satisfy all conditions in `all`, taking into
    /// account equivalent/opposite `v_cmp` lane atoms and 1D integer bounds.
    fn feasible(&self, all: &[(Bool, bool)]) -> bool {
        let Some(lanes) = self.lanes else {
            return true;
        };
        if !lanes.bools.meet(all) {
            return false;
        }
        let mut lane_atoms = Vec::new();
        for &(b, _) in all {
            for atom in lanes.bools.support(b) {
                if matches!(atom, Atom::Lane(_)) && !lane_atoms.contains(&atom) {
                    lane_atoms.push(atom);
                }
            }
        }
        if lane_atoms.len() < 2 {
            return true;
        }
        let mut groups: HashMap<CanonKey, usize> = HashMap::new();
        let mut equiv = Vec::new();
        for &atom in &lane_atoms {
            let Atom::Lane(v) = atom else { continue };
            if let Some((key, inv)) = self.canon_cmp(v) {
                let next_id = groups.len();
                let group = *groups.entry(key).or_insert(next_id);
                equiv.push((atom, group, inv));
            }
        }
        if equiv.is_empty() {
            return true;
        }
        let groups_len = groups.len();
        lanes
            .bools
            .meet_with_facts(all, &equiv, groups_len, |assigned| {
                self.cmp_leaf_ok(assigned)
            })
    }

    fn feasible_with(&self, known: &[(Bool, bool)], extra: &[(Bool, bool)]) -> bool {
        let mut conds = Vec::with_capacity(known.len() + extra.len());
        conds.extend_from_slice(known);
        conds.extend_from_slice(extra);
        self.feasible(&conds)
    }

    /// Whether two mask values evaluate to the same boolean on every lane in
    /// `known` (and both are tracked in `lanes.masks` when `lanes` is present).
    fn same_mask(&self, a: Value, b: Value, known: &[(Bool, bool)]) -> bool {
        if a == b {
            return true;
        }
        let Some(lanes) = self.lanes else {
            return false;
        };
        let (Some(&ma), Some(&mb)) = (lanes.masks.get(&a), lanes.masks.get(&b)) else {
            return false;
        };
        ma == mb
            || (!self.feasible_with(known, &[(ma, true), (mb, false)])
                && !self.feasible_with(known, &[(ma, false), (mb, true)]))
    }

    /// Canonicalizes a pure `v_cmp_*` / `v_cmpx_*` instruction defining `v`
    /// into `(canonical_key, inverted)` so equivalent or negated compares
    /// share the same key.
    fn canon_cmp(&self, v: Value) -> Option<(CanonKey, bool)> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if f.dpp.is_some() || f.clamp || f.omod != 0 {
            return None;
        }
        let rest = inst
            .name()
            .strip_prefix("v_cmp_")
            .or_else(|| inst.name().strip_prefix("v_cmpx_"))?;
        let (op, ty_str) = rest.rsplit_once('_')?;
        let ty = match ty_str {
            "u16" => CmpType::U16,
            "i16" => CmpType::I16,
            "f16" => CmpType::F16,
            "u32" => CmpType::U32,
            "i32" => CmpType::I32,
            "f32" => CmpType::F32,
            "u64" => CmpType::U64,
            "i64" => CmpType::I64,
            "f64" => CmpType::F64,
            _ => return None,
        };
        let a = self.cmp_operand(inst, f, 0, ty)?;
        let b = self.cmp_operand(inst, f, 1, ty)?;
        let ordered = |x: CmpOperand, y: CmpOperand| if x <= y { (x, y) } else { (y, x) };
        let (pred, inv) = match ty {
            CmpType::U16
            | CmpType::I16
            | CmpType::U32
            | CmpType::I32
            | CmpType::U64
            | CmpType::I64 => match op {
                "f" => (CanonPred::Const(false), false),
                "t" => (CanonPred::Const(true), false),
                "lt" => (CanonPred::IntLt(a, b), false),
                "ge" => (CanonPred::IntLt(a, b), true),
                "gt" => (CanonPred::IntLt(b, a), false),
                "le" => (CanonPred::IntLt(b, a), true),
                "eq" => {
                    let (lo, hi) = ordered(a, b);
                    (CanonPred::IntEq(lo, hi), false)
                }
                "ne" | "lg" => {
                    let (lo, hi) = ordered(a, b);
                    (CanonPred::IntEq(lo, hi), true)
                }
                _ => return None,
            },
            CmpType::F16 | CmpType::F32 | CmpType::F64 => match op {
                "f" => (CanonPred::Const(false), false),
                "tru" => (CanonPred::Const(true), false),
                "lt" => (CanonPred::FloatLt(a, b), false),
                "nlt" => (CanonPred::FloatLt(a, b), true),
                "gt" => (CanonPred::FloatLt(b, a), false),
                "ngt" => (CanonPred::FloatLt(b, a), true),
                "le" => (CanonPred::FloatLe(a, b), false),
                "nle" => (CanonPred::FloatLe(a, b), true),
                "ge" => (CanonPred::FloatLe(b, a), false),
                "nge" => (CanonPred::FloatLe(b, a), true),
                "eq" => {
                    let (lo, hi) = ordered(a, b);
                    (CanonPred::FloatEq(lo, hi), false)
                }
                "neq" => {
                    let (lo, hi) = ordered(a, b);
                    (CanonPred::FloatEq(lo, hi), true)
                }
                "lg" => {
                    let (lo, hi) = ordered(a, b);
                    (CanonPred::FloatLg(lo, hi), false)
                }
                "nlg" => {
                    let (lo, hi) = ordered(a, b);
                    (CanonPred::FloatLg(lo, hi), true)
                }
                "o" => {
                    let (lo, hi) = ordered(a, b);
                    (CanonPred::FloatOrd(lo, hi), false)
                }
                "u" => {
                    let (lo, hi) = ordered(a, b);
                    (CanonPred::FloatOrd(lo, hi), true)
                }
                "class" => (CanonPred::FloatClass(a, b), false),
                _ => return None,
            },
        };
        Some((CanonKey { ty, pred }, inv))
    }

    fn cmp_operand(
        &self,
        inst: &Inst,
        f: &crate::gfx10::Valu,
        k: usize,
        ty: CmpType,
    ) -> Option<CmpOperand> {
        let op = *f.src.get(k)?;
        let words = match ty {
            CmpType::U64 | CmpType::I64 | CmpType::F64 => match op {
                Operand::Vector { reg, .. } => [
                    self.lane_unconditional(inst.read(Reg::V(reg))?),
                    self.lane_unconditional(inst.read(Reg::V(reg + 1))?),
                ],
                Operand::Scalar { code: NULL, .. } => [Term::Const(0), Term::Const(0)],
                Operand::Scalar { code, .. } => [
                    self.origin(inst.read(Reg::S(code))?),
                    self.origin(inst.read(Reg::S(code + 1))?),
                ],
                Operand::Const(c @ 128..=208) => {
                    let low = inline(c)?;
                    [Term::Const(low), Term::Const(((low as i32) >> 31) as u32)]
                }
                Operand::Literal(x) => [Term::Const(x), Term::Const(0)],
                _ => return None,
            },
            _ => {
                let w = match op {
                    Operand::Vector { reg, .. } => self.lane_unconditional(inst.read(Reg::V(reg))?),
                    Operand::Const(c) if ty == CmpType::F16 && c >= 240 => return None,
                    _ => self.term(inst, &op)?,
                };
                [w, Term::Const(0)]
            }
        };
        let is16 = matches!(ty, CmpType::U16 | CmpType::I16 | CmpType::F16);
        Some(CmpOperand {
            words,
            abs: (f.abs >> k) & 1 != 0,
            neg: (f.neg >> k) & 1 != 0,
            opsel: is16 && ((f.opsel >> k) & 1 != 0),
            sdwa: f.sdwa.map(|s| (s.src_sel[k], (s.sext >> k) & 1 != 0)),
        })
    }

    /// Follows unconditional lane-preserving copies (those with no guard)
    /// from `v`.
    fn lane_unconditional(&self, mut v: Value) -> Term {
        for _ in 0..MAX_JOINS * 2 {
            let Some(site) = self.defs[v.0 as usize] else {
                return Term::Value(v);
            };
            if self
                .lanes
                .is_some_and(|l| l.guards[site.0][site.1].is_some())
            {
                return Term::Value(v);
            }
            let Some((_, next)) = self.lane_copy(v) else {
                return Term::Value(v);
            };
            match next {
                Term::Value(u) if u != v => v = u,
                other => return other,
            }
        }
        Term::Value(v)
    }

    /// Checks whether a leaf assignment of `Atom::Lane` compares is consistent
    /// with 1D integer constant bounds (e.g., `(x < 8) && !(x < 16)` is false).
    fn cmp_leaf_ok(&self, assigned: &[(Atom, bool)]) -> bool {
        struct Bounds {
            lo: i128,
            hi: i128,
            neq: Vec<i128>,
        }
        fn entry_for(
            map: &mut HashMap<(CmpType, CmpOperand), Bounds>,
            ty: CmpType,
            op: CmpOperand,
        ) -> &mut Bounds {
            let (min, max) = match ty {
                CmpType::U16 => (0i128, i128::from(u16::MAX)),
                CmpType::I16 => (i128::from(i16::MIN), i128::from(i16::MAX)),
                CmpType::U32 => (0i128, i128::from(u32::MAX)),
                CmpType::I32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
                CmpType::U64 => (0i128, i128::from(u64::MAX)),
                CmpType::I64 => (i128::from(i64::MIN), i128::from(i64::MAX)),
                _ => (0, 0),
            };
            map.entry((ty, op)).or_insert(Bounds {
                lo: min,
                hi: max,
                neq: Vec::new(),
            })
        }
        let mut bounds: HashMap<(CmpType, CmpOperand), Bounds> = HashMap::new();
        let int_const = |ty: CmpType, op: CmpOperand| -> Option<i128> {
            if op.abs || op.neg || op.opsel || op.sdwa.is_some() {
                return None;
            }
            let (Term::Const(w0), Term::Const(w1)) = (op.words[0], op.words[1]) else {
                return None;
            };
            Some(match ty {
                CmpType::U16 => i128::from(w0 as u16),
                CmpType::I16 => i128::from(w0 as u16 as i16),
                CmpType::U32 => i128::from(w0),
                CmpType::I32 => i128::from(w0 as i32),
                CmpType::U64 => i128::from(u64::from(w0) | (u64::from(w1) << 32)),
                CmpType::I64 => i128::from((u64::from(w0) | (u64::from(w1) << 32)) as i64),
                _ => return None,
            })
        };
        for &(atom, truth) in assigned {
            let Atom::Lane(v) = atom else { continue };
            let Some((CanonKey { ty, pred }, inv)) = self.canon_cmp(v) else {
                continue;
            };
            let holds = truth ^ inv;
            match pred {
                CanonPred::Const(c) => {
                    if c != holds {
                        return false;
                    }
                }
                CanonPred::IntLt(a, b) => {
                    if a == b {
                        if holds {
                            return false;
                        }
                        continue;
                    }
                    if let (Some(ka), Some(kb)) = (int_const(ty, a), int_const(ty, b)) {
                        if (ka < kb) != holds {
                            return false;
                        }
                        continue;
                    }
                    if let Some(k) = int_const(ty, b) {
                        let e = entry_for(&mut bounds, ty, a);
                        if holds {
                            e.hi = e.hi.min(k - 1);
                        } else {
                            e.lo = e.lo.max(k);
                        }
                    } else if let Some(k) = int_const(ty, a) {
                        let e = entry_for(&mut bounds, ty, b);
                        if holds {
                            e.lo = e.lo.max(k + 1);
                        } else {
                            e.hi = e.hi.min(k);
                        }
                    }
                }
                CanonPred::IntEq(a, b) => {
                    if a == b {
                        if !holds {
                            return false;
                        }
                        continue;
                    }
                    if let (Some(ka), Some(kb)) = (int_const(ty, a), int_const(ty, b)) {
                        if (ka == kb) != holds {
                            return false;
                        }
                        continue;
                    }
                    if let Some(k) = int_const(ty, b) {
                        let e = entry_for(&mut bounds, ty, a);
                        if holds {
                            e.lo = e.lo.max(k);
                            e.hi = e.hi.min(k);
                        } else {
                            e.neq.push(k);
                        }
                    } else if let Some(k) = int_const(ty, a) {
                        let e = entry_for(&mut bounds, ty, b);
                        if holds {
                            e.lo = e.lo.max(k);
                            e.hi = e.hi.min(k);
                        } else {
                            e.neq.push(k);
                        }
                    }
                }
                _ => {}
            }
        }
        bounds
            .values()
            .all(|b| b.lo <= b.hi && (b.lo != b.hi || !b.neq.contains(&b.lo)))
    }

    /// Fails unless every lane in `known` that reaches the vector
    /// instruction at `site` runs it: one with its EXEC bit clear there
    /// keeps what the registers held before.
    fn runs(&self, site: Site, known: &[(Bool, bool)]) -> Result<(), &'static str> {
        let Some(lanes) = self.lanes else {
            return Ok(());
        };
        let Some(&Some(guard)) = lanes.guards.get(site.0).and_then(|g| g.get(site.1)) else {
            return Ok(());
        };
        let mut skips = known.to_vec();
        skips.extend([(lanes.reach[site.0], true), (guard, false)]);
        if self.feasible(&skips) {
            return Err("a per-lane descriptor written where some lanes may keep the old one");
        }
        Ok(())
    }

    /// The source of descriptor dwords each lane holds its own of (a
    /// waterfall loop reads them from the first active lane): the same in
    /// every lane, loaded per lane from memory a heap address points to, or
    /// picked per lane between such sources, by a lane mask or by the path
    /// the lane took. `known` holds in every lane that accesses them.
    fn per_lane(
        &self,
        site: Site,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Source, &'static str> {
        let words = words
            .iter()
            .map(|&w| self.lane_value(w, known))
            .collect::<Result<Vec<_>, _>>()?;
        let words = &words[..];
        let Some(&lane) = words.iter().find(|&&w| self.varies(w)) else {
            return self.classify(site, words, joins, known);
        };
        if let Some(source) = self.fetched(words, known)? {
            return Ok(source);
        }
        if let Some(source) = self.lane_select(site, words, joins, known)? {
            return Ok(source);
        }
        if let Some(source) = self.lane_phi(site, words, joins, known)? {
            return Ok(source);
        }
        let Term::Value(v) = lane else { unreachable!() };
        let name = self.def(v).map(|i| i.name());
        Err(match name {
            _ if self.inputs[v.0 as usize].is_some() => {
                "a descriptor read from the first lane of an input VGPR"
            }
            None => "a descriptor read from the first lane of a phi",
            Some(n) if n.starts_with("ds_read") => {
                "a descriptor read from the first lane of an LDS read"
            }
            Some(n) if n.starts_with("buffer_load") => {
                "a descriptor read from the first lane of a buffer load"
            }
            Some(n) if n.starts_with("global_load") || n.starts_with("flat_load") => {
                "a descriptor read from the first lane of a partial or mixed load"
            }
            Some(_) => "a descriptor read from the first lane of computed VGPRs",
        })
    }

    /// Dwords of `global_load_dword*` that line up: `first..` of one load,
    /// then the bytes that follow (an image descriptor takes two
    /// `global_load_dwordx4`).
    fn fetched(&self, words: &[Term], known: &Known) -> Result<Option<Source>, &'static str> {
        let at = |w: &Term| -> Option<(Site, usize)> {
            let Term::Value(v) = *w else { return None };
            let site = self.defs[v.0 as usize]?;
            let inst = self.inst(site);
            if !inst.name().starts_with("global_load_dword") {
                return None;
            }
            let k = inst
                .writes
                .iter()
                .filter(|w| matches!(w.0, Reg::V(_)))
                .position(|w| w.1 == v)?;
            Some((site, k))
        };
        let Some(sites) = words.iter().map(at).collect::<Option<Vec<_>>>() else {
            return Ok(None);
        };
        let address = |site: Site| {
            let inst = self.inst(site);
            let Fields::Flat {
                addr,
                saddr,
                offset,
                ..
            } = &inst.fields
            else {
                return Err("a global load");
            };
            self.flat_address(inst, addr.as_ref(), saddr.as_ref(), *offset)
        };
        let (load, first) = sites[0];
        let a0 = address(load)?;
        for (k, &(site, dword)) in sites.iter().enumerate() {
            self.runs(site, known)?;
            let a = address(site)?;
            let want = a0.offset.wrapping_add(4 * (first + k) as u32);
            if (a.param, a.dynamic) != (a0.param, a0.dynamic)
                || a.offset.wrapping_add(4 * dword as u32) != want
            {
                return Err("a descriptor read from the first lane of loads that do not line up");
            }
        }
        Ok(Some(Source::Fetched {
            load,
            first: first as u8,
        }))
    }

    /// Words that `v_cndmask_b32` picks by one lane mask, traced per outcome
    /// (other words are the same for both).
    fn lane_select(
        &self,
        site: Site,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let picks: Vec<_> = words
            .iter()
            .map(|w| match *w {
                Term::Value(v) => self.lane_picked(v),
                Term::Const(_) => None,
            })
            .collect();
        let Some(mask) = picks.iter().flatten().map(|p| p.0).next() else {
            return Ok(None);
        };
        if picks
            .iter()
            .flatten()
            .any(|p| !self.same_mask(p.0, mask, known))
        {
            return Err("a descriptor chosen per lane by several masks");
        }
        for (w, p) in words.iter().zip(&picks) {
            if let (Term::Value(v), Some(_)) = (*w, p)
                && let Some(def_site) = self.defs[v.0 as usize]
            {
                self.runs(def_site, known)?;
            }
        }
        let bit = self.lanes.and_then(|l| l.masks.get(&mask).copied());
        let side = |set: bool| -> Vec<Term> {
            words
                .iter()
                .zip(&picks)
                .map(|(w, p)| match p {
                    Some((_, t, f)) => {
                        if set {
                            *t
                        } else {
                            *f
                        }
                    }
                    None => *w,
                })
                .collect()
        };
        if let Some(b) = bit {
            let t_ok = self.feasible_with(known, &[(b, true)]);
            let f_ok = self.feasible_with(known, &[(b, false)]);
            if t_ok != f_ok {
                return self.per_lane(site, &side(t_ok), joins, known).map(Some);
            }
        }
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(usize::MAX);
        // Each side holds where the lane's bit of the mask is set, or clear.
        let mut results = [true, false].into_iter().map(|set| {
            let n = known.len();
            known.extend(bit.map(|b| (b, set)));
            let arg = self.per_lane(site, &side(set), joins, known);
            known.truncate(n);
            arg
        });
        let t_res = results.next().unwrap();
        let f_res = results.next().unwrap();
        joins.pop();
        match (t_res, f_res) {
            (Ok(t_src), Ok(f_src)) => Ok(Some(Source::Lanes {
                mask,
                args: vec![t_src, f_src],
            })),
            (Ok(good_src), Err(err)) => {
                if let Some(b) = bit
                    && self.dead_for_bad_lanes(site, known, &[(b, false)])
                {
                    Ok(Some(good_src))
                } else {
                    Err(err)
                }
            }
            (Err(err), Ok(good_src)) => {
                if let Some(b) = bit
                    && self.dead_for_bad_lanes(site, known, &[(b, true)])
                {
                    Ok(Some(good_src))
                } else {
                    Err(err)
                }
            }
            (Err(err), Err(_)) => Err(err),
        }
    }

    /// For `v` written by a plain `v_cndmask_b32`: the low dword of the
    /// lane mask it reads, and the dword it is where the lane's bit is set
    /// (src1), then clear (src0).
    fn lane_picked(&self, v: Value) -> Option<(Value, Term, Term)> {
        let (_, mask, set, clear) = self.lane_picked_with_reg(v)?;
        Some((mask, set, clear))
    }

    fn lane_picked_with_reg(&self, v: Value) -> Option<(Reg, Value, Term, Term)> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        let Some(Operand::Vector { reg, .. }) = f.dst else {
            return None;
        };
        if inst.name() != "v_cndmask_b32"
            || inst.write(Reg::V(reg)) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.clamp
            || f.omod != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        let [clear, set, Operand::Scalar { code, .. }] = f.src[..] else {
            return None;
        };
        let mask = inst.read(Reg::S(code))?;
        Some((
            Reg::V(reg),
            mask,
            self.raw(inst, &set)?,
            self.raw(inst, &clear)?,
        ))
    }

    /// Per-lane words that phis of one block join, as an if/else on a
    /// per-lane condition leaves them: traced on each edge into the block
    /// that lanes in `known` can take (other words hold on every edge).
    /// Lanes take their own paths ([`Structured`]), so each gets the value
    /// of the edge it came by; an edge they cannot take is `None` (a flow
    /// block's phi joins an arm's dwords with what lanes that skipped the
    /// arm held before, which only they read).
    fn lane_phi(
        &self,
        site: Site,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let phi_of = |w: &Term| match *w {
            Term::Value(v) => self.phis[v.0 as usize],
            Term::Const(_) => None,
        };
        let Some((block, _)) = words.iter().find_map(phi_of) else {
            return Ok(None);
        };
        if words.iter().filter_map(phi_of).any(|(b, _)| b != block) {
            return Err("a descriptor joined in several blocks");
        }
        if joins.contains(&block) {
            return Err("a descriptor that changes in a loop");
        }
        let preds = &self.p.blocks[block].preds;
        if preds.iter().any(|&q| self.rpo[q] >= self.rpo[block]) {
            // shortcut: what a lane holds after some passes is not traced.
            return Err("a descriptor read from the first lane of a phi at a loop header");
        }
        let Some(lanes) = self.lanes else {
            return Err("a descriptor read from the first lane of a phi, without the lanes' paths");
        };
        // The other words must be there before the join: defined in a
        // block that dominates it, which (both dominating the access) is
        // one the join does not reach.
        for w in words {
            if let Term::Value(v) = *w
                && phi_of(w).is_none()
                && let Some((d, _)) = self.defs[v.0 as usize]
                && (d == block || self.reaches(block, d))
            {
                return Err("a descriptor partly joined by control flow");
            }
        }
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(block);
        let phis = &self.p.blocks[block].phis;
        let mut results: Vec<(Result<Option<Source>, &'static str>, Known)> =
            Vec::with_capacity(preds.len());
        for (j, &q) in preds.iter().enumerate() {
            let n = known.len();
            known.push((lanes.reach[q], true));
            if let LaneExit::Branch {
                when,
                taken,
                not_taken,
            } = lanes.exits[q]
                && taken != not_taken
            {
                known.push((when, taken == block));
            }
            let edge_cond = known[n..].to_vec();
            let arg = if self.feasible(known) {
                let edge: Vec<Term> = words
                    .iter()
                    .map(|w| match phi_of(w) {
                        Some((_, i)) => Term::Value(phis[i].args[j]),
                        None => *w,
                    })
                    .collect();
                self.per_lane(site, &edge, joins, known).map(Some)
            } else {
                Ok(None)
            };
            known.truncate(n);
            results.push((arg, edge_cond));
        }
        joins.pop();
        let has_valid = results.iter().any(|(r, _)| matches!(r, Ok(Some(_))));
        let mut args = Vec::with_capacity(results.len());
        let mut first_err = None;
        for (res, edge_cond) in results {
            match res {
                Ok(a) => args.push(a),
                Err(e) => {
                    if has_valid
                        && first_err.is_none()
                        && self.dead_for_bad_lanes(site, known, &edge_cond)
                    {
                        args.push(None);
                    } else {
                        first_err.get_or_insert(e);
                    }
                }
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        Ok(Some(Source::LanePhi { block, args }))
    }

    fn uses(&self) -> &[Vec<UseSite>] {
        self.uses.get_or_init(|| {
            let mut uses = vec![Vec::new(); self.defs.len()];
            for (block, b) in self.p.blocks.iter().enumerate() {
                for (phi_idx, phi) in b.phis.iter().enumerate() {
                    for (pred_idx, &arg) in phi.args.iter().enumerate() {
                        uses[arg.0 as usize].push(UseSite::Phi {
                            block,
                            pred_idx,
                            phi_idx,
                        });
                    }
                }
                for (inst_idx, inst) in b.insts.iter().enumerate() {
                    for &(_, v) in &inst.reads {
                        let list = &mut uses[v.0 as usize];
                        if !matches!(
                            list.last(),
                            Some(&UseSite::Inst { block: pb, inst_idx: pi })
                                if pb == block && pi == inst_idx
                        ) {
                            list.push(UseSite::Inst { block, inst_idx });
                        }
                    }
                }
            }
            uses
        })
    }

    fn bool_born_before(&self, b: Bool, header_rpo: usize) -> bool {
        let Some(lanes) = self.lanes else {
            return false;
        };
        lanes.bools.support(b).into_iter().all(|atom| {
            let born_rpo = match atom {
                Atom::Constant(_) => 0,
                Atom::Lane(v)
                | Atom::Uniform(v)
                | Atom::Phi(v)
                | Atom::Quad(v)
                | Atom::Opaque(v) => {
                    if self.inputs[v.0 as usize].is_some() {
                        0
                    } else if let Some((db, _)) = self.defs[v.0 as usize] {
                        self.rpo[db]
                    } else if let Some((pb, _)) = self.phis[v.0 as usize] {
                        self.rpo[pb]
                    } else {
                        return false;
                    }
                }
            };
            born_rpo < header_rpo
        })
    }

    fn cndmask_bool(&self, inst: &Inst, mask_op: &Operand, lanes: &Structured) -> Option<Bool> {
        let Operand::Scalar { code, .. } = *mask_op else {
            return None;
        };
        if code == NULL {
            return Some(Bool::FALSE);
        }
        let lo = inst.read(Reg::S(code))?;
        let hi = if self.p.wave64 {
            Some(inst.read(Reg::S(code + 1))?)
        } else {
            None
        };
        lanes.mask_bits(lo, hi)
    }

    fn cndmask_const(&self, mask_op: &Operand) -> Option<bool> {
        match *mask_op {
            Operand::Scalar { code: NULL, .. } | Operand::Const(128) | Operand::Literal(0) => {
                Some(false)
            }
            Operand::Const(193) => Some(true),
            Operand::Literal(u32::MAX) if !self.p.wave64 => Some(true),
            _ => None,
        }
    }

    /// Proves that the VGPR outputs of the pure read/sample at `site` are never
    /// consumed by any lane satisfying `known` extended with `extra`.
    fn dead_for_bad_lanes(
        &self,
        site: Site,
        known: &[(Bool, bool)],
        extra: &[(Bool, bool)],
    ) -> bool {
        let Some(lanes) = self.lanes else {
            return false;
        };
        let inst = self.inst(site);
        let is_pure_read = match &inst.fields {
            Fields::Image { tfe, lwe, .. } => {
                !*tfe
                    && !*lwe
                    && !inst.name().contains("_store")
                    && !inst.name().contains("_atomic_")
            }
            Fields::Buffer { lds, tfe, .. } => {
                !*lds
                    && !*tfe
                    && !inst.name().contains("_store_")
                    && !inst.name().contains("_atomic_")
            }
            _ => false,
        };
        if !is_pure_read
            || inst.writes.is_empty()
            || inst.writes.iter().any(|w| !matches!(w.0, Reg::V(_)))
        {
            return false;
        }
        let Some(bad_known) = extend_known(&[], known.iter().chain(extra).copied()) else {
            return true;
        };
        if !self.feasible(&bad_known) {
            return true;
        }

        let uses = self.uses();
        let mut visited: HashMap<Value, Vec<Known>> = HashMap::new();
        let mut worklist: Vec<(Value, Known)> = Vec::new();
        let mut states = 0usize;

        let push = |v: Value,
                    cond: Known,
                    visited: &mut HashMap<Value, Vec<Known>>,
                    worklist: &mut Vec<(Value, Known)>,
                    states: &mut usize|
         -> bool {
            if !self.feasible(&cond) {
                return true;
            }
            let seen = visited.entry(v).or_default();
            if seen
                .iter()
                .any(|prev| prev.iter().all(|item| cond.contains(item)))
            {
                return true;
            }
            *states += 1;
            if *states > MAX_TAINT_STATES {
                return false;
            }
            seen.push(cond.clone());
            worklist.push((v, cond));
            true
        };

        for &(_, v) in &inst.writes {
            if !push(
                v,
                bad_known.clone(),
                &mut visited,
                &mut worklist,
                &mut states,
            ) {
                return false;
            }
        }

        while let Some((v, cond)) = worklist.pop() {
            for &use_site in &uses[v.0 as usize] {
                match use_site {
                    UseSite::Phi {
                        block,
                        pred_idx,
                        phi_idx,
                    } => {
                        let phi = &self.p.blocks[block].phis[phi_idx];
                        if !matches!(phi.reg, Reg::V(_)) {
                            return false;
                        }
                        let q = self.p.blocks[block].preds[pred_idx];
                        let next_cond = if self.rpo[q] >= self.rpo[block] {
                            // Backedge: the condition must be loop-invariant so lane
                            // membership cannot change across iterations.
                            if !cond
                                .iter()
                                .all(|&(b, _)| self.bool_born_before(b, self.rpo[block]))
                            {
                                return false;
                            }
                            extend_known(&cond, [(lanes.reach[block], true)])
                        } else {
                            let branch = match lanes.exits[q] {
                                LaneExit::Branch {
                                    when,
                                    taken,
                                    not_taken,
                                } if taken != not_taken => Some((when, taken == block)),
                                _ => None,
                            };
                            extend_known(
                                &cond,
                                [(lanes.reach[q], true), (lanes.reach[block], true)]
                                    .into_iter()
                                    .chain(branch),
                            )
                        };
                        if let Some(next_cond) = next_cond
                            && !push(
                                phi.value,
                                next_cond,
                                &mut visited,
                                &mut worklist,
                                &mut states,
                            )
                        {
                            return false;
                        }
                    }
                    UseSite::Inst {
                        block: bu,
                        inst_idx: iu,
                    } => {
                        let Some(reach_cond) = extend_known(&cond, [(lanes.reach[bu], true)])
                        else {
                            continue;
                        };
                        if !self.feasible(&reach_cond) {
                            continue;
                        }
                        let u_inst = &self.p.blocks[bu].insts[iu];
                        let guard = lanes.guards[bu][iu];

                        // Disabled lanes skip the instruction and keep destination VGPRs' old values.
                        if let Some(g) = guard
                            && let Some(skip_cond) = extend_known(&reach_cond, [(g, false)])
                            && self.feasible(&skip_cond)
                        {
                            for &(w_reg, w_val) in &u_inst.writes {
                                if matches!(w_reg, Reg::V(_))
                                    && u_inst.read(w_reg) == Some(v)
                                    && !push(
                                        w_val,
                                        skip_cond.clone(),
                                        &mut visited,
                                        &mut worklist,
                                        &mut states,
                                    )
                                {
                                    return false;
                                }
                            }
                        }

                        let Some(exec_cond) = extend_known(&reach_cond, guard.map(|g| (g, true)))
                        else {
                            continue;
                        };
                        if !self.feasible(&exec_cond) {
                            continue;
                        }

                        match &u_inst.fields {
                            Fields::Valu(f)
                                if u_inst.name() == "v_cndmask_b32"
                                    && f.dpp.is_none()
                                    && f.sdwa.is_none() =>
                            {
                                let [clear_op, set_op, mask_op] = &f.src[..] else {
                                    return false;
                                };
                                if op_reads_value(u_inst, mask_op, v) {
                                    return false;
                                }
                                let reads_clear = op_reads_value(u_inst, clear_op, v);
                                let reads_set = op_reads_value(u_inst, set_op, v);
                                if !reads_clear && !reads_set {
                                    continue;
                                }
                                let selected_cond = if reads_clear && reads_set {
                                    Some(exec_cond.clone())
                                } else if let Some(m) = self.cndmask_bool(u_inst, mask_op, lanes) {
                                    extend_known(&exec_cond, [(m, reads_set)])
                                } else if let Some(const_set) = self.cndmask_const(mask_op) {
                                    if (const_set && reads_set) || (!const_set && reads_clear) {
                                        Some(exec_cond.clone())
                                    } else {
                                        None
                                    }
                                } else {
                                    Some(exec_cond.clone())
                                };
                                let Some(selected_cond) = selected_cond else {
                                    continue;
                                };
                                if !self.feasible(&selected_cond) {
                                    continue;
                                }
                                for &(w_reg, w_val) in &u_inst.writes {
                                    match w_reg {
                                        Reg::V(_) => {
                                            if !push(
                                                w_val,
                                                selected_cond.clone(),
                                                &mut visited,
                                                &mut worklist,
                                                &mut states,
                                            ) {
                                                return false;
                                            }
                                        }
                                        _ if w_reg == Reg::S(EXEC)
                                            || w_reg == Reg::S(EXEC + 1)
                                            || !uses[w_val.0 as usize].is_empty() =>
                                        {
                                            return false;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            Fields::Valu(f)
                                if u_inst.name() == "v_swap_b32"
                                    && f.dpp.is_none()
                                    && f.sdwa.is_none() =>
                            {
                                let (
                                    Some(Operand::Vector { reg: dreg, .. }),
                                    Some(&Operand::Vector { reg: sreg, .. }),
                                ) = (f.dst, f.src.first())
                                else {
                                    return false;
                                };
                                if u_inst.read(Reg::V(dreg)) == Some(v)
                                    && let Some(w_val) = u_inst.write(Reg::V(sreg))
                                    && !push(
                                        w_val,
                                        exec_cond.clone(),
                                        &mut visited,
                                        &mut worklist,
                                        &mut states,
                                    )
                                {
                                    return false;
                                }
                                if u_inst.read(Reg::V(sreg)) == Some(v)
                                    && let Some(w_val) = u_inst.write(Reg::V(dreg))
                                    && !push(
                                        w_val,
                                        exec_cond.clone(),
                                        &mut visited,
                                        &mut worklist,
                                        &mut states,
                                    )
                                {
                                    return false;
                                }
                            }
                            Fields::Valu(f) if is_pure_per_lane_valu(u_inst, f) => {
                                let reads_src =
                                    f.src.iter().any(|op| op_reads_value(u_inst, op, v));
                                let partial_dst = is_partial_dst_valu(u_inst.name(), f);
                                for &(w_reg, w_val) in &u_inst.writes {
                                    match w_reg {
                                        Reg::V(_) => {
                                            let keeps_old =
                                                partial_dst && u_inst.read(w_reg) == Some(v);
                                            if (reads_src || keeps_old)
                                                && !push(
                                                    w_val,
                                                    exec_cond.clone(),
                                                    &mut visited,
                                                    &mut worklist,
                                                    &mut states,
                                                )
                                            {
                                                return false;
                                            }
                                        }
                                        _ if reads_src
                                            && (w_reg == Reg::S(EXEC)
                                                || w_reg == Reg::S(EXEC + 1)
                                                || !uses[w_val.0 as usize].is_empty()) =>
                                        {
                                            return false;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            Fields::Image {
                                addr,
                                rsrc,
                                sampler,
                                d16,
                                ..
                            } if !u_inst.name().contains("_store")
                                && !u_inst.name().contains("_atomic_")
                                && ((bu, iu) == site || !*d16) =>
                            {
                                if addr
                                    .iter()
                                    .chain([rsrc])
                                    .chain(sampler)
                                    .any(|op| op_reads_value(u_inst, op, v))
                                {
                                    return false;
                                }
                            }
                            Fields::Buffer {
                                addr,
                                rsrc,
                                soffset,
                                offen,
                                idxen,
                                lds,
                                ..
                            } if !*lds
                                && !u_inst.name().contains("_store_")
                                && !u_inst.name().contains("_atomic_")
                                && ((bu, iu) == site || !u_inst.name().contains("_d16")) =>
                            {
                                let uses_addr =
                                    (*offen || *idxen) && op_reads_value(u_inst, addr, v);
                                if uses_addr
                                    || op_reads_value(u_inst, rsrc, v)
                                    || op_reads_value(u_inst, soffset, v)
                                {
                                    return false;
                                }
                            }
                            _ => return false,
                        }
                    }
                }
            }
        }
        true
    }

    /// Whether a path leads from block `from` to block `to` (to `from`
    /// itself only through a back edge).
    fn reaches(&self, from: usize, to: usize) -> bool {
        let mut seen = vec![false; self.p.blocks.len()];
        let mut stack = self.p.blocks[from].exit.successors();
        while let Some(x) = stack.pop() {
            if x == to {
                return true;
            }
            if std::mem::replace(&mut seen[x], true) {
                continue;
            }
            stack.extend(self.p.blocks[x].exit.successors());
        }
        false
    }

    /// Prove a single-block waterfall at the descriptor access. In each
    /// iteration the access serves exactly the matching lanes, and the
    /// back edge keeps exactly the other lanes active. Otherwise, why not.
    ///
    // shortcut: only the accesses are proven. Any other per-lane work in the
    // loop is taken to run once for each lane, as the access does; a body
    // whose effect adds up over the passes a lane waits through (a counter
    // before the mask is narrowed) would differ. Compilers put nothing but
    // the reads, the compares and the access in these loops.
    fn waterfall(
        &self,
        b: usize,
        accesses: &[usize],
        reads: &[Value],
    ) -> Result<Waterfall, &'static str> {
        const LOOP: &str = "a descriptor read from the first lane outside a waterfall loop";
        const SPLIT: &str = "a waterfall loop nested in another or split over blocks";
        const SELECT: &str = "a waterfall loop whose compares do not select the access's lanes";
        const RETRY: &str = "a waterfall loop whose retry does not keep the lanes still waiting";
        let access = accesses[0];
        let block = &self.p.blocks[b];
        let lanes = self.lanes.ok_or(LOOP)?;
        let exit = block.insts.last().ok_or(LOOP)?;
        if !matches!(block.exit, Exit::Branch { taken, not_taken } if taken == b && not_taken != b)
            || exit.name() != "s_cbranch_execnz"
        {
            return Err(LOOP);
        }
        let mask = |low: Option<Value>, high: Option<Value>| {
            let high = if self.p.wave64 { Some(high?) } else { None };
            lanes.mask_bits(low?, high)
        };
        let exec = |inst: &Inst| mask(inst.read(Reg::S(EXEC)), inst.read(Reg::S(EXEC + 1)));
        // The reads belong to this loop's own block: a loop around this one
        // (one per operand, as LLVM nests them for a per-lane image and
        // sampler) is not followed.
        if reads
            .iter()
            .any(|v| self.defs[v.0 as usize].is_none_or(|at| at.0 != b))
        {
            return Err(SPLIT);
        }
        let selected = exec(&block.insts[access]).ok_or(SELECT)?;
        if accesses
            .iter()
            .any(|&i| exec(&block.insts[i]) != Some(selected))
        {
            return Err(SELECT);
        }
        let retry = exec(exit).ok_or(RETRY)?;
        let first = self.defs[reads[0].0 as usize].ok_or(LOOP)?;
        let remaining = exec(self.inst(first)).ok_or(SELECT)?;
        if reads.iter().any(|v| {
            self.defs[v.0 as usize]
                .is_none_or(|at| at.1 >= access || exec(self.inst(at)) != Some(remaining))
        }) {
            return Err(SELECT);
        }
        let reach = (lanes.reach[b], true);
        let loop_known = [reach, (remaining, true)];
        let meet = |conditions: &[(Bool, bool)]| self.feasible(conditions);
        let mut out = Waterfall::default();
        let mut required = vec![(remaining, true)];
        for (i, inst) in block.insts[..access].iter().enumerate() {
            let Fields::Valu(f) = &inst.fields else {
                continue;
            };
            let Some(ty) = inst
                .name()
                .strip_prefix("v_cmp_eq_")
                .or(inst.name().strip_prefix("v_cmpx_eq_"))
            else {
                continue;
            };
            let dwords = match ty {
                "u32" | "i32" => 1,
                "u64" | "i64" => 2,
                _ => continue,
            };
            if f.abs != 0 || f.neg != 0 || f.sdwa.is_some() || f.dpp.is_some() {
                continue;
            }
            // The first lane's dwords on one side, the lane's own on the
            // other.
            let pair = |first: &Operand, own: &Operand| -> Option<Vec<Value>> {
                let (&Operand::Scalar { code, .. }, &Operand::Vector { reg, .. }) = (first, own)
                else {
                    return None;
                };
                (0..dwords)
                    .map(|k| {
                        let Term::Value(x) = self.origin(inst.read(Reg::S(code + k))?) else {
                            return None;
                        };
                        let lane = inst.read(Reg::V(reg + k))?;
                        let (_, read) = self.defs[x.0 as usize]?;
                        let same = reads.contains(&x)
                            && read < i
                            && match (
                                self.lane_value(
                                    Term::Value(self.first_lane_source(x)?),
                                    &loop_known,
                                ),
                                self.lane_value(Term::Value(lane), &loop_known),
                            ) {
                                (Ok(lhs), Ok(rhs)) => lhs == rhs,
                                _ => false,
                            };
                        same.then_some(x)
                    })
                    .collect()
            };
            let (Some(a), Some(c)) = (f.src.first(), f.src.get(1)) else {
                continue;
            };
            if let Some(xs) = pair(a, c).or_else(|| pair(c, a)) {
                let code = if inst.name().starts_with("v_cmpx_") {
                    EXEC
                } else if let Some(Operand::Scalar { code, .. }) = f.dst {
                    code
                } else {
                    continue;
                };
                let Some(matched) = mask(inst.write(Reg::S(code)), inst.write(Reg::S(code + 1))) else {
                    continue;
                };
                if meet(&[reach, (selected, true), (matched, false)]) {
                    continue;
                }
                required.push((matched, true));
                out.compares.push((b, i));
                for x in xs {
                    if !out.reads.contains(&x) {
                        out.reads.push(x);
                    }
                }
            }
        }
        if reads.iter().any(|v| !out.reads.contains(v))
            || meet(&[reach, (selected, true), (remaining, false)])
        {
            return Err(SELECT);
        }
        // The access must take every match.
        required.extend([reach, (selected, false)]);
        if meet(&required) {
            return Err(SELECT);
        }
        // The retry must keep exactly the lanes that were eligible but did
        // not perform the access.
        if meet(&[reach, (retry, true), (remaining, false)])
            || meet(&[reach, (retry, true), (selected, true)])
            || meet(&[reach, (remaining, true), (selected, false), (retry, false)])
        {
            return Err(RETRY);
        }
        Ok(out)
    }

    /// Consecutive dwords of one scalar load.
    fn loaded(&self, words: &[Term]) -> Option<Source> {
        let Term::Value(v0) = words[0] else {
            return None;
        };
        let load = self.defs[v0.0 as usize]?;
        let inst = self.inst(load);
        if !inst.name().starts_with("s_load_") {
            return None;
        }
        let first = inst.writes.iter().position(|w| w.1 == v0)?;
        let same = words.iter().enumerate().all(|(k, w)| {
            inst.writes
                .get(first + k)
                .is_some_and(|x| *w == Term::Value(x.1))
        });
        same.then_some(Source::Loaded {
            load,
            first: first as u8,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CmpType {
    U16,
    I16,
    F16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CmpOperand {
    words: [Term; 2],
    abs: bool,
    neg: bool,
    opsel: bool,
    sdwa: Option<(u8, bool)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CanonPred {
    Const(bool),
    IntLt(CmpOperand, CmpOperand),
    IntEq(CmpOperand, CmpOperand),
    FloatLt(CmpOperand, CmpOperand),
    FloatLe(CmpOperand, CmpOperand),
    FloatEq(CmpOperand, CmpOperand),
    FloatLg(CmpOperand, CmpOperand),
    FloatOrd(CmpOperand, CmpOperand),
    FloatClass(CmpOperand, CmpOperand),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CanonKey {
    ty: CmpType,
    pred: CanonPred,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Moved {
    Const(u32),
    Value(Value),
}

impl Tracer<'_> {
    /// For `v_readlane_b32` of a constant lane: the SGPR spilled to that
    /// lane, found by following the VGPR's `v_writelane_b32` writes.
    fn restored(&self, inst: &Inst) -> Option<Moved> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        if inst.name() != "v_readlane_b32" {
            return None;
        }
        let [Operand::Vector { reg, .. }, lane] = v.src[..] else {
            return None;
        };
        self.spilled(inst.read(Reg::V(reg))?, lane_of(lane)?, &mut Vec::new())
            .ok()
            .flatten()
    }

    /// What lane `lane` of VGPR value `v` holds: `Ok(None)` for a phi in
    /// `seen` (already being resolved), `Err` if not a spilled SGPR.
    fn spilled(&self, v: Value, lane: u32, seen: &mut Vec<Value>) -> Result<Option<Moved>, ()> {
        if let Some((b, i)) = self.phis[v.0 as usize] {
            if seen.contains(&v) {
                return Ok(None);
            }
            seen.push(v);
            let mut found = None;
            for &a in &self.p.blocks[b].phis[i].args {
                if let Some(m) = self.spilled(a, lane, seen)? {
                    if found.is_some_and(|f| f != m) {
                        return Err(());
                    }
                    found = Some(m);
                }
            }
            return Ok(found);
        }
        let inst = self.def(v).ok_or(())?;
        let Fields::Valu(w) = &inst.fields else {
            return Err(());
        };
        let (Some(Operand::Vector { reg, .. }), [value, at]) = (w.dst, &w.src[..]) else {
            return Err(());
        };
        if inst.name() != "v_writelane_b32" {
            return Err(());
        }
        if lane_of(*at).ok_or(())? != lane {
            return self.spilled(inst.read(Reg::V(reg)).ok_or(())?, lane, seen);
        }
        match *value {
            Operand::Scalar { code, .. } if code != NULL => {
                inst.read(Reg::S(code)).map(Moved::Value)
            }
            Operand::Literal(x) => Some(Moved::Const(x)),
            Operand::Const(c) => inline(c).map(Moved::Const),
            _ => None,
        }
        .map(Some)
        .ok_or(())
    }
}

/// A lane select given as a constant.
fn lane_of(op: Operand) -> Option<u32> {
    match op {
        Operand::Const(c) => inline(c),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

/// Extends `base` with `extra`, returning `None` on a trivial contradiction.
fn extend_known(
    base: &[(Bool, bool)],
    extra: impl IntoIterator<Item = (Bool, bool)>,
) -> Option<Known> {
    let mut out = base.to_vec();
    for (b, truth) in extra {
        if b == Bool::TRUE {
            if !truth {
                return None;
            }
            continue;
        }
        if b == Bool::FALSE {
            if truth {
                return None;
            }
            continue;
        }
        if out.contains(&(b, !truth)) {
            return None;
        }
        if !out.contains(&(b, truth)) {
            out.push((b, truth));
        }
    }
    Some(out)
}

/// Whether `op` of `inst` reads `v` through any of its VGPR dwords.
fn op_reads_value(inst: &Inst, op: &Operand, v: Value) -> bool {
    match *op {
        Operand::Vector { reg, dwords } => {
            (0..dwords).any(|k| inst.read(Reg::V(reg + k)) == Some(v))
        }
        _ => false,
    }
}

/// Whether a VALU instruction is pure per-lane (does not read across lanes
/// or affect masks/EXEC).
fn is_pure_per_lane_valu(inst: &Inst, f: &crate::gfx10::Valu) -> bool {
    if f.dpp.is_some() {
        return false;
    }
    let name = inst.name();
    !name.is_empty()
        && !name.starts_with("v_cmp")
        && !name.starts_with("v_cmpx")
        && !name.starts_with("v_permlane")
        && !name.starts_with("v_interp")
        && !name.starts_with("v_movrel")
        && !name.starts_with("v_swaprel")
        && !name.starts_with("v_mbcnt")
        && !matches!(
            name,
            "v_readfirstlane_b32" | "v_readlane_b32" | "v_writelane_b32"
        )
}

fn is_partial_dst_valu(name: &str, f: &crate::gfx10::Valu) -> bool {
    f.sdwa.is_some_and(|s| s.dst_sel != 6)
        || name.ends_with("_f16")
        || name.ends_with("_u16")
        || name.ends_with("_i16")
        || name.ends_with("_b16")
        || matches!(
            name,
            "v_cvt_f16_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" | "v_cvt_pk_u8_f32"
        )
}

/// What dword `k` of a move's result copies.
fn copy_source(inst: &Inst, k: usize) -> Option<Moved> {
    let Fields::Sop { src, imm, .. } = &inst.fields else {
        return None;
    };
    if inst.name() == "s_movk_i32" {
        return Some(Moved::Const(i32::from((*imm)? as i16) as u32));
    }
    let src = src.first()?;
    match (inst.name(), *src) {
        ("s_mov_b32" | "s_mov_b64", Operand::Scalar { code, .. }) if code != NULL => {
            inst.read(Reg::S(code + k as u8)).map(Moved::Value)
        }
        ("s_mov_b32" | "s_mov_b64", Operand::Scalar { .. }) => Some(Moved::Const(0)),
        ("s_mov_b32", Operand::Literal(x)) => Some(Moved::Const(x)),
        ("s_mov_b32", Operand::Const(c)) => inline(c).map(Moved::Const),
        // shortcut: only the one-operand ways the compiler makes a constant
        // that has no inline form; fold more if descriptors need it.
        ("s_brev_b32", Operand::Const(c)) => inline(c).map(|x| Moved::Const(x.reverse_bits())),
        ("s_not_b32", Operand::Const(c)) => inline(c).map(|x| Moved::Const(!x)),
        // 64-bit inline integers are sign-extended.
        ("s_mov_b64", Operand::Const(c @ 128..=208)) => {
            let low = inline(c)?;
            Some(Moved::Const(if k == 0 {
                low
            } else {
                ((low as i32) >> 31) as u32
            }))
        }
        _ => None,
    }
}

/// The 32-bit value of an inline constant operand.
pub(crate) fn inline(code: u8) -> Option<u32> {
    let float = |f: f32| Some(f.to_bits());
    match code {
        128..=192 => Some(u32::from(code) - 128),
        193..=208 => Some((192 - i32::from(code)) as u32),
        240 => float(0.5),
        241 => float(-0.5),
        242 => float(1.0),
        243 => float(-1.0),
        244 => float(2.0),
        245 => float(-2.0),
        246 => float(4.0),
        247 => float(-4.0),
        248 => Some(0x3E22_F983),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::native::RootType::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn track_words(words: &[u32], layout: &[RootParam], stage: HwStage) -> Resources {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        track(&p, layout, stage, lanes(&p).as_ref()).unwrap()
    }

    /// `p` as one lane runs it, if it structures.
    fn lanes(p: &Program) -> Option<Structured> {
        crate::structure::structure(p).ok()
    }

    fn param(root: RootType, user_data: u8, sgprs: u32) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot: 0,
        }
    }

    fn at(param: usize, offset: u32) -> Address {
        Address {
            param,
            offset,
            dynamic: None,
        }
    }

    #[test]
    fn table_descriptors_are_loaded_from_their_range() {
        let r = track_words(
            &[
                0xBE82_0309, // s_mov_b32 s2, s9
                0xBE89_0380, // s_mov_b32 s9, 0
                0xBE83_0380, // s_mov_b32 s3, 0
                0xF408_0204,
                0xFA00_0000, // s_load_dwordx4 s[8:11], s[8:9], null
                0xF40C_0001,
                0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
                0xF080_0F08,
                0x0040_0002, // image_sample v[0:3], v[2:3], s[0:7], s[8:11]
                S_ENDPGM,
            ],
            &[param(Table, 8, 1), param(Table, 9, 1)],
            HwStage::Ps,
        );
        assert_eq!(r.scalar, [((0, 3), at(0, 0)), ((0, 4), at(1, 0))]);
        let roles: Vec<_> = r.descriptors.iter().map(|d| (d.role, &d.source)).collect();
        assert_eq!(
            roles,
            [
                (
                    Role::Resource,
                    &Source::Loaded {
                        load: (0, 4),
                        first: 0
                    }
                ),
                (
                    Role::Sampler,
                    &Source::Loaded {
                        load: (0, 3),
                        first: 0
                    }
                ),
            ]
        );
    }

    #[test]
    fn root_descriptors_are_built_in_code() {
        let r = track_words(
            &[
                0x8801_FF0B,
                0x0010_0000, // s_or_b32 s1, s11, 0x100000
                0xBE80_030A, // s_mov_b32 s0, s10
                0xBE82_03C1, // s_mov_b32 s2, -1
                0xBE83_03FF,
                0x1101_4FAC, // s_mov_b32 s3, 0x11014fac
                0xF430_0000,
                0xFA00_0000, // s_buffer_load_dwordx16 s[0:15], s[0:3], null
                S_ENDPGM,
            ],
            &[param(Descriptor, 10, 2)],
            HwStage::Vs,
        );
        assert_eq!(
            r.descriptors[0].source,
            Source::Built {
                address: at(0, 0),
                bits: 0x10_0000,
                words: [u32::MAX, 0x1101_4FAC]
            }
        );
        assert_eq!(
            r.params.iter().map(|p| (p.1, p.2)).collect::<Vec<_>>(),
            [(0, 0), (0, 1)]
        );
    }

    #[test]
    fn heap_indices_from_root_constants_in_a_merged_stage() {
        // User data starts at s8: the constant (index 6) is s14, the heap
        // (index 10) s18.
        let r = track_words(
            &[
                0x8F08_850E, // s_lshl_b32 s8, s14, 5
                0xBE93_0380, // s_mov_b32 s19, 0
                0xF408_0209,
                0x1000_0000, // s_load_dwordx4 s[8:11], s[18:19], s8
                0xE000_2000,
                0x8002_0400, // buffer_load_format_x v4, v0, s[8:11], 0 idxen
                S_ENDPGM,
            ],
            &[param(Constants, 6, 1), param(ResourceHeap, 10, 1)],
            HwStage::Ms,
        );
        let index = r.params.iter().find(|p| p.1 == 0).unwrap().0;
        let shift = r.scalar[0].1.dynamic.unwrap();
        assert_eq!(r.scalar[0].1.param, 1);
        // The computed term is the shifted constant, not the constant itself.
        assert_ne!(shift, index);
        assert_eq!(
            r.descriptors[0].source,
            Source::Loaded {
                load: (0, 2),
                first: 0
            }
        );
    }

    #[test]
    fn spilled_sgprs_are_followed_through_vgpr_lanes() {
        let r = track_words(
            &[
                0xD761_0000,
                0x0001_0609, // v_writelane_b32 v0, s9, 3
                0xD760_0002,
                0x0001_0700, // v_readlane_b32 s2, v0, 3
                0xBE83_0380, // s_mov_b32 s3, 0
                0xF40C_0001,
                0xFA00_0020, // s_load_dwordx8 s[0:7], s[2:3], 0x20
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        assert_eq!(r.scalar, [((0, 3), at(0, 0x20))]);
    }

    #[test]
    fn descriptors_joined_by_control_flow_are_traced_per_edge() {
        let r = track_words(
            &[
                0xBE82_0309, // s_mov_b32 s2, s9
                0xBE83_0380, // s_mov_b32 s3, 0
                0xBF06_8001, // s_cmp_eq_u32 s1, 0
                0xBF85_0003, // s_cbranch_scc1 -> 0x1c
                0xF408_0001,
                0xFA00_0000, // s_load_dwordx4 s[0:3], s[2:3], null
                0xBF82_0002, // s_branch -> 0x24
                0xF408_0001,
                0xFA00_0010, // 0x1c: s_load_dwordx4 s[0:3], s[2:3], 0x10
                0xF420_0100,
                0xFA00_0000, // 0x24: s_buffer_load_dword s4, s[0:3], null
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        let Source::Phi { args, .. } = &r.descriptors[0].source else {
            panic!("{:?}", r.descriptors)
        };
        let offsets: Vec<_> = args
            .iter()
            .map(|a| match a {
                Source::Loaded { load, .. } => {
                    r.scalar.iter().find(|s| s.0 == *load).unwrap().1.offset
                }
                a => panic!("{a:?}"),
            })
            .collect();
        assert_eq!(offsets.len(), 2);
        assert!(
            offsets.contains(&0) && offsets.contains(&0x10),
            "{offsets:?}"
        );
    }

    #[test]
    fn descriptors_selected_by_scc_are_traced_per_outcome() {
        // Two buffer descriptors loaded from the table at s9 and one
        // picked with s_cselect_b64 per half, as the compiler makes a
        // conditional choice of resources without a branch.
        let r = track_words(
            &[
                0xBE8C_0309, // s_mov_b32 s12, s9
                0xBE8D_0380, // s_mov_b32 s13, 0
                0xF408_0006,
                0xFA00_0000, // s_load_dwordx4 s[0:3], s[12:13], null
                0xF408_0106,
                0xFA00_0010, // s_load_dwordx4 s[4:7], s[12:13], 0x10
                0xBF06_800E, // s_cmp_eq_u32 s14, 0
                0x8588_0400, // s_cselect_b64 s[8:9], s[0:1], s[4:5]
                0x858A_0602, // s_cselect_b64 s[10:11], s[2:3], s[6:7]
                0xF420_0404,
                0xFA00_0000, // s_buffer_load_dword s16, s[8:11], null
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        let Source::Select { args, .. } = &r.descriptors[0].source else {
            panic!("{:?}", r.descriptors)
        };
        // SCC set picks the first operand.
        assert_eq!(
            args[..],
            [
                Source::Loaded {
                    load: (0, 2),
                    first: 0
                },
                Source::Loaded {
                    load: (0, 3),
                    first: 0
                },
            ]
        );
        assert_eq!(r.scalar, [((0, 2), at(0, 0)), ((0, 3), at(0, 0x10))]);
    }

    #[test]
    fn a_selection_that_mixes_two_descriptors_fails() {
        // The high half picked from the first load whatever SCC is: when
        // clear, the descriptor is half of one load and half of the other.
        let code: Vec<u8> = [
            0xBE8C_0309, // s_mov_b32 s12, s9
            0xBE8D_0380, // s_mov_b32 s13, 0
            0xF408_0006,
            0xFA00_0000, // s_load_dwordx4 s[0:3], s[12:13], null
            0xF408_0106,
            0xFA00_0010, // s_load_dwordx4 s[4:7], s[12:13], 0x10
            0xBF06_800E, // s_cmp_eq_u32 s14, 0
            0x8588_0400, // s_cselect_b64 s[8:9], s[0:1], s[4:5]
            0x858A_0202, // s_cselect_b64 s[10:11], s[2:3], s[2:3]
            0xF420_0404,
            0xFA00_0000, // s_buffer_load_dword s16, s[8:11], null
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w: &u32| w.to_le_bytes())
        .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let err = track(&p, &[param(Table, 9, 1)], HwStage::Ps, lanes(&p).as_ref()).unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 36,
                what: "a descriptor changed after loading"
            }
        );
    }

    #[test]
    fn untraceable_addresses_fail_with_a_reason() {
        let code: Vec<u8> = [0xF408_0001, 0xFA00_0000, S_ENDPGM] // s_load_dwordx4 s[0:3], s[2:3]
            .iter()
            .flat_map(|w: &u32| w.to_le_bytes())
            .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let err = track(&p, &[param(Table, 9, 1)], HwStage::Ps, lanes(&p).as_ref()).unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 0,
                what: "an address in an SGPR that is no root parameter"
            }
        );
    }

    /// A waterfall loop over an image descriptor each lane loads from the
    /// table at s0 (wave32, assembled with llvm-mc for gfx1030): lane i
    /// reads entry i with two `global_load_dwordx4`, the loop reads the
    /// first active lane's descriptor into s[12:19], samples with the lanes
    /// whose descriptor it is, and repeats for the others. The sampler comes
    /// from the table at s1; the result goes to LDS so the sample is needed.
    pub(crate) const WATERFALL_FETCHED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0x3402_0085, // v_lshlrev_b32 v1, 5, v0
        0xDC38_8000,
        0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38_8010,
        0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7E18_0504, // 0x3c: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0x3c
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        0xBF81_0000, // s_endpgm
    ];

    /// [`WATERFALL_FETCHED_WAVE32`] in wave64: 64-bit lane masks.
    pub(crate) const WATERFALL_FETCHED_WAVE64: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0x3402_0085, // v_lshlrev_b32 v1, 5, v0
        0xDC38_8000,
        0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38_8010,
        0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0x7E18_0504, // 0x3c: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0016,
        0x0002_080C, // v_cmp_eq_u64_e64 s[22:23], s[12:13], v[4:5]
        0xD4E2_0018,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s[24:25], s[14:15], v[6:7]
        0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
        0xD4E2_0018,
        0x0002_1010, // v_cmp_eq_u64_e64 s[24:25], s[16:17], v[8:9]
        0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
        0xD4E2_0018,
        0x0002_1412, // v_cmp_eq_u64_e64 s[24:25], s[18:19], v[10:11]
        0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
        0xBE96_2416, // s_and_saveexec_b64 s[22:23], s[22:23]
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x89FE_167E, // s_xor_b64 exec, exec, s[22:23]
        0xBF89_FFE6, // s_cbranch_execnz -> 0x3c
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        0xBF81_0000, // s_endpgm
    ];

    /// The same waterfall over one of two image descriptors loaded from the
    /// table at s0, picked per lane by `v_cndmask_b32` (lanes below 16 take
    /// the first).
    pub(crate) const WATERFALL_PICKED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E08_0218, // v_mov_b32 v4, s24
        0x0208_0820, // v_cndmask_b32 v4, s32, v4, vcc_lo
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x020A_0A21, // v_cndmask_b32 v5, s33, v5, vcc_lo
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x020C_0C22, // v_cndmask_b32 v6, s34, v6, vcc_lo
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x020E_0E23, // v_cndmask_b32 v7, s35, v7, vcc_lo
        0x7E10_021C, // v_mov_b32 v8, s28
        0x0210_1024, // v_cndmask_b32 v8, s36, v8, vcc_lo
        0x7E12_021D, // v_mov_b32 v9, s29
        0x0212_1225, // v_cndmask_b32 v9, s37, v9, vcc_lo
        0x7E14_021E, // v_mov_b32 v10, s30
        0x0214_1426, // v_cndmask_b32 v10, s38, v10, vcc_lo
        0x7E16_021F, // v_mov_b32 v11, s31
        0x0216_1627, // v_cndmask_b32 v11, s39, v11, vcc_lo
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7E18_0504, // 0x7c: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0x7c
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        0xBF81_0000, // s_endpgm
    ];

    /// The waterfall loop of [`WATERFALL_PICKED_WAVE32`] over one of two
    /// image descriptors loaded from the table at s0, chosen by control flow
    /// as the compiler lowers an if/else on a per-lane condition: lanes below
    /// 16 move the first into v[4:11], the others the second, each arm
    /// skipped when no lane takes it. The sample goes to the buffer at dword
    /// 16 of the table, at lane * 4.
    pub(crate) const WATERFALL_JOINED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0x8914_147E, // s_xor_b32 s20, exec_lo, s20
        0xBF88_0008, // s_cbranch_execz -> 0x70
        0x7E08_0218, // v_mov_b32 v4, s24
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x7E10_021C, // v_mov_b32 v8, s28
        0x7E12_021D, // v_mov_b32 v9, s29
        0x7E14_021E, // v_mov_b32 v10, s30
        0x7E16_021F, // v_mov_b32 v11, s31
        0xBE94_3D14, // 0x70: s_or_saveexec_b32 s20, s20
        0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
        0xBF88_0008, // s_cbranch_execz -> 0x9c
        0x7E08_0220, // v_mov_b32 v4, s32
        0x7E0A_0221, // v_mov_b32 v5, s33
        0x7E0C_0222, // v_mov_b32 v6, s34
        0x7E0E_0223, // v_mov_b32 v7, s35
        0x7E10_0224, // v_mov_b32 v8, s36
        0x7E12_0225, // v_mov_b32 v9, s37
        0x7E14_0226, // v_mov_b32 v10, s38
        0x7E16_0227, // v_mov_b32 v11, s39
        0x887E_147E, // 0x9c: s_or_b32 exec_lo, exec_lo, s20
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0x7E18_0504, // 0xa4: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0xa4
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    /// The console compiler's waterfall (Grand Theft Auto VI's ray-tracing
    /// resolve and others): each lane holds its entry's byte offset in the
    /// table at s0 (v7 = (lane & 1) * 32); the loop reads the first active
    /// lane's offset, narrows EXEC to the lanes with the same one
    /// (`v_cmpx_eq_u32`), loads that entry with `s_load_dwordx8` and samples,
    /// then drops those lanes (`s_andn2_wrexec_b32`). The sample goes to the
    /// buffer at dword 16 of the table, at lane * 4.
    pub(crate) const WATERFALL_INDEXED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x360E_0081, // v_and_b32 v7, 1, v0
        0x340E_0E85, // v_lshlrev_b32 v7, 5, v7
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0xBE95_037E, // s_mov_b32 s21, exec_lo
        0x7E28_0507, // 0x40: v_readfirstlane_b32 s20, v7
        0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
        0xF40C_0302,
        0x2800_0000, // s_load_dwordx8 s[12:19], s[4:5], s20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
        0xBF89_FFF6, // s_cbranch_execnz -> 0x40
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    #[test]
    fn the_console_waterfall_over_a_first_lane_offset_is_one() {
        let r = waterfall(WATERFALL_INDEXED_WAVE32, false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| d.role == Role::Resource)
            .unwrap();
        let Source::Loaded { load, first: 0 } = image.source else {
            panic!("{:?}", image.source)
        };
        let (_, address) = r.scalar.iter().find(|s| s.0 == load).unwrap();
        assert!(address.dynamic.is_some(), "{address:?}");
        assert_eq!(r.waterfall.compares.len(), 1);
        assert_eq!(r.waterfall.reads.len(), 1);
        assert_eq!(r.waterfall.loads, [load]);
    }

    /// [`WATERFALL_JOINED_WAVE32`] with other sources on each path: lanes
    /// below 16 load table entry (lane & 1) with two `global_load_dwordx4`,
    /// the others pick per lane with `v_cndmask_b32` (lanes below 24 the
    /// first entry, the rest the second).
    pub(crate) const WATERFALL_JOINED_MIXED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x3602_0081, // v_and_b32 v1, 1, v0
        0x3402_0285, // v_lshlrev_b32 v1, 5, v1
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0x8914_147E, // s_xor_b32 s20, exec_lo, s20
        0xBF88_0004, // s_cbranch_execz -> 0x68
        0xDC38_8000,
        0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38_8010,
        0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xBE94_3D14, // 0x68: s_or_saveexec_b32 s20, s20
        0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
        0xBF88_0011, // s_cbranch_execz -> 0xb8
        0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
        0x7E08_0218, // v_mov_b32 v4, s24
        0x0208_0820, // v_cndmask_b32 v4, s32, v4, vcc_lo
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x020A_0A21, // v_cndmask_b32 v5, s33, v5, vcc_lo
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x020C_0C22, // v_cndmask_b32 v6, s34, v6, vcc_lo
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x020E_0E23, // v_cndmask_b32 v7, s35, v7, vcc_lo
        0x7E10_021C, // v_mov_b32 v8, s28
        0x0210_1024, // v_cndmask_b32 v8, s36, v8, vcc_lo
        0x7E12_021D, // v_mov_b32 v9, s29
        0x0212_1225, // v_cndmask_b32 v9, s37, v9, vcc_lo
        0x7E14_021E, // v_mov_b32 v10, s30
        0x0214_1426, // v_cndmask_b32 v10, s38, v10, vcc_lo
        0x7E16_021F, // v_mov_b32 v11, s31
        0x0216_1627, // v_cndmask_b32 v11, s39, v11, vcc_lo
        0x887E_147E, // 0xb8: s_or_b32 exec_lo, exec_lo, s20
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E18_0504, // 0xc4: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0xc4
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    /// A waterfall loop whose descriptor changes on each pass: v[4:11] holds
    /// the first table entry on entry and the second after every pass, so
    /// the first lane's dwords are phis of the loop header.
    pub(crate) const WATERFALL_LOOPED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7E08_0218, // v_mov_b32 v4, s24
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x7E10_021C, // v_mov_b32 v8, s28
        0x7E12_021D, // v_mov_b32 v9, s29
        0x7E14_021E, // v_mov_b32 v10, s30
        0x7E16_021F, // v_mov_b32 v11, s31
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0x7E18_0504, // 0x64: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0x7E08_0220, // v_mov_b32 v4, s32
        0x7E0A_0221, // v_mov_b32 v5, s33
        0x7E0C_0222, // v_mov_b32 v6, s34
        0x7E0E_0223, // v_mov_b32 v7, s35
        0x7E10_0224, // v_mov_b32 v8, s36
        0x7E12_0225, // v_mov_b32 v9, s37
        0x7E14_0226, // v_mov_b32 v10, s38
        0x7E16_0227, // v_mov_b32 v11, s39
        0xBF89_FFDE, // s_cbranch_execnz -> 0x64
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    /// A waterfall loop over one of two buffer descriptors (four dwords)
    /// loaded from the table at s0, chosen by nested if/else: lanes below 16
    /// move the first into v[4:7], lanes from 16 to 23 the second, and the
    /// others keep what v[4:7] held. Only lanes below 24 (the same mask)
    /// run the loop, which loads dword `lane` of the chosen buffer, and
    /// store it to the buffer at dword 16 of the table.
    pub(crate) const WATERFALL_JOINED_BUFFER_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0602,
        0xFA00_0000, // s_load_dwordx4 s[24:27], s[4:5], 0x0
        0xF408_0802,
        0xFA00_0020, // s_load_dwordx4 s[32:35], s[4:5], 0x20
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xD4C4_001E,
        0x0002_0098, // v_cmp_gt_u32_e64 s30, 24, v0
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0x8914_147E, // s_xor_b32 s20, exec_lo, s20
        0xBF88_0004, // s_cbranch_execz -> 0x50
        0x7E08_0218, // 0x40: v_mov_b32 v4, s24
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x7E0E_021B, // v_mov_b32 v7, s27
        0xBE94_3D14, // 0x50: s_or_saveexec_b32 s20, s20
        0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
        0xBF88_0007, // s_cbranch_execz -> 0x78
        0xBE9F_3C1E, // s_and_saveexec_b32 s31, s30
        0xBF88_0004, // s_cbranch_execz -> 0x74
        0x7E08_0220, // 0x64: v_mov_b32 v4, s32
        0x7E0A_0221, // v_mov_b32 v5, s33
        0x7E0C_0222, // v_mov_b32 v6, s34
        0x7E0E_0223, // v_mov_b32 v7, s35
        0x887E_1F7E, // 0x74: s_or_b32 exec_lo, exec_lo, s31
        0x887E_147E, // 0x78: s_or_b32 exec_lo, exec_lo, s20
        0xBE97_3C1E, // s_and_saveexec_b32 s23, s30
        0xBF88_0013, // s_cbranch_execz -> 0xd0
        0xBE9D_037E, // s_mov_b32 s29, exec_lo
        0x7E18_0504, // 0x88: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xE030_1000,
        0x8003_0C10, // buffer_load_dword v12, v16, s[12:15], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFF1, // s_cbranch_execnz -> 0x88
        0xBEFE_031D, // s_mov_b32 exec_lo, s29
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0x887E_177E, // 0xd0: s_or_b32 exec_lo, exec_lo, s23
        0xBF81_0000, // s_endpgm
    ];

    fn waterfall(words: &[u32], wave64: bool) -> Result<Resources, ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, wave64, 256).unwrap();
        let layout = [param(Table, 0, 1), param(Table, 1, 1)];
        track(&p, &layout, HwStage::Cs, lanes(&p).as_ref())
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_per_lane_loads_are_traced_to_them() {
        for (words, wave64) in [
            (WATERFALL_FETCHED_WAVE32, false),
            (WATERFALL_FETCHED_WAVE64, true),
        ] {
            let r = waterfall(words, wave64).unwrap();
            let image = r
                .descriptors
                .iter()
                .find(|d| d.role == Role::Resource)
                .unwrap();
            let Source::Fetched { load, first: 0 } = image.source else {
                panic!("{:?}", image.source)
            };
            // The first of the two loads: entry v0 * 32 of the table at s0.
            let a = r.scalar.iter().find(|s| s.0 == load).unwrap().1;
            assert_eq!((a.param, a.offset), (0, 0));
            assert!(a.dynamic.is_some());
            // The sampler is the same for every lane.
            let sampler = r.descriptors.iter().find(|d| d.role == Role::Sampler);
            assert!(matches!(
                sampler.unwrap().source,
                Source::Loaded { first: 0, .. }
            ));
            // Four 64-bit compares of the first lane's dwords with the lane's
            // own; the eight reads become the lane's own dwords.
            assert_eq!(r.waterfall.compares.len(), 4);
            assert_eq!(r.waterfall.reads.len(), 8);
        }
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_a_per_lane_choice_are_traced_per_lane() {
        let r = waterfall(WATERFALL_PICKED_WAVE32, false).unwrap();
        let image = &r.descriptors[0];
        let Source::Lanes { args, .. } = &image.source else {
            panic!("{:?}", image.source)
        };
        // Lanes with their bit set take the first load (offset 0), the
        // others the second (0x20).
        let offsets: Vec<u32> = args
            .iter()
            .map(|a| match *a {
                Source::Loaded { load, first: 0 } => {
                    r.scalar.iter().find(|s| s.0 == load).unwrap().1.offset
                }
                ref a => panic!("{a:?}"),
            })
            .collect();
        assert_eq!(offsets, [0, 0x20]);
        assert_eq!(r.waterfall.compares.len(), 4);
    }

    #[test]
    fn a_first_lane_read_outside_a_loop_is_no_waterfall() {
        // The waterfall without its back edge: the s_cbranch_execnz is an
        // s_nop, so lanes the first one does not match never get a pass.
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        let back = words.iter().position(|&w| w == 0xBF89_FFE6).unwrap();
        words[back] = 0xBF80_0000;
        let err = waterfall(&words, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane outside a waterfall loop",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// Why resource tracking refuses `words`.
    fn refusal(words: &[u32], wave64: bool) -> &'static str {
        match waterfall(words, wave64) {
            Err(ShaderError::Unsupported { what, .. }) => what,
            other => panic!("not refused: {other:?}"),
        }
    }

    const NO_LOOP: &str = "a descriptor read from the first lane outside a waterfall loop";
    const NOT_SELECTED: &str = "a waterfall loop whose compares do not select the access's lanes";
    const BAD_RETRY: &str = "a waterfall loop whose retry does not keep the lanes still waiting";
    const SPLIT: &str = "a waterfall loop nested in another or split over blocks";

    /// The index of `word` in `words`.
    fn word_at(words: &[u32], word: u32) -> usize {
        words.iter().position(|&w| w == word).unwrap()
    }

    #[test]
    fn a_first_lane_read_without_its_compare_is_no_waterfall() {
        // The loop without the compare of dwords 0 and 1: nothing makes
        // the first lane's dwords the lane's own there.
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        let cmp = word_at(&words, 0x0002_080C);
        words[cmp - 1..=cmp].copy_from_slice(&[0xBF80_0000, 0xBF80_0000]);
        assert_eq!(refusal(&words, false), NOT_SELECTED);
    }

    #[test]
    fn a_compare_that_does_not_mask_the_access_is_no_waterfall() {
        // The compares are all there, but their result never narrows EXEC:
        // every lane samples with the first lane's descriptor (reported by
        // ifBars on upstream PR #566).
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let save = word_at(&words, 0xBE95_3C15);
        words[save] = 0xBE95_037E; // s_mov_b32 s21, exec_lo
        assert_eq!(refusal(&words, false), NOT_SELECTED);
    }

    #[test]
    fn an_access_under_a_further_mask_is_no_waterfall() {
        // The access runs for the matching lanes that VCC also holds, yet
        // the retry removes every matching lane.
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let nop = word_at(&words, 0xBE95_3C15) + 1;
        assert_eq!(words[nop], 0xBF80_0000);
        words[nop] = 0x877E_6A7E; // s_and_b32 exec_lo, exec_lo, vcc_lo
        assert_eq!(refusal(&words, false), NOT_SELECTED);
    }

    #[test]
    fn a_loop_that_drops_unmatched_lanes_is_no_waterfall() {
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let retry = word_at(&words, 0x897E_157E);
        words[retry] = 0xBEFE_0380; // s_mov_b32 exec_lo, 0
        assert_eq!(refusal(&words, false), BAD_RETRY);
    }

    #[test]
    fn a_retry_that_keeps_other_lanes_than_those_waiting_is_no_waterfall() {
        // The retry keeps the lanes of VCC that did not just sample, not
        // the waiting ones: lanes 16 and up never sample on the console.
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let retry = word_at(&words, 0x897E_157E);
        words[retry] = 0x8A7E_7E6A; // s_andn2_b32 exec_lo, vcc_lo, exec_lo
        assert_eq!(refusal(&words, false), BAD_RETRY);
    }

    #[test]
    fn a_wave64_retry_of_one_half_is_no_waterfall() {
        // Only the low half of EXEC loses the lanes just served.
        let mut words = WATERFALL_FETCHED_WAVE64.to_vec();
        let retry = word_at(&words, 0x89FE_167E);
        words[retry] = 0x897E_167E; // s_xor_b32 exec_lo, exec_lo, s22
        assert_eq!(refusal(&words, true), BAD_RETRY);
    }

    #[test]
    fn a_descriptor_used_after_exec_is_restored_is_no_waterfall() {
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let sample = word_at(&words, 0xF09C_0F08);
        let restore = word_at(&words, 0xBEFE_0317);
        let image = [words[sample], words[sample + 1]];
        words[sample..sample + 2].fill(0xBF80_0000); // Keep the branch offsets.
        words.splice(restore + 1..restore + 1, image);
        assert_eq!(refusal(&words, false), NO_LOOP);
    }

    #[test]
    fn a_wave64_access_with_exec_hi_overwritten_is_no_waterfall() {
        let mut words = WATERFALL_FETCHED_WAVE64.to_vec();
        let save = word_at(&words, 0xBE96_2416);
        words[save + 1] = 0xBEFF_03C1; // s_mov_b32 exec_hi, -1
        assert_eq!(refusal(&words, true), NOT_SELECTED);
    }

    /// The waterfall with `v_cmpx_eq` narrowing EXEC itself, and the retry
    /// as `todo & !EXEC` (`folded`: in one `s_andn2_wrexec`), as LLVM's
    /// AMDGPU backend writes it for GFX10 and later.
    pub(crate) fn cmpx_waterfall(wave64: bool, folded: bool) -> Vec<u32> {
        let (base, todo, restore): (&[u32], u32, u32) = if wave64 {
            (WATERFALL_FETCHED_WAVE64, 0xBE94_047E, 0xBEFE_0414)
        } else {
            (WATERFALL_JOINED_WAVE32, 0xBE97_037E, 0xBEFE_0317)
        };
        let reads = word_at(base, 0x7E18_0504);
        let sample = word_at(base, 0xF09C_0F08);
        let mut w = base[..=word_at(base, todo)].to_vec();
        // s_mov_b32 s21, exec_lo / s_mov_b64 s[22:23], exec
        w.push(if wave64 { 0xBE96_047E } else { 0xBE95_037E });
        w.extend(&base[reads..reads + 8]); // v_readfirstlane_b32 s12..s19
        w.extend([
            0x7DE4_080C, // v_cmpx_eq_u64 s[12:13], v[4:5]
            0x7DE4_0C0E, // v_cmpx_eq_u64 s[14:15], v[6:7]
            0x7DE4_1010, // v_cmpx_eq_u64 s[16:17], v[8:9]
            0x7DE4_1412, // v_cmpx_eq_u64 s[18:19], v[10:11]
        ]);
        w.extend(&base[sample..sample + 2]); // image_sample_lz
        match (wave64, folded) {
            // s_andn2_wrexec_b32 s21, s21; s_cbranch_execnz
            (false, true) => w.extend([0xBE95_4715, 0xBF89_FFF0]),
            // s_andn2_b32 s21, s21, exec_lo; s_mov_b32 exec_lo, s21
            (false, false) => w.extend([0x8A15_7E15, 0xBEFE_0315, 0xBF89_FFEF]),
            // s_andn2_wrexec_b64 s[22:23], s[22:23]
            (true, true) => w.extend([0xBE96_3A16, 0xBF89_FFF0]),
            // s_andn2_b64 s[22:23], s[22:23], exec; s_mov_b64 exec, s[22:23]
            (true, false) => w.extend([0x8A96_7E16, 0xBEFE_0416, 0xBF89_FFEF]),
        }
        w.extend(&base[word_at(base, restore)..]);
        w
    }

    #[test]
    fn a_waterfall_narrowed_by_v_cmpx_is_one() {
        for wave64 in [false, true] {
            for folded in [false, true] {
                let r = waterfall(&cmpx_waterfall(wave64, folded), wave64)
                    .unwrap_or_else(|e| panic!("wave64 {wave64} folded {folded}: {e:?}"));
                assert_eq!(r.waterfall.compares.len(), 4, "{wave64} {folded}");
                assert_eq!(r.waterfall.reads.len(), 8, "{wave64} {folded}");
            }
        }
    }

    #[test]
    fn a_v_cmpx_waterfall_whose_retry_drops_lanes_is_none() {
        let mut words = cmpx_waterfall(false, false);
        let retry = word_at(&words, 0x8A15_7E15);
        words[retry] = 0x8A15_7E6A; // s_andn2_b32 s21, vcc_lo, exec_lo
        assert_eq!(refusal(&words, false), BAD_RETRY);
    }

    #[test]
    fn a_waterfall_nested_in_another_is_refused_by_name() {
        // An image whose resource and sampler are both per lane, with one
        // loop per operand, the one for the sampler inside the one for the
        // resource (what the SelectionDAG legalisation in LLVM would give).
        // The console gives each lane its own pair. Refused for now: the
        // access of the inner loop reads the first-lane dwords of the outer
        // loop, which another block holds.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0x7E04_0D00, // v_cvt_f32_u32 v2, v0
            0x7E06_0302, // v_mov_b32 v3, v2
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0x3602_0081, // v_and_b32 v1, 1, v0
            0x3402_0285, // v_lshlrev_b32 v1, 5, v1
            0x3622_0080, // v_and_b32 v17, 0, v0
            0x3422_2284, // v_lshlrev_b32 v17, 4, v17
            0xDC38_8000,
            0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
            0xDC38_8010,
            0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
            0xDC38_8000,
            0x1406_0011, // global_load_dwordx4 v[20:23], v17, s[6:7]
            0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
            0xBE94_037E, // s_mov_b32 s20, exec_lo
            // L1, the loop over the resource:
            0x7E18_0504, // v_readfirstlane_b32 s12, v4
            0x7E1A_0505,
            0x7E1C_0506,
            0x7E1E_0507,
            0x7E20_0508,
            0x7E22_0509,
            0x7E24_050A,
            0x7E26_050B, // v_readfirstlane_b32 s19, v11
            0xD4E2_0015,
            0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
            0xD4E2_0016,
            0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xD4E2_0016,
            0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xD4E2_0016,
            0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xBE95_3C15, // s_and_saveexec_b32 s21, s21
            0xBE9A_037E, // s_mov_b32 s26, exec_lo
            // L2, the loop over the sampler:
            0x7E10_0514, // v_readfirstlane_b32 s8, v20
            0x7E12_0515, // v_readfirstlane_b32 s9, v21
            0x7E14_0516, // v_readfirstlane_b32 s10, v22
            0x7E16_0517, // v_readfirstlane_b32 s11, v23
            0xD4E2_0018,
            0x0002_2808, // v_cmp_eq_u64_e64 s24, s[8:9], v[20:21]
            0xD4E2_0019,
            0x0002_2C0A, // v_cmp_eq_u64_e64 s25, s[10:11], v[22:23]
            0x8718_1918, // s_and_b32 s24, s24, s25
            0xBE98_3C18, // s_and_saveexec_b32 s24, s24
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11]
            0x897E_187E, // s_xor_b32 exec_lo, exec_lo, s24
            0xBF89_FFF2, // s_cbranch_execnz L2
            0xBEFE_031A, // s_mov_b32 exec_lo, s26
            0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
            0xBF89_FFDA, // s_cbranch_execnz L1
            0xBEFE_0314, // s_mov_b32 exec_lo, s20
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        assert_eq!(refusal(&words, false), SPLIT);
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_lds_fail_naming_it() {
        // The waterfall over descriptors each lane read from LDS:
        // ds_read_b128 v[4:7], v1 and ds_read_b128 v[8:11], v1 offset:16
        // in place of the global loads.
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        words[5..9].copy_from_slice(&[0xDBFC_0000, 0x0400_0001, 0xDBFC_0010, 0x0800_0001]);
        let err = waterfall(&words, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane of an LDS read",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// The block of `p` at byte `offset`.
    fn block_at(words: &[u32], offset: usize) -> usize {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        p.blocks.iter().position(|b| b.offset == offset).unwrap()
    }

    /// The table offset of what the scalar load of `s` reads.
    fn table_offset(r: &Resources, s: &Source) -> u32 {
        match *s {
            Source::Loaded { load, first: 0 } => {
                r.scalar.iter().find(|x| x.0 == load).unwrap().1.offset
            }
            ref s => panic!("{s:?}"),
        }
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_an_if_else_join_are_traced_per_path() {
        let words = WATERFALL_JOINED_WAVE32;
        let r = waterfall(words, false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        // The join after the else arm: lanes that ran it (lanes from 16 on)
        // take the second entry (0x20); the others come through the flow
        // block, whose join gives lanes that ran the then arm the first one
        // (0). No lane that skipped the then arm comes that way, so the
        // dwords they held before (the inputs) are not traced.
        let Source::LanePhi { block, args } = &image.source else {
            unreachable!()
        };
        assert_eq!(*block, block_at(words, 0x9c));
        let [Some(flow), Some(other)] = &args[..] else {
            panic!("{args:?}")
        };
        assert_eq!(table_offset(&r, other), 0x20);
        let Source::LanePhi { block, args } = flow else {
            panic!("{flow:?}")
        };
        assert_eq!(*block, block_at(words, 0x70));
        let [None, Some(then)] = &args[..] else {
            panic!("{args:?}")
        };
        assert_eq!(table_offset(&r, then), 0);
        // The loop's compares are of the joined dwords; still recognised.
        assert_eq!(r.waterfall.compares.len(), 4);
        assert_eq!(r.waterfall.reads.len(), 8);
        // Without the lanes' paths the join is refused.
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let layout = [param(Table, 0, 1), param(Table, 1, 1)];
        let err = track(&p, &layout, HwStage::Cs, None).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane of a phi, without the lanes' paths",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn descriptors_joined_from_a_per_lane_load_and_a_per_lane_choice_are_traced_per_path() {
        let words = WATERFALL_JOINED_MIXED_WAVE32;
        let r = waterfall(words, false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        let Source::LanePhi { block, args } = &image.source else {
            unreachable!()
        };
        assert_eq!(*block, block_at(words, 0xb8));
        let [
            Some(Source::LanePhi { args: flow, .. }),
            Some(Source::Lanes { args: picked, .. }),
        ] = &args[..]
        else {
            panic!("{args:?}")
        };
        // The then arm loads each lane's entry from the table at s0.
        let [None, Some(Source::Fetched { load, first: 0 })] = flow[..] else {
            panic!("{flow:?}")
        };
        let a = r.scalar.iter().find(|s| s.0 == load).unwrap().1;
        assert_eq!((a.param, a.offset), (0, 0));
        assert!(a.dynamic.is_some());
        // The else arm picks the first entry where vcc is set.
        let offsets: Vec<u32> = picked.iter().map(|s| table_offset(&r, s)).collect();
        assert_eq!(offsets, [0, 0x20]);
        assert_eq!(r.waterfall.compares.len(), 4);
    }

    #[test]
    fn a_descriptor_read_from_the_first_lane_of_a_loop_phi_fails_naming_it() {
        let err = waterfall(WATERFALL_LOOPED_WAVE32, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane of a phi at a loop header",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_per_lane_descriptor_some_lanes_do_not_write_fails() {
        // The else arm of WATERFALL_JOINED_WAVE32 not skipped (its
        // s_cbranch_execz an s_nop): it runs with EXEC clear in the lanes
        // of the then arm, which keep the first entry; the second is not
        // every lane's.
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let skip = words.iter().rposition(|&w| w == 0xBF88_0008).unwrap();
        words[skip] = 0xBF80_0000;
        let err = waterfall(&words, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a per-lane descriptor written where some lanes may keep the old one",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_buffer_descriptor_joined_per_lane_is_traced_on_the_paths_of_lanes_that_load() {
        let words = WATERFALL_JOINED_BUFFER_WAVE32;
        let r = waterfall(words, false).unwrap();
        let buffer = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        // The outer join: lanes from the else arm's inner join, or (below
        // 16) through the flow block.
        let Source::LanePhi { block, args } = &buffer.source else {
            unreachable!()
        };
        assert_eq!(*block, block_at(words, 0x78));
        let [
            Some(Source::LanePhi { args: flow, .. }),
            Some(Source::LanePhi {
                block: inner,
                args: nested,
            }),
        ] = &args[..]
        else {
            panic!("{args:?}")
        };
        let [None, Some(then)] = &flow[..] else {
            panic!("{flow:?}")
        };
        assert_eq!(table_offset(&r, then), 0);
        // Lanes from 24 on skip the inner arm and keep the inputs, but do
        // not load: that edge is not traced.
        assert_eq!(*inner, block_at(words, 0x74));
        let [None, Some(other)] = &nested[..] else {
            panic!("{nested:?}")
        };
        assert_eq!(table_offset(&r, other), 0x20);
        assert_eq!(r.waterfall.compares.len(), 2);
        assert_eq!(r.waterfall.reads.len(), 4);
    }

    pub(crate) fn joined_with_redundant_write() -> Vec<u32> {
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let at = word_at(&words, 0x887E_147E) + 1;
        words.splice(
            at..at,
            [
                0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
                0xBE92_3C6A, // s_and_saveexec_b32 s18, vcc_lo
                0x7E08_0218, // v_mov_b32 v4, s24 (then arm's word 0)
                0xBEFE_0312, // s_mov_b32 exec_lo, s18
            ],
        );
        words
    }

    /// 32-thread synthetic compute shader joining two 4-dword sampler
    /// descriptors across an if/else (`lane < 16` takes entry 0, else entry 1)
    /// and sampling a 2x1 texture at U=2.0.
    pub(crate) fn joined_sampler_shader(redundant_write: Option<u32>) -> Vec<u32> {
        let mut words = vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF40C_0202,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
            0xF408_0603,
            0xFA00_0000, // s_load_dwordx4 s[24:27], s[6:7], 0x0
            0xF408_0703,
            0xFA00_0010, // s_load_dwordx4 s[28:31], s[6:7], 0x10
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0x7E04_02F4, // v_mov_b32 v2, 2.0
            0x7E06_0280, // v_mov_b32 v3, 0
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
            0x8914_147E, // s_xor_b32 s20, exec_lo, s20
            0xBF88_0004, // s_cbranch_execz -> +4 (else flow)
            0x7E08_0218, // v_mov_b32 v4, s24
            0x7E0A_0219, // v_mov_b32 v5, s25
            0x7E0C_021A, // v_mov_b32 v6, s26
            0x7E0E_021B, // v_mov_b32 v7, s27
            0xBE94_3D14, // s_or_saveexec_b32 s20, s20
            0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
            0xBF88_0004, // s_cbranch_execz -> +4 (join)
            0x7E08_021C, // v_mov_b32 v4, s28
            0x7E0A_021D, // v_mov_b32 v5, s29
            0x7E0C_021E, // v_mov_b32 v6, s30
            0x7E0E_021F, // v_mov_b32 v7, s31
            0x887E_147E, // s_or_b32 exec_lo, exec_lo, s20
        ];
        if let Some(mov) = redundant_write {
            words.extend([
                0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
                0xBE92_3C6A, // s_and_saveexec_b32 s18, vcc_lo
                mov,
                0xBEFE_0312, // s_mov_b32 exec_lo, s18
            ]);
        }
        words.extend([
            0xBE97_037E, // s_mov_b32 s23, exec_lo
            0x7E20_0504, // loop: v_readfirstlane_b32 s16, v4
            0x7E22_0505, // v_readfirstlane_b32 s17, v5
            0x7E24_0506, // v_readfirstlane_b32 s18, v6
            0x7E26_0507, // v_readfirstlane_b32 s19, v7
            0xD4E2_0015,
            0x0002_0810, // v_cmp_eq_u64_e64 s21, s[16:17], v[4:5]
            0xD4E2_0016,
            0x0002_0C12, // v_cmp_eq_u64_e64 s22, s[18:19], v[6:7]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xBE95_3C15, // s_and_saveexec_b32 s21, s21
            0xBF80_0000, // s_nop 0
            0xF09C_0F08,
            0x0082_0C02, // image_sample_lz v[12:15], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
            0xBF89_FFF0, // s_cbranch_execnz -> loop (-16 dwords)
            0xBEFE_0317, // s_mov_b32 exec_lo, s23
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            0xBF81_0000, // s_endpgm
        ]);
        words
    }

    #[test]
    fn joined_descriptor_redundant_partial_write() {
        let r = waterfall(&joined_with_redundant_write(), false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        let Source::LanePhi { args, .. } = &image.source else {
            unreachable!()
        };
        let [Some(Source::LanePhi { args: flow, .. }), Some(other)] = &args[..] else {
            panic!("{args:?}")
        };
        assert_eq!(table_offset(&r, other), 0x20);
        let [None, Some(then)] = &flow[..] else {
            panic!("{flow:?}")
        };
        assert_eq!(table_offset(&r, then), 0);
        assert_eq!(r.waterfall.compares.len(), 4);
        assert_eq!(r.waterfall.reads.len(), 8);
    }

    #[test]
    fn joined_descriptor_redundant_partial_write_subset_and_swapped_cmp() {
        // Strict subset mask (lane < 8) writing v4 = s24.
        let mut subset = joined_with_redundant_write();
        let cmp = word_at(&subset, 0x7D88_0090);
        let second_cmp = cmp
            + 1
            + subset[cmp + 1..]
                .iter()
                .position(|&w| w == 0x7D88_0090)
                .unwrap();
        subset[second_cmp] = 0x7D88_0088; // v_cmp_gt_u32 vcc_lo, 8, v0
        assert!(waterfall(&subset, false).is_ok());

        // Swapped VOP3 comparison: v_cmp_lt_u32 vcc_lo, v0, 16.
        let mut swapped = WATERFALL_JOINED_WAVE32.to_vec();
        let at = word_at(&swapped, 0x887E_147E) + 1;
        swapped.splice(
            at..at,
            [
                0xD4C1_006A,
                0x0001_2100, // v_cmp_lt_u32 vcc_lo, v0, 16
                0xBE92_3C6A, // s_and_saveexec_b32 s18, vcc_lo
                0x7E08_0218, // v_mov_b32 v4, s24
                0xBEFE_0312, // s_mov_b32 exec_lo, s18
            ],
        );
        assert!(waterfall(&swapped, false).is_ok());

        // Independent sampler control (#750): redundant v4 = s24 passes, non-identity v4 = s28 fails.
        assert!(waterfall(&joined_sampler_shader(Some(0x7E08_0218)), false).is_ok());
        assert_eq!(
            refusal(&joined_sampler_shader(Some(0x7E08_021C)), false),
            "a per-lane descriptor written where some lanes may keep the old one"
        );
    }

    #[test]
    fn joined_descriptor_non_identity_or_overlapping_partial_writes_are_refused() {
        // Non-identity masked write: v4 = s32 on lanes < 16 corrupts the then-arm descriptor.
        let mut wrong_val = joined_with_redundant_write();
        let mov = word_at(&wrong_val, 0xBE92_3C6A) + 1;
        wrong_val[mov] = 0x7E08_0220; // v_mov_b32 v4, s32
        assert_eq!(
            refusal(&wrong_val, false),
            "a per-lane descriptor written where some lanes may keep the old one"
        );

        // Overlapping masked write: widen mask to lane < 24 so lanes 16..23 overwrite else-arm word 0.
        let mut widened = joined_with_redundant_write();
        let cmp = word_at(&widened, 0xBE92_3C6A) - 1;
        widened[cmp] = 0x7D88_0098; // v_cmp_gt_u32 vcc_lo, 24, v0
        assert_eq!(
            refusal(&widened, false),
            "a per-lane descriptor written where some lanes may keep the old one"
        );

        // Opposite-arm non-matching write: v_cmp_le_u32 16, v0 (lanes >= 16) writing s24.
        let mut flipped = WATERFALL_JOINED_WAVE32.to_vec();
        let at = word_at(&flipped, 0x887E_147E) + 1;
        flipped.splice(
            at..at,
            [
                0x7D86_0090, // v_cmp_le_u32 vcc_lo, 16, v0
                0xBE92_3C6A, // s_and_saveexec_b32 s18, vcc_lo
                0x7E08_0218, // v_mov_b32 v4, s24
                0xBEFE_0312, // s_mov_b32 exec_lo, s18
            ],
        );
        assert_eq!(
            refusal(&flipped, false),
            "a per-lane descriptor written where some lanes may keep the old one"
        );
    }

    pub(crate) fn fetched_with_lane_preserving_transforms() -> Vec<u32> {
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        // After global_load_dwordx4 v[4:7], v[8:11] and s_waitcnt (0xBF8C_0070),
        // shuffle/identity-move/extract the loaded descriptor dwords before the waterfall loop.
        let at = word_at(&words, 0xBF8C_0070) + 1;
        words.splice(
            at..at,
            [
                // v_swap_b32 v4, v5; v_swap_b32 v5, v4 (back to original order via swaps)
                0x7E08_CB05,
                0x7E0A_CB04,
                // Identity bitwise/shift moves: v4 = 0 | v4, v5 = -1 & v5, v6 = v6 << 0
                0x3808_0880,
                0x360A_0AC1,
                0x340C_0C80,
                // 64-bit word extract: v_lshrrev_b64 v[20:21], 32, v[6:7] -> v20 = v7
                0xD700_0014,
                0x0002_0CA0,
                // v_mov_b32 v7, v20
                0x7E0E_0314,
                // Byte-preserving v_perm_b32 v22, v8, v9, 0x07060504 -> v22 = v8
                0xD744_0016,
                0x03FE_1308,
                0x0706_0504,
                // v_mov_b32 v8, v22
                0x7E10_0316,
                // v_alignbit_b32 v23, v8, v9, 0 -> v23 = v9
                0xD54E_0017,
                0x0202_1308,
                // v_mov_b32 v9, v23
                0x7E12_0317,
                // v_cndmask_b32 v10, v10, v10, vcc_lo (same operand on both sides)
                0x0214_150A,
            ],
        );
        // Adjust s_cbranch_execnz backward offset in loop by +16 dwords (-0x1a - 16 = -0x2a) —
        // wait: the loop starts AFTER `at`, so its own backward offset is unchanged!
        words
    }

    #[test]
    fn first_lane_of_lane_preserving_moves_and_extracts_is_traced() {
        let r = waterfall(&fetched_with_lane_preserving_transforms(), false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| d.role == Role::Resource)
            .unwrap();
        assert!(matches!(image.source, Source::Fetched { first: 0, .. }));
        assert_eq!(r.waterfall.compares.len(), 4);
        assert_eq!(r.waterfall.reads.len(), 8);
    }

    #[test]
    fn first_lane_of_non_identity_computed_vgprs_still_fails() {
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        let at = word_at(&words, 0xBF8C_0070) + 1;
        // Non-identity computed VGPR: v_or_b32 v4, 1, v4
        words.insert(at, 0x3808_0881);
        assert_eq!(
            refusal(&words, false),
            "a descriptor read from the first lane of computed VGPRs"
        );

        // Non-dword 64-bit shift: v_lshrrev_b64 v[20:21], 16, v[6:7]; v_mov_b32 v4, v20
        let mut bad_shift = WATERFALL_FETCHED_WAVE32.to_vec();
        bad_shift.splice(
            at..at,
            [
                0xD700_0014,
                0x0002_0C90, // shift by 16
                0x7E08_0314, // v4 = v20
            ],
        );
        assert_eq!(
            refusal(&bad_shift, false),
            "a descriptor read from the first lane of computed VGPRs"
        );

        // Non-identity permute selector: 0x07060405 into v4
        let mut bad_perm = WATERFALL_FETCHED_WAVE32.to_vec();
        bad_perm.splice(
            at..at,
            [
                0xD744_0016,
                0x03FE_1308,
                0x0706_0405,
                0x7E08_0316, // v4 = v22
            ],
        );
        assert_eq!(
            refusal(&bad_perm, false),
            "a descriptor read from the first lane of computed VGPRs"
        );

        // Guarded v_swap_b32 under partial EXEC (not redundant since v4 != v5) must fail.
        let mut guarded_swap = WATERFALL_FETCHED_WAVE32.to_vec();
        guarded_swap.splice(
            at..at,
            [
                0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
                0xBE92_3C6A, // s_and_saveexec_b32 s18, vcc_lo
                0x7E08_CB05, // v_swap_b32 v4, v5
                0xBEFE_0312, // s_mov_b32 exec_lo, s18
            ],
        );
        assert_eq!(
            refusal(&guarded_swap, false),
            "a per-lane descriptor written where some lanes may keep the old one"
        );
    }

    #[test]
    fn lane_select_accepts_equivalent_masks_and_refuses_different_ones() {
        // Recompute an equivalent mask in s18 for the high word of WATERFALL_PICKED_WAVE32
        // and verify it still traces, while a narrower mask (lane < 8) is refused.
        let mut eq_mask = WATERFALL_PICKED_WAVE32.to_vec();
        let at = word_at(&eq_mask, 0x0216_1627);
        eq_mask.splice(
            at..at,
            [
                0xD4C1_0012,
                0x0001_2100, // v_cmp_lt_u32_e64 s18, v0, 16
            ],
        );
        let cnd = at + 2;
        // Replace v_cndmask_b32 v11, s39, v11, vcc_lo with VOP3 using s18.
        eq_mask[cnd] = 0xD501_000B;
        eq_mask.insert(cnd + 1, (18 << 18) | ((256 + 11) << 9) | 39);
        assert!(waterfall(&eq_mask, false).is_ok());

        let mut diff_mask = eq_mask;
        diff_mask[at + 1] = 0x0001_1100; // v_cmp_lt_u32_e64 s18, v0, 8
        assert_eq!(
            refusal(&diff_mask, false),
            "a descriptor chosen per lane by several masks"
        );
    }

    #[test]
    fn v_cndmask_literal_all_ones_mask_is_refused_in_wave64() {
        // v_cndmask_b32_e64 v4, v20, v4, 0xFFFF_FFFF: in wave32 all 32 lanes
        // take v4 (set), whereas in wave64 the literal zero-extends to
        // 0x0000_0000_FFFF_FFFF so lanes 32..63 take v20 (clear).
        let lit_mask = [
            0xD501_0004,
            (255 << 18) | ((256 + 4) << 9) | (256 + 20),
            u32::MAX,
        ];
        let mut w32 = WATERFALL_FETCHED_WAVE32.to_vec();
        let at32 = word_at(&w32, 0xBF8C_0070) + 1;
        w32.splice(at32..at32, lit_mask);
        assert!(waterfall(&w32, false).is_ok());

        let mut w64_lit = WATERFALL_FETCHED_WAVE64.to_vec();
        let at64 = word_at(&w64_lit, 0xBF8C_0070) + 1;
        w64_lit.splice(at64..at64, lit_mask);
        assert_eq!(
            refusal(&w64_lit, true),
            "a descriptor read from the first lane of computed VGPRs"
        );

        // Inline -1 (Const(193)) sign-extends to all 64 lanes in wave64.
        let mut w64_inline = WATERFALL_FETCHED_WAVE64.to_vec();
        w64_inline.splice(
            at64..at64,
            [0xD501_0004, (193 << 18) | ((256 + 4) << 9) | (256 + 20)],
        );
        assert!(waterfall(&w64_inline, true).is_ok());
    }

    /// 32-thread synthetic compute shader matching the foliage mesh sampler
    /// pattern: `v[4:7]` hold pre-split computed VGPRs (`v_cvt_f32_u32`,
    /// `v_sub_f32`, `v_mul_legacy_f32`, `v_add_f32`). A per-lane if/else
    /// (`v_cmp_gt_u32` + `s_and_saveexec_b32`, then-arm leaves `v[4:7]`
    /// untouched, `s_andn2_b32` + `s_cbranch_execz`, else-arm loads sampler 1
    /// and copies to `v[4:7]`, `s_mov_b32 exec_lo` at join) feeds a full-mask
    /// waterfall loop over `v[4:7]`. With `downstream`, `v[12:15]` is consumed
    /// only by else-arm lanes; otherwise `v12` is stored unconditionally.
    pub(crate) fn foliage_joined_sampler_shader(downstream: &[u32]) -> Vec<u32> {
        let mut words = vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF40C_0202,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0x7E04_02F4, // v_mov_b32 v2, 2.0
            0x7E06_0280, // v_mov_b32 v3, 0
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0x7E24_0D00, // v_cvt_f32_u32 v18, v0 (then-arm result)
            // Pre-split non-descriptor computed VGPRs in v[4:7]:
            0x7E08_0D00, // v_cvt_f32_u32 v4, v0
            0x080A_0902, // v_sub_f32 v5, v2, v4
            0x0E0C_0902, // v_mul_legacy_f32 v6, v2, v4
            0x060E_0902, // v_add_f32 v7, v2, v4
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            // Per-lane if/else: then (lane < 16) leaves v[4:7] untouched;
            // else (lane >= 16) loads sampler 1 and copies into v[4:7].
            0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
            0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
            0x8A7E_7E14, // s_andn2_b32 exec_lo, s20, exec_lo
            0xBF88_0007, // s_cbranch_execz -> join (+7 dwords)
            0xF408_0703,
            0xFA00_0010, // s_load_dwordx4 s[28:31], s[6:7], 0x10
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x7E08_021C, // v_mov_b32 v4, s28
            0x7E0A_021D, // v_mov_b32 v5, s29
            0x7E0C_021E, // v_mov_b32 v6, s30
            0x7E0E_021F, // v_mov_b32 v7, s31
            0xBEFE_0314, // join: s_mov_b32 exec_lo, s20
            // Full-mask waterfall loop over v[4:7]:
            0xBE97_037E, // s_mov_b32 s23, exec_lo
            0x7E20_0504, // loop: v_readfirstlane_b32 s16, v4
            0x7E22_0505, // v_readfirstlane_b32 s17, v5
            0x7E24_0506, // v_readfirstlane_b32 s18, v6
            0x7E26_0507, // v_readfirstlane_b32 s19, v7
            0xD4E2_0015,
            0x0002_0810, // v_cmp_eq_u64_e64 s21, s[16:17], v[4:5]
            0xD4E2_0016,
            0x0002_0C12, // v_cmp_eq_u64_e64 s22, s[18:19], v[6:7]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xBE95_3C15, // s_and_saveexec_b32 s21, s21
            0xBF80_0000, // s_nop 0
            0xF09C_0F08,
            0x0082_0C02, // image_sample_lz v[12:15], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
            0xBF89_FFF0, // s_cbranch_execnz -> loop
            0xBEFE_0317, // s_mov_b32 exec_lo, s23
        ];
        words.extend_from_slice(downstream);
        words
    }

    #[test]
    fn joined_sampler_with_computed_non_descriptor_arm_accepted_when_dead_downstream() {
        // Positive 1: downstream v_cndmask_b32 selects v12 only for else lanes
        // (vcc_lo clear, i.e., lane >= 16) and v18 for then lanes.
        let cndmask_shader = foliage_joined_sampler_shader(&[
            0x0218_250C, // v_cndmask_b32 v12, v12, v18, vcc_lo
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            S_ENDPGM,
        ]);
        let r = waterfall(&cndmask_shader, false).unwrap();
        let sampler = r
            .descriptors
            .iter()
            .find(|d| d.role == Role::Sampler)
            .unwrap();
        let Source::LanePhi { args, .. } = &sampler.source else {
            panic!("{:?}", sampler.source);
        };
        assert_eq!(args.len(), 2);
        assert!(args.iter().any(Option::is_none));
        assert!(
            args.iter()
                .flatten()
                .any(|s| matches!(s, Source::Loaded { .. }))
        );
        assert_eq!(r.waterfall.compares.len(), 2);
        assert_eq!(r.waterfall.reads.len(), 4);

        // Positive 2: downstream transitive VALU before v_cndmask_b32.
        let transitive_shader = foliage_joined_sampler_shader(&[
            0x0E18_050C, // v_mul_legacy_f32 v12, v12, v2
            0x0218_250C, // v_cndmask_b32 v12, v12, v18, vcc_lo
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            S_ENDPGM,
        ]);
        assert!(waterfall(&transitive_shader, false).is_ok());

        // Positive 3: downstream guarded consumer under the else-arm mask (!vcc_lo).
        let guarded_shader = foliage_joined_sampler_shader(&[
            0xBE96_076A, // s_not_b32 s22, vcc_lo
            0xBE94_3C16, // s_and_saveexec_b32 s20, s22
            0xBF88_0002, // s_cbranch_execz -> +2
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            0xBEFE_0314, // s_mov_b32 exec_lo, s20
            S_ENDPGM,
        ]);
        assert!(waterfall(&guarded_shader, false).is_ok());
    }

    #[test]
    fn joined_sampler_with_computed_non_descriptor_arm_refused_when_live() {
        // Negative 1: exact same 4-step pattern with unconditional full-mask store.
        let unconditional = foliage_joined_sampler_shader(&[
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            S_ENDPGM,
        ]);
        assert_eq!(
            refusal(&unconditional, false),
            "a descriptor read from the first lane of computed VGPRs"
        );

        // Negative 2: v_cndmask_b32 with swapped polarity (then-lanes select v12!).
        let wrong_polarity = foliage_joined_sampler_shader(&[
            0x0218_1912, // v_cndmask_b32 v12, v18, v12, vcc_lo
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            S_ENDPGM,
        ]);
        assert_eq!(
            refusal(&wrong_polarity, false),
            "a descriptor read from the first lane of computed VGPRs"
        );

        // Negative 3: v_cndmask_b32 with an unrelated mask (lane < 8 instead of lane < 16).
        let wrong_mask = foliage_joined_sampler_shader(&[
            0x7D88_0088, // v_cmp_gt_u32 vcc_lo, 8, v0
            0x0218_250C, // v_cndmask_b32 v12, v12, v18, vcc_lo
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            S_ENDPGM,
        ]);
        assert_eq!(
            refusal(&wrong_mask, false),
            "a descriptor read from the first lane of computed VGPRs"
        );

        // Negative 4: second channel v13 stored unconditionally even though v12 is masked.
        let partial_channels = foliage_joined_sampler_shader(&[
            0x0218_250C, // v_cndmask_b32 v12, v12, v18, vcc_lo
            0xE070_1000,
            0x800A_0D10, // buffer_store_dword v13, v16, s[40:43], 0 offen
            S_ENDPGM,
        ]);
        assert_eq!(
            refusal(&partial_channels, false),
            "a descriptor read from the first lane of computed VGPRs"
        );

        // Negative 5: cross-lane v_readfirstlane_b32 on v12 before v_cndmask_b32.
        let cross_lane = foliage_joined_sampler_shader(&[
            0x7E2E_050C, // v_readfirstlane_b32 s23, v12
            0x0218_250C, // v_cndmask_b32 v12, v12, v18, vcc_lo
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            S_ENDPGM,
        ]);
        assert_eq!(
            refusal(&cross_lane, false),
            "a descriptor read from the first lane of computed VGPRs"
        );

        // Negative 6: guarded v_add_f32 v12, v12, v2 under !vcc_lo leaves old v12
        // on then-lanes, which is then stored unconditionally.
        let guarded_overwrite_leak = foliage_joined_sampler_shader(&[
            0xBE96_076A, // s_not_b32 s22, vcc_lo
            0xBE94_3C16, // s_and_saveexec_b32 s20, s22
            0x0618_050C, // v_add_f32 v12, v12, v2
            0xBEFE_0314, // s_mov_b32 exec_lo, s20
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            S_ENDPGM,
        ]);
        assert_eq!(
            refusal(&guarded_overwrite_leak, false),
            "a descriptor read from the first lane of computed VGPRs"
        );

        // Negative 7: v_cvt_f16_f32 v12, v2 preserves upper 16 bits of v12 on
        // all lanes, leaking bad-lane sample bits into the unconditional store.
        let cvt_f16_f32_leak = foliage_joined_sampler_shader(&[
            0x7E18_1502, // v_cvt_f16_f32 v12, v2
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            S_ENDPGM,
        ]);
        assert_eq!(
            refusal(&cvt_f16_f32_leak, false),
            "a descriptor read from the first lane of computed VGPRs"
        );

        // Negative 8: buffer_load_short_d16 v12, ... preserves upper 16 bits of
        // v12 on all lanes, leaking bad-lane sample bits into the unconditional store.
        let d16_buffer_load_leak = foliage_joined_sampler_shader(&[
            0xE090_1000,
            0x800A_0C10, // buffer_load_short_d16 v12, v16, s[40:43], 0 offen
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            S_ENDPGM,
        ]);
        assert_eq!(
            refusal(&d16_buffer_load_leak, false),
            "a descriptor read from the first lane of computed VGPRs"
        );
    }

    #[test]
    fn cndmask_joined_sampler_with_computed_non_descriptor_arm_respects_liveness() {
        // Branchless v_cndmask_b32 joining computed VGPRs (v[4:7], when vcc_lo=1)
        // with uniform sampler (s[28:31], when vcc_lo=0).
        let build_cndmask = |masked_downstream: bool| {
            let mut words = vec![
                0xBE84_0300, // s_mov_b32 s4, s0
                0xBE85_0380, // s_mov_b32 s5, 0
                0xBE86_0301, // s_mov_b32 s6, s1
                0xBE87_0380, // s_mov_b32 s7, 0
                0xF40C_0202,
                0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
                0xF408_0703,
                0xFA00_0010, // s_load_dwordx4 s[28:31], s[6:7], 0x10
                0xF408_0A02,
                0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
                0x7E04_02F4, // v_mov_b32 v2, 2.0
                0x7E06_0280, // v_mov_b32 v3, 0
                0x3420_0082, // v_lshlrev_b32 v16, 2, v0
                0x7E24_0D00, // v_cvt_f32_u32 v18, v0
                0x7E08_0D00, // v_cvt_f32_u32 v4, v0
                0x080A_0902, // v_sub_f32 v5, v2, v4
                0x0E0C_0902, // v_mul_legacy_f32 v6, v2, v4
                0x060E_0902, // v_add_f32 v7, v2, v4
                0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
                0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
                0x0208_081C, // v_cndmask_b32 v4, s28, v4, vcc_lo
                0x020A_0A1D, // v_cndmask_b32 v5, s29, v5, vcc_lo
                0x020C_0C1E, // v_cndmask_b32 v6, s30, v6, vcc_lo
                0x020E_0E1F, // v_cndmask_b32 v7, s31, v7, vcc_lo
                0xBE97_037E, // s_mov_b32 s23, exec_lo
                0x7E20_0504, // loop: v_readfirstlane_b32 s16, v4
                0x7E22_0505, // v_readfirstlane_b32 s17, v5
                0x7E24_0506, // v_readfirstlane_b32 s18, v6
                0x7E26_0507, // v_readfirstlane_b32 s19, v7
                0xD4E2_0015,
                0x0002_0810, // v_cmp_eq_u64_e64 s21, s[16:17], v[4:5]
                0xD4E2_0016,
                0x0002_0C12, // v_cmp_eq_u64_e64 s22, s[18:19], v[6:7]
                0x8715_1615, // s_and_b32 s21, s21, s22
                0xBE95_3C15, // s_and_saveexec_b32 s21, s21
                0xBF80_0000, // s_nop 0
                0xF09C_0F08,
                0x0082_0C02, // image_sample_lz v[12:15], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
                0xBF8C_3F70, // s_waitcnt vmcnt(0)
                0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
                0xBF89_FFF0, // s_cbranch_execnz -> loop
                0xBEFE_0317, // s_mov_b32 exec_lo, s23
            ];
            if masked_downstream {
                words.push(0x0218_250C); // v_cndmask_b32 v12, v12, v18, vcc_lo
            }
            words.extend([
                0xE070_1000,
                0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
                S_ENDPGM,
            ]);
            words
        };
        assert!(waterfall(&build_cndmask(true), false).is_ok());
        assert_eq!(
            refusal(&build_cndmask(false), false),
            "a descriptor read from the first lane of computed VGPRs"
        );
    }

    #[test]
    fn foliage_wave64_buffer_load_and_float_math_non_descriptor_arm() {
        // Wave64 variant matching the exact console pattern:
        // buffer_load_dwordx3 + v_sub_f32 + v_mul_legacy_f32 into v[4:7],
        // v_cmp_gt_u32_e64 s[6:7], s_and_saveexec_b64 s[64:65], s_andn2_b64,
        // s_cbranch_execz, s_load_dwordx4 + v_mov_b32, s_mov_b64 exec,
        // waterfall over v[4:7], and downstream v_cndmask_b32_e64 using s[6:7].
        let build_w64 = |masked_downstream: bool| {
            let mut words = vec![
                0xBE84_0300, // s_mov_b32 s4, s0
                0xBE85_0380, // s_mov_b32 s5, 0
                0xBE82_0301, // s_mov_b32 s2, s1
                0xBE83_0380, // s_mov_b32 s3, 0
                0xF40C_0202,
                0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
                0xF408_0A02,
                0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
                0x7E04_02F4, // v_mov_b32 v2, 2.0
                0x7E06_0280, // v_mov_b32 v3, 0
                0x3420_0082, // v_lshlrev_b32 v16, 2, v0
                0x7E24_0D00, // v_cvt_f32_u32 v18, v0
                0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
                // Pre-split buffer_load_dwordx3 v[4:6] + float math into v[4:7]:
                0xE03C_1000,
                0x800A_0410, // buffer_load_dwordx3 v[4:6], v16, s[40:43], 0 offen
                0xBF8C_3F70, // s_waitcnt vmcnt(0)
                0x0808_0902, // v_sub_f32 v4, v2, v4
                0x0E0E_0B02, // v_mul_legacy_f32 v7, v2, v5
                // Per-lane if/else on s[6:7]:
                0xD4C4_0006,
                0x0002_0090, // v_cmp_gt_u32_e64 s[6:7], 16, v0
                0xBEC0_2406, // s_and_saveexec_b64 s[64:65], s[6:7]
                0x8AFE_7E40, // s_andn2_b64 exec, s[64:65], exec
                0xBF88_0007, // s_cbranch_execz -> join (+7 dwords)
                0xF408_0701,
                0xFA00_0010, // s_load_dwordx4 s[28:31], s[2:3], 0x10
                0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
                0x7E08_021C, // v_mov_b32 v4, s28
                0x7E0A_021D, // v_mov_b32 v5, s29
                0x7E0C_021E, // v_mov_b32 v6, s30
                0x7E0E_021F, // v_mov_b32 v7, s31
                0xBEFE_0440, // join: s_mov_b64 exec, s[64:65]
                // Full-mask waterfall loop over v[4:7]:
                0xBE94_047E, // s_mov_b64 s[20:21], exec
                0x7E20_0504, // loop: v_readfirstlane_b32 s16, v4
                0x7E22_0505, // v_readfirstlane_b32 s17, v5
                0x7E24_0506, // v_readfirstlane_b32 s18, v6
                0x7E26_0507, // v_readfirstlane_b32 s19, v7
                0xD4E2_0018,
                0x0002_0810, // v_cmp_eq_u64_e64 s[24:25], s[16:17], v[4:5]
                0xD4E2_001A,
                0x0002_0C12, // v_cmp_eq_u64_e64 s[26:27], s[18:19], v[6:7]
                0x8798_1A18, // s_and_b64 s[24:25], s[24:25], s[26:27]
                0xBE98_2418, // s_and_saveexec_b64 s[24:25], s[24:25]
                0xBF80_0000, // s_nop 0
                0xF09C_0F08,
                0x0082_0C02, // image_sample_lz v[12:15], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
                0xBF8C_3F70, // s_waitcnt vmcnt(0)
                0x89FE_187E, // s_xor_b64 exec, exec, s[24:25]
                0xBF89_FFF0, // s_cbranch_execnz -> loop
                0xBEFE_0414, // s_mov_b64 exec, s[20:21]
            ];
            if masked_downstream {
                words.extend([
                    0xD501_000C,
                    (6 << 18) | ((256 + 18) << 9) | (256 + 12), // v_cndmask_b32_e64 v12, v12, v18, s[6:7]
                ]);
            }
            words.extend([
                0xE070_1000,
                0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
                S_ENDPGM,
            ]);
            words
        };
        assert!(waterfall(&build_w64(true), true).is_ok());
        assert_eq!(
            refusal(&build_w64(false), true),
            "a descriptor read from the first lane of computed VGPRs"
        );
    }
}
