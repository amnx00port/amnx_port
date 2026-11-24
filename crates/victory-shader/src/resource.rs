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

use std::cell::RefCell;
use std::collections::HashMap;

use crate::ShaderError;
use crate::gfx10::{Fields, NULL, Operand, Valu};
use crate::ir::{Inst, Program, Reg, Value};
use crate::native::{HwStage, RootParam, RootType};

/// An instruction: block and index in [`crate::ir::Block::insts`].
pub type Site = (usize, usize);

/// Relative byte offset `value - base` for multi-term, scaled, subtracted, or
/// phi-joined addresses, plus an optional extra dynamic offset operand (such as
/// SMEM `soffset` or a `global_*` VGPR offset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelOffset {
    pub value: Value,
    pub base: Value,
    pub extra_dynamic: Option<Value>,
}

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
    pub rel: Option<RelOffset>,
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
}

impl Source {
    /// The sources this one picks from (itself unless a phi).
    pub fn leaves(&self) -> Vec<&Source> {
        match self {
            Source::Phi { args, .. } => args.iter().flat_map(Source::leaves).collect(),
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
}

/// Traces every descriptor and scalar address of `p`, which runs as
/// `stage`, to the root parameters in `layout`. Fails on the first one it
/// cannot trace.
pub fn track(p: &Program, layout: &[RootParam], stage: HwStage) -> Result<Resources, ShaderError> {
    let t = Tracer::new(p, layout, stage)?;
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
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let site = (b, i);
            let fail = |what| ShaderError::Unsupported {
                offset: inst.machine.offset,
                what,
            };
            let mut descriptor = |op: &Operand, role| -> Result<(), ShaderError> {
                let source = t.descriptor(inst, op).map_err(fail)?;
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
        }
    }
    Ok(out)
}

/// A dword after copy propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Term {
    Const(u32),
    Value(Value),
}

/// A 32-bit sum: optional root parameter (as its first dword), constant,
/// optional computed term, and whether non-linear/multi-term arithmetic was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sum {
    base: Option<Value>,
    offset: u32,
    dynamic: Option<Value>,
    multi: bool,
}

#[derive(Debug, Clone, Copy)]
struct Trace {
    sum: Sum,
    /// Shallowest active phi index in `visiting` depended on, if any.
    cycle: Option<usize>,
}

impl Trace {
    fn has_ptr(&self) -> bool {
        self.sum.base.is_some() || self.cycle.is_some()
    }
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
    /// Memoized non-cyclic sums by value.
    sums: RefCell<HashMap<Value, Sum>>,
}

const MAX_ROUNDS: usize = 64;
const MAX_JOINS: usize = 8;

impl<'a> Tracer<'a> {
    fn new(p: &'a Program, layout: &'a [RootParam], stage: HwStage) -> Result<Self, ShaderError> {
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
        let mut t = Tracer {
            p,
            layout,
            stage,
            defs,
            phis,
            inputs,
            origin: Vec::new(),
            sums: RefCell::new(HashMap::new()),
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

    fn is_base_root(&self, param: usize) -> bool {
        matches!(
            self.layout[param].root,
            RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap | RootType::Descriptor
        )
    }

    fn or_align_limit(&self, base: Value) -> Option<u32> {
        if self.getpc(base).is_some() {
            return Some(4);
        }
        let reg = self.inputs[base.0 as usize]?;
        let (param, 0) = self.param_of(reg)? else {
            return None;
        };
        match self.layout[param].root {
            RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap => Some(16),
            RootType::Descriptor => Some(4),
            _ => None,
        }
    }

    fn valu_ops(&self, v: Value, names: &[&str]) -> Option<(&'a Inst, &'a Valu)> {
        let inst = self.def(v)?;
        if !names.contains(&inst.name()) {
            return None;
        }
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if f.abs != 0 || f.neg != 0 || f.sdwa.is_some() || f.dpp.is_some() || f.clamp || f.omod != 0
        {
            return None;
        }
        let writes_v = match f.dst? {
            Operand::Vector { reg, .. } => inst.write(Reg::V(reg)) == Some(v),
            Operand::Scalar { code, .. } => inst.write(Reg::S(code)) == Some(v),
            _ => false,
        };
        writes_v.then_some((inst, f))
    }

    fn valu_bin(&self, v: Value, names: &[&str]) -> Option<[Term; 2]> {
        let (inst, f) = self.valu_ops(v, names)?;
        if !matches!(f.dst, Some(Operand::Vector { .. })) {
            return None;
        }
        Some([
            self.vterm(inst, f.src.first()?)?,
            self.vterm(inst, f.src.get(1)?)?,
        ])
    }

    fn valu_tri(&self, v: Value, names: &[&str]) -> Option<[Term; 3]> {
        let (inst, f) = self.valu_ops(v, names)?;
        if !matches!(f.dst, Some(Operand::Vector { .. })) {
            return None;
        }
        Some([
            self.vterm(inst, f.src.first()?)?,
            self.vterm(inst, f.src.get(1)?)?,
            self.vterm(inst, f.src.get(2)?)?,
        ])
    }

    /// Whether a VALU source is wave-uniform (a scalar/constant, or copied from
    /// one through unmodified `v_mov_b32`s).
    fn uniform_valu_src(&self, mut inst: &'a Inst, mut op: &'a Operand) -> bool {
        loop {
            match *op {
                Operand::Scalar { .. } | Operand::Const(_) | Operand::Literal(_) => return true,
                Operand::Vector { reg, .. } => {
                    let Some(val) = inst.read(Reg::V(reg)) else {
                        return false;
                    };
                    let Some((next, f)) = self.valu_ops(val, &["v_mov_b32"]) else {
                        return false;
                    };
                    let Some(src) = f.src.first() else {
                        return false;
                    };
                    inst = next;
                    op = src;
                }
                Operand::Attr { .. } => return false,
            }
        }
    }

    /// Address-local `v_readfirstlane_b32` or `v_readlane_b32` reading `v`.
    fn lane_read(&self, v: Value) -> Option<(Term, bool)> {
        let (inst, f) = self.valu_ops(v, &["v_readfirstlane_b32", "v_readlane_b32"])?;
        if !matches!(f.dst, Some(Operand::Scalar { .. })) {
            return None;
        }
        let src0 = f.src.first()?;
        let uniform = self.uniform_valu_src(inst, src0);
        let t = self.vterm(inst, src0)?;
        Some((t, uniform))
    }

    /// A VA-preserving low-bit mask (`s_and_b32` / `v_and_b32` with `mask & 0xFF == 0xFF`).
    fn va_masked(&self, v: Value) -> Option<Term> {
        let [a, b] = self
            .sop2(v, &["s_and_b32"])
            .map(|(_, t)| t)
            .or_else(|| self.valu_bin(v, &["v_and_b32"]))?;
        match [a, b] {
            [x, Term::Const(mask)] | [Term::Const(mask), x] if mask & 0xFF == 0xFF => Some(x),
            _ => None,
        }
    }

    fn strip_va_mask(&self, mut t: Term) -> Term {
        while let Term::Value(v) = t {
            if let Some(x) = self.va_masked(v) {
                t = x;
            } else if let Some((x, _)) = self.lane_read(v) {
                t = x;
            } else {
                break;
            }
        }
        t
    }

    fn sum(&self, t: Term) -> Sum {
        self.sum_rec(t, &mut Vec::new(), &mut Vec::new()).sum
    }

    fn sum_rec(
        &self,
        t: Term,
        visiting: &mut Vec<Value>,
        frames: &mut Vec<HashMap<Value, Trace>>,
    ) -> Trace {
        let v = match t {
            Term::Const(c) => {
                return Trace {
                    sum: Sum {
                        base: None,
                        offset: c,
                        dynamic: None,
                        multi: false,
                    },
                    cycle: None,
                };
            }
            Term::Value(v) => v,
        };
        if let Some(&sum) = self.sums.borrow().get(&v) {
            return Trace { sum, cycle: None };
        }
        if let Some(depth) = visiting.iter().position(|&x| x == v) {
            return Trace {
                sum: Sum {
                    base: None,
                    offset: 0,
                    dynamic: None,
                    multi: true,
                },
                cycle: Some(depth),
            };
        }
        if let Some(&cached) = frames.last().and_then(|f| f.get(&v)) {
            return cached;
        }
        let (res, cache_global) = self.sum_value(v, visiting, frames);
        if cache_global && res.cycle.is_none() {
            self.sums.borrow_mut().insert(v, res.sum);
        } else if let Some(frame) = frames.last_mut() {
            frame.insert(v, res);
        }
        res
    }

    fn sum_value(
        &self,
        v: Value,
        visiting: &mut Vec<Value>,
        frames: &mut Vec<HashMap<Value, Trace>>,
    ) -> (Trace, bool) {
        let plain_dynamic = Trace {
            sum: Sum {
                base: None,
                offset: 0,
                dynamic: Some(v),
                multi: false,
            },
            cycle: None,
        };
        if let Some(reg) = self.inputs[v.0 as usize] {
            return match self.param_of(reg) {
                Some((param, 0)) if self.is_base_root(param) => (
                    Trace {
                        sum: Sum {
                            base: Some(v),
                            offset: 0,
                            dynamic: None,
                            multi: false,
                        },
                        cycle: None,
                    },
                    true,
                ),
                _ => (plain_dynamic, true),
            };
        }
        if self.getpc(v).is_some() {
            return (
                Trace {
                    sum: Sum {
                        base: Some(v),
                        offset: 0,
                        dynamic: None,
                        multi: false,
                    },
                    cycle: None,
                },
                true,
            );
        }
        if let Some((phi_b, phi_i)) = self.phis[v.0 as usize] {
            let my_depth = visiting.len();
            visiting.push(v);
            frames.push(HashMap::new());
            let mut shared_base: Option<Value> = None;
            let mut outer_cycle: Option<usize> = None;
            let mut touched_outer = false;
            let mut ok = !self.p.blocks[phi_b].phis[phi_i].args.is_empty();
            for &arg in &self.p.blocks[phi_b].phis[phi_i].args {
                let e = self.sum_rec(self.vorigin(arg), visiting, frames);
                if let Some(d) = e.cycle
                    && d < my_depth
                {
                    touched_outer = true;
                }
                if !e.has_ptr() {
                    ok = false;
                    break;
                }
                if let Some(b) = e.sum.base {
                    if shared_base.is_some_and(|sb| sb != b) {
                        ok = false;
                        break;
                    }
                    shared_base = Some(b);
                }
                if let Some(d) = e.cycle
                    && d < my_depth
                {
                    outer_cycle = Some(outer_cycle.map_or(d, |o| o.min(d)));
                }
            }
            frames.pop();
            visiting.pop();
            let res = if ok {
                if let Some(base) = shared_base {
                    Trace {
                        sum: Sum {
                            base: Some(base),
                            offset: 0,
                            dynamic: Some(v),
                            multi: true,
                        },
                        cycle: outer_cycle,
                    }
                } else if let Some(d) = outer_cycle {
                    Trace {
                        sum: Sum {
                            base: None,
                            offset: 0,
                            dynamic: None,
                            multi: true,
                        },
                        cycle: Some(d),
                    }
                } else {
                    plain_dynamic
                }
            } else {
                plain_dynamic
            };
            return (res, !touched_outer);
        }

        let combine_add = |x: Trace, y: Trace| -> Trace {
            if x.has_ptr() && y.has_ptr() {
                return plain_dynamic;
            }
            if !x.has_ptr() && !y.has_ptr() {
                if x.sum.dynamic.is_some() && y.sum.dynamic.is_some() {
                    return plain_dynamic;
                }
                return Trace {
                    sum: Sum {
                        base: None,
                        offset: x.sum.offset.wrapping_add(y.sum.offset),
                        dynamic: x.sum.dynamic.or(y.sum.dynamic),
                        multi: false,
                    },
                    cycle: None,
                };
            }
            let (p, o) = if x.has_ptr() { (x, y) } else { (y, x) };
            let multi = p.sum.multi || (p.sum.dynamic.is_some() && o.sum.dynamic.is_some());
            let dynamic = if multi {
                Some(v)
            } else {
                p.sum.dynamic.or(o.sum.dynamic)
            };
            Trace {
                sum: Sum {
                    base: p.sum.base,
                    offset: p.sum.offset.wrapping_add(o.sum.offset),
                    dynamic,
                    multi,
                },
                cycle: p.cycle,
            }
        };

        let Some(inst) = self.def(v) else {
            return (plain_dynamic, true);
        };

        if let Some((_, f)) = self.valu_ops(v, &["v_mov_b32"])
            && matches!(f.dst, Some(Operand::Vector { .. }))
            && let Some(src_t) = f.src.first().and_then(|s| self.vterm(inst, s))
        {
            return (self.sum_rec(src_t, visiting, frames), true);
        }

        if let Some((src_t, uniform)) = self.lane_read(v) {
            let inner = self.sum_rec(src_t, visiting, frames);
            if !inner.has_ptr() {
                return if inner.sum.dynamic.is_none() {
                    (inner, true)
                } else {
                    (plain_dynamic, true)
                };
            }
            if uniform || (!inner.sum.multi && inner.sum.dynamic.is_none() && inner.cycle.is_none())
            {
                return (inner, true);
            }
            return (
                Trace {
                    sum: Sum {
                        base: inner.sum.base,
                        offset: inner.sum.offset,
                        dynamic: Some(v),
                        multi: true,
                    },
                    cycle: inner.cycle,
                },
                true,
            );
        }

        if let Some([a, b]) = self
            .sop2(v, &["s_add_u32", "s_add_i32"])
            .map(|(_, t)| t)
            .or_else(|| self.valu_bin(v, &["v_add_co_u32", "v_add_nc_u32", "v_add_nc_i32"]))
        {
            let x = self.sum_rec(a, visiting, frames);
            let y = self.sum_rec(b, visiting, frames);
            return (combine_add(x, y), true);
        }

        if inst.name() == "s_addk_i32"
            && let Fields::Sop {
                dst: Some(Operand::Scalar { code, .. }),
                imm: Some(imm),
                ..
            } = &inst.fields
            && inst.writes.first().is_some_and(|w| w.1 == v)
            && let Some(prev) = inst.read(Reg::S(*code)).map(|p| self.origin(p))
        {
            let x = self.sum_rec(prev, visiting, frames);
            let k = Trace {
                sum: Sum {
                    base: None,
                    offset: i32::from(*imm as i16) as u32,
                    dynamic: None,
                    multi: false,
                },
                cycle: None,
            };
            return (combine_add(x, k), true);
        }

        if let Some((inst, [a, b])) = self.sop2(
            v,
            &[
                "s_lshl1_add_u32",
                "s_lshl2_add_u32",
                "s_lshl3_add_u32",
                "s_lshl4_add_u32",
            ],
        ) {
            let shift = u32::from(inst.name().as_bytes()[6] - b'0');
            let x = self.sum_rec(a, visiting, frames);
            let y = self.sum_rec(b, visiting, frames);
            if x.has_ptr() {
                return (plain_dynamic, true);
            }
            if !y.has_ptr() {
                if x.sum.dynamic.is_none() && y.sum.dynamic.is_none() {
                    return (
                        Trace {
                            sum: Sum {
                                base: None,
                                offset: x.sum.offset.wrapping_shl(shift).wrapping_add(y.sum.offset),
                                dynamic: None,
                                multi: false,
                            },
                            cycle: None,
                        },
                        true,
                    );
                }
                return (plain_dynamic, true);
            }
            if x.sum.dynamic.is_none() {
                let k = Trace {
                    sum: Sum {
                        base: None,
                        offset: x.sum.offset.wrapping_shl(shift),
                        dynamic: None,
                        multi: false,
                    },
                    cycle: None,
                };
                return (combine_add(y, k), true);
            }
            return (
                Trace {
                    sum: Sum {
                        base: y.sum.base,
                        offset: y.sum.offset,
                        dynamic: Some(v),
                        multi: true,
                    },
                    cycle: y.cycle,
                },
                true,
            );
        }

        if let Some([a, sh, b]) = self.valu_tri(v, &["v_lshl_add_u32"]) {
            let ta = self.sum_rec(a, visiting, frames);
            let tsh = self.sum_rec(sh, visiting, frames);
            let tb = self.sum_rec(b, visiting, frames);
            if ta.has_ptr() || tsh.has_ptr() {
                return (plain_dynamic, true);
            }
            if !tb.has_ptr() {
                if ta.sum.dynamic.is_none() && tsh.sum.dynamic.is_none() && tb.sum.dynamic.is_none()
                {
                    return (
                        Trace {
                            sum: Sum {
                                base: None,
                                offset: ta
                                    .sum
                                    .offset
                                    .wrapping_shl(tsh.sum.offset & 31)
                                    .wrapping_add(tb.sum.offset),
                                dynamic: None,
                                multi: false,
                            },
                            cycle: None,
                        },
                        true,
                    );
                }
                return (plain_dynamic, true);
            }
            if ta.sum.dynamic.is_none() && tsh.sum.dynamic.is_none() {
                let k = Trace {
                    sum: Sum {
                        base: None,
                        offset: ta.sum.offset.wrapping_shl(tsh.sum.offset & 31),
                        dynamic: None,
                        multi: false,
                    },
                    cycle: None,
                };
                return (combine_add(tb, k), true);
            }
            return (
                Trace {
                    sum: Sum {
                        base: tb.sum.base,
                        offset: tb.sum.offset,
                        dynamic: Some(v),
                        multi: true,
                    },
                    cycle: tb.cycle,
                },
                true,
            );
        }

        if let Some([a, b, c]) = self.valu_tri(v, &["v_add3_u32"]) {
            let ta = self.sum_rec(a, visiting, frames);
            let tb = self.sum_rec(b, visiting, frames);
            let tc = self.sum_rec(c, visiting, frames);
            let ptrs =
                usize::from(ta.has_ptr()) + usize::from(tb.has_ptr()) + usize::from(tc.has_ptr());
            if ptrs > 1 {
                return (plain_dynamic, true);
            }
            return (combine_add(combine_add(ta, tb), tc), true);
        }

        if let Some((minuend, subtrahend)) = self
            .sop2(v, &["s_sub_u32", "s_sub_i32"])
            .map(|(_, [a, b])| (a, b))
            .or_else(|| {
                self.valu_bin(v, &["v_sub_co_u32", "v_sub_nc_u32"])
                    .map(|[a, b]| (a, b))
            })
            .or_else(|| {
                self.valu_bin(v, &["v_subrev_co_u32", "v_subrev_nc_u32"])
                    .map(|[a, b]| (b, a))
            })
        {
            let m = self.sum_rec(minuend, visiting, frames);
            let s = self.sum_rec(subtrahend, visiting, frames);
            if s.has_ptr() {
                return (plain_dynamic, true);
            }
            if !m.has_ptr() {
                if m.sum.dynamic.is_none() && s.sum.dynamic.is_none() {
                    return (
                        Trace {
                            sum: Sum {
                                base: None,
                                offset: m.sum.offset.wrapping_sub(s.sum.offset),
                                dynamic: None,
                                multi: false,
                            },
                            cycle: None,
                        },
                        true,
                    );
                }
                return (plain_dynamic, true);
            }
            return (
                Trace {
                    sum: Sum {
                        base: m.sum.base,
                        offset: m.sum.offset,
                        dynamic: Some(v),
                        multi: true,
                    },
                    cycle: m.cycle,
                },
                true,
            );
        }

        if let Some([a, b]) = self
            .sop2(v, &["s_or_b32"])
            .map(|(_, t)| t)
            .or_else(|| self.valu_bin(v, &["v_or_b32"]))
        {
            let x = self.sum_rec(a, visiting, frames);
            let y = self.sum_rec(b, visiting, frames);
            let or_align = |p: Trace, k: Trace| -> Option<Trace> {
                if p.cycle.is_some() || p.sum.multi || p.sum.dynamic.is_some() {
                    return None;
                }
                let base = p.sum.base?;
                if k.has_ptr() || k.sum.dynamic.is_some() {
                    return None;
                }
                let c = k.sum.offset;
                let lim = self.or_align_limit(base)?;
                ((p.sum.offset & c) == 0 && c < lim).then_some(Trace {
                    sum: Sum {
                        base: Some(base),
                        offset: p.sum.offset | c,
                        dynamic: None,
                        multi: false,
                    },
                    cycle: None,
                })
            };
            if let Some(res) = or_align(x, y).or_else(|| or_align(y, x)) {
                return (res, true);
            }
            return (plain_dynamic, true);
        }

        (plain_dynamic, true)
    }

    /// The `s_getpc_b64` writing `lo` as its low dword.
    fn getpc(&self, lo: Value) -> Option<&'a Inst> {
        let inst = self.def(lo)?;
        (inst.name() == "s_getpc_b64" && inst.writes.first().is_some_and(|w| w.1 == lo))
            .then_some(inst)
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
                        multi: false,
                    },
                )
            }
        }
    }

    /// The address `lo` and `hi` hold, plus `extra` (an instruction's
    /// immediate and offset register).
    fn address(&self, lo: Term, hi: Term, extra: Sum) -> Result<Address, &'static str> {
        let s = self.sum(lo);
        let base = s.base.ok_or_else(|| self.why_no_base(s.dynamic))?;
        let use_rel = s.multi || (s.dynamic.is_some() && extra.dynamic.is_some());
        let lo_val = match lo {
            Term::Value(v) => v,
            Term::Const(_) => unreachable!("base implies value"),
        };
        let rel = use_rel.then_some(RelOffset {
            value: lo_val,
            base,
            extra_dynamic: extra.dynamic,
        });
        let dynamic = if rel.is_some() {
            extra.dynamic.or(Some(lo_val))
        } else {
            s.dynamic.or(extra.dynamic)
        };
        if let Some(pc) = self.getpc(base) {
            // A table in the code: the base is the next instruction's offset.
            let high = pc.writes.get(1).map(|w| w.1);
            let want = high.map_or(Term::Const(0), Term::Value);
            if !self.high_half(hi, high)
                && !(high.is_some() && self.carried(hi, want, &mut Vec::new()))
            {
                return Err("an address with an untracked high half");
            }
            let offset = if rel.is_some() {
                (pc.machine.offset as u32 + 4).wrapping_add(extra.offset)
            } else {
                (pc.machine.offset as u32 + 4)
                    .wrapping_add(s.offset)
                    .wrapping_add(extra.offset)
            };
            return Ok(Address {
                param: CODE,
                offset,
                dynamic,
                rel,
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
        let want = high.map_or(Term::Const(0), Term::Value);
        if !self.high_half(hi, high)
            && !(dynamic.is_some() && self.carried(hi, want, &mut Vec::new()))
        {
            return Err("an address with an untracked high half");
        }
        let offset = if rel.is_some() {
            extra.offset
        } else {
            s.offset.wrapping_add(extra.offset)
        };
        Ok(Address {
            param,
            offset,
            dynamic,
            rel,
        })
    }

    /// `hi` is `high` (0 for a table), possibly plus a carry.
    fn high_half(&self, hi: Term, high: Option<Value>) -> bool {
        let want = high.map_or(Term::Const(0), Term::Value);
        let hi = self.strip_va_mask(hi);
        if hi == want {
            return true;
        }
        let Term::Value(v) = hi else {
            return false;
        };
        let Some([a, b]) = self
            .sop2(v, &["s_addc_u32"])
            .map(|(_, t)| t)
            .or_else(|| self.valu_bin(v, &["v_add_co_ci_u32"]))
        else {
            return false;
        };
        let (a, b) = (self.strip_va_mask(a), self.strip_va_mask(b));
        a == Term::Const(0) && self.high_half(b, high)
            || b == Term::Const(0) && self.high_half(a, high)
    }

    /// `hi` adds something (with carry) to `want` (`high` or 0 for a table),
    /// through carry-adds, adds, subtractions (minuend), VA-preserving masks,
    /// and high-half phis.
    fn carried(&self, hi: Term, want: Term, seen: &mut Vec<Value>) -> bool {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Carry {
            No,
            Cycle,
            Yes,
        }
        fn best(a: Carry, b: Carry) -> Carry {
            match (a, b) {
                (Carry::Yes, _) | (_, Carry::Yes) => Carry::Yes,
                (Carry::Cycle, _) | (_, Carry::Cycle) => Carry::Cycle,
                _ => Carry::No,
            }
        }
        fn rec(t: &Tracer<'_>, hi: Term, want: Term, seen: &mut Vec<Value>) -> Carry {
            let hi = t.strip_va_mask(hi);
            if hi == want {
                return Carry::Yes;
            }
            let Term::Value(v) = hi else {
                return Carry::No;
            };
            if seen.contains(&v) {
                return Carry::Cycle;
            }
            if let Some((phi_b, phi_i)) = t.phis[v.0 as usize] {
                if t.p.blocks[phi_b].phis[phi_i].args.is_empty() {
                    return Carry::No;
                }
                seen.push(v);
                let mut any_yes = false;
                for &arg in &t.p.blocks[phi_b].phis[phi_i].args {
                    match rec(t, t.vorigin(arg), want, seen) {
                        Carry::No => {
                            seen.pop();
                            return Carry::No;
                        }
                        Carry::Yes => any_yes = true,
                        Carry::Cycle => {}
                    }
                }
                seen.pop();
                return if any_yes { Carry::Yes } else { Carry::Cycle };
            }
            if let Some([a, b]) = t
                .sop2(v, &["s_addc_u32", "s_add_u32", "s_add_i32"])
                .map(|(_, args)| args)
                .or_else(|| {
                    t.valu_bin(
                        v,
                        &[
                            "v_add_co_ci_u32",
                            "v_add_nc_u32",
                            "v_add_co_u32",
                            "v_add_nc_i32",
                        ],
                    )
                })
            {
                return best(rec(t, a, want, seen), rec(t, b, want, seen));
            }
            if let Some(minuend) = t
                .sop2(v, &["s_subb_u32", "s_sub_u32", "s_sub_i32"])
                .map(|(_, [a, _])| a)
                .or_else(|| {
                    t.valu_bin(v, &["v_sub_co_ci_u32", "v_sub_nc_u32", "v_sub_co_u32"])
                        .map(|[a, _]| a)
                })
                .or_else(|| {
                    t.valu_bin(
                        v,
                        &["v_subrev_co_ci_u32", "v_subrev_nc_u32", "v_subrev_co_u32"],
                    )
                    .map(|[_, b]| b)
                })
            {
                return rec(t, minuend, want, seen);
            }
            Carry::No
        }
        rec(self, hi, want, seen) == Carry::Yes
    }

    fn why_no_base(&self, dynamic: Option<Value>) -> &'static str {
        let Some(v) = dynamic else {
            return "a constant address";
        };
        if let Some(reg) = self.inputs[v.0 as usize] {
            return if self.param_of(reg).is_some() {
                "an address from a root parameter that is no address"
            } else {
                "an address in an SGPR that is no root parameter"
            };
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

    fn descriptor(&self, inst: &Inst, op: &Operand) -> Result<Source, &'static str> {
        let Operand::Scalar { code, dwords } = *op else {
            return Err("a descriptor not in SGPRs");
        };
        let words = (0..dwords)
            .map(|k| inst.read(Reg::S(code + k)).map(|v| self.origin(v)))
            .collect::<Option<Vec<_>>>()
            .ok_or("a descriptor not in SGPRs")?;
        self.classify(&words, &mut Vec::new())
    }

    /// The source of a descriptor's words; `joins` are the phi blocks
    /// being resolved.
    fn classify(&self, words: &[Term], joins: &mut Vec<usize>) -> Result<Source, &'static str> {
        if let Some(source) = self.phi(words, joins)? {
            return Ok(source);
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
            while let Term::Value(v) = hi {
                if let Some((x, more)) = self.set_bits(v) {
                    (hi, bits) = (x, bits | more);
                } else if let Some(x) = self.va_masked(v) {
                    hi = x;
                } else {
                    break;
                }
            }
            let address = self.address(
                lo,
                hi,
                Sum {
                    base: None,
                    offset: 0,
                    dynamic: None,
                    multi: false,
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
    fn phi(&self, words: &[Term], joins: &mut Vec<usize>) -> Result<Option<Source>, &'static str> {
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
                self.classify(&edge, joins)
            })
            .collect::<Result<_, _>>();
        joins.pop();
        Ok(Some(Source::Phi { block, args: args? }))
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
mod tests {
    use super::*;
    use crate::native::RootType::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn track_words(words: &[u32], layout: &[RootParam], stage: HwStage) -> Resources {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        track(&crate::ir::build(&code, false, 256).unwrap(), layout, stage).unwrap()
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
            rel: None,
        }
    }

    fn try_track_words(
        words: &[u32],
        layout: &[RootParam],
        stage: HwStage,
    ) -> Result<Resources, ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        track(&crate::ir::build(&code, false, 256).unwrap(), layout, stage)
    }

    const fn s_mov(sdst: u8, src0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
    }
    const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
        0x8000_0000
            | (op as u32 & 0x7F) << 23
            | (sdst as u32 & 0x7F) << 16
            | (src1 as u32) << 8
            | src0 as u32
    }
    const fn s_addk_i32(sdst: u8, imm: i16) -> u32 {
        0xB000_0000 | 0x0F << 23 | (sdst as u32 & 0x7F) << 16 | imm as u16 as u32
    }
    const fn sopc(op: u8, src0: u8, src1: u8) -> u32 {
        0xBF00_0000 | (op as u32 & 0x7F) << 16 | (src1 as u32) << 8 | src0 as u32
    }
    const fn sopp(op: u8, simm16: i16) -> u32 {
        0xBF80_0000 | (op as u32 & 0x7F) << 16 | simm16 as u16 as u32
    }
    const fn s_load_x4(dst: u8, base: u8, soffset: u8, offset: i32) -> [u32; 2] {
        [
            0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
            (soffset as u32) << 25 | (offset as u32 & 0x1F_FFFF),
        ]
    }
    const fn s_buf_load(dst: u8, base: u8) -> [u32; 2] {
        [
            0x3D << 26 | 0x08 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
            125 << 25,
        ]
    }
    const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
        0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
    }
    const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
        (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
    }
    const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
        [
            0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
            (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
        ]
    }
    const fn vop3b(op: u16, vdst: u8, sdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
        [
            0x35 << 26 | (op as u32 & 0x3FF) << 16 | (sdst as u32 & 0x7F) << 8 | vdst as u32,
            (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
        ]
    }
    const fn global_load(vdst: u8, vaddr: u8, saddr: u8, offset: i16) -> [u32; 2] {
        [
            0x37 << 26 | 0x0C << 18 | 2 << 14 | (offset as u32 & 0xFFF),
            (vdst as u32) << 24 | (saddr as u32 & 0x7F) << 16 | vaddr as u32,
        ]
    }
    const fn sgpr(n: u8) -> u16 {
        n as u16
    }
    const fn vgpr(n: u8) -> u16 {
        256 + n as u16
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
    fn untraceable_addresses_fail_with_a_reason() {
        let code: Vec<u8> = [0xF408_0001, 0xFA00_0000, S_ENDPGM] // s_load_dwordx4 s[0:3], s[2:3]
            .iter()
            .flat_map(|w: &u32| w.to_le_bytes())
            .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let err = track(&p, &[param(Table, 9, 1)], HwStage::Ps).unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 0,
                what: "an address in an SGPR that is no root parameter"
            }
        );
    }

    #[test]
    fn root_constants_dword0_is_offset_not_base() {
        // s0 = RootType::Constants (dword 0), s4 = Table, s[6:7] = Descriptor.
        let layout = [
            param(Constants, 0, 2),
            param(Table, 4, 1),
            param(Descriptor, 6, 2),
        ];
        // Added to Table (s_add_u32 s8, s0, s4) and Descriptor (s_add_u32 s10, s0, s6; s_addc_u32 s11, s7, 0),
        // plus passed directly as soffset and global vaddr.
        let load_table = s_load_x4(12, 8, 125, 0);
        let load_soffset = s_load_x4(16, 4, 0, 0);
        let load_desc = global_load(0, 0, 10, 0);
        let load_vaddr = global_load(1, 0, 6, 0);
        let r = track_words(
            &[
                sop2(0x00, 8, 0, 4), // s_add_u32 s8, s0, s4
                s_mov(9, 0x80),      // s_mov_b32 s9, 0
                load_table[0],
                load_table[1],
                s_mov(5, 0x80), // s_mov_b32 s5, 0
                load_soffset[0],
                load_soffset[1],
                sop2(0x00, 10, 0, 6),    // s_add_u32 s10, s0, s6
                sop2(0x04, 11, 7, 0x80), // s_addc_u32 s11, s7, 0
                vop1(0x001, 0, sgpr(0)), // v_mov_b32 v0, s0
                load_desc[0],
                load_desc[1],
                load_vaddr[0],
                load_vaddr[1],
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        );
        assert_eq!(r.scalar.len(), 4);
        assert_eq!(r.scalar[0].1.param, 1);
        assert!(r.scalar[0].1.dynamic.is_some());
        assert_eq!(r.scalar[1].1.param, 1);
        assert!(r.scalar[1].1.dynamic.is_some());
        assert_eq!(r.scalar[2].1.param, 2);
        assert!(r.scalar[2].1.dynamic.is_some());
        assert_eq!(r.scalar[3].1.param, 2);
        assert!(r.scalar[3].1.dynamic.is_some());

        // RootType::Constants dword 0 alone is NOT a base address.
        let only_const = s_load_x4(8, 0, 125, 0);
        let err = try_track_words(
            &[s_mov(1, 0x80), only_const[0], only_const[1], S_ENDPGM],
            &[param(Constants, 0, 2)],
            HwStage::Cs,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 4,
                what: "an address from a root parameter that is no address"
            }
        );
    }

    #[test]
    fn two_term_dynamic_addresses_produce_rel_offset() {
        let layout = [
            param(Constants, 0, 2), // s0 = dyn1, s1 = dyn2
            param(Table, 4, 1),     // s4 = table
        ];
        // 1) (base + dyn1) + dyn2 via two s_add_u32s.
        // 2) (base + 0x10 + dyn1) with soffset = dyn2 and immediate offset = 0x20.
        let load1 = s_load_x4(12, 8, 125, 0);
        let load2 = s_load_x4(16, 10, 1, 0x20);
        let r = track_words(
            &[
                sop2(0x00, 8, 4, 0), // s_add_u32 s8, s4, s0
                sop2(0x00, 8, 8, 1), // s_add_u32 s8, s8, s1
                s_mov(9, 0x80),      // s_mov_b32 s9, 0
                load1[0],
                load1[1],
                sop2(0x00, 10, 4, 0x90), // s_add_u32 s10, s4, 16
                sop2(0x00, 10, 10, 0),   // s_add_u32 s10, s10, s0
                s_mov(11, 0x80),         // s_mov_b32 s11, 0
                load2[0],
                load2[1],
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        );
        let base_val = r.params.iter().find(|p| p.1 == 1).unwrap().0;
        let dyn2_val = r.params.iter().find(|p| p.1 == 0 && p.2 == 1).unwrap().0;

        let a0 = r.scalar[0].1;
        assert_eq!(a0.param, 1);
        assert_eq!(a0.offset, 0);
        assert!(a0.dynamic.is_some());
        let rel0 = a0.rel.expect("two-term add needs rel");
        assert_eq!(rel0.base, base_val);
        assert_eq!(rel0.extra_dynamic, None);
        assert_eq!(a0.dynamic, Some(rel0.value));

        let a1 = r.scalar[1].1;
        assert_eq!(a1.param, 1);
        // s_offset (0x10) is inside rel_offset(value, base), so offset holds only extra (0x20).
        assert_eq!(a1.offset, 0x20);
        let rel1 = a1.rel.expect("lo dyn + soffset dyn needs rel");
        assert_eq!(rel1.base, base_val);
        assert_eq!(rel1.extra_dynamic, Some(dyn2_val));
        assert_eq!(a1.dynamic, Some(dyn2_val));
    }

    #[test]
    fn scaled_subtracted_add3_addk_and_aligned_or_addresses() {
        let layout = [
            param(Constants, 0, 2),  // s0 = dyn1, s1 = dyn2
            param(Table, 4, 1),      // s4 = table
            param(Descriptor, 6, 2), // s[6:7] = desc
        ];
        // Positive cases:
        // - s_addk_i32 (+0x10 and -0x4)
        // - s_lshl{1..4}_add_u32 (base on src1)
        // - v_lshl_add_u32 (base on src2)
        // - v_add3_u32 (base + dyn1 + dyn2)
        // - s_sub_u32, s_sub_i32, v_sub_co_u32, v_sub_nc_u32, v_subrev_co_u32, v_subrev_nc_u32 (minuend)
        // - aligned s_or_b32 (table | 8, (table + 8) | 4, desc | 2) and v_or_b32 (table | 8)
        // - uniform v_mov_b32 -> v_readfirstlane_b32 and v_readlane_b32
        let mut words = vec![
            s_mov(9, 0x80), // s9 = 0 (table hi)
            s_mov(8, 4),    // s8 = table
            s_addk_i32(8, 16),
            s_addk_i32(8, -4),
        ];
        words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [0] offset = 12
        for op in [0x2Eu8, 0x2F, 0x30, 0x31] {
            words.push(sop2(op, 8, 0, 4));
            words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [1..=4]
        }
        // v_lshl_add_u32 v2, s0, 2, s4 -> global_load v3, v[2:3], off (with v3 = 0)
        words.push(vop1(0x001, 3, 0x80)); // v3 = 0
        words.extend_from_slice(&vop3(0x346, 2, sgpr(0), 0x82, sgpr(4)));
        words.extend_from_slice(&global_load(4, 2, 125, 0)); // [5]
        // v_add3_u32 v2, s0, s4, s1 -> global_load
        words.extend_from_slice(&vop3(0x36D, 2, sgpr(0), sgpr(4), sgpr(1)));
        words.extend_from_slice(&global_load(4, 2, 125, 0)); // [6]
        // s_sub_u32 and s_sub_i32 on table minuend
        words.push(sop2(0x01, 8, 4, 0));
        words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [7]
        words.push(sop2(0x03, 8, 4, 0));
        words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [8]
        // v_sub_co_u32, v_sub_nc_u32, v_subrev_co_u32, v_subrev_nc_u32
        words.push(vop1(0x001, 5, sgpr(0))); // v5 = dyn1
        words.push(vop1(0x001, 6, sgpr(4))); // v6 = table
        words.extend_from_slice(&vop3b(0x310, 2, 106, vgpr(6), vgpr(5), 0));
        words.extend_from_slice(&global_load(4, 2, 125, 0)); // [9]
        words.push(vop2(0x26, 2, vgpr(6), 5)); // v_sub_nc_u32 v2, v6, v5
        words.extend_from_slice(&global_load(4, 2, 125, 0)); // [10]
        words.extend_from_slice(&vop3b(0x319, 2, 106, vgpr(5), vgpr(6), 0));
        words.extend_from_slice(&global_load(4, 2, 125, 0)); // [11]
        words.push(vop2(0x27, 2, vgpr(5), 6)); // v_subrev_nc_u32 v2, v5, v6
        words.extend_from_slice(&global_load(4, 2, 125, 0)); // [12]
        // Aligned OR: table | 8, (table + 8) | 4, desc | 2, v_or_b32(table, 8)
        words.push(sop2(0x10, 8, 4, 0x88)); // s_or_b32 s8, s4, 8
        words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [13] offset 8
        words.push(sop2(0x10, 8, 8, 0x84)); // s_or_b32 s8, s8, 4
        words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [14] offset 12
        words.push(sop2(0x10, 10, 6, 0x82)); // s_or_b32 s10, s6, 2
        words.push(s_mov(11, 7));
        words.extend_from_slice(&global_load(4, 3, 10, 0)); // [15] desc offset 2
        words.push(vop2(0x1C, 2, sgpr(4), 7)); // v7 needs to be 8:
        words.pop();
        words.push(vop1(0x001, 7, 0x88)); // v7 = 8
        words.push(vop2(0x1C, 2, sgpr(4), 7)); // v_or_b32 v2, s4, v7? wait v7 is not Const in copy-prop
        // Use inline constant 8 as src0 in v_or_b32 v2, 8, v6 (v6 = table)
        words.pop();
        words.push(vop2(0x1C, 2, 0x88, 6));
        words.extend_from_slice(&global_load(4, 2, 125, 0)); // [16] table offset 8
        // Uniform v_mov_b32 -> v_readfirstlane_b32 / v_readlane_b32
        words.push(vop1(0x001, 7, sgpr(4))); // v7 = s4
        words.push(vop1(0x002, 8, vgpr(7))); // v_readfirstlane_b32 s8, v7
        words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [17]
        words.extend_from_slice(&vop3(0x360, 8, vgpr(7), 0x80, 0)); // v_readlane_b32 s8, v7, 0
        words.extend_from_slice(&s_load_x4(12, 8, 125, 0)); // [18]
        words.push(S_ENDPGM);

        let r = track_words(&words, &layout, HwStage::Cs);
        assert_eq!(r.scalar.len(), 19);
        assert_eq!(r.scalar[0].1, at(1, 12));
        for idx in 1..=6 {
            assert!(r.scalar[idx].1.rel.is_some(), "scalar[{idx}] missing rel");
        }
        for idx in 7..=12 {
            assert!(r.scalar[idx].1.rel.is_some(), "sub[{idx}] missing rel");
        }
        assert_eq!(r.scalar[13].1, at(1, 8));
        assert_eq!(r.scalar[14].1, at(1, 12));
        assert_eq!(r.scalar[15].1, at(2, 2));
        assert_eq!(r.scalar[16].1, at(1, 8));
        assert_eq!(r.scalar[17].1, at(1, 0));
        assert_eq!(r.scalar[18].1, at(1, 0));

        // Negative cases that MUST fail:
        let bad_cases: &[&[u32]] = &[
            // Base on shifted src0 of s_lshl1_add_u32
            &[
                s_mov(9, 0x80),
                sop2(0x2E, 8, 4, 0),
                s_load_x4(12, 8, 125, 0)[0],
                s_load_x4(12, 8, 125, 0)[1],
                S_ENDPGM,
            ],
            // Base on shifted src0 of v_lshl_add_u32
            &[
                vop1(0x001, 3, 0x80),
                vop3(0x346, 2, sgpr(4), 0x82, sgpr(0))[0],
                vop3(0x346, 2, sgpr(4), 0x82, sgpr(0))[1],
                global_load(4, 2, 125, 0)[0],
                global_load(4, 2, 125, 0)[1],
                S_ENDPGM,
            ],
            // Two bases in v_add3_u32
            &[
                vop1(0x001, 3, 0x80),
                vop3(0x36D, 2, sgpr(4), sgpr(6), sgpr(0))[0],
                vop3(0x36D, 2, sgpr(4), sgpr(6), sgpr(0))[1],
                global_load(4, 2, 125, 0)[0],
                global_load(4, 2, 125, 0)[1],
                S_ENDPGM,
            ],
            // Base on subtrahend (const - base) in s_sub_u32
            &[
                s_mov(9, 0x80),
                sop2(0x01, 8, 0x90, 4),
                s_load_x4(12, 8, 125, 0)[0],
                s_load_x4(12, 8, 125, 0)[1],
                S_ENDPGM,
            ],
            // Base on subtrahend in v_subrev_nc_u32 (vsrc1 - src0 = 16 - table)
            &[
                vop1(0x001, 3, 0x80),
                vop1(0x001, 5, 0x90),      // v5 = 16
                vop2(0x27, 2, sgpr(4), 5), // v2 = v5 - s4
                global_load(4, 2, 125, 0)[0],
                global_load(4, 2, 125, 0)[1],
                S_ENDPGM,
            ],
            // Base on subtrahend in v_sub_nc_u32 (src0 - vsrc1 = 16 - table)
            &[
                vop1(0x001, 3, 0x80),
                vop1(0x001, 5, sgpr(4)), // v5 = table
                vop2(0x26, 2, 0x90, 5),  // v2 = 16 - v5
                global_load(4, 2, 125, 0)[0],
                global_load(4, 2, 125, 0)[1],
                S_ENDPGM,
            ],
            // Unaligned / out-of-range OR: table | 16
            &[
                s_mov(9, 0x80),
                sop2(0x10, 8, 4, 0x90),
                s_load_x4(12, 8, 125, 0)[0],
                s_load_x4(12, 8, 125, 0)[1],
                S_ENDPGM,
            ],
            // Overlapping OR: (table + 4) | 4
            &[
                s_mov(9, 0x80),
                sop2(0x00, 8, 4, 0x84),
                sop2(0x10, 8, 8, 0x84),
                s_load_x4(12, 8, 125, 0)[0],
                s_load_x4(12, 8, 125, 0)[1],
                S_ENDPGM,
            ],
            // Descriptor | 4 (>= 4 limit)
            &[
                sop2(0x10, 10, 6, 0x84),
                s_mov(11, 7),
                s_load_x4(12, 10, 125, 0)[0],
                s_load_x4(12, 10, 125, 0)[1],
                S_ENDPGM,
            ],
            // Dynamic term + OR: (table + dyn) | 4
            &[
                s_mov(9, 0x80),
                sop2(0x00, 8, 4, 0),
                sop2(0x10, 8, 8, 0x84),
                s_load_x4(12, 8, 125, 0)[0],
                s_load_x4(12, 8, 125, 0)[1],
                S_ENDPGM,
            ],
        ];
        for (idx, words) in bad_cases.iter().enumerate() {
            assert!(
                try_track_words(words, &layout, HwStage::Cs).is_err(),
                "bad case {idx} unexpectedly succeeded"
            );
        }
    }

    #[test]
    fn high_half_carry_va_mask_and_table_carry() {
        let layout = [
            param(Constants, 0, 2),  // s0, s1
            param(Table, 4, 1),      // s4
            param(Descriptor, 6, 2), // s[6:7]
        ];
        // 1) Constant-offset v_add_co_u32 + v_add_co_ci_u32(s7, 0, vcc).
        // 2) Table 64-bit dynamic carry: s_mov_b32 s9, 0; s_add_u32 s8, s4, s0; s_addc_u32 s9, s9, s1.
        // 3) VA mask s_and_b32 hi, 0xFFFF on global_load_dword.
        // 4) VA mask s_and_b32 hi, 0xFFFF inside Source::Built before s_or_b32 0x100000.
        let r = track_words(
            &[
                vop3b(0x30F, 2, 106, sgpr(6), 0x90, 0)[0],
                vop3b(0x30F, 2, 106, sgpr(6), 0x90, 0)[1],
                vop3b(0x128, 3, 106, sgpr(7), 0x80, sgpr(106))[0],
                vop3b(0x128, 3, 106, sgpr(7), 0x80, sgpr(106))[1],
                global_load(4, 2, 125, 0)[0],
                global_load(4, 2, 125, 0)[1],
                s_mov(9, 0x80),
                sop2(0x00, 8, 4, 0),
                sop2(0x04, 9, 9, 1),
                s_load_x4(12, 8, 125, 0)[0],
                s_load_x4(12, 8, 125, 0)[1],
                sop2(0x00, 10, 6, 0),
                sop2(0x04, 11, 7, 0x80),
                sop2(0x0E, 11, 11, 0xFF),
                0x0000_FFFF, // s_and_b32 s11, s11, 0xFFFF
                vop1(0x001, 5, 0x80),
                global_load(4, 5, 10, 0)[0],
                global_load(4, 5, 10, 0)[1],
                sop2(0x0E, 1, 7, 0xFF),
                0x0000_FFFF, // s_and_b32 s1, s7, 0xFFFF
                sop2(0x10, 1, 1, 0xFF),
                0x0010_0000, // s_or_b32 s1, s1, 0x100000
                s_mov(0, 6),
                s_mov(2, 0xC1),
                s_mov(3, 0xFF),
                0x1101_4FAC,
                s_buf_load(0, 0)[0],
                s_buf_load(0, 0)[1],
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        );
        assert_eq!(r.scalar[0].1, at(2, 16));
        assert_eq!(r.scalar[1].1.param, 1);
        assert!(r.scalar[1].1.dynamic.is_some());
        assert_eq!(r.scalar[2].1.param, 2);
        assert_eq!(
            r.descriptors[0].source,
            Source::Built {
                address: at(2, 0),
                bits: 0x10_0000,
                words: [u32::MAX, 0x1101_4FAC],
            }
        );

        // Invalid VA mask (0x0F -> 0x0F & 0xFF != 0xFF) MUST fail.
        let err_mask = try_track_words(
            &[
                sop2(0x0E, 9, 7, 0x8F), // s_and_b32 s9, s7, 15
                s_mov(8, 6),
                vop1(0x001, 5, 0x80),
                global_load(4, 5, 8, 0)[0],
                global_load(4, 5, 8, 0)[1],
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        )
        .unwrap_err();
        assert_eq!(
            err_mask,
            ShaderError::Unsupported {
                offset: 12,
                what: "an address with an untracked high half"
            }
        );

        // Unrelated high half SGPR MUST fail (keep s7 referenced so Descriptor high half exists).
        let err_hi = try_track_words(
            &[
                s_mov(12, 7),
                s_mov(8, 6),
                s_mov(9, 0), // unrelated constant s0 as hi
                vop1(0x001, 5, 0x80),
                global_load(4, 5, 8, 0)[0],
                global_load(4, 5, 8, 0)[1],
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        )
        .unwrap_err();
        assert_eq!(
            err_hi,
            ShaderError::Unsupported {
                offset: 16,
                what: "an address with an untracked high half"
            }
        );
    }

    #[test]
    fn phi_joined_and_loop_carried_addresses() {
        let layout = [
            param(Constants, 0, 1),  // s0
            param(Descriptor, 2, 2), // s[2:3]
            param(Descriptor, 4, 2), // s[4:5]
        ];
        // 1) If/else diamond joining (s2 + 4) and (s2 + 8).
        let r_if = track_words(
            &[
                sopc(0x06, 0, 0x80),    // s_cmp_eq_u32 s0, 0
                sopp(0x05, 2),          // s_cbranch_scc1 -> +2
                sop2(0x00, 8, 2, 0x84), // s8 = s2 + 4
                sopp(0x02, 1),          // s_branch -> join
                sop2(0x00, 8, 2, 0x88), // s8 = s2 + 8
                s_mov(9, 3),            // join: s9 = s3
                vop1(0x001, 0, 0x80),
                global_load(1, 0, 8, 0)[0],
                global_load(1, 0, 8, 0)[1],
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        );
        let base1 = r_if.params.iter().find(|p| p.1 == 1 && p.2 == 0).unwrap().0;
        let rel_if = r_if.scalar[0].1.rel.expect("if/else phi needs rel");
        assert_eq!(r_if.scalar[0].1.param, 1);
        assert_eq!(rel_if.base, base1);

        // 2) Loop-carried pointer induction phi (s_add_u32 + s_addc_u32 back-edge).
        let r_loop = track_words(
            &[
                s_mov(8, 2),                // s8 = s2
                s_mov(9, 3),                // s9 = s3
                s_mov(10, 0x80),            // s10 = 0
                vop1(0x001, 0, 0x80),       // v0 = 0
                global_load(1, 0, 8, 0)[0], // loop:
                global_load(1, 0, 8, 0)[1],
                sop2(0x00, 8, 8, 0x84), // s8 += 4
                sop2(0x04, 9, 9, 0x80), // s9 += carry
                sop2(0x00, 10, 10, 0x81),
                sopc(0x0A, 10, 0x84),
                sopp(0x05, -7), // s_cbranch_scc1 loop
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        );
        let base_loop = r_loop
            .params
            .iter()
            .find(|p| p.1 == 1 && p.2 == 0)
            .unwrap()
            .0;
        let rel_loop = r_loop.scalar[0].1.rel.expect("loop phi needs rel");
        assert_eq!(r_loop.scalar[0].1.param, 1);
        assert_eq!(rel_loop.base, base_loop);

        // 3) Negative: mismatched-base phi joining s2 (param 1) and s4 (param 2) MUST fail.
        let err = try_track_words(
            &[
                sopc(0x06, 0, 0x80), // s_cmp_eq_u32 s0, 0
                sopp(0x05, 3),       // s_cbranch_scc1 -> else
                s_mov(8, 2),
                s_mov(9, 3),
                sopp(0x02, 2), // s_branch -> join
                s_mov(8, 4),
                s_mov(9, 5),
                vop1(0x001, 0, 0x80),
                global_load(1, 0, 8, 0)[0],
                global_load(1, 0, 8, 0)[1],
                S_ENDPGM,
            ],
            &layout,
            HwStage::Cs,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 32,
                what: "an address from a phi"
            }
        );
    }
}
