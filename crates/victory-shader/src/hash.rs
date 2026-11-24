//! The hash tables of the translator.
//!
//! Every key here is a small integer (a [`crate::ir::Value`], a register, a
//! block index, a tuple of them), and the translator asks about millions of
//! them: one program is tens of thousands of instructions, and a build is
//! tens of thousands of programs. SipHash's setup and finalisation dominate
//! those lookups, so these tables use FxHash's multiply-rotate step instead:
//! no seeding, one multiply per word.
//!
//! This is not a defence against a hostile program: the input is a console
//! shader the user already owns, not a request from the network. Iteration
//! order is unspecified here exactly as it is for `HashMap`, so nothing may
//! depend on it; the one place the translator needs order (`user_data`) uses
//! a `BTreeMap`.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

pub(crate) type Map<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;
pub(crate) type Set<T> = HashSet<T, BuildHasherDefault<Fx>>;

/// FxHash's multiply-rotate step.
#[derive(Default)]
pub(crate) struct Fx(u64);

impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }

    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}
