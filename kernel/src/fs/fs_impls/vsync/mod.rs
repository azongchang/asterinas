// SPDX-License-Identifier: MPL-2.0

#![allow(dead_code)]

//! In-memory VSync implementation for Asterinas.
//!
//! This module ports the core VSync ideas from the Linux implementation
//! (domain-aware `sync_until`, per-sync-group batching, shard distribution),
//! while intentionally limiting this pass to filesystem-internal integration.
//! VFS syscall plumbing and async worker threads are still deferred.
//!
//! Design notes:
//! - Rust-first ownership and synchronization (`Arc`, `SpinLock`, `Mutex`, `WaitQueue`)
//! - No `unsafe` code and no raw-pointer state machines
//! - C-like public API names are preserved where useful for future filesystem integration
//!   (notably ext5-side adapter work)

pub const VSYNC_NUM_SHARDS: usize = 8;
pub const VSYNC_NUM_SYNC_GROUPS: usize = 8;
pub const VSYNC_COALESCE_HASH_BITS: usize = 8;
pub const VSYNC_COALESCE_HASH_SIZE: usize = 1 << VSYNC_COALESCE_HASH_BITS;
pub const VSYNC_CKPT_HASH_BITS: usize = 14;
pub const VSYNC_CKPT_HASH_SIZE: usize = 1 << VSYNC_CKPT_HASH_BITS;
pub const VSYNC_CKPT_ONLINE_HASH_BITS: usize = 17;
pub const VSYNC_CKPT_ONLINE_HASH_SIZE: usize = 1 << VSYNC_CKPT_ONLINE_HASH_BITS;
pub const VSYNC_CKPT_POOL_SIZE: usize = 131_072;
pub const VSYNC_DOMAIN_HASH_BITS: usize = 8;
pub const VSYNC_DOMAIN_HASH_SIZE: usize = 1 << VSYNC_DOMAIN_HASH_BITS;

pub const VSYNC_BATCH_THRESHOLD: usize = 64;
pub const VSYNC_PENDING_HIGH_MARK: usize = 4096;
pub const VSYNC_JOURNAL_PRESSURE_NUM: u64 = 3;
pub const VSYNC_JOURNAL_PRESSURE_DEN: u64 = 4;

pub type GfpT = u32;
pub type VsyncSeq = u64;

mod coalesce;
mod pcpu_slot;
mod record;
mod shard;
mod sync_group;
mod vspace;

pub use coalesce::*;
pub use pcpu_slot::*;
pub use record::*;
pub use shard::*;
pub use sync_group::*;
pub use vspace::*;

#[cfg(ktest)]
mod test;
