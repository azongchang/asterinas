// SPDX-License-Identifier: MPL-2.0

//! Per-sync-group slot queues.
//!
//! Linux VSync uses per-CPU lock-free lists. In this in-memory Rust pass we
//! model the same producer/consumer behavior with Asterinas synchronization
//! primitives while preserving ordering guarantees required by `sync_until`.

use alloc::{
    collections::{BTreeSet, VecDeque},
    vec::Vec,
};
use core::sync::atomic::{AtomicU64, Ordering};

use ostd::sync::{SpinLock, WaitQueue};

use super::*;

/// One producer/consumer slot for a sync group.
pub struct VsyncSlot {
    records: SpinLock<VecDeque<Record>>,
    fences: SpinLock<VecDeque<VsyncFence>>,
    inflight: SpinLock<BTreeSet<VsyncSeq>>,
    inflight_wait: WaitQueue,
    last_published_seq: AtomicU64,
}

impl Default for VsyncSlot {
    fn default() -> Self {
        Self {
            records: SpinLock::new(VecDeque::new()),
            fences: SpinLock::new(VecDeque::new()),
            inflight: SpinLock::new(BTreeSet::new()),
            inflight_wait: WaitQueue::new(),
            last_published_seq: AtomicU64::new(0),
        }
    }
}

/// Per-vspace slot array.
pub struct VsyncPcpu {
    pub slots: [VsyncSlot; VSYNC_NUM_SYNC_GROUPS],
}

impl core::fmt::Debug for VsyncPcpu {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VsyncPcpu").finish_non_exhaustive()
    }
}

impl Default for VsyncPcpu {
    fn default() -> Self {
        Self {
            slots: core::array::from_fn(|_| VsyncSlot::default()),
        }
    }
}

impl VsyncSlot {
    /// Marks a sequence as in-flight.
    pub fn begin_inflight(&self, seq: VsyncSeq) {
        self.inflight.lock().insert(seq);
    }

    /// Marks a sequence as no longer in-flight.
    pub fn finish_inflight(&self, seq: VsyncSeq) {
        if self.inflight.lock().remove(&seq) {
            self.inflight_wait.wake_all();
        }
    }

    /// Waits until no in-flight sequence less than or equal to `target_seq` exists.
    pub fn wait_inflight_until(&self, target_seq: VsyncSeq) {
        self.inflight_wait.wait_until(|| {
            let min = self.inflight.lock().first().copied();
            (min.is_none() || min > Some(target_seq)).then_some(())
        });
    }

    /// Publishes one completed record.
    pub fn publish_record(&self, record: Record) {
        self.last_published_seq
            .fetch_max(record.seq(), Ordering::Relaxed);
        self.records.lock().push_back(record);
    }

    /// Publishes one fence record.
    pub fn publish_fence(&self, fence: VsyncFence) {
        self.last_published_seq
            .fetch_max(fence.target_seq, Ordering::Relaxed);
        self.fences.lock().push_back(fence);
    }

    /// Drains records with sequence not greater than `target_seq`.
    ///
    /// Returns at most `max_count` records in oldest-first order.
    pub fn drain_until(&self, target_seq: VsyncSeq, max_count: usize) -> Vec<Record> {
        let mut records = self.records.lock();
        let mut drained = Vec::with_capacity(max_count.min(records.len()));
        while drained.len() < max_count {
            let Some(front) = records.front() else {
                break;
            };
            if front.seq() > target_seq {
                break;
            }
            if let Some(record) = records.pop_front() {
                drained.push(record);
            }
        }
        drained
    }

    /// Drains all currently queued fences.
    pub fn drain_fences(&self) -> Vec<VsyncFence> {
        self.fences.lock().drain(..).collect()
    }

    /// Returns the most recent published sequence.
    pub fn last_published_seq(&self) -> VsyncSeq {
        self.last_published_seq.load(Ordering::Acquire)
    }

    /// Returns current queued record count.
    pub fn queued_records(&self) -> usize {
        self.records.lock().len()
    }
}
