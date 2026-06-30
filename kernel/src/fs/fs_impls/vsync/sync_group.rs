// SPDX-License-Identifier: MPL-2.0

//! Sync-group state and helpers.

use alloc::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::Arc,
    vec::Vec,
};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use ostd::sync::{SpinLock, WaitQueue};

use super::*;
use crate::error::Errno;

/// Per-domain durability state in one sync group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VsyncDomainState {
    pub domain_id: u32,
    pub stable_seq: VsyncSeq,
}

/// A pending durability fence.
#[derive(Clone, Debug)]
pub struct VsyncFence {
    pub token: Arc<CountdownLatch>,
    pub domain_id: u32,
    pub target_seq: VsyncSeq,
}

/// Ordered completion state for submitted transactions.
pub struct VsyncCompletionState {
    in_flight: SpinLock<VecDeque<SgTransaction>>,
    next_complete_id: AtomicU64,
    in_flight_count: AtomicU64,
    drain_wq: WaitQueue,
}

impl core::fmt::Debug for VsyncCompletionState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VsyncCompletionState")
            .field("in_flight", &self.in_flight)
            .field("next_complete_id", &self.next_complete_id)
            .field("in_flight_count", &self.in_flight_count)
            .finish_non_exhaustive()
    }
}

impl VsyncCompletionState {
    /// Creates a new completion state.
    pub fn new() -> Self {
        Self {
            in_flight: SpinLock::new(VecDeque::new()),
            next_complete_id: AtomicU64::new(0),
            in_flight_count: AtomicU64::new(0),
            drain_wq: WaitQueue::new(),
        }
    }
}

impl Default for VsyncCompletionState {
    fn default() -> Self {
        Self::new()
    }
}

impl VsyncCompletionState {
    /// Enqueues one transaction into the in-flight queue.
    pub fn push_inflight(&self, txn: SgTransaction) {
        self.in_flight.lock().push_back(txn);
        self.in_flight_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Drains all currently in-flight transactions.
    pub fn drain_inflight(&self) -> Vec<SgTransaction> {
        let mut queue = self.in_flight.lock();
        let drained = queue.drain(..).collect::<Vec<_>>();
        self.in_flight_count
            .fetch_sub(drained.len() as u64, Ordering::Relaxed);
        self.drain_wq.wake_all();
        drained
    }

    /// Returns current in-flight transaction count.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight_count.load(Ordering::Acquire) as usize
    }

    /// Waits until all in-flight transactions are drained.
    pub fn wait_drain(&self) {
        self.drain_wq
            .wait_until(|| (self.in_flight_count() == 0).then_some(()));
    }

    /// Advances `next_complete_id`.
    pub fn set_next_complete_id(&self, txn_id: u64) {
        self.next_complete_id.store(txn_id, Ordering::Release);
    }

    /// Returns `next_complete_id`.
    pub fn next_complete_id(&self) -> u64 {
        self.next_complete_id.load(Ordering::Acquire)
    }
}

/// One sync group.
#[derive(Debug)]
pub struct VsyncSyncGroup {
    pub sg_id: u32,
    domains: SpinLock<BTreeMap<u32, VsyncSeq>>,
    pending_files: SpinLock<BTreeSet<u64>>,
    commit_lock: SpinLock<()>,
    commit_delegated: AtomicBool,
    stable_seq: AtomicU64,
    fence_list: SpinLock<VecDeque<VsyncFence>>,
    next_txn_id: AtomicU64,
    pub completion: VsyncCompletionState,
    pub commit_pending: AtomicU64,
    pub pending_record_count: AtomicU64,
    pub stopping: AtomicBool,
    pub commits: AtomicU64,
    pub fences_satisfied: AtomicU64,
    pub records_processed: AtomicU64,
    pub txns_created: AtomicU64,
    last_error: SpinLock<Option<(Errno, VsyncSeq)>>,
}

impl VsyncSyncGroup {
    /// Creates a sync group with the given ID.
    pub fn new(sg_id: u32) -> Self {
        Self {
            sg_id,
            domains: SpinLock::new(BTreeMap::new()),
            pending_files: SpinLock::new(BTreeSet::new()),
            commit_lock: SpinLock::new(()),
            commit_delegated: AtomicBool::new(false),
            stable_seq: AtomicU64::new(0),
            fence_list: SpinLock::new(VecDeque::new()),
            next_txn_id: AtomicU64::new(0),
            completion: VsyncCompletionState::default(),
            commit_pending: AtomicU64::new(0),
            pending_record_count: AtomicU64::new(0),
            stopping: AtomicBool::new(false),
            commits: AtomicU64::new(0),
            fences_satisfied: AtomicU64::new(0),
            records_processed: AtomicU64::new(0),
            txns_created: AtomicU64::new(0),
            last_error: SpinLock::new(None),
        }
    }

    /// Returns this group's stable sequence.
    pub fn stable_seq(&self) -> VsyncSeq {
        self.stable_seq.load(Ordering::Acquire)
    }

    /// Returns one domain entry, creating it if missing.
    pub fn get_domain(&self, domain_id: u32) -> VsyncDomainState {
        let stable_seq = *self.domains.lock().entry(domain_id).or_insert(0);
        VsyncDomainState {
            domain_id,
            stable_seq,
        }
    }

    /// Returns one domain stable sequence.
    pub fn domain_stable_seq(&self, domain_id: u32) -> VsyncSeq {
        self.domains.lock().get(&domain_id).copied().unwrap_or(0)
    }

    /// Updates domain and group stable sequence.
    pub fn mark_domain_stable(&self, domain_id: u32, seq: VsyncSeq) {
        let mut domains = self.domains.lock();
        let entry = domains.entry(domain_id).or_insert(0);
        *entry = (*entry).max(seq);
        self.stable_seq.fetch_max(seq, Ordering::Relaxed);
    }

    /// Updates this group's stable sequence.
    pub fn mark_group_stable(&self, seq: VsyncSeq) {
        self.stable_seq.fetch_max(seq, Ordering::Relaxed);
    }

    /// Registers one pending file.
    pub fn add_pending_file(&self, ino: u64) {
        if ino != 0 {
            self.pending_files.lock().insert(ino);
        }
    }

    /// Removes one pending file.
    pub fn remove_pending_file(&self, ino: u64) {
        self.pending_files.lock().remove(&ino);
    }

    /// Returns whether this sync group still tracks a pending file.
    pub fn has_pending_file(&self, ino: u64) -> bool {
        self.pending_files.lock().contains(&ino)
    }

    /// Sets whether commit is delegated to a logical group.
    pub fn set_commit_delegated(&self, delegated: bool) {
        self.commit_delegated.store(delegated, Ordering::Release);
    }

    /// Returns whether commit is delegated to a logical group.
    pub fn commit_delegated(&self) -> bool {
        self.commit_delegated.load(Ordering::Acquire)
    }

    /// Enqueues one fence.
    pub fn enqueue_fence(&self, fence: VsyncFence) {
        self.fence_list.lock().push_back(fence);
    }

    /// Increments pending-record counter.
    pub fn inc_pending_records(&self) -> u64 {
        self.pending_record_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrements pending-record counter by `count`.
    pub fn dec_pending_records(&self, count: u64) {
        self.pending_record_count
            .fetch_sub(count, Ordering::Relaxed);
    }

    /// Returns pending-record counter.
    pub fn pending_records(&self) -> u64 {
        self.pending_record_count.load(Ordering::Acquire)
    }

    /// Checks and satisfies pending fences.
    ///
    /// Returns whether there are still unsatisfied fences.
    pub fn check_fences(&self) -> bool {
        let mut fences = self.fence_list.lock();
        let mut remaining = VecDeque::new();
        let mut satisfied = 0_u64;
        while let Some(fence) = fences.pop_front() {
            let stable = if fence.domain_id == 0 {
                self.stable_seq()
            } else {
                self.domain_stable_seq(fence.domain_id)
            };
            if stable >= fence.target_seq {
                if fence.token.count_down() {
                    satisfied += 1;
                }
            } else {
                remaining.push_back(fence);
            }
        }
        *fences = remaining;
        if satisfied > 0 {
            self.fences_satisfied
                .fetch_add(satisfied, Ordering::Relaxed);
        }
        !fences.is_empty()
    }

    /// Returns the next transaction ID.
    pub fn next_txn_id(&self) -> u64 {
        self.next_txn_id.load(Ordering::Acquire)
    }

    /// Sets the next transaction ID (used during recovery).
    pub fn set_next_txn_id(&self, val: u64) {
        self.next_txn_id.store(val, Ordering::Release);
    }

    /// Allocates one transaction for this sync group.
    pub fn transaction_alloc(&self) -> SgTransaction {
        let txn_id = self.next_txn_id.fetch_add(1, Ordering::Relaxed);
        self.txns_created.fetch_add(1, Ordering::Relaxed);
        SgTransaction::new(self.sg_id, txn_id)
    }

    /// Records transaction completion and updates stability.
    pub fn complete_transaction(&self, txn: SgTransaction) {
        self.commits.fetch_add(1, Ordering::Relaxed);
        self.records_processed
            .fetch_add(txn.records.len() as u64, Ordering::Relaxed);
        for record in &txn.records {
            self.mark_domain_stable(record.domain_id(), record.seq());
        }
        self.completion.set_next_complete_id(txn.txn_id + 1);
        self.check_fences();
    }

    /// Tries to complete all queued in-flight transactions.
    pub fn try_complete_txns(&self) {
        for txn in self.completion.drain_inflight() {
            self.complete_transaction(txn);
        }
    }

    /// Executes one closure under the group commit lock.
    pub fn with_commit_lock<R>(&self, f: impl FnOnce() -> R) -> R {
        let _guard = self.commit_lock.lock();
        f()
    }

    /// Coalesces a batch of records.
    pub fn coalesce_batch(
        &self,
        batch: Vec<Record>,
        coalesce_ops: Option<&dyn VsyncCoalesceOps>,
    ) -> Vec<Record> {
        coalesce_batch(batch, coalesce_ops)
    }

    /// Sets the last error state.
    pub fn set_last_error(&self, errno: Errno, seq: VsyncSeq) {
        *self.last_error.lock() = Some((errno, seq));
    }

    /// Returns the last error state.
    pub fn last_error(&self) -> Option<(Errno, VsyncSeq)> {
        *self.last_error.lock()
    }

    /// Placeholder commit thread entry.
    pub fn commit_thread_fn() -> i32 {
        0
    }
}

/// Per-file tracking state.
#[derive(Clone, Debug, Default)]
pub struct VsyncFileState {
    pub ino: u64,
    pub sg_targets: BTreeMap<u32, VsyncSeq>,
    pub sg_refs: BTreeMap<u32, VsyncSgFileRef>,
}

impl VsyncFileState {
    /// Creates per-file tracking state.
    pub fn new(ino: u64) -> Self {
        Self {
            ino,
            sg_targets: BTreeMap::new(),
            sg_refs: BTreeMap::new(),
        }
    }
}

/// Per-(sg,file) reference counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VsyncSgFileRef {
    pub sg_id: u32,
    pub inflight_ops: i32,
    pub uncommitted_ops: i32,
}

/// Logical-group state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum VsyncLgState {
    #[default]
    Active,
    Dismissing,
    Destroyed,
}

/// A logical group of sync groups that share files.
#[derive(Clone, Debug, Default)]
pub struct VsyncLogicalGroup {
    pub lg_id: u32,
    pub member_sgs: BTreeSet<u32>,
    pub shared_files: BTreeSet<u64>,
    pub state: VsyncLgState,
}

impl VsyncLogicalGroup {
    /// Creates one logical group with two members.
    pub fn create(lg_id: u32, sg1: u32, sg2: u32) -> Self {
        let mut member_sgs = BTreeSet::new();
        member_sgs.insert(sg1);
        member_sgs.insert(sg2);
        Self {
            lg_id,
            member_sgs,
            shared_files: BTreeSet::new(),
            state: VsyncLgState::Active,
        }
    }

    /// Adds one member sync group.
    pub fn add_sg(&mut self, sg_id: u32) -> bool {
        self.member_sgs.insert(sg_id)
    }

    /// Merges `other` into `self`.
    pub fn merge(&mut self, other: &mut VsyncLogicalGroup) {
        self.member_sgs.extend(other.member_sgs.iter().copied());
        self.shared_files.extend(other.shared_files.iter().copied());
        other.state = VsyncLgState::Destroyed;
    }

    /// Checks whether this group can be dismissed.
    pub fn check_dismissal(&self) -> bool {
        self.shared_files.is_empty()
    }

    /// Marks this group as dismissing/destroyed.
    pub fn dismiss(&mut self) {
        self.state = if self.shared_files.is_empty() {
            VsyncLgState::Destroyed
        } else {
            VsyncLgState::Dismissing
        };
    }

    /// Placeholder for triggering one group commit.
    pub fn trigger_commit(&self) {}

    /// Placeholder for running one group commit.
    pub fn do_commit(&self) {}
}

/// Sync-group transaction state.
#[derive(Clone, Debug)]
pub struct SgTransaction {
    pub txn_id: u64,
    pub shard_id: u32,
    pub state: VsyncTxnState,
    pub sg_id: u32,
    pub records: Vec<Record>,
    pub max_seq: VsyncSeq,
    pub start_sector: u64,
    pub sector_count: u64,
    pub error: Option<Errno>,
    pub io_completed: bool,
}

impl SgTransaction {
    /// Creates an empty transaction.
    pub fn new(sg_id: u32, txn_id: u64) -> Self {
        Self {
            txn_id,
            shard_id: 0,
            state: VsyncTxnState::Collecting,
            sg_id,
            records: Vec::new(),
            max_seq: 0,
            start_sector: 0,
            sector_count: 0,
            error: None,
            io_completed: false,
        }
    }

    /// Appends one record to this transaction.
    pub fn push_record(&mut self, record: Record) {
        self.max_seq = self.max_seq.max(record.seq());
        self.records.push(record);
    }

    /// Drops this transaction.
    pub fn free(self) {}
}

/// Transaction metadata retained after commit.
#[derive(Clone, Debug, Default)]
pub struct CommittedTransaction {
    pub sg_id: u32,
    pub txn_id: u64,
    pub shard_id: u32,
    pub size_sectors: u64,
    pub start_sector: u64,
    pub records: Vec<Record>,
    pub io_complete: bool,
    pub processed: bool,
    pub journal_freed: bool,
}

impl CommittedTransaction {
    /// Drops this committed transaction.
    pub fn free(self) {}
}
