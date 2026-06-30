// SPDX-License-Identifier: MPL-2.0

//! VSpace: per-mount in-memory VSync state.

use core::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};

use aster_block::BlockDevice;
use ostd::mm::VmIo;

use super::*;
use crate::{
    prelude::*,
    process::{Process, posix_thread::AsPosixThread},
    thread::Thread,
};

static VSYNC_SEQ_COUNTER: AtomicU64 = AtomicU64::new(1);
pub static VSYNC_GLOBAL_CFG: SpinLock<VsyncConfig> = SpinLock::new(VsyncConfig::new());

fn journal_superblock_sectors() -> u64 {
    (aster_block::BLOCK_SIZE / aster_block::SECTOR_SIZE) as u64
}

/// Transaction state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum VsyncTxnState {
    #[default]
    Collecting,
    Serializing,
    Submitted,
    Completed,
}

/// Domain mapping mode.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum VsyncDomainType {
    Program,
    Global,
    #[default]
    Thread,
}

/// Runtime VSync configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VsyncConfig {
    pub debug_level: i32,
    pub domain_type: VsyncDomainType,
    pub debug_seq: bool,
    pub debug_no_coalesce: bool,
    pub debug_sg_shard: bool,
    pub online_checkpoint: bool,
    pub umount_checkpoint: bool,
    pub back_pressure: bool,
    pub per_file_sync: bool,
    pub coalesce_us: i32,
    pub numa_aware: bool,
}

impl VsyncConfig {
    pub const fn new() -> Self {
        Self {
            debug_level: 0,
            domain_type: VsyncDomainType::Thread,
            debug_seq: false,
            debug_no_coalesce: false,
            debug_sg_shard: false,
            online_checkpoint: true,
            umount_checkpoint: true,
            back_pressure: true,
            per_file_sync: true,
            coalesce_us: 250,
            numa_aware: false,
        }
    }
}

impl Default for VsyncConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Filesystem callback used during checkpoint/replay style apply.
pub type VsyncCheckpointFn = Arc<dyn Fn(&dyn Any, &[Record]) -> Result<()> + Send + Sync>;

/// Optional shadow-materialization callback.
pub type VsyncShadowMaterializeFn = Arc<dyn Fn(&dyn Any, u32) -> Vec<Record> + Send + Sync>;

/// Optional shadow-reset callback.
pub type VsyncShadowResetFn = Arc<dyn Fn(&dyn Any, u32) + Send + Sync>;

/// Handle returned by `vsync_op_start`.
#[derive(Debug)]
pub struct VsyncHandle {
    record: Option<Record>,
    vspace: Arc<VsyncVspace>,
    sg_id: u32,
    cpu: i32,
    cancelled: bool,
    completed: bool,
}

impl VsyncHandle {
    /// Returns sequence of the underlying record.
    pub fn seq(&self) -> VsyncSeq {
        self.record.as_ref().map(Record::seq).unwrap_or(0)
    }

    /// Returns sync-group ID.
    pub fn sg_id(&self) -> u32 {
        self.sg_id
    }

    /// Returns slot CPU index (in this pass always `0`).
    pub fn cpu(&self) -> i32 {
        self.cpu
    }

    /// Sets pre-commit barrier on the record.
    pub fn set_pre_commit_barrier(&mut self, barrier: Arc<CountdownLatch>) {
        if let Some(record) = &mut self.record {
            record.set_pre_commit_barrier(Some(barrier));
        }
    }
}

/// VSpace lifecycle state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum VsyncVspaceState {
    #[default]
    Created = 0,
    Active = 1,
    Unmounting = 2,
    Destroyed = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VsyncStableScope {
    Domain(u32),
    Group,
}

/// Per-mount in-memory VSync instance.
pub struct VsyncVspace {
    pub sync_groups: [Arc<VsyncSyncGroup>; VSYNC_NUM_SYNC_GROUPS],
    pub shards: [Arc<VsyncShard>; VSYNC_NUM_SHARDS],
    pub slots: Arc<VsyncPcpu>,
    pub journal_start_sector: u64,
    pub journal_total_sectors: u64,
    pub shard_journal_sectors: u64,
    config: SpinLock<VsyncConfig>,
    checkpoint_fn: RwLock<Option<VsyncCheckpointFn>>,
    pub fs_context: RwLock<Option<Arc<dyn Any + Send + Sync>>>,
    pub coalesce_ops: RwLock<Option<Arc<dyn VsyncCoalesceOps>>>,
    no_batch_coalesce: AtomicBool,
    shadow_materialize: RwLock<Option<VsyncShadowMaterializeFn>>,
    shadow_reset: RwLock<Option<VsyncShadowResetFn>>,
    logical_groups: SpinLock<BTreeMap<u32, VsyncLogicalGroup>>,
    next_lg_id: AtomicU64,
    file_tracker: SpinLock<BTreeMap<u64, VsyncFileState>>,
    pub block_device: Option<Arc<dyn BlockDevice>>,
    pub journal_super_flags: AtomicU32,
    state: AtomicU8,
    total_ops: AtomicU64,
    total_syncs: AtomicU64,
    pub checkpointed_txn_id: [AtomicU32; VSYNC_NUM_SYNC_GROUPS],
}

impl Debug for VsyncVspace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VsyncVspace")
            .field("journal_start_sector", &self.journal_start_sector)
            .field("journal_total_sectors", &self.journal_total_sectors)
            .field("no_batch_coalesce", &self.no_batch_coalesce)
            .field("state", &self.state)
            .field("total_ops", &self.total_ops)
            .field("total_syncs", &self.total_syncs)
            .finish_non_exhaustive()
    }
}

impl VsyncVspace {
    /// Creates one vspace (optionally backed by a block device for on-disk journal).
    pub fn new(
        config: VsyncConfig,
        checkpoint_fn: Option<VsyncCheckpointFn>,
        fs_context: Option<Arc<dyn Any + Send + Sync>>,
        block_device: Option<Arc<dyn BlockDevice>>,
        journal_start_sector: u64,
        journal_total_sectors: u64,
    ) -> Arc<Self> {
        let superblock_sectors = journal_superblock_sectors();
        let data_sectors = journal_total_sectors
            .saturating_sub(superblock_sectors)
            .max(VSYNC_NUM_SHARDS as u64);
        let shard_journal_sectors = {
            let block_sectors = journal_superblock_sectors();
            let per_shard_sectors = data_sectors / VSYNC_NUM_SHARDS as u64;
            (per_shard_sectors / block_sectors * block_sectors).max(block_sectors)
        };
        let sync_groups = core::array::from_fn(|idx| Arc::new(VsyncSyncGroup::new(idx as u32)));
        let shards = core::array::from_fn(|idx| {
            Arc::new(VsyncShard::new(
                idx as u32,
                journal_start_sector + superblock_sectors + idx as u64 * shard_journal_sectors,
                shard_journal_sectors,
            ))
        });
        let vspace = Arc::new(Self {
            sync_groups,
            shards,
            slots: Arc::new(VsyncPcpu::default()),
            journal_start_sector,
            journal_total_sectors,
            shard_journal_sectors,
            config: SpinLock::new(config),
            checkpoint_fn: RwLock::new(checkpoint_fn),
            fs_context: RwLock::new(fs_context),
            coalesce_ops: RwLock::new(None),
            no_batch_coalesce: AtomicBool::new(false),
            shadow_materialize: RwLock::new(None),
            shadow_reset: RwLock::new(None),
            logical_groups: SpinLock::new(BTreeMap::new()),
            next_lg_id: AtomicU64::new(1),
            file_tracker: SpinLock::new(BTreeMap::new()),
            block_device,
            journal_super_flags: AtomicU32::new(0),
            state: AtomicU8::new(VsyncVspaceState::Created as u8),
            total_ops: AtomicU64::new(0),
            total_syncs: AtomicU64::new(0),
            checkpointed_txn_id: core::array::from_fn(|_| AtomicU32::new(0)),
        });
        vspace.set_state(VsyncVspaceState::Active);
        vspace
    }

    /// Returns current vspace state.
    pub fn state(&self) -> VsyncVspaceState {
        match self.state.load(Ordering::Acquire) {
            1 => VsyncVspaceState::Active,
            2 => VsyncVspaceState::Unmounting,
            3 => VsyncVspaceState::Destroyed,
            _ => VsyncVspaceState::Created,
        }
    }

    /// Stores vspace state.
    pub fn set_state(&self, state: VsyncVspaceState) {
        self.state.store(state as u8, Ordering::Release);
    }

    /// Returns current config snapshot.
    pub fn config(&self) -> VsyncConfig {
        self.config.lock().clone()
    }

    /// Sets config snapshot.
    pub fn set_config(&self, config: VsyncConfig) {
        *self.config.lock() = config;
    }

    pub fn task_to_domain(config: &VsyncConfig) -> u32 {
        // In ktest context there may be no current process/thread; use a stable fallback.
        let pid: u32 = Process::current().map(|p| p.pid().into()).unwrap_or(0);
        let tid: u32 = if let Some(current_thread) = Thread::current()
            && let Some(posix) = current_thread.as_posix_thread()
        {
            posix.tid()
        } else {
            pid
        };
        match config.domain_type {
            VsyncDomainType::Global => 0,
            VsyncDomainType::Program => pid,
            VsyncDomainType::Thread => tid,
        }
    }

    pub fn domain_to_sg(domain_id: u32) -> u32 {
        domain_id % VSYNC_NUM_SYNC_GROUPS as u32
    }

    fn sg_to_shard(config: &VsyncConfig, sg_id: u32, txn_id: u64) -> u32 {
        if config.debug_sg_shard {
            sg_id % VSYNC_NUM_SHARDS as u32
        } else {
            (txn_id % VSYNC_NUM_SHARDS as u64) as u32
        }
    }

    fn mark_sync_target_stable(
        sg: &VsyncSyncGroup,
        target_seq: VsyncSeq,
        stable_scope: VsyncStableScope,
    ) {
        match stable_scope {
            VsyncStableScope::Domain(domain_id) => sg.mark_domain_stable(domain_id, target_seq),
            VsyncStableScope::Group => sg.mark_group_stable(target_seq),
        }
    }

    fn commit_one_group(
        &self,
        sg_id: u32,
        target_seq: VsyncSeq,
        stable_scope: VsyncStableScope,
    ) -> Result<()> {
        let sg = &self.sync_groups[sg_id as usize];
        sg.with_commit_lock(|| {
            let slot = &self.slots.slots[sg_id as usize];
            slot.wait_inflight_until(target_seq);

            for fence in slot.drain_fences() {
                sg.enqueue_fence(fence);
            }

            let mut batch = slot.drain_until(target_seq, usize::MAX);
            if let Some(materialize) = self.shadow_materialize.read().as_ref() {
                let materialized = if let Some(context) = self.fs_context.read().as_ref() {
                    materialize(context.as_ref(), sg_id)
                } else {
                    materialize(&(), sg_id)
                };
                batch.extend(materialized);
            }

            if batch.is_empty() {
                Self::mark_sync_target_stable(sg, target_seq, stable_scope);
                sg.check_fences();
                return Ok(());
            }

            let coalesce_ops = self.coalesce_ops.read();
            if !self.no_batch_coalesce.load(Ordering::Acquire)
                && !self.config.lock().debug_no_coalesce
            {
                batch = sg.coalesce_batch(batch, coalesce_ops.as_deref());
            }

            let mut txn = sg.transaction_alloc();
            for record in batch {
                txn.push_record(record);
            }
            let shard_id = Self::sg_to_shard(&self.config.lock(), sg_id, txn.txn_id);
            txn.shard_id = shard_id;
            txn.state = VsyncTxnState::Submitted;
            self.shards[shard_id as usize].enqueue_transaction(txn);

            let committed_transactions = self.shards[shard_id as usize]
                .flush_all(self.block_device.as_deref())
                .map_err(|err| {
                    sg.set_last_error(err.error(), target_seq);
                    err
                })?;

            for committed in committed_transactions {
                let committed_sg = &self.sync_groups[committed.sg_id as usize];
                let mut completed_txn = SgTransaction::new(committed.sg_id, committed.txn_id);
                completed_txn.shard_id = committed.shard_id;
                completed_txn.state = VsyncTxnState::Completed;
                completed_txn.records = committed.records.clone();
                completed_txn.max_seq = completed_txn
                    .records
                    .iter()
                    .map(Record::seq)
                    .max()
                    .unwrap_or(0);
                completed_txn.io_completed = true;
                committed_sg.complete_transaction(completed_txn);
                committed_sg.dec_pending_records(committed.records.len() as u64);

                for record in &committed.records {
                    if let Some(ino) = record.ino() {
                        self.file_on_commit(ino, committed.sg_id, record.seq());
                    }
                }
            }

            Self::mark_sync_target_stable(sg, target_seq, stable_scope);
            sg.check_fences();

            if let Some(reset) = self.shadow_reset.read().as_ref() {
                if let Some(context) = self.fs_context.read().as_ref() {
                    reset(context.as_ref(), sg_id);
                } else {
                    reset(&(), sg_id);
                }
            }

            Ok(())
        })
    }

    /// Starts one journaling operation and returns its handle.
    pub fn op_start(self: &Arc<Self>, rtype: VsyncRecordType, ino: u64) -> Result<VsyncHandle> {
        if self.state() != VsyncVspaceState::Active {
            return_errno_with_message!(Errno::EINVAL, "vspace is not active");
        }

        let config = self.config();
        let domain_id = Self::task_to_domain(&config);
        let sg_id = Self::domain_to_sg(domain_id);
        let seq = vsync_seq_now();
        let record = match rtype {
            VsyncRecordType::Write => Record::new_write(seq, domain_id, ino, Vec::new()),
            VsyncRecordType::Meta => Record::new_meta(seq, domain_id, ino, Vec::new()),
            VsyncRecordType::Fence => {
                Record::new_fence(seq, domain_id, Arc::new(CountdownLatch::new_token(1)), seq)
            }
            VsyncRecordType::TxnHdr => {
                Record::new_txn_hdr(seq, domain_id, VsyncTxnHdrPayload::default())
            }
        };

        let slot = &self.slots.slots[sg_id as usize];
        slot.begin_inflight(seq);
        let pending = self.sync_groups[sg_id as usize].inc_pending_records();
        if pending > VSYNC_PENDING_HIGH_MARK as u64 {
            self.sync_groups[sg_id as usize]
                .commit_pending
                .fetch_add(1, Ordering::Relaxed);
        }

        if ino != 0 {
            let _ = self.handle_shared_file(ino, sg_id);
        }

        Ok(VsyncHandle {
            record: Some(record),
            vspace: self.clone(),
            sg_id,
            cpu: 0,
            cancelled: false,
            completed: false,
        })
    }

    /// Completes one started operation.
    pub fn op_complete(&self, handle: &mut VsyncHandle, payload: Option<Vec<u8>>) -> bool {
        if handle.completed {
            return false;
        }
        let Some(mut record) = handle.record.take() else {
            return false;
        };

        let seq = record.seq();
        let sg_id = handle.sg_id;
        let slot = &self.slots.slots[sg_id as usize];
        let sg = &self.sync_groups[sg_id as usize];

        slot.finish_inflight(seq);
        if handle.cancelled {
            sg.dec_pending_records(1);
            if let Some(ino) = record.ino() {
                self.file_cancel_sg(ino, sg_id);
            }
            handle.completed = true;
            return false;
        }

        match record.record_type() {
            VsyncRecordType::Write | VsyncRecordType::Meta => {
                let Some(payload) = payload else {
                    sg.dec_pending_records(1);
                    if let Some(ino) = record.ino() {
                        self.file_cancel_sg(ino, sg_id);
                    }
                    handle.completed = true;
                    return false;
                };
                if payload.is_empty() {
                    sg.dec_pending_records(1);
                    if let Some(ino) = record.ino() {
                        self.file_cancel_sg(ino, sg_id);
                    }
                    handle.completed = true;
                    return false;
                }
                record.set_payload(payload);
                if let Some(ino) = record.ino() {
                    self.file_add_seq(ino, sg_id, seq);
                    self.file_remove_sg(ino, sg_id);
                }
            }
            _ => {
                if let Some(ino) = record.ino() {
                    self.file_remove_sg(ino, sg_id);
                }
            }
        }

        slot.publish_record(record);
        self.total_ops.fetch_add(1, Ordering::Relaxed);
        handle.completed = true;

        if sg.pending_records() >= VSYNC_BATCH_THRESHOLD as u64 {
            sg.commit_pending.fetch_add(1, Ordering::Relaxed);
        }

        true
    }

    /// Runs one checkpoint cycle on committed in-memory transactions.
    ///
    /// Applies records via coalesce_fn or checkpoint_fn, then advances
    /// shard tails to free journal space. Writes the superblock when tails change.
    pub fn checkpoint(&self, _sectors_needed: u64) -> Result<u64> {
        let mut records = Vec::new();
        let mut shard_tails: [Option<u64>; VSYNC_NUM_SHARDS] = [None; VSYNC_NUM_SHARDS];

        for shard in &self.shards {
            for mut committed in shard.take_committed_transactions() {
                // Track furthest end sector for this shard
                let end_sector = committed.start_sector + committed.size_sectors;
                let sid = committed.shard_id as usize;
                shard_tails[sid] = Some(
                    shard_tails[sid]
                        .map(|existing| existing.max(end_sector))
                        .unwrap_or(end_sector),
                );
                committed.journal_freed = true;
                records.extend(committed.records);
            }
        }
        if records.is_empty() {
            return Ok(0);
        }

        let fs_context = self.fs_context.read().clone();
        let coalesce_ops = self.coalesce_ops.read().clone();
        let checkpoint_fn = self.checkpoint_fn.read().clone();
        if let Some(fs_context) = fs_context {
            if let Some(coalesce_ops) = coalesce_ops {
                checkpoint_coalesce(&records, coalesce_ops.as_ref(), fs_context.as_ref(), true)?;
            } else if let Some(checkpoint_fn) = checkpoint_fn {
                checkpoint_fn(fs_context.as_ref(), &records)?;
            }
        }

        // Advance shard tails to free journal space
        let mut tail_advanced = false;
        for (sid, new_tail) in shard_tails.iter().enumerate() {
            if let Some(tail) = new_tail {
                self.shards[sid].journal.advance_tail(*tail);
                tail_advanced = true;
            }
        }
        if tail_advanced {
            self.write_journal_super();
        }

        Ok(records.len() as u64)
    }

    /// Placeholder checkpoint thread body.
    pub fn checkpoint_thread_fn() -> i32 {
        0
    }

    /// Queues a checkpoint immediately in this in-memory implementation.
    pub fn queue_checkpoint(&self) {
        let _ = self.checkpoint(0);
    }

    /// Runs unmount checkpoint if configured.
    pub fn unmount_checkpoint(&self) {
        if self.config().umount_checkpoint {
            let _ = self.checkpoint(0);
        }
    }

    /// Writes journal superblock to the block device at `journal_start_sector`.
    ///
    /// Serializes `VsyncJournalSuper` into a page-sized buffer, pads to one block,
    /// writes via the block device, and issues a cache flush for durability.
    pub fn write_journal_super(&self) -> i32 {
        let Some(ref bdev) = self.block_device else {
            return 0;
        };

        let mut superblock = VsyncJournalSuper {
            magic: VSYNC_JOURNAL_MAGIC,
            version: VSYNC_JOURNAL_VERSION,
            num_shards: VSYNC_NUM_SHARDS as u32,
            lb_size: aster_block::BLOCK_SIZE as u32,
            journal_start_sector: self.journal_start_sector,
            journal_total_sectors: self.journal_total_sectors,
            shard_journal_sectors: self.shard_journal_sectors,
            flags: self.journal_super_flags(),
            checksum: 0,
            fs_uuid: [0u8; 16],
            shards: [VsyncShardOndisk::default(); VSYNC_NUM_SHARDS],
            next_txn_id: [0u32; VSYNC_NUM_SYNC_GROUPS],
            checkpointed_txn_id: [0u32; VSYNC_NUM_SYNC_GROUPS],
        };

        // Populate per-shard head/tail as relative offsets
        for (i, shard) in self.shards.iter().enumerate() {
            superblock.shards[i] = VsyncShardOndisk {
                head_sector: shard
                    .journal
                    .head_sector
                    .load(Ordering::Acquire)
                    .saturating_sub(shard.journal.start_sector),
                tail_sector: shard
                    .journal
                    .tail_sector
                    .load(Ordering::Acquire)
                    .saturating_sub(shard.journal.start_sector),
            };
        }
        // Populate per-SG state
        for (i, sg) in self.sync_groups.iter().enumerate() {
            superblock.next_txn_id[i] = sg.next_txn_id() as u32;
            superblock.checkpointed_txn_id[i] = self.checkpointed_txn_id[i].load(Ordering::Acquire);
        }

        // Serialize and write
        let mut buf = vec![0u8; aster_block::BLOCK_SIZE];
        serialize_superblock(&superblock, &mut buf);

        let super_offset = self.journal_start_sector as usize * aster_block::SECTOR_SIZE;
        if bdev
            .write(
                super_offset,
                &mut VmReader::from(buf.as_slice()).to_fallible(),
            )
            .is_err()
        {
            return -(Errno::EIO as i32);
        }
        if bdev.sync().is_err() {
            return -(Errno::EIO as i32);
        }
        0
    }

    /// Reads journal superblock from the block device and restores shard/SG state.
    ///
    /// Returns 0 on success, negative errno on failure (e.g., bad magic, version mismatch,
    /// checksum error, or no block device).
    pub fn read_journal_super(&self) -> i32 {
        let Some(ref bdev) = self.block_device else {
            return -(Errno::ENODEV as i32);
        };

        let super_offset = self.journal_start_sector as usize * aster_block::SECTOR_SIZE;
        let mut buf = vec![0u8; aster_block::BLOCK_SIZE];
        if bdev
            .read(
                super_offset,
                &mut VmWriter::from(buf.as_mut_slice()).to_fallible(),
            )
            .is_err()
        {
            return -(Errno::EIO as i32);
        }

        let Some(superblock) = deserialize_superblock(&buf) else {
            return -(Errno::EINVAL as i32);
        };

        // Validate magic and version
        if superblock.magic != VSYNC_JOURNAL_MAGIC || superblock.version != VSYNC_JOURNAL_VERSION {
            return -(Errno::EINVAL as i32);
        }

        // Validate shard count matches
        if superblock.num_shards as usize != VSYNC_NUM_SHARDS {
            return -(Errno::EINVAL as i32);
        }

        if superblock.lb_size != aster_block::BLOCK_SIZE as u32
            || superblock.journal_start_sector != self.journal_start_sector
            || superblock.journal_total_sectors != self.journal_total_sectors
            || superblock.shard_journal_sectors != self.shard_journal_sectors
        {
            return -(Errno::EINVAL as i32);
        }

        // Restore per-shard state
        for (i, shard) in self.shards.iter().enumerate() {
            let s = superblock.shards[i];
            if s.head_sector >= self.shard_journal_sectors
                || s.tail_sector >= self.shard_journal_sectors
            {
                return -(Errno::EINVAL as i32);
            }
            let abs_head = shard.journal.start_sector.saturating_add(s.head_sector);
            let abs_tail = shard.journal.start_sector.saturating_add(s.tail_sector);
            shard.journal.head_sector.store(abs_head, Ordering::Release);
            shard
                .journal
                .committed_head_sector
                .store(abs_head, Ordering::Release);
            shard.journal.tail_sector.store(abs_tail, Ordering::Release);
        }

        // Restore per-SG txn state
        for (i, sg) in self.sync_groups.iter().enumerate() {
            // next_txn_id is recovered as lower-bound of what was on disk
            let nid = superblock.next_txn_id[i] as u64;
            sg.set_next_txn_id(nid);
            self.checkpointed_txn_id[i].store(superblock.checkpointed_txn_id[i], Ordering::Release);
        }

        // Store flags for mount/unmount lifecycle
        self.set_journal_super_flags(superblock.flags);

        0
    }

    /// Returns journal superblock flags.
    fn journal_super_flags(&self) -> u32 {
        self.journal_super_flags.load(Ordering::Acquire)
    }

    /// Stores journal superblock flags.
    fn set_journal_super_flags(&self, flags: u32) {
        self.journal_super_flags.store(flags, Ordering::Release);
    }

    /// Returns one file state.
    pub fn get_file_state(&self, ino: u64) -> Option<VsyncFileState> {
        self.file_tracker.lock().get(&ino).cloned()
    }

    /// Stores one file state.
    pub fn put_file_state(&self, file_state: VsyncFileState) {
        self.file_tracker.lock().insert(file_state.ino, file_state);
    }

    pub fn file_add_seq(&self, ino: u64, sg_id: u32, seq: VsyncSeq) {
        let mut tracker = self.file_tracker.lock();
        let file_state = tracker
            .entry(ino)
            .or_insert_with(|| VsyncFileState::new(ino));
        let entry = file_state.sg_targets.entry(sg_id).or_insert(0);
        *entry = (*entry).max(seq);
    }

    /// Adds one sync-group participant for a file.
    pub fn file_add_sg(&self, ino: u64, sg_id: u32) -> i32 {
        let mut tracker = self.file_tracker.lock();
        let file_state = tracker
            .entry(ino)
            .or_insert_with(|| VsyncFileState::new(ino));
        file_state.sg_targets.entry(sg_id).or_insert(0);
        let sg_ref = file_state
            .sg_refs
            .entry(sg_id)
            .or_insert_with(|| VsyncSgFileRef {
                sg_id,
                inflight_ops: 0,
                uncommitted_ops: 0,
            });
        sg_ref.inflight_ops = sg_ref.inflight_ops.saturating_add(1);
        self.sync_groups[sg_id as usize].add_pending_file(ino);
        0
    }

    /// Moves one file operation from in-flight to uncommitted.
    pub fn file_remove_sg(&self, ino: u64, sg_id: u32) {
        let mut tracker = self.file_tracker.lock();
        let Some(file_state) = tracker.get_mut(&ino) else {
            return;
        };
        let Some(sg_ref) = file_state.sg_refs.get_mut(&sg_id) else {
            return;
        };
        if sg_ref.inflight_ops > 0 {
            sg_ref.inflight_ops -= 1;
        }
        sg_ref.uncommitted_ops = sg_ref.uncommitted_ops.saturating_add(1);
    }

    /// Drops one file operation that never became an uncommitted journal record.
    pub fn file_cancel_sg(&self, ino: u64, sg_id: u32) {
        let mut should_remove_pending = false;
        {
            let mut tracker = self.file_tracker.lock();
            let Some(file_state) = tracker.get_mut(&ino) else {
                return;
            };
            let Some(sg_ref) = file_state.sg_refs.get_mut(&sg_id) else {
                return;
            };
            if sg_ref.inflight_ops > 0 {
                sg_ref.inflight_ops -= 1;
            }
            if sg_ref.inflight_ops == 0 && sg_ref.uncommitted_ops == 0 {
                file_state.sg_refs.remove(&sg_id);
                should_remove_pending = true;
            }
        }
        if should_remove_pending {
            self.sync_groups[sg_id as usize].remove_pending_file(ino);
        }
    }

    /// Handles shared-file tracking for operation start.
    pub fn handle_shared_file(&self, ino: u64, sg_id: u32) -> i32 {
        self.file_add_sg(ino, sg_id)
    }

    /// Fills sync-group targets for one file.
    ///
    /// Returns true when file had at least one tracked SG target.
    pub fn file_get_lg_targets(
        &self,
        ino: u64,
        target_seq: VsyncSeq,
        sg_targets: &mut [VsyncSeq; VSYNC_NUM_SYNC_GROUPS],
    ) -> bool {
        sg_targets.fill(0);
        let Some(file_state) = self.file_tracker.lock().get(&ino).cloned() else {
            return false;
        };
        for (sg_id, seq) in file_state.sg_targets {
            let idx = sg_id as usize % VSYNC_NUM_SYNC_GROUPS;
            sg_targets[idx] = if target_seq == 0 {
                seq
            } else {
                seq.max(target_seq)
            };
        }
        sg_targets.iter().any(|target| *target > 0)
    }

    /// Updates file tracking after commit.
    pub fn file_on_commit(&self, ino: u64, sg_id: u32, seq: VsyncSeq) {
        self.file_add_seq(ino, sg_id, seq);
        let mut should_remove_pending = false;
        {
            let mut tracker = self.file_tracker.lock();
            let Some(file_state) = tracker.get_mut(&ino) else {
                return;
            };
            if let Some(sg_ref) = file_state.sg_refs.get_mut(&sg_id) {
                if sg_ref.uncommitted_ops > 0 {
                    sg_ref.uncommitted_ops -= 1;
                }
                if sg_ref.inflight_ops == 0 && sg_ref.uncommitted_ops == 0 {
                    file_state.sg_refs.remove(&sg_id);
                    should_remove_pending = true;
                }
            }
        }
        if should_remove_pending {
            self.sync_groups[sg_id as usize].remove_pending_file(ino);
        }
    }

    /// Initializes file tracking state.
    pub fn init_file_tracker(&self) -> i32 {
        self.file_tracker.lock().clear();
        0
    }

    /// Destroys file tracking state.
    pub fn destroy_file_tracker(&self) {
        self.file_tracker.lock().clear();
    }

    /// Sets no-batch-coalesce mode.
    pub fn set_no_batch_coalesce(&self, val: bool) {
        self.no_batch_coalesce.store(val, Ordering::Release);
    }

    /// Registers coalescing callbacks.
    pub fn set_coalesce_ops(&self, ops: Option<Arc<dyn VsyncCoalesceOps>>) -> i32 {
        *self.coalesce_ops.write() = ops;
        0
    }

    /// Registers optional shadow callbacks.
    pub fn set_shadow_ops(
        &self,
        shadow_materialize: Option<VsyncShadowMaterializeFn>,
        shadow_reset: Option<VsyncShadowResetFn>,
    ) {
        *self.shadow_materialize.write() = shadow_materialize;
        *self.shadow_reset.write() = shadow_reset;
    }

    /// Returns vspace journal start sector.
    pub fn start_sector(&self) -> u64 {
        self.journal_start_sector
    }

    /// Returns vspace journal total sectors.
    pub fn total_sectors(&self) -> u64 {
        self.journal_total_sectors
    }

    /// Returns global stable sequence (minimum over sync groups).
    pub fn stable_seq(&self) -> VsyncSeq {
        self.sync_groups
            .iter()
            .map(|sg| sg.stable_seq())
            .min()
            .unwrap_or(0)
    }

    /// Returns whether vspace is still in recovery stage.
    pub fn is_recovering(&self) -> bool {
        self.state() == VsyncVspaceState::Created
    }

    /// Performs one domain-aware `sync_until`.
    pub fn sync_until(&self, target_seq: VsyncSeq) -> Result<()> {
        let config = self.config();
        let domain_id = Self::task_to_domain(&config);
        let sg_id = Self::domain_to_sg(domain_id);
        let sg = &self.sync_groups[sg_id as usize];

        if sg.domain_stable_seq(domain_id) >= target_seq {
            return Ok(());
        }

        let token = Arc::new(CountdownLatch::new_token(1));
        self.slots.slots[sg_id as usize].publish_fence(VsyncFence {
            token: token.clone(),
            domain_id,
            target_seq,
        });

        self.commit_one_group(sg_id, target_seq, VsyncStableScope::Domain(domain_id))?;
        token.wait();

        if let Some((errno, error_seq)) = sg.last_error() {
            if error_seq >= target_seq {
                return_errno_with_message!(errno, "sync_until range has transaction error");
            }
        }

        self.total_syncs.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Performs sync for all sync groups up to current published watermarks.
    pub fn sync_all(&self) -> Result<()> {
        let mut targets = [0; VSYNC_NUM_SYNC_GROUPS];
        for (idx, target) in targets.iter_mut().enumerate() {
            *target = self.slots.slots[idx].last_published_seq();
        }
        self.sync_multi_sg(&targets)
    }

    /// Performs sync for all sync groups up to `target_seq`.
    pub fn sync_all_until(&self, target_seq: VsyncSeq) -> Result<()> {
        self.sync_multi_sg(&[target_seq; VSYNC_NUM_SYNC_GROUPS])
    }

    /// Performs sync for selected sync groups.
    pub fn sync_multi_sg(&self, sg_targets: &[VsyncSeq; VSYNC_NUM_SYNC_GROUPS]) -> Result<()> {
        let mut targets = Vec::new();
        for (sg_id, target) in sg_targets.iter().copied().enumerate() {
            if target == 0 {
                continue;
            }
            if self.sync_groups[sg_id].stable_seq() >= target {
                continue;
            }
            targets.push((sg_id as u32, target));
        }
        if targets.is_empty() {
            return Ok(());
        }

        let token = Arc::new(CountdownLatch::new_token(targets.len() as u64));
        for (sg_id, target_seq) in &targets {
            self.slots.slots[*sg_id as usize].publish_fence(VsyncFence {
                token: token.clone(),
                domain_id: 0,
                target_seq: *target_seq,
            });
        }
        for (sg_id, target_seq) in targets {
            self.commit_one_group(sg_id, target_seq, VsyncStableScope::Group)?;
        }
        token.wait();

        for (sg_id, target_seq) in sg_targets.iter().copied().enumerate() {
            if target_seq == 0 {
                continue;
            }
            if let Some((errno, error_seq)) = self.sync_groups[sg_id].last_error() {
                if error_seq >= target_seq {
                    return_errno_with_message!(errno, "sync_multi_sg range has transaction error");
                }
            }
        }

        self.total_syncs.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Performs file-scoped sync, including cross-SG shared-file targets.
    pub fn sync_file(&self, ino: u64, target_seq: VsyncSeq) -> Result<()> {
        if !self.config().per_file_sync {
            return self.sync_until(target_seq);
        }

        let mut sg_targets = [0; VSYNC_NUM_SYNC_GROUPS];
        if self.file_get_lg_targets(ino, target_seq, &mut sg_targets) {
            self.sync_multi_sg(&sg_targets)
        } else {
            self.sync_until(target_seq)
        }
    }
}

/// Returns current sequence number.
pub fn vsync_seq_now() -> VsyncSeq {
    VSYNC_SEQ_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Returns whether per-file sync is globally enabled.
pub fn vsync_per_file_sync_enabled() -> bool {
    VSYNC_GLOBAL_CFG.lock().per_file_sync
}

/// Returns current sync-group ID under global config.
pub fn vsync_current_sg_id() -> u32 {
    let cfg = VSYNC_GLOBAL_CFG.lock().clone();
    VsyncVspace::domain_to_sg(VsyncVspace::task_to_domain(&cfg))
}

/// Creates one vspace (optionally backed by a block device for on-disk journal).
pub fn vsync_vspace_create(
    checkpoint_fn: Option<VsyncCheckpointFn>,
    fs_context: Option<Arc<dyn Any + Send + Sync>>,
    config: Option<VsyncConfig>,
    block_device: Option<Arc<dyn BlockDevice>>,
    journal_start_sector: u64,
    journal_total_sectors: u64,
) -> Arc<VsyncVspace> {
    let cfg = config.unwrap_or_default();
    *VSYNC_GLOBAL_CFG.lock() = cfg.clone();
    let vspace = VsyncVspace::new(
        cfg,
        checkpoint_fn,
        fs_context,
        block_device,
        journal_start_sector,
        journal_total_sectors,
    );
    // Set MOUNTED flag and write the superblock if backed by a block device.
    if vspace.block_device.is_some() {
        let _ = vspace.read_journal_super();
        vspace
            .journal_super_flags
            .fetch_or(VSYNC_JOURNAL_FLAG_MOUNTED, Ordering::Release);
        vspace.write_journal_super();
    }
    vspace
}

/// Destroys one vspace, flushing pending data and clearing the MOUNTED flag.
pub fn vsync_vspace_destroy(vspace: &Arc<VsyncVspace>) {
    vspace.set_state(VsyncVspaceState::Unmounting);
    vspace.unmount_checkpoint();
    // Clear MOUNTED flag for clean unmount
    let flags = vspace.journal_super_flags.load(Ordering::Acquire);
    vspace
        .journal_super_flags
        .store(flags & !VSYNC_JOURNAL_FLAG_MOUNTED, Ordering::Release);
    vspace.write_journal_super();
    vspace.destroy_file_tracker();
    vspace.set_state(VsyncVspaceState::Destroyed);
}

/// Recovers committed records from the on-disk journal after a crash.
///
/// Returns 0 on success (including skipped clean-unmount), negative errno on failure.
pub fn vsync_vspace_replay(vspace: &Arc<VsyncVspace>) -> i32 {
    let Some(ref _bdev) = vspace.block_device else {
        return 0; // No block device → nothing to replay
    };

    // Phase 1: Read journal superblock
    if vspace.read_journal_super() != 0 {
        // Fresh or invalid superblock — nothing to replay
        return 0;
    }

    // Phase 1.5: Skip replay if clean unmount
    let flags = vspace.journal_super_flags.load(Ordering::Acquire);
    let all_empty = vspace.shards.iter().all(|s| {
        s.journal.head_sector.load(Ordering::Acquire)
            == s.journal.tail_sector.load(Ordering::Acquire)
    });
    if flags & VSYNC_JOURNAL_FLAG_MOUNTED == 0 && all_empty {
        return 0; // Clean unmount — nothing to replay
    }

    // Phase 2: Scan each shard's journal
    let mut all_records = Vec::new();
    for shard in &vspace.shards {
        let mut shard_records = Vec::new();
        if shard.scan_journal(vspace.block_device.as_deref(), &mut shard_records) == 0 {
            all_records.extend(shard_records);
        }
    }

    if all_records.is_empty() {
        // Reset shards to clean state
        for shard in &vspace.shards {
            let start = shard.journal.start_sector;
            shard.journal.head_sector.store(start, Ordering::Release);
            shard
                .journal
                .committed_head_sector
                .store(start, Ordering::Release);
            shard.journal.tail_sector.store(start, Ordering::Release);
        }
        return 0;
    }

    // Phase 3: Sort records by sequence
    all_records.sort_by_key(Record::seq);

    // Phase 4: Replay through checkpoint
    {
        let fs_context = vspace.fs_context.read().clone();
        let coalesce_ops = vspace.coalesce_ops.read().clone();
        let checkpoint_fn = vspace.checkpoint_fn.read().clone();
        let unit_context = ();
        let context = fs_context
            .as_ref()
            .map(|context| context.as_ref() as &dyn Any)
            .unwrap_or(&unit_context);
        let result = if let Some(coalesce_ops) = coalesce_ops {
            checkpoint_coalesce(&all_records, coalesce_ops.as_ref(), context, false)
        } else if let Some(checkpoint_fn) = checkpoint_fn {
            checkpoint_fn(context, &all_records)
        } else {
            Ok(())
        };
        if result.is_err() {
            return -(Errno::EIO as i32);
        }
    }

    // Phase 5: Reset shard state and mark as mounted
    for shard in &vspace.shards {
        let start = shard.journal.start_sector;
        shard.journal.head_sector.store(start, Ordering::Release);
        shard
            .journal
            .committed_head_sector
            .store(start, Ordering::Release);
        shard.journal.tail_sector.store(start, Ordering::Release);
    }
    vspace
        .journal_super_flags
        .store(VSYNC_JOURNAL_FLAG_MOUNTED, Ordering::Release);
    vspace.write_journal_super();

    0
}

/// Starts one operation.
pub fn vsync_op_start(
    vspace: &Arc<VsyncVspace>,
    rtype: VsyncRecordType,
    ino: u64,
) -> Result<VsyncHandle> {
    vspace.op_start(rtype, ino)
}

/// Completes one operation.
pub fn vsync_op_complete(handle: &mut VsyncHandle, payload: Option<Vec<u8>>) -> bool {
    let vspace = handle.vspace.clone();
    vspace.op_complete(handle, payload)
}

/// Sets one pre-commit barrier.
pub fn vsync_handle_set_pre_commit_barrier(handle: &mut VsyncHandle, barrier: Arc<CountdownLatch>) {
    handle.set_pre_commit_barrier(barrier);
}

/// Returns handle sequence.
pub fn vsync_handle_seq(handle: &VsyncHandle) -> VsyncSeq {
    handle.seq()
}

/// Returns handle sync-group ID.
pub fn vsync_handle_sg_id(handle: &VsyncHandle) -> u32 {
    handle.sg_id()
}

/// Returns handle CPU slot index.
pub fn vsync_handle_cpu(handle: &VsyncHandle) -> i32 {
    handle.cpu()
}

/// Runs one domain-aware sync-until.
pub fn vsync_sync_until(vspace: &Arc<VsyncVspace>, target_seq: VsyncSeq) -> Result<()> {
    vspace.sync_until(target_seq)
}

/// Runs one global sync.
pub fn vsync_sync_all(vspace: &Arc<VsyncVspace>) -> Result<()> {
    vspace.sync_all()
}

/// Runs one global sync-until.
pub fn vsync_sync_all_until(vspace: &Arc<VsyncVspace>, target_seq: VsyncSeq) -> Result<()> {
    vspace.sync_all_until(target_seq)
}

/// Runs one multi-group sync.
pub fn vsync_sync_multi_sg(
    vspace: &Arc<VsyncVspace>,
    sg_targets: &[VsyncSeq; VSYNC_NUM_SYNC_GROUPS],
) -> Result<()> {
    vspace.sync_multi_sg(sg_targets)
}

/// Runs one file-scoped sync.
pub fn vsync_sync_file(vspace: &Arc<VsyncVspace>, ino: u64, target_seq: VsyncSeq) -> Result<()> {
    vspace.sync_file(ino, target_seq)
}

/// Runs one checkpoint.
pub fn vsync_vspace_checkpoint(vspace: &Arc<VsyncVspace>, _target_seq: VsyncSeq) -> Result<u64> {
    vspace.checkpoint(0)
}

/// Sets no-batch-coalesce mode.
pub fn vsync_vspace_set_no_batch_coalesce(vspace: &Arc<VsyncVspace>, val: bool) {
    vspace.set_no_batch_coalesce(val);
}

/// Registers coalescing callbacks.
pub fn vsync_vspace_set_coalesce_ops(
    vspace: &Arc<VsyncVspace>,
    ops: Option<Arc<dyn VsyncCoalesceOps>>,
) -> i32 {
    vspace.set_coalesce_ops(ops)
}

/// Registers optional shadow callbacks.
pub fn vsync_vspace_set_shadow_ops(
    vspace: &Arc<VsyncVspace>,
    shadow_materialize: Option<VsyncShadowMaterializeFn>,
    shadow_reset: Option<VsyncShadowResetFn>,
) {
    vspace.set_shadow_ops(shadow_materialize, shadow_reset);
}

/// Returns journal start sector.
pub fn vsync_vspace_start_sector(vspace: &Arc<VsyncVspace>) -> u64 {
    vspace.start_sector()
}

/// Returns journal total sectors.
pub fn vsync_vspace_total_sectors(vspace: &Arc<VsyncVspace>) -> u64 {
    vspace.total_sectors()
}

/// Returns global stable sequence.
pub fn vsync_vspace_stable_seq(vspace: &Arc<VsyncVspace>) -> VsyncSeq {
    vspace.stable_seq()
}

/// Returns whether vspace is in recovery stage.
pub fn vsync_vspace_is_recovering(vspace: &Arc<VsyncVspace>) -> bool {
    vspace.is_recovering()
}
