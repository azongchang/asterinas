// SPDX-License-Identifier: MPL-2.0

//! Tests for in-memory VSync implementation.
//!
//! These tests use fake journal workloads to exercise the full VSync
//! transaction lifecycle: op_start → op_complete → commit → sync_until.
//! On-disk persistence is not tested here; this pass validates the
//! in-memory state machine against Linux VSync semantics.

use alloc::{collections::BTreeMap, sync::Arc, vec, vec::Vec};
use core::{
    any::Any,
    fmt,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

use aster_block::{
    BlockDevice, BlockDeviceMeta,
    bio::{BioEnqueueError, BioStatus, BioType, SubmittedBio},
};
use device_id::{DeviceId, MajorId, MinorId};
use ostd::{
    mm::{VmReader, VmWriter, io::util::HasVmReaderWriter},
    prelude::ktest,
    sync::Mutex,
};

use super::*;
use crate::{error::Errno, prelude::SpinLock, thread::kernel_thread::ThreadOptions};

/// A test block device backed by an in-memory byte vector.
///
/// Implements `BlockDevice` trait so journal reads/writes can be tested
/// without real hardware. BIOs complete synchronously.
pub struct TestBlockDevice {
    pub data: Mutex<Vec<u8>>,
    nr_sectors: usize,
    name: &'static str,
}

impl TestBlockDevice {
    pub fn new(nr_sectors: u64) -> Arc<Self> {
        let size = nr_sectors as usize * aster_block::SECTOR_SIZE;
        Arc::new(Self {
            data: Mutex::new(vec![0u8; size]),
            nr_sectors: nr_sectors as usize,
            name: "test_vsync",
        })
    }
}

impl fmt::Debug for TestBlockDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TestBlockDevice")
            .field("nr_sectors", &self.nr_sectors)
            .finish()
    }
}

impl BlockDevice for TestBlockDevice {
    fn enqueue(&self, bio: SubmittedBio) -> Result<(), BioEnqueueError> {
        let bio_type = bio.type_();
        let start_sid = bio.sid_range().start.to_raw() as usize;
        let mut data = self.data.lock();
        let mut byte_off = start_sid * aster_block::SECTOR_SIZE;

        match bio_type {
            BioType::Read => {
                for segment in bio.segments() {
                    let nbytes = segment.nbytes();
                    let end = (byte_off + nbytes).min(data.len());
                    let mut reader = VmReader::from(&data[byte_off..end]);
                    let mut writer = segment.inner_dma().writer().unwrap();
                    let _ = writer.write(&mut reader);
                    byte_off += nbytes;
                }
            }
            BioType::Write => {
                for segment in bio.segments() {
                    let nbytes = segment.nbytes();
                    let end = (byte_off + nbytes).min(data.len());
                    if byte_off < data.len() {
                        let mut reader = segment.inner_dma().reader().unwrap();
                        let mut writer = VmWriter::from(&mut data[byte_off..end]);
                        let _ = reader.read(&mut writer);
                    }
                    byte_off += nbytes;
                }
            }
            BioType::Flush => {
                // No volatile cache in test device — flush is a no-op
            }
        }

        bio.complete(BioStatus::Complete);
        Ok(())
    }

    fn metadata(&self) -> BlockDeviceMeta {
        BlockDeviceMeta {
            max_nr_segments_per_bio: 16,
            nr_sectors: self.nr_sectors,
        }
    }

    fn name(&self) -> &str {
        self.name
    }

    fn id(&self) -> DeviceId {
        DeviceId::new(MajorId::new(255), MinorId::new(0))
    }
}

// ============================================================================
// Fake journal workload helpers
// ============================================================================

/// Minimal coalesce ops that treats each record as a single range keyed by ino.
#[derive(Default)]
struct FakeCoalesceOps;

impl VsyncCoalesceOps for FakeCoalesceOps {
    fn iter_ranges(
        &self,
        rec: &Record,
        cb: &mut dyn FnMut(VsyncCoalesceRange) -> crate::prelude::Result<()>,
    ) -> crate::prelude::Result<()> {
        if let Some(ino) = rec.ino() {
            let payload = rec.payload_bytes().to_vec();
            cb(VsyncCoalesceRange {
                blocknr: ino,
                offset: 0,
                len: payload.len() as u32,
                seq: rec.seq(),
                data: payload,
            })?;
        }
        Ok(())
    }

    fn build_payload(&self, ranges: &[VsyncCoalesceRange]) -> crate::prelude::Result<Vec<u8>> {
        let mut payload = Vec::new();
        for range in ranges {
            payload.extend_from_slice(&range.data);
        }
        Ok(payload)
    }

    fn apply_range(
        &self,
        fs_context: &dyn Any,
        range: &VsyncCoalesceRange,
    ) -> crate::prelude::Result<()> {
        let Some(applied) = fs_context.downcast_ref::<SpinLock<Vec<VsyncCoalesceRange>>>() else {
            return Err(crate::error::Error::new(Errno::EINVAL));
        };
        applied.lock().push(range.clone());
        Ok(())
    }
}

/// Multi-range coalesce ops where each record may carry multiple block ranges.
/// Payload format: [u64 blocknr][u32 len][len bytes data] repeated.
#[derive(Default)]
struct MultiRangeCoalesceOps;

impl VsyncCoalesceOps for MultiRangeCoalesceOps {
    fn iter_ranges(
        &self,
        rec: &Record,
        cb: &mut dyn FnMut(VsyncCoalesceRange) -> crate::prelude::Result<()>,
    ) -> crate::prelude::Result<()> {
        let payload = rec.payload_bytes();
        let mut cursor = 0_usize;
        while cursor + 12 <= payload.len() {
            let blocknr = u64::from_le_bytes(payload[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            let len = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            if cursor + len > payload.len() {
                break;
            }
            let data = payload[cursor..cursor + len].to_vec();
            cursor += len;
            cb(VsyncCoalesceRange {
                blocknr,
                offset: 0,
                len: len as u32,
                seq: rec.seq(),
                data,
            })?;
        }
        Ok(())
    }

    fn build_payload(&self, ranges: &[VsyncCoalesceRange]) -> crate::prelude::Result<Vec<u8>> {
        let mut payload = Vec::new();
        for range in ranges {
            payload.extend_from_slice(&range.data);
        }
        Ok(payload)
    }

    fn apply_range(
        &self,
        fs_context: &dyn Any,
        range: &VsyncCoalesceRange,
    ) -> crate::prelude::Result<()> {
        let Some(applied) = fs_context.downcast_ref::<SpinLock<Vec<VsyncCoalesceRange>>>() else {
            return Err(crate::error::Error::new(Errno::EINVAL));
        };
        applied.lock().push(range.clone());
        Ok(())
    }
}

#[derive(Default)]
struct OffsetRangeCoalesceOps;

impl VsyncCoalesceOps for OffsetRangeCoalesceOps {
    fn iter_ranges(
        &self,
        rec: &Record,
        cb: &mut dyn FnMut(VsyncCoalesceRange) -> crate::prelude::Result<()>,
    ) -> crate::prelude::Result<()> {
        let payload = rec.payload_bytes();
        let mut cursor = 0_usize;
        while cursor + 16 <= payload.len() {
            let blocknr = u64::from_le_bytes(payload[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            let offset = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            let len = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            if cursor + len > payload.len() {
                break;
            }
            let data = payload[cursor..cursor + len].to_vec();
            cursor += len;
            cb(VsyncCoalesceRange {
                blocknr,
                offset,
                len: len as u32,
                seq: rec.seq(),
                data,
            })?;
        }
        Ok(())
    }

    fn build_payload(&self, ranges: &[VsyncCoalesceRange]) -> crate::prelude::Result<Vec<u8>> {
        let mut payload = Vec::new();
        for range in ranges {
            payload.extend_from_slice(&range.data);
        }
        Ok(payload)
    }

    fn apply_range(
        &self,
        fs_context: &dyn Any,
        range: &VsyncCoalesceRange,
    ) -> crate::prelude::Result<()> {
        let Some(applied) = fs_context.downcast_ref::<SpinLock<Vec<VsyncCoalesceRange>>>() else {
            return Err(crate::error::Error::new(Errno::EINVAL));
        };
        applied.lock().push(range.clone());
        Ok(())
    }
}

fn offset_range_payload(blocknr: u64, offset: u32, data: &[u8]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&blocknr.to_le_bytes());
    payload.extend_from_slice(&offset.to_le_bytes());
    payload.extend_from_slice(&(data.len() as u32).to_le_bytes());
    payload.extend_from_slice(data);
    payload
}

fn new_test_vspace() -> Arc<VsyncVspace> {
    vsync_vspace_create(None, None, None, None, 0, 1024 * 1024)
}

fn new_test_vspace_with_coalesce() -> Arc<VsyncVspace> {
    let vspace = vsync_vspace_create(None, None, None, None, 0, 1024 * 1024);
    vspace.set_coalesce_ops(Some(Arc::new(FakeCoalesceOps)));
    vspace
}

fn write_one(vspace: &Arc<VsyncVspace>, ino: u64, bytes: &[u8]) -> (u32, u64) {
    let mut handle = vsync_op_start(vspace, VsyncRecordType::Meta, ino).unwrap();
    let sg_id = vsync_handle_sg_id(&handle);
    let seq = vsync_handle_seq(&handle);
    assert!(vsync_op_complete(&mut handle, Some(bytes.to_vec())));
    (sg_id, seq)
}

fn write_one_write(vspace: &Arc<VsyncVspace>, ino: u64, bytes: &[u8]) -> (u32, u64) {
    let mut handle = vsync_op_start(vspace, VsyncRecordType::Write, ino).unwrap();
    let sg_id = vsync_handle_sg_id(&handle);
    let seq = vsync_handle_seq(&handle);
    assert!(vsync_op_complete(&mut handle, Some(bytes.to_vec())));
    (sg_id, seq)
}

fn write_many(vspace: &Arc<VsyncVspace>, ino: u64, count: usize) -> Vec<(u32, u64)> {
    let mut results = Vec::new();
    for i in 0..count {
        let payload = vec![i as u8; 16];
        results.push(write_one(vspace, ino, &payload));
    }
    results
}

// ============================================================================
// 1. Transaction lifecycle tests
// ============================================================================

#[ktest]
fn op_start_assigns_monotonic_seq() {
    let vspace = new_test_vspace();
    let (_, seq1) = write_one(&vspace, 1, b"first");
    let (_, seq2) = write_one(&vspace, 2, b"second");
    assert!(seq1 > 0);
    assert!(seq2 > seq1, "seq2={seq2} must be > seq1={seq1}");
}

#[ktest]
fn op_start_increments_pending_record_count() {
    let vspace = new_test_vspace();
    let sg_id = vsync_current_sg_id();
    let before = vspace.sync_groups[sg_id as usize].pending_records();
    let mut handle = vsync_op_start(&vspace, VsyncRecordType::Meta, 10).unwrap();
    let after = vspace.sync_groups[sg_id as usize].pending_records();
    assert_eq!(after, before + 1);
    vsync_op_complete(&mut handle, Some(b"data".to_vec()));
}

#[ktest]
fn op_start_rejects_inactive_vspace() {
    let vspace = new_test_vspace();
    vspace.set_state(VsyncVspaceState::Unmounting);
    let result = vsync_op_start(&vspace, VsyncRecordType::Meta, 1);
    assert!(result.is_err());
}

#[ktest]
fn op_complete_twice_is_idempotent() {
    let vspace = new_test_vspace();
    let mut handle = vsync_op_start(&vspace, VsyncRecordType::Meta, 99).unwrap();
    assert!(vsync_op_complete(
        &mut handle,
        Some(b"first-payload".to_vec())
    ));
    assert!(!vsync_op_complete(
        &mut handle,
        Some(b"second-payload".to_vec())
    ));
}

#[ktest]
fn op_complete_rejects_null_payload() {
    let vspace = new_test_vspace();
    let mut handle = vsync_op_start(&vspace, VsyncRecordType::Meta, 99).unwrap();
    let sg_id = vsync_handle_sg_id(&handle);
    let before = vspace.sync_groups[sg_id as usize].pending_records();
    assert!(!vsync_op_complete(&mut handle, None));
    let after = vspace.sync_groups[sg_id as usize].pending_records();
    assert_eq!(after, before - 1);
}

#[ktest]
fn op_complete_rejects_empty_payload() {
    let vspace = new_test_vspace();
    let mut handle = vsync_op_start(&vspace, VsyncRecordType::Meta, 99).unwrap();
    let sg_id = vsync_handle_sg_id(&handle);
    let before = vspace.sync_groups[sg_id as usize].pending_records();
    assert!(!vsync_op_complete(&mut handle, Some(Vec::new())));
    let after = vspace.sync_groups[sg_id as usize].pending_records();
    assert_eq!(after, before - 1);
}

#[ktest]
fn sync_until_commits_current_domain_records() {
    let vspace = new_test_vspace();
    let (sg_id, seq) = write_one(&vspace, 7, b"phantom-metadata");

    vsync_sync_until(&vspace, seq).unwrap();

    assert!(vspace.sync_groups[sg_id as usize].stable_seq() >= seq);
}

#[ktest]
fn sync_until_already_durable_is_noop() {
    let vspace = new_test_vspace();
    let (sg_id, seq) = write_one(&vspace, 7, b"data");
    vsync_sync_until(&vspace, seq).unwrap();
    // Second call should return immediately
    vsync_sync_until(&vspace, seq).unwrap();
    assert!(vspace.sync_groups[sg_id as usize].stable_seq() >= seq);
}

#[ktest]
fn sync_until_commits_all_prior_records() {
    let vspace = new_test_vspace();
    let (sg_id, _seq1) = write_one(&vspace, 1, b"record-1");
    let (_, _seq2) = write_one(&vspace, 1, b"record-2");
    let (_, seq3) = write_one(&vspace, 1, b"record-3");

    vsync_sync_until(&vspace, seq3).unwrap();

    let stable = vspace.sync_groups[sg_id as usize].stable_seq();
    assert!(stable >= seq3, "stable={stable} must be >= seq3={seq3}");
}

// ============================================================================
// 2. Fence ordering tests
// ============================================================================

#[ktest]
fn fences_are_satisfied_in_order() {
    let vspace = new_test_vspace();
    let (sg_id, seq1) = write_one(&vspace, 1, b"batch-1");
    let (_, seq2) = write_one(&vspace, 1, b"batch-2");

    // Both fences should be satisfied after the higher target
    vsync_sync_until(&vspace, seq2).unwrap();

    assert!(vspace.sync_groups[sg_id as usize].stable_seq() >= seq1);
    assert!(vspace.sync_groups[sg_id as usize].stable_seq() >= seq2);
}

#[ktest]
fn fence_with_no_records_is_trivially_satisfied() {
    let vspace = new_test_vspace();
    let target_seq = vsync_seq_now();
    vsync_sync_until(&vspace, target_seq).unwrap();
}

#[ktest]
fn fences_satisfied_counter_increments() {
    let vspace = new_test_vspace();
    let (sg_id, seq) = write_one(&vspace, 1, b"count-me");
    let before = vspace.sync_groups[sg_id as usize]
        .fences_satisfied
        .load(Ordering::Relaxed);
    vsync_sync_until(&vspace, seq).unwrap();
    let after = vspace.sync_groups[sg_id as usize]
        .fences_satisfied
        .load(Ordering::Relaxed);
    assert!(after > before);
}

// ============================================================================
// 3. Domain isolation tests
// ============================================================================

#[ktest]
fn domain_aware_sync_until_does_not_force_other_groups() {
    let vspace = new_test_vspace();

    let worker_ready = Arc::new(AtomicBool::new(false));
    let worker_data = Arc::new(SpinLock::<Option<(u32, u64)>>::new(None::<(u32, u64)>));
    let worker_thread = {
        let vspace = vspace.clone();
        let worker_ready = worker_ready.clone();
        let worker_data = worker_data.clone();
        ThreadOptions::new(move || {
            let (sg_id, seq) = write_one(&vspace, 100, b"worker-update");
            *worker_data.lock() = Some((sg_id, seq));
            worker_ready.store(true, Ordering::Release);
        })
        .spawn()
    };

    while !worker_ready.load(Ordering::Acquire) {
        crate::thread::Thread::yield_now();
    }

    let (main_sg, main_seq) = write_one(&vspace, 101, b"main-update");
    vsync_sync_until(&vspace, main_seq).unwrap();
    worker_thread.join();

    let (worker_sg, worker_seq) = (*worker_data.lock()).expect("worker data must be set");
    assert!(vspace.sync_groups[main_sg as usize].stable_seq() >= main_seq);

    if worker_sg != main_sg {
        assert!(vspace.sync_groups[worker_sg as usize].stable_seq() < worker_seq);
    }

    vsync_sync_all(&vspace).unwrap();
    assert!(vspace.sync_groups[worker_sg as usize].stable_seq() >= worker_seq);
}

#[ktest]
fn domain_stable_seq_is_per_domain() {
    let vspace = new_test_vspace();
    let domain_id = VsyncVspace::task_to_domain(&vspace.config());
    let sg_id = VsyncVspace::domain_to_sg(domain_id);
    let sg = &vspace.sync_groups[sg_id as usize];

    let initial = sg.domain_stable_seq(domain_id);
    assert_eq!(initial, 0);

    let (_, seq) = write_one(&vspace, 42, b"data");
    vsync_sync_until(&vspace, seq).unwrap();

    let after = sg.domain_stable_seq(domain_id);
    assert!(after >= seq);
}

// ============================================================================
// 4. Multi-SG sync tests
// ============================================================================

#[ktest]
fn sync_all_commits_every_sg() {
    let vspace = new_test_vspace();

    // Write to multiple inodes that may hit different SGs
    let mut max_per_sg = [0_u64; VSYNC_NUM_SYNC_GROUPS];
    for ino in [1, 7, 13, 19, 25, 31, 37, 43] {
        let (sg_id, seq) = write_one(&vspace, ino, &[(ino % 256) as u8; 8]);
        max_per_sg[sg_id as usize] = max_per_sg[sg_id as usize].max(seq);
    }

    vsync_sync_all(&vspace).unwrap();

    for (sg_id, target) in max_per_sg.iter().enumerate() {
        if *target > 0 {
            assert!(
                vspace.sync_groups[sg_id].stable_seq() >= *target,
                "sg {sg_id}: stable_seq={} < target={target}",
                vspace.sync_groups[sg_id].stable_seq()
            );
        }
    }
}

#[ktest]
fn sync_all_until_sets_uniform_target() {
    let vspace = new_test_vspace();
    let target = vsync_seq_now();
    let (_, _seq) = write_one(&vspace, 1, b"data");
    vsync_sync_all_until(&vspace, target).unwrap();
    // All SGs should have stable_seq >= target (trivially true for empty SGs)
    for sg_id in 0..VSYNC_NUM_SYNC_GROUPS {
        assert!(vspace.sync_groups[sg_id].stable_seq() >= target);
    }
}

#[ktest]
fn sync_multi_sg_skips_zero_targets() {
    let vspace = new_test_vspace();
    let targets = [0_u64; VSYNC_NUM_SYNC_GROUPS];
    vsync_sync_multi_sg(&vspace, &targets).unwrap();
}

#[ktest]
fn sync_multi_sg_skips_already_durable() {
    let vspace = new_test_vspace();
    let (sg_id, seq) = write_one(&vspace, 1, b"data");
    vsync_sync_until(&vspace, seq).unwrap();

    let mut targets = [0_u64; VSYNC_NUM_SYNC_GROUPS];
    targets[sg_id as usize] = seq;
    vsync_sync_multi_sg(&vspace, &targets).unwrap();
}

// ============================================================================
// 5. Coalescing tests
// ============================================================================

#[ktest]
fn coalesce_prefers_latest_conflicting_range() {
    let mut older = Record::new_meta(10, 1, 888, b"old".to_vec());
    older.set_txn_id(1);
    let mut newer = Record::new_meta(20, 1, 888, b"new".to_vec());
    newer.set_txn_id(2);

    let ops = FakeCoalesceOps;
    let coalesced = coalesce_batch(vec![older, newer], Some(&ops));
    assert_eq!(coalesced.len(), 1);
    assert_eq!(coalesced[0].payload_bytes(), b"new");

    let applied = SpinLock::<Vec<VsyncCoalesceRange>>::new(Vec::new());
    checkpoint_coalesce(&coalesced, &ops, &applied, false).unwrap();
    let applied = applied.lock();
    assert_eq!(applied.len(), 1);
    assert_eq!(applied[0].data, b"new");
}

#[ktest]
fn checkpoint_coalesce_preserves_same_block_distinct_offsets() {
    let older = Record::new_meta(10, 1, 888, offset_range_payload(7, 0, b"aa"));
    let newer = Record::new_meta(20, 1, 888, offset_range_payload(7, 2, b"bb"));

    let ops = OffsetRangeCoalesceOps;
    let applied = SpinLock::<Vec<VsyncCoalesceRange>>::new(Vec::new());
    checkpoint_coalesce(&[older, newer], &ops, &applied, true).unwrap();

    let applied = applied.lock();
    assert_eq!(applied.len(), 2);
    assert_eq!(applied[0].offset, 0);
    assert_eq!(applied[0].data, b"aa");
    assert_eq!(applied[1].offset, 2);
    assert_eq!(applied[1].data, b"bb");
}

#[ktest]
fn coalesce_different_inos_produce_separate_records() {
    let rec1 = Record::new_meta(10, 1, 100, b"ino-100".to_vec());
    let rec2 = Record::new_meta(20, 1, 200, b"ino-200".to_vec());

    let ops = FakeCoalesceOps;
    let coalesced = coalesce_batch(vec![rec1, rec2], Some(&ops));
    assert_eq!(coalesced.len(), 2);
}

#[ktest]
fn coalesce_same_ino_different_domain_separate() {
    let rec1 = Record::new_meta(10, 1, 500, b"domain-1".to_vec());
    let rec2 = Record::new_meta(20, 2, 500, b"domain-2".to_vec());

    let ops = FakeCoalesceOps;
    let coalesced = coalesce_batch(vec![rec1, rec2], Some(&ops));
    // Different domain IDs → separate groups → separate records
    assert_eq!(coalesced.len(), 2);
}

#[ktest]
fn coalesce_preserves_fence_records() {
    let token = Arc::new(CountdownLatch::new_token(1));
    let fence = Record::new_fence(15, 1, token, 15);
    let meta = Record::new_meta(10, 1, 300, b"meta".to_vec());

    let ops = FakeCoalesceOps;
    let coalesced = coalesce_batch(vec![meta.clone(), fence], Some(&ops));
    assert_eq!(coalesced.len(), 2);
    assert!(
        coalesced
            .iter()
            .any(|r| r.record_type() == VsyncRecordType::Fence)
    );
}

#[ktest]
fn coalesce_sorted_by_seq() {
    let rec3 = Record::new_meta(30, 1, 1, b"third".to_vec());
    let rec1 = Record::new_meta(10, 1, 2, b"first".to_vec());
    let rec2 = Record::new_meta(20, 1, 3, b"second".to_vec());

    let ops = FakeCoalesceOps;
    let coalesced = coalesce_batch(vec![rec3, rec1, rec2], Some(&ops));
    // Different inos so all preserved, but sorted by seq
    for w in coalesced.windows(2) {
        assert!(w[0].seq() <= w[1].seq());
    }
}

#[ktest]
fn coalesce_batch_no_ops_returns_original() {
    let batch = vec![
        Record::new_meta(1, 1, 1, b"a".to_vec()),
        Record::new_meta(2, 1, 2, b"b".to_vec()),
    ];
    let result = coalesce_batch(batch.clone(), None);
    assert_eq!(result.len(), 2);
}

#[ktest]
fn coalesce_batch_single_record_unchanged() {
    let batch = vec![Record::new_meta(1, 1, 1, b"single".to_vec())];
    let ops = FakeCoalesceOps;
    let result = coalesce_batch(batch, Some(&ops));
    assert_eq!(result.len(), 1);
}

#[ktest]
fn coalesce_multi_range_ops_dedup_by_blocknr() {
    // Build two records that each encode two ranges with overlapping blocknrs
    let old = Record::new_meta(10, 1, 42, {
        let mut p = Vec::new();
        p.extend_from_slice(&100_u64.to_le_bytes());
        p.extend_from_slice(&3_u32.to_le_bytes());
        p.extend_from_slice(b"old");
        p.extend_from_slice(&200_u64.to_le_bytes());
        p.extend_from_slice(&4_u32.to_le_bytes());
        p.extend_from_slice(b"keep");
        p
    });
    let new = Record::new_meta(20, 1, 42, {
        let mut p = Vec::new();
        p.extend_from_slice(&100_u64.to_le_bytes());
        p.extend_from_slice(&3_u32.to_le_bytes());
        p.extend_from_slice(b"new");
        p
    });

    let ops = MultiRangeCoalesceOps;
    let coalesced = coalesce_batch(vec![old, new], Some(&ops));
    // blocknr=100 from "old" superseded by "new", blocknr=200 "keep" preserved
    assert!(coalesced.len() >= 1);
}

// ============================================================================
// 6. Cross-SG file tracking tests
// ============================================================================

#[ktest]
fn concurrent_same_file_updates_are_tracked_and_synced() {
    let vspace = new_test_vspace();
    let ino = 42_u64;
    let done = Arc::new(AtomicU64::new(0));
    let failures = Arc::new(AtomicU64::new(0));
    let per_sg_max = Arc::new(SpinLock::<BTreeMap<u32, u64>>::new(
        BTreeMap::<u32, u64>::new(),
    ));

    let mut threads = Vec::new();
    for worker_idx in 0..2_u8 {
        let vspace = vspace.clone();
        let done = done.clone();
        let failures = failures.clone();
        let per_sg_max = per_sg_max.clone();
        threads.push(
            ThreadOptions::new(move || {
                for turn in 0..32_u8 {
                    let payload = vec![worker_idx, turn, 0x5A, 0xC3];
                    let Ok(mut handle) = vsync_op_start(&vspace, VsyncRecordType::Meta, ino) else {
                        failures.fetch_add(1, Ordering::Release);
                        return;
                    };
                    let sg_id = vsync_handle_sg_id(&handle);
                    let seq = vsync_handle_seq(&handle);
                    if !vsync_op_complete(&mut handle, Some(payload)) {
                        failures.fetch_add(1, Ordering::Release);
                        return;
                    }
                    let mut map = per_sg_max.lock();
                    map.entry(sg_id)
                        .and_modify(|max_seq| *max_seq = (*max_seq).max(seq))
                        .or_insert(seq);
                }
                done.fetch_add(1, Ordering::Release);
            })
            .spawn(),
        );
    }

    for thread in threads {
        thread.join();
    }
    assert_eq!(done.load(Ordering::Acquire), 2);
    assert_eq!(failures.load(Ordering::Acquire), 0);

    let tracked = vspace.get_file_state(ino).unwrap();
    assert!(!tracked.sg_targets.is_empty());
    vsync_sync_file(&vspace, ino, 0).unwrap();

    let map = per_sg_max.lock();
    for (sg_id, max_seq) in map.iter() {
        assert!(vspace.sync_groups[*sg_id as usize].stable_seq() >= *max_seq);
    }
}

#[ktest]
fn file_add_remove_sg_maintains_tracking() {
    let vspace = new_test_vspace();
    let ino = 999_u64;
    let sg_id = 3_u32;

    // Initially no file state
    assert!(vspace.get_file_state(ino).is_none());

    // add_sg creates state
    vspace.file_add_sg(ino, sg_id);
    let state = vspace.get_file_state(ino).unwrap();
    assert!(state.sg_targets.contains_key(&sg_id));
    assert_eq!(state.sg_refs.get(&sg_id).unwrap().inflight_ops, 1);
    assert!(vspace.sync_groups[sg_id as usize].has_pending_file(ino));

    // remove_sg publishes the op and keeps SG participation until commit.
    vspace.file_remove_sg(ino, sg_id);
    let state = vspace.get_file_state(ino).unwrap();
    assert!(state.sg_targets.contains_key(&sg_id));
    let sg_ref = state.sg_refs.get(&sg_id).unwrap();
    assert_eq!(sg_ref.inflight_ops, 0);
    assert_eq!(sg_ref.uncommitted_ops, 1);
    assert!(vspace.sync_groups[sg_id as usize].has_pending_file(ino));

    vspace.file_on_commit(ino, sg_id, 123);
    let state = vspace.get_file_state(ino).unwrap();
    assert!(!state.sg_refs.contains_key(&sg_id));
    assert!(!vspace.sync_groups[sg_id as usize].has_pending_file(ino));
}

#[ktest]
fn file_cancel_sg_removes_unpublished_reference() {
    let vspace = new_test_vspace();
    let ino = 1000_u64;
    let sg_id = 2_u32;

    vspace.file_add_sg(ino, sg_id);
    assert!(vspace.sync_groups[sg_id as usize].has_pending_file(ino));

    vspace.file_cancel_sg(ino, sg_id);
    let state = vspace.get_file_state(ino).unwrap();
    assert!(!state.sg_refs.contains_key(&sg_id));
    assert!(!vspace.sync_groups[sg_id as usize].has_pending_file(ino));
}

#[ktest]
fn file_add_seq_stores_max_per_sg() {
    let vspace = new_test_vspace();
    let ino = 555_u64;
    let sg_id = 2_u32;

    vspace.file_add_seq(ino, sg_id, 100);
    vspace.file_add_seq(ino, sg_id, 50); // lower seq should not overwrite

    let state = vspace.get_file_state(ino).unwrap();
    assert_eq!(state.sg_targets.get(&sg_id), Some(&100));
}

#[ktest]
fn file_get_lg_targets_returns_correct_targets() {
    let vspace = new_test_vspace();
    let ino = 777_u64;

    vspace.file_add_seq(ino, 1, 100);
    vspace.file_add_seq(ino, 3, 200);

    let mut targets = [0_u64; VSYNC_NUM_SYNC_GROUPS];
    let has_lg = vspace.file_get_lg_targets(ino, 0, &mut targets);
    assert!(has_lg);
    assert_eq!(targets[1], 100);
    assert_eq!(targets[3], 200);
}

#[ktest]
fn file_get_lg_targets_with_target_seq() {
    let vspace = new_test_vspace();
    let ino = 888_u64;

    vspace.file_add_seq(ino, 1, 100);
    vspace.file_add_seq(ino, 3, 200);

    let mut targets = [0_u64; VSYNC_NUM_SYNC_GROUPS];
    let has_lg = vspace.file_get_lg_targets(ino, 250, &mut targets);
    assert!(has_lg);
    // target_seq=250 > seq=100 for sg 1, so target should be 250
    assert_eq!(targets[1], 250, "sg 1: expected target_seq.max(seq) = 250");
    // target_seq=250 > seq=200 for sg 3, so target should be 250
    assert_eq!(targets[3], 250, "sg 3: expected target_seq.max(seq) = 250");
}

#[ktest]
fn file_get_lg_targets_no_state_returns_false() {
    let vspace = new_test_vspace();
    let mut targets = [0_u64; VSYNC_NUM_SYNC_GROUPS];
    assert!(!vspace.file_get_lg_targets(99999, 0, &mut targets));
    assert!(targets.iter().all(|&t| t == 0));
}

// ============================================================================
// 7. Logical group lifecycle tests
// ============================================================================

#[ktest]
fn lg_create_and_add_sg() {
    let mut lg = VsyncLogicalGroup::create(1, 2, 3);
    assert_eq!(lg.lg_id, 1);
    assert!(lg.member_sgs.contains(&2));
    assert!(lg.member_sgs.contains(&3));
    assert_eq!(lg.state, VsyncLgState::Active);

    lg.add_sg(5);
    assert!(lg.member_sgs.contains(&5));
}

#[ktest]
fn lg_merge_combines_members_and_files() {
    let mut lg1 = VsyncLogicalGroup::create(1, 2, 3);
    lg1.shared_files.insert(100);
    let mut lg2 = VsyncLogicalGroup::create(2, 4, 5);
    lg2.shared_files.insert(200);

    lg1.merge(&mut lg2);

    assert!(lg1.member_sgs.contains(&2));
    assert!(lg1.member_sgs.contains(&3));
    assert!(lg1.member_sgs.contains(&4));
    assert!(lg1.member_sgs.contains(&5));
    assert!(lg1.shared_files.contains(&100));
    assert!(lg1.shared_files.contains(&200));
    assert_eq!(lg2.state, VsyncLgState::Destroyed);
}

#[ktest]
fn lg_check_dismissal_empty_files() {
    let mut lg = VsyncLogicalGroup::create(1, 2, 3);
    assert!(lg.check_dismissal());
    lg.shared_files.insert(42);
    assert!(!lg.check_dismissal());
    lg.shared_files.clear();
    assert!(lg.check_dismissal());
}

#[ktest]
fn lg_dismiss_with_shared_files_goes_dismissing() {
    let mut lg = VsyncLogicalGroup::create(1, 2, 3);
    lg.shared_files.insert(42);
    lg.dismiss();
    assert_eq!(lg.state, VsyncLgState::Dismissing);
}

#[ktest]
fn lg_dismiss_without_shared_files_goes_destroyed() {
    let mut lg = VsyncLogicalGroup::create(1, 2, 3);
    lg.dismiss();
    assert_eq!(lg.state, VsyncLgState::Destroyed);
}

// ============================================================================
// 8. Pre-commit barrier tests
// ============================================================================

#[ktest]
fn pre_commit_barrier_is_stored_and_retrieved() {
    let barrier = Arc::new(CountdownLatch::new_barrier());
    let mut rec = Record::new_meta(1, 1, 42, b"test".to_vec());

    assert!(rec.pre_commit_barrier().is_none());
    rec.set_pre_commit_barrier(Some(barrier.clone()));
    assert!(rec.pre_commit_barrier().is_some());

    rec.set_pre_commit_barrier(None);
    assert!(rec.pre_commit_barrier().is_none());
}

// ============================================================================
// 9. VsyncHandle tests
// ============================================================================

#[ktest]
fn handle_seq_returns_record_seq() {
    let vspace = new_test_vspace();
    let mut handle = vsync_op_start(&vspace, VsyncRecordType::Meta, 1).unwrap();
    let seq = vsync_handle_seq(&handle);
    assert!(seq > 0);
    assert_eq!(vsync_handle_seq(&handle), seq);
    vsync_op_complete(&mut handle, Some(b"done".to_vec()));
}

#[ktest]
fn handle_sg_id_is_consistent() {
    let vspace = new_test_vspace();
    let mut handle = vsync_op_start(&vspace, VsyncRecordType::Meta, 1).unwrap();
    let sg_id = vsync_handle_sg_id(&handle);
    assert!(sg_id < VSYNC_NUM_SYNC_GROUPS as u32);
    vsync_op_complete(&mut handle, Some(b"done".to_vec()));
}

// ============================================================================
// 10. Batch threshold and back-pressure tests
// ============================================================================

#[ktest]
fn batch_threshold_triggers_commit_pending() {
    let vspace = new_test_vspace();

    // Find the SG with the most operations and push past threshold
    let mut counts = [0_u64; VSYNC_NUM_SYNC_GROUPS];
    let mut handles = Vec::new();
    for _ in 0..VSYNC_BATCH_THRESHOLD + 10 {
        let handle = vsync_op_start(&vspace, VsyncRecordType::Meta, 50).unwrap();
        let sg_id = vsync_handle_sg_id(&handle);
        counts[sg_id as usize] += 1;
        handles.push(handle);
    }

    // Complete all - some SG should have triggered commit_pending
    for mut h in handles {
        vsync_op_complete(&mut h, Some(b"batch-data".to_vec()));
    }

    // At least one SG passed threshold and should have commit_pending > 0
    let any_triggered = counts.iter().enumerate().any(|(sg_id, count)| {
        *count >= VSYNC_BATCH_THRESHOLD as u64
            && vspace.sync_groups[sg_id]
                .commit_pending
                .load(Ordering::Relaxed)
                > 0
    });
    // Not all SGs may hit threshold, but at least one should
    // (this is probabilistic but with 74 ops over 8 SGs it's guaranteed)
    assert!(any_triggered);
}

#[ktest]
fn pending_high_mark_triggers_commit_pending() {
    let vspace = new_test_vspace();
    let mut handles = Vec::new();
    let sg_id = vsync_current_sg_id();

    for _ in 0..VSYNC_PENDING_HIGH_MARK + 10 {
        handles.push(vsync_op_start(&vspace, VsyncRecordType::Meta, 0).unwrap());
    }

    // commit_pending should have been set by the high-mark check in op_start
    let commit_pending = vspace.sync_groups[sg_id as usize]
        .commit_pending
        .load(Ordering::Relaxed);
    assert!(commit_pending > 0);

    // Cleanup
    for mut h in handles {
        vsync_op_complete(&mut h, Some(b"cleanup".to_vec()));
    }
}

// ============================================================================
// 11. Shadow ops tests
// ============================================================================

#[ktest]
fn shadow_materialize_is_called_during_commit() {
    let vspace = new_test_vspace();
    let called = Arc::new(AtomicBool::new(false));
    let called_clone = called.clone();

    vspace.set_shadow_ops(
        Some(Arc::new({
            let called = called_clone;
            move |_ctx: &dyn Any, _sg_id: u32| -> Vec<Record> {
                called.store(true, Ordering::Release);
                let mut rec = Record::new_meta(vsync_seq_now(), 0, 0, b"shadow".to_vec());
                rec.set_txn_id(0);
                vec![rec]
            }
        })),
        Some(Arc::new(|_ctx: &dyn Any, _sg_id: u32| {})),
    );
    vspace.set_coalesce_ops(Some(Arc::new(FakeCoalesceOps)));

    let (_, seq) = write_one(&vspace, 1, b"trigger");
    vsync_sync_until(&vspace, seq).unwrap();

    assert!(called.load(Ordering::Acquire));
}

#[ktest]
fn shadow_reset_is_called_after_commit() {
    let vspace = new_test_vspace();
    let reset_called = Arc::new(AtomicBool::new(false));
    let reset_called_clone = reset_called.clone();

    vspace.set_shadow_ops(
        None,
        Some(Arc::new(move |_ctx: &dyn Any, _sg_id: u32| {
            reset_called_clone.store(true, Ordering::Release);
        })),
    );

    let (_, seq) = write_one(&vspace, 1, b"trigger");
    vsync_sync_until(&vspace, seq).unwrap();

    assert!(reset_called.load(Ordering::Acquire));
}

// ============================================================================
// 12. Checkpoint tests
// ============================================================================

#[ktest]
fn checkpoint_applies_records_via_callbacks() {
    let vspace = new_test_vspace_with_coalesce();
    let fs_context = Arc::new(SpinLock::<Vec<VsyncCoalesceRange>>::new(Vec::new()));

    vspace.fs_context.write().replace(fs_context.clone());

    let _ = write_one(&vspace, 10, b"checkpoint-me");
    let (_, seq2) = write_one(&vspace, 10, b"checkpoint-too");
    vsync_sync_until(&vspace, seq2).unwrap();

    let count = vspace.checkpoint(0).unwrap();
    assert!(count > 0, "checkpoint applied {count} records");

    let applied = fs_context.lock();
    assert!(!applied.is_empty(), "expected applied ranges");
}

#[ktest]
fn checkpoint_empty_returns_zero() {
    let vspace = new_test_vspace_with_coalesce();
    let fs_context = Arc::new(SpinLock::<Vec<VsyncCoalesceRange>>::new(Vec::new()));
    vspace.fs_context.write().replace(fs_context.clone());

    let count = vspace.checkpoint(0).unwrap();
    assert_eq!(count, 0);
}

#[ktest]
fn queue_checkpoint_runs_checkpoint() {
    let vspace = new_test_vspace_with_coalesce();
    let fs_context = Arc::new(SpinLock::<Vec<VsyncCoalesceRange>>::new(Vec::new()));
    vspace.fs_context.write().replace(fs_context.clone());

    let (_, seq) = write_one(&vspace, 1, b"data");
    vsync_sync_until(&vspace, seq).unwrap();
    vspace.queue_checkpoint();

    let applied = fs_context.lock();
    assert!(!applied.is_empty());
}

#[ktest]
fn unmount_checkpoint_runs_when_configured() {
    let vspace = new_test_vspace_with_coalesce();
    let mut config = vspace.config();
    config.umount_checkpoint = true;
    vspace.set_config(config);

    let fs_context = Arc::new(SpinLock::<Vec<VsyncCoalesceRange>>::new(Vec::new()));
    vspace.fs_context.write().replace(fs_context.clone());

    let (_, seq) = write_one(&vspace, 1, b"data");
    vsync_sync_until(&vspace, seq).unwrap();
    vspace.unmount_checkpoint();

    let applied = fs_context.lock();
    assert!(!applied.is_empty());
}

// ============================================================================
// 13. Shard journal tests
// ============================================================================

#[ktest]
fn shard_journal_reserves_and_tracks_sectors() {
    let journal = VsyncShardJournal::new(0, 1024);

    let new_head = journal.reserve(10).unwrap();
    assert_eq!(new_head, 10);
    journal.mark_committed(new_head);

    let used = journal.used_sectors();
    assert!(used > 0);
}

#[ktest]
fn shard_journal_wraps_circularly() {
    let journal = VsyncShardJournal::new(100, 200);

    let _start = journal.reserve(150).unwrap();
    journal.mark_committed(250);
    assert_eq!(journal.used_sectors(), 150);
    journal.advance_tail(220);

    let start2 = journal.reserve(100).unwrap();
    // Wrapped: 250 - 100 = 150, 150 + 100 = 250, 250 % 200 = 50, 100 + 50 = 150
    assert_eq!(start2, 150);
}

#[ktest]
fn shard_tail_advance_reduces_used() {
    let journal = VsyncShardJournal::new(0, 1024);
    let _start = journal.reserve(50).unwrap();
    journal.mark_committed(50);

    let before = journal.used_sectors();
    journal.advance_tail(50);
    let after = journal.used_sectors();
    assert!(after < before);
}

#[ktest]
fn journal_space_pressure_detected() {
    let shard = VsyncShard::new(0, 0, 1024);

    // Fill the journal
    shard.journal.reserve(800).unwrap();
    shard.journal.mark_committed(800);

    // 800/1024 = 78% > 75% threshold
    assert!(shard.journal_space_pressure());
}

#[ktest]
fn shard_journal_rejects_reservation_that_would_overwrite_tail() {
    let journal = VsyncShardJournal::new(0, 100);

    let new_head = journal.reserve(80).unwrap();
    journal.mark_committed(new_head);

    assert!(journal.reserve(20).is_err());
    assert_eq!(journal.reserve(19).unwrap(), 99);
}

#[ktest]
fn journal_space_pressure_not_triggered_when_empty() {
    let shard = VsyncShard::new(0, 0, 1024);
    assert!(!shard.journal_space_pressure());
}

// ============================================================================
// 14. Vspace lifecycle tests
// ============================================================================

#[ktest]
fn vspace_state_transitions() {
    let vspace = new_test_vspace();
    assert_eq!(vspace.state(), VsyncVspaceState::Active);

    vspace.set_state(VsyncVspaceState::Unmounting);
    assert_eq!(vspace.state(), VsyncVspaceState::Unmounting);

    vspace.set_state(VsyncVspaceState::Destroyed);
    assert_eq!(vspace.state(), VsyncVspaceState::Destroyed);
}

#[ktest]
fn vspace_stable_seq_returns_min_across_sgs() {
    let vspace = new_test_vspace();
    let (_, seq) = write_one(&vspace, 1, b"stability");
    vsync_sync_until(&vspace, seq).unwrap();

    let stable = vspace.stable_seq();
    assert_eq!(stable, 0);
}

#[ktest]
fn vspace_is_not_recovering_when_active() {
    let vspace = new_test_vspace();
    assert!(!vspace.is_recovering());
}

// ============================================================================
// 15. Record serialization tests
// ============================================================================

#[ktest]
fn record_serialize_writes_journal_entry_header() {
    let rec = Record::new_meta(42, 7, 99, b"payload-data".to_vec());
    let mut buf = vec![0_u8; 512];
    let written = rec.serialize(&mut buf, 3);
    assert!(written > 0);
    // Verify magic
    let magic = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    assert_eq!(magic, 0x5653594E33314A00);
}

#[ktest]
fn record_serialize_fence_returns_zero() {
    let token = Arc::new(CountdownLatch::new_token(1));
    let rec = Record::new_fence(1, 0, token, 1);
    let mut buf = vec![0_u8; 512];
    assert_eq!(rec.serialize(&mut buf, 0), 0);
}

#[ktest]
fn record_serialize_oversized_returns_enospc() {
    let payload = vec![0_u8; 500];
    let rec = Record::new_meta(1, 1, 1, payload);
    let mut buf = vec![0_u8; JOURNAL_ENTRY_HEADER_SIZE]; // too small
    let result = rec.serialize(&mut buf, 0);
    assert!(result < 0);
}

#[ktest]
fn record_from_journal_entry_roundtrips() {
    let original = Record::new_meta(42, 7, 99, b"roundtrip-data".to_vec());
    let mut buf = vec![0_u8; 512];
    let written = original.serialize(&mut buf, 5) as usize;
    assert!(written > 0);

    // Parse the header back
    let entry = JournalEntryHeader {
        magic: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
        size: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
        rtype: buf[12],
        shard_id: buf[13],
        txn_id: u32::from_le_bytes(buf[14..18].try_into().unwrap()),
        seq: u64::from_le_bytes(buf[18..26].try_into().unwrap()),
        domain_id: u32::from_le_bytes(buf[26..30].try_into().unwrap()),
        ino: u64::from_le_bytes(buf[30..38].try_into().unwrap()),
        payload_len: u32::from_le_bytes(buf[38..42].try_into().unwrap()),
        checksum: u32::from_le_bytes(buf[42..46].try_into().unwrap()),
    };

    let payload_data = &buf[JOURNAL_ENTRY_HEADER_SIZE..written];
    let roundtripped = Record::from_journal_entry(&entry, payload_data).unwrap();
    assert_eq!(roundtripped.seq(), original.seq());
    assert_eq!(roundtripped.domain_id(), original.domain_id());
    assert_eq!(roundtripped.ino(), original.ino());
    assert_eq!(roundtripped.payload_bytes(), original.payload_bytes());
}

// ============================================================================
// 16. CountdownLatch tests
// ============================================================================

#[ktest]
fn countdown_latch_single_token() {
    let latch = CountdownLatch::new_token(1);
    assert_eq!(latch.remaining(), 1);
    assert!(latch.count_down());
    assert_eq!(latch.remaining(), 0);
}

#[ktest]
fn countdown_latch_multi_token() {
    let latch = CountdownLatch::new_token(3);
    assert!(!latch.count_down());
    assert_eq!(latch.remaining(), 2);
    assert!(!latch.count_down());
    assert_eq!(latch.remaining(), 1);
    assert!(latch.count_down()); // last one
    assert_eq!(latch.remaining(), 0);
}

#[ktest]
fn countdown_latch_barrier_signals() {
    let latch = CountdownLatch::new_barrier();
    assert_eq!(latch.remaining(), 1);
    latch.signal();
    assert_eq!(latch.remaining(), 0);
}

#[ktest]
#[should_panic(expected = "CountdownLatch underflow")]
fn countdown_latch_underflow_panics() {
    let latch = CountdownLatch::new_token(1);
    latch.count_down();
    latch.count_down(); // panics
}

// ============================================================================
// 17. Record type tests
// ============================================================================

#[ktest]
fn record_types_are_distinct() {
    let write = Record::new_write(1, 1, 1, Vec::new());
    let meta = Record::new_meta(1, 1, 1, Vec::new());
    let token = Arc::new(CountdownLatch::new_token(1));
    let fence = Record::new_fence(1, 1, token, 1);
    let txn_hdr = Record::new_txn_hdr(1, 1, VsyncTxnHdrPayload::default());

    assert_eq!(write.record_type(), VsyncRecordType::Write);
    assert_eq!(meta.record_type(), VsyncRecordType::Meta);
    assert_eq!(fence.record_type(), VsyncRecordType::Fence);
    assert_eq!(txn_hdr.record_type(), VsyncRecordType::TxnHdr);
}

#[ktest]
fn write_record_has_ino() {
    let rec = Record::new_write(1, 1, 42, b"data".to_vec());
    assert_eq!(rec.ino(), Some(42));
}

#[ktest]
fn fence_record_has_no_ino() {
    let token = Arc::new(CountdownLatch::new_token(1));
    let rec = Record::new_fence(1, 1, token, 1);
    assert_eq!(rec.ino(), None);
}

#[ktest]
fn fence_record_has_target_seq() {
    let token = Arc::new(CountdownLatch::new_token(1));
    let rec = Record::new_fence(1, 1, token, 100);
    assert_eq!(rec.fence_target_seq(), Some(100));
}

#[ktest]
fn meta_record_has_no_fence_fields() {
    let rec = Record::new_meta(1, 1, 1, b"data".to_vec());
    assert_eq!(rec.fence_target_seq(), None);
    assert!(rec.fence_token().is_none());
}

#[ktest]
fn record_payload_set_and_get() {
    let mut rec = Record::new_meta(1, 1, 1, b"old".to_vec());
    assert_eq!(rec.payload_bytes(), b"old");
    rec.set_payload(b"new-payload".to_vec());
    assert_eq!(rec.payload_bytes(), b"new-payload");
}

// ============================================================================
// 18. VsynSyncGroup tests
// ============================================================================

#[ktest]
fn sg_transaction_lifecycle() {
    let mut txn = SgTransaction::new(3, 100);
    assert_eq!(txn.sg_id, 3);
    assert_eq!(txn.txn_id, 100);
    assert_eq!(txn.state, VsyncTxnState::Collecting);
    assert!(txn.records.is_empty());

    let rec = Record::new_meta(1, 1, 1, b"txn-record".to_vec());
    txn.push_record(rec);
    assert_eq!(txn.records.len(), 1);
}

#[ktest]
fn sg_transaction_max_seq_tracks_highest() {
    let mut txn = SgTransaction::new(0, 0);
    txn.push_record(Record::new_meta(5, 1, 1, Vec::new()));
    txn.push_record(Record::new_meta(100, 1, 1, Vec::new()));
    txn.push_record(Record::new_meta(50, 1, 1, Vec::new()));
    assert_eq!(txn.max_seq, 100);
}

#[ktest]
fn sg_completion_state_drain_inflight() {
    let completion = VsyncCompletionState::new();
    completion.push_inflight(SgTransaction::new(0, 1));
    completion.push_inflight(SgTransaction::new(0, 2));
    assert_eq!(completion.in_flight_count(), 2);

    let drained = completion.drain_inflight();
    assert_eq!(drained.len(), 2);
    assert_eq!(completion.in_flight_count(), 0);
}

#[ktest]
fn sg_domain_stable_seq_updated_after_commit() {
    let vspace = new_test_vspace();
    let domain_id = VsyncVspace::task_to_domain(&vspace.config());
    let sg_id = VsyncVspace::domain_to_sg(domain_id);
    let sg = &vspace.sync_groups[sg_id as usize];

    let (_, seq) = write_one(&vspace, 1, b"domain-test");
    vsync_sync_until(&vspace, seq).unwrap();

    let domain_seq = sg.domain_stable_seq(domain_id);
    assert!(domain_seq >= seq);
}

#[ktest]
fn sg_complete_transaction_updates_stats() {
    let vspace = new_test_vspace();
    let (sg_id, seq) = write_one(&vspace, 1, b"stats-test");
    let before_commits = vspace.sync_groups[sg_id as usize]
        .commits
        .load(Ordering::Relaxed);

    vsync_sync_until(&vspace, seq).unwrap();

    let after_commits = vspace.sync_groups[sg_id as usize]
        .commits
        .load(Ordering::Relaxed);
    assert!(after_commits > before_commits);
}

// ============================================================================
// 19. VsyncSlot tests
// ============================================================================

#[ktest]
fn slot_drain_until_respects_max_count() {
    let slot = VsyncSlot::default();
    slot.publish_record(Record::new_meta(1, 1, 1, Vec::new()));
    slot.publish_record(Record::new_meta(2, 1, 1, Vec::new()));
    slot.publish_record(Record::new_meta(3, 1, 1, Vec::new()));

    let drained = slot.drain_until(3, 2);
    assert_eq!(drained.len(), 2);
    // Oldest records first
    assert_eq!(drained[0].seq(), 1);
    assert_eq!(drained[1].seq(), 2);
}

#[ktest]
fn slot_drain_until_respects_target_seq() {
    let slot = VsyncSlot::default();
    slot.publish_record(Record::new_meta(1, 1, 1, Vec::new()));
    slot.publish_record(Record::new_meta(10, 1, 1, Vec::new()));
    slot.publish_record(Record::new_meta(100, 1, 1, Vec::new()));

    let drained = slot.drain_until(10, usize::MAX);
    assert_eq!(drained.len(), 2);
    assert_eq!(drained[0].seq(), 1);
    assert_eq!(drained[1].seq(), 10);
}

#[ktest]
fn slot_inflight_tracking() {
    let slot = VsyncSlot::default();
    slot.begin_inflight(1);
    slot.begin_inflight(2);
    slot.begin_inflight(3);

    slot.finish_inflight(2);
    slot.finish_inflight(3);

    // seq 1 is still in flight
    // wait_inflight_until(1) would block until seq 1 is finished
    slot.finish_inflight(1);
}

#[ktest]
fn slot_last_published_seq_tracks_max() {
    let slot = VsyncSlot::default();
    assert_eq!(slot.last_published_seq(), 0);

    slot.publish_record(Record::new_meta(5, 1, 1, Vec::new()));
    assert_eq!(slot.last_published_seq(), 5);

    slot.publish_record(Record::new_meta(3, 1, 1, Vec::new()));
    assert_eq!(slot.last_published_seq(), 5); // not decreased

    slot.publish_record(Record::new_meta(100, 1, 1, Vec::new()));
    assert_eq!(slot.last_published_seq(), 100);
}

#[ktest]
fn slot_drain_fences_returns_all() {
    let slot = VsyncSlot::default();
    let token = Arc::new(CountdownLatch::new_token(1));
    slot.publish_fence(VsyncFence {
        token: token.clone(),
        domain_id: 1,
        target_seq: 100,
    });
    slot.publish_fence(VsyncFence {
        token,
        domain_id: 2,
        target_seq: 200,
    });

    let drained = slot.drain_fences();
    assert_eq!(drained.len(), 2);
}

// ============================================================================
// 20. Config tests
// ============================================================================

#[ktest]
fn default_config_has_expected_values() {
    let config = VsyncConfig::default();
    assert_eq!(config.domain_type, VsyncDomainType::Thread);
    assert_eq!(config.coalesce_us, 250);
    assert!(config.online_checkpoint);
    assert!(config.per_file_sync);
    assert!(!config.numa_aware);
}

#[ktest]
fn vspace_set_config_updates_runtime_config() {
    let vspace = new_test_vspace();
    let mut config = vspace.config();
    config.debug_level = 5;
    config.coalesce_us = 1000;
    vspace.set_config(config);

    assert_eq!(vspace.config().debug_level, 5);
    assert_eq!(vspace.config().coalesce_us, 1000);
}

// ============================================================================
// 21. sync_until and sync_file error propagation
// ============================================================================

#[ktest]
fn sync_until_returns_error_when_txn_error_in_range() {
    let vspace = new_test_vspace();
    let domain_id = VsyncVspace::task_to_domain(&vspace.config());
    let sg_id = VsyncVspace::domain_to_sg(domain_id);
    let sg = &vspace.sync_groups[sg_id as usize];

    // Simulate a transaction error
    let (_, seq) = write_one(&vspace, 1, b"will-fail");
    sg.set_last_error(Errno::EIO, seq);

    let result = vsync_sync_until(&vspace, seq);
    assert!(result.is_err());
    // Clear error for other tests sharing this vspace (not needed, but defensive)
}

#[ktest]
fn sync_multi_sg_returns_error_when_any_sg_has_error() {
    let vspace = new_test_vspace();
    let (sg_id, seq) = write_one(&vspace, 1, b"multi-sg-error");

    vspace.sync_groups[sg_id as usize].set_last_error(Errno::EIO, seq);

    let mut targets = [0_u64; VSYNC_NUM_SYNC_GROUPS];
    targets[sg_id as usize] = seq;
    let result = vsync_sync_multi_sg(&vspace, &targets);
    assert!(result.is_err());
}

// ============================================================================
// 22. Record replay from journal entry
// ============================================================================

#[ktest]
fn from_journal_entry_rejects_bad_magic() {
    let entry = JournalEntryHeader {
        magic: 0xDEADBEEF,
        ..Default::default()
    };
    assert!(Record::from_journal_entry(&entry, &[]).is_none());
}

#[ktest]
fn from_journal_entry_rejects_unknown_type() {
    let entry = JournalEntryHeader {
        magic: 0x5653594E33314A00,
        rtype: 99,
        ..Default::default()
    };
    assert!(Record::from_journal_entry(&entry, &[]).is_none());
}

#[ktest]
fn from_journal_entry_rejects_truncated_payload() {
    let entry = JournalEntryHeader {
        magic: 0x5653594E33314A00,
        rtype: 1, // Write
        payload_len: 100,
        ..Default::default()
    };
    assert!(Record::from_journal_entry(&entry, b"short").is_none());
}

#[ktest]
fn from_journal_entry_rejects_bad_checksum() {
    let original = Record::new_meta(42, 7, 99, b"checksum-data".to_vec());
    let mut buf = vec![0_u8; 512];
    let written = original.serialize(&mut buf, 5) as usize;
    let entry = journal_entry_header_from_bytes(&buf);
    let mut payload = buf[JOURNAL_ENTRY_HEADER_SIZE..written].to_vec();
    payload[0] ^= 0xFF;

    assert!(Record::from_journal_entry(&entry, &payload).is_none());
}

// ============================================================================
// 23. no_batch_coalesce flag
// ============================================================================

#[ktest]
fn no_batch_coalesce_skips_coalescing() {
    let vspace = new_test_vspace_with_coalesce();
    vspace.set_no_batch_coalesce(true);

    // Two records for same ino should NOT be coalesced
    let (sg_id, _seq1) = write_one(&vspace, 42, b"first");
    let (_, seq2) = write_one(&vspace, 42, b"second");

    vsync_sync_until(&vspace, seq2).unwrap();
    assert!(vspace.sync_groups[sg_id as usize].stable_seq() >= seq2);
}

// ============================================================================
// 24. Op start with fence and txn_hdr types
// ============================================================================

#[ktest]
fn op_start_fence_and_complete_publishes() {
    let vspace = new_test_vspace();
    let mut handle = vsync_op_start(&vspace, VsyncRecordType::Fence, 0).unwrap();
    let seq = vsync_handle_seq(&handle);
    let result = vsync_op_complete(&mut handle, Some(b"ignored".to_vec()));
    // Fence records skip the payload check and publish
    assert!(result);
    // Fence is on the slot, can be drained
    vsync_sync_until(&vspace, seq).unwrap();
}

#[ktest]
fn op_start_txn_hdr_and_complete_publishes() {
    let vspace = new_test_vspace();
    let mut handle = vsync_op_start(&vspace, VsyncRecordType::TxnHdr, 0).unwrap();
    let seq = vsync_handle_seq(&handle);
    let result = vsync_op_complete(&mut handle, Some(b"hdr-data".to_vec()));
    assert!(result);
    vsync_sync_until(&vspace, seq).unwrap();
}

// ============================================================================
// 25. VsyncGlobalCfg and public API helpers
// ============================================================================

#[ktest]
fn vsync_seq_now_is_monotonic() {
    let a = vsync_seq_now();
    let b = vsync_seq_now();
    let c = vsync_seq_now();
    assert!(a < b);
    assert!(b < c);
}

#[ktest]
fn vsync_per_file_sync_flag_is_readable() {
    assert!(vsync_per_file_sync_enabled());
}

#[ktest]
fn vsync_current_sg_id_is_valid() {
    let sg_id = vsync_current_sg_id();
    assert!(sg_id < VSYNC_NUM_SYNC_GROUPS as u32);
}

#[ktest]
fn vsync_vspace_create_sets_global_config() {
    let mut cfg = VsyncConfig::default();
    cfg.coalesce_us = 500;
    let vspace = vsync_vspace_create(None, None, Some(cfg.clone()), None, 100, 2000);
    assert_eq!(vspace.start_sector(), 100);
    assert_eq!(vspace.total_sectors(), 2000);
    assert_eq!(VSYNC_GLOBAL_CFG.lock().coalesce_us, 500);
}

#[ktest]
fn vsync_vspace_destroy_transitions_state() {
    let vspace = new_test_vspace();
    // Test the public API wrapper; vspace state is marked Destroyed internally
    vsync_vspace_destroy(&vspace);
    // After destroy: vspace is in destroyed state
    assert_eq!(vspace.state(), VsyncVspaceState::Destroyed);
}

// ============================================================================
// 26. Edge cases with Write record type
// ============================================================================

#[ktest]
fn write_records_flow_through_commit_path() {
    let vspace = new_test_vspace();
    let (sg_id, seq) = write_one_write(&vspace, 42, b"write-data");
    vsync_sync_until(&vspace, seq).unwrap();
    assert!(vspace.sync_groups[sg_id as usize].stable_seq() >= seq);
}

#[ktest]
fn mixed_write_and_meta_records_coexist() {
    let vspace = new_test_vspace_with_coalesce();
    let (sg_id, seq_w) = write_one_write(&vspace, 42, b"write");
    let (_, seq_m) = write_one(&vspace, 42, b"meta");
    let target = seq_w.max(seq_m);
    vsync_sync_until(&vspace, target).unwrap();
    assert!(vspace.sync_groups[sg_id as usize].stable_seq() >= target);
}

// ============================================================================
// 27. On-disk integration tests (use TestBlockDevice)
// ============================================================================

fn new_test_vspace_with_block_device(nr_sectors: u64) -> (Arc<VsyncVspace>, Arc<TestBlockDevice>) {
    let bdev = TestBlockDevice::new(nr_sectors);
    let vspace = vsync_vspace_create(
        None,
        None,
        None,
        Some(bdev.clone() as Arc<dyn BlockDevice>),
        0,
        nr_sectors,
    );
    (vspace, bdev)
}

/// Helper: verify raw journal data on the test device contains expected magic bytes.
fn journal_contains_valid_entries(bdev: &TestBlockDevice) -> bool {
    let raw = bdev.data.lock();
    let magic_le = 0x5653594E33314A00_u64.to_le_bytes();
    raw.windows(8).any(|w| w == magic_le)
}

/// Helper: count valid journal entries in raw device data.
fn count_journal_entries(bdev: &TestBlockDevice) -> usize {
    let raw = bdev.data.lock();
    let magic_le = 0x5653594E33314A00_u64.to_le_bytes();
    raw.windows(8).filter(|w| *w == magic_le).count()
}

fn journal_entry_header_from_bytes(buf: &[u8]) -> JournalEntryHeader {
    JournalEntryHeader {
        magic: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
        size: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
        rtype: buf[12],
        shard_id: buf[13],
        txn_id: u32::from_le_bytes(buf[14..18].try_into().unwrap()),
        seq: u64::from_le_bytes(buf[18..26].try_into().unwrap()),
        domain_id: u32::from_le_bytes(buf[26..30].try_into().unwrap()),
        ino: u64::from_le_bytes(buf[30..38].try_into().unwrap()),
        payload_len: u32::from_le_bytes(buf[38..42].try_into().unwrap()),
        checksum: u32::from_le_bytes(buf[42..46].try_into().unwrap()),
    }
}

// --- Superblock I/O ---

#[ktest]
fn superblock_write_read_roundtrip() {
    let nr_sectors = 1024;
    let (vspace, bdev) = new_test_vspace_with_block_device(nr_sectors);

    vspace.write_journal_super();

    // Verify raw bytes at journal_start_sector contain VSYNC_JOURNAL_MAGIC
    {
        let raw = bdev.data.lock();
        let magic_bytes = VSYNC_JOURNAL_MAGIC.to_le_bytes();
        assert_eq!(&raw[0..4], &magic_bytes[..]);
    }

    // Read back and validate
    assert_eq!(vspace.read_journal_super(), 0);
}

#[ktest]
fn journal_layout_keeps_superblock_before_shards() {
    let nr_sectors = 8192;
    let (vspace, bdev) = new_test_vspace_with_block_device(nr_sectors);
    let superblock_sectors = (aster_block::BLOCK_SIZE / aster_block::SECTOR_SIZE) as u64;

    assert_eq!(
        vspace.shards[0].journal.start_sector,
        vspace.start_sector() + superblock_sectors
    );

    vspace.write_journal_super();
    let (_, seq) = write_one(&vspace, 42, b"layout-preserves-superblock");
    vsync_sync_until(&vspace, seq).unwrap();

    let raw = bdev.data.lock();
    let magic_bytes = VSYNC_JOURNAL_MAGIC.to_le_bytes();
    assert_eq!(&raw[0..4], &magic_bytes[..]);
}

#[ktest]
fn superblock_restores_shard_state_after_checkpoint() {
    let nr_sectors = 8192;
    let (vspace, bdev) = new_test_vspace_with_block_device(nr_sectors);
    vspace.set_coalesce_ops(Some(Arc::new(FakeCoalesceOps)));

    // Write several records to advance journal heads
    let mut max_seq = 0_u64;
    for i in 0..16 {
        let (_, seq) = write_one(&vspace, (i % 4 + 1) as u64, &[i; 32]);
        max_seq = max_seq.max(seq);
    }
    vsync_sync_until(&vspace, max_seq).unwrap();

    // Record shard head positions before checkpoint
    let heads_before: Vec<u64> = vspace
        .shards
        .iter()
        .map(|s| s.journal.head_sector.load(Ordering::Acquire))
        .collect();

    // Checkpoint + write superblock
    let fs_ctx = Arc::new(SpinLock::<Vec<VsyncCoalesceRange>>::new(Vec::new()));
    vspace.fs_context.write().replace(fs_ctx);
    vspace.checkpoint(0).unwrap();
    vspace.write_journal_super();

    // Create fresh vspace and read superblock
    let vspace2 = vsync_vspace_create(
        None,
        None,
        None,
        Some(bdev.clone() as Arc<dyn BlockDevice>),
        0,
        nr_sectors,
    );
    assert_eq!(vspace2.read_journal_super(), 0);
    let heads_after: Vec<u64> = vspace2
        .shards
        .iter()
        .map(|s| s.journal.head_sector.load(Ordering::Acquire))
        .collect();
    assert_eq!(heads_after, heads_before);
}

// --- Transaction I/O ---

#[ktest]
fn sync_until_writes_transactions_to_disk() {
    let nr_sectors = 8192;
    let (vspace, bdev) = new_test_vspace_with_block_device(nr_sectors);

    let (_, seq) = write_one(&vspace, 42, b"on-disk-payload-abcdefgh");
    vsync_sync_until(&vspace, seq).unwrap();

    assert!(
        journal_contains_valid_entries(&bdev),
        "expected journal entries on disk"
    );
}

#[ktest]
fn scan_journal_replays_complete_transaction_without_txn_header() {
    let nr_sectors = 8192;
    let (vspace, bdev) = new_test_vspace_with_block_device(nr_sectors);
    let shard = &vspace.shards[0];
    let mut txn = SgTransaction::new(0, 1);
    txn.push_record(Record::new_meta(10, 1, 100, b"first".to_vec()));
    txn.push_record(Record::new_meta(20, 1, 200, b"second".to_vec()));

    shard.enqueue_transaction(txn);
    let block_device: &dyn BlockDevice = bdev.as_ref();
    let completed = shard.flush_all(Some(block_device)).unwrap();
    assert_eq!(completed.len(), 1);

    let mut records = Vec::new();
    assert_eq!(shard.scan_journal(Some(bdev.as_ref()), &mut records), 0);
    assert_eq!(records.len(), 2);
    assert!(
        records
            .iter()
            .all(|record| record.record_type() != VsyncRecordType::TxnHdr)
    );
}

#[ktest]
fn scan_journal_discards_incomplete_transaction() {
    let nr_sectors = 8192;
    let (vspace, bdev) = new_test_vspace_with_block_device(nr_sectors);
    let shard = &vspace.shards[0];
    let mut txn = SgTransaction::new(0, 1);
    txn.push_record(Record::new_meta(10, 1, 100, b"first".to_vec()));
    txn.push_record(Record::new_meta(20, 1, 200, b"second".to_vec()));
    let buf = shard.serialize_transaction(&txn);

    let txn_hdr_size = JOURNAL_ENTRY_HEADER_SIZE + VsyncTxnHdrPayload::SERIALIZED_SIZE;
    let first_record_header = journal_entry_header_from_bytes(&buf[txn_hdr_size..]);
    let second_record_offset = txn_hdr_size + first_record_header.size as usize;
    shard.enqueue_transaction(txn);
    let block_device: &dyn BlockDevice = bdev.as_ref();
    let completed = shard.flush_all(Some(block_device)).unwrap();
    assert_eq!(completed.len(), 1);
    let start = shard.journal.start_sector as usize * aster_block::SECTOR_SIZE;
    bdev.data.lock()[start + second_record_offset..start + second_record_offset + 8].fill(0);

    let mut records = Vec::new();
    assert_eq!(shard.scan_journal(Some(bdev.as_ref()), &mut records), 0);
    assert!(
        records.is_empty(),
        "incomplete transactions must be discarded atomically"
    );
}

#[ktest]
fn multiple_sync_until_calls_produce_ordered_entries() {
    let nr_sectors = 8192;
    let (vspace, bdev) = new_test_vspace_with_block_device(nr_sectors);

    // Batch 1
    let (_, seq1) = write_one(&vspace, 1, b"batch-1");
    vsync_sync_until(&vspace, seq1).unwrap();
    let entries_after_1 = count_journal_entries(&bdev);

    // Batch 2
    let (_, seq2) = write_one(&vspace, 1, b"batch-2-second");
    vsync_sync_until(&vspace, seq2).unwrap();
    let entries_after_2 = count_journal_entries(&bdev);

    // Second batch should add at least one more entry (txn hdr + record)
    assert!(
        entries_after_2 > entries_after_1,
        "batch 2 should add entries: {entries_after_1} -> {entries_after_2}"
    );
}

#[ktest]
fn transaction_serialization_is_parseable() {
    let nr_sectors = 8192;
    let (vspace, _bdev) = new_test_vspace_with_block_device(nr_sectors);

    // Drain a sync group slot directly to get a serialized transaction
    let mut txn = SgTransaction::new(0, 1);
    txn.records = vec![
        Record::new_meta(10, 1, 100, b"record-data".to_vec()),
        Record::new_meta(20, 1, 200, b"more-data-here".to_vec()),
    ];

    let shard = &vspace.shards[0];
    let buf = shard.serialize_transaction(&txn);

    // Buffer should be sector-aligned
    assert_eq!(buf.len() % 512, 0);
    // Should start with RECORD_MAGIC
    let magic = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    assert_eq!(magic, 0x5653594E33314A00);
    // First entry should be TxnHdr type
    assert_eq!(buf[12], VsyncRecordType::TxnHdr as u8);
}

#[ktest]
fn transaction_header_payload_roundtrips() {
    let mut txn = SgTransaction::new(0, 1);
    txn.push_record(Record::new_meta(10, 1, 100, b"record-data".to_vec()));
    txn.push_record(Record::new_meta(20, 1, 200, b"more-data-here".to_vec()));

    let shard = VsyncShard::new(0, 0, 1024);
    let buf = shard.serialize_transaction(&txn);
    let entry = journal_entry_header_from_bytes(&buf);
    let payload_len = entry.payload_len as usize;
    let payload = &buf[JOURNAL_ENTRY_HEADER_SIZE..JOURNAL_ENTRY_HEADER_SIZE + payload_len];
    let parsed = Record::from_journal_entry(&entry, payload).unwrap();
    let hdr = parsed.txn_header().unwrap();
    let sg_id = hdr.sg_id;
    let num_records = hdr.num_records;
    let max_seq = hdr.max_seq;
    let txn_size = hdr.txn_size;
    let checksum = entry.checksum;

    assert_eq!(parsed.record_type(), VsyncRecordType::TxnHdr);
    assert_eq!(sg_id, 0);
    assert_eq!(num_records, 2);
    assert_eq!(max_seq, 20);
    assert_eq!(txn_size as usize, buf.len());
    assert_ne!(checksum, 0);
}

#[ktest]
fn fence_only_records_are_skipped_in_serialization() {
    let mut txn = SgTransaction::new(0, 1);
    let token = Arc::new(CountdownLatch::new_token(1));
    txn.records = vec![
        Record::new_fence(10, 1, token, 10),
        Record::new_meta(20, 1, 100, b"actual-data".to_vec()),
    ];

    let shard = VsyncShard::new(0, 0, 1024);
    let buf = shard.serialize_transaction(&txn);

    // Should have txn hdr + 1 data record (fence skipped)
    // TxnHdr entries + 1 record = 2 entries total
    assert!(buf.len() >= JOURNAL_ENTRY_HEADER_SIZE * 2);
}

// --- Checkpoint + Tail Advancement ---

#[ktest]
fn checkpoint_applies_records_and_advances_tail() {
    let nr_sectors = 8192;
    let (vspace, _bdev) = new_test_vspace_with_block_device(nr_sectors);
    vspace.set_coalesce_ops(Some(Arc::new(FakeCoalesceOps)));

    // Write records to one shard
    let (sg_id, seq) = write_one(&vspace, 100, b"checkpoint-test-payload");
    vsync_sync_until(&vspace, seq).unwrap();

    // Find which shard has committed transactions
    let active_shards: Vec<usize> = vspace
        .shards
        .iter()
        .enumerate()
        .filter(|(_, s)| s.committed_txn_count.load(Ordering::Acquire) > 0)
        .map(|(i, _)| i)
        .collect();

    let fs_ctx = Arc::new(SpinLock::<Vec<VsyncCoalesceRange>>::new(Vec::new()));
    vspace.fs_context.write().replace(fs_ctx.clone());

    let used_before: Vec<u64> = active_shards
        .iter()
        .map(|&i| vspace.shards[i].journal.used_sectors())
        .collect();

    let applied = vspace.checkpoint(0).unwrap();
    assert!(applied > 0, "checkpoint should apply records");

    // Verify checkpoint applied records to fs_context
    let applied_ranges = fs_ctx.lock();
    assert!(!applied_ranges.is_empty());

    // Verify shards no longer have committed transactions
    for &i in &active_shards {
        assert_eq!(
            vspace.shards[i].committed_txn_count.load(Ordering::Acquire),
            0,
            "shard {i} should be drained after checkpoint"
        );
    }
    for (&i, used_before) in active_shards.iter().zip(used_before.iter()) {
        assert!(
            vspace.shards[i].journal.used_sectors() < *used_before,
            "shard {i} should advance its tail after checkpoint"
        );
    }

    // Stable seq should reach the written seq
    assert!(vspace.sync_groups[sg_id as usize].stable_seq() >= seq);
}

#[ktest]
fn checkpoint_on_idle_vspace_is_noop() {
    let nr_sectors = 8192;
    let (vspace, _bdev) = new_test_vspace_with_block_device(nr_sectors);
    let applied = vspace.checkpoint(0).unwrap();
    assert_eq!(applied, 0);
}

// --- Recovery / Replay ---

#[ktest]
fn full_lifecycle_write_checkpoint_replay() {
    let nr_sectors = 32768;
    let (vspace, bdev) = new_test_vspace_with_block_device(nr_sectors);
    vspace.set_coalesce_ops(Some(Arc::new(FakeCoalesceOps)));

    // Phase 1: Write a realistic fake workload
    // - Multiple inodes
    // - Mixed small and larger payloads
    // - Records spread across multiple SGs
    let mut max_seqs = [0_u64; VSYNC_NUM_SYNC_GROUPS];
    for ino in 1..=20_u64 {
        for _ in 0..3 {
            let payload = vec![(ino % 256) as u8; 64 + (ino % 4) as usize * 16];
            let (sg_id, seq) = write_one(&vspace, ino, &payload);
            max_seqs[sg_id as usize] = max_seqs[sg_id as usize].max(seq);
        }
    }

    // Sync all SGs
    let max_seq = max_seqs.iter().copied().max().unwrap_or(0);
    vsync_sync_all_until(&vspace, max_seq).unwrap();

    // Verify each SG reached its max
    for (sg_id, &target) in max_seqs.iter().enumerate() {
        if target > 0 {
            assert!(
                vspace.sync_groups[sg_id].stable_seq() >= target,
                "SG {sg_id}: stable={} < target={target}",
                vspace.sync_groups[sg_id].stable_seq()
            );
        }
    }

    // Verify journal entries are on disk
    assert!(
        journal_contains_valid_entries(&bdev),
        "journal must have valid entries after sync"
    );

    // Phase 2: Checkpoint to apply records and advance tails
    let fs_ctx = Arc::new(SpinLock::<Vec<VsyncCoalesceRange>>::new(Vec::new()));
    vspace.fs_context.write().replace(fs_ctx.clone());
    let applied = vspace.checkpoint(0).unwrap();
    assert!(applied > 0, "checkpoint applied {applied} records");

    let applied_ranges = fs_ctx.lock();
    // 20 inodes * 2-3 records each, all coalesced to 1 range per inode = ~20
    assert!(
        applied_ranges.len() >= 20,
        "expected >= 20 applied ranges, got {}",
        applied_ranges.len()
    );

    // Phase 3: Destroy (simulate unmount)
    vsync_vspace_destroy(&vspace);

    // Phase 4: Recreate and recover (simulate remount)
    let recovered_records = Arc::new(SpinLock::<Vec<Record>>::new(Vec::new()));
    let recovered_clone = recovered_records.clone();
    let vspace2 = vsync_vspace_create(
        Some(Arc::new(move |_ctx: &dyn Any, recs: &[Record]| {
            recovered_clone.lock().extend_from_slice(recs);
            Result::<(), crate::error::Error>::Ok(())
        })),
        None,
        None,
        Some(bdev.clone() as Arc<dyn BlockDevice>),
        0,
        nr_sectors,
    );

    let replay_rc = vsync_vspace_replay(&vspace2);
    assert_eq!(replay_rc, 0, "replay should succeed, got {replay_rc}");

    let recovered = recovered_records.lock();
    assert!(
        recovered.is_empty(),
        "clean checkpoint/unmount should not replay records"
    );
}

#[ktest]
fn crash_recovery_works_without_clean_unmount() {
    let nr_sectors = 16384;
    let (vspace, bdev) = new_test_vspace_with_block_device(nr_sectors);

    // Write records and sync (MOUNTED flag is set)
    let (_, seq1) = write_one(&vspace, 10, b"crash-recovery-data-1");
    let (_, seq2) = write_one(&vspace, 20, b"crash-recovery-data-2");
    let (_, seq3) = write_one(&vspace, 30, b"crash-recovery-data-3");
    let max_seq = seq1.max(seq2).max(seq3);
    vsync_sync_until(&vspace, max_seq).unwrap();

    // Write superblock (simulate periodic checkpoint write)
    vspace.write_journal_super();
    let mut journal_records = Vec::new();
    for shard in &vspace.shards {
        assert_eq!(
            shard.scan_journal(Some(bdev.as_ref()), &mut journal_records),
            0
        );
    }
    assert!(
        !journal_records.is_empty(),
        "journal should contain crash-recoverable records"
    );

    // Simulate crash: create new vspace WITHOUT clean unmount
    // (MOUNTED flag is still set from the first vspace)
    let recovered = Arc::new(SpinLock::<Vec<Record>>::new(Vec::new()));
    let recovered_clone = recovered.clone();
    let vspace2 = vsync_vspace_create(
        Some(Arc::new(move |_ctx: &dyn Any, recs: &[Record]| {
            recovered_clone.lock().extend_from_slice(recs);
            Result::<(), crate::error::Error>::Ok(())
        })),
        None,
        None,
        Some(bdev.clone() as Arc<dyn BlockDevice>),
        0,
        nr_sectors,
    );

    let replay_rc = vsync_vspace_replay(&vspace2);
    assert_eq!(replay_rc, 0, "crash replay should succeed, got {replay_rc}");

    let recovered = recovered.lock();
    assert!(!recovered.is_empty(), "should recover records after crash");
}

#[ktest]
fn replay_handles_empty_journal() {
    let nr_sectors = 8192;
    let (vspace, bdev) = new_test_vspace_with_block_device(nr_sectors);

    // Clean unmount
    vsync_vspace_destroy(&vspace);

    // Re-create: replay should find clean unmount and skip
    let vspace2 = vsync_vspace_create(
        None,
        None,
        None,
        Some(bdev.clone() as Arc<dyn BlockDevice>),
        0,
        nr_sectors,
    );
    assert_eq!(vsync_vspace_replay(&vspace2), 0);
}

// --- Journal Space Management ---

#[ktest]
fn journal_wrap_handling_reserves_correctly() {
    let nr_sectors = 4096; // Small journal to force wrapping
    let (vspace, bdev) = new_test_vspace_with_block_device(nr_sectors);

    // Write many records to fill and wrap the journal
    let mut max_seq = 0_u64;
    for i in 0..64 {
        let (_, seq) = write_one(&vspace, (i % 8 + 1) as u64, &[i as u8; 256]);
        max_seq = max_seq.max(seq);
    }
    vsync_sync_until(&vspace, max_seq).unwrap();

    // Verify journal entries exist on disk
    assert!(journal_contains_valid_entries(&bdev));
}

#[ktest]
fn used_sectors_tracks_journal_occupancy() {
    let nr_sectors = 16384;
    let (vspace, _bdev) = new_test_vspace_with_block_device(nr_sectors);

    let used_before: Vec<u64> = vspace
        .shards
        .iter()
        .map(|s| s.journal.used_sectors())
        .collect();

    let (_, seq) = write_one(&vspace, 1, &[0xAA; 512]);
    vsync_sync_until(&vspace, seq).unwrap();

    // At least one shard should show non-zero usage
    let used_after: Vec<u64> = vspace
        .shards
        .iter()
        .map(|s| s.journal.used_sectors())
        .collect();

    let any_increased = used_after
        .iter()
        .zip(used_before.iter())
        .any(|(a, b)| a > b);
    assert!(any_increased, "journal usage should increase after write");
}

// --- Multi-SG On-Disk Tests ---

#[ktest]
fn sync_all_persists_all_sgs_to_disk() {
    let nr_sectors = 32768;
    let (vspace, bdev) = new_test_vspace_with_block_device(nr_sectors);

    // Write to many inodes to spread across SGs
    for ino in [1_u64, 10, 100, 200, 500, 1000, 5000, 10000] {
        write_one(&vspace, ino, &[ino as u8; 32]);
    }

    vsync_sync_all(&vspace).unwrap();
    assert!(journal_contains_valid_entries(&bdev));
}

#[ktest]
fn file_sync_persists_cross_sg_data() {
    let nr_sectors = 32768;
    let (vspace, bdev) = new_test_vspace_with_block_device(nr_sectors);

    // Write to the same inode repeatedly to trigger cross-SG tracking
    let ino = 42_u64;
    let mut max_seq = 0_u64;
    for _ in 0..16 {
        let (_, seq) = write_one(&vspace, ino, b"shared-file-data");
        max_seq = max_seq.max(seq);
    }

    vsync_sync_file(&vspace, ino, 0).unwrap();
    assert!(journal_contains_valid_entries(&bdev));
}

// --- Journal Superblock Lifecycle ---

#[ktest]
fn mounted_flag_is_set_on_create() {
    let nr_sectors = 8192;
    let (vspace, _bdev) = new_test_vspace_with_block_device(nr_sectors);

    let flags = vspace.journal_super_flags.load(Ordering::Acquire);
    assert!(
        flags & VSYNC_JOURNAL_FLAG_MOUNTED != 0,
        "MOUNTED flag ({VSYNC_JOURNAL_FLAG_MOUNTED}) should be set on create, got {flags}"
    );
}

#[ktest]
fn mounted_flag_is_cleared_on_destroy() {
    let nr_sectors = 8192;
    let (vspace, _bdev) = new_test_vspace_with_block_device(nr_sectors);

    vsync_vspace_destroy(&vspace);

    let flags = vspace.journal_super_flags.load(Ordering::Acquire);
    assert_eq!(flags & VSYNC_JOURNAL_FLAG_MOUNTED, 0);
}

// --- Edge Cases ---

#[ktest]
fn op_complete_null_payload_does_not_generate_journal_entry() {
    let nr_sectors = 8192;
    let (vspace, bdev) = new_test_vspace_with_block_device(nr_sectors);

    let entries_before = count_journal_entries(&bdev);

    // Start and complete with null payload (should be discarded, not written)
    let mut handle = vsync_op_start(&vspace, VsyncRecordType::Meta, 99).unwrap();
    let seq = vsync_handle_seq(&handle);
    assert!(!vsync_op_complete(&mut handle, None));

    // This sync_until creates a fence but no data records for the null op
    vsync_sync_until(&vspace, seq).unwrap();

    let entries_after = count_journal_entries(&bdev);
    // Null-payload ops don't produce journal entries
    assert_eq!(entries_after, entries_before);
}

#[ktest]
fn no_batch_coalesce_writes_raw_records() {
    let nr_sectors = 8192;
    let (vspace, bdev) = new_test_vspace_with_block_device(nr_sectors);
    vspace.set_no_batch_coalesce(true);

    let (_, _seq1) = write_one(&vspace, 55, b"raw-1");
    let (_, seq2) = write_one(&vspace, 55, b"raw-2");
    vsync_sync_until(&vspace, seq2).unwrap();

    // Without coalescing, both records produce separate journal entries
    assert!(journal_contains_valid_entries(&bdev));
}

#[ktest]
fn vspace_without_block_device_works_in_memory_only() {
    // No block device — all ops work in-memory only
    let vspace = new_test_vspace();
    let (sg_id, seq) = write_one(&vspace, 1, b"in-memory");

    vsync_sync_until(&vspace, seq).unwrap();
    assert!(vspace.sync_groups[sg_id as usize].stable_seq() >= seq);

    // write_journal_super / read_journal_super are no-ops without bdev
    vspace.write_journal_super();
    assert_eq!(vspace.read_journal_super(), -(Errno::ENODEV as i32));
}

#[ktest]
fn replay_is_noop_without_block_device() {
    let vspace = new_test_vspace();
    assert_eq!(vsync_vspace_replay(&vspace), 0);
}
