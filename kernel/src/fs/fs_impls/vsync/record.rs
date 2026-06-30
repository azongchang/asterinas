// SPDX-License-Identifier: MPL-2.0

//! VSync record primitives.
//!
//! This file implements purely in-memory record and barrier primitives.
//! Journal persistence/replay helpers are intentionally lightweight in this pass.

use alloc::{sync::Arc, vec, vec::Vec};
use core::sync::atomic::{AtomicU64, Ordering};

use ostd::sync::WaitQueue;

use super::*;
use crate::vm::page_cache::CachePage;

const RECORD_MAGIC: u64 = 0x5653594E33314A00; // "VSYN31J\0"

pub const VSYNC_JOURNAL_MAGIC: u32 = 0x56534A4E;
pub const VSYNC_JOURNAL_VERSION: u32 = 4;
pub const VSYNC_JOURNAL_FLAG_MOUNTED: u32 = 1 << 0;
pub const JOURNAL_ENTRY_HEADER_SIZE: usize = 46;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum VsyncRecordType {
    Write = 1,
    Meta = 2,
    Fence = 3,
    TxnHdr = 4,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VsyncShardOndisk {
    pub head_sector: u64,
    pub tail_sector: u64,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VsyncJournalSuper {
    pub magic: u32,
    pub version: u32,
    pub num_shards: u32,
    pub lb_size: u32,
    pub journal_start_sector: u64,
    pub journal_total_sectors: u64,
    pub shard_journal_sectors: u64,
    pub flags: u32,
    pub checksum: u32,
    pub fs_uuid: [u8; 16],
    pub shards: [VsyncShardOndisk; VSYNC_NUM_SHARDS],
    pub next_txn_id: [u32; VSYNC_NUM_SYNC_GROUPS],
    pub checkpointed_txn_id: [u32; VSYNC_NUM_SYNC_GROUPS],
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VsyncTxnHdrPayload {
    pub sg_id: u32,
    pub num_records: u32,
    pub max_seq: u64,
    pub txn_size: u64,
}

impl VsyncTxnHdrPayload {
    pub const SERIALIZED_SIZE: usize = size_of::<Self>();

    /// Deserializes a transaction-header payload.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SERIALIZED_SIZE {
            return None;
        }

        Some(Self {
            sg_id: read_u32(buf, 0),
            num_records: read_u32(buf, 4),
            max_seq: read_u64(buf, 8),
            txn_size: read_u64(buf, 16),
        })
    }
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct JournalEntryHeader {
    pub magic: u64,
    pub size: u32,
    pub rtype: u8,
    pub shard_id: u8,
    pub txn_id: u32,
    pub seq: u64,
    pub domain_id: u32,
    pub ino: u64,
    pub payload_len: u32,
    pub checksum: u32,
}

/// Creates payload storage for a journalled record.
pub fn vsync_payload_alloc(size: usize, _gfp: GfpT) -> Option<Vec<u8>> {
    Some(vec![0; size])
}

/// Releases payload storage for a journalled record.
pub fn vsync_payload_free(_payload: Vec<u8>) {}

/// Shared countdown/event primitive used by both fence token and pre-commit barrier flows.
pub struct CountdownLatch {
    remaining: AtomicU64,
    wait_queue: WaitQueue,
}

impl core::fmt::Debug for CountdownLatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CountdownLatch")
            .field("remaining", &self.remaining)
            .finish_non_exhaustive()
    }
}

impl CountdownLatch {
    /// Creates a latch with an arbitrary initial countdown value.
    pub fn new(remaining: u64) -> Self {
        Self {
            remaining: AtomicU64::new(remaining),
            wait_queue: WaitQueue::new(),
        }
    }

    /// Creates a latch for token-like use (multi-fence completion).
    pub fn new_token(num_fences: u64) -> Self {
        Self::new(num_fences)
    }

    /// Creates a latch for barrier-like use (single completion event).
    pub fn new_barrier() -> Self {
        Self::new(1)
    }

    /// Decrements countdown by one and wakes waiters when reaching zero.
    pub fn count_down(&self) -> bool {
        let prev = self.remaining.fetch_sub(1, Ordering::AcqRel);
        assert!(prev > 0, "CountdownLatch underflow.");
        if prev == 1 {
            self.wait_queue.wake_all();
            true
        } else {
            false
        }
    }

    /// Signals completion for barrier semantics.
    pub fn signal(&self) {
        if self.remaining.swap(0, Ordering::AcqRel) > 0 {
            self.wait_queue.wake_all();
        }
    }

    /// Waits until the countdown reaches zero.
    pub fn wait(&self) {
        self.wait_queue
            .wait_until(|| (self.remaining.load(Ordering::Acquire) == 0).then_some(()));
    }

    /// Returns current remaining countdown value.
    pub fn remaining(&self) -> u64 {
        self.remaining.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug)]
pub(super) struct WriteMetaPayload {
    pub(super) rtype: VsyncRecordType,
    pub(super) ino: u64,
    pub(super) payload: Option<Vec<u8>>,
    pub(super) page: Option<CachePage>,
    pub(super) offset: u32,
    pub(super) len: u32,
}

#[derive(Clone, Debug)]
pub(super) struct FencePayload {
    pub(super) token: Arc<CountdownLatch>,
    pub(super) target_seq: u64,
}

#[derive(Clone, Debug)]
pub(super) enum RecordTypePayload {
    WriteMeta(WriteMetaPayload),
    Fence(FencePayload),
    TxnHdr(VsyncTxnHdrPayload),
}

/// In-memory representation of a journal record.
#[derive(Clone, Debug)]
pub struct Record {
    pub(super) seq: VsyncSeq,
    pub(super) domain_id: u32,
    pub(super) txn_id: u32,
    pub(super) payload: RecordTypePayload,
    pub(super) pre_commit_barrier: Option<Arc<CountdownLatch>>,
}

impl Record {
    /// Creates a write record.
    pub fn new_write(seq: VsyncSeq, domain_id: u32, ino: u64, payload: Vec<u8>) -> Self {
        Self {
            seq,
            domain_id,
            txn_id: 0,
            payload: RecordTypePayload::WriteMeta(WriteMetaPayload {
                rtype: VsyncRecordType::Write,
                ino,
                payload: Some(payload),
                page: None,
                offset: 0,
                len: 0,
            }),
            pre_commit_barrier: None,
        }
    }

    /// Creates a metadata record.
    pub fn new_meta(seq: VsyncSeq, domain_id: u32, ino: u64, payload: Vec<u8>) -> Self {
        Self {
            seq,
            domain_id,
            txn_id: 0,
            payload: RecordTypePayload::WriteMeta(WriteMetaPayload {
                rtype: VsyncRecordType::Meta,
                ino,
                payload: Some(payload),
                page: None,
                offset: 0,
                len: 0,
            }),
            pre_commit_barrier: None,
        }
    }

    /// Creates a fence record.
    pub fn new_fence(
        seq: VsyncSeq,
        domain_id: u32,
        token: Arc<CountdownLatch>,
        target_seq: u64,
    ) -> Self {
        Self {
            seq,
            domain_id,
            txn_id: 0,
            payload: RecordTypePayload::Fence(FencePayload { token, target_seq }),
            pre_commit_barrier: None,
        }
    }

    /// Creates a transaction-header record.
    pub fn new_txn_hdr(seq: VsyncSeq, domain_id: u32, hdr: VsyncTxnHdrPayload) -> Self {
        Self {
            seq,
            domain_id,
            txn_id: 0,
            payload: RecordTypePayload::TxnHdr(hdr),
            pre_commit_barrier: None,
        }
    }

    /// Allocates a default metadata record.
    pub fn alloc(_gfp: GfpT) -> Self {
        Self::new_meta(0, 0, 0, Vec::new())
    }

    /// Drops this record.
    pub fn free(self) {}

    /// Drops this replay record.
    pub fn free_replay_record(self) {}

    /// Creates a record from a journal entry.
    pub fn from_journal_entry(entry: &JournalEntryHeader, payload_data: &[u8]) -> Option<Self> {
        if entry.magic != RECORD_MAGIC {
            return None;
        }
        let payload_len = usize::try_from(entry.payload_len).ok()?;
        if payload_len > payload_data.len() {
            return None;
        }
        let payload = &payload_data[..payload_len];
        if checksum_bytes(payload) != entry.checksum {
            return None;
        }
        let rtype = match entry.rtype {
            1 => VsyncRecordType::Write,
            2 => VsyncRecordType::Meta,
            3 => VsyncRecordType::Fence,
            4 => VsyncRecordType::TxnHdr,
            _ => return None,
        };

        let mut rec = match rtype {
            VsyncRecordType::Write => {
                Self::new_write(entry.seq, entry.domain_id, entry.ino, payload.to_vec())
            }
            VsyncRecordType::Meta => {
                Self::new_meta(entry.seq, entry.domain_id, entry.ino, payload.to_vec())
            }
            VsyncRecordType::Fence => Self::new_fence(
                entry.seq,
                entry.domain_id,
                Arc::new(CountdownLatch::new_token(1)),
                entry.seq,
            ),
            VsyncRecordType::TxnHdr => {
                let hdr = VsyncTxnHdrPayload::from_bytes(payload)?;
                Self::new_txn_hdr(entry.seq, entry.domain_id, hdr)
            }
        };
        rec.txn_id = entry.txn_id;
        Some(rec)
    }

    /// Materializes deferred payload buffers for write/meta records.
    ///
    /// In this in-memory pass, if a record still references a page but has no
    /// direct payload, a zeroed payload is synthesized with the same length.
    pub fn materialize_payload(&mut self) {
        if let RecordTypePayload::WriteMeta(write_meta) = &mut self.payload
            && write_meta.payload.is_none()
            && write_meta.page.is_some()
            && write_meta.len > 0
        {
            write_meta.payload = Some(vec![0; write_meta.len as usize]);
            write_meta.page = None;
            write_meta.offset = 0;
            write_meta.len = 0;
        }
    }

    /// Serializes a single record into `buf`.
    ///
    /// Returns bytes written, or `-errno` on failure.
    pub fn serialize(&self, buf: &mut [u8], shard_id: u8) -> i32 {
        if self.record_type() == VsyncRecordType::Fence {
            return 0;
        }

        let payload = self.payload_bytes();
        let payload_len = payload.len();
        let total = JOURNAL_ENTRY_HEADER_SIZE + payload_len;
        if total > buf.len() {
            return -(crate::error::Errno::ENOSPC as i32);
        }

        let checksum = payload
            .iter()
            .fold(0u32, |acc, byte| acc.wrapping_add(u32::from(*byte)));
        let ino = self.ino().unwrap_or(0);
        let mut cursor = 0;

        cursor += write_u64(buf, cursor, RECORD_MAGIC);
        cursor += write_u32(buf, cursor, total as u32);
        cursor += write_u8(buf, cursor, self.record_type() as u8);
        cursor += write_u8(buf, cursor, shard_id);
        cursor += write_u32(buf, cursor, self.txn_id);
        cursor += write_u64(buf, cursor, self.seq);
        cursor += write_u32(buf, cursor, self.domain_id);
        cursor += write_u64(buf, cursor, ino);
        cursor += write_u32(buf, cursor, payload_len as u32);
        cursor += write_u32(buf, cursor, checksum);
        buf[cursor..cursor + payload_len].copy_from_slice(payload);

        total as i32
    }

    /// Returns this record type.
    pub fn record_type(&self) -> VsyncRecordType {
        match self.payload {
            RecordTypePayload::WriteMeta(ref write_meta) => write_meta.rtype,
            RecordTypePayload::Fence(_) => VsyncRecordType::Fence,
            RecordTypePayload::TxnHdr(_) => VsyncRecordType::TxnHdr,
        }
    }

    /// Returns this record sequence.
    pub fn seq(&self) -> VsyncSeq {
        self.seq
    }

    /// Returns this record domain ID.
    pub fn domain_id(&self) -> u32 {
        self.domain_id
    }

    /// Returns this record transaction ID.
    pub fn txn_id(&self) -> u32 {
        self.txn_id
    }

    /// Sets this record transaction ID.
    pub fn set_txn_id(&mut self, txn_id: u32) {
        self.txn_id = txn_id;
    }

    /// Returns the inode number for write/meta records.
    pub fn ino(&self) -> Option<u64> {
        match &self.payload {
            RecordTypePayload::WriteMeta(write_meta) => Some(write_meta.ino),
            _ => None,
        }
    }

    /// Returns record payload bytes for write/meta records.
    pub fn payload_bytes(&self) -> &[u8] {
        match &self.payload {
            RecordTypePayload::WriteMeta(write_meta) => {
                write_meta.payload.as_deref().unwrap_or_default()
            }
            _ => &[],
        }
    }

    /// Sets record payload bytes for write/meta records.
    pub fn set_payload(&mut self, payload: Vec<u8>) {
        if let RecordTypePayload::WriteMeta(write_meta) = &mut self.payload {
            write_meta.payload = Some(payload);
            write_meta.page = None;
            write_meta.offset = 0;
            write_meta.len = 0;
        }
    }

    /// Returns fence target sequence for fence records.
    pub fn fence_target_seq(&self) -> Option<VsyncSeq> {
        match &self.payload {
            RecordTypePayload::Fence(fence) => Some(fence.target_seq),
            _ => None,
        }
    }

    /// Returns fence token for fence records.
    pub fn fence_token(&self) -> Option<&Arc<CountdownLatch>> {
        match &self.payload {
            RecordTypePayload::Fence(fence) => Some(&fence.token),
            _ => None,
        }
    }

    /// Returns transaction-header payload for transaction-header records.
    pub fn txn_header(&self) -> Option<VsyncTxnHdrPayload> {
        match &self.payload {
            RecordTypePayload::TxnHdr(hdr) => Some(*hdr),
            _ => None,
        }
    }

    /// Sets or clears the pre-commit barrier.
    pub fn set_pre_commit_barrier(&mut self, barrier: Option<Arc<CountdownLatch>>) {
        self.pre_commit_barrier = barrier;
    }

    /// Returns the pre-commit barrier.
    pub fn pre_commit_barrier(&self) -> Option<&Arc<CountdownLatch>> {
        self.pre_commit_barrier.as_ref()
    }
}

fn write_u8(buf: &mut [u8], offset: usize, value: u8) -> usize {
    buf[offset] = value;
    1
}

fn write_u32(buf: &mut [u8], offset: usize, value: u32) -> usize {
    let bytes = value.to_le_bytes();
    buf[offset..offset + bytes.len()].copy_from_slice(&bytes);
    bytes.len()
}

fn write_u64(buf: &mut [u8], offset: usize, value: u64) -> usize {
    let bytes = value.to_le_bytes();
    buf[offset..offset + bytes.len()].copy_from_slice(&bytes);
    bytes.len()
}

fn read_u8(buf: &[u8], offset: usize) -> u8 {
    buf[offset]
}

fn read_u32(buf: &[u8], offset: usize) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&buf[offset..offset + 4]);
    u32::from_le_bytes(bytes)
}

fn read_u64(buf: &[u8], offset: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&buf[offset..offset + 8]);
    u64::from_le_bytes(bytes)
}

/// Simple additive checksum over a byte slice.
pub fn checksum_bytes(data: &[u8]) -> u32 {
    data.iter()
        .fold(0u32, |acc, &byte| acc.wrapping_add(u32::from(byte)))
}

/// Superblock on-disk layout offsets (byte positions within the packed struct).
/// These match the `#[repr(C, packed)]` layout of `VsyncJournalSuper`.
const SUPER_MAGIC_OFF: usize = 0;
const SUPER_VERSION_OFF: usize = 4;
const SUPER_NUM_SHARDS_OFF: usize = 8;
const SUPER_LB_SIZE_OFF: usize = 12;
const SUPER_JOURNAL_START_OFF: usize = 16;
const SUPER_JOURNAL_TOTAL_OFF: usize = 24;
const SUPER_SHARD_JOURNAL_OFF: usize = 32;
const SUPER_FLAGS_OFF: usize = 40;
const SUPER_CHECKSUM_OFF: usize = 44;
const SUPER_FS_UUID_OFF: usize = 48;
const SUPER_SHARDS_OFF: usize = 64;
const SUPER_SHARD_ENTRY_SIZE: usize = 16;
const SUPER_NEXT_TXN_ID_OFF: usize = 64 + VSYNC_NUM_SHARDS * 16;
const SUPER_CKPT_TXN_ID_OFF: usize = 64 + VSYNC_NUM_SHARDS * 16 + VSYNC_NUM_SYNC_GROUPS * 4;
const SUPER_TOTAL_SIZE: usize = 64 + VSYNC_NUM_SHARDS * 16 + VSYNC_NUM_SYNC_GROUPS * 4 * 2;

/// Serializes a `VsyncJournalSuper` into a byte buffer.
/// Returns the number of bytes written.
pub fn serialize_superblock(superblock: &VsyncJournalSuper, buf: &mut [u8]) -> usize {
    assert!(buf.len() >= SUPER_TOTAL_SIZE);
    let mut off = 0;
    off += write_u32(buf, off, superblock.magic);
    off += write_u32(buf, off, superblock.version);
    off += write_u32(buf, off, superblock.num_shards);
    off += write_u32(buf, off, superblock.lb_size);
    off += write_u64(buf, off, superblock.journal_start_sector);
    off += write_u64(buf, off, superblock.journal_total_sectors);
    off += write_u64(buf, off, superblock.shard_journal_sectors);
    off += write_u32(buf, off, superblock.flags);
    let checksum_pos = off;
    off += write_u32(buf, off, 0); // placeholder for checksum
    // fs_uuid
    buf[off..off + 16].copy_from_slice(&superblock.fs_uuid);
    off += 16;
    // shards — copy to locals to avoid unaligned refs on packed struct
    let shards = superblock.shards;
    for shard in shards.iter() {
        off += write_u64(buf, off, shard.head_sector);
        off += write_u64(buf, off, shard.tail_sector);
    }
    // next_txn_id — copy to locals to avoid unaligned refs on packed struct
    let next_txn_id = superblock.next_txn_id;
    for val in next_txn_id {
        off += write_u32(buf, off, val);
    }
    // checkpointed_txn_id
    let checkpointed_txn_id = superblock.checkpointed_txn_id;
    for val in checkpointed_txn_id {
        off += write_u32(buf, off, val);
    }
    // Compute checksum (over all fields except checksum itself)
    let cs = checksum_bytes(&buf[SUPER_MAGIC_OFF..checksum_pos])
        .wrapping_add(checksum_bytes(&buf[checksum_pos + 4..off]));
    write_u32(buf, checksum_pos, cs);
    off
}

/// Deserializes a `VsyncJournalSuper` from a byte buffer.
/// Returns the superblock if parsing succeeds.
pub fn deserialize_superblock(buf: &[u8]) -> Option<VsyncJournalSuper> {
    if buf.len() < SUPER_TOTAL_SIZE {
        return None;
    }
    let mut off = 0;
    let magic = read_u32(buf, off);
    off += 4;
    let version = read_u32(buf, off);
    off += 4;
    let num_shards = read_u32(buf, off);
    off += 4;
    let lb_size = read_u32(buf, off);
    off += 4;
    let journal_start_sector = read_u64(buf, off);
    off += 8;
    let journal_total_sectors = read_u64(buf, off);
    off += 8;
    let shard_journal_sectors = read_u64(buf, off);
    off += 8;
    let flags = read_u32(buf, off);
    off += 4;
    let checksum = read_u32(buf, off);
    off += 4;
    let mut fs_uuid = [0u8; 16];
    fs_uuid.copy_from_slice(&buf[off..off + 16]);
    off += 16;

    let mut shards = [VsyncShardOndisk::default(); VSYNC_NUM_SHARDS];
    for s in shards.iter_mut() {
        s.head_sector = read_u64(buf, off);
        off += 8;
        s.tail_sector = read_u64(buf, off);
        off += 8;
    }

    let mut next_txn_id = [0u32; VSYNC_NUM_SYNC_GROUPS];
    for val in next_txn_id.iter_mut() {
        *val = read_u32(buf, off);
        off += 4;
    }

    let mut checkpointed_txn_id = [0u32; VSYNC_NUM_SYNC_GROUPS];
    for val in checkpointed_txn_id.iter_mut() {
        *val = read_u32(buf, off);
        off += 4;
    }

    // Validate checksum
    let computed = checksum_bytes(&buf[SUPER_MAGIC_OFF..SUPER_CHECKSUM_OFF])
        .wrapping_add(checksum_bytes(&buf[SUPER_CHECKSUM_OFF + 4..off]));
    if computed != checksum {
        return None;
    }

    Some(VsyncJournalSuper {
        magic,
        version,
        num_shards,
        lb_size,
        journal_start_sector,
        journal_total_sectors,
        shard_journal_sectors,
        flags,
        checksum,
        fs_uuid,
        shards,
        next_txn_id,
        checkpointed_txn_id,
    })
}
