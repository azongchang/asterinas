// SPDX-License-Identifier: MPL-2.0

//! In-memory shard journal state.

use alloc::{collections::VecDeque, vec, vec::Vec};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use aster_block::BlockDevice;
use ostd::{
    mm::{VmIo, VmReader, VmWriter},
    sync::SpinLock,
};

use super::*;
use crate::prelude::{Errno, Error, Result, return_errno_with_message};

// Inline serialization helpers for transaction buffer construction.
fn write_u8_inline(buf: &mut [u8], offset: usize, value: u8) -> usize {
    buf[offset] = value;
    1
}
fn write_u32_inline(buf: &mut [u8], offset: usize, value: u32) -> usize {
    let bytes = value.to_le_bytes();
    buf[offset..offset + bytes.len()].copy_from_slice(&bytes);
    bytes.len()
}
fn write_u64_inline(buf: &mut [u8], offset: usize, value: u64) -> usize {
    let bytes = value.to_le_bytes();
    buf[offset..offset + bytes.len()].copy_from_slice(&bytes);
    bytes.len()
}
/// Result of reserving journal space, including wrap information.
#[derive(Clone, Copy, Debug, Default)]
pub struct JournalReservation {
    pub start_sector: u64,
    pub sectors: u64,
    pub final_head_sector: u64,
    pub wrap_start: Option<u64>,
    pub wrap_sectors: u64,
}

#[derive(Debug)]
struct PendingReplayTxn {
    txn_id: u32,
    expected_records: usize,
    records: Vec<Record>,
}

impl PendingReplayTxn {
    fn new(txn_id: u32, hdr: VsyncTxnHdrPayload) -> Self {
        Self {
            txn_id,
            expected_records: hdr.num_records as usize,
            records: Vec::new(),
        }
    }

    fn is_complete(&self) -> bool {
        self.records.len() == self.expected_records
    }

    fn finish(self, valid_records: &mut Vec<Record>) {
        if self.is_complete() {
            valid_records.extend(self.records);
        }
    }
}

/// Per-SG transaction slot in one shard.
#[derive(Debug)]
pub struct ShardSgSlot {
    queue: SpinLock<VecDeque<SgTransaction>>,
}

impl Default for ShardSgSlot {
    fn default() -> Self {
        Self {
            queue: SpinLock::new(VecDeque::new()),
        }
    }
}

impl ShardSgSlot {
    /// Pushes one transaction.
    pub fn push(&self, txn: SgTransaction) {
        self.queue.lock().push_back(txn);
    }

    /// Drains all queued transactions.
    pub fn drain(&self) -> Vec<SgTransaction> {
        self.queue.lock().drain(..).collect()
    }
}

/// Circular-journal accounting for one shard.
#[derive(Debug)]
pub struct VsyncShardJournal {
    pub start_sector: u64,
    pub size_sectors: u64,
    pub head_sector: AtomicU64,
    pub committed_head_sector: AtomicU64,
    pub tail_sector: AtomicU64,
    lock: SpinLock<()>,
}

impl VsyncShardJournal {
    /// Creates one shard journal range.
    pub fn new(start_sector: u64, size_sectors: u64) -> Self {
        Self {
            start_sector,
            size_sectors,
            head_sector: AtomicU64::new(start_sector),
            committed_head_sector: AtomicU64::new(start_sector),
            tail_sector: AtomicU64::new(start_sector),
            lock: SpinLock::new(()),
        }
    }

    /// Reserves sectors and returns the new head sector.
    pub fn reserve(&self, sectors: u64) -> Result<u64> {
        Ok(self.reserve_with_wrap(sectors)?.final_head_sector)
    }

    /// Reserves sectors and returns full wrap-aware reservation.
    ///
    /// Handles the case where the write spans the shard end boundary.
    pub fn reserve_with_wrap(&self, sectors: u64) -> Result<JournalReservation> {
        let _guard = self.lock.lock();
        if sectors == 0 || sectors >= self.size_sectors {
            return_errno_with_message!(Errno::ENOSPC, "VSync journal reservation is too large");
        }

        let start = self.head_sector.load(Ordering::Relaxed);
        let tail = self.tail_sector.load(Ordering::Acquire);
        let used = used_sectors_between(start, tail, self.size_sectors);
        let free = self.size_sectors.saturating_sub(used).saturating_sub(1);
        if sectors > free {
            return_errno_with_message!(Errno::ENOSPC, "VSync journal has no free space");
        }

        let shard_end = self.start_sector + self.size_sectors;

        let (new_head, wrap_start, wrap_sectors) = if start + sectors > shard_end {
            let space_at_end = shard_end - start;
            let at_start = sectors - space_at_end;
            let next = self.start_sector + at_start;
            (next, Some(self.start_sector), at_start)
        } else {
            (start + sectors, None, 0)
        };

        // Normalize: if head lands exactly at end, wrap to start
        let final_head = if new_head == shard_end {
            self.start_sector
        } else {
            new_head
        };
        self.head_sector.store(final_head, Ordering::Relaxed);

        Ok(JournalReservation {
            start_sector: start,
            sectors,
            final_head_sector: final_head,
            wrap_start,
            wrap_sectors,
        })
    }

    /// Marks I/O complete to `new_head`.
    pub fn mark_committed(&self, new_head: u64) {
        self.committed_head_sector
            .store(new_head, Ordering::Release);
    }

    /// Frees journal space up to `new_tail`.
    pub fn advance_tail(&self, new_tail: u64) {
        self.tail_sector.store(new_tail, Ordering::Release);
    }

    /// Returns used sectors.
    pub fn used_sectors(&self) -> u64 {
        let head = self.committed_head_sector.load(Ordering::Acquire);
        let tail = self.tail_sector.load(Ordering::Acquire);
        used_sectors_between(head, tail, self.size_sectors)
    }
}

fn used_sectors_between(head: u64, tail: u64, size_sectors: u64) -> u64 {
    if head >= tail {
        head - tail
    } else {
        size_sectors.saturating_sub(tail - head)
    }
}

/// One shard in a vspace.
#[derive(Debug)]
pub struct VsyncShard {
    pub shard_id: u32,
    pub sg_slots: [ShardSgSlot; VSYNC_NUM_SYNC_GROUPS],
    pub stopping: AtomicBool,
    pub journal: VsyncShardJournal,
    committed_transactions: SpinLock<VecDeque<CommittedTransaction>>,
    pub committed_txn_count: AtomicU64,
    pub records_written: AtomicU64,
    pub bytes_written: AtomicU64,
}

impl VsyncShard {
    /// Creates one shard.
    pub fn new(shard_id: u32, start_sector: u64, size_sectors: u64) -> Self {
        Self {
            shard_id,
            sg_slots: core::array::from_fn(|_| ShardSgSlot::default()),
            stopping: AtomicBool::new(false),
            journal: VsyncShardJournal::new(start_sector, size_sectors),
            committed_transactions: SpinLock::new(VecDeque::new()),
            committed_txn_count: AtomicU64::new(0),
            records_written: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
        }
    }

    /// Enqueues one transaction in the corresponding SG slot.
    pub fn enqueue_transaction(&self, txn: SgTransaction) {
        let sg_idx = txn.sg_id as usize % VSYNC_NUM_SYNC_GROUPS;
        self.sg_slots[sg_idx].push(txn);
    }

    /// Serializes all records in one transaction into a sector-aligned buffer.
    ///
    /// Format: txn header entry (JournalEntryHeader with TxnHdr + VsyncTxnHdrPayload),
    /// followed by each non-Fence record's JournalEntryHeader + payload.
    /// Total is zero-padded to next sector boundary.
    pub fn serialize_transaction(&self, txn: &SgTransaction) -> Vec<u8> {
        let data_records: Vec<&Record> = txn
            .records
            .iter()
            .filter(|r| {
                matches!(
                    r.record_type(),
                    VsyncRecordType::Write | VsyncRecordType::Meta
                )
            })
            .collect();
        let mut total = JOURNAL_ENTRY_HEADER_SIZE + size_of::<VsyncTxnHdrPayload>();
        for rec in &data_records {
            total += JOURNAL_ENTRY_HEADER_SIZE + rec.payload_bytes().len();
        }
        let aligned = total.div_ceil(aster_block::BLOCK_SIZE) * aster_block::BLOCK_SIZE;
        let mut buf = vec![0u8; aligned];

        // Write txn header entry
        let hdr_payload = VsyncTxnHdrPayload {
            sg_id: txn.sg_id,
            num_records: data_records.len() as u32,
            max_seq: txn.max_seq,
            txn_size: aligned as u64,
        };
        let mut hdr_payload_buf = [0u8; VsyncTxnHdrPayload::SERIALIZED_SIZE];
        let mut payload_cursor = 0;
        payload_cursor += write_u32_inline(&mut hdr_payload_buf, payload_cursor, hdr_payload.sg_id);
        payload_cursor += write_u32_inline(
            &mut hdr_payload_buf,
            payload_cursor,
            hdr_payload.num_records,
        );
        payload_cursor +=
            write_u64_inline(&mut hdr_payload_buf, payload_cursor, hdr_payload.max_seq);
        _ = write_u64_inline(&mut hdr_payload_buf, payload_cursor, hdr_payload.txn_size);
        let hdr_payload_checksum = checksum_bytes(&hdr_payload_buf);
        let hdr = JournalEntryHeader {
            magic: 0x5653594E33314A00,
            size: (JOURNAL_ENTRY_HEADER_SIZE + size_of::<VsyncTxnHdrPayload>()) as u32,
            rtype: VsyncRecordType::TxnHdr as u8,
            shard_id: self.shard_id as u8,
            txn_id: txn.txn_id as u32,
            seq: txn.max_seq,
            domain_id: 0,
            ino: 0,
            payload_len: size_of::<VsyncTxnHdrPayload>() as u32,
            checksum: hdr_payload_checksum,
        };
        // Serialize header + txn payload manually via Record::serialize-style writes
        let mut hdr_buf = vec![0u8; JOURNAL_ENTRY_HEADER_SIZE + size_of::<VsyncTxnHdrPayload>()];
        {
            let mut c = 0;
            c += write_u64_inline(&mut hdr_buf, c, hdr.magic);
            c += write_u32_inline(&mut hdr_buf, c, hdr.size);
            c += write_u8_inline(&mut hdr_buf, c, hdr.rtype);
            c += write_u8_inline(&mut hdr_buf, c, hdr.shard_id);
            c += write_u32_inline(&mut hdr_buf, c, hdr.txn_id);
            c += write_u64_inline(&mut hdr_buf, c, hdr.seq);
            c += write_u32_inline(&mut hdr_buf, c, hdr.domain_id);
            c += write_u64_inline(&mut hdr_buf, c, hdr.ino);
            c += write_u32_inline(&mut hdr_buf, c, hdr.payload_len);
            c += write_u32_inline(&mut hdr_buf, c, hdr.checksum);
            hdr_buf[c..c + hdr_payload_buf.len()].copy_from_slice(&hdr_payload_buf);
        }
        buf[..hdr_buf.len()].copy_from_slice(&hdr_buf);
        let mut off = hdr_buf.len();

        // Write each data record
        for rec in &data_records {
            let rec_size = JOURNAL_ENTRY_HEADER_SIZE + rec.payload_bytes().len();
            let mut rec_buf = vec![0u8; rec_size];
            let payload = rec.payload_bytes();
            {
                let mut c = 0;
                c += write_u64_inline(&mut rec_buf, c, 0x5653594E33314A00); // magic
                c += write_u32_inline(&mut rec_buf, c, rec_size as u32);
                c += write_u8_inline(&mut rec_buf, c, rec.record_type() as u8);
                c += write_u8_inline(&mut rec_buf, c, self.shard_id as u8);
                c += write_u32_inline(&mut rec_buf, c, txn.txn_id as u32);
                c += write_u64_inline(&mut rec_buf, c, rec.seq());
                c += write_u32_inline(&mut rec_buf, c, rec.domain_id());
                c += write_u64_inline(&mut rec_buf, c, rec.ino().unwrap_or(0));
                c += write_u32_inline(&mut rec_buf, c, payload.len() as u32);
                _ = write_u32_inline(&mut rec_buf, c, checksum_bytes(payload));
            }
            rec_buf[JOURNAL_ENTRY_HEADER_SIZE..].copy_from_slice(payload);
            buf[off..off + rec_buf.len()].copy_from_slice(&rec_buf);
            off += rec_buf.len();
        }
        buf
    }

    /// Flushes all queued transactions and returns them as completed transactions.
    ///
    /// When `block_device` is `Some`, transactions are written to the on-disk journal
    /// with crash-consistent ordering (write + flush). When `None`, only in-memory
    /// journal positions are advanced.
    pub fn flush_all(
        &self,
        block_device: Option<&dyn BlockDevice>,
    ) -> Result<Vec<CommittedTransaction>> {
        let mut completed = Vec::new();
        let shard_start_bytes = (self.journal.start_sector as usize) * aster_block::SECTOR_SIZE;
        let shard_end_bytes =
            shard_start_bytes + (self.journal.size_sectors as usize) * aster_block::SECTOR_SIZE;

        for slot in &self.sg_slots {
            for mut txn in slot.drain() {
                // Serialize to sector-aligned buffer
                let buf = self.serialize_transaction(&txn);
                let total_bytes = buf.len() as u64;
                let sectors = total_bytes.div_ceil(aster_block::SECTOR_SIZE as u64);

                // Reserve journal space
                let sectors = sectors.max(1);
                let reservation = self.journal.reserve_with_wrap(sectors)?;

                // Write to block device if available
                if let Some(bdev) = block_device
                    && sectors > 0
                {
                    let write_offset =
                        (reservation.start_sector as usize) * aster_block::SECTOR_SIZE;

                    if let Some(wrap_start) = reservation.wrap_start
                        && reservation.wrap_sectors > 0
                    {
                        // Two-part write across shard boundary
                        let part1_len = shard_end_bytes - write_offset;
                        let part1_buf = &buf[..part1_len];
                        let part2_buf = &buf[part1_len
                            ..part1_len
                                + (reservation.wrap_sectors as usize) * aster_block::SECTOR_SIZE];
                        if bdev
                            .write(write_offset, &mut VmReader::from(part1_buf).to_fallible())
                            .is_err()
                        {
                            return Err(Error::with_message(
                                Errno::EIO,
                                "failed to write first wrapped journal segment",
                            ));
                        }
                        if bdev
                            .write(
                                (wrap_start as usize) * aster_block::SECTOR_SIZE,
                                &mut VmReader::from(part2_buf).to_fallible(),
                            )
                            .is_err()
                        {
                            return Err(Error::with_message(
                                Errno::EIO,
                                "failed to write second wrapped journal segment",
                            ));
                        }
                    } else {
                        if bdev
                            .write(
                                write_offset,
                                &mut VmReader::from(buf.as_slice()).to_fallible(),
                            )
                            .is_err()
                        {
                            return Err(Error::with_message(
                                Errno::EIO,
                                "failed to write journal transaction",
                            ));
                        }
                    }
                    // Crash-consistent flush
                    if bdev.sync().is_err() {
                        return Err(Error::with_message(
                            Errno::EIO,
                            "failed to flush journal transaction",
                        ));
                    }
                }

                // Determine committed_head (may have wrapped)
                self.journal.mark_committed(reservation.final_head_sector);

                txn.state = VsyncTxnState::Completed;
                txn.start_sector = reservation.start_sector;
                txn.sector_count = sectors.max(1);
                txn.io_completed = true;

                let committed = CommittedTransaction {
                    sg_id: txn.sg_id,
                    txn_id: txn.txn_id,
                    shard_id: self.shard_id,
                    size_sectors: txn.sector_count,
                    start_sector: txn.start_sector,
                    records: txn.records,
                    io_complete: true,
                    processed: false,
                    journal_freed: false,
                };

                self.records_written
                    .fetch_add(committed.records.len() as u64, Ordering::Relaxed);
                self.bytes_written.fetch_add(total_bytes, Ordering::Relaxed);
                self.committed_txn_count.fetch_add(1, Ordering::Relaxed);
                self.committed_transactions
                    .lock()
                    .push_back(committed.clone());
                completed.push(committed);
            }
        }
        Ok(completed)
    }

    /// Pops and returns all committed transactions currently tracked in the shard.
    pub fn take_committed_transactions(&self) -> Vec<CommittedTransaction> {
        let drained = self
            .committed_transactions
            .lock()
            .drain(..)
            .collect::<Vec<_>>();
        self.committed_txn_count
            .fetch_sub(drained.len() as u64, Ordering::Relaxed);
        drained
    }

    /// Returns used journal sectors.
    pub fn journal_used_sectors(&self) -> u64 {
        self.journal.used_sectors()
    }

    /// Returns whether journal space pressure is high.
    pub fn journal_space_pressure(&self) -> bool {
        self.journal_used_sectors() * VSYNC_JOURNAL_PRESSURE_DEN
            >= self.journal.size_sectors * VSYNC_JOURNAL_PRESSURE_NUM
    }

    /// Placeholder shard flush thread entry for future async implementation.
    pub fn flush_thread_fn() -> i32 {
        0
    }

    /// Scans this shard's journal region for valid records after a crash.
    ///
    /// Reads from `tail_sector` to `committed_head_sector`, parsing each entry
    /// and reconstructing `Record` objects via `Record::from_journal_entry`.
    pub fn scan_journal(
        &self,
        block_device: Option<&dyn BlockDevice>,
        valid_records: &mut Vec<Record>,
    ) -> i32 {
        let tail = self.journal.tail_sector.load(Ordering::Acquire);
        let head = self.journal.committed_head_sector.load(Ordering::Acquire);
        if tail == head {
            return 0;
        }
        self.scan_journal_range(block_device, tail, head, valid_records)
    }

    /// Scans from a specific sector in this shard.
    pub fn scan_journal_from(
        &self,
        block_device: Option<&dyn BlockDevice>,
        from_sector: u64,
        valid_records: &mut Vec<Record>,
    ) -> i32 {
        let head = self.journal.committed_head_sector.load(Ordering::Acquire);
        if from_sector == head {
            return 0;
        }
        self.scan_journal_range(block_device, from_sector, head, valid_records)
    }

    /// Core scanning logic: reads `start..end` region, scans for RECORD_MAGIC,
    /// validates entries, and reconstructs records.
    fn scan_journal_range(
        &self,
        block_device: Option<&dyn BlockDevice>,
        start_sector: u64,
        end_sector: u64,
        valid_records: &mut Vec<Record>,
    ) -> i32 {
        let Some(bdev) = block_device else {
            return 0;
        };
        let shard_end = self.journal.start_sector + self.journal.size_sectors;

        // Handle normal (start < end) and wrapped (start > end) regions
        let ranges: Vec<(u64, u64)> = if start_sector <= end_sector {
            vec![(start_sector, end_sector)]
        } else {
            vec![
                (start_sector, shard_end),
                (self.journal.start_sector, end_sector),
            ]
        };

        const CHUNK_SECTORS: u64 = 512; // 256KB chunks
        let mut buf = vec![0u8; (CHUNK_SECTORS as usize) * aster_block::SECTOR_SIZE];
        let mut pending_txn: Option<PendingReplayTxn> = None;

        for (range_start, range_end) in ranges {
            let mut cursor = range_start;
            while cursor < range_end {
                let chunk_sectors = CHUNK_SECTORS.min(range_end - cursor);
                let chunk_bytes = (chunk_sectors as usize) * aster_block::SECTOR_SIZE;
                let slice_len = chunk_bytes;
                let buf_slice = &mut buf[..chunk_bytes];

                let byte_offset = (cursor as usize) * aster_block::SECTOR_SIZE;
                if bdev
                    .read(byte_offset, &mut VmWriter::from(buf_slice).to_fallible())
                    .is_err()
                {
                    break;
                }

                // Scan byte-by-byte for RECORD_MAGIC
                let buf_slice = &buf[..slice_len];
                let mut pos = 0;
                while pos + JOURNAL_ENTRY_HEADER_SIZE <= buf_slice.len() {
                    let magic = u64::from_le_bytes(buf_slice[pos..pos + 8].try_into().unwrap());
                    if magic != 0x5653594E33314A00 {
                        pos += 1;
                        continue;
                    }

                    // Parse entry header from buffer (copy to aligned local)
                    let entry = JournalEntryHeader {
                        magic,
                        size: u32::from_le_bytes(buf_slice[pos + 8..pos + 12].try_into().unwrap()),
                        rtype: buf_slice[pos + 12],
                        shard_id: buf_slice[pos + 13],
                        txn_id: u32::from_le_bytes(
                            buf_slice[pos + 14..pos + 18].try_into().unwrap(),
                        ),
                        seq: u64::from_le_bytes(buf_slice[pos + 18..pos + 26].try_into().unwrap()),
                        domain_id: u32::from_le_bytes(
                            buf_slice[pos + 26..pos + 30].try_into().unwrap(),
                        ),
                        ino: u64::from_le_bytes(buf_slice[pos + 30..pos + 38].try_into().unwrap()),
                        payload_len: u32::from_le_bytes(
                            buf_slice[pos + 38..pos + 42].try_into().unwrap(),
                        ),
                        checksum: u32::from_le_bytes(
                            buf_slice[pos + 42..pos + 46].try_into().unwrap(),
                        ),
                    };

                    // Validate entry fits in buffer
                    let entry_total = entry.size as usize;
                    if entry_total < JOURNAL_ENTRY_HEADER_SIZE
                        || pos + entry_total > buf_slice.len()
                    {
                        pos += 1;
                        continue;
                    }
                    let payload_len = entry.payload_len as usize;
                    if payload_len > entry_total - JOURNAL_ENTRY_HEADER_SIZE {
                        pos += entry_total;
                        continue;
                    }

                    let payload_data = &buf_slice[pos + JOURNAL_ENTRY_HEADER_SIZE
                        ..pos + JOURNAL_ENTRY_HEADER_SIZE + payload_len];
                    if let Some(rec) = Record::from_journal_entry(&entry, payload_data) {
                        match rec.record_type() {
                            VsyncRecordType::TxnHdr => {
                                if let Some(txn) = pending_txn.take() {
                                    txn.finish(valid_records);
                                }
                                if let Some(hdr) = rec.txn_header()
                                    && hdr.num_records > 0
                                {
                                    pending_txn = Some(PendingReplayTxn::new(rec.txn_id(), hdr));
                                }
                            }
                            VsyncRecordType::Write | VsyncRecordType::Meta => {
                                if let Some(txn) = pending_txn.as_mut()
                                    && txn.txn_id == rec.txn_id()
                                {
                                    txn.records.push(rec);
                                    if txn.is_complete()
                                        && let Some(txn) = pending_txn.take()
                                    {
                                        txn.finish(valid_records);
                                    }
                                }
                            }
                            VsyncRecordType::Fence => {}
                        }
                    }
                    pos += entry_total.max(1);
                }
                cursor += chunk_sectors;
            }
        }
        if let Some(txn) = pending_txn.take() {
            txn.finish(valid_records);
        }
        0
    }
}
