// SPDX-License-Identifier: MPL-2.0

use ostd::mm::io::util::HasVmReaderWriter;

use super::{prelude::*, super_block::SuperBlock};
use crate::fs::fs_impls::vsync::{
    Record, VSYNC_NUM_SHARDS, VsyncCoalesceOps, VsyncCoalesceRange, VsyncConfig, VsyncRecordType,
    VsyncSeq, VsyncVspace, vsync_handle_seq, vsync_op_complete, vsync_op_start, vsync_sync_all,
    vsync_sync_until, vsync_vspace_checkpoint, vsync_vspace_create, vsync_vspace_destroy,
    vsync_vspace_replay, vsync_vspace_set_coalesce_ops,
};

const PAYLOAD_HEADER_SIZE: usize = 8;
const RANGE_HEADER_SIZE: usize = 24;
const DEFAULT_IN_MEMORY_JOURNAL_SECTORS: u64 = 1024 * 1024;

/// The ext2 VSync journal for one mounted filesystem.
pub(super) struct Ext2Journal {
    vspace: Arc<VsyncVspace>,
    persistent: bool,
    journal_start_sector: u64,
    journal_total_sectors: u64,
}

/// The ext2 write category to publish into VSync.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Ext2WriteKind {
    Metadata,
    Data { ino: u64 },
}

impl Ext2WriteKind {
    pub(super) fn record_type(self) -> VsyncRecordType {
        match self {
            Self::Metadata => VsyncRecordType::Meta,
            Self::Data { .. } => VsyncRecordType::Write,
        }
    }

    pub(super) fn ino(self) -> u64 {
        match self {
            Self::Metadata => 0,
            Self::Data { ino } => ino,
        }
    }
}

impl Ext2Journal {
    /// Opens the VSync journal for an ext2 filesystem.
    pub fn open(block_device: Arc<dyn BlockDevice>, super_block: &SuperBlock) -> Result<Arc<Self>> {
        let journal_range = Self::persistent_journal_range(block_device.as_ref(), super_block)?;
        let (journal_start_sector, journal_total_sectors, persistent_block_device) =
            if let Some((start_sector, total_sectors)) = journal_range {
                (start_sector, total_sectors, Some(block_device.clone()))
            } else {
                (0, DEFAULT_IN_MEMORY_JOURNAL_SECTORS, None)
            };

        let context = Arc::new(Ext2JournalContext {
            block_device: block_device.clone(),
        });
        let vspace = vsync_vspace_create(
            None,
            Some(context),
            Some(VsyncConfig::default()),
            persistent_block_device,
            journal_start_sector,
            journal_total_sectors,
        );
        vsync_vspace_set_coalesce_ops(&vspace, Some(Arc::new(Ext2CoalesceOps)));

        if journal_range.is_some() {
            let replay_result = vsync_vspace_replay(&vspace);
            if replay_result != 0 {
                return Err(Error::with_message(
                    Errno::EIO,
                    "failed to replay ext2 VSync journal",
                ));
            }
        }

        Ok(Arc::new(Self {
            vspace,
            persistent: journal_range.is_some(),
            journal_start_sector,
            journal_total_sectors,
        }))
    }

    /// Returns whether this journal persists records on the block device.
    pub fn is_persistent(&self) -> bool {
        self.persistent
    }

    /// Logs a physical write before the caller writes it in place.
    pub fn log_write(
        &self,
        offset: usize,
        bytes: &[u8],
        kind: Ext2WriteKind,
    ) -> Result<Option<VsyncSeq>> {
        if bytes.is_empty() || self.overlaps_journal_range(offset, bytes.len())? {
            return Ok(None);
        }
        if offset % SECTOR_SIZE != 0 || bytes.len() % SECTOR_SIZE != 0 {
            return_errno_with_message!(Errno::EINVAL, "ext2 VSync write is not sector aligned");
        }

        let mut handle = vsync_op_start(&self.vspace, kind.record_type(), kind.ino())?;
        let seq = vsync_handle_seq(&handle);
        let payload = match build_payload_from_bytes(offset, bytes, seq) {
            Ok(payload) => payload,
            Err(err) => {
                let _ = vsync_op_complete(&mut handle, None);
                return Err(err);
            }
        };

        if !vsync_op_complete(&mut handle, Some(payload)) {
            return Ok(None);
        }

        vsync_sync_until(&self.vspace, seq)?;
        self.persist_journal_super()?;
        Ok(Some(seq))
    }

    /// Checkpoints all committed VSync records.
    pub fn checkpoint(&self) -> Result<()> {
        vsync_sync_all(&self.vspace)?;
        let _ = vsync_vspace_checkpoint(&self.vspace, 0)?;
        Ok(())
    }

    /// Destroys the VSync journal.
    pub fn destroy(&self) {
        vsync_vspace_destroy(&self.vspace);
    }

    fn persistent_journal_range(
        block_device: &dyn BlockDevice,
        super_block: &SuperBlock,
    ) -> Result<Option<(u64, u64)>> {
        let fs_bytes = (super_block.total_blocks() as usize)
            .checked_mul(super_block.block_size())
            .ok_or_else(|| Error::with_message(Errno::EINVAL, "ext2 size overflow"))?;
        let fs_sectors = fs_bytes.div_ceil(SECTOR_SIZE) as u64;
        let device_sectors = block_device.metadata().nr_sectors as u64;
        let spare_sectors = device_sectors.saturating_sub(fs_sectors);
        let min_journal_sectors =
            ((VSYNC_NUM_SHARDS + 1) * BLOCK_SIZE).div_ceil(SECTOR_SIZE) as u64;

        if spare_sectors < min_journal_sectors {
            return Ok(None);
        }

        Ok(Some((fs_sectors, spare_sectors)))
    }

    fn overlaps_journal_range(&self, offset: usize, len: usize) -> Result<bool> {
        if !self.persistent {
            return Ok(false);
        }

        let write_start = offset as u64;
        let write_end = write_start
            .checked_add(len as u64)
            .ok_or_else(|| Error::with_message(Errno::EINVAL, "write range overflow"))?;
        let journal_start = self
            .journal_start_sector
            .checked_mul(SECTOR_SIZE as u64)
            .ok_or_else(|| Error::with_message(Errno::EINVAL, "journal start overflow"))?;
        let journal_len = self
            .journal_total_sectors
            .checked_mul(SECTOR_SIZE as u64)
            .ok_or_else(|| Error::with_message(Errno::EINVAL, "journal size overflow"))?;
        let journal_end = journal_start
            .checked_add(journal_len)
            .ok_or_else(|| Error::with_message(Errno::EINVAL, "journal range overflow"))?;

        Ok(write_start < journal_end && write_end > journal_start)
    }

    fn persist_journal_super(&self) -> Result<()> {
        if !self.persistent {
            return Ok(());
        }

        if self.vspace.write_journal_super() < 0 {
            return_errno_with_message!(Errno::EIO, "failed to persist ext2 VSync journal head");
        }

        Ok(())
    }
}

impl Debug for Ext2Journal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Ext2Journal")
            .field("persistent", &self.persistent)
            .field("journal_start_sector", &self.journal_start_sector)
            .field("journal_total_sectors", &self.journal_total_sectors)
            .finish_non_exhaustive()
    }
}

struct Ext2JournalContext {
    block_device: Arc<dyn BlockDevice>,
}

impl Ext2JournalContext {
    fn apply_bytes(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        if offset % SECTOR_SIZE != 0 || bytes.len() % SECTOR_SIZE != 0 {
            return_errno_with_message!(Errno::EINVAL, "ext2 VSync replay is not sector aligned");
        }

        self.block_device
            .write(offset, &mut VmReader::from(bytes).to_fallible())
            .map_err(|_| Error::with_message(Errno::EIO, "failed to apply ext2 VSync range"))?;
        self.block_device
            .sync()
            .map_err(|_| Error::with_message(Errno::EIO, "failed to flush ext2 VSync range"))?;
        Ok(())
    }
}

#[derive(Default)]
struct Ext2CoalesceOps;

impl VsyncCoalesceOps for Ext2CoalesceOps {
    fn iter_ranges(
        &self,
        rec: &Record,
        cb: &mut dyn FnMut(VsyncCoalesceRange) -> Result<()>,
    ) -> Result<()> {
        for range in parse_payload_ranges(rec.payload_bytes())? {
            cb(VsyncCoalesceRange {
                blocknr: range.blocknr,
                offset: range.offset,
                len: range.data.len() as u32,
                seq: if range.seq == 0 { rec.seq() } else { range.seq },
                data: range.data,
            })?;
        }
        Ok(())
    }

    fn build_payload(&self, ranges: &[VsyncCoalesceRange]) -> Result<Vec<u8>> {
        build_payload_from_ranges(ranges)
    }

    fn apply_range(&self, fs_context: &dyn Any, range: &VsyncCoalesceRange) -> Result<()> {
        if range.data.len() != range.len as usize {
            return_errno_with_message!(Errno::EINVAL, "invalid ext2 VSync replay range length");
        }
        if range.offset as usize + range.data.len() > BLOCK_SIZE {
            return_errno_with_message!(Errno::EINVAL, "ext2 VSync range crosses block boundary");
        }

        let Some(context) = fs_context.downcast_ref::<Ext2JournalContext>() else {
            return_errno_with_message!(Errno::EINVAL, "invalid ext2 VSync replay context");
        };
        let block_offset = (range.blocknr as usize)
            .checked_mul(BLOCK_SIZE)
            .ok_or_else(|| Error::with_message(Errno::EINVAL, "ext2 VSync block overflow"))?;
        let offset = block_offset
            .checked_add(range.offset as usize)
            .ok_or_else(|| Error::with_message(Errno::EINVAL, "ext2 VSync offset overflow"))?;

        context.apply_bytes(offset, &range.data)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Ext2PayloadRange {
    blocknr: u64,
    offset: u32,
    seq: VsyncSeq,
    data: Vec<u8>,
}

fn build_payload_from_bytes(offset: usize, bytes: &[u8], seq: VsyncSeq) -> Result<Vec<u8>> {
    let mut ranges = Vec::new();
    let mut cursor = 0;
    while cursor < bytes.len() {
        let absolute_offset = offset
            .checked_add(cursor)
            .ok_or_else(|| Error::with_message(Errno::EINVAL, "ext2 VSync payload overflow"))?;
        let blocknr = (absolute_offset / BLOCK_SIZE) as u64;
        let block_offset = absolute_offset % BLOCK_SIZE;
        let chunk_len = (BLOCK_SIZE - block_offset).min(bytes.len() - cursor);
        ranges.push(VsyncCoalesceRange {
            blocknr,
            offset: block_offset as u32,
            len: chunk_len as u32,
            seq,
            data: bytes[cursor..cursor + chunk_len].to_vec(),
        });
        cursor += chunk_len;
    }
    build_payload_from_ranges(&ranges)
}

fn build_payload_from_ranges(ranges: &[VsyncCoalesceRange]) -> Result<Vec<u8>> {
    let mut total_size = PAYLOAD_HEADER_SIZE;
    for range in ranges {
        if range.data.len() != range.len as usize {
            return_errno_with_message!(Errno::EINVAL, "invalid ext2 VSync payload range length");
        }
        if range.offset as usize + range.data.len() > BLOCK_SIZE {
            return_errno_with_message!(Errno::EINVAL, "ext2 VSync range crosses block boundary");
        }
        total_size = total_size
            .checked_add(RANGE_HEADER_SIZE)
            .and_then(|size| size.checked_add(range.data.len()))
            .ok_or_else(|| Error::with_message(Errno::EINVAL, "ext2 VSync payload overflow"))?;
    }

    let mut payload = Vec::with_capacity(total_size);
    payload.extend_from_slice(&(ranges.len() as u32).to_le_bytes());
    payload.extend_from_slice(&(total_size as u32).to_le_bytes());
    for range in ranges {
        payload.extend_from_slice(&range.blocknr.to_le_bytes());
        payload.extend_from_slice(&range.offset.to_le_bytes());
        payload.extend_from_slice(&(range.data.len() as u32).to_le_bytes());
        payload.extend_from_slice(&range.seq.to_le_bytes());
        payload.extend_from_slice(&range.data);
    }
    Ok(payload)
}

fn parse_payload_ranges(payload: &[u8]) -> Result<Vec<Ext2PayloadRange>> {
    if payload.len() < PAYLOAD_HEADER_SIZE {
        return_errno_with_message!(Errno::EINVAL, "truncated ext2 VSync payload header");
    }

    let num_entries = read_u32(payload, 0)? as usize;
    let total_size = read_u32(payload, 4)? as usize;
    if total_size != payload.len() {
        return_errno_with_message!(Errno::EINVAL, "invalid ext2 VSync payload size");
    }

    let mut cursor = PAYLOAD_HEADER_SIZE;
    let mut ranges = Vec::with_capacity(num_entries);
    for _ in 0..num_entries {
        if cursor + RANGE_HEADER_SIZE > payload.len() {
            return_errno_with_message!(Errno::EINVAL, "truncated ext2 VSync range header");
        }

        let blocknr = read_u64(payload, cursor)?;
        cursor += 8;
        let offset = read_u32(payload, cursor)?;
        cursor += 4;
        let len = read_u32(payload, cursor)? as usize;
        cursor += 4;
        let seq = read_u64(payload, cursor)?;
        cursor += 8;

        let end = cursor
            .checked_add(len)
            .ok_or_else(|| Error::with_message(Errno::EINVAL, "ext2 VSync range overflow"))?;
        if end > payload.len() {
            return_errno_with_message!(Errno::EINVAL, "truncated ext2 VSync range data");
        }
        if offset as usize + len > BLOCK_SIZE {
            return_errno_with_message!(Errno::EINVAL, "ext2 VSync range crosses block boundary");
        }

        ranges.push(Ext2PayloadRange {
            blocknr,
            offset,
            seq,
            data: payload[cursor..end].to_vec(),
        });
        cursor = end;
    }

    if cursor != payload.len() {
        return_errno_with_message!(Errno::EINVAL, "trailing ext2 VSync payload bytes");
    }

    Ok(ranges)
}

fn read_u32(buf: &[u8], offset: usize) -> Result<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| Error::with_message(Errno::EINVAL, "ext2 VSync read overflow"))?;
    if end > buf.len() {
        return_errno_with_message!(Errno::EINVAL, "truncated ext2 VSync u32");
    }

    let mut bytes = [0; 4];
    bytes.copy_from_slice(&buf[offset..end]);
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(buf: &[u8], offset: usize) -> Result<u64> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| Error::with_message(Errno::EINVAL, "ext2 VSync read overflow"))?;
    if end > buf.len() {
        return_errno_with_message!(Errno::EINVAL, "truncated ext2 VSync u64");
    }

    let mut bytes = [0; 8];
    bytes.copy_from_slice(&buf[offset..end]);
    Ok(u64::from_le_bytes(bytes))
}

pub(super) fn bio_segment_bytes(bio_segment: &BioSegment) -> Result<Vec<u8>> {
    let mut bytes = vec![0; bio_segment.nbytes()];
    let mut writer = VmWriter::from(bytes.as_mut_slice()).to_fallible();
    bio_segment
        .inner_dma_slice()
        .reader()?
        .read_fallible(&mut writer)?;
    Ok(bytes)
}

#[cfg(ktest)]
mod tests {
    use ostd::prelude::ktest;

    use super::*;

    #[ktest]
    fn ext2_vsync_payload_round_trip() {
        let mut bytes = vec![0u8; BLOCK_SIZE + SECTOR_SIZE];
        for (idx, byte) in bytes.iter_mut().enumerate() {
            *byte = (idx % 251) as u8;
        }

        let payload = build_payload_from_bytes(SECTOR_SIZE, &bytes, 42).unwrap();
        let ranges = parse_payload_ranges(&payload).unwrap();

        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].blocknr, 0);
        assert_eq!(ranges[0].offset, SECTOR_SIZE as u32);
        assert_eq!(ranges[0].seq, 42);
        assert_eq!(ranges[0].data, bytes[..BLOCK_SIZE - SECTOR_SIZE]);
        assert_eq!(ranges[1].blocknr, 1);
        assert_eq!(ranges[1].offset, 0);
        assert_eq!(ranges[1].data, bytes[BLOCK_SIZE - SECTOR_SIZE..]);
    }

    #[ktest]
    fn ext2_vsync_payload_rejects_truncated_range() {
        let payload = build_payload_from_bytes(0, &[7u8; SECTOR_SIZE], 9).unwrap();
        let truncated = &payload[..payload.len() - 1];

        assert!(parse_payload_ranges(truncated).is_err());
    }

    #[ktest]
    fn ext2_vsync_payload_rejects_out_of_block_range() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&(PAYLOAD_HEADER_SIZE + RANGE_HEADER_SIZE + 2).to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes());
        payload.extend_from_slice(&((BLOCK_SIZE - 1) as u32).to_le_bytes());
        payload.extend_from_slice(&2u32.to_le_bytes());
        payload.extend_from_slice(&1u64.to_le_bytes());
        payload.extend_from_slice(&[1, 2]);

        assert!(parse_payload_ranges(&payload).is_err());
    }

    #[ktest]
    fn ext2_vsync_reads_to_device_bio_segment() {
        let bio_segment = BioSegment::alloc(1, BioDirection::ToDevice);
        let mut expected = vec![0u8; BLOCK_SIZE];
        for (idx, byte) in expected.iter_mut().enumerate() {
            *byte = (idx % 251) as u8;
        }

        bio_segment
            .writer()
            .unwrap()
            .write_fallible(&mut VmReader::from(expected.as_slice()).to_fallible())
            .unwrap();

        assert_eq!(bio_segment_bytes(&bio_segment).unwrap(), expected);
    }
}
