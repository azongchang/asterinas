// SPDX-License-Identifier: MPL-2.0

use aster_block::{
    BlockDeviceMeta,
    bio::{BioEnqueueError, BioType, SubmittedBio},
};
use device_id::{DeviceId, MajorId, MinorId};
use ostd::{mm::io::util::HasVmReaderWriter, prelude::ktest};

use super::{
    inode::RawInode,
    journal::{Ext2Journal, Ext2WriteKind},
    prelude::*,
    super_block::{ErrorsBehaviour, MAGIC_NUM, OsId, RawSuperBlock, RevLevel, SuperBlock},
};
use crate::fs::fs_impls::vsync::VsyncRecordType;

const TEST_FS_BLOCKS: u32 = 32;
const TEST_JOURNAL_SPARE_SECTORS: u64 = 4096;

#[derive(Debug)]
struct TestExt2Device {
    data: Mutex<Vec<u8>>,
    nr_sectors: usize,
}

impl TestExt2Device {
    fn new() -> Arc<Self> {
        let fs_sectors = (TEST_FS_BLOCKS as u64 * BLOCK_SIZE as u64).div_ceil(SECTOR_SIZE as u64);
        let nr_sectors = fs_sectors + TEST_JOURNAL_SPARE_SECTORS;
        Arc::new(Self {
            data: Mutex::new(vec![0; nr_sectors as usize * SECTOR_SIZE]),
            nr_sectors: nr_sectors as usize,
        })
    }

    fn read_at(&self, offset: usize, len: usize) -> Vec<u8> {
        self.data.lock()[offset..offset + len].to_vec()
    }

    fn overwrite_at(&self, offset: usize, bytes: &[u8]) {
        self.data.lock()[offset..offset + bytes.len()].copy_from_slice(bytes);
    }
}

impl BlockDevice for TestExt2Device {
    fn enqueue(&self, bio: SubmittedBio) -> core::result::Result<(), BioEnqueueError> {
        let mut data = self.data.lock();
        let mut byte_offset = bio.sid_range().start.to_raw() as usize * SECTOR_SIZE;

        match bio.type_() {
            BioType::Read => {
                for segment in bio.segments() {
                    let nbytes = segment.nbytes();
                    let end = byte_offset + nbytes;
                    let mut reader = VmReader::from(&data[byte_offset..end]);
                    let mut writer = segment.inner_dma().writer().unwrap();
                    let _ = writer.write(&mut reader);
                    byte_offset = end;
                }
            }
            BioType::Write => {
                for segment in bio.segments() {
                    let nbytes = segment.nbytes();
                    let end = byte_offset + nbytes;
                    let mut reader = segment.inner_dma().reader().unwrap();
                    let mut writer = VmWriter::from(&mut data[byte_offset..end]);
                    let _ = reader.read(&mut writer);
                    byte_offset = end;
                }
            }
            BioType::Flush => {}
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
        "ext2-vsync-test"
    }

    fn id(&self) -> DeviceId {
        DeviceId::new(MajorId::new(255), MinorId::new(1))
    }
}

fn test_super_block() -> SuperBlock {
    let mut raw = RawSuperBlock::default();
    raw.inodes_count = 128;
    raw.blocks_count = TEST_FS_BLOCKS;
    raw.free_blocks_count = TEST_FS_BLOCKS - 8;
    raw.free_inodes_count = 120;
    raw.log_block_size = 2;
    raw.log_frag_size = 2;
    raw.blocks_per_group = TEST_FS_BLOCKS;
    raw.frags_per_group = TEST_FS_BLOCKS;
    raw.inodes_per_group = 128;
    raw.magic = MAGIC_NUM;
    raw.state = 1;
    raw.errors = ErrorsBehaviour::Continue as u16;
    raw.creator_os = OsId::Linux as u32;
    raw.rev_level = RevLevel::Dynamic as u32;
    raw.first_ino = 11;
    raw.inode_size = size_of::<RawInode>() as u16;
    raw.feature_compat = 0;
    raw.feature_incompat = 0;
    raw.feature_ro_compat = 0;

    SuperBlock::try_from(raw).unwrap()
}

#[ktest]
fn ext2_vsync_write_kind_maps_to_vsync_record_type() {
    assert_eq!(Ext2WriteKind::Metadata.record_type(), VsyncRecordType::Meta);
    assert_eq!(Ext2WriteKind::Metadata.ino(), 0);
    assert_eq!(
        Ext2WriteKind::Data { ino: 42 }.record_type(),
        VsyncRecordType::Write
    );
    assert_eq!(Ext2WriteKind::Data { ino: 42 }.ino(), 42);
}

#[ktest]
fn ext2_vsync_persistent_replay_applies_logged_range() {
    let block_device = TestExt2Device::new();
    let super_block = test_super_block();
    let offset = 4 * BLOCK_SIZE;
    let expected = vec![0x5a; SECTOR_SIZE];

    let journal = Ext2Journal::open(block_device.clone(), &super_block).unwrap();
    assert!(journal.is_persistent());
    journal
        .log_write(offset, &expected, Ext2WriteKind::Metadata)
        .unwrap();
    drop(journal);

    assert_eq!(
        block_device.read_at(offset, expected.len()),
        vec![0; expected.len()]
    );

    let replayed = Ext2Journal::open(block_device.clone(), &super_block).unwrap();
    drop(replayed);

    assert_eq!(block_device.read_at(offset, expected.len()), expected);
}

#[ktest]
fn ext2_vsync_checkpoint_applies_and_prevents_replay() {
    let block_device = TestExt2Device::new();
    let super_block = test_super_block();
    let offset = 5 * BLOCK_SIZE;
    let checkpointed = vec![0x11; SECTOR_SIZE];
    let later_unjournaled = vec![0x22; SECTOR_SIZE];

    let journal = Ext2Journal::open(block_device.clone(), &super_block).unwrap();
    journal
        .log_write(offset, &checkpointed, Ext2WriteKind::Data { ino: 7 })
        .unwrap();
    journal.checkpoint().unwrap();
    assert_eq!(
        block_device.read_at(offset, checkpointed.len()),
        checkpointed
    );

    block_device.overwrite_at(offset, &later_unjournaled);
    drop(journal);

    let replayed = Ext2Journal::open(block_device.clone(), &super_block).unwrap();
    drop(replayed);

    assert_eq!(
        block_device.read_at(offset, later_unjournaled.len()),
        later_unjournaled
    );
}

#[ktest]
fn ext2_vsync_skips_writes_inside_journal_range() {
    let block_device = TestExt2Device::new();
    let super_block = test_super_block();
    let journal = Ext2Journal::open(block_device, &super_block).unwrap();
    let journal_offset = TEST_FS_BLOCKS as usize * BLOCK_SIZE;

    assert_eq!(
        journal
            .log_write(
                journal_offset,
                &[0x33; SECTOR_SIZE],
                Ext2WriteKind::Metadata
            )
            .unwrap(),
        None
    );
}
