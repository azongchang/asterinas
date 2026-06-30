// SPDX-License-Identifier: MPL-2.0

//! In-memory coalescing for VSync batches and checkpoint apply.

use super::*;
use crate::prelude::*;

/// A coalesced range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VsyncCoalesceRange {
    pub blocknr: u64,
    pub offset: u32,
    pub len: u32,
    pub seq: VsyncSeq,
    pub data: Vec<u8>,
}

/// Filesystem-provided range coalescing callbacks.
pub trait VsyncCoalesceOps: Send + Sync {
    /// Iterates all ranges encoded by one record.
    fn iter_ranges(
        &self,
        rec: &Record,
        cb: &mut dyn FnMut(VsyncCoalesceRange) -> Result<()>,
    ) -> Result<()>;

    /// Builds one payload from coalesced ranges.
    fn build_payload(&self, ranges: &[VsyncCoalesceRange]) -> Result<Vec<u8>>;

    /// Applies one coalesced range to filesystem state.
    fn apply_range(&self, fs_context: &dyn Any, range: &VsyncCoalesceRange) -> Result<()>;

    /// Finalizes one coalescing pass.
    fn finalize(&self, _fs_context: &dyn Any, _range_count: u64, _coalesced_count: u64) {}
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct BatchKey {
    ino: u64,
    rtype: VsyncRecordType,
    domain_id: u32,
    blocknr: u64,
    offset: u32,
    len: u32,
}

/// Coalesces one batch of records.
pub fn coalesce_batch(
    batch: Vec<Record>,
    coalesce_ops: Option<&dyn VsyncCoalesceOps>,
) -> Vec<Record> {
    let Some(ops) = coalesce_ops else {
        return batch;
    };
    if batch.len() < 2 {
        return batch;
    }

    let mut preserved = VecDeque::new();
    let mut dedup = BTreeMap::<BatchKey, VsyncCoalesceRange>::new();
    let mut groups = BTreeMap::<(u64, VsyncRecordType, u32), Vec<VsyncCoalesceRange>>::new();
    let mut fallback = Vec::new();

    for record in batch {
        match record.record_type() {
            VsyncRecordType::Write | VsyncRecordType::Meta => {
                let Some(ino) = record.ino() else {
                    fallback.push(record);
                    continue;
                };
                let mut parsed = Vec::new();
                let mut parse_failed = false;
                let mut cb = |range: VsyncCoalesceRange| {
                    parsed.push(range);
                    Ok(())
                };
                if ops.iter_ranges(&record, &mut cb).is_err() || parsed.is_empty() {
                    parse_failed = true;
                }
                if parse_failed {
                    fallback.push(record);
                    continue;
                }

                for mut range in parsed {
                    if range.data.len() != range.len as usize {
                        range.len = range.data.len() as u32;
                    }
                    let key = BatchKey {
                        ino,
                        rtype: record.record_type(),
                        domain_id: record.domain_id(),
                        blocknr: range.blocknr,
                        offset: range.offset,
                        len: range.len,
                    };
                    dedup
                        .entry(key)
                        .and_modify(|existing| {
                            if range.seq >= existing.seq {
                                *existing = range.clone();
                            }
                        })
                        .or_insert(range);
                }
            }
            _ => preserved.push_back(record),
        }
    }

    for (key, range) in dedup {
        groups
            .entry((key.ino, key.rtype, key.domain_id))
            .or_default()
            .push(range);
    }

    let mut rebuilt = Vec::new();
    for ((ino, rtype, domain_id), mut ranges) in groups {
        ranges.sort_by_key(|range| (range.blocknr, range.seq, range.offset));
        let Ok(payload) = ops.build_payload(&ranges) else {
            continue;
        };
        let max_seq = ranges.iter().map(|range| range.seq).max().unwrap_or(0);
        let rebuilt_record = match rtype {
            VsyncRecordType::Write => Record::new_write(max_seq, domain_id, ino, payload),
            VsyncRecordType::Meta => Record::new_meta(max_seq, domain_id, ino, payload),
            _ => continue,
        };
        rebuilt.push(rebuilt_record);
    }

    rebuilt.extend(fallback);
    rebuilt.extend(preserved);
    rebuilt.sort_by_key(Record::seq);
    rebuilt
}

/// Applies checkpoint coalescing.
pub fn checkpoint_coalesce(
    records: &[Record],
    ops: &dyn VsyncCoalesceOps,
    fs_context: &dyn Any,
    _online: bool,
) -> Result<()> {
    let mut dedup = BTreeMap::<(u64, u32, u32), VsyncCoalesceRange>::new();
    let mut total_ranges = 0_u64;

    for record in records {
        if matches!(record.record_type(), VsyncRecordType::Fence) {
            continue;
        }
        let mut cb = |mut range: VsyncCoalesceRange| {
            total_ranges += 1;
            if range.data.len() != range.len as usize {
                range.len = range.data.len() as u32;
            }
            let key = (range.blocknr, range.offset, range.len);
            dedup
                .entry(key)
                .and_modify(|existing| {
                    if range.seq >= existing.seq {
                        *existing = range.clone();
                    }
                })
                .or_insert(range);
            Ok(())
        };
        ops.iter_ranges(record, &mut cb)?;
    }

    let mut merged = dedup.into_values().collect::<Vec<_>>();
    merged.sort_by_key(|range| (range.blocknr, range.seq, range.offset));

    for range in &merged {
        ops.apply_range(fs_context, range)?;
    }

    let coalesced = total_ranges.saturating_sub(merged.len() as u64);
    ops.finalize(fs_context, merged.len() as u64, coalesced);
    Ok(())
}

impl VsyncVspace {
    /// Applies checkpoint coalescing with the vspace's registered callbacks.
    pub fn checkpoint_coalesce(
        &self,
        records: &[Record],
        fs_context: &dyn Any,
        online: bool,
    ) -> Result<()> {
        let ops = self.coalesce_ops.read().clone();
        let Some(ops) = ops else {
            return Ok(());
        };
        checkpoint_coalesce(records, ops.as_ref(), fs_context, online)
    }
}
