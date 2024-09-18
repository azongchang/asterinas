// SPDX-License-Identifier: MPL-2.0

use alloc::collections::btree_map::BTreeMap;

use super::Layout;

// Save the mapping between thread and thread cache.
struct ThreadCacheTable {
    thread_cache_table: BTreeMap<usize, ThreadCache>,
}

struct ThreadCache {
    // TODO: Use linked-lists to implement freelists.
    free_list: FreeList<size_class>,    // A series of free lists of size classes
    size: usize,                        // Combined size of data
    max_size: usize,                    // size > max_size --> scavenge()
}

// Basic operations of `ThreadCache`.
impl ThreadCache {
    // TODO: Allocate default thread cache for a new thread.
    fn new() -> Self {

    }

    // TODO: Release thread cache of a thread.
    fn cleaup(&mut self) {

    }

    // TODO: Convert layout to nearest size class and allocate / deallocate an object.
    unsafe fn allocate(&mut self, layout: Layout) -> *mut u8 {

    }

    unsafe fn deallocate(&mut self, ptr: *mut u8, layout: Layout) {

    }
}

// Interact with other threads.
impl ThreadCache {
    fn init_tsd() {

    }
    fn get_cache() -> Self {

    }
    fn get_cache_if_present() -> Self {
        
    }
    fn become_idle() {

    }
}

// Interact with `TCMalloc` middle end.
impl ThreadCache {
    // Gets and returns an object from the transfer cache, and, if possible,
    // also adds some objects of that size class to this thread cache.
    fn fetch_from_transfer_cache(&mut self, size_class: SizeClassType, byte_size: usize) {

    }

    // Releases `count` items from this thread cache.
    fn release_to_transfer_cache(src: FreeList, size_class: SizeClassType, count: usize) {

    }

    // Releases some number of items from src.  Adjusts the list's max_length
    // to eventually converge on `num_objects_to_move(size_class)`.
    fn list_too_long(list: FreeList, size_class: SizeClassType) {

    }

    fn scavenge() {

    }
}