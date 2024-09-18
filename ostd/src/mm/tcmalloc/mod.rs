// SPDX-License-Identifier: MPL-2.0

use core::alloc::{GlobalAlloc, Layout};

mod size_class;
pub mod thread_cache;

struct Tcmallocator;

unsafe impl GlobalAlloc for Tcmallocator {
    // TODO: fn `alloc` and `dealloc` should recognize the tid and hand over to thread's allocator.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        todo!();
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        todo!();
    }
}