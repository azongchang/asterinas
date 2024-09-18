// SPDX-License-Identifier: MPL-2.0

use crate::mm::PAGE_SIZE;

struct SizeClassInfo {
    // Max size storable in that class
    size: u32,

    // Number of pages to allocate at a time
    pages: u8,

    // Number of objects to move between a per-thread list and a central list in
    // one shot.  We want this to be not too small so we can amortize the lock
    // overhead for accessing the central list.  Making it too big may temporarily
    // cause unnecessary memory wastage in the per-thread free list until the
    // scavenger cleans up the list.
    num_to_move: u8,

    // Max per-CPU slab capacity for the default 256KB slab size.
    // Scaled up/down for larger/smaller slab sizes.
    max_capacity: u32,
}

impl SizeClassInfo {
    fn new(size: u32, pages: u8, num_to_move: u8, max_capacity: u32) -> Self {
        SizeClassInfo {
            size,
            pages,
            num_to_move,
            max_capacity,
        }
    }
}

struct SizeClassAssumptions {
    has_expanded_classes: bool,     // kHasExpandedClasses
    span_size: usize,               // sizeof(Span)
    sampling_interval: usize,       // kDefaultProfileSamplingInterval
    large_size: usize,              // SizeMap::kLargeSize
    large_size_alignment: usize     // SizeMap::kLargeSizeAlignment
}

struct SizeClasses {
    assumptions: SizeClassAssumptions,
    // TODO: Use a proper type to store size classes.
    // BTreeMap may be a desirable option.
    classes: Span<SizeClassInfo>,
}

// FIXME: Select a group of size classes based on page size.
#[cfg(PAGE_SIZE = "4096")]
static_assertions::const_assert_eq!(kMaxSize, 8192);
#[cfg(PAGE_SIZE = "4096")]
// TODO: To organize size classes properly.
static SIZE_CLASSES: SizeClasses = SizeClasses {
    assumptions: SizeClassAssumptions {
        has_expanded_classes: false,
        span_size: 48,
        sampling_interval: 524288,
        large_size: 1024,
        large_size_alignment: 128,
    },
    // TODO: Use a macro to fetch size classes.
    classes: Span<SizeClassInfo>::new(
    //                                                                                                   |    waste     |
    //                           bytes        pages             batch                 cap    class  objs |fixed sampling|    inc
    //  SizeClassInfo::new(          0,           0,                0,                  0),  //  0     0  0.00%    0.00%   0.00%
        SizeClassInfo::new(     8,    1,   32, 5811),  //  0   512  1.16%    0.92%   0.00%
        SizeClassInfo::new(    16,    1,   32, 5811),  //  1   256  1.16%    0.92% 100.00%
        SizeClassInfo::new(    32,    1,   32, 5811),  //  2   128  1.16%    0.92% 100.00%
        SizeClassInfo::new(    64,    1,   32, 5811),  //  3    64  1.16%    0.92% 100.00%
        SizeClassInfo::new(    80,    1,   32, 5811),  //  4    51  1.54%    0.92%  25.00%
        SizeClassInfo::new(    96,    1,   32, 3615),  //  5    42  2.70%    0.92%  20.00%
        SizeClassInfo::new(   112,    1,   32, 2468),  //  6    36  2.70%    0.92%  16.67%
        SizeClassInfo::new(   128,    1,   32, 2667),  //  7    32  1.16%    0.92%  14.29%
        SizeClassInfo::new(   144,    1,   32, 2037),  //  8    28  2.70%    0.92%  12.50%
        SizeClassInfo::new(   160,    1,   32, 2017),  //  9    25  3.47%    0.92%  11.11%
        SizeClassInfo::new(   176,    1,   32,  973),  // 10    23  2.32%    0.92%  10.00%
        SizeClassInfo::new(   192,    1,   32,  999),  // 11    21  2.70%    0.92%   9.09%
        SizeClassInfo::new(   208,    1,   32,  885),  // 12    19  4.63%    0.92%   8.33%
        SizeClassInfo::new(   224,    1,   32,  820),  // 13    18  2.70%    0.92%   7.69%
        SizeClassInfo::new(   240,    1,   32,  800),  // 14    17  1.54%    0.92%   7.14%
        SizeClassInfo::new(   256,    1,   32, 1226),  // 15    16  1.16%    0.92%   6.67%
        SizeClassInfo::new(   272,    1,   32,  582),  // 16    15  1.54%    0.92%   6.25%
        SizeClassInfo::new(   288,    1,   32,  502),  // 17    14  2.70%    0.92%   5.88%
        SizeClassInfo::new(   304,    1,   32,  460),  // 18    13  4.63%    0.92%   5.56%
        SizeClassInfo::new(   336,    1,   32,  854),  // 19    12  2.70%    0.92%  10.53%
        SizeClassInfo::new(   368,    1,   32,  485),  // 20    11  2.32%    0.92%   9.52%
        SizeClassInfo::new(   448,    1,   32,  559),  // 21     9  2.70%    0.92%  21.74%
        SizeClassInfo::new(   512,    1,   32, 1370),  // 22     8  1.16%    0.92%  14.29%
        SizeClassInfo::new(   576,    2,   32,  684),  // 23    14  2.14%    0.92%  12.50%
        SizeClassInfo::new(   640,    2,   32,  403),  // 24    12  6.80%    0.92%  11.11%
        SizeClassInfo::new(   704,    2,   32,  389),  // 25    11  6.02%    0.92%  10.00%
        SizeClassInfo::new(   768,    2,   32,  497),  // 26    10  6.80%    0.93%   9.09%
        SizeClassInfo::new(   896,    2,   32,  721),  // 27     9  2.14%    0.92%  16.67%
        SizeClassInfo::new(  1024,    2,   32, 3115),  // 28     8  0.58%    0.92%  14.29%
        SizeClassInfo::new(  1152,    3,   32,  451),  // 29    10  6.61%    0.93%  12.50%
        SizeClassInfo::new(  1280,    3,   32,  372),  // 30     9  6.61%    0.93%  11.11%
        SizeClassInfo::new(  1536,    3,   32,  420),  // 31     8  0.39%    0.92%  20.00%
        SizeClassInfo::new(  1792,    4,   32,  406),  // 32     9  1.85%    0.92%  16.67%
        SizeClassInfo::new(  2048,    4,   32,  562),  // 33     8  0.29%    0.92%  14.29%
        SizeClassInfo::new(  2304,    4,   28,  380),  // 34     7  1.85%    0.92%  12.50%
        SizeClassInfo::new(  2688,    4,   24,  394),  // 35     6  1.85%    0.93%  16.67%
        SizeClassInfo::new(  3200,    4,   20,  389),  // 36     5  2.63%    0.93%  19.05%
        SizeClassInfo::new(  3584,    7,   18,  409),  // 37     8  0.17%    0.92%  12.00%
        SizeClassInfo::new(  4096,    4,   16, 1430),  // 38     4  0.29%    0.92%  14.29%
        SizeClassInfo::new(  4736,    5,   13,  440),  // 39     4  7.72%    1.77%  15.62%
        SizeClassInfo::new(  5376,    4,   12,  361),  // 40     3  1.85%    1.72%  13.51%
        SizeClassInfo::new(  6144,    3,   10,  369),  // 41     2  0.39%    1.70%  14.29%
        SizeClassInfo::new(  7168,    7,    9,  377),  // 42     4  0.17%    1.70%  16.67%
        SizeClassInfo::new(  8192,    4,    8,  505),  // 43     2  0.29%    1.70%  14.29%
    )
};