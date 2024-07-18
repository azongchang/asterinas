// SPDX-License-Identifier: MPL-2.0

//! This module is used to compress kernel EFI.

use std::io::Write;
use libflate::gzip::Encoder;

/// This function compresses kernel EFI.
pub fn compress_kernel(kernel: &[u8]) -> Vec<u8> {
    let mut encoder = Encoder::new(Vec::new()).unwrap();
    encoder.write_all(kernel).unwrap();
    encoder.finish().into_result().unwrap()
}