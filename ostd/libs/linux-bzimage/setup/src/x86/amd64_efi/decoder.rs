// SPDX-License-Identifier: MPL-2.0

//! This module is used to decompress payload.

extern crate alloc;

use core2::io::Read;
use libflate::gzip::Decoder;
pub use alloc::vec::Vec;

/// This function decompresses payload.
pub fn decompress_payload(payload: &[u8]) -> Vec<u8> {
    let mut decoder = Decoder::new(payload).unwrap();
    let mut kernel = Vec::new();
    decoder.read_to_end(&mut kernel).unwrap();
    kernel
}