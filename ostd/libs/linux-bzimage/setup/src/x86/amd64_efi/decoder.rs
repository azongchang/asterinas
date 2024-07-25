// SPDX-License-Identifier: MPL-2.0

//! This module is used to decompress payload.

extern crate alloc;

pub use alloc::vec::Vec;

use core2::io::Read;
use libflate::{deflate, gzip, zlib};

/// Checking the magic number of the payload and matching decompression methods to
/// decompress it. 
pub fn decompress_payload(payload: &[u8]) -> Vec<u8> {
    let mut kernel = Vec::new();
    const GZIP_MAGIC_NUMBER: &[u8] = &[0x1F, 0x8B];
    const ZLIB_MAGIC_NUMBER: &[u8] = &[0x78, 0x9C];
    match &payload[..2] {
        GZIP_MAGIC_NUMBER => {
            let mut decoder = gzip::Decoder::new(payload).unwrap();
            decoder.read_to_end(&mut kernel).unwrap();
        }
        ZLIB_MAGIC_NUMBER => {
            let mut decoder = zlib::Decoder::new(payload).unwrap();
            decoder.read_to_end(&mut kernel).unwrap();
        }
        _ => {
            let mut decoder = deflate::Decoder::new(payload);
            decoder.read_to_end(&mut kernel).unwrap();
        }
    };
    kernel
}
