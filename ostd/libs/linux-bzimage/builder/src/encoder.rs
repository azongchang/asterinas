// SPDX-License-Identifier: MPL-2.0

//! This module is used to compress kernel ELF.

use std::io::Write;

use libflate::*;

pub fn compress_kernel(kernel: &[u8]) -> Vec<u8> {
    match std::env::var("COMPRESSION_FORMAT") {
        Ok(typ) if typ == "gzip" => {
            let mut encoder = gzip::Encoder::new(Vec::new()).unwrap();
            encoder.write_all(kernel).unwrap();
            encoder.finish().into_result().unwrap()
        },
        Ok(typ) if typ == "zlib" => {
            let mut encoder = zlib::Encoder::new(Vec::new()).unwrap();
            encoder.write_all(kernel).unwrap();
            encoder.finish().into_result().unwrap()
        },
        _ => {
            let mut encoder = deflate::Encoder::new(Vec::new());
            encoder.write_all(kernel).unwrap();
            encoder.finish().into_result().unwrap()
        },
    }
}
