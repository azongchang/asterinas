// SPDX-License-Identifier: MPL-2.0

//! This module is used to compress kernel ELF.

use std::{
    ffi::{OsStr, OsString},
    io::Write,
    str::FromStr,
};

use libflate::{deflate, gzip, zlib};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionFormat {
    #[default]
    #[serde(rename = "plain")]
    Plain,
    #[serde(rename = "gzip")]
    Gzip,
    #[serde(rename = "zlib")]
    Zlib,
    #[serde(rename = "deflate")]
    Deflate,
}

impl FromStr for CompressionFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "plain" => Ok(CompressionFormat::Plain),
            "gzip" => Ok(CompressionFormat::Gzip),
            "zlib" => Ok(CompressionFormat::Zlib),
            "deflate" => Ok(CompressionFormat::Deflate),
            _ => Err(format!("Invalid compression format: {}", s)),
        }
    }
}

impl From<OsString> for CompressionFormat {
    fn from(os_string: OsString) -> Self {
        CompressionFormat::from_str(&os_string.to_string_lossy()).unwrap_or_default()
    }
}

impl From<&OsStr> for CompressionFormat {
    fn from(os_str: &OsStr) -> Self {
        CompressionFormat::from_str(&os_str.to_string_lossy()).unwrap_or_default()
    }
}

pub fn compress_kernel(kernel: &[u8], compression_format: CompressionFormat) -> Vec<u8> {
    match compression_format {
        CompressionFormat::Gzip => {
            let mut encoder = gzip::Encoder::new(Vec::new()).unwrap();
            encoder.write_all(kernel).unwrap();
            encoder.finish().into_result().unwrap()
        }
        CompressionFormat::Zlib => {
            let mut encoder = zlib::Encoder::new(Vec::new()).unwrap();
            encoder.write_all(kernel).unwrap();
            encoder.finish().into_result().unwrap()
        }
        CompressionFormat::Deflate => {
            let mut encoder = deflate::Encoder::new(Vec::new());
            encoder.write_all(kernel).unwrap();
            encoder.finish().into_result().unwrap()
        }
        _ => {
            panic!("Unsupported compression format!");
        }
    }
}
