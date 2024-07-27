// SPDX-License-Identifier: MPL-2.0

//! This module is used to compress kernel ELF.

use std::{
    fs::read_to_string,
    io::Write,
};

use libflate::{deflate, gzip, zlib};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TomlManifest {
    config: Config,
}

#[derive(Debug, Deserialize)]
struct Config {
    run: RunConfig,
}

#[derive(Debug, Deserialize)]
struct RunConfig {
    build: BuildConfig,
}

#[derive(Debug, Deserialize)]
struct BuildConfig {
    compression_format: String,
}

pub fn compress_kernel(kernel: &[u8]) -> Vec<u8> {
    let mut current_dir = std::env::current_dir().unwrap();
    current_dir.pop();
    current_dir.push("asterinas/bundle.toml");
    let toml_str = read_to_string(current_dir).unwrap();
    let config: TomlManifest = toml::from_str(&toml_str).unwrap();
    match config.config.run.build.compression_format {
        typ if typ == "gzip" => {
            let mut encoder = gzip::Encoder::new(Vec::new()).unwrap();
            encoder.write_all(kernel).unwrap();
            encoder.finish().into_result().unwrap()
        }
        typ if typ == "zlib" => {
            let mut encoder = zlib::Encoder::new(Vec::new()).unwrap();
            encoder.write_all(kernel).unwrap();
            encoder.finish().into_result().unwrap()
        }
        typ if typ == "deflate" => {
            let mut encoder = deflate::Encoder::new(Vec::new());
            encoder.write_all(kernel).unwrap();
            encoder.finish().into_result().unwrap()
        }
        _ => {
            panic!("Unsupported compression format!");
        }
    }
}
