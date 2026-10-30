#!/bin/sh
# llguidance C FFI as static lib (crate llguidance via llg/, header copied to Sources/CLLG). Run before ./build.sh.
cd "$(dirname "$0")/../llg" && cargo build --release && cp "$(ls -d ~/.cargo/registry/src/*/llguidance-1.9.1 | head -1)/llguidance.h" ../Sources/CLLG/
