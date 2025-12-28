#!/usr/bin/env bash
set -euo pipefail

# Run ignored (heavy) tests with conservative environment to limit resource usage
export CARGO_BUILD_JOBS=1
export RUST_TEST_THREADS=1
export RUSTFLAGS='-C codegen-units=1 -C opt-level=0'

echo "Running heavy (ignored) tests with conservative settings..."
cargo test -- --ignored --test-threads=1 --nocapture