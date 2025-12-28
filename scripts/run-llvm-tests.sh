#!/usr/bin/env bash
set -euo pipefail

# Try to add Homebrew LLVM to PATH on macOS
if [[ "$(uname)" == "Darwin" ]]; then
  if command -v brew >/dev/null 2>&1; then
    prefix=$(brew --prefix llvm 2>/dev/null || true)
    if [[ -n "$prefix" ]]; then
      export PATH="$prefix/bin:$PATH"
      echo "Added $prefix/bin to PATH"
    fi
  fi
fi

export CARGO_BUILD_JOBS=1
export RUST_TEST_THREADS=1
export RUSTFLAGS='-C codegen-units=1 -C opt-level=0'

echo "Running LLVM-enabled tests..."
cargo test --features llvm -- --test-threads=1 --nocapture
