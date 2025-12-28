# Contributing & Developer Setup

Thanks for contributing! This document covers setting up LLVM locally (required for LLVM-enabled codegen tests) and running the test matrix.

## Install LLVM/Clang

### macOS (Homebrew)

1. Install LLVM:

   brew install llvm

2. Make Homebrew's LLVM available on your PATH for the current shell:

   export PATH="$(brew --prefix llvm)/bin:$PATH"

   To persist, add the export command to your shell profile (e.g., `~/.zshrc`).

### Ubuntu / Debian

1. Install system packages:

   sudo apt-get update
   sudo apt-get install -y clang llvm lld libclang-dev

## Running tests

- Fast tests (default):

  ```bash
  cargo test
  ```

- Heavy/ignored tests (conservative):

  ```bash
  CARGO_BUILD_JOBS=1 RUST_TEST_THREADS=1 RUSTFLAGS="-C codegen-units=1 -C opt-level=0" cargo test -- --ignored
  ```

- LLVM-enabled tests (requires LLVM/Clang available on PATH):

  ```bash
  cargo test --features llvm
  ```

You can also run the helper script `scripts/run-llvm-tests.sh` which attempts to add Homebrew's LLVM to PATH (on macOS) and runs the LLVM-enabled tests conservatively.

## Notes

- CI runs the LLVM-enabled tests on Ubuntu and macOS (stable and nightly toolchains) and uploads logs on failure.
- If you have trouble building `inkwell`, ensure `clang` and `libclang` are installed and available on your system.
