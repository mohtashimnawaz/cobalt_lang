# Cobalt

A small statically-typed language implemented in Rust (experimental).

## Testing

We run a split test strategy so heavy/slow tests don't overload developer machines:

- Fast tests (run locally):

  ```bash
  cargo test
  ```

- Heavy tests (ignored by default, run in CI):

  Run locally with conservative limits to avoid OOM on machines with limited RAM (e.g., 16 GB):

  ```bash
  CARGO_BUILD_JOBS=1 RUST_TEST_THREADS=1 RUSTFLAGS="-C codegen-units=1 -C opt-level=0" cargo test -- --ignored
  ```

- LLVM-enabled tests (require LLVM/clang):

  To run codegen tests locally (if you have LLVM/clang installed):

  ```bash
  # macOS (Homebrew):
  brew install llvm
  # Linux (Ubuntu):
  sudo apt-get update && sudo apt-get install -y clang llvm lld libclang-dev

  # Then run LLVM-enabled tests:
  cargo test --features llvm
  ```

- Helper script:

  A convenience script is provided at `scripts/heavy-test.sh` that runs the conservative heavy test command.

## Contributing

Open a PR and CI will run both fast and heavy tests (including LLVM-enabled job, if applicable). See `.github/workflows/ci.yml` for details.