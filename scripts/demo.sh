#!/usr/bin/env bash
set -euo pipefail

# Demo script: build, link and run the example program
# - Builds object with llvm feature (requires clang/llvm installed)
# - Also generates textual IR (non-llvm) so you can inspect the IR

EXAMPLE=examples/promo.cobalt
OUT_OBJ=promo.o
OUT_LL=promo.ll
OUT_EXE=promo_run
MAIN_C=main.c

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

echo "1) Generating C fallback (non-LLVM)"
# Use conservative build settings to avoid OOM on low-RAM machines
CARGO_BUILD_JOBS=1 RUST_TEST_THREADS=1 RUSTFLAGS='-C codegen-units=1 -C opt-level=0' cargo run -- build $EXAMPLE --output promo.c || true

# Optional: build with LLVM if you want (uncomment the next lines)
# echo "(optional) Build with LLVM-enabled codegen (requires --features llvm)"
# cargo run --features llvm -- build $EXAMPLE --output $OUT_OBJ || true

# If the cargo build didn't generate promo.c, fall back to a small generator
if [[ ! -f promo.c ]]; then
  echo "promo.c not found, running small Python generator as fallback"
  python3 scripts/generate_c_from_example.py
fi

# Write a small C main that calls to_int()
cat > $MAIN_C <<'C'
#include <stdio.h>
extern int to_int();
extern float to_f();
int main(){
    int v = to_int();
    float fv = to_f();
    printf("to_int() -> %d\n", v);
    printf("to_f() -> %f\n", fv);
    return v;
}
C

echo "3) Linking with clang (requires clang available)"
CC=${CC:-clang}
# Prefer compiling promo.c directly if object wasn't produced
if [[ -f promo.o ]]; then
  $CC promo.o $MAIN_C -o $OUT_EXE
else
  $CC -x c promo.c $MAIN_C -o $OUT_EXE
fi

echo "4) Running the executable"
./$OUT_EXE
RC=$?

echo "exit code: $RC"

echo "\n=== Show snippet of IR (first 200 lines) ==="
if [[ -f $OUT_LL ]]; then
  head -n 200 $OUT_LL
else
  echo "IR not produced (non-llvm code path may be disabled)."
fi

echo "Demo complete."
