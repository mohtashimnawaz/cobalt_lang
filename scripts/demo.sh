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

echo "1) Building object (requires --features llvm)..."
cargo run --features llvm -- build $EXAMPLE --output $OUT_OBJ

echo "2) Generating C fallback (non-LLVM)"
cargo run -- build $EXAMPLE --output promo.c || true

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
$CC $OUT_OBJ $MAIN_C -o $OUT_EXE

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
