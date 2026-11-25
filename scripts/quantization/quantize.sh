#!/usr/bin/env bash
# Quantize Dark-Scarlett-v1.0-26B-A4B-Uncensored from the bf16 GGUF down to the
# usual llama.cpp set: Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0.
#
# The source is the bf16 GGUF conversion of
#   https://huggingface.co/rakshazi/Dark-Scarlett-v1.0-26B-A4B-Uncensored
# (`model-bf16.gguf`, 50.5 GB, 658 tensors, 392 of them already F32).
# Quantizing from bf16 (not from an existing quant) keeps every tensor at its
# native precision until the single lossy step.
#
# Usage
#   ./quantize.sh                     # all of the default types
#   ./quantize.sh Q4_K_M Q8_0         # only these
#   FORCE=1 ./quantize.sh Q4_K_M      # overwrite existing output
#   ./quantize.sh --verify Q4_K_M     # quantize, then smoke-test with llama-cli
#
# Environment
#   SRC        input GGUF                (default ./model-bf16.gguf, or $SRC)
#   OUTDIR     output directory          (default ./hf)
#   PREFIX     output basename          (default Dark-Scarlett-v1.0-26B-A4B-Uncensored)
#   LLAMA_BIN  dir containing llama-quantize  (default: autodetect, else download)
#   LLAMA_BUILD release tag to download  (default b12723, the build that produced the bf16 file)
#   THREADS    threads for llama-quantize (default: nproc)
#   NO_HASH=1  skip sha256 of the 50 GB input (saves ~1 min)
#
# Requirements: bash, curl (only if llama.cpp has to be downloaded), tar.
# No python, no pytorch, no cmake: it uses the prebuilt llama.cpp CPU release,
# so this runs on any x86-64 Linux box, CPU only.
set -euo pipefail

DEFAULT_QUANTS=(Q2_K Q3_K_M Q4_K_M Q5_K_M Q6_K Q8_0)
LLAMA_BUILD="${LLAMA_BUILD:-b12723}"
LLAMA_BIN="${LLAMA_BIN:-}"
SRC="${SRC:-./model-bf16.gguf}"
OUTDIR="${OUTDIR:-./hf}"
PREFIX="${PREFIX:-Dark-Scarlett-v1.0-26B-A4B-Uncensored}"
THREADS="${THREADS:-$(nproc)}"
FORCE="${FORCE:-0}"
NO_HASH="${NO_HASH:-0}"

verify=0
quants=()
for arg in "$@"; do
    case "$arg" in
        --verify) verify=1 ;;
        -h|--help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown flag: $arg" >&2; exit 2 ;;
        *) quants+=("$arg") ;;
    esac
done
[ ${#quants[@]} -eq 0 ] && quants=("${DEFAULT_QUANTS[@]}")

say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
hr() { printf '%s\n' "------------------------------------------------------------------------"; }

# ---------------------------------------------------------------- toolchain ---
find_quantize() {
    if [ -n "$LLAMA_BIN" ] && [ -x "$LLAMA_BIN/llama-quantize" ]; then
        echo "$LLAMA_BIN/llama-quantize"; return 0
    fi
    if command -v llama-quantize >/dev/null 2>&1; then
        command -v llama-quantize; return 0
    fi
    return 1
}

download_llama() {
    local dir=".llama.cpp/$LLAMA_BUILD" tgz="llama-$LLAMA_BUILD-bin-ubuntu-x64.tar.gz"
    local url="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD/$tgz"
    if [ -x "$dir/llama-quantize" ]; then
        echo "$dir/llama-quantize"; return 0
    fi
    say "no llama-quantize found, downloading the $LLAMA_BUILD CPU release (~24 MB)" >&2
    mkdir -p ".llama.cpp"
    curl -fL --retry 3 -o ".llama.cpp/$tgz" "$url" >&2
    tar -xzf ".llama.cpp/$tgz" -C ".llama.cpp" >&2
    rm -f ".llama.cpp/$tgz"
    # the tarball unpacks to llama-$LLAMA_BUILD/
    if [ -x "$dir/llama-quantize" ]; then echo "$dir/llama-quantize"; else
        echo "download did not produce $dir/llama-quantize" >&2; return 1
    fi
}

QUANTIZE="$(find_quantize || download_llama)"
LLAMA_DIR="$(cd "$(dirname "$QUANTIZE")" && pwd)"
export LD_LIBRARY_PATH="$LLAMA_DIR:${LD_LIBRARY_PATH:-}"

# -------------------------------------------------------------------- input ---
[ -f "$SRC" ] || { echo "missing input: $SRC (set SRC=...)" >&2; exit 1; }
mkdir -p "$OUTDIR"

hr
say "llama-quantize: $QUANTIZE"
say "input:          $SRC ($(du -h --apparent-size "$SRC" | cut -f1))"
say "output dir:     $OUTDIR"
say "threads:        $THREADS"
say "types:          ${quants[*]}"
if [ "$NO_HASH" != "1" ]; then
    say "sha256(input):  $(sha256sum "$SRC" | cut -d' ' -f1)  (NO_HASH=1 to skip)"
fi
hr

# ------------------------------------------------------------------- convert ---
results=()
for q in "${quants[@]}"; do
    out="$OUTDIR/$PREFIX-$q.gguf"
    if [ -f "$out" ] && [ "$FORCE" != "1" ]; then
        say "skip $q (exists: $out, FORCE=1 to redo)"
        results+=("$q $(stat -c %s "$out")")
        continue
    fi
    log="$OUTDIR/.quantize-$q.log"
    say "quantizing $q -> $out"
    t0=$SECONDS
    "$LLAMA_DIR/llama-quantize" --allow-requantize "$SRC" "$out" "$q" "$THREADS" >"$log" 2>&1 || {
        echo "FAILED: $q, log follows" >&2; tail -20 "$log" >&2; exit 1; }
    elapsed=$((SECONDS - t0))
    size=$(stat -c %s "$out")
    bpw=$(awk -v s="$size" 'BEGIN{printf "%.2f", s*8/25.234e9}')
    fallbacks=$(grep -o 'WARNING: [0-9]* of [0-9]* tensor(s)' "$log" | head -1 || true)
    say "done $q in ${elapsed}s, $(du -h --apparent-size "$out" | cut -f1), ${bpw} bpw over 25.23B params${fallbacks:+, $fallbacks fell back to a higher type}"
    results+=("$q $size")
    rm -f "$log"
done

# -------------------------------------------------------------------- report ---
hr
say "outputs in $OUTDIR"
printf '%-8s %12s %10s\n' TYPE BYTES GiB
for r in "${results[@]}"; do
    set -- $r
    printf '%-8s %12s %10.2f\n' "$1" "$2" "$(awk -v b="$2" 'BEGIN{print b/1024/1024/1024}')"
done

if [ "$NO_HASH" != "1" ]; then
    ( cd "$OUTDIR" && for r in "${results[@]}"; do set -- $r; echo "$PREFIX-$1.gguf"; done \
        | xargs -r sha256sum ) > "$OUTDIR/SHA256SUMS.txt"
    say "wrote $OUTDIR/SHA256SUMS.txt"
fi

if [ "$verify" = "1" ]; then
    CLI="$LLAMA_DIR/llama-cli"
    for r in "${results[@]}"; do
        set -- $r
        [ -x "$CLI" ] || { say "no llama-cli next to llama-quantize, skipping verify"; break; }
        say "verify $1: load, chat template, 64 greedy tokens"
        "$CLI" -m "$OUTDIR/$PREFIX-$1.gguf" -p "Name three prime numbers under 10. Answer with just the numbers." \
            -n 64 --temp 0 -st --no-display-prompt -co off > "$OUTDIR/.verify.out" 2> "$OUTDIR/.verify.err"
        ans=$(awk '/^> /{f = 1; next} /^\[ Prompt: /{f = 0} f' "$OUTDIR/.verify.out" \
              | tr -d '\r' | tr '\n' ' ' | sed 's/\[Start thinking\]//; s/  */ /g; s/^ //; s/ $//')
        tps=$(sed -n 's/.*Generation: *\([0-9][0-9.]*\) t\/s.*/\1/p' "$OUTDIR/.verify.out" | tail -1)
        if [ -n "$ans" ]; then
            say "  ok${tps:+, $tps tok/s}: $(printf '%s' "$ans" | cut -c1-100)"
        else
            say "  no output, see $OUTDIR/.verify.err"
        fi
        rm -f "$OUTDIR/.verify.out" "$OUTDIR/.verify.err"
    done
fi
hr
say "all done"
