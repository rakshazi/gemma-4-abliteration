#!/usr/bin/env bash
# Host-side test suite for the Dark-Scarlett GGUF quants.
#
# Quantizing needs no GPU, but this part does, so run it next to the .gguf files
# on the machine that has one:
#
#   cd <dir with the .gguf files>
#   ./test-quants.sh                 # every quant: bench, chat smoke test, MTP A/B
#   QUANTS="Q4_K_M" ./test-quants.sh # just one
#   IMAGE=photo.jpg ./test-quants.sh # also exercise the vision tower
#   MTP=0 ./test-quants.sh           # skip the draft-mtp comparison
#
# Needs llama-bench from llama.cpp on PATH. Everything else is used when present:
# llama-server + curl + python3 (preferred, gives clean JSON answers and exact
# timings), llama-completion as a fallback, llama-mtmd-cli for the vision test.
#
# Output: a summary on stdout, logs in .test/, and a paste-ready markdown table
# in .test/test-report.md with throughput per quant and the machine it ran on.
#
# Environment
#   QUANTS     space separated list to test (default: every -Q*.gguf found)
#   PREFIX     file basename (default Dark-Scarlett-v1.0-26B-A4B-Uncensored)
#   NGL        gpu layers for bench/server/cli (default 999 = offload everything)
#   NCMOE      -ncmoe for the MoE experts (default 0; raise it if VRAM is short)
#   REP        llama-bench repetitions (default 2)
#   PP / TG    llama-bench prefill / gen sizes (default 512 / 128)
#   CTX        context for the smoke test (default 2048)
#   NTEST      tokens per smoke completion (default 96)
#   MTP_N      tokens for each side of the MTP A/B (default 128)
#   MTP_PROMPT prompt for the MTP A/B
#   IMAGE      optional image path for the llama-mtmd-cli test
#   VUB        -ub for the vision test (default 1024)
#   MTP        1 to run the speculative decoding comparison (default 1)
#   MTP_FILE   MTP gguf (default mtp-gemma-4-26B-A4B-it-Q4_0.gguf)
#   MMPROJ     mmproj gguf (default mmproj-Dark-Scarlett-v1.0-26B-A4B-Uncensored-BF16.gguf)
set -uo pipefail

PREFIX="${PREFIX:-Dark-Scarlett-v1.0-26B-A4B-Uncensored}"
NGL="${NGL:-999}"
NCMOE="${NCMOE:-0}"
REP="${REP:-2}"
PP="${PP:-512}"
TG="${TG:-128}"
CTX="${CTX:-2048}"
NTEST="${NTEST:-96}"
MTP_N="${MTP_N:-128}"
MTP_PROMPT="${MTP_PROMPT:-Write a short paragraph about the sea at night.}"
VUB="${VUB:-1024}"
MTP="${MTP:-1}"
MTP_FILE="${MTP_FILE:-mtp-gemma-4-26B-A4B-it-Q4_0.gguf}"
MMPROJ="${MMPROJ:-mmproj-Dark-Scarlett-v1.0-26B-A4B-Uncensored-BF16.gguf}"
OUT=".test"
mkdir -p "$OUT"

say() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
fail=0
note() { fail=$((fail + 1)); say "FAIL: $*"; }
have() { command -v "$1" >/dev/null 2>&1; }
# numbers come out as "44.5" or "44,5" depending on the locale
num() { printf '%s' "${1:-0}" | tr -d ' ' | tr ',' '.'; }
round1() { awk -v v="${1:-0}" 'BEGIN{printf "%.1f", v}'; }

# ------------------------------------------------------------------ toolchain --
have llama-bench || { echo "missing llama-bench in PATH (put the llama.cpp release dir on PATH)" >&2; exit 1; }
use_server=0
if have llama-server && have curl && have python3; then use_server=1; fi
if [ "$use_server" = "1" ]; then
    say "smoke tests and the MTP A/B go through llama-server (clean json, exact timings)"
else
    say "no llama-server+curl+python3 here, falling back to llama-completion for the smoke test"
    have llama-completion || { echo "missing llama-completion in PATH as well" >&2; exit 1; }
fi
have llama-mtmd-cli || say "note: no llama-mtmd-cli in PATH, the vision test will be skipped"

# jget KEY  <- json on stdin, prints the value or nothing
jget() {
    python3 -c 'import json,sys
d = json.load(sys.stdin)
for k in sys.argv[1].split("."):
    d = d[int(k)] if isinstance(d, list) else (d or {}).get(k)
    if d is None:
        print("")
        raise SystemExit(0)
print(d)' "$1" 2>/dev/null
}

server_start() { # $1 = model, $2 = tag, rest = extra args -> sets SRV_PID, SRV_PORT
    local m="$1" tag="$2"; shift 2
    local try
    for try in 1 2 3; do
        SRV_PORT=$(( 30000 + RANDOM % 20000 ))
        llama-server -m "$m" -c "$CTX" --jinja -ngl "$NGL" -ncmoe "$NCMOE" \
            --host 127.0.0.1 --port "$SRV_PORT" "$@" > "$OUT/server-$tag.log" 2>&1 &
        SRV_PID=$!
        local i
        for i in $(seq 1 240); do
            if curl -sf "http://127.0.0.1:$SRV_PORT/health" >/dev/null 2>&1; then return 0; fi
            kill -0 "$SRV_PID" 2>/dev/null || break
            sleep 1
        done
        kill "$SRV_PID" 2>/dev/null
        wait "$SRV_PID" 2>/dev/null
        say "server on port $SRV_PORT did not come up, retrying (see $OUT/server-$tag.log)"
    done
    return 1
}
server_stop() { kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; }

srv_ask() { # $1 = prompt, $2 = max tokens -> sets ANS, ANS_TPS, ANS_PPS
    local body resp
    body=$(python3 -c 'import json,sys
print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]}],"temperature":0,"max_tokens":int(sys.argv[2])}))' "$1" "$2")
    resp=$(curl -sf "http://127.0.0.1:$SRV_PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
   ANS=$(printf '%s' "$resp" | jget choices.0.message.content)
    [ -n "$ANS" ] || ANS=$(printf '%s' "$resp" | jget choices.0.message.reasoning_content)
    ANS=$(printf '%s' "$ANS" | tr '\n' ' ' | sed 's/  */ /g; s/^ //; s/ $//')
    ANS_TPS=$(round1 "$(num "$(printf '%s' "$resp" | jget timings.predicted_per_second)")")
    ANS_PPS=$(round1 "$(num "$(printf '%s' "$resp" | jget timings.prompt_per_second)")")
}

# --------------------------------------------------------------------- inputs --
quants=()
if [ -n "${QUANTS:-}" ]; then
    for q in $QUANTS; do quants+=("$q"); done
else
    while IFS= read -r f; do
        b=$(basename "$f" .gguf)
        quants+=("${b#"$PREFIX"-}")
    done < <(ls -1 "$PREFIX"-Q*.gguf 2>/dev/null | sort -r)
fi
[ ${#quants[@]} -eq 0 ] && { echo "no $PREFIX-Q*.gguf files here" >&2; exit 1; }

best="${quants[0]}"
for q in "${quants[@]}"; do [ "$q" = "Q4_K_M" ] && best="Q4_K_M"; done

# ----------------------------------------------------------------------- host --
{
    echo "### run $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "| | |"
    echo "|---|---|"
    echo "| llama.cpp | $(llama-bench --version 2>&1 | grep -iE 'version|build' | head -1 | cut -c1-100) |"
    echo "| cpu | $(awk -F: '/model name/{print $2; exit}' /proc/cpuinfo | sed 's/^ //') |"
    echo "| cores | $(nproc) |"
    echo "| offload | ngl=$NGL ncmoe=$NCMOE |"
    echo "| devices | $(llama-bench --list-devices 2>&1 | grep -vE '^load_backend|^$' | tr '\n' ' ' | cut -c1-140) |"
} > "$OUT/host.md"
cat "$OUT/host.md"

: > "$OUT/results.tsv"

# ---------------------------------------------------------------------- bench --
say "benchmarking ${#quants[@]} quant(s): ${quants[*]}"
for q in "${quants[@]}"; do
    m="$PREFIX-$q.gguf"
    [ -f "$m" ] || { note "$m missing"; continue; }
    size_gib=$(awk -v b="$(stat -c %s "$m")" 'BEGIN{printf "%.1f", b/1024/1024/1024}')
    say "llama-bench $q (${size_gib} GiB), $REP rep(s)"
    llama-bench -m "$m" -ngl "$NGL" -ncmoe "$NCMOE" -p "$PP" -n "$TG" -r "$REP" -o md \
        > "$OUT/bench-$q.md" 2> "$OUT/bench-$q.err"
    if [ $? -ne 0 ]; then
        note "llama-bench $q (see $OUT/bench-$q.err)"
        echo "$q|$size_gib|fail|fail" >> "$OUT/results.tsv"
        continue
    fi
    col() { # $1 = test name such as pp512 / tg128, $2 = llama-bench md file
        awk -F'|' -v t="$1" '{for (i = 1; i <= NF; i++) if ($i ~ "^ *" t " *$") {n = split($(i+1), a, " "); print a[1]; exit}}' "$2"
    }
    pp=$(num "$(col "pp$PP" "$OUT/bench-$q.md")")
    tg=$(num "$(col "tg$TG" "$OUT/bench-$q.md")")
    echo "$q|$size_gib|${pp:-fail}|${tg:-fail}" >> "$OUT/results.tsv"
done

# --------------------------------------------------------------- text smoke ---
say "smoke test: chat template, $NTEST greedy tokens, 'three primes under 10'"
for q in "${quants[@]}"; do
    m="$PREFIX-$q.gguf"
    [ -f "$m" ] || continue
    ans=""; sgen=""; spare=""
    if [ "$use_server" = "1" ]; then
        if server_start "$m" "smoke-$q"; then
            srv_ask "Name three prime numbers under 10. Answer with just the numbers." "$NTEST"
            ans="$ANS"; sgen="$ANS_TPS"; spare="$ANS_PPS"
            server_stop
        else
            note "$q server did not start"
        fi
    else
        llama-completion -m "$m" -ngl "$NGL" -ncmoe "$NCMOE" -c "$CTX" -n "$NTEST" \
            --temp 0 -no-cnv -p "The first three prime numbers are" \
            > "$OUT/cli-$q.out" 2> "$OUT/cli-$q.err"
        ans=$(tr -d '\r' < "$OUT/cli-$q.out" | tr '\n' ' ' | sed 's/  */ /g; s/^ //; s/ $//')
    fi
    if [ -z "$ans" ]; then
        note "$q produced no text (see $OUT/server-smoke-$q.log or $OUT/cli-$q.err)"
        ans="(no output)"
    elif printf '%s' "$ans" | grep -qE '2[^0-9]{0,10}3'; then
        say "$q ok, ${sgen:-?} tok/s: '${ans:0:70}'"
    else
        say "$q loaded, answer looks off (${sgen:-?} tok/s): '${ans:0:70}'"
    fi
    printf '%s|%s|%s|%s\n' "$sgen" "$spare" "$ans" > "$OUT/smoke-$q.txt"
done

# ------------------------------------------------------------------- vision ---
if [ -n "${IMAGE:-}" ] && have llama-mtmd-cli && [ -f "$MMPROJ" ]; then
    say "vision test: $best + $MMPROJ + $IMAGE"
    llama-mtmd-cli -m "$PREFIX-$best.gguf" -ngl "$NGL" --mmproj "$MMPROJ" \
        --image "$IMAGE" -p "Describe this image in one sentence." -n 64 -ub "$VUB" --jinja \
        > "$OUT/vision.out" 2> "$OUT/vision.err"
    vout=$(tr -d '\r' < "$OUT/vision.out" | tr '\n' ' ')
    if [ -n "${vout// /}" ] && ! grep -qE '^error|invalid argument' "$OUT/vision.err"; then
        say "vision ok: '$(printf '%s' "$vout" | cut -c1-100)'"
    else
        note "vision produced no usable text (see $OUT/vision.err)"
    fi
elif [ -z "${IMAGE:-}" ]; then
    say "vision test skipped (set IMAGE=path/to.jpg)"
fi

# ---------------------------------------------------------------------- mtp ---
if [ "$MTP" = "1" ] && [ -f "$MTP_FILE" ] && [ "$use_server" = "1" ]; then
    say "MTP A/B on $best: same prompt and token budget with and without draft-mtp"
    plain=""; draft=""
    if server_start "$PREFIX-$best.gguf" "mtp-plain"; then
        srv_ask "$MTP_PROMPT" "$MTP_N"; plain="$ANS_TPS"; server_stop
    else note "the plain MTP baseline server did not start"; fi
    if server_start "$PREFIX-$best.gguf" "mtp-draft" \
            -md "$MTP_FILE" --spec-type draft-mtp --spec-draft-n-max 3; then
        srv_ask "$MTP_PROMPT" "$MTP_N"; draft="$ANS_TPS"; server_stop
    else note "the draft-mtp server did not start (does this build support -md/--spec-type?)"; fi
    say "generation tok/s: plain ${plain:-?}, draft-mtp ${draft:-?}"
    if awk -v a="${plain:-0}" -v b="${draft:-0}" 'BEGIN{exit !(a > 0 && b > 0)}'; then
        gain=$(awk -v a="$plain" -v b="$draft" 'BEGIN{printf "%+.0f%%", 100*(b-a)/a}')
        say "draft-mtp is $gain against plain"
    fi
    { echo "| generation, $MTP_N tokens | tok/s |"; echo "|---|---|"
      echo "| plain | ${plain:-n/a} |"; echo "| draft-mtp | ${draft:-n/a} |"; } > "$OUT/mtp.md"
elif [ "$MTP" = "1" ] && [ ! -f "$MTP_FILE" ]; then
    say "MTP A/B skipped, $MTP_FILE is not here"
fi

# ------------------------------------------------------------------- report ---
{
    cat "$OUT/host.md"
    echo
    echo "| quant | file GiB | llama-bench prefill $PP | llama-bench gen $TG | server gen $NTEST | answer |"
    echo "|---|---|---|---|---|---|"
    while IFS='|' read -r q s pp tg; do
        sgen=""; ans="-"
        if [ -f "$OUT/smoke-$q.txt" ]; then
            sgen=$(cut -d'|' -f1 "$OUT/smoke-$q.txt")
            ans=$(cut -d'|' -f3- "$OUT/smoke-$q.txt" | cut -c1-60)
        fi
        echo "| $q | $s | ${pp:-fail} | ${tg:-fail} | ${sgen:--} | ${ans} |"
    done < "$OUT/results.tsv"
    echo
    if [ -s "$OUT/mtp.md" ]; then cat "$OUT/mtp.md"; echo; fi
    echo "failures: $fail"
} > "$OUT/test-report.md"

say "wrote $OUT/test-report.md"
cat "$OUT/test-report.md"
[ "$fail" -eq 0 ] || { say "$fail check(s) failed"; exit 1; }
say "all checks passed"
