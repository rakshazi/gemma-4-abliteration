#!/usr/bin/env bash
# Pipeline driver for Gemma 4 abliteration. Run ./run.sh <command> --help for details.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-}"
MODEL="${MODEL:-ReadyArt/Dark-Scarlett-v1.0-26B-A4B}"
PREBAKED="${PREBAKED:-models/ega-experts-26b}"
FINAL="${FINAL:-models/ega-all-heretic-t1}"
DATASETS="${DATASETS:-mlabonne}"

heretic() {
  # argv[0] must not stay "-c" or heretic's own -c option swallows the next flag
  HERETIC_CONFIG_FILE="$1" "$PY" -c 'import sys; sys.argv[0] = "heretic"; from heretic.main import main; main()' "${@:2}"
}

usage() {
  cat <<'EOF'
Usage: ./run.sh <command> [extra args]

Commands:
  smoke              Tiny-model end-to-end test on CPU (~1 min, no GPU, no 26B)
  gpu-smoke          Real-model smoke: 1-layer prebake dry run + 2 heretic trials
  prebake            EGA-prebake the MoE experts only (recommended stage 1)
  prebake-all        One-shot EGA: dense + experts, no heretic run needed
  heretic            Run the production heretic study (config.toml, 8-10 h)
  heretic-baseline   Optional A/B: dense-only heretic on the original model
  heretic-stack      Third run: heretic search on top of the EGA dense+experts prebake (6 trials)
  eval               Refusal rates + KL for original vs prebaked vs final
  status             Show models, reports and completed trial counts

Environment overrides: PY, MODEL, PREBAKED, FINAL, DATASETS.
EOF
}

resolve_python() {
  [ -n "$PY" ] && return
  if [ -x venv/bin/python ]; then
    PY="venv/bin/python"
  elif [ -x .venv/bin/python ]; then
    PY=".venv/bin/python"
  else
    echo "No virtual environment found. Create one and install requirements.txt (see README)." >&2
    exit 1
  fi
}

case "${1:-}" in
  ""|-h|--help|help)
    usage
    exit 0
    ;;
esac

resolve_python

case "${1:-}" in
  smoke)
    shift
    "$PY" scripts/smoke_test.py "$@"
    ;;
  gpu-smoke)
    shift
    "$PY" scripts/ega_prebake.py --model "$MODEL" --config config.toml \
      --parts all --layers 0 --max-prompts 4 --batch-size 4 --dry-run "$@"
    heretic config.smoke-26b.toml --model "$MODEL" "$@"
    ;;
  prebake)
    shift
    "$PY" scripts/ega_prebake.py --model "$MODEL" --config config.toml \
      --output "$PREBAKED" --parts experts "$@"
    ;;
  prebake-all)
    shift
    "$PY" scripts/ega_prebake.py --model "$MODEL" --config config.toml \
      --output "${ALL_ONE_SHOT:-models/ega-all-26b}" --parts all --show 3 "$@"
    ;;
  heretic)
    shift
    heretic config.toml "$@"
    ;;
  heretic-baseline)
    shift
    heretic config.dense-baseline.toml "$@"
    ;;
  heretic-stack)
    shift
    heretic config.ega-all-heretic.toml "$@"
    ;;
  eval)
    shift
    models=("$PREBAKED")
    if [ -d "$FINAL" ]; then
      models+=("$FINAL")
    fi
    "$PY" scripts/eval_models.py --baseline-model "$MODEL" \
      --models "${models[@]}" --datasets "$DATASETS" \
      --json reports/eval.json "$@"
    ;;
  status)
    found=0
    for path in "$PREBAKED" "$FINAL" models/ega-all-26b; do
      if [ -d "$path" ]; then
        found=1
        printf '%-34s %s\n' "$path" "$(du -sh "$path" | cut -f1)"
        if [ -f "$path/ega-report.json" ]; then
          "$PY" - "$path/ega-report.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
c = r["checks"]
print(f"  parts={r['parts']} layers={len(r['selected_layers'])}/{r['n_layers']} "
      f"dense={r['dense_modified']} experts={r['expert_slices_modified']}")
print(f"  norm_err_max={c['row_norm_max_rel_err']} proj_ratio_max={c['projection_ratio_max']}")
PYEOF
        fi
      fi
    done
    for checkpoint in checkpoints/ega-dense/*.jsonl checkpoints/dense-baseline/*.jsonl checkpoints/smoke-26b/*.jsonl; do
      [ -f "$checkpoint" ] || continue
      trials=$("$PY" - "$checkpoint" <<'PYEOF'
import json, sys
print(sum(1 for line in open(sys.argv[1]) if json.loads(line).get("op_code") == 6
          and json.loads(line).get("state") == 1))
PYEOF
)
      printf '%-52s %s completed trials\n' "$checkpoint" "$trials"
      found=1
    done
    if [ "$found" -eq 0 ]; then
      echo "No models or checkpoints yet. Run ./run.sh smoke, then ./run.sh prebake."
    fi
    ;;
  *)
    echo "Unknown command: $1" >&2
    usage
    exit 1
    ;;
esac
