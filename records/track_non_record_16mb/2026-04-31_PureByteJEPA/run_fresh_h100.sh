#!/usr/bin/env bash
set -euo pipefail

# Fresh-machine runner for the compact final JEPA script.
# Defaults target the challenge rule: train for <=10 minutes on an 8xH100 box.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv_jepa}"
TRAIN_SCRIPT="$HERE/train_jepa.py"
RUN_ID="${RUN_ID:-jepa_lagmixer_10min_$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
GPU_ID="${GPU_ID:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
ITERATIONS="${ITERATIONS:-20000}"
WARMDOWN_ITERS="${WARMDOWN_ITERS:-1200}"
VAL_MAX_BYTES="${VAL_MAX_BYTES:-0}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SUBMISSION_AUTHOR="${SUBMISSION_AUTHOR:-TODO}"
SUBMISSION_GITHUB_ID="${SUBMISSION_GITHUB_ID:-TODO}"
SUBMISSION_NAME="${SUBMISSION_NAME:-PureByte JEPA LagMixer}"
SUBMISSION_TRACK="${SUBMISSION_TRACK:-10min_16mb}"
SUBMISSION_BLURB="${SUBMISSION_BLURB:-Pure byte260 contextual-delta JEPA with LagMixer, auxiliary LM-head scoring, Muon training, and mixed int8+zlib export.}"
MAX_SUBMISSION_BYTES="${MAX_SUBMISSION_BYTES:-16000000}"

DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_byte260"
TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_pure_byte_260.json"
RUN_DIR="$HERE/runs/$RUN_ID"
BUNDLE_ROOT="${BUNDLE_ROOT:-$HERE/submissions}"
BUNDLE_DIR="$BUNDLE_ROOT/$RUN_ID"
BUNDLE_TAR="$BUNDLE_ROOT/${RUN_ID}.tar.gz"

log() {
  printf '[fresh-h100] %s\n' "$*"
}

run() {
  log "+ $*"
  "$@"
}

setup_env() {
  if [[ "${SKIP_ENV_SETUP:-0}" == "1" ]]; then
    log "Skipping env setup; using current Python: $(command -v python)"
    return
  fi
  if [[ ! -d "$VENV_DIR" ]]; then
    run "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  run python -m pip install --upgrade pip setuptools wheel
  tmp_req="$(mktemp)"
  grep -vE '^torch([<=> ].*)?$' requirements.txt > "$tmp_req"
  run python -m pip install --index-url "$TORCH_INDEX_URL" torch
  run python -m pip install -r "$tmp_req"
  rm -f "$tmp_req"
}

activate_env_if_needed() {
  if [[ "${SKIP_ENV_SETUP:-0}" != "1" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
  fi
}

check_machine() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "WARNING: nvidia-smi not found; continuing anyway."
    return
  fi
  nvidia-smi -L
  gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
  if [[ "$gpu_count" -lt 8 ]]; then
    log "WARNING: detected $gpu_count GPU(s), not 8. The challenge record budget is 8xH100."
  fi
  log "This compact trainer is single-process and will use CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES."
}

download_data() {
  log "Downloading cached byte260 data if missing: train_shards=$TRAIN_SHARDS"
  run python data/cached_challenge_fineweb.py --variant byte260 --train-shards "$TRAIN_SHARDS"
  test -f "$TOKENIZER_PATH"
  test -n "$(find "$DATA_PATH" -name 'fineweb_val_*.bin' -print -quit)"
  test -n "$(find "$DATA_PATH" -name 'fineweb_train_*.bin' -print -quit)"
}

run_training() {
  log "Starting training run_id=$RUN_ID with a ${MAX_WALLCLOCK_SECONDS}s training cap."
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  RUN_ID="$RUN_ID" \
  DATA_PATH="$DATA_PATH" \
  TOKENIZER_PATH="$TOKENIZER_PATH" \
  MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
  ITERATIONS="$ITERATIONS" \
  WARMDOWN_ITERS="$WARMDOWN_ITERS" \
  VAL_MAX_BYTES="$VAL_MAX_BYTES" \
  VAL_LOSS_EVERY="$VAL_LOSS_EVERY" \
  TRAIN_LOG_EVERY="$TRAIN_LOG_EVERY" \
  python "$TRAIN_SCRIPT"
}

prepare_submission() {
  summary="$RUN_DIR/summary.json"
  artifact="$RUN_DIR/final_model.int8.ptz"
  test -f "$summary"
  test -f "$artifact"
  rm -rf "$BUNDLE_DIR"
  mkdir -p "$BUNDLE_DIR"
  cp "$TRAIN_SCRIPT" "$BUNDLE_DIR/train_jepa.py"
  cp "$HERE/RUN_COMMANDS.md" "$BUNDLE_DIR/RUN_COMMANDS.md"
  cp "$RUN_DIR/train.log" "$RUN_DIR/config.json" "$summary" "$artifact" "$BUNDLE_DIR/"
  cat > "$BUNDLE_DIR/run_command.sh" <<EOF
#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
RUN_ID=$RUN_ID DATA_PATH='$DATA_PATH' TOKENIZER_PATH='$TOKENIZER_PATH' \\
MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS ITERATIONS=$ITERATIONS WARMDOWN_ITERS=$WARMDOWN_ITERS \\
VAL_MAX_BYTES=$VAL_MAX_BYTES VAL_LOSS_EVERY=$VAL_LOSS_EVERY TRAIN_LOG_EVERY=$TRAIN_LOG_EVERY \\
python records/track_non_record_16mb/2026-05-01_PureByteJEPA_FinalWrap/train_jepa.py
EOF
  chmod +x "$BUNDLE_DIR/run_command.sh"

  SUMMARY_PATH="$summary" \
  BUNDLE_DIR="$BUNDLE_DIR" \
  RUN_ID="$RUN_ID" \
  SUBMISSION_AUTHOR="$SUBMISSION_AUTHOR" \
  SUBMISSION_GITHUB_ID="$SUBMISSION_GITHUB_ID" \
  SUBMISSION_NAME="$SUBMISSION_NAME" \
  SUBMISSION_BLURB="$SUBMISSION_BLURB" \
  SUBMISSION_TRACK="$SUBMISSION_TRACK" \
  MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
  MAX_SUBMISSION_BYTES="$MAX_SUBMISSION_BYTES" \
  python - <<'PY'
import datetime, json, os
from pathlib import Path

summary_path = Path(os.environ["SUMMARY_PATH"])
bundle = Path(os.environ["BUNDLE_DIR"])
summary = json.loads(summary_path.read_text())
val_bpb = summary.get("val_bpb") if summary.get("val_bpb") is not None else summary.get("subset_val_bpb")
payload = {
    "author": os.environ["SUBMISSION_AUTHOR"],
    "github_id": os.environ["SUBMISSION_GITHUB_ID"],
    "name": os.environ["SUBMISSION_NAME"],
    "blurb": os.environ["SUBMISSION_BLURB"],
    "date": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "track": os.environ["SUBMISSION_TRACK"],
    "val_loss": summary.get("proxy_val_loss"),
    "val_bpb": val_bpb,
    "bytes_total": summary.get("artifact_total_bytes_int8_zlib"),
    "bytes_code": summary.get("artifact_code_bytes"),
    "run_id": summary.get("run_id", os.environ["RUN_ID"]),
    "hardware": "8xH100 target machine; compact trainer uses one visible CUDA device",
    "training_cap_seconds": float(os.environ["MAX_WALLCLOCK_SECONDS"]),
    "artifact": "final_model.int8.ptz",
}
bundle.joinpath("submission.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
total = int(summary.get("artifact_total_bytes_int8_zlib") or 0)
limit = int(os.environ["MAX_SUBMISSION_BYTES"])
if total > limit:
    raise SystemExit(f"submission artifact is too large: {total} > {limit}")
PY

  mkdir -p "$BUNDLE_ROOT"
  tar -czf "$BUNDLE_TAR" -C "$BUNDLE_ROOT" "$RUN_ID"
  log "Submission directory: $BUNDLE_DIR"
  log "Submission tarball:   $BUNDLE_TAR"
}

main() {
  setup_env
  activate_env_if_needed
  check_machine
  download_data
  run_training
  prepare_submission
}

main "$@"
