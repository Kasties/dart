#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./start_vrcai_services.sh [start|stop|restart|start-vlm|stop-vlm|restart-vlm|start-depth|stop-depth|restart-depth|start-gemma|stop-gemma|restart-gemma|status|logs|check-env]

Environment overrides:
  VRCAI_SERVICES_ENV      Optional env file to source. Default: $DART_DIR/vrcai_services.local.env.
  CONDA_ENV               Conda env to activate before starting Python services. Default: DART.
  CONDA_SH                Optional path to conda.sh if Conda is not on PATH.
  PYTHON_BIN              Explicit Python override. Default: python from the activated Conda env.
  DART_DIR                DART checkout. Default: this script's directory.
  DENOISER_CHECKPOINT     DART checkpoint path.
  DART_HOST               DART service bind host. Default: 127.0.0.1.
  DART_PORT               DART service port. Default: 8765.
  DART_DEVICE             DART torch device. Default: cuda.
  DART_DATASET            DART dataset. Default: babel.
  DART_SHOW_VIEWER        Set to 1 to pass --show-viewer.
  GOAL_POLICY_CHECKPOINT  Reach-location policy checkpoint for --goal-location requests.
  GOAL_INIT_DATA_PATH     Initial seed motion for --goal-location requests.
  GOAL_NUM_ENVS           Parallel reach-location policy rollouts. Default: 4.
  GOAL_NUM_STEPS          Maximum reach-location policy steps. Default: 256.
  LLAMA_SERVER_BIN        llama.cpp server binary. Default: llama-server.
  VLM_BACKEND             VLM backend: llama or openrouter. Default: llama.
  VLM_MODEL_ID            VLM model id. Defaults to ggml-org/gemma-4-E4B-it-GGUF for llama, google/gemini-2.5-flash for openrouter.
  LLAMA_MODEL_PATH        Local GGUF model path. Overrides VLM_MODEL_ID.
  LLAMA_MODEL_URL         Direct GGUF model URL. Overrides VLM_MODEL_ID.
  LLAMA_HF_FILE           Explicit Hugging Face GGUF file for older llama.cpp builds.
  LLAMA_MMPROJ_PATH       Local multimodal projector path.
  LLAMA_MMPROJ_URL        Direct multimodal projector URL.
  LLAMA_HOST              llama-server bind host. Default: 127.0.0.1.
  LLAMA_PORT              llama-server port. Default: 8778.
  LLAMA_CTX_SIZE          llama-server context size. Default: 8192.
  LLAMA_REASONING         llama-server reasoning mode. Default: off.
  LLAMA_REASONING_BUDGET  Optional llama-server reasoning token budget.
  LLAMA_EXTRA_ARGS        Extra llama-server args, split by shell words.
  VLM_HOST                VLM adapter bind host. Default: 0.0.0.0.
  VLM_PORT                VLM adapter port. Default: 8777.
  VLM_MAX_NEW_TOKENS      Adapter max response tokens. Default: 160.
  VLM_TEMPERATURE         Adapter sampling temperature. Default: 0.2.
  OPENROUTER_API_KEY      Required when VLM_BACKEND=openrouter.
  OPENROUTER_MODEL_ID     OpenRouter model id fallback. Default: google/gemini-2.5-flash.
  OPENROUTER_BASE_URL     OpenRouter API base URL. Default: https://openrouter.ai/api/v1.
  OPENROUTER_REFERER      Optional OpenRouter HTTP-Referer header.
  OPENROUTER_TITLE        Optional OpenRouter title header. Default: VRCAI.
  OPENROUTER_PROVIDER     Optional provider slug/order, comma-separated. Example: alibaba.
  OPENROUTER_ALLOW_FALLBACKS  Set to 1 to allow fallback providers when OPENROUTER_PROVIDER is set. Default: 0 when a provider is pinned.
  OPENROUTER_RESPONSE_FORMAT  Set to 0 for OpenRouter models that do not support response_format JSON mode.
  DEPTH_ENABLE            Start Depth Anything with the full stack when set to 1. Default: 0.
  DEPTH_HOST              Depth adapter bind host. Default: 0.0.0.0.
  DEPTH_PORT              Depth adapter port. Default: 8779.
  DEPTH_BACKEND           hf, da2, da3, or custom. Default: hf.
  DEPTH_MODEL_ID          HF/DA3 model id. Default: depth-anything/Depth-Anything-V2-Small-hf.
  DEPTH_DEVICE            Torch device. Default: cuda.
  DEPTH_REPO_DIR          Optional local Depth Anything repository path.
  DEPTH_CHECKPOINT        DA2 checkpoint path for DEPTH_BACKEND=da2.
  DEPTH_ENCODER           DA2 encoder: vits, vitb, vitl, or vitg. Default: vits.
  DEPTH_INPUT_SIZE        DA2 input size. Default: 518.
  DEPTH_CUSTOM_FACTORY    module:function factory for DEPTH_BACKEND=custom.
  DEPTH_EXTRA_ARGS        Extra depth adapter args, split by shell words.
  LOG_DIR                 Log and pid directory. Default: $DART_DIR/logs/vrcai_services.
  PYTHONNOUSERSITE        Ignore ~/.local Python packages. Default: 1.

Deprecated aliases:
  GEMMA_MODEL_ID, GEMMA_HOST, and GEMMA_PORT are still accepted as fallbacks for VLM_*.
EOF
}

ACTION="${1:-start}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DART_DIR="${DART_DIR:-$SCRIPT_DIR}"

LOCAL_ENV_FILE="${VRCAI_SERVICES_ENV:-$DART_DIR/vrcai_services.local.env}"
if [[ -f "$LOCAL_ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$LOCAL_ENV_FILE"
  set +a
fi

CONDA_ENV="${CONDA_ENV:-DART}"

DENOISER_CHECKPOINT="${DENOISER_CHECKPOINT:-$DART_DIR/mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt}"
DART_HOST="${DART_HOST:-127.0.0.1}"
DART_PORT="${DART_PORT:-8765}"
DART_DEVICE="${DART_DEVICE:-cuda}"
DART_DATASET="${DART_DATASET:-babel}"
DART_MAX_FRAME_COUNT="${DART_MAX_FRAME_COUNT:-196}"
GOAL_POLICY_CHECKPOINT="${GOAL_POLICY_CHECKPOINT:-$DART_DIR/policy_train/reach_location_mld/fixtext_repeat_floor100_hop10_skate100/iter_2000.pth}"
GOAL_INIT_DATA_PATH="${GOAL_INIT_DATA_PATH:-$DART_DIR/data/stand.pkl}"
GOAL_NUM_ENVS="${GOAL_NUM_ENVS:-4}"
GOAL_NUM_STEPS="${GOAL_NUM_STEPS:-256}"

if [[ -z "${LLAMA_SERVER_BIN:-}" ]]; then
  if command -v llama-server >/dev/null 2>&1; then
    LLAMA_SERVER_BIN="$(command -v llama-server)"
  elif [[ -x /mnt/ssd/llama.cpp/build/bin/llama-server ]]; then
    LLAMA_SERVER_BIN="/mnt/ssd/llama.cpp/build/bin/llama-server"
  else
    LLAMA_SERVER_BIN="llama-server"
  fi
fi
VLM_BACKEND="${VLM_BACKEND:-llama}"
case "$VLM_BACKEND" in
  llama|openrouter)
    ;;
  *)
    echo "Unsupported VLM_BACKEND: $VLM_BACKEND. Use llama or openrouter." >&2
    exit 2
    ;;
esac
if [[ -z "${VLM_MODEL_ID:-}" ]]; then
  if [[ "$VLM_BACKEND" == "openrouter" ]]; then
    VLM_MODEL_ID="${OPENROUTER_MODEL_ID:-google/gemini-2.5-flash}"
  else
    VLM_MODEL_ID="${GEMMA_MODEL_ID:-ggml-org/gemma-4-E4B-it-GGUF}"
  fi
fi
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-}"
LLAMA_MODEL_URL="${LLAMA_MODEL_URL:-}"
LLAMA_HF_FILE="${LLAMA_HF_FILE:-}"
LLAMA_MMPROJ_PATH="${LLAMA_MMPROJ_PATH:-}"
LLAMA_MMPROJ_URL="${LLAMA_MMPROJ_URL:-}"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8778}"
LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-8192}"
LLAMA_REASONING="${LLAMA_REASONING:-off}"
LLAMA_REASONING_BUDGET="${LLAMA_REASONING_BUDGET:-}"
LLAMA_URL="${LLAMA_URL:-http://127.0.0.1:${LLAMA_PORT}}"
VLM_HOST="${VLM_HOST:-${GEMMA_HOST:-0.0.0.0}}"
VLM_PORT="${VLM_PORT:-${GEMMA_PORT:-8777}}"
VLM_MAX_NEW_TOKENS="${VLM_MAX_NEW_TOKENS:-160}"
VLM_TEMPERATURE="${VLM_TEMPERATURE:-0.2}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
OPENROUTER_REFERER="${OPENROUTER_REFERER:-}"
OPENROUTER_TITLE="${OPENROUTER_TITLE:-VRCAI}"
OPENROUTER_PROVIDER="${OPENROUTER_PROVIDER:-${OPENROUTER_PROVIDER_ORDER:-}}"
OPENROUTER_ALLOW_FALLBACKS="${OPENROUTER_ALLOW_FALLBACKS:-}"
OPENROUTER_RESPONSE_FORMAT="${OPENROUTER_RESPONSE_FORMAT:-}"
DEPTH_ENABLE="${DEPTH_ENABLE:-0}"
DEPTH_HOST="${DEPTH_HOST:-0.0.0.0}"
DEPTH_PORT="${DEPTH_PORT:-8779}"
DEPTH_BACKEND="${DEPTH_BACKEND:-hf}"
DEPTH_MODEL_ID="${DEPTH_MODEL_ID:-depth-anything/Depth-Anything-V2-Small-hf}"
DEPTH_DEVICE="${DEPTH_DEVICE:-cuda}"
DEPTH_REPO_DIR="${DEPTH_REPO_DIR:-}"
DEPTH_CHECKPOINT="${DEPTH_CHECKPOINT:-}"
DEPTH_ENCODER="${DEPTH_ENCODER:-vits}"
DEPTH_INPUT_SIZE="${DEPTH_INPUT_SIZE:-518}"
DEPTH_CUSTOM_FACTORY="${DEPTH_CUSTOM_FACTORY:-}"

LOG_DIR="${LOG_DIR:-$DART_DIR/logs/vrcai_services}"
PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONNOUSERSITE

mkdir -p "$LOG_DIR"
DART_PID="$LOG_DIR/dart_session_service.pid"
LLAMA_PID="$LOG_DIR/llama_server.pid"
VLM_PID="$LOG_DIR/gemma_vlm_service.pid"
DEPTH_PID="$LOG_DIR/depth_anything_service.pid"
DART_LOG="$LOG_DIR/dart_session_service.log"
LLAMA_LOG="$LOG_DIR/llama_server.log"
VLM_LOG="$LOG_DIR/vlm_adapter.log"
DEPTH_LOG="$LOG_DIR/depth_anything_service.log"

is_alive() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

find_conda_sh() {
  local conda_base=""
  local candidates=()
  if [[ -n "${CONDA_SH:-}" ]]; then
    candidates+=("$CONDA_SH")
  fi
  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$conda_base" ]]; then
      candidates+=("$conda_base/etc/profile.d/conda.sh")
    fi
  fi
  candidates+=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/mnt/ssd/anna/etc/profile.d/conda.sh"
    "/mnt/ssd/miniconda3/etc/profile.d/conda.sh"
    "/mnt/ssd/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

activate_runtime() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "Using explicit Python: $PYTHON_BIN"
    return
  fi

  local conda_sh
  if ! conda_sh="$(find_conda_sh)"; then
    echo "Could not find conda.sh. Set CONDA_SH=/path/to/conda.sh or PYTHON_BIN=/path/to/python." >&2
    exit 1
  fi

  set +u
  # shellcheck source=/dev/null
  source "$conda_sh"
  conda activate "$CONDA_ENV"
  local activate_status="$?"
  set -u
  if [[ "$activate_status" -ne 0 ]]; then
    echo "Failed to activate Conda env: $CONDA_ENV" >&2
    exit "$activate_status"
  fi
  PYTHON_BIN="$(command -v python)"
  echo "Activated conda env: $CONDA_ENV"
}

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -E "(:|\\])${port}$" >/dev/null 2>&1
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 1
  fi
}

require_free_port() {
  local name="$1"
  local port="$2"
  local pid_file="$3"
  if ! is_alive "$pid_file" && port_in_use "$port"; then
    echo "$name port $port is already listening, but no script pid file owns it." >&2
    echo "Stop the old manual process or choose a different port before starting." >&2
    exit 1
  fi
}

require_command() {
  local command_name="$1"
  if [[ -x "$command_name" ]] || command -v "$command_name" >/dev/null 2>&1; then
    return
  fi
  echo "Missing executable: $command_name. Set LLAMA_SERVER_BIN or add llama-server to PATH." >&2
  exit 1
}

confirm_started() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  sleep "${STARTUP_GRACE_SEC:-1}"
  if ! is_alive "$pid_file"; then
    echo "$name exited during startup. Last log lines:" >&2
    tail -n 40 "$log_file" >&2 || true
    exit 1
  fi
}

start_dart() {
  if is_alive "$DART_PID"; then
    echo "DART session service already running with pid $(cat "$DART_PID")."
    return
  fi
  require_free_port "DART session service" "$DART_PORT" "$DART_PID"
  if [[ ! -f "$DART_DIR/dart_session_service.py" ]]; then
    echo "Missing dart_session_service.py in $DART_DIR" >&2
    exit 1
  fi
  if [[ ! -f "$DENOISER_CHECKPOINT" ]]; then
    echo "Missing DART checkpoint: $DENOISER_CHECKPOINT" >&2
    exit 1
  fi

  local args=(
    "$DART_DIR/dart_session_service.py"
    --dart-dir "$DART_DIR"
    --denoiser-checkpoint "$DENOISER_CHECKPOINT"
    --dataset "$DART_DATASET"
    --device "$DART_DEVICE"
    --host "$DART_HOST"
    --port "$DART_PORT"
    --max-frame-count "$DART_MAX_FRAME_COUNT"
    --goal-policy-checkpoint "$GOAL_POLICY_CHECKPOINT"
    --goal-init-data-path "$GOAL_INIT_DATA_PATH"
    --goal-num-envs "$GOAL_NUM_ENVS"
    --goal-num-steps "$GOAL_NUM_STEPS"
  )
  if [[ "${DART_SHOW_VIEWER:-0}" == "1" ]]; then
    args+=(--show-viewer)
  fi
  if [[ -n "${DART_RESPACING:-}" ]]; then
    args+=(--respacing "$DART_RESPACING")
  fi

  cd "$DART_DIR"
  nohup "$PYTHON_BIN" -u "${args[@]}" >"$DART_LOG" 2>&1 </dev/null &
  echo "$!" >"$DART_PID"
  confirm_started "DART session service" "$DART_PID" "$DART_LOG"
  echo "Started DART session service on $DART_HOST:$DART_PORT with pid $(cat "$DART_PID")."
}

start_llama() {
  if [[ "$VLM_BACKEND" == "openrouter" ]]; then
    echo "Skipping llama-server because VLM_BACKEND=openrouter."
    return
  fi
  if is_alive "$LLAMA_PID"; then
    echo "llama-server already running with pid $(cat "$LLAMA_PID")."
    return
  fi
  require_command "$LLAMA_SERVER_BIN"
  require_free_port "llama-server" "$LLAMA_PORT" "$LLAMA_PID"

  local args=()
  if [[ -n "$LLAMA_MODEL_PATH" ]]; then
    args+=(-m "$LLAMA_MODEL_PATH")
  elif [[ -n "$LLAMA_MODEL_URL" ]]; then
    args+=(--model-url "$LLAMA_MODEL_URL")
  else
    args+=(-hf "$VLM_MODEL_ID")
    if [[ -n "$LLAMA_HF_FILE" ]]; then
      args+=(-hff "$LLAMA_HF_FILE")
    fi
  fi
  if [[ -n "$LLAMA_MMPROJ_PATH" ]]; then
    args+=(--mmproj "$LLAMA_MMPROJ_PATH")
  elif [[ -n "$LLAMA_MMPROJ_URL" ]]; then
    args+=(--mmproj-url "$LLAMA_MMPROJ_URL")
  fi
  args+=(
    --host "$LLAMA_HOST"
    --port "$LLAMA_PORT"
    -c "$LLAMA_CTX_SIZE"
    --reasoning "$LLAMA_REASONING"
  )
  if [[ -n "$LLAMA_REASONING_BUDGET" ]]; then
    args+=(--reasoning-budget "$LLAMA_REASONING_BUDGET")
  fi
  if [[ -n "${LLAMA_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra_args=( $LLAMA_EXTRA_ARGS )
    args+=("${extra_args[@]}")
  fi

  cd "$DART_DIR"
  nohup "$LLAMA_SERVER_BIN" "${args[@]}" >"$LLAMA_LOG" 2>&1 </dev/null &
  echo "$!" >"$LLAMA_PID"
  confirm_started "llama-server" "$LLAMA_PID" "$LLAMA_LOG"
  echo "Started llama-server on $LLAMA_HOST:$LLAMA_PORT with pid $(cat "$LLAMA_PID")."
}

start_vlm_adapter() {
  if is_alive "$VLM_PID"; then
    echo "VLM adapter already running with pid $(cat "$VLM_PID")."
    return
  fi
  if [[ "$VLM_BACKEND" == "openrouter" && -z "$OPENROUTER_API_KEY" ]]; then
    echo "OPENROUTER_API_KEY is required when VLM_BACKEND=openrouter." >&2
    exit 1
  fi
  require_free_port "VLM adapter" "$VLM_PORT" "$VLM_PID"
  if [[ ! -f "$DART_DIR/gemma_vlm_service.py" ]]; then
    echo "Missing gemma_vlm_service.py in $DART_DIR" >&2
    exit 1
  fi

  local args=(
    "$DART_DIR/gemma_vlm_service.py"
    --backend "$VLM_BACKEND"
    --model-id "$VLM_MODEL_ID"
    --host "$VLM_HOST"
    --port "$VLM_PORT"
    --llama-url "$LLAMA_URL"
    --max-new-tokens "$VLM_MAX_NEW_TOKENS"
    --temperature "$VLM_TEMPERATURE"
    --openrouter-url "$OPENROUTER_BASE_URL"
    --openrouter-referer "$OPENROUTER_REFERER"
    --openrouter-title "$OPENROUTER_TITLE"
    --openrouter-provider "$OPENROUTER_PROVIDER"
  )
  if [[ -n "$OPENROUTER_PROVIDER" || -n "$OPENROUTER_ALLOW_FALLBACKS" ]]; then
    if [[ "$OPENROUTER_ALLOW_FALLBACKS" == "1" || "$OPENROUTER_ALLOW_FALLBACKS" == "true" ]]; then
      args+=(--openrouter-allow-fallbacks)
    else
      args+=(--openrouter-no-provider-fallbacks)
    fi
  fi
  if [[ -n "$OPENROUTER_RESPONSE_FORMAT" ]]; then
    if [[ "$OPENROUTER_RESPONSE_FORMAT" == "0" || "$OPENROUTER_RESPONSE_FORMAT" == "false" ]]; then
      args+=(--openrouter-no-response-format)
    else
      args+=(--openrouter-response-format)
    fi
  fi

  cd "$DART_DIR"
  nohup "$PYTHON_BIN" -u "${args[@]}" >"$VLM_LOG" 2>&1 </dev/null &
  echo "$!" >"$VLM_PID"
  confirm_started "VLM adapter" "$VLM_PID" "$VLM_LOG"
  echo "Started VLM adapter on $VLM_HOST:$VLM_PORT with pid $(cat "$VLM_PID")."
}

start_depth_adapter() {
  if is_alive "$DEPTH_PID"; then
    echo "Depth Anything adapter already running with pid $(cat "$DEPTH_PID")."
    return
  fi
  require_free_port "Depth Anything adapter" "$DEPTH_PORT" "$DEPTH_PID"
  if [[ ! -f "$DART_DIR/depth_anything_service.py" ]]; then
    echo "Missing depth_anything_service.py in $DART_DIR" >&2
    exit 1
  fi

  local args=(
    "$DART_DIR/depth_anything_service.py"
    --host "$DEPTH_HOST"
    --port "$DEPTH_PORT"
    --backend "$DEPTH_BACKEND"
    --model-id "$DEPTH_MODEL_ID"
    --device "$DEPTH_DEVICE"
    --encoder "$DEPTH_ENCODER"
    --input-size "$DEPTH_INPUT_SIZE"
  )
  if [[ -n "$DEPTH_REPO_DIR" ]]; then
    args+=(--repo-dir "$DEPTH_REPO_DIR")
  fi
  if [[ -n "$DEPTH_CHECKPOINT" ]]; then
    args+=(--checkpoint "$DEPTH_CHECKPOINT")
  fi
  if [[ -n "$DEPTH_CUSTOM_FACTORY" ]]; then
    args+=(--custom-factory "$DEPTH_CUSTOM_FACTORY")
  fi
  if [[ -n "${DEPTH_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra_args=( $DEPTH_EXTRA_ARGS )
    args+=("${extra_args[@]}")
  fi

  cd "$DART_DIR"
  nohup "$PYTHON_BIN" -u "${args[@]}" >"$DEPTH_LOG" 2>&1 </dev/null &
  echo "$!" >"$DEPTH_PID"
  confirm_started "Depth Anything adapter" "$DEPTH_PID" "$DEPTH_LOG"
  echo "Started Depth Anything adapter on $DEPTH_HOST:$DEPTH_PORT with pid $(cat "$DEPTH_PID")."
}

stop_one() {
  local name="$1"
  local pid_file="$2"
  if ! is_alive "$pid_file"; then
    rm -f "$pid_file"
    echo "$name is not running."
    return
  fi

  local pid
  pid="$(cat "$pid_file")"
  kill "$pid" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pid_file"
      echo "Stopped $name."
      return
    fi
    sleep 0.25
  done
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$pid_file"
  echo "Stopped $name with SIGKILL."
}

status_one() {
  local name="$1"
  local pid_file="$2"
  local bind="$3"
  local port="$4"
  if is_alive "$pid_file"; then
    echo "$name running: pid $(cat "$pid_file"), bind $bind"
  elif port_in_use "$port"; then
    echo "$name has no script pid file, but port $port is listening."
  else
    echo "$name stopped."
  fi
}

start_all() {
  activate_runtime
  echo "Using Python: $PYTHON_BIN"
  echo "Using DART dir: $DART_DIR"
  echo "Using PYTHONNOUSERSITE: $PYTHONNOUSERSITE"
  echo "Using VLM backend: $VLM_BACKEND"
  start_dart
  start_llama
  start_vlm_adapter
  if [[ "$DEPTH_ENABLE" == "1" ]]; then
    start_depth_adapter
  fi
}

start_vlm_stack() {
  activate_runtime
  echo "Using Python: $PYTHON_BIN"
  echo "Using DART dir: $DART_DIR"
  echo "Using PYTHONNOUSERSITE: $PYTHONNOUSERSITE"
  echo "Using VLM backend: $VLM_BACKEND"
  start_llama
  start_vlm_adapter
}

stop_vlm_stack() {
  stop_one "VLM adapter" "$VLM_PID"
  stop_one "llama-server" "$LLAMA_PID"
}

case "$ACTION" in
  start)
    start_all
    ;;
  stop)
    stop_one "Depth Anything adapter" "$DEPTH_PID"
    stop_vlm_stack
    stop_one "DART session service" "$DART_PID"
    ;;
  restart)
    stop_one "Depth Anything adapter" "$DEPTH_PID"
    stop_vlm_stack
    stop_one "DART session service" "$DART_PID"
    start_all
    ;;
  start-vlm|start-gemma)
    start_vlm_stack
    ;;
  stop-vlm|stop-gemma)
    stop_vlm_stack
    ;;
  restart-vlm|restart-gemma)
    stop_vlm_stack
    start_vlm_stack
    ;;
  start-depth)
    activate_runtime
    echo "Using Python: $PYTHON_BIN"
    echo "Using DART dir: $DART_DIR"
    echo "Using PYTHONNOUSERSITE: $PYTHONNOUSERSITE"
    start_depth_adapter
    ;;
  stop-depth)
    stop_one "Depth Anything adapter" "$DEPTH_PID"
    ;;
  restart-depth)
    stop_one "Depth Anything adapter" "$DEPTH_PID"
    activate_runtime
    echo "Using Python: $PYTHON_BIN"
    echo "Using DART dir: $DART_DIR"
    echo "Using PYTHONNOUSERSITE: $PYTHONNOUSERSITE"
    start_depth_adapter
    ;;
  status)
    status_one "DART session service" "$DART_PID" "$DART_HOST:$DART_PORT" "$DART_PORT"
    status_one "llama-server" "$LLAMA_PID" "$LLAMA_HOST:$LLAMA_PORT" "$LLAMA_PORT"
    status_one "VLM adapter" "$VLM_PID" "$VLM_HOST:$VLM_PORT" "$VLM_PORT"
    status_one "Depth Anything adapter" "$DEPTH_PID" "$DEPTH_HOST:$DEPTH_PORT" "$DEPTH_PORT"
    ;;
  logs)
    touch "$DART_LOG" "$LLAMA_LOG" "$VLM_LOG" "$DEPTH_LOG"
    tail -n "${TAIL_LINES:-80}" -f "$DART_LOG" "$LLAMA_LOG" "$VLM_LOG" "$DEPTH_LOG"
    ;;
  check-env)
    activate_runtime
    echo "Using Python: $PYTHON_BIN"
    echo "Using PYTHONNOUSERSITE: $PYTHONNOUSERSITE"
    "$PYTHON_BIN" --version
    "$PYTHON_BIN" - <<'PY'
import json
import urllib.request

print("VLM adapter Python imports OK")
PY
    if [[ "$VLM_BACKEND" == "llama" ]]; then
      require_command "$LLAMA_SERVER_BIN"
      "$LLAMA_SERVER_BIN" --version || true
    else
      echo "OpenRouter VLM backend selected; llama-server is not required."
    fi
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
