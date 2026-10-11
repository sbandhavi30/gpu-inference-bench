#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../.env"

COMPOSE_PROJECT="gpu-inference-bench"
VLLM_CONTAINER="${COMPOSE_PROJECT}-vllm-1"
TIMESTAMP=$(date +%s)
HUMAN_TIME=$(date)

echo "[inject] T_inject = $TIMESTAMP ($HUMAN_TIME)"

# Push T_inject annotation to Grafana
curl -sf -X POST \
  -H "Content-Type: application/json" \
  -u "admin:${GRAFANA_ADMIN_PASSWORD}" \
  "http://localhost:${GRAFANA_PORT}/api/annotations" \
  -d "{\"text\":\"T_inject: TP rank 1 SIGKILL\",\"tags\":[\"experiment\",\"inject\"],\"time\":$((TIMESTAMP * 1000))}" \
  > /dev/null && echo "[inject] Grafana annotation posted"

# vLLM TP workers are spawned with --local-rank=N argument
echo "[inject] Searching for GPU rank 1 worker..."
RANK1_PID=$(docker exec "$VLLM_CONTAINER" pgrep -f -- "--local-rank=1" 2>/dev/null | head -1 || true)

if [ -n "$RANK1_PID" ]; then
    echo "[inject] Found rank 1 PID: $RANK1_PID — sending SIGKILL"
    docker exec "$VLLM_CONTAINER" kill -9 "$RANK1_PID"
else
    echo "[inject] Rank 1 not found by --local-rank=1. Falling back to main process."
    MAIN_PID=$(docker exec "$VLLM_CONTAINER" pgrep -f "vllm.entrypoints" 2>/dev/null | head -1 || true)
    if [ -n "$MAIN_PID" ]; then
        echo "[inject] Killing vLLM main PID: $MAIN_PID"
        docker exec "$VLLM_CONTAINER" kill -9 "$MAIN_PID"
    else
        echo "[inject] ERROR: No vLLM process found. Container running?"
        docker ps --filter "name=$VLLM_CONTAINER"
        exit 1
    fi
fi

echo "[inject] Done. vLLM exits within 5-10s. Watch: docker compose logs -f vllm"
