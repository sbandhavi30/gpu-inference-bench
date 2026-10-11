#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../.env"

TIMESTAMP=$(date +%s)
echo "[recover] T_recover = $TIMESTAMP ($(date))"

curl -sf -X POST \
  -H "Content-Type: application/json" \
  -u "admin:${GRAFANA_ADMIN_PASSWORD}" \
  "http://localhost:${GRAFANA_PORT}/api/annotations" \
  -d "{\"text\":\"T_recover: docker compose restart vllm\",\"tags\":[\"experiment\",\"recover\"],\"time\":$((TIMESTAMP * 1000))}" \
  > /dev/null && echo "[recover] Grafana annotation posted"

echo "[recover] Restarting vLLM service..."
cd "$SCRIPT_DIR/.."
docker compose restart vllm

echo "[recover] Waiting for vLLM health check (up to 10 min)..."
RETRIES=60
until curl -sf http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; do
    RETRIES=$((RETRIES - 1))
    if [ $RETRIES -le 0 ]; then
        echo "[recover] ERROR: vLLM did not become healthy in time."
        docker compose logs --tail=30 vllm
        exit 1
    fi
    echo "[recover] ... waiting ($RETRIES retries left)"
    sleep 10
done

echo "[recover] vLLM healthy."
