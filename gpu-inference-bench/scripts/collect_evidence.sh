#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../.env"

PROM="http://localhost:${PROMETHEUS_PORT}"
OUTPUT="$SCRIPT_DIR/../postmortem/evidence-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT"

NOW=$(date +%s)
START=$((NOW - 900))
STEP=5

echo "[evidence] Window: last 15 min. Saving to $OUTPUT"

snapshot() {
    local name="$1"
    local query="$2"
    curl -sG "$PROM/api/v1/query_range" \
        --data-urlencode "query=$query" \
        --data-urlencode "start=$START" \
        --data-urlencode "end=$NOW" \
        --data-urlencode "step=${STEP}s" \
        > "$OUTPUT/${name}.json"
    echo "[evidence] $name.json"
}

snapshot "p99_ttft"          "histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[1m])) by (le))"
snapshot "p50_ttft"          "histogram_quantile(0.50, sum(rate(vllm:time_to_first_token_seconds_bucket[1m])) by (le))"
snapshot "output_tps"        "sum(rate(vllm:generation_tokens_total[1m]))"
snapshot "queue_depth"       "vllm:num_requests_waiting"
snapshot "kv_cache_pct"      "vllm:gpu_cache_usage_perc * 100"
snapshot "gpu_sm_util"       "DCGM_FI_DEV_GPU_UTIL"
snapshot "gpu_power"         "DCGM_FI_DEV_POWER_USAGE"
snapshot "gpu_temp"          "DCGM_FI_DEV_GPU_TEMP"
snapshot "nvlink_tx_gbps"    "rate(DCGM_FI_PROF_NVLINK_TX_BYTES[1m]) / 1e9"
snapshot "nvlink_rx_gbps"    "rate(DCGM_FI_PROF_NVLINK_RX_BYTES[1m]) / 1e9"
snapshot "tensor_core_pct"   "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE * 100"
snapshot "vram_used_gib"     "DCGM_FI_DEV_FB_USED / 1024"
snapshot "vllm_up"           "up{job=\"vllm\"}"

echo "[evidence] Done. $(ls "$OUTPUT" | wc -l) files in $OUTPUT"
ls -lh "$OUTPUT"
