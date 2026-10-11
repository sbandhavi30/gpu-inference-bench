# GPU Inference Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy a 4x H100 SXM vLLM inference stack on RunPod with full GPU and inference observability, execute a TP rank failure injection, and produce a written postmortem.

**Architecture:** Docker Compose on a single RunPod Secure Cloud 4xH100 SXM host. vLLM serves Llama-3.3-70B-Instruct with TP=4 across all four GPUs. DCGM Exporter collects GPU hardware metrics including NVLink profiling fields, feeding Prometheus, which Grafana visualizes in 7 dashboard rows. Locust generates concurrent chat-completion load. Three scripts handle failure injection, recovery, and evidence collection using the Grafana annotation API for timeline markers.

**Tech Stack:** Docker Compose, vLLM (latest), DCGM Exporter 3.3.5-3.4.0-ubuntu22.04, Prometheus 2.51, Grafana 10.4, node_exporter 1.7, Locust 2.24, Bash

**Spec:** `docs/superpowers/specs/2026-10-10-gpu-inference-bench-design.md`

## Global Constraints

- Host: RunPod Secure Cloud 4x H100 SXM only — NVLink fabric required for `DCGM_FI_PROF_NVLINK_*` fields
- vLLM env var `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0` must be set (RunPod image workaround)
- DCGM container requires `cap_add: [SYS_ADMIN]` for profiling metrics
- Model `meta-llama/Llama-3.3-70B-Instruct` is gated on HuggingFace — `HF_TOKEN` in `.env` required
- SLO: p99 TTFT < 2.0 s
- Locust steady-state: 50 concurrent users
- Network volume: 500 GB minimum (model ~140 GB BF16 + Docker images)
- All scripts must be executable (`chmod +x`)
- Grafana annotations use the HTTP annotation API (POST `:3000/api/annotations`) — no Pushgateway needed

## Review Focus

- DCGM NVLink profiling fields (`DCGM_FI_PROF_NVLINK_TX_BYTES`, `DCGM_FI_PROF_NVLINK_RX_BYTES`) silently absent without `SYS_ADMIN` — Task 2 Step 2 verifies before proceeding; Task 8 Step 4 checks Prometheus
- vLLM `/metrics` metric names vary by version — Task 8 Step 3 adds a verification step to confirm actual names before relying on Grafana panels
- `inject_failure.sh` uses `pgrep -f -- "--local-rank=1"` pattern — Task 6 Step 5 runs a dry-run pgrep check before actual injection to confirm the pattern matches
- Model download (~140 GB) billed as uptime — Task 8 Step 2 explicitly waits for `Application startup complete` in logs; Task 8 Step 3 confirms health endpoint before starting Locust
- Grafana annotation API requires admin credentials — scripts source `.env` for `GRAFANA_ADMIN_PASSWORD`; Task 6 Step 6 tests the API call before the live run

---

## File Map

| File | Task | Purpose |
|---|---|---|
| `gpu-inference-bench/docker-compose.yml` | 1 | All 6 services wired together |
| `gpu-inference-bench/.env` | 1 | HF_TOKEN, ports, passwords |
| `gpu-inference-bench/dcgm/metrics.csv` | 2 | DCGM field IDs including NVLink profiling |
| `gpu-inference-bench/prometheus/prometheus.yml` | 3 | Scrape configs for all exporters |
| `gpu-inference-bench/prometheus/alerts.yml` | 3 | SLO alert: p99 TTFT > 2s |
| `gpu-inference-bench/grafana/provisioning/datasources/prometheus.yml` | 4 | Auto-wires Prometheus datasource |
| `gpu-inference-bench/grafana/provisioning/dashboards/dashboard.yml` | 4 | Points Grafana at dashboard JSON dir |
| `gpu-inference-bench/grafana/dashboards/gpu-inference.json` | 4 | 7-row dashboard with all panels |
| `gpu-inference-bench/locust/locustfile.py` | 5 | Chat completion load workload |
| `gpu-inference-bench/locust/Dockerfile` | 5 | Locust container image |
| `gpu-inference-bench/scripts/inject_failure.sh` | 6 | kill -9 vLLM GPU rank 1 worker |
| `gpu-inference-bench/scripts/recover.sh` | 6 | docker compose restart vllm |
| `gpu-inference-bench/scripts/collect_evidence.sh` | 6 | Prometheus range-query snapshot to JSON |
| `gpu-inference-bench/postmortem/template.md` | 7 | Pre-filled postmortem skeleton |

---

### Task 1: Project Scaffold

**Files:**
- Create: `gpu-inference-bench/docker-compose.yml`
- Create: `gpu-inference-bench/.env`

**Interfaces:**
- Produces: compose project name `gpu-inference-bench`; service names `vllm`, `dcgm-exporter`, `node-exporter`, `prometheus`, `grafana`, `locust`; `.env` keys consumed by all tasks

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p gpu-inference-bench/{dcgm,prometheus,grafana/provisioning/{datasources,dashboards},grafana/dashboards,locust,scripts,postmortem}
```

- [ ] **Step 2: Create `.env`**

Create `gpu-inference-bench/.env`:
```
HF_TOKEN=hf_your_token_here
MODEL=meta-llama/Llama-3.3-70B-Instruct
VLLM_PORT=8000
GRAFANA_ADMIN_PASSWORD=changeme
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
LOCUST_PORT=8089
DCGM_PORT=9400
NODE_EXPORTER_PORT=9100
```

- [ ] **Step 3: Create `docker-compose.yml`**

Create `gpu-inference-bench/docker-compose.yml`:
```yaml
version: "3.9"

x-gpu-deploy: &gpu-deploy
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]

volumes:
  huggingface_cache:
  prometheus_data:
  grafana_data:

services:
  vllm:
    image: vllm/vllm-openai:latest
    <<: *gpu-deploy
    environment:
      - VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
      - HF_TOKEN=${HF_TOKEN}
    command: >
      --model ${MODEL}
      --tensor-parallel-size 4
      --max-model-len 8192
      --gpu-memory-utilization 0.90
      --enable-metrics
    ports:
      - "${VLLM_PORT}:8000"
    volumes:
      - huggingface_cache:/root/.cache/huggingface
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 600s

  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04
    <<: *gpu-deploy
    cap_add:
      - SYS_ADMIN
    volumes:
      - ./dcgm/metrics.csv:/etc/dcgm-exporter/custom-counters.csv
    command: ["-f", "/etc/dcgm-exporter/custom-counters.csv", "-d", "f"]
    ports:
      - "${DCGM_PORT}:9400"
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:v1.7.0
    command:
      - "--path.rootfs=/host"
    pid: host
    volumes:
      - "/:/host:ro,rslave"
    ports:
      - "${NODE_EXPORTER_PORT}:9100"
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:v2.51.0
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--storage.tsdb.retention.time=24h"
      - "--web.enable-lifecycle"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/alerts.yml:/etc/prometheus/alerts.yml
      - prometheus_data:/prometheus
    ports:
      - "${PROMETHEUS_PORT}:9090"
    depends_on: [vllm, dcgm-exporter, node-exporter]
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.4.0
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_ANALYTICS_REPORTING_ENABLED=false
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
      - grafana_data:/var/lib/grafana
    ports:
      - "${GRAFANA_PORT}:3000"
    depends_on: [prometheus]
    restart: unless-stopped

  locust:
    build: ./locust
    command: >
      -f /mnt/locust/locustfile.py
      --host http://vllm:8000
      --web-port 8089
    volumes:
      - ./locust:/mnt/locust
    ports:
      - "${LOCUST_PORT}:8089"
    depends_on: [vllm]
    restart: unless-stopped
```

- [ ] **Step 4: Validate compose syntax**

```bash
cd gpu-inference-bench && docker compose config
```
Expected: full YAML printed, no errors.

- [ ] **Step 5: Commit**

```bash
git add gpu-inference-bench/docker-compose.yml gpu-inference-bench/.env
git commit -m "feat: docker compose scaffold for GPU inference bench stack"
```

---

### Task 2: DCGM Exporter Metrics Config

**Files:**
- Create: `gpu-inference-bench/dcgm/metrics.csv`

**Interfaces:**
- Consumed by: `dcgm-exporter` service via volume mount at `/etc/dcgm-exporter/custom-counters.csv`
- Produces: Prometheus metric names used in Task 4 PromQL queries (exact names listed in comments below)

- [ ] **Step 1: Create `dcgm/metrics.csv`**

Create `gpu-inference-bench/dcgm/metrics.csv`:
```csv
# DCGM custom counters — GPU inference benchmarking
# Format: FIELD_NAME, prometheus_type, help_text
#
# Clocks
DCGM_FI_DEV_SM_CLOCK,  gauge, SM clock frequency (MHz)
DCGM_FI_DEV_MEM_CLOCK, gauge, Memory clock frequency (MHz)
#
# Utilization
DCGM_FI_DEV_GPU_UTIL,      gauge, GPU SM utilization (percent)
DCGM_FI_DEV_MEM_COPY_UTIL, gauge, GPU memory copy engine utilization (percent)
#
# Framebuffer memory
DCGM_FI_DEV_FB_FREE, gauge, Framebuffer memory free (MiB)
DCGM_FI_DEV_FB_USED, gauge, Framebuffer memory used (MiB)
#
# Power and thermals
DCGM_FI_DEV_POWER_USAGE,     gauge,   GPU power draw (W)
DCGM_FI_DEV_GPU_TEMP,        gauge,   GPU temperature (C)
DCGM_FI_DEV_POWER_VIOLATION, counter, Cumulative power throttle duration (us)
#
# ECC errors
DCGM_FI_DEV_ECC_SBE_VOL_TOTAL, counter, ECC single-bit errors (volatile total)
DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, counter, ECC double-bit errors (volatile total)
#
# Profiling — require SYS_ADMIN capability on dcgm-exporter container
DCGM_FI_PROF_GR_ENGINE_ACTIVE,   gauge, Fraction of time graphics engine active (0-1)
DCGM_FI_PROF_SM_ACTIVE,          gauge, Fraction of cycles at least one thread active (0-1)
DCGM_FI_PROF_SM_OCCUPANCY,       gauge, Fraction of warps resident per SM (0-1)
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, gauge, Fraction of cycles tensor core active (0-1)
DCGM_FI_PROF_DRAM_ACTIVE,        gauge, Fraction of cycles DRAM interface active (0-1)
DCGM_FI_PROF_PCIE_TX_BYTES,      counter, PCIe transmit bytes
DCGM_FI_PROF_PCIE_RX_BYTES,      counter, PCIe receive bytes
#
# NVLink — H100 SXM specific, require SYS_ADMIN
DCGM_FI_PROF_NVLINK_TX_BYTES, counter, NVLink transmit bytes
DCGM_FI_PROF_NVLINK_RX_BYTES, counter, NVLink receive bytes
```

- [ ] **Step 2: Verify DCGM exporter starts and exposes NVLink metrics**

```bash
cd gpu-inference-bench
docker compose up dcgm-exporter -d
sleep 15
curl -s http://localhost:9400/metrics | grep -E "DCGM_FI_PROF_NVLINK|DCGM_FI_DEV_GPU_UTIL" | head -10
```
Expected: lines like `DCGM_FI_DEV_GPU_UTIL{gpu="0",...} 0` and `DCGM_FI_PROF_NVLINK_TX_BYTES{gpu="0",...} 0`.

If NVLink lines absent: check `docker compose logs dcgm-exporter | grep -iE "error|prof"`. Missing `SYS_ADMIN` cap or driver profiling unsupported — verify compose `cap_add: [SYS_ADMIN]` is present.

- [ ] **Step 3: Stop DCGM**

```bash
docker compose stop dcgm-exporter
```

- [ ] **Step 4: Commit**

```bash
git add gpu-inference-bench/dcgm/metrics.csv
git commit -m "feat: DCGM custom metrics config with NVLink profiling fields"
```

---

### Task 3: Prometheus Config + SLO Alert Rules

**Files:**
- Create: `gpu-inference-bench/prometheus/prometheus.yml`
- Create: `gpu-inference-bench/prometheus/alerts.yml`

**Interfaces:**
- Consumes: Docker service names from Task 1 — `vllm:8000`, `dcgm-exporter:9400`, `node-exporter:9100`
- Produces: PromQL metric namespace consumed by Task 4 Grafana panels

- [ ] **Step 1: Create `prometheus/prometheus.yml`**

Create `gpu-inference-bench/prometheus/prometheus.yml`:
```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 5s

rule_files:
  - /etc/prometheus/alerts.yml

scrape_configs:
  - job_name: vllm
    static_configs:
      - targets: ["vllm:8000"]
    metrics_path: /metrics

  - job_name: dcgm
    static_configs:
      - targets: ["dcgm-exporter:9400"]

  - job_name: node
    static_configs:
      - targets: ["node-exporter:9100"]
```

- [ ] **Step 2: Create `prometheus/alerts.yml`**

Create `gpu-inference-bench/prometheus/alerts.yml`:
```yaml
groups:
  - name: inference_slo
    interval: 5s
    rules:
      - alert: P99TTFTBreached
        expr: >
          histogram_quantile(0.99,
            sum(rate(vllm:time_to_first_token_seconds_bucket[1m])) by (le)
          ) > 2.0
        for: 10s
        labels:
          severity: critical
        annotations:
          summary: "p99 TTFT > 2s SLO breached"
          description: "p99 TTFT is {{ $value | humanizeDuration }}"

      - alert: VLLMDown
        expr: up{job="vllm"} == 0
        for: 10s
        labels:
          severity: critical
        annotations:
          summary: "vLLM is unreachable"

      - alert: GPUSMUtilSuddenDrop
        expr: >
          avg(DCGM_FI_DEV_GPU_UTIL) < 10
          and on()
          avg(DCGM_FI_DEV_GPU_UTIL offset 1m) > 50
        for: 15s
        labels:
          severity: critical
        annotations:
          summary: "GPU SM utilization sudden drop — possible TP rank failure"
```

- [ ] **Step 3: Validate Prometheus config**

```bash
cd gpu-inference-bench
docker run --rm \
  -v "$(pwd)/prometheus:/etc/prometheus" \
  prom/prometheus:v2.51.0 \
  promtool check config /etc/prometheus/prometheus.yml
```
Expected: `SUCCESS: 1 rule files found` and no errors.

- [ ] **Step 4: Commit**

```bash
git add gpu-inference-bench/prometheus/
git commit -m "feat: Prometheus scrape config + SLO alert rules for inference bench"
```

---

### Task 4: Grafana Provisioning + Dashboard

**Files:**
- Create: `gpu-inference-bench/grafana/provisioning/datasources/prometheus.yml`
- Create: `gpu-inference-bench/grafana/provisioning/dashboards/dashboard.yml`
- Create: `gpu-inference-bench/grafana/dashboards/gpu-inference.json`

**Interfaces:**
- Consumes: Prometheus at `http://prometheus:9090`
- Produces: Dashboard UID `gpu-inference-bench` (used by `collect_evidence.sh` for export URL)

**PromQL reference per row:**
```
# Row 1 — SLO Tracker
p99 TTFT:     histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[1m])) by (le))
Failure rate: 1 - (sum(rate(vllm:request_success_total[1m])) / sum(rate(vllm:request_total[1m])))

# Row 2 — Throughput
Output tok/s: sum(rate(vllm:generation_tokens_total[1m]))
Prompt tok/s: sum(rate(vllm:prompt_tokens_total[1m]))
Queue depth:  vllm:num_requests_waiting
KV cache %:   vllm:gpu_cache_usage_perc * 100

# Row 3 — Latency
TTFT p50/95/99: histogram_quantile(0.50|0.95|0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[1m])) by (le))
TPOT p50/95/99: histogram_quantile(0.50|0.95|0.99, sum(rate(vllm:time_per_output_token_seconds_bucket[1m])) by (le))

# Row 4 — GPU Compute
SM util:       DCGM_FI_DEV_GPU_UTIL
Tensor core:   DCGM_FI_PROF_PIPE_TENSOR_ACTIVE * 100

# Row 5 — GPU Memory
VRAM GiB:      DCGM_FI_DEV_FB_USED / 1024
Mem BW active: DCGM_FI_PROF_DRAM_ACTIVE * 100

# Row 6 — NVLink
TX GB/s:  rate(DCGM_FI_PROF_NVLINK_TX_BYTES[1m]) / 1e9
RX GB/s:  rate(DCGM_FI_PROF_NVLINK_RX_BYTES[1m]) / 1e9
ECC err:  DCGM_FI_DEV_ECC_SBE_VOL_TOTAL + DCGM_FI_DEV_ECC_DBE_VOL_TOTAL

# Row 7 — Power + Thermals
Power W:    DCGM_FI_DEV_POWER_USAGE
Temp C:     DCGM_FI_DEV_GPU_TEMP
SM clock:   DCGM_FI_DEV_SM_CLOCK
```

- [ ] **Step 1: Create datasource provisioning**

Create `gpu-inference-bench/grafana/provisioning/datasources/prometheus.yml`:
```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    uid: prometheus
    url: http://prometheus:9090
    access: proxy
    isDefault: true
    jsonData:
      timeInterval: "5s"
```

- [ ] **Step 2: Create dashboard provisioning**

Create `gpu-inference-bench/grafana/provisioning/dashboards/dashboard.yml`:
```yaml
apiVersion: 1
providers:
  - name: default
    folder: Inference Bench
    type: file
    options:
      path: /var/lib/grafana/dashboards
```

- [ ] **Step 3: Create dashboard JSON**

Create `gpu-inference-bench/grafana/dashboards/gpu-inference.json`:
```json
{
  "uid": "gpu-inference-bench",
  "title": "GPU Inference Benchmark",
  "schemaVersion": 39,
  "refresh": "5s",
  "time": { "from": "now-15m", "to": "now" },
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": { "type": "grafana", "uid": "-- Grafana --" },
        "enable": true,
        "hide": false,
        "iconColor": "red",
        "name": "Experiment Events",
        "type": "dashboard"
      }
    ]
  },
  "panels": [
    {
      "id": 1, "type": "stat", "title": "p99 TTFT",
      "gridPos": { "x": 0, "y": 0, "w": 8, "h": 4 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [{ "expr": "histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[1m])) by (le))", "legendFormat": "p99 TTFT" }],
      "options": { "reduceOptions": { "calcs": ["lastNotNull"] }, "colorMode": "background" },
      "fieldConfig": { "defaults": { "unit": "s", "thresholds": { "mode": "absolute", "steps": [{ "color": "green", "value": null }, { "color": "yellow", "value": 1.5 }, { "color": "red", "value": 2.0 }] } } }
    },
    {
      "id": 2, "type": "stat", "title": "Request Failure Rate",
      "gridPos": { "x": 8, "y": 0, "w": 8, "h": 4 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [{ "expr": "1 - (sum(rate(vllm:request_success_total[1m])) / (sum(rate(vllm:request_total[1m])) + 0.0001))", "legendFormat": "Failure Rate" }],
      "options": { "reduceOptions": { "calcs": ["lastNotNull"] }, "colorMode": "background" },
      "fieldConfig": { "defaults": { "unit": "percentunit", "thresholds": { "mode": "absolute", "steps": [{ "color": "green", "value": null }, { "color": "red", "value": 0.001 }] } } }
    },
    {
      "id": 3, "type": "stat", "title": "Active Requests",
      "gridPos": { "x": 16, "y": 0, "w": 8, "h": 4 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [{ "expr": "vllm:num_requests_running", "legendFormat": "running" }],
      "options": { "reduceOptions": { "calcs": ["lastNotNull"] }, "colorMode": "value" }
    },
    {
      "id": 4, "type": "timeseries", "title": "Throughput — Tokens/sec",
      "gridPos": { "x": 0, "y": 4, "w": 12, "h": 6 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [
        { "expr": "sum(rate(vllm:generation_tokens_total[1m]))", "legendFormat": "output tok/s" },
        { "expr": "sum(rate(vllm:prompt_tokens_total[1m]))", "legendFormat": "prompt tok/s" }
      ],
      "fieldConfig": { "defaults": { "unit": "none" } }
    },
    {
      "id": 5, "type": "timeseries", "title": "Queue Depth + KV Cache %",
      "gridPos": { "x": 12, "y": 4, "w": 12, "h": 6 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [
        { "expr": "vllm:num_requests_waiting", "legendFormat": "waiting" },
        { "expr": "vllm:num_requests_running", "legendFormat": "running" },
        { "expr": "vllm:gpu_cache_usage_perc * 100", "legendFormat": "KV cache %" }
      ]
    },
    {
      "id": 6, "type": "timeseries", "title": "TTFT Percentiles",
      "gridPos": { "x": 0, "y": 10, "w": 12, "h": 6 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [
        { "expr": "histogram_quantile(0.50, sum(rate(vllm:time_to_first_token_seconds_bucket[1m])) by (le))", "legendFormat": "p50" },
        { "expr": "histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket[1m])) by (le))", "legendFormat": "p95" },
        { "expr": "histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[1m])) by (le))", "legendFormat": "p99" }
      ],
      "fieldConfig": { "defaults": { "unit": "s" } }
    },
    {
      "id": 7, "type": "timeseries", "title": "TPOT Percentiles",
      "gridPos": { "x": 12, "y": 10, "w": 12, "h": 6 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [
        { "expr": "histogram_quantile(0.50, sum(rate(vllm:time_per_output_token_seconds_bucket[1m])) by (le))", "legendFormat": "p50" },
        { "expr": "histogram_quantile(0.95, sum(rate(vllm:time_per_output_token_seconds_bucket[1m])) by (le))", "legendFormat": "p95" },
        { "expr": "histogram_quantile(0.99, sum(rate(vllm:time_per_output_token_seconds_bucket[1m])) by (le))", "legendFormat": "p99" }
      ],
      "fieldConfig": { "defaults": { "unit": "s" } }
    },
    {
      "id": 8, "type": "timeseries", "title": "GPU SM Utilization per GPU (%)",
      "gridPos": { "x": 0, "y": 16, "w": 12, "h": 6 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [{ "expr": "DCGM_FI_DEV_GPU_UTIL", "legendFormat": "GPU {{gpu}}" }],
      "fieldConfig": { "defaults": { "unit": "percent", "min": 0, "max": 100 } }
    },
    {
      "id": 9, "type": "timeseries", "title": "Tensor Core Active (%)",
      "gridPos": { "x": 12, "y": 16, "w": 12, "h": 6 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [{ "expr": "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE * 100", "legendFormat": "GPU {{gpu}}" }],
      "fieldConfig": { "defaults": { "unit": "percent" } }
    },
    {
      "id": 10, "type": "timeseries", "title": "VRAM Used (GiB)",
      "gridPos": { "x": 0, "y": 22, "w": 12, "h": 6 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [{ "expr": "DCGM_FI_DEV_FB_USED / 1024", "legendFormat": "GPU {{gpu}}" }],
      "fieldConfig": { "defaults": { "unit": "decgbytes", "thresholds": { "mode": "absolute", "steps": [{ "color": "green", "value": null }, { "color": "red", "value": 75 }] } } }
    },
    {
      "id": 11, "type": "timeseries", "title": "Memory Bandwidth Active (%)",
      "gridPos": { "x": 12, "y": 22, "w": 12, "h": 6 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [{ "expr": "DCGM_FI_PROF_DRAM_ACTIVE * 100", "legendFormat": "GPU {{gpu}}" }],
      "fieldConfig": { "defaults": { "unit": "percent" } }
    },
    {
      "id": 12, "type": "timeseries", "title": "NVLink TX GB/s",
      "gridPos": { "x": 0, "y": 28, "w": 12, "h": 6 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [{ "expr": "rate(DCGM_FI_PROF_NVLINK_TX_BYTES[1m]) / 1e9", "legendFormat": "GPU {{gpu}} TX" }],
      "fieldConfig": { "defaults": { "unit": "GBs" } }
    },
    {
      "id": 13, "type": "timeseries", "title": "NVLink RX GB/s",
      "gridPos": { "x": 12, "y": 28, "w": 12, "h": 6 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [{ "expr": "rate(DCGM_FI_PROF_NVLINK_RX_BYTES[1m]) / 1e9", "legendFormat": "GPU {{gpu}} RX" }],
      "fieldConfig": { "defaults": { "unit": "GBs" } }
    },
    {
      "id": 14, "type": "stat", "title": "ECC Errors (total)",
      "gridPos": { "x": 0, "y": 34, "w": 8, "h": 3 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [{ "expr": "sum(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL + DCGM_FI_DEV_ECC_DBE_VOL_TOTAL)", "legendFormat": "ECC Errors" }],
      "options": { "reduceOptions": { "calcs": ["lastNotNull"] }, "colorMode": "background" },
      "fieldConfig": { "defaults": { "thresholds": { "mode": "absolute", "steps": [{ "color": "green", "value": null }, { "color": "red", "value": 1 }] } } }
    },
    {
      "id": 15, "type": "timeseries", "title": "GPU Power Draw (W)",
      "gridPos": { "x": 0, "y": 37, "w": 8, "h": 6 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [{ "expr": "DCGM_FI_DEV_POWER_USAGE", "legendFormat": "GPU {{gpu}}" }],
      "fieldConfig": { "defaults": { "unit": "watt", "thresholds": { "mode": "absolute", "steps": [{ "color": "green", "value": null }, { "color": "red", "value": 680 }] } } }
    },
    {
      "id": 16, "type": "timeseries", "title": "GPU Temperature (°C)",
      "gridPos": { "x": 8, "y": 37, "w": 8, "h": 6 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [{ "expr": "DCGM_FI_DEV_GPU_TEMP", "legendFormat": "GPU {{gpu}}" }],
      "fieldConfig": { "defaults": { "unit": "celsius", "thresholds": { "mode": "absolute", "steps": [{ "color": "green", "value": null }, { "color": "yellow", "value": 75 }, { "color": "red", "value": 83 }] } } }
    },
    {
      "id": 17, "type": "timeseries", "title": "SM Clock (MHz)",
      "gridPos": { "x": 16, "y": 37, "w": 8, "h": 6 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [{ "expr": "DCGM_FI_DEV_SM_CLOCK", "legendFormat": "GPU {{gpu}}" }],
      "fieldConfig": { "defaults": { "unit": "none", "custom": { "displayName": "MHz" } } }
    }
  ]
}
```

- [ ] **Step 4: Verify Grafana starts with provisioned dashboard**

```bash
cd gpu-inference-bench
source .env
docker compose up grafana prometheus node-exporter -d
sleep 20
curl -s -u "admin:${GRAFANA_ADMIN_PASSWORD}" \
  http://localhost:3000/api/dashboards/uid/gpu-inference-bench \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK:', d['dashboard']['title'])"
```
Expected: `OK: GPU Inference Benchmark`

- [ ] **Step 5: Stop services**

```bash
docker compose stop grafana prometheus node-exporter
```

- [ ] **Step 6: Commit**

```bash
git add gpu-inference-bench/grafana/
git commit -m "feat: Grafana provisioning and 7-row GPU inference dashboard"
```

---

### Task 5: Locust Load Generator

**Files:**
- Create: `gpu-inference-bench/locust/Dockerfile`
- Create: `gpu-inference-bench/locust/locustfile.py`

**Interfaces:**
- Consumes: vLLM at `http://vllm:8000/v1/chat/completions`
- Produces: client-side latency at Locust web UI `:8089`; server-side metrics via vLLM `/metrics`

- [ ] **Step 1: Create `locust/Dockerfile`**

Create `gpu-inference-bench/locust/Dockerfile`:
```dockerfile
FROM locustio/locust:2.24.0
RUN pip install --no-cache-dir httpx
```

- [ ] **Step 2: Create `locust/locustfile.py`**

Create `gpu-inference-bench/locust/locustfile.py`:
```python
import random
from locust import HttpUser, task, between

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."

# (prompt, max_output_tokens) — short-medium I/O targeting ~50-200 input, ~100-400 output tokens
MESSAGES = [
    ("What is gradient descent?", 256),
    ("Explain transformer attention in 3 sentences.", 300),
    ("What is the difference between L1 and L2 regularization?", 256),
    ("How does KV caching improve LLM inference throughput?", 400),
    ("What is tensor parallelism and why does NVLink matter for it?", 400),
    ("Explain the difference between TTFT and TPOT in LLM serving.", 300),
    ("What is vLLM's PagedAttention and what problem does it solve?", 400),
    ("Summarize the key tradeoffs between beam search and sampling.", 350),
    ("What is quantization and how does FP8 affect model quality?", 300),
    ("How does prefix caching reduce TTFT for repeated prompts?", 350),
]


class ChatUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task
    def chat_completion(self):
        prompt, max_tokens = random.choice(MESSAGES)
        with self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False,
            },
            timeout=60,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
            else:
                data = response.json()
                if not data.get("choices"):
                    response.failure("No choices in response")
                else:
                    response.success()
```

- [ ] **Step 3: Build Locust image**

```bash
cd gpu-inference-bench && docker compose build locust
```
Expected: build completes, no errors.

- [ ] **Step 4: Verify Locust web UI starts**

```bash
docker compose run --rm -p 8089:8089 locust \
  locust -f /mnt/locust/locustfile.py --host http://localhost:8000 --web-port 8089 &
sleep 5
curl -sf http://localhost:8089/ | grep -qi locust && echo "Locust UI OK"
kill %1 2>/dev/null || true
```
Expected: `Locust UI OK`

- [ ] **Step 5: Commit**

```bash
git add gpu-inference-bench/locust/
git commit -m "feat: Locust chat completion load generator"
```

---

### Task 6: Failure Injection + Recovery Scripts

**Files:**
- Create: `gpu-inference-bench/scripts/inject_failure.sh`
- Create: `gpu-inference-bench/scripts/recover.sh`
- Create: `gpu-inference-bench/scripts/collect_evidence.sh`

**Interfaces:**
- Consumes: Docker service `gpu-inference-bench-vllm-1`; Grafana admin API at `:3000`; Prometheus at `:9090`
- Produces: `postmortem/evidence-<timestamp>/` with JSON metric snapshots; Grafana annotation markers for `T_inject`, `T_detect`, `T_recover`

- [ ] **Step 1: Create `scripts/inject_failure.sh`**

Create `gpu-inference-bench/scripts/inject_failure.sh`:
```bash
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
```

- [ ] **Step 2: Create `scripts/recover.sh`**

Create `gpu-inference-bench/scripts/recover.sh`:
```bash
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
```

- [ ] **Step 3: Create `scripts/collect_evidence.sh`**

Create `gpu-inference-bench/scripts/collect_evidence.sh`:
```bash
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
```

- [ ] **Step 4: Make all scripts executable**

```bash
chmod +x gpu-inference-bench/scripts/inject_failure.sh \
         gpu-inference-bench/scripts/recover.sh \
         gpu-inference-bench/scripts/collect_evidence.sh
```

- [ ] **Step 5: Dry-run pgrep check (vLLM must be running)**

After vLLM is fully loaded (Task 8 Step 2), run before live injection:
```bash
docker exec gpu-inference-bench-vllm-1 pgrep -a python | grep -E "local-rank|vllm.entrypoints"
```
Expected: lines showing `--local-rank=0`, `--local-rank=1`, `--local-rank=2`, `--local-rank=3`.
If only `vllm.entrypoints` appears (no local-rank args): the fallback path in `inject_failure.sh` will activate — acceptable.

- [ ] **Step 6: Test Grafana annotation API**

```bash
cd gpu-inference-bench && source .env
curl -sf -X POST \
  -H "Content-Type: application/json" \
  -u "admin:${GRAFANA_ADMIN_PASSWORD}" \
  "http://localhost:${GRAFANA_PORT}/api/annotations" \
  -d '{"text":"test annotation","tags":["test"]}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('Annotation ID:', r['id'])"
```
Expected: `Annotation ID: <number>`

- [ ] **Step 7: Commit**

```bash
git add gpu-inference-bench/scripts/
git commit -m "feat: inject_failure, recover, and collect_evidence scripts"
```

---

### Task 7: Postmortem Template

**Files:**
- Create: `gpu-inference-bench/postmortem/template.md`

- [ ] **Step 1: Create `postmortem/template.md`**

Create `gpu-inference-bench/postmortem/template.md`:
```markdown
# Postmortem: vLLM Tensor Parallel Rank Failure

**Date:** 2026-10-10
**Severity:** SEV-2
**Duration:** ___ min
**Author:** ___

---

## Incident Summary

A single GPU worker process (TP rank 1) was killed on a 4xH100 vLLM server running
Llama-3.3-70B-Instruct with tensor-parallel-size=4. vLLM has no partial fault tolerance
for TP rank loss — the entire inference server exited within ~5 seconds. All in-flight
requests failed. Service was unavailable until manual restart (~___ min).

**Impact:**
- Requests failed: ___ (from Locust CSV export)
- SLO breach window: ___ min (p99 TTFT > 2s)
- Data loss: None (stateless inference)

---

## Timeline

| Time (T+) | Event |
|-----------|-------|
| T+0:00 | Locust started, ramp to 50 concurrent users |
| T+2:00 | Baseline stable — p99 TTFT: ___s, output tok/s: ___ |
| T+5:00 | `inject_failure.sh` executed — rank 1 SIGKILL sent |
| T+5:___ | vLLM process exited (NCCL collective broken) |
| T+5:___ | Prometheus: `up{job="vllm"}` → 0 |
| T+5:___ | Alert fired: `VLLMDown` or `P99TTFTBreached` |
| T+7:00 | `collect_evidence.sh` ran — metrics snapshot saved |
| T+7:30 | `recover.sh` ran — `docker compose restart vllm` |
| T+9:___ | vLLM health check passed, NVLink BW restored |
| T+10:__ | p99 TTFT back under 2s SLO |
| T+12:00 | Locust stopped |

---

## Root Cause

vLLM tensor parallelism (TP=4) distributes each transformer layer across 4 GPU workers
using NCCL all-reduce over NVLink. Every forward pass requires a synchronized collective
across all ranks. When rank 1 exited via SIGKILL, the remaining ranks deadlocked waiting
for rank 1's all-reduce response. The vLLM main process raised `RuntimeError: NCCL error`
and the entire server exited.

**There is no partial fault tolerance in synchronous tensor parallelism.** Losing 1 of N
TP ranks terminates the server. This is a property of the collective-communication model,
not a vLLM defect.

---

## Contributing Factors

1. **No auto-restart:** vLLM service had no `restart: unless-stopped` — recovery required manual intervention
2. **No pre-crash rank monitoring:** Rank failure manifested as sudden process exit, not degraded state — no leading indicator
3. **30s healthcheck interval:** Prometheus alert fired up to 60s after actual failure (2× scrape interval)

---

## Dashboard Evidence

_Attach Grafana panel screenshots for:_

- [ ] SM utilization cliff (Row 4) — all 4 GPUs drop to 0% at T_inject annotation
- [ ] NVLink TX GB/s (Row 6) — all lines drop to 0 at T_inject
- [ ] p99 TTFT stat (Row 1) — goes red / shows no data at T_inject
- [ ] Request failure rate (Row 1) — spikes to 100%
- [ ] Queue depth (Row 2) — spike then drain

_Evidence JSON: `postmortem/evidence-<timestamp>/`_

---

## What Went Well

- DCGM captured all 4 GPU SM drops simultaneously — failure visible in Grafana immediately
- Grafana `T_inject` annotation marked exact moment across all rows
- `collect_evidence.sh` captured full 15-min window for all key metrics
- `recover.sh` restored service — full NVLink fabric came back clean

---

## Action Items

| # | Action | Owner | Priority |
|---|--------|-------|----------|
| 1 | Add `restart: unless-stopped` to vLLM in `docker-compose.yml` | SRE | P1 |
| 2 | Add Prometheus alert: SM util avg < 10% when prior avg > 50% | SRE | P1 |
| 3 | Reduce vLLM healthcheck interval from 30s to 10s | SRE | P2 |
| 4 | Evaluate vLLM disaggregated prefill for partial fault isolation | Eng | P2 |
| 5 | Document TP rank failure blast radius in GPU inference runbook | Eng | P2 |

---

## Metrics Summary

| Metric | Baseline (T+2) | During Failure | Post-Recovery (T+10) |
|--------|----------------|----------------|----------------------|
| p99 TTFT | ___s | N/A (down) | ___s |
| Output tok/s | ___ | 0 | ___ |
| NVLink TX avg | ___ GB/s | 0 GB/s | ___ GB/s |
| GPU SM util avg | ___% | 0% | ___% |
| Request failure rate | 0% | 100% | 0% |
```

- [ ] **Step 2: Commit**

```bash
git add gpu-inference-bench/postmortem/template.md
git commit -m "docs: postmortem template for TP rank failure experiment"
```

---

### Task 8: Full Stack Smoke Test + Baseline

Run on the RunPod 4xH100 SXM host. All prior tasks must be committed and files present.

**Prerequisites:** `.env` populated with real `HF_TOKEN`; 500GB network volume mounted.

- [ ] **Step 1: Start all services**

```bash
cd gpu-inference-bench
docker compose up -d
```

- [ ] **Step 2: Wait for model load (~10 min)**

```bash
docker compose logs -f vllm | grep -E "startup complete|ERROR|error"
```
Do not proceed until `Application startup complete.` appears. Model download is ~140GB.

- [ ] **Step 3: Verify actual vLLM metric names**

vLLM metric names vary by version. Before trusting dashboard panels:
```bash
curl -s http://localhost:8000/metrics | grep "^# HELP vllm" | sort
```
Confirm these names exist (or note actual names and update Grafana PromQL):
- `vllm:time_to_first_token_seconds`
- `vllm:time_per_output_token_seconds`
- `vllm:generation_tokens_total`
- `vllm:prompt_tokens_total`
- `vllm:num_requests_running`
- `vllm:num_requests_waiting`
- `vllm:gpu_cache_usage_perc`
- `vllm:request_success_total`
- `vllm:request_total`

If names differ: update affected panels in `grafana/dashboards/gpu-inference.json`, then reload Grafana:
```bash
curl -X POST -u "admin:${GRAFANA_ADMIN_PASSWORD}" http://localhost:3000/api/admin/provisioning/dashboards/reload
```

- [ ] **Step 4: Verify all services healthy**

```bash
source .env
curl -sf http://localhost:${VLLM_PORT}/health      && echo "vLLM OK"
curl -sf http://localhost:${DCGM_PORT}/metrics | grep DCGM_FI_DEV_GPU_UTIL | head -2 && echo "DCGM OK"
curl -sf http://localhost:${NODE_EXPORTER_PORT}/metrics | grep node_cpu | head -1     && echo "node_exporter OK"
curl -sf http://localhost:${PROMETHEUS_PORT}/-/ready                                  && echo "Prometheus OK"
curl -sf http://localhost:${GRAFANA_PORT}/api/health                                  && echo "Grafana OK"
curl -sf http://localhost:${LOCUST_PORT}/                                              && echo "Locust OK"
```
Expected: all 6 print OK.

- [ ] **Step 5: Verify NVLink metrics in Prometheus**

```bash
curl -sG "http://localhost:${PROMETHEUS_PORT}/api/v1/query" \
  --data-urlencode "query=DCGM_FI_PROF_NVLINK_TX_BYTES" \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'NVLink series: {len(r[\"data\"][\"result\"])}')"
```
Expected: `NVLink series: 4`. If 0: DCGM profiling not working — see Task 2 Step 2 debug steps.

- [ ] **Step 6: Run 2-minute baseline**

Open Locust UI at `http://<host>:8089`. Set **Users=50**, **Spawn rate=5**, **Host=http://vllm:8000**. Click Start. Wait 2 min.

Record baseline values (fill into postmortem template):
- p99 TTFT: ___
- Output tok/s: ___
- NVLink TX avg: ___
- GPU SM util avg: ___

---

### Task 9: Failure Injection + Evidence Collection

Execute spec's injection sequence. Follow timing exactly.

- [ ] **Step 1: Confirm stable baseline**

Locust failure rate = 0%. Grafana p99 TTFT panel = green (< 2s).

- [ ] **Step 2: Run dry-run pgrep check (Task 6 Step 5)**

```bash
docker exec gpu-inference-bench-vllm-1 pgrep -a python | grep -E "local-rank|vllm.entrypoints"
```
Confirm which process pattern inject script will use.

- [ ] **Step 3: Inject at T+5:00**

```bash
bash scripts/inject_failure.sh
```

- [ ] **Step 4: Observe Grafana for 2 min**

Within 30s, verify ALL of these:
- SM util all 4 GPUs → 0%
- NVLink TX/RX all → 0
- p99 TTFT → no data or spiked
- Failure rate → 100%
- `T_inject` annotation line visible across rows

- [ ] **Step 5: Collect evidence at T+7:00**

```bash
bash scripts/collect_evidence.sh
```

- [ ] **Step 6: Recover at T+7:30**

```bash
bash scripts/recover.sh
```

- [ ] **Step 7: Verify recovery in Grafana**

After ~2 min:
- SM util: all 4 GPUs return to ~80%
- NVLink TX/RX: bandwidth restores
- p99 TTFT: drops back below 2s
- `T_recover` annotation visible

- [ ] **Step 8: Stop Locust at T+12:00**

Click Stop in Locust UI. Download CSV report (Stats + Failures + Exceptions).

- [ ] **Step 9: Export Grafana dashboard**

In Grafana: Share → Export → Download JSON. Also screenshot each row.
Save to `postmortem/grafana-export-<date>.json`.

- [ ] **Step 10: Write postmortem**

```bash
cp postmortem/template.md postmortem/2026-10-10-tp-rank-failure.md
```
Fill all `___` fields from: Locust CSV + evidence JSON files + Grafana screenshots.

- [ ] **Step 11: Commit everything**

```bash
git add gpu-inference-bench/postmortem/
git commit -m "docs: completed TP rank failure postmortem + evidence 2026-10-10"
```

---

## Self-Review

**Spec coverage:**
- Provider + model + TP=4 + RunPod workaround ✓ (Task 1 compose + .env)
- All DCGM metrics from spec ✓ (Task 2 metrics.csv: SM util, VRAM, BW, tensor core, NVLink, ECC, power, temp, clock)
- All vLLM metrics from spec ✓ (Task 4 panels: TTFT, TPOT, tok/s, queue, KV cache, failure rate)
- All 7 dashboard rows ✓ (panels 1-17 cover SLO, throughput, latency, compute, memory, NVLink, power+thermals)
- SLO alert p99 TTFT > 2s ✓ (Task 3 alerts.yml `P99TTFTBreached`)
- GPU SM drop alert ✓ (Task 3 `GPUSMUtilSuddenDrop`)
- Grafana annotations for T_inject/T_recover ✓ (Task 6 scripts POST to Grafana API)
- Failure injection sequence matches spec timeline ✓ (Task 9 steps)
- Postmortem sections match spec ✓ (Task 7 template: summary, timeline, root cause, contributing factors, evidence, what went well, action items)
- NVLink metrics verification before live run ✓ (Task 8 Step 5)

**Placeholder scan:** No TBD/TODO in code blocks. All `___` are operational fill-ins during live run — not implementation placeholders.

**Type consistency:** Script variables (`GRAFANA_ADMIN_PASSWORD`, `GRAFANA_PORT`, `VLLM_PORT`, `PROMETHEUS_PORT`) sourced from `.env` in Task 1 and used identically across Tasks 6 and 8.

**Review Focus mitigations all present:** NVLink profiling verification in Tasks 2+8; histogram_quantile PromQL throughout Task 4; pgrep dry-run in Tasks 6+9; Grafana API test in Task 6 Step 6; model download wait in Task 8 Step 2.
