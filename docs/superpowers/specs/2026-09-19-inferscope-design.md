# inferscope — LLM Inference Diagnostics CLI

**Date:** 2026-09-19  
**Status:** Draft  
**Goal:** OSS adoption — a tool practitioners reach for when LLM inference breaks in production

---

## Problem

Engineering teams running LLM inference in production spend 30–60 minutes per incident manually correlating signals across kubectl, Grafana, nvidia-smi, and vLLM metrics to find root cause. Existing tools cover individual layers:

- App-layer traces: Langfuse, Arize, OpenLIT
- Metrics/dashboards: Prometheus + Grafana
- GPU metrics: DCGM exporter

No tool bridges all three layers simultaneously — serving engine (KV cache, queue, batch), infrastructure (GPU memory, OOM events, K8s events), and application (output quality, truncation, tool-call failures) — and correlates them in time to produce an actionable root cause.

---

## What inferscope does

`inferscope` is a CLI tool that collects signals from 12 parallel sources, correlates them on a timeline, sends structured anomaly data to an LLM, and produces a root-cause diagnosis with recommended actions. It maintains persistent incident history so teams accumulate diagnostic memory across incidents.

---

## Architecture

```
User
  │
  ▼
inferscope CLI
  │
  ├── Collector Layer (12 collectors, all parallel)
  │
  ├── Aggregator → unified CollectionResult
  │
  ├── Pipeline
  │     ├── Timeline Builder   (1-min buckets, fuzzy alignment ±30s, K8s events as anchors)
  │     ├── Anomaly Detector   (flags deviations, drops stable signals)
  │     └── Signal Formatter   (semantic deltas, 3000-token hard budget, priority truncation)
  │
  ├── LLM Reasoner             (Claude / OpenAI / Ollama)
  │
  ├── State Store              (SQLite / K8s ConfigMap / JSON file)
  │
  └── Reporter                 (terminal / JSON)
```

**Key properties:**
- Stateless binary — no server, no daemon required for Level 1–3 use
- Read-only — never mutates K8s resources
- Zero cluster footprint — runs from laptop with kubeconfig
- Language: Go — single binary, fast startup, client-go ecosystem

---

## Collector Layer (12 collectors)

Every collector implements:

```go
type Collector interface {
    Name() string
    Collect(ctx context.Context, target Target) (CollectorResult, error)
    Available(ctx context.Context, target Target) bool  // skip gracefully if absent
}
```

All collectors normalize output to a universal `Signal` struct:

```go
type Signal struct {
    Name      string
    Value     float64
    Unit      string
    Source    string
    Pod       string
    Node      string
    Timestamp time.Time
    Tags      map[string]string
}
```

### The 12 collectors

| # | Collector | What it collects | Source |
|---|-----------|-----------------|--------|
| 1 | K8s API Client | Pod status, events, restarts, HPA, OOMKilled | K8s API |
| 2 | Metrics Scraper | vLLM/TGI /metrics per pod (kv_cache_usage, latency, queue, TTFT) | Pod /metrics via port-forward |
| 3 | GPU Collector | VRAM used/free, GPU util, XID errors | DCGM exporter → fallback nvidia-smi exec |
| 4 | Prometheus Client | Time-range metric history (optional) | Prometheus HTTP API |
| 5 | Log Collector | CUDA OOM errors, cache misses, engine state, truncation events | kubectl logs + pattern extraction |
| 6 | Model Config Collector | max_model_len, tp_size, quantization, dtype, max_batch_size | Pod args + /v1/models + env vars |
| 7 | Node Collector | Node conditions (MemoryPressure, DiskPressure), GPU device plugin, taints | K8s Node API |
| 8 | Request Distribution | Prompt/output length histograms, TTFT per request class | vLLM request-level metrics |
| 9 | Network Collector | Envoy/Istio connection errors, circuit breaker trips, traffic distribution per replica | Istio metrics (skipped if absent) |
| 10 | Scheduler Collector | CA events, pending pods, ResourceQuota state, HPA desired vs current | kube-system events + HPA API |
| 11 | Thermal / Power | GPU temperature, SM clock throttle, power cap | DCGM (DCGM_FI_DEV_GPU_TEMP, DCGM_FI_DEV_SM_CLOCK, DCGM_FI_DEV_POWER_USAGE) |
| 12 | LoRA Adapter | Adapter load/eviction state, eviction rate, per-adapter load latency | vLLM LoRA API (skipped if no LoRA detected) |

Collectors 9 and 12 skip silently if Istio or LoRA are not detected. All others attempt collection and record errors in `CollectError` without blocking the pipeline.

---

## Data Pipeline

### Timeline Builder

Bins all signals into 1-minute buckets across the lookback window. Computes rolling baseline (first 80% of window) and flags anomalies in the last 20%.

```
14:00  kv_cache=71%  gpu_mem=72%  p99=2.1s  waiting=3
14:10  kv_cache=78%  gpu_mem=79%  p99=2.4s  waiting=8
14:20  kv_cache=85%  gpu_mem=86%  p99=4.1s  waiting=19
14:21  [EVENT] OOMKilled pod vllm-llama3-abc
14:23  kv_cache=89%  gpu_mem=89%  p99=18.4s waiting=47
```

**Fuzzy time alignment (clock-drift problem):**

Different sources have different clock sources and collection delays: Prometheus scrape lag (15–30s), kubectl log timestamps, K8s event `firstTime` vs `lastTime`, DCGM clock. Naive bucketing miscorrelates events within the same window.

Strategy:
- K8s events used as **anchor points** — highest timestamp confidence, used to align other signals
- All other signals aligned relative to anchors with configurable tolerance: `--time-tolerance 30s` (default)
- Each signal carries a `clock_source` tag; approximate correlations are flagged in LLM payload: `[~correlated ±30s]`
- Buckets are open intervals — a signal at 14:21:45 and an event at 14:21:55 land in the same bucket at 30s tolerance

### Anomaly Detector

Flags signals where current value deviates significantly from baseline. Stable signals are dropped — they never reach the LLM payload. This compresses thousands of metric data points to the ~20–40 signals that actually matter for diagnosis.

### Signal Formatter

Produces **semantic delta sentences**, not raw data. Hard token budget of 3000 tokens.

```
INSTEAD OF (raw):
  kv_cache_usage_perc: [71, 72, 74, 75, 78, 81, 85, 87, 89]

EMITS (semantic):
  kv_cache_usage_perc spiked +25% above baseline, reaching 89% at 14:23 UTC

INSTEAD OF (raw logs):
  [log line 1] ... [log line 12]

EMITS (pattern):
  12x "prefix cache miss" between 14:10–14:23 (rate: 1/min → 4/min, accelerating)
```

**Token budget with priority ordering (truncate lowest first):**

```
1. K8s Events        (chronological, deduplicated)           — never truncated
2. Metric anomalies  (semantic delta sentences)               — truncated last
3. Log patterns      (count + time range, not raw lines)      — truncated second
4. Model config      (key serving parameters)                 — truncated third
5. History summary   (prior incidents, last resolution)       — truncated first

If budget exceeded → append:
  "[N lower-priority signals omitted — run with --verbose for full output]"
```

**Example LLM payload:**

```
DEPLOYMENT:  vllm-llama3 | namespace: inference | window: 30m
MODEL:       llama-3-70b | tp=4 | max_model_len=32768 | quant=awq

ANOMALOUS SIGNALS:
  kv_cache_usage_perc   spiked +25% above baseline (71% → 89%) at 14:23 UTC
  e2e_latency_p99       spiked +776% above baseline (2.1s → 18.4s) at 14:23 UTC
  requests_waiting      rose +1467% above baseline (3 → 47) from 14:20 UTC
  gpu_temp              rose +18% on node-7 (71°C → 84°C), SM clock throttle active
  lora_evictions        new: 3 evictions/min since 14:19 UTC (was 0)

EVENTS (chronological):
  14:21 UTC  OOMKilled     pod vllm-llama3-abc (node-7)
  14:22 UTC  BackOff       pod vllm-llama3-abc restart
  14:23 UTC  ScalingFailed HPA: desired=4 current=3 (ResourceQuota exceeded)

LOGS (patterns):
  3x  "CUDA out of memory" at 14:21 UTC
  12x "prefix cache miss"  between 14:10–14:23 (accelerating 1/min → 4/min)
  2x  "LoRA adapter evicted: adapter-finance" at 14:19 UTC

HISTORY: 3 incidents in 7 days — same root cause. Last resolution: added replica (Sep 16), temporary.

COLLECTORS UNAVAILABLE: prometheus (time-range history limited to snapshot)
```

---

## LLM Reasoner

### Interface

```go
type LLMClient interface {
    Diagnose(ctx context.Context, payload SignalSummary) (Finding, error)
}
```

### Backends

```bash
--llm=claude   # Claude API (default, best reasoning)
--llm=ollama   # Local Ollama (private, free, offline)
--llm=openai   # OpenAI API
```

### Output schema (structured JSON from LLM)

```json
{
  "root_cause": "kv_cache_eviction_pressure",
  "confidence": "high",
  "reasoning": "...",
  "contributing_factors": ["thermal_throttle_node7", "no_prefix_caching"],
  "recommended_actions": [
    "Enable --enable-prefix-caching",
    "Reduce max_model_len 32768 → 16384",
    "Investigate node-7 cooling (GPU at 84°C)",
    "Add 1 replica to reduce per-replica memory pressure"
  ]
}
```

LLM is forced to return structured JSON via function-calling / tool-use. No free-text parsing.

---

## State Store

### Interface

```go
type StateStore interface {
    SaveDiagnosis(d Diagnosis) error
    GetHistory(deployment, namespace string, since time.Time) ([]Diagnosis, error)
    GetTrends(namespace string, window time.Duration) (Trends, error)
    MarkResolved(id string, resolution string) error
}
```

### Backends

| Backend | Default | Flag | Use case |
|---------|---------|------|---------|
| SQLite | ✅ | `~/.inferscope/history.db` | Single engineer |
| K8s ConfigMap | — | `--store=k8s` | Shared team history |
| JSON file | — | `--store=file ./history.json` | CI/CD, GitOps |

### Schema (SQLite)

```sql
CREATE TABLE diagnoses (
  id           TEXT PRIMARY KEY,
  timestamp    DATETIME,
  deployment   TEXT,
  namespace    TEXT,
  root_cause   TEXT,
  confidence   TEXT,
  factors      JSON,
  actions      JSON,
  resolved_at  DATETIME,
  resolution   TEXT
);
```

---

## CLI Commands

```bash
# Core
inferscope diagnose   --deployment <name> --namespace <ns> [--window 30m] [--llm claude] [--output json]
inferscope watch      --deployment <name> --namespace <ns> [--interval 60s] [--alert-webhook <url>]
inferscope incidents  --deployment <name> --namespace <ns> [--since 7d] [--output json]
inferscope trends     --namespace <ns> [--window 30d]

# Utility
inferscope check      # verify cluster + LLM connectivity
inferscope setup      # create RBAC + shared ConfigMap state store
inferscope config     # manage ~/.inferscope/config.yaml
```

### `diagnose` terminal output

```
inferscope v0.1 — vllm-llama3 (inference)  [12 collectors, 2.3s]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROOT CAUSE  [HIGH]
  KV cache eviction pressure causing latency spike.
  GPU memory on node-7 at 89%, SM clock throttled to 67% base.
  OOMKilled at 14:21. Requests queued 47 deep by 14:23.

EVIDENCE
  kv_cache_usage   89%  (↑ from 71% baseline)
  p99_latency      18.4s (↑ from 2.1s, spike 14:23 UTC)
  gpu_temp         84°C  → thermal throttle active
  requests_waiting 47    (↑ from 3 baseline)
  logs             3x CUDA OOM, 12x prefix cache miss

ACTIONS
  1. Enable --enable-prefix-caching
  2. Reduce max_model_len 32768 → 16384
  3. Check node-7 cooling — GPU at 84°C
  4. Add 1 replica to relieve memory pressure

HISTORY   ↻ Same root cause 3x in 7 days (last: Sep 16 14:23)
TREND     GPU memory: 71% → 83% → 89% over 7 days
          ⚠ OOM threshold projected in ~2 days
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Healthy — no issues found |
| 1 | Issues found |
| 2 | Collection error (can't reach cluster or metrics) |

---

## Integration Levels

### Level 1: Zero setup (day-1 adoption)

```bash
brew install inferscope
export INFERSCOPE_LLM_KEY=sk-...
inferscope diagnose --deployment vllm-llama3 --namespace inference
```

Requirements: kubeconfig with read access + LLM API key. Zero cluster changes.

### Level 2: Team setup (shared history)

```bash
inferscope setup --namespace inference
# creates ServiceAccount, ClusterRole (read-only), ConfigMap state store
```

### Level 3: CI/CD gate

```yaml
- name: Post-deploy inference health check
  run: |
    inferscope diagnose --deployment $DEPLOYMENT \
      --namespace inference --output json --window 10m > diagnosis.json
    jq -e '.confidence != "high"' diagnosis.json
```

### Level 4: Continuous watch (Helm)

```bash
helm install inferscope inferscope/inferscope \
  --namespace inferscope-system \
  --set llm.apiKey=$CLAUDE_API_KEY \
  --set watch.deployments="{vllm-llama3,tgi-codegen}" \
  --set alert.webhook=$SLACK_WEBHOOK
```

---

## Configuration

```yaml
# ~/.inferscope/config.yaml
llm:
  backend: claude
  api_key: $INFERSCOPE_LLM_KEY
  model: claude-haiku-4-5-20251001

store:
  backend: local              # local | k8s | file
  path: ~/.inferscope/history.db

prometheus:
  url: http://prometheus.monitoring:9090   # optional
  timeout: 10s                            # per-query timeout
  max_parallel_queries: 20               # all metrics queried concurrently
  fallback_on_timeout: true              # fall back to /metrics scrape if >50% queries timeout
  # Adaptive step size targets ~30 data points per metric regardless of window:
  # window=30m → step=1m | window=1h → step=2m | window=6h → step=12m

collectors:
  window: 30m
  log_lines: 500
  redact: false
  time_tolerance: 30s                    # fuzzy alignment tolerance across clock sources

watch:
  interval: 60s
  alert_threshold: medium
  webhook: ""
```

---

## Project Structure

```
inferscope/
├── cmd/inferscope/main.go
├── internal/
│   ├── cli/               # command handlers (diagnose, watch, incidents, trends, check, setup)
│   ├── collector/         # 12 collectors + aggregator + Collector interface
│   ├── pipeline/          # timeline.go, anomaly.go, formatter.go
│   ├── llm/               # LLMClient interface + claude.go, openai.go, ollama.go
│   ├── store/             # StateStore interface + sqlite.go, k8s.go, file.go
│   ├── reporter/          # terminal.go, json.go
│   └── config/            # config.go
├── deploy/
│   ├── helm/inferscope/   # Level 4 Helm chart
│   └── rbac.yaml          # Level 2 RBAC manifests
├── go.mod
└── README.md
```

---

## Key Dependencies

```
k8s.io/client-go              K8s API
prometheus/client_model        /metrics parsing
anthropic-sdk-go               Claude API
openai/openai-go               OpenAI API
modernc.org/sqlite             Pure Go SQLite (no CGO)
spf13/cobra                    CLI framework
spf13/viper                    Config + env vars
charmbracelet/lipgloss         Terminal formatting
```

---

## Testing Strategy

| Layer | Approach |
|-------|---------|
| Pipeline (timeline, anomaly, formatter) | Pure unit tests |
| Collectors | envtest mock K8s + mock /metrics HTTP server |
| LLM client | Mock LLMClient returning fixture Finding |
| State store | SQLite in-memory |
| E2E | kind cluster + mock vLLM metrics endpoint |
| LLM payload | Golden file tests — assert compact output |

---

## Distribution

```
GitHub Releases   darwin/linux × amd64/arm64
Homebrew          brew install inferscope/tap/inferscope
Docker            ghcr.io/inferscope/inferscope:latest  (watch mode)
Helm chart        helm repo add inferscope https://inferscope.github.io/helm-charts
```

---

## Supported backends (v0.1)

| Inference | GPU | K8s |
|-----------|-----|-----|
| vLLM | NVIDIA (DCGM + nvidia-smi) | EKS / GKE / AKS / on-prem |
| TGI | — | Any kubeconfig |
| Ollama | — | — |

TGI v0.1, Triton + SGLang v0.2.

---

## v0.2 scope (out of v0.1)

- Triton Inference Server + SGLang collector
- AMD ROCm GPU support
- LLM-generated remediation runbooks (not just action bullets)
- Argo Rollouts native integration
- Web UI for incident history
- Multi-cluster support
