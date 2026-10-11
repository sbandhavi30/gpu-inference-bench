# GPU Inference Benchmark — Design Spec

**Date:** 2026-10-10
**Status:** Approved

---

## Intent

Benchmark chat-completion inference on a 4x H100 SXM (NVLink) host using vLLM + Llama-3.3-70B-Instruct with tensor parallelism across all 4 GPUs. Collect full-stack GPU and inference metrics in a Grafana dashboard. Inject a deliberate TP rank failure, observe the impact, recover, and produce a written postmortem.

**Success criteria:**
- Stable baseline: p99 TTFT < 2s at 50 concurrent users
- All GPU, NVLink, and vLLM metrics visible in Grafana before failure injection
- Dashboard annotations mark injection, detection, and recovery precisely
- Postmortem identifies root cause, blast radius, and action items

---

## Provider + Model

| Decision | Choice | Reason |
|---|---|---|
| Provider | RunPod Secure Cloud | Fastest startup (0.4 min), 7% cheaper than Lambda, full H100 perf |
| Instance | 4x H100 SXM | NVLink fabric required for TP all-reduce bandwidth measurement |
| Model | Llama-3.3-70B-Instruct | Open, fits 4xH100 at BF16, well-tested on vLLM TP=4 |
| Serving | vLLM TP=4 | Tensor parallel splits model across 4 GPUs |
| Workaround | `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0` | Required on RunPod for vLLM official image |

---

## Architecture

Single RunPod host. Docker Compose — no external dependencies.

```
RunPod 4x H100 SXM Host
├── vLLM :8000          (TP=4, OpenAI-compatible API + /metrics)
├── DCGM Exporter :9400 (GPU hardware metrics, NVLink profiling)
├── node_exporter :9100 (CPU, RAM, NVMe, network)
├── Prometheus :9090    (scrapes all three, 5s interval)
├── Grafana :3000       (pre-provisioned dashboards)
└── Locust :8089        (load generator, chat-shaped workload)
```

NVLink all-reduce traffic is generated organically by vLLM's TP communication between GPU ranks during each forward pass.

---

## Directory Layout

```
gpu-inference-bench/
├── docker-compose.yml
├── .env
├── dcgm/
│   └── metrics.csv               # custom DCGM field IDs (NVLink + tensor core)
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/prometheus.yml
│   │   └── dashboards/dashboard.yml
│   └── dashboards/
│       └── gpu-inference.json
├── locust/
│   ├── locustfile.py
│   └── Dockerfile
├── scripts/
│   ├── inject_failure.sh         # kill -9 vLLM GPU rank 1 process
│   ├── recover.sh                # docker compose restart vllm
│   └── collect_evidence.sh       # snapshot Prometheus metrics to JSON
└── postmortem/
    └── template.md
```

---

## Metrics Collected

### GPU Hardware (DCGM Exporter)
- SM utilization per GPU (0–3)
- GPU memory used / total per GPU
- Memory bandwidth utilization %
- Tensor core active cycles (BF16/FP16)
- NVLink TX / RX bytes per link per GPU
- NVLink error counters (correctable + uncorrectable)
- ECC errors (single-bit / double-bit)
- GPU power draw (vs 700W TDP)
- GPU temperature per GPU
- SM clock / memory clock (throttle detection)
- PCIe host bandwidth

### vLLM /metrics (Prometheus)
- TTFT p50 / p95 / p99
- TPOT p50 / p95 / p99
- End-to-end request latency p50 / p95 / p99
- Tokens generated/sec
- Prompt tokens/sec
- Running requests
- Waiting queue depth
- KV cache usage %
- Preemption count
- Request success / failure rate
- Prefix cache hit rate

### System (node_exporter)
- CPU utilization
- System RAM
- NVMe throughput
- Network I/O

### Load Generator (Locust)
- RPS over time
- Active concurrent users
- Client-side p50 / p95 / p99 / p99.9 latency
- Failure rate %

---

## Dashboard Layout (Grafana)

7 rows, top-to-bottom by incident urgency:

| Row | Panels |
|---|---|
| SLO Tracker | p99 TTFT vs 2s target, failure rate, alert badge |
| Inference Throughput | tok/s, RPS, queue depth, KV cache % |
| Latency Distribution | TTFT p50/p95/p99, TPOT p50/p95/p99 |
| GPU Compute | SM util × 4, tensor core active % overlay |
| GPU Memory | VRAM used × 4 (stacked, 80GiB ceiling), memory BW % × 4 |
| NVLink Fabric | TX GB/s × 4, RX GB/s × 4, error counters (red if > 0) |
| Power + Thermals + Load | Power × 4 vs TDP, temp × 4, SM clock, Locust users + client p99 |

Vertical annotations (`T_inject`, `T_detect`, `T_recover`) pushed to Prometheus by scripts, displayed across all rows.

**SLO:** p99 TTFT < 2s, measured from vLLM /metrics.

---

## Failure Injection Sequence

```
T+0:00   Start Locust, ramp to 50 concurrent users
T+2:00   Baseline stable — record p99 TTFT, tok/s, NVLink BW
T+5:00   inject_failure.sh → kill -9 GPU rank 1 worker
T+5:00   vLLM TP coordination breaks — server exits
T+5:0x   Prometheus scrape miss → Grafana panels red
T+5:0x   Locust: 100% failure rate, p99 TTFT → ∞
T+7:00   collect_evidence.sh → metrics snapshot to JSON
T+7:30   recover.sh → docker compose restart vllm
T+9:00   vLLM healthy, NVLink BW restores, all 4 GPUs active
T+10:02  p99 TTFT < 2s SLO restored
T+12:00  Stop load test, export dashboard PDF
```

**Expected dashboard signal at T+5:00:**

| Panel | Before | After |
|---|---|---|
| SM util GPU1 | ~85% | 0% (all 4 GPUs drop) |
| NVLink TX GPU1 | ~180 GB/s | 0 |
| p99 TTFT | ~0.8s | no data |
| Queue depth | ~12 | spikes then drains |
| Request failure rate | 0% | 100% |

---

## Postmortem Structure

File: `postmortem/2026-10-10-tp-rank-failure.md`

Sections:
1. **Incident Summary** — severity, duration, impact
2. **Timeline** — T+0 through T+12, annotated
3. **Impact** — failed request count, SLO breach window, data loss assessment
4. **Root Cause** — vLLM TP=4 has no partial fault tolerance; single rank loss exits whole server
5. **Contributing Factors** — no auto-restart, no pre-crash rank health monitoring
6. **Dashboard Evidence** — annotated screenshots of SM util cliff, NVLink TX drop, p99 TTFT spike
7. **What Went Well** — DCGM captured drop instantly, annotations precise, recovery < 2 min manual
8. **Action Items** — restart policy, SM util drop alert, disaggregated prefill evaluation, blast radius docs

---

## Constraints

- All services run on the RunPod host itself — no egress to external Prometheus/Grafana
- DCGM requires `--privileged` container or direct host NVIDIA driver access
- vLLM model download (~140GB BF16) billed as startup time — use RunPod network volume or pre-cached image
- NVLink profiling metrics (`DCGM_FI_PROF_*`) require DCGM profiling permissions; must run as root or with `CAP_SYS_ADMIN`
- Locust runs on same host — subtract ~4 CPU cores from system metrics baseline

---

## Out of Scope

- Multi-node inference (Ray cluster)
- Prefix caching optimization
- Quantized model variants (FP8, AWQ)
- Kubernetes deployment
- Persistent storage for metrics beyond experiment window
