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
