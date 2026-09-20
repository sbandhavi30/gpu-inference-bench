# KEDA Token-Aware External Scaler — Design Spec

**Date:** 2026-09-20
**Status:** Approved for implementation

---

## Problem

KEDA scales Kubernetes workloads on message count (SQS `ApproximateNumberOfMessagesVisible`, Kafka consumer lag). For LLM batch inference, message count is the wrong signal. Ten requests at 100k tokens each requires 10× more GPU capacity than 100 requests at 50 tokens. Scaling on message count causes OOM or chronic GPU underutilization.

**Scope:** Batch inference jobs (offline embeddings, document summarization, structured extraction). Explicitly out of scope: open-ended agentic loops where execution density is non-deterministic from the entry payload.

---

## Solution

A lightweight Go gRPC server implementing KEDA's `ExternalScaler` interface. It reads a token backlog from Redis (populated by the upstream caller via a dual-write contract) and returns the total token sum as an `AverageValue` metric. KEDA hands this to the HPA controller, which scales vLLM pods accordingly.

---

## Architecture

```
Upstream Client
  │  publishes job → SQS / Kafka
  │  HSET token_backlog:{queue} {job_id} {token_count}  → Redis  (Script A)
  │  ZADD token_expiry:{queue} {expiry_epoch} {job_id}  → Redis
  │  PUBLISH token_events:{queue} "active"              → Redis Pub/Sub (if backlog was empty)
  ▼
SQS / Kafka ──────────────────────────────────────────────────────┐
                                                                   │
Redis ◄─── token-scaler (gRPC :9090) ◄─── KEDA polls GetMetrics  │
           │                                                       │
           │  cleanup expired → HVALS sum → return total_tokens   │
           │  Prometheus /metrics :8080                            │
           ▼                                                       │
      vLLM pods (HPA target) ◄─────────────────────────────────── ┘
           │
           │  on completion: HDEL + ZREM  → Redis (caller contract)
```

**Two ports:**
- `:9090` — gRPC (KEDA polling path, high-performance)
- `:8080` — HTTP `/metrics` (Prometheus scrape, isolated)

---

## Caller Contract

The scaler ships no SDK. Upstream applications must implement the dual-write protocol.

### Publisher (on job submit) — Script A (atomic Lua)

```lua
-- KEYS[1]: token_backlog:{queue}
-- KEYS[2]: token_expiry:{queue}
-- KEYS[3]: token_events:{queue}
-- ARGV[1]: job_id, ARGV[2]: token_count, ARGV[3]: expiry_epoch, ARGV[4]: safety_ttl

local was_empty = redis.call("HLEN", KEYS[1]) == 0
redis.call("HSET", KEYS[1], ARGV[1], ARGV[2])
redis.call("ZADD", KEYS[2], ARGV[3], ARGV[1])
redis.call("PERSIST", KEYS[1])
redis.call("PERSIST", KEYS[2])
if was_empty then
    redis.call("PUBLISH", KEYS[3], "active")
end
return 1
```

- `token_count` = `metadata.estimated_tokens` integer from job payload. Upstream computes this using tiktoken or HuggingFace tokenizers before publishing. The scaler never tokenizes raw strings.
- `expiry_epoch` = `now + jobTimeoutSeconds`
- `PERSIST` removes any dangling key-level TTL from a previous empty state.
- `PUBLISH` fires only when backlog transitions empty → non-empty (no polling amplification).

### Worker (on job complete)

```
HDEL token_backlog:{queue} {job_id}
ZREM token_expiry:{queue} {job_id}
```

### Payload schema requirement

```json
{
  "job_id": "uuid",
  "payload": "...",
  "metadata": {
    "estimated_tokens": 42500
  }
}
```

---

## Redis Schema

| Key | Type | Contents |
|---|---|---|
| `token_backlog:{queue}` | Hash | `job_id → token_count` |
| `token_expiry:{queue}` | Sorted Set | `job_id` scored by expiry epoch |
| `token_events:{queue}` | Pub/Sub channel | `"active"` published on empty→non-empty transition |

---

## gRPC Interface

Implements KEDA's `ExternalScaler` proto exactly (vendored, not modified).

### Methods

**`IsActive`**
- `HLEN token_backlog:{queue} > 0` → true
- Enables scale-from-zero.
- On Redis error: return `codes.Internal`. KEDA freezes current replica count. If already at zero, stays at zero.
- Records `token_scaler_healthy{phase="is_active"} = 0` on error for alerting.

**`GetMetricSpec`**
- Returns metric name `token_backlog`, type `AverageValue`, target = `tokensPerReplica` from `scalerMetadata`.
- `AverageValue` is critical: HPA divides `total_token_backlog` by current running pod count before comparing to target. `TargetValue` (absolute) breaks math at scale.

**`GetMetrics`**
- Executes Script B (atomic Lua evaluator) in Redis.
- Returns raw `total_token_backlog` int64.
- On Redis error: `codes.Internal` + `token_scaler_healthy{phase="get_metrics"} = 0`.

**`StreamIsActive`**
- Subscribes to `token_events:{queue}` Pub/Sub (blocking, no polling).
- 50 ScaledObjects = 50 subscriptions on one multiplexed Redis connection, not 50 timer goroutines.
- On Redis connection drop: return `codes.Unavailable`. Reconnect on next KEDA retry.

### Scaling formula

```
desired_replicas = ceil(total_token_backlog / tokensPerReplica)
```

KEDA applies min/max replica bounds on top.

---

## Script B — Atomic Scaler Evaluator

Runs inside `GetMetrics`. Single Lua execution = zero race conditions between cleanup and summation.

```lua
-- KEYS[1]: token_backlog:{queue}, KEYS[2]: token_expiry:{queue}
-- ARGV[1]: now_epoch, ARGV[2]: safety_ttl, ARGV[3]: chunk_size

local expired_ids = redis.call("ZRANGEBYSCORE", KEYS[2], 0, ARGV[1], "LIMIT", 0, tonumber(ARGV[3]))
if #expired_ids > 0 then
    for i = 1, #expired_ids do
        redis.call("HDEL", KEYS[1], expired_ids[i])
        redis.call("ZREM", KEYS[2], expired_ids[i])
    end
end

local total_tokens = 0
for _, val in ipairs(redis.call("HVALS", KEYS[1])) do
    total_tokens = total_tokens + tonumber(val)
end

if redis.call("HLEN", KEYS[1]) == 0 then
    redis.call("EXPIRE", KEYS[1], ARGV[2])
    redis.call("EXPIRE", KEYS[2], ARGV[2])
end

return total_tokens
```

- `LIMIT 0, chunk_size` (default 100): caps cleanup per cycle. Prevents single-threaded Redis blocking on mass expiry events. Remainder cleaned next KEDA poll (~5s later).
- `EXPIRE` on empty keys: self-healing memory protection against orphaned structures from crashed workers.
- Scripts loaded at startup via `SCRIPT LOAD`, called by SHA. On `NOSCRIPT` error (Redis restart), `client.go` re-loads and retries once.

---

## Pluggable Queue Adapter Interface

Used for health checking and drift telemetry only. **Not on the critical scaling path.**

```go
package adapter

import "context"

type QueueTelemetry struct {
    ApproximateDepth int64
    IsHealthy        bool
}

type QueueAdapter interface {
    HealthCheck(ctx context.Context, config map[string]string) error
    FetchTelemetry(ctx context.Context, config map[string]string) (QueueTelemetry, error)
}
```

### SQS adapter
- `HealthCheck`: `GetQueueAttributes` on `queueUrl` (not queue name — avoids runtime `GetQueueUrl` calls).
- `FetchTelemetry`: reads `ApproximateNumberOfMessagesVisible`.
- Auth: env vars or IRSA (workload identity). No hardcoded credentials.

### Kafka adapter
- `HealthCheck`: connect to broker, verify topic exists.
- `FetchTelemetry`: `ReadHighWatermarks` on topic partitions. **Not `DescribeGroups`** — watermark fetch is an order of magnitude cheaper and does not stress the Kafka controller.

`ApproximateDepth` exposed as `token_scaler_queue_depth_approx{queue}` Prometheus metric. If Redis backlog diverges significantly from queue depth, the caller contract is broken. Alert on sustained divergence.

Any queue adapter failure degrades observability only — `GetMetrics` continues unaffected.

---

## ScaledObject Metadata Contract

### Required fields (both queue types)

| Key | Description |
|---|---|
| `queueType` | `"sqs"` or `"kafka"` — adapter selector |
| `queueName` | Redis key suffix: `token_backlog:{queueName}` |
| `redisAddress` | Redis endpoint (`host:port`) |
| `tokensPerReplica` | Scaling denominator (string integer) |
| `jobTimeoutSeconds` | Per-ScaledObject TTL override (default: `defaultJobTimeoutSeconds`) |

### SQS-specific

| Key | Description |
|---|---|
| `queueUrl` | Fully qualified SQS URL (required; missing → `InvalidArgument` at startup) |
| `awsRegion` | AWS region |

### Kafka-specific

| Key | Description |
|---|---|
| `kafkaBrokers` | Bootstrap servers |
| `kafkaTopic` | Topic name for watermark fetch |

### Example SQS ScaledObject

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-batch-scaler
spec:
  scaleTargetRef:
    name: vllm-deployment
  minReplicaCount: 0
  maxReplicaCount: 10
  triggers:
    - type: external
      metadata:
        scalerAddress: token-scaler.keda.svc:9090
        queueType: sqs
        queueName: batch-jobs
        queueUrl: "https://sqs.us-east-1.amazonaws.com/123456789/batch-jobs"
        awsRegion: us-east-1
        redisAddress: redis.default.svc:6379
        tokensPerReplica: "500000"
        jobTimeoutSeconds: "3600"
```

---

## Project Structure

```
token-scaler/
├── cmd/token-scaler/
│   ├── main.go               # entrypoint, flag parsing, wiring
│   └── main_test.go          # integration tests (testcontainers)
├── internal/
│   ├── scaler/
│   │   ├── scaler.go         # ExternalScalerServer gRPC impl
│   │   ├── metrics.go        # Prometheus registry, RecordQueueDrift(), /healthz
│   │   ├── scaler_test.go
│   │   └── gen/              # generated gRPC Go code (committed, no protoc required to build)
│   ├── redis/
│   │   ├── client.go         # connection pool, NOSCRIPT retry, ExecuteEvaluator()
│   │   ├── scripts.go        # ScriptA, ScriptB constants + ScriptManager + LoadScripts()
│   │   └── client_test.go
│   └── adapter/
│       ├── adapter.go        # QueueAdapter interface + QueueTelemetry
│       ├── sqs/
│       │   ├── sqs.go
│       │   └── sqs_test.go
│       └── kafka/
│           ├── kafka.go
│           └── kafka_test.go
├── proto/
│   └── externalscaler.proto  # vendored KEDA proto
├── deploy/helm/token-scaler/
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── serviceaccount.yaml
│       └── scaled-object-example.yaml
├── docs/
│   └── contract.md           # caller contract spec
├── Dockerfile
├── go.mod
└── Makefile
```

---

## Helm Configuration

### `values.yaml`

```yaml
image:
  repository: ghcr.io/your-org/token-scaler
  tag: latest
  pullPolicy: IfNotPresent

grpc:
  port: 9090

metrics:
  port: 8080

redis:
  address: ""           # REQUIRED
  passwordSecretRef:
    name: ""
    key: ""

resources:
  requests:
    cpu: 50m
    memory: 64Mi
  limits:
    cpu: 200m
    memory: 128Mi

defaultJobTimeoutSeconds: 3600
cleanupChunkSize: 100
```

### Environment variables (injected by deployment.yaml)

```yaml
env:
  - name: REDIS_ADDRESS
    value: "{{ .Values.redis.address }}"
  - name: CLEANUP_CHUNK_SIZE
    value: "{{ .Values.cleanupChunkSize }}"
  - name: DEFAULT_JOB_TIMEOUT_SECONDS
    value: "{{ .Values.defaultJobTimeoutSeconds }}"
  - name: GRPC_PORT
    value: "{{ .Values.grpc.port }}"
  - name: METRICS_PORT
    value: "{{ .Values.metrics.port }}"
```

`cleanupChunkSize` flows: `values.yaml` → env var → Go flag → Script B `ARGV[3]`. No hardcoded constants in Lua.

### Probes

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
readinessProbe:
  grpc:
    port: 9090     # native K8s gRPC probe (K8s 1.24+)
```

### RBAC

None required. Scaler holds no K8s API access. ServiceAccount exists only for IRSA annotation (SQS workload identity).

---

## Error Handling

| Failure | Behavior |
|---|---|
| Redis unreachable on `GetMetrics` | `codes.Internal` → KEDA holds current replica count; `token_scaler_healthy{phase="get_metrics"} = 0` |
| Redis unreachable on `IsActive` | `codes.Internal` → KEDA freezes; stays at zero if already zero; `token_scaler_healthy{phase="is_active"} = 0` |
| Redis drops during `StreamIsActive` | `codes.Unavailable` → KEDA retries; pool self-heals on reconnect |
| `NOSCRIPT` error (Redis restart) | `client.go` re-loads scripts → retries once |
| `queueName` missing | `codes.InvalidArgument` at first call |
| `queueUrl` missing (SQS) | `codes.InvalidArgument` at startup validation |
| Queue adapter `FetchTelemetry` failure | Log + `IsHealthy: false` metric; does NOT affect `GetMetrics` |

**Alert rule:** `token_scaler_healthy == 0` for > 30s → page.

---

## Testing Strategy

### Unit tests

| File | Coverage |
|---|---|
| `redis/client_test.go` | Script B: empty, partial expiry, 100-ceiling, 101 entries (2 cycles), NOSCRIPT retry — via `miniredis` |
| `scaler/scaler_test.go` | Missing metadata → `InvalidArgument`; Redis error → `Internal` + health metric = 0; normal path AverageValue correct |
| `adapter/sqs/sqs_test.go` | `queueUrl` required; throttling error; mock AWS SDK |
| `adapter/kafka/kafka_test.go` | Watermark fetch; broker unreachable; mock Kafka client |

### Integration tests (`cmd/.../main_test.go`, `testcontainers-go`)

| Test | Infrastructure |
|---|---|
| Full lifecycle: publish 3 jobs → GetMetrics → expire 1 → sum decrements | Real Redis container |
| Cleanup ceiling: 105 expired → 100 purged cycle 1, 5 purged cycle 2 | Real Redis container |
| HPA boundary: 510k vs 560k backlog against 500k `tokensPerReplica` — verify no precision truncation | Real Redis container |
| Network fracture: partition mid-`StreamIsActive` → `codes.Unavailable` (not hang) → heal → self-recover | Redis + toxiproxy container |

Network partition via `testcontainers/modules/toxiproxy`. Scaler dials toxiproxy address; test adds/removes timeout toxic to simulate partition + heal.

---

## Key Constraints

- No SDK shipped. Caller contract is a documentation + schema contract only.
- Scaler is stateless beyond Redis. Multiple replicas safe.
- Batch inference jobs only. Agentic loops explicitly unsupported (non-deterministic token density from entry payload).
- Cloud-agnostic. No AWS-specific defaults baked in.
- `internal/` only — no exported packages. This is a binary, not a library.
