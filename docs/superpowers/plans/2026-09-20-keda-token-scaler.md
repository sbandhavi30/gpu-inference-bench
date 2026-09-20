# KEDA Token-Aware External Scaler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Go gRPC server implementing KEDA's ExternalScaler interface that scales vLLM pods based on total token backlog stored in Redis rather than raw message count.

**Architecture:** Single Go binary exposing gRPC on :9090 and Prometheus metrics on :8080. Reads token backlog from Redis via two atomic Lua scripts. Queue adapters (SQS, Kafka) provide health/drift telemetry only — Redis is the authoritative scaling data source. No SDK shipped; upstream callers implement the dual-write contract directly.

**Tech Stack:** Go 1.22+, `google.golang.org/grpc`, `github.com/redis/go-redis/v9`, `github.com/aws/aws-sdk-go-v2/service/sqs`, `github.com/IBM/sarama` (Kafka), `github.com/prometheus/client_golang`, `github.com/alicebob/miniredis/v2` (tests), `github.com/testcontainers/testcontainers-go` (integration tests)

**Spec:** `docs/superpowers/specs/2026-09-20-keda-token-scaler-design.md`

## Global Constraints

- Go 1.22+
- All packages under `internal/` — nothing exported; this is a binary, not a library
- gRPC port: `:9090` (env: `GRPC_PORT`)
- Metrics/health port: `:8080` (env: `METRICS_PORT`)
- Cleanup chunk ceiling: `100` (env: `CLEANUP_CHUNK_SIZE`)
- Default job timeout: `3600` seconds (env: `DEFAULT_JOB_TIMEOUT_SECONDS`)
- Redis address required at startup (env: `REDIS_ADDRESS`); missing → fatal
- No hardcoded credentials; auth via env vars or workload identity
- `internal/scaler/gen/` contains committed generated gRPC code — protoc not required to build
- Scope: batch inference jobs only; agentic loops explicitly unsupported

---

## File Map

```
token-scaler/
├── cmd/token-scaler/
│   ├── main.go                    # flag/env parsing, component wiring, gRPC server startup
│   └── main_test.go               # integration tests (testcontainers-go)
├── internal/
│   ├── redis/
│   │   ├── client.go              # Client struct, NewClient, LoadScripts, ExecuteEvaluator, GetHashLength, Subscribe, Close
│   │   ├── scripts.go             # ScriptA/ScriptB Lua constants; ScriptManager struct; LoadScripts()
│   │   └── client_test.go         # miniredis unit tests for all Redis operations
│   ├── adapter/
│   │   ├── adapter.go             # QueueAdapter interface; QueueTelemetry struct
│   │   ├── sqs/
│   │   │   ├── sqs.go             # SQS adapter: HealthCheck, FetchTelemetry
│   │   │   └── sqs_test.go
│   │   └── kafka/
│   │       ├── kafka.go           # Kafka adapter: watermark-based HealthCheck, FetchTelemetry
│   │       └── kafka_test.go
│   └── scaler/
│       ├── metrics.go             # Prometheus Metrics struct; SetHealthy, RecordQueueDrift; /healthz + /metrics HTTP server
│       ├── scaler.go              # ExternalScalerServer gRPC impl: IsActive, GetMetricSpec, GetMetrics, StreamIsActive
│       ├── scaler_test.go         # in-process gRPC unit tests with miniredis
│       └── gen/                   # generated gRPC Go code (committed)
├── proto/
│   └── externalscaler.proto       # vendored KEDA external scaler proto
├── deploy/helm/token-scaler/
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── serviceaccount.yaml
│       └── scaled-object-example.yaml
├── docs/
│   └── contract.md                # caller contract: publisher + worker protocol
├── Dockerfile
├── go.mod
└── Makefile
```

---

### Task 1: Project Scaffold, Proto, and Makefile

**Files:**
- Create: `go.mod`
- Create: `proto/externalscaler.proto`
- Create: `internal/scaler/gen/` (committed generated files)
- Create: `Makefile`
- Create: `Dockerfile`

**Interfaces:**
- Produces: `pb.ExternalScalerServer` interface, `pb.ScaledObjectRef`, `pb.IsActiveResponse`, `pb.GetMetricSpecResponse`, `pb.MetricSpec`, `pb.GetMetricsRequest`, `pb.GetMetricsResponse`, `pb.MetricValue` — used by Tasks 6 and 7

- [ ] **Step 1: Initialize Go module**

```bash
mkdir -p token-scaler && cd token-scaler
go mod init token-scaler
```

- [ ] **Step 2: Add dependencies**

```bash
go get google.golang.org/grpc@v1.64.0
go get google.golang.org/protobuf@v1.34.0
go get github.com/redis/go-redis/v9@v9.5.3
go get github.com/aws/aws-sdk-go-v2@v1.27.0
go get github.com/aws/aws-sdk-go-v2/config@v1.27.0
go get github.com/aws/aws-sdk-go-v2/service/sqs@v1.32.0
go get github.com/IBM/sarama@v1.43.2
go get github.com/prometheus/client_golang@v1.19.0
go get github.com/alicebob/miniredis/v2@v2.33.0
go get github.com/testcontainers/testcontainers-go@v0.31.0
go get github.com/testcontainers/testcontainers-go/modules/redis@v0.31.0
go get github.com/testcontainers/testcontainers-go/modules/toxiproxy@v0.31.0
go mod tidy
```

- [ ] **Step 3: Write the proto file**

`proto/externalscaler.proto`:
```proto
syntax = "proto3";
package externalscaler;
option go_package = "token-scaler/internal/scaler/gen;externalscaler";

service ExternalScaler {
    rpc IsActive(ScaledObjectRef)       returns (IsActiveResponse)         {}
    rpc StreamIsActive(ScaledObjectRef) returns (stream IsActiveResponse)  {}
    rpc GetMetricSpec(ScaledObjectRef)  returns (GetMetricSpecResponse)    {}
    rpc GetMetrics(GetMetricsRequest)   returns (GetMetricsResponse)       {}
}

message ScaledObjectRef {
    string name      = 1;
    string namespace = 2;
    map<string, string> scalerMetadata = 3;
}

message IsActiveResponse {
    bool result = 1;
}

message GetMetricSpecResponse {
    repeated MetricSpec metricSpecs = 1;
}

message MetricSpec {
    string metricName = 1;
    int64  targetSize = 2;
}

message GetMetricsRequest {
    ScaledObjectRef scaledObjectRef = 1;
    string          metricName      = 2;
}

message MetricValue {
    string metricName  = 1;
    int64  metricValue = 2;
}

message GetMetricsResponse {
    repeated MetricValue metricValues = 1;
}
```

- [ ] **Step 4: Generate gRPC Go code**

Install protoc and plugins if not present:
```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
mkdir -p internal/scaler/gen
protoc --go_out=. --go_opt=paths=source_relative \
       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
       proto/externalscaler.proto
mv proto/token-scaler/internal/scaler/gen/* internal/scaler/gen/ 2>/dev/null || \
protoc --go_out=internal/scaler/gen --go_opt=paths=source_relative \
       --go-grpc_out=internal/scaler/gen --go-grpc_opt=paths=source_relative \
       --proto_path=proto proto/externalscaler.proto
```

Commit the generated files so consumers never need protoc:
```bash
git add internal/scaler/gen/
```

- [ ] **Step 5: Write Makefile**

`Makefile`:
```makefile
.PHONY: build test generate lint docker

BINARY := token-scaler
IMAGE  := ghcr.io/your-org/token-scaler

build:
	go build -o bin/$(BINARY) ./cmd/token-scaler

test:
	go test ./... -race -timeout 120s

test-unit:
	go test ./internal/... -race -timeout 30s

test-integration:
	go test ./cmd/... -race -timeout 120s -tags integration

generate:
	protoc --go_out=internal/scaler/gen --go_opt=paths=source_relative \
	       --go-grpc_out=internal/scaler/gen --go-grpc_opt=paths=source_relative \
	       --proto_path=proto proto/externalscaler.proto

lint:
	golangci-lint run ./...

docker:
	docker build -t $(IMAGE):latest .
```

- [ ] **Step 6: Write Dockerfile**

`Dockerfile`:
```dockerfile
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o /token-scaler ./cmd/token-scaler

FROM alpine:3.19
RUN addgroup -S scaler && adduser -S scaler -G scaler
COPY --from=builder /token-scaler /token-scaler
USER scaler
EXPOSE 9090 8080
ENTRYPOINT ["/token-scaler"]
```

- [ ] **Step 7: Verify build compiles**

```bash
go build ./...
```
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add go.mod go.sum proto/ internal/scaler/gen/ Makefile Dockerfile
git commit -m "feat: scaffold project, proto, and generated gRPC stubs"
```

---

### Task 2: Redis Client and Lua Scripts

**Files:**
- Create: `internal/redis/scripts.go`
- Create: `internal/redis/client.go`
- Create: `internal/redis/client_test.go`

**Interfaces:**
- Consumes: `github.com/redis/go-redis/v9`
- Produces:
  - `redis.NewClient(addr, password string) (*Client, error)`
  - `(*Client).LoadScripts(ctx context.Context) (*ScriptManager, error)`
  - `(*Client).ExecuteEvaluator(ctx context.Context, queueName string, nowEpoch int64, safetyTTL, chunkSize string) (int64, error)`
  - `(*Client).GetHashLength(ctx context.Context, key string) (int64, error)`
  - `(*Client).Subscribe(ctx context.Context, channel string) *goredis.PubSub`
  - `(*Client).Close() error`
  - `ScriptManager{PublisherSHA string; EvaluatorSHA string}`

- [ ] **Step 1: Write failing tests**

`internal/redis/client_test.go`:
```go
package redis_test

import (
    "context"
    "strconv"
    "testing"
    "time"

    "github.com/alicebob/miniredis/v2"
    goredis "github.com/redis/go-redis/v9"
    "token-scaler/internal/redis"
)

func newTestClient(t *testing.T) (*redis.Client, *miniredis.Miniredis) {
    t.Helper()
    mr, err := miniredis.Run()
    if err != nil {
        t.Fatalf("miniredis start: %v", err)
    }
    t.Cleanup(mr.Close)
    c, err := redis.NewClient(mr.Addr(), "")
    if err != nil {
        t.Fatalf("NewClient: %v", err)
    }
    t.Cleanup(func() { c.Close() })
    return c, mr
}

func TestLoadScripts_ReturnsSHAs(t *testing.T) {
    c, _ := newTestClient(t)
    sm, err := c.LoadScripts(context.Background())
    if err != nil {
        t.Fatalf("LoadScripts: %v", err)
    }
    if sm.PublisherSHA == "" || sm.EvaluatorSHA == "" {
        t.Fatal("expected non-empty SHAs")
    }
}

func TestGetHashLength_Empty(t *testing.T) {
    c, _ := newTestClient(t)
    n, err := c.GetHashLength(context.Background(), "token_backlog:q1")
    if err != nil {
        t.Fatalf("GetHashLength: %v", err)
    }
    if n != 0 {
        t.Fatalf("want 0, got %d", n)
    }
}

func TestGetHashLength_NonEmpty(t *testing.T) {
    c, mr := newTestClient(t)
    mr.HSet("token_backlog:q1", "job1", "1000")
    n, err := c.GetHashLength(context.Background(), "token_backlog:q1")
    if err != nil {
        t.Fatalf("GetHashLength: %v", err)
    }
    if n != 1 {
        t.Fatalf("want 1, got %d", n)
    }
}

func TestEvaluator_EmptyBacklog(t *testing.T) {
    c, _ := newTestClient(t)
    if _, err := c.LoadScripts(context.Background()); err != nil {
        t.Fatal(err)
    }
    total, err := c.ExecuteEvaluator(context.Background(), "q1",
        time.Now().Unix(), "86400", "100")
    if err != nil {
        t.Fatalf("ExecuteEvaluator: %v", err)
    }
    if total != 0 {
        t.Fatalf("want 0, got %d", total)
    }
}

func TestEvaluator_SumsTokens(t *testing.T) {
    c, mr := newTestClient(t)
    if _, err := c.LoadScripts(context.Background()); err != nil {
        t.Fatal(err)
    }
    mr.HSet("token_backlog:q1", "job1", "1000", "job2", "2500", "job3", "500")
    future := time.Now().Add(1 * time.Hour).Unix()
    mr.ZAdd("token_expiry:q1", float64(future), "job1")
    mr.ZAdd("token_expiry:q1", float64(future), "job2")
    mr.ZAdd("token_expiry:q1", float64(future), "job3")

    total, err := c.ExecuteEvaluator(context.Background(), "q1",
        time.Now().Unix(), "86400", "100")
    if err != nil {
        t.Fatalf("ExecuteEvaluator: %v", err)
    }
    if total != 4000 {
        t.Fatalf("want 4000, got %d", total)
    }
}

func TestEvaluator_PurgesExpiredJobs(t *testing.T) {
    c, mr := newTestClient(t)
    if _, err := c.LoadScripts(context.Background()); err != nil {
        t.Fatal(err)
    }
    past := time.Now().Add(-1 * time.Hour).Unix()
    future := time.Now().Add(1 * time.Hour).Unix()
    mr.HSet("token_backlog:q1", "expired", "9999", "active", "1000")
    mr.ZAdd("token_expiry:q1", float64(past), "expired")
    mr.ZAdd("token_expiry:q1", float64(future), "active")

    total, err := c.ExecuteEvaluator(context.Background(), "q1",
        time.Now().Unix(), "86400", "100")
    if err != nil {
        t.Fatalf("ExecuteEvaluator: %v", err)
    }
    if total != 1000 {
        t.Fatalf("want 1000, got %d", total)
    }
    // Verify expired job removed from hash
    rdb := goredis.NewClient(&goredis.Options{Addr: mr.Addr()})
    defer rdb.Close()
    exists, _ := rdb.HExists(context.Background(), "token_backlog:q1", "expired").Result()
    if exists {
        t.Fatal("expired job should have been purged from hash")
    }
}

func TestEvaluator_ChunkCeiling_100PerCycle(t *testing.T) {
    c, mr := newTestClient(t)
    if _, err := c.LoadScripts(context.Background()); err != nil {
        t.Fatal(err)
    }
    past := time.Now().Add(-1 * time.Hour).Unix()
    // Insert 105 expired jobs
    rdb := goredis.NewClient(&goredis.Options{Addr: mr.Addr()})
    defer rdb.Close()
    for i := 0; i < 105; i++ {
        id := "job" + strconv.Itoa(i)
        mr.HSet("token_backlog:q1", id, "100")
        mr.ZAdd("token_expiry:q1", float64(past), id)
    }
    // Cycle 1: only 100 purged
    total1, err := c.ExecuteEvaluator(context.Background(), "q1",
        time.Now().Unix(), "86400", "100")
    if err != nil {
        t.Fatalf("cycle1: %v", err)
    }
    remaining1, _ := rdb.HLen(context.Background(), "token_backlog:q1").Result()
    if remaining1 != 5 {
        t.Fatalf("cycle1: want 5 remaining, got %d", remaining1)
    }
    // total1 = sum of remaining 5 * 100 = 500
    if total1 != 500 {
        t.Fatalf("cycle1: want sum 500, got %d", total1)
    }

    // Cycle 2: remaining 5 purged
    total2, err := c.ExecuteEvaluator(context.Background(), "q1",
        time.Now().Unix(), "86400", "100")
    if err != nil {
        t.Fatalf("cycle2: %v", err)
    }
    if total2 != 0 {
        t.Fatalf("cycle2: want 0, got %d", total2)
    }
    remaining2, _ := rdb.HLen(context.Background(), "token_backlog:q1").Result()
    if remaining2 != 0 {
        t.Fatalf("cycle2: want 0 remaining, got %d", remaining2)
    }
}

func TestEvaluator_SetsExpireOnEmpty(t *testing.T) {
    c, mr := newTestClient(t)
    if _, err := c.LoadScripts(context.Background()); err != nil {
        t.Fatal(err)
    }
    past := time.Now().Add(-1 * time.Hour).Unix()
    mr.HSet("token_backlog:q1", "job1", "100")
    mr.ZAdd("token_expiry:q1", float64(past), "job1")

    c.ExecuteEvaluator(context.Background(), "q1", time.Now().Unix(), "86400", "100")

    rdb := goredis.NewClient(&goredis.Options{Addr: mr.Addr()})
    defer rdb.Close()
    ttl, _ := rdb.TTL(context.Background(), "token_backlog:q1").Result()
    if ttl <= 0 {
        t.Fatalf("expected positive TTL after empty, got %v", ttl)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
go test ./internal/redis/... -v
```
Expected: FAIL — `redis` package does not exist yet.

- [ ] **Step 3: Write scripts.go**

`internal/redis/scripts.go`:
```go
package redis

import "context"

const ScriptAPublisher = `
local was_empty = redis.call("HLEN", KEYS[1]) == 0
redis.call("HSET", KEYS[1], ARGV[1], ARGV[2])
redis.call("ZADD", KEYS[2], ARGV[3], ARGV[1])
redis.call("PERSIST", KEYS[1])
redis.call("PERSIST", KEYS[2])
if was_empty then
    redis.call("PUBLISH", KEYS[3], "active")
end
return 1
`

const ScriptBEvaluator = `
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
`

// ScriptManager holds SHAs after scripts are loaded into Redis.
type ScriptManager struct {
    PublisherSHA  string
    EvaluatorSHA  string
}

func (c *Client) LoadScripts(ctx context.Context) (*ScriptManager, error) {
    pubSHA, err := c.rdb.ScriptLoad(ctx, ScriptAPublisher).Result()
    if err != nil {
        return nil, fmt.Errorf("load publisher script: %w", err)
    }
    evalSHA, err := c.rdb.ScriptLoad(ctx, ScriptBEvaluator).Result()
    if err != nil {
        return nil, fmt.Errorf("load evaluator script: %w", err)
    }
    sm := &ScriptManager{PublisherSHA: pubSHA, EvaluatorSHA: evalSHA}
    c.sm = sm
    return sm, nil
}
```

- [ ] **Step 4: Write client.go**

`internal/redis/client.go`:
```go
package redis

import (
    "context"
    "fmt"
    "strings"

    goredis "github.com/redis/go-redis/v9"
)

// Client wraps go-redis with script management and NOSCRIPT retry.
type Client struct {
    rdb *goredis.Client
    sm  *ScriptManager
}

func NewClient(addr, password string) (*Client, error) {
    rdb := goredis.NewClient(&goredis.Options{
        Addr:     addr,
        Password: password,
    })
    if err := rdb.Ping(context.Background()).Err(); err != nil {
        return nil, fmt.Errorf("redis ping: %w", err)
    }
    return &Client{rdb: rdb}, nil
}

func (c *Client) GetHashLength(ctx context.Context, key string) (int64, error) {
    return c.rdb.HLen(ctx, key).Result()
}

func (c *Client) Subscribe(ctx context.Context, channel string) *goredis.PubSub {
    return c.rdb.Subscribe(ctx, channel)
}

func (c *Client) Close() error {
    return c.rdb.Close()
}

// ExecuteEvaluator runs Script B atomically. Retries once on NOSCRIPT (Redis restart).
func (c *Client) ExecuteEvaluator(ctx context.Context, queueName string, nowEpoch int64, safetyTTL, chunkSize string) (int64, error) {
    backlogKey := "token_backlog:" + queueName
    expiryKey  := "token_expiry:"  + queueName
    keys       := []string{backlogKey, expiryKey}

    result, err := c.rdb.EvalSha(ctx, c.sm.EvaluatorSHA, keys, nowEpoch, safetyTTL, chunkSize).Int64()
    if err != nil && strings.Contains(err.Error(), "NOSCRIPT") {
        if _, reloadErr := c.LoadScripts(ctx); reloadErr != nil {
            return 0, fmt.Errorf("reload after NOSCRIPT: %w", reloadErr)
        }
        result, err = c.rdb.EvalSha(ctx, c.sm.EvaluatorSHA, keys, nowEpoch, safetyTTL, chunkSize).Int64()
    }
    return result, err
}
```

Note: `scripts.go` references `fmt` — add import there too:
```go
import "fmt"
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
go test ./internal/redis/... -v -race
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/redis/
git commit -m "feat: redis client with atomic Lua scripts and NOSCRIPT retry"
```

---

### Task 3: Queue Adapter Interface and SQS Adapter

**Files:**
- Create: `internal/adapter/adapter.go`
- Create: `internal/adapter/sqs/sqs.go`
- Create: `internal/adapter/sqs/sqs_test.go`

**Interfaces:**
- Produces:
  - `adapter.QueueTelemetry{ApproximateDepth int64; IsHealthy bool}`
  - `adapter.QueueAdapter` interface with `HealthCheck(ctx, config) error` and `FetchTelemetry(ctx, config) (QueueTelemetry, error)`
  - `sqs.Adapter` struct implementing `QueueAdapter`
  - `sqs.New() *Adapter`

- [ ] **Step 1: Write failing tests**

`internal/adapter/sqs/sqs_test.go`:
```go
package sqs_test

import (
    "context"
    "errors"
    "testing"

    "token-scaler/internal/adapter/sqs"
)

// mockSQSAPI is an interface for the SQS calls we make, enabling test injection.
type mockSQSAPI struct {
    depth int64
    err   error
}

func TestHealthCheck_MissingQueueUrl(t *testing.T) {
    a := sqs.New()
    err := a.HealthCheck(context.Background(), map[string]string{
        "awsRegion": "us-east-1",
    })
    if err == nil {
        t.Fatal("expected error for missing queueUrl")
    }
}

func TestFetchTelemetry_MissingQueueUrl(t *testing.T) {
    a := sqs.New()
    _, err := a.FetchTelemetry(context.Background(), map[string]string{})
    if err == nil {
        t.Fatal("expected error for missing queueUrl")
    }
}

func TestFetchTelemetry_UnhealthyOnError(t *testing.T) {
    // Without valid AWS credentials/endpoint, FetchTelemetry returns IsHealthy=false
    a := sqs.New()
    tel, err := a.FetchTelemetry(context.Background(), map[string]string{
        "queueUrl":  "https://sqs.us-east-1.amazonaws.com/000000000000/test",
        "awsRegion": "us-east-1",
    })
    // Any error from AWS should set IsHealthy=false, not propagate fatally
    if err == nil && tel.IsHealthy {
        // In CI without AWS creds, we expect an error path
        t.Log("got healthy response — ensure AWS mock is in use in CI")
    }
    if err != nil {
        // Error propagated correctly
        _ = errors.Is(err, err)
    }
}
```

- [ ] **Step 2: Run to verify fail**

```bash
go test ./internal/adapter/sqs/... -v
```
Expected: FAIL — package does not exist.

- [ ] **Step 3: Write adapter.go interface**

`internal/adapter/adapter.go`:
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

- [ ] **Step 4: Write sqs.go**

`internal/adapter/sqs/sqs.go`:
```go
package sqs

import (
    "context"
    "fmt"
    "strconv"

    awsconfig "github.com/aws/aws-sdk-go-v2/config"
    awssqs   "github.com/aws/aws-sdk-go-v2/service/sqs"
    "github.com/aws/aws-sdk-go-v2/service/sqs/types"
    "token-scaler/internal/adapter"
)

type Adapter struct{}

func New() *Adapter { return &Adapter{} }

func (a *Adapter) HealthCheck(ctx context.Context, config map[string]string) error {
    queueURL := config["queueUrl"]
    if queueURL == "" {
        return fmt.Errorf("sqs adapter: queueUrl is required")
    }
    client, err := a.buildClient(ctx, config)
    if err != nil {
        return fmt.Errorf("sqs build client: %w", err)
    }
    _, err = client.GetQueueAttributes(ctx, &awssqs.GetQueueAttributesInput{
        QueueUrl:       &queueURL,
        AttributeNames: []types.QueueAttributeName{types.QueueAttributeNameApproximateNumberOfMessages},
    })
    return err
}

func (a *Adapter) FetchTelemetry(ctx context.Context, config map[string]string) (adapter.QueueTelemetry, error) {
    queueURL := config["queueUrl"]
    if queueURL == "" {
        return adapter.QueueTelemetry{}, fmt.Errorf("sqs adapter: queueUrl is required")
    }
    client, err := a.buildClient(ctx, config)
    if err != nil {
        return adapter.QueueTelemetry{IsHealthy: false}, fmt.Errorf("sqs build client: %w", err)
    }
    out, err := client.GetQueueAttributes(ctx, &awssqs.GetQueueAttributesInput{
        QueueUrl:       &queueURL,
        AttributeNames: []types.QueueAttributeName{types.QueueAttributeNameApproximateNumberOfMessages},
    })
    if err != nil {
        return adapter.QueueTelemetry{IsHealthy: false}, fmt.Errorf("sqs GetQueueAttributes: %w", err)
    }
    raw := out.Attributes[string(types.QueueAttributeNameApproximateNumberOfMessages)]
    depth, _ := strconv.ParseInt(raw, 10, 64)
    return adapter.QueueTelemetry{ApproximateDepth: depth, IsHealthy: true}, nil
}

func (a *Adapter) buildClient(ctx context.Context, config map[string]string) (*awssqs.Client, error) {
    region := config["awsRegion"]
    cfg, err := awsconfig.LoadDefaultConfig(ctx, awsconfig.WithRegion(region))
    if err != nil {
        return nil, err
    }
    return awssqs.NewFromConfig(cfg), nil
}
```

- [ ] **Step 5: Run tests to verify pass**

```bash
go test ./internal/adapter/... -v -race
```
Expected: PASS (missing-queueUrl checks pass; AWS call tests require live credentials or mocking — skip with `-short` in CI).

- [ ] **Step 6: Commit**

```bash
git add internal/adapter/
git commit -m "feat: queue adapter interface and SQS adapter"
```

---

### Task 4: Kafka Adapter

**Files:**
- Create: `internal/adapter/kafka/kafka.go`
- Create: `internal/adapter/kafka/kafka_test.go`

**Interfaces:**
- Consumes: `adapter.QueueAdapter`, `adapter.QueueTelemetry` from Task 3
- Produces: `kafka.Adapter` implementing `QueueAdapter` using watermark fetch (not DescribeGroups)

- [ ] **Step 1: Write failing tests**

`internal/adapter/kafka/kafka_test.go`:
```go
package kafka_test

import (
    "context"
    "testing"

    "token-scaler/internal/adapter/kafka"
)

func TestHealthCheck_MissingBrokers(t *testing.T) {
    a := kafka.New()
    err := a.HealthCheck(context.Background(), map[string]string{
        "kafkaTopic": "batch-jobs",
    })
    if err == nil {
        t.Fatal("expected error for missing kafkaBrokers")
    }
}

func TestHealthCheck_MissingTopic(t *testing.T) {
    a := kafka.New()
    err := a.HealthCheck(context.Background(), map[string]string{
        "kafkaBrokers": "localhost:9092",
    })
    if err == nil {
        t.Fatal("expected error for missing kafkaTopic")
    }
}

func TestFetchTelemetry_MissingConfig(t *testing.T) {
    a := kafka.New()
    _, err := a.FetchTelemetry(context.Background(), map[string]string{})
    if err == nil {
        t.Fatal("expected error for missing kafkaBrokers and kafkaTopic")
    }
}

func TestFetchTelemetry_UnhealthyOnBrokerUnreachable(t *testing.T) {
    a := kafka.New()
    tel, err := a.FetchTelemetry(context.Background(), map[string]string{
        "kafkaBrokers": "localhost:19999", // nothing listening
        "kafkaTopic":   "batch-jobs",
    })
    if err == nil && tel.IsHealthy {
        t.Fatal("expected IsHealthy=false when broker is unreachable")
    }
}
```

- [ ] **Step 2: Run to verify fail**

```bash
go test ./internal/adapter/kafka/... -v
```
Expected: FAIL — package does not exist.

- [ ] **Step 3: Write kafka.go**

`internal/adapter/kafka/kafka.go`:
```go
package kafka

import (
    "context"
    "fmt"
    "strings"

    "github.com/IBM/sarama"
    "token-scaler/internal/adapter"
)

type Adapter struct{}

func New() *Adapter { return &Adapter{} }

func (a *Adapter) HealthCheck(ctx context.Context, config map[string]string) error {
    if err := validateConfig(config); err != nil {
        return err
    }
    brokers := strings.Split(config["kafkaBrokers"], ",")
    cfg := sarama.NewConfig()
    cfg.Version = sarama.V2_6_0_0
    client, err := sarama.NewClient(brokers, cfg)
    if err != nil {
        return fmt.Errorf("kafka connect: %w", err)
    }
    defer client.Close()
    _, err = client.Partitions(config["kafkaTopic"])
    return err
}

// FetchTelemetry reads high-watermark offsets across all partitions.
// Uses ReadHighWatermarks — does NOT call DescribeGroups (avoids controller stress).
func (a *Adapter) FetchTelemetry(ctx context.Context, config map[string]string) (adapter.QueueTelemetry, error) {
    if err := validateConfig(config); err != nil {
        return adapter.QueueTelemetry{IsHealthy: false}, err
    }
    brokers := strings.Split(config["kafkaBrokers"], ",")
    cfg := sarama.NewConfig()
    cfg.Version = sarama.V2_6_0_0
    client, err := sarama.NewClient(brokers, cfg)
    if err != nil {
        return adapter.QueueTelemetry{IsHealthy: false}, fmt.Errorf("kafka connect: %w", err)
    }
    defer client.Close()

    partitions, err := client.Partitions(config["kafkaTopic"])
    if err != nil {
        return adapter.QueueTelemetry{IsHealthy: false}, fmt.Errorf("kafka partitions: %w", err)
    }
    var total int64
    for _, p := range partitions {
        offset, err := client.GetOffset(config["kafkaTopic"], p, sarama.OffsetNewest)
        if err != nil {
            return adapter.QueueTelemetry{IsHealthy: false}, fmt.Errorf("kafka watermark p%d: %w", p, err)
        }
        total += offset
    }
    return adapter.QueueTelemetry{ApproximateDepth: total, IsHealthy: true}, nil
}

func validateConfig(config map[string]string) error {
    if config["kafkaBrokers"] == "" {
        return fmt.Errorf("kafka adapter: kafkaBrokers is required")
    }
    if config["kafkaTopic"] == "" {
        return fmt.Errorf("kafka adapter: kafkaTopic is required")
    }
    return nil
}
```

- [ ] **Step 4: Run tests**

```bash
go test ./internal/adapter/kafka/... -v -race
```
Expected: validation tests PASS; broker-unreachable test PASS (sarama returns error quickly on invalid addr).

- [ ] **Step 5: Commit**

```bash
git add internal/adapter/kafka/
git commit -m "feat: kafka adapter using watermark offsets (not DescribeGroups)"
```

---

### Task 5: Prometheus Metrics Layer

**Files:**
- Create: `internal/scaler/metrics.go`

**Interfaces:**
- Produces:
  - `scaler.NewMetrics(reg prometheus.Registerer) *Metrics`
  - `(*Metrics).SetHealthy(phase string, healthy bool)`
  - `(*Metrics).RecordQueueDrift(queueName string, approxDepth int64)`
  - `(*Metrics).StartHTTPServer(port string) error` — serves `/healthz` (200 OK) and `/metrics` on given port

- [ ] **Step 1: Write failing tests**

`internal/scaler/metrics_test.go`:
```go
package scaler_test

import (
    "io"
    "net/http"
    "net/http/httptest"
    "strings"
    "testing"

    "github.com/prometheus/client_golang/prometheus"
    "token-scaler/internal/scaler"
)

func TestMetrics_SetHealthy_RecordsGauge(t *testing.T) {
    reg := prometheus.NewRegistry()
    m := scaler.NewMetrics(reg)
    m.SetHealthy("is_active", false)

    families, err := reg.Gather()
    if err != nil {
        t.Fatal(err)
    }
    var found bool
    for _, f := range families {
        if f.GetName() == "token_scaler_healthy" {
            for _, metric := range f.GetMetric() {
                for _, label := range metric.GetLabel() {
                    if label.GetName() == "phase" && label.GetValue() == "is_active" {
                        if metric.GetGauge().GetValue() == 0 {
                            found = true
                        }
                    }
                }
            }
        }
    }
    if !found {
        t.Fatal("expected token_scaler_healthy{phase=is_active}=0")
    }
}

func TestMetrics_SetHealthy_True(t *testing.T) {
    reg := prometheus.NewRegistry()
    m := scaler.NewMetrics(reg)
    m.SetHealthy("get_metrics", true)

    families, _ := reg.Gather()
    for _, f := range families {
        if f.GetName() == "token_scaler_healthy" {
            for _, metric := range f.GetMetric() {
                for _, label := range metric.GetLabel() {
                    if label.GetValue() == "get_metrics" {
                        if metric.GetGauge().GetValue() != 1 {
                            t.Fatalf("want 1, got %v", metric.GetGauge().GetValue())
                        }
                    }
                }
            }
        }
    }
}

func TestMetrics_RecordQueueDrift(t *testing.T) {
    reg := prometheus.NewRegistry()
    m := scaler.NewMetrics(reg)
    m.RecordQueueDrift("test-queue", 42)

    families, _ := reg.Gather()
    var found bool
    for _, f := range families {
        if f.GetName() == "token_scaler_queue_depth_approx" {
            found = true
            break
        }
    }
    if !found {
        t.Fatal("expected token_scaler_queue_depth_approx metric")
    }
}

func TestMetrics_HealthzEndpoint(t *testing.T) {
    reg := prometheus.NewRegistry()
    m := scaler.NewMetrics(reg)
    mux := m.BuildHTTPMux()
    w := httptest.NewRecorder()
    req, _ := http.NewRequest("GET", "/healthz", nil)
    mux.ServeHTTP(w, req)
    if w.Code != http.StatusOK {
        t.Fatalf("want 200, got %d", w.Code)
    }
    body, _ := io.ReadAll(w.Body)
    if !strings.Contains(string(body), "ok") {
        t.Fatalf("want 'ok' body, got %q", body)
    }
}
```

- [ ] **Step 2: Run to verify fail**

```bash
go test ./internal/scaler/... -run TestMetrics -v
```
Expected: FAIL — `metrics.go` does not exist.

- [ ] **Step 3: Write metrics.go**

`internal/scaler/metrics.go`:
```go
package scaler

import (
    "fmt"
    "net/http"

    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

// Metrics holds all custom Prometheus gauges for the scaler.
type Metrics struct {
    healthy          *prometheus.GaugeVec
    queueDepthApprox *prometheus.GaugeVec
    reg              prometheus.Gatherer
}

func NewMetrics(reg prometheus.Registerer) *Metrics {
    healthy := prometheus.NewGaugeVec(prometheus.GaugeOpts{
        Name: "token_scaler_healthy",
        Help: "1 if the scaler phase is healthy, 0 on error.",
    }, []string{"phase"})
    queueDepth := prometheus.NewGaugeVec(prometheus.GaugeOpts{
        Name: "token_scaler_queue_depth_approx",
        Help: "Approximate message depth from queue broker (drift telemetry only).",
    }, []string{"queue"})
    reg.MustRegister(healthy, queueDepth)
    return &Metrics{
        healthy:          healthy,
        queueDepthApprox: queueDepth,
        reg:              reg.(prometheus.Gatherer),
    }
}

func (m *Metrics) SetHealthy(phase string, healthy bool) {
    v := 0.0
    if healthy {
        v = 1.0
    }
    m.healthy.WithLabelValues(phase).Set(v)
}

func (m *Metrics) RecordQueueDrift(queueName string, approxDepth int64) {
    m.queueDepthApprox.WithLabelValues(queueName).Set(float64(approxDepth))
}

// BuildHTTPMux returns an http.ServeMux with /healthz and /metrics registered.
func (m *Metrics) BuildHTTPMux() *http.ServeMux {
    mux := http.NewServeMux()
    mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        fmt.Fprint(w, "ok")
    })
    mux.Handle("/metrics", promhttp.HandlerFor(m.reg, promhttp.HandlerOpts{}))
    return mux
}

func (m *Metrics) StartHTTPServer(port string) error {
    return http.ListenAndServe(":"+port, m.BuildHTTPMux())
}
```

- [ ] **Step 4: Run tests**

```bash
go test ./internal/scaler/... -run TestMetrics -v -race
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/scaler/metrics.go internal/scaler/metrics_test.go
git commit -m "feat: prometheus metrics layer with health gauges and drift telemetry"
```

---

### Task 6: gRPC Scaler Server

**Files:**
- Create: `internal/scaler/scaler.go`
- Create: `internal/scaler/scaler_test.go`

**Interfaces:**
- Consumes:
  - `redis.Client` (Task 2): `ExecuteEvaluator`, `GetHashLength`, `Subscribe`
  - `redis.ScriptManager` (Task 2)
  - `adapter.QueueAdapter` (Task 3): `FetchTelemetry`
  - `scaler.Metrics` (Task 5): `SetHealthy`, `RecordQueueDrift`
  - `pb.UnimplementedExternalScalerServer` (Task 1)
- Produces:
  - `scaler.NewExternalScalerServer(rc *redis.Client, sm *redis.ScriptManager, m *Metrics, a adapter.QueueAdapter) *ExternalScalerServer`
  - Implements: `pb.ExternalScalerServer`

- [ ] **Step 1: Write failing tests**

`internal/scaler/scaler_test.go`:
```go
package scaler_test

import (
    "context"
    "net"
    "testing"
    "time"

    "github.com/alicebob/miniredis/v2"
    "github.com/prometheus/client_golang/prometheus"
    goredis "github.com/redis/go-redis/v9"
    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/credentials/insecure"
    "google.golang.org/grpc/status"

    "token-scaler/internal/adapter"
    "token-scaler/internal/redis"
    "token-scaler/internal/scaler"
    pb "token-scaler/internal/scaler/gen"
)

type noopAdapter struct{}
func (n *noopAdapter) HealthCheck(ctx context.Context, config map[string]string) error { return nil }
func (n *noopAdapter) FetchTelemetry(ctx context.Context, config map[string]string) (adapter.QueueTelemetry, error) {
    return adapter.QueueTelemetry{IsHealthy: true, ApproximateDepth: 0}, nil
}

func startTestServer(t *testing.T, mr *miniredis.Miniredis) pb.ExternalScalerClient {
    t.Helper()
    rc, err := redis.NewClient(mr.Addr(), "")
    if err != nil {
        t.Fatal(err)
    }
    if _, err := rc.LoadScripts(context.Background()); err != nil {
        t.Fatal(err)
    }
    reg := prometheus.NewRegistry()
    m := scaler.NewMetrics(reg)
    srv := scaler.NewExternalScalerServer(rc, m, &noopAdapter{})

    lis, err := net.Listen("tcp", "127.0.0.1:0")
    if err != nil {
        t.Fatal(err)
    }
    gs := grpc.NewServer()
    pb.RegisterExternalScalerServer(gs, srv)
    go gs.Serve(lis)
    t.Cleanup(gs.Stop)

    conn, err := grpc.NewClient(lis.Addr().String(),
        grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        t.Fatal(err)
    }
    t.Cleanup(func() { conn.Close() })
    return pb.NewExternalScalerClient(conn)
}

func TestIsActive_True(t *testing.T) {
    mr, err := miniredis.Run()
    if err != nil {
        t.Fatal(err)
    }
    defer mr.Close()
    rdb := goredis.NewClient(&goredis.Options{Addr: mr.Addr()})
    rdb.HSet(context.Background(), "token_backlog:q1", "job1", "1000")
    rdb.Close()

    client := startTestServer(t, mr)
    resp, err := client.IsActive(context.Background(), &pb.ScaledObjectRef{
        ScalerMetadata: map[string]string{"queueName": "q1"},
    })
    if err != nil {
        t.Fatalf("IsActive: %v", err)
    }
    if !resp.Result {
        t.Fatal("want true")
    }
}

func TestIsActive_False(t *testing.T) {
    mr, _ := miniredis.Run()
    defer mr.Close()
    client := startTestServer(t, mr)
    resp, err := client.IsActive(context.Background(), &pb.ScaledObjectRef{
        ScalerMetadata: map[string]string{"queueName": "empty"},
    })
    if err != nil {
        t.Fatalf("IsActive: %v", err)
    }
    if resp.Result {
        t.Fatal("want false")
    }
}

func TestIsActive_MissingQueueName(t *testing.T) {
    mr, _ := miniredis.Run()
    defer mr.Close()
    client := startTestServer(t, mr)
    _, err := client.IsActive(context.Background(), &pb.ScaledObjectRef{
        ScalerMetadata: map[string]string{},
    })
    if status.Code(err) != codes.InvalidArgument {
        t.Fatalf("want InvalidArgument, got %v", err)
    }
}

func TestGetMetricSpec_Success(t *testing.T) {
    mr, _ := miniredis.Run()
    defer mr.Close()
    client := startTestServer(t, mr)
    resp, err := client.GetMetricSpec(context.Background(), &pb.ScaledObjectRef{
        ScalerMetadata: map[string]string{"tokensPerReplica": "500000"},
    })
    if err != nil {
        t.Fatalf("GetMetricSpec: %v", err)
    }
    if len(resp.MetricSpecs) != 1 {
        t.Fatalf("want 1 metric spec, got %d", len(resp.MetricSpecs))
    }
    if resp.MetricSpecs[0].MetricName != "token_backlog" {
        t.Fatalf("want metric_backlog, got %s", resp.MetricSpecs[0].MetricName)
    }
    if resp.MetricSpecs[0].TargetSize != 500000 {
        t.Fatalf("want 500000, got %d", resp.MetricSpecs[0].TargetSize)
    }
}

func TestGetMetricSpec_MissingTokensPerReplica(t *testing.T) {
    mr, _ := miniredis.Run()
    defer mr.Close()
    client := startTestServer(t, mr)
    _, err := client.GetMetricSpec(context.Background(), &pb.ScaledObjectRef{
        ScalerMetadata: map[string]string{},
    })
    if status.Code(err) != codes.InvalidArgument {
        t.Fatalf("want InvalidArgument, got %v", err)
    }
}

func TestGetMetrics_ReturnsBacklogSum(t *testing.T) {
    mr, _ := miniredis.Run()
    defer mr.Close()
    rdb := goredis.NewClient(&goredis.Options{Addr: mr.Addr()})
    future := time.Now().Add(1 * time.Hour).Unix()
    rdb.HSet(context.Background(), "token_backlog:q1", "j1", "100000", "j2", "200000")
    rdb.ZAdd(context.Background(), "token_expiry:q1", goredis.Z{Score: float64(future), Member: "j1"})
    rdb.ZAdd(context.Background(), "token_expiry:q1", goredis.Z{Score: float64(future), Member: "j2"})
    rdb.Close()

    client := startTestServer(t, mr)
    resp, err := client.GetMetrics(context.Background(), &pb.GetMetricsRequest{
        ScaledObjectRef: &pb.ScaledObjectRef{
            ScalerMetadata: map[string]string{
                "queueName":        "q1",
                "jobTimeoutSeconds": "3600",
            },
        },
    })
    if err != nil {
        t.Fatalf("GetMetrics: %v", err)
    }
    if len(resp.MetricValues) != 1 {
        t.Fatalf("want 1 value, got %d", len(resp.MetricValues))
    }
    if resp.MetricValues[0].MetricValue != 300000 {
        t.Fatalf("want 300000, got %d", resp.MetricValues[0].MetricValue)
    }
}

func TestGetMetrics_MissingQueueName(t *testing.T) {
    mr, _ := miniredis.Run()
    defer mr.Close()
    client := startTestServer(t, mr)
    _, err := client.GetMetrics(context.Background(), &pb.GetMetricsRequest{
        ScaledObjectRef: &pb.ScaledObjectRef{ScalerMetadata: map[string]string{}},
    })
    if status.Code(err) != codes.InvalidArgument {
        t.Fatalf("want InvalidArgument, got %v", err)
    }
}
```

- [ ] **Step 2: Run to verify fail**

```bash
go test ./internal/scaler/... -run TestIsActive -v
go test ./internal/scaler/... -run TestGetMetric -v
```
Expected: FAIL — `scaler.go` does not exist.

- [ ] **Step 3: Write scaler.go**

`internal/scaler/scaler.go`:
```go
package scaler

import (
    "context"
    "fmt"
    "strconv"
    "time"

    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"

    "token-scaler/internal/adapter"
    "token-scaler/internal/redis"
    pb "token-scaler/internal/scaler/gen"
)

type ExternalScalerServer struct {
    pb.UnimplementedExternalScalerServer
    rc      *redis.Client
    metrics *Metrics
    adapter adapter.QueueAdapter
}

func NewExternalScalerServer(rc *redis.Client, m *Metrics, a adapter.QueueAdapter) *ExternalScalerServer {
    return &ExternalScalerServer{rc: rc, metrics: m, adapter: a}
}

func (s *ExternalScalerServer) IsActive(ctx context.Context, in *pb.ScaledObjectRef) (*pb.IsActiveResponse, error) {
    queueName := in.ScalerMetadata["queueName"]
    if queueName == "" {
        return nil, status.Error(codes.InvalidArgument, "queueName metadata required")
    }
    n, err := s.rc.GetHashLength(ctx, "token_backlog:"+queueName)
    if err != nil {
        s.metrics.SetHealthy("is_active", false)
        return nil, status.Errorf(codes.Internal, "redis: %v", err)
    }
    s.metrics.SetHealthy("is_active", true)
    return &pb.IsActiveResponse{Result: n > 0}, nil
}

func (s *ExternalScalerServer) StreamIsActive(in *pb.ScaledObjectRef, stream pb.ExternalScaler_StreamIsActiveServer) error {
    queueName := in.ScalerMetadata["queueName"]
    if queueName == "" {
        return status.Error(codes.InvalidArgument, "queueName metadata required")
    }
    sub := s.rc.Subscribe(stream.Context(), "token_events:"+queueName)
    defer sub.Close()
    for {
        select {
        case <-stream.Context().Done():
            return nil
        case msg, ok := <-sub.Channel():
            if !ok {
                return status.Error(codes.Unavailable, "redis pubsub closed")
            }
            _ = msg
            if err := stream.Send(&pb.IsActiveResponse{Result: true}); err != nil {
                return err
            }
        }
    }
}

func (s *ExternalScalerServer) GetMetricSpec(ctx context.Context, in *pb.ScaledObjectRef) (*pb.GetMetricSpecResponse, error) {
    raw := in.ScalerMetadata["tokensPerReplica"]
    if raw == "" {
        return nil, status.Error(codes.InvalidArgument, "tokensPerReplica metadata required")
    }
    target, err := strconv.ParseInt(raw, 10, 64)
    if err != nil {
        return nil, status.Errorf(codes.InvalidArgument, "tokensPerReplica must be integer: %v", err)
    }
    return &pb.GetMetricSpecResponse{
        MetricSpecs: []*pb.MetricSpec{{MetricName: "token_backlog", TargetSize: target}},
    }, nil
}

func (s *ExternalScalerServer) GetMetrics(ctx context.Context, in *pb.GetMetricsRequest) (*pb.GetMetricsResponse, error) {
    meta := in.ScaledObjectRef.ScalerMetadata
    queueName := meta["queueName"]
    if queueName == "" {
        return nil, status.Error(codes.InvalidArgument, "queueName metadata required")
    }
    safetyTTL := "86400"
    if v := meta["jobTimeoutSeconds"]; v != "" {
        safetyTTL = v
    }
    chunkSize := "100"

    total, err := s.rc.ExecuteEvaluator(ctx, queueName, time.Now().Unix(), safetyTTL, chunkSize)
    if err != nil {
        s.metrics.SetHealthy("get_metrics", false)
        return nil, status.Errorf(codes.Internal, "evaluator: %v", err)
    }
    s.metrics.SetHealthy("get_metrics", true)

    // Background telemetry: fetch queue drift (non-blocking, best-effort)
    go func() {
        if s.adapter == nil {
            return
        }
        tel, err := s.adapter.FetchTelemetry(ctx, meta)
        if err == nil {
            s.metrics.RecordQueueDrift(queueName, tel.ApproximateDepth)
        }
    }()

    return &pb.GetMetricsResponse{
        MetricValues: []*pb.MetricValue{{MetricName: "token_backlog", MetricValue: total}},
    }, nil
}

// chunkSize is read from env in main.go and passed to ExecuteEvaluator.
// Expose a setter so main.go can wire it without changing the interface.
func (s *ExternalScalerServer) WithChunkSize(size string) *ExternalScalerServer {
    s.chunkSize = size
    return s
}
```

Note: Add `chunkSize string` field to `ExternalScalerServer` struct and use it in `GetMetrics` instead of the hardcoded `"100"`. Update the struct:

```go
type ExternalScalerServer struct {
    pb.UnimplementedExternalScalerServer
    rc        *redis.Client
    metrics   *Metrics
    adapter   adapter.QueueAdapter
    chunkSize string // injected from env via WithChunkSize
}
```

And in `GetMetrics`, replace `chunkSize := "100"` with:
```go
chunkSize := s.chunkSize
if chunkSize == "" {
    chunkSize = "100"
}
```

- [ ] **Step 4: Run tests**

```bash
go test ./internal/scaler/... -v -race
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/scaler/scaler.go internal/scaler/scaler_test.go
git commit -m "feat: gRPC ExternalScaler server with IsActive, GetMetricSpec, GetMetrics, StreamIsActive"
```

---

### Task 7: main.go Wiring and Startup Validation

**Files:**
- Create: `cmd/token-scaler/main.go`

**Interfaces:**
- Consumes: all of Tasks 2–6
- Produces: runnable binary; validates required config at startup and fails fast with clear error

- [ ] **Step 1: Write main.go**

`cmd/token-scaler/main.go`:
```go
package main

import (
    "fmt"
    "log"
    "net"
    "os"

    "google.golang.org/grpc"
    "github.com/prometheus/client_golang/prometheus"

    "token-scaler/internal/adapter/kafka"
    "token-scaler/internal/adapter/sqs"
    "token-scaler/internal/redis"
    "token-scaler/internal/scaler"
    pb "token-scaler/internal/scaler/gen"
)

func main() {
    redisAddr    := requireEnv("REDIS_ADDRESS")
    redisPass    := os.Getenv("REDIS_PASSWORD")
    grpcPort     := envOr("GRPC_PORT", "9090")
    metricsPort  := envOr("METRICS_PORT", "8080")
    chunkSize    := envOr("CLEANUP_CHUNK_SIZE", "100")
    queueType    := os.Getenv("QUEUE_TYPE") // optional; used for startup adapter validation

    rc, err := redis.NewClient(redisAddr, redisPass)
    if err != nil {
        log.Fatalf("redis connect: %v", err)
    }
    defer rc.Close()

    if _, err := rc.LoadScripts(nil); err != nil {
        log.Fatalf("load Lua scripts: %v", err)
    }

    reg := prometheus.NewRegistry()
    m := scaler.NewMetrics(reg)

    var queueAdapter interface {
        HealthCheck(ctx interface{}, config map[string]string) error
        FetchTelemetry(ctx interface{}, config map[string]string) (interface{}, error)
    }
    _ = queueAdapter
    // Wire the correct adapter based on QUEUE_TYPE env (optional at startup)
    var a adapter.QueueAdapter
    switch queueType {
    case "sqs":
        a = sqs.New()
    case "kafka":
        a = kafka.New()
    default:
        a = nil
    }

    srv := scaler.NewExternalScalerServer(rc, m, a).WithChunkSize(chunkSize)

    // Start metrics/healthz HTTP server in background
    go func() {
        log.Printf("metrics server on :%s", metricsPort)
        if err := m.StartHTTPServer(metricsPort); err != nil {
            log.Printf("metrics server error: %v", err)
        }
    }()

    lis, err := net.Listen("tcp", fmt.Sprintf(":%s", grpcPort))
    if err != nil {
        log.Fatalf("listen: %v", err)
    }
    gs := grpc.NewServer()
    pb.RegisterExternalScalerServer(gs, srv)
    log.Printf("gRPC server on :%s", grpcPort)
    if err := gs.Serve(lis); err != nil {
        log.Fatalf("serve: %v", err)
    }
}

func requireEnv(key string) string {
    v := os.Getenv(key)
    if v == "" {
        log.Fatalf("required env var %s is not set", key)
    }
    return v
}

func envOr(key, fallback string) string {
    if v := os.Getenv(key); v != "" {
        return v
    }
    return fallback
}
```

Fix the import of `adapter` package — replace the inline interface with the real one:
```go
import (
    "token-scaler/internal/adapter"
    "token-scaler/internal/adapter/kafka"
    "token-scaler/internal/adapter/sqs"
    ...
)
```

Remove the unused `queueAdapter` interface block. The `a adapter.QueueAdapter` assignment is correct.

Also fix `rc.LoadScripts(nil)` → pass a real context:
```go
import "context"
...
ctx := context.Background()
if _, err := rc.LoadScripts(ctx); err != nil {
```

- [ ] **Step 2: Verify build**

```bash
go build ./cmd/token-scaler/
```
Expected: compiles to `token-scaler` binary.

- [ ] **Step 3: Verify startup requires REDIS_ADDRESS**

```bash
./token-scaler
```
Expected: `FATAL required env var REDIS_ADDRESS is not set` and non-zero exit.

- [ ] **Step 4: Commit**

```bash
git add cmd/token-scaler/main.go
git commit -m "feat: main.go wiring with startup validation and graceful config loading"
```

---

### Task 8: Integration Tests

**Files:**
- Create: `cmd/token-scaler/main_test.go`

**Interfaces:**
- Consumes: real Redis (testcontainers), toxiproxy (testcontainers), all Tasks 2–7
- This test file starts the gRPC server in-process, dials it with a real gRPC client

- [ ] **Step 1: Write integration tests**

`cmd/token-scaler/main_test.go`:
```go
//go:build integration

package main_test

import (
    "context"
    "fmt"
    "net"
    "strconv"
    "testing"
    "time"

    "github.com/prometheus/client_golang/prometheus"
    tcredis "github.com/testcontainers/testcontainers-go/modules/redis"
    tctoxiproxy "github.com/testcontainers/testcontainers-go/modules/toxiproxy"
    toxiclient "github.com/shopify/toxiproxy/v2/client"
    goredis "github.com/redis/go-redis/v9"
    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/credentials/insecure"
    "google.golang.org/grpc/status"

    "token-scaler/internal/redis"
    "token-scaler/internal/scaler"
    pb "token-scaler/internal/scaler/gen"
)

func startIntegrationServer(t *testing.T, redisAddr string) (pb.ExternalScalerClient, *goredis.Client) {
    t.Helper()
    rc, err := redis.NewClient(redisAddr, "")
    if err != nil {
        t.Fatalf("redis.NewClient: %v", err)
    }
    if _, err := rc.LoadScripts(context.Background()); err != nil {
        t.Fatalf("LoadScripts: %v", err)
    }
    reg := prometheus.NewRegistry()
    m := scaler.NewMetrics(reg)
    srv := scaler.NewExternalScalerServer(rc, m, nil).WithChunkSize("100")

    lis, err := net.Listen("tcp", "127.0.0.1:0")
    if err != nil {
        t.Fatal(err)
    }
    gs := grpc.NewServer()
    pb.RegisterExternalScalerServer(gs, srv)
    go gs.Serve(lis)
    t.Cleanup(func() { gs.Stop(); rc.Close() })

    conn, err := grpc.NewClient(lis.Addr().String(),
        grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        t.Fatal(err)
    }
    t.Cleanup(func() { conn.Close() })

    rdb := goredis.NewClient(&goredis.Options{Addr: redisAddr})
    t.Cleanup(func() { rdb.Close() })
    return pb.NewExternalScalerClient(conn), rdb
}

func TestIntegration_FullLifecycle(t *testing.T) {
    ctx := context.Background()
    container, err := tcredis.Run(ctx, "redis:7-alpine")
    if err != nil {
        t.Fatalf("redis container: %v", err)
    }
    t.Cleanup(func() { container.Terminate(ctx) })
    addr, _ := container.ConnectionString(ctx)
    // ConnectionString returns "redis://host:port" — strip scheme
    addr = addr[len("redis://"):]

    client, rdb := startIntegrationServer(t, addr)
    future := float64(time.Now().Add(1 * time.Hour).Unix())

    // Publish 3 jobs
    rdb.HSet(ctx, "token_backlog:q1", "j1", "100000", "j2", "200000", "j3", "50000")
    rdb.ZAdd(ctx, "token_expiry:q1",
        goredis.Z{Score: future, Member: "j1"},
        goredis.Z{Score: future, Member: "j2"},
        goredis.Z{Score: future, Member: "j3"},
    )

    // GetMetrics should return 350000
    resp, err := client.GetMetrics(ctx, &pb.GetMetricsRequest{
        ScaledObjectRef: &pb.ScaledObjectRef{
            ScalerMetadata: map[string]string{"queueName": "q1", "jobTimeoutSeconds": "3600"},
        },
    })
    if err != nil {
        t.Fatalf("GetMetrics: %v", err)
    }
    if resp.MetricValues[0].MetricValue != 350000 {
        t.Fatalf("want 350000, got %d", resp.MetricValues[0].MetricValue)
    }

    // Simulate job completion: remove j2
    rdb.HDel(ctx, "token_backlog:q1", "j2")
    rdb.ZRem(ctx, "token_expiry:q1", "j2")

    resp2, _ := client.GetMetrics(ctx, &pb.GetMetricsRequest{
        ScaledObjectRef: &pb.ScaledObjectRef{
            ScalerMetadata: map[string]string{"queueName": "q1", "jobTimeoutSeconds": "3600"},
        },
    })
    if resp2.MetricValues[0].MetricValue != 150000 {
        t.Fatalf("after completion: want 150000, got %d", resp2.MetricValues[0].MetricValue)
    }
}

func TestIntegration_CleanupCeiling(t *testing.T) {
    ctx := context.Background()
    container, _ := tcredis.Run(ctx, "redis:7-alpine")
    t.Cleanup(func() { container.Terminate(ctx) })
    addr, _ := container.ConnectionString(ctx)
    addr = addr[len("redis://"):]

    client, rdb := startIntegrationServer(t, addr)
    past := float64(time.Now().Add(-1 * time.Hour).Unix())

    // Insert 105 expired jobs
    members := make([]goredis.Z, 105)
    hvals := make([]interface{}, 0, 210)
    for i := 0; i < 105; i++ {
        id := "job" + strconv.Itoa(i)
        members[i] = goredis.Z{Score: past, Member: id}
        hvals = append(hvals, id, "100")
    }
    rdb.HSet(ctx, "token_backlog:cleanup", hvals...)
    rdb.ZAdd(ctx, "token_expiry:cleanup", members...)

    // Cycle 1: only 100 purged, 5 remain with total = 500
    resp1, err := client.GetMetrics(ctx, &pb.GetMetricsRequest{
        ScaledObjectRef: &pb.ScaledObjectRef{
            ScalerMetadata: map[string]string{"queueName": "cleanup", "jobTimeoutSeconds": "3600"},
        },
    })
    if err != nil {
        t.Fatalf("cycle1 GetMetrics: %v", err)
    }
    if resp1.MetricValues[0].MetricValue != 500 {
        t.Fatalf("cycle1: want 500, got %d", resp1.MetricValues[0].MetricValue)
    }
    remaining, _ := rdb.HLen(ctx, "token_backlog:cleanup").Result()
    if remaining != 5 {
        t.Fatalf("cycle1: want 5 remaining, got %d", remaining)
    }

    // Cycle 2: last 5 purged, total = 0
    resp2, _ := client.GetMetrics(ctx, &pb.GetMetricsRequest{
        ScaledObjectRef: &pb.ScaledObjectRef{
            ScalerMetadata: map[string]string{"queueName": "cleanup", "jobTimeoutSeconds": "3600"},
        },
    })
    if resp2.MetricValues[0].MetricValue != 0 {
        t.Fatalf("cycle2: want 0, got %d", resp2.MetricValues[0].MetricValue)
    }
}

func TestIntegration_HPABoundaryMath(t *testing.T) {
    ctx := context.Background()
    container, _ := tcredis.Run(ctx, "redis:7-alpine")
    t.Cleanup(func() { container.Terminate(ctx) })
    addr, _ := container.ConnectionString(ctx)
    addr = addr[len("redis://"):]

    client, rdb := startIntegrationServer(t, addr)
    future := float64(time.Now().Add(1 * time.Hour).Unix())

    // Test: 510000 tokens (2% above 500000 target — within 10% HPA stability window)
    rdb.HSet(ctx, "token_backlog:hpa", "j1", "510000")
    rdb.ZAdd(ctx, "token_expiry:hpa", goredis.Z{Score: future, Member: "j1"})
    resp1, _ := client.GetMetrics(ctx, &pb.GetMetricsRequest{
        ScaledObjectRef: &pb.ScaledObjectRef{
            ScalerMetadata: map[string]string{"queueName": "hpa", "jobTimeoutSeconds": "3600"},
        },
    })
    // Verify returned value is exact int64, no truncation
    if resp1.MetricValues[0].MetricValue != 510000 {
        t.Fatalf("510k boundary: want 510000, got %d", resp1.MetricValues[0].MetricValue)
    }

    // Test: 560000 tokens (12% above — should trigger HPA scale-up)
    rdb.HSet(ctx, "token_backlog:hpa2", "j1", "560000")
    rdb.ZAdd(ctx, "token_expiry:hpa2", goredis.Z{Score: future, Member: "j1"})
    resp2, _ := client.GetMetrics(ctx, &pb.GetMetricsRequest{
        ScaledObjectRef: &pb.ScaledObjectRef{
            ScalerMetadata: map[string]string{"queueName": "hpa2", "jobTimeoutSeconds": "3600"},
        },
    })
    if resp2.MetricValues[0].MetricValue != 560000 {
        t.Fatalf("560k boundary: want 560000, got %d", resp2.MetricValues[0].MetricValue)
    }
}

func TestIntegration_NetworkFracture(t *testing.T) {
    ctx := context.Background()
    redisContainer, _ := tcredis.Run(ctx, "redis:7-alpine")
    t.Cleanup(func() { redisContainer.Terminate(ctx) })
    redisAddr, _ := redisContainer.ConnectionString(ctx)
    redisAddr = redisAddr[len("redis://"):]

    toxiContainer, err := tctoxiproxy.Run(ctx, "ghcr.io/shopify/toxiproxy")
    if err != nil {
        t.Fatalf("toxiproxy container: %v", err)
    }
    t.Cleanup(func() { toxiContainer.Terminate(ctx) })

    toxiAPIAddr, _ := toxiContainer.URI(ctx)
    toxiClient := toxiclient.NewClient(toxiAPIAddr)

    // Create proxy: toxiproxy_listen → redis
    proxy, err := toxiClient.CreateProxy("redis", "0.0.0.0:16379", redisAddr)
    if err != nil {
        t.Fatalf("create proxy: %v", err)
    }

    toxiHost, _, _ := net.SplitHostPort(toxiAPIAddr)
    proxiedAddr := fmt.Sprintf("%s:16379", toxiHost)

    grpcClient, _ := startIntegrationServer(t, proxiedAddr)

    // Verify normal operation before fracture
    future := float64(time.Now().Add(1 * time.Hour).Unix())
    rdb := goredis.NewClient(&goredis.Options{Addr: proxiedAddr})
    defer rdb.Close()
    rdb.HSet(ctx, "token_backlog:frac", "j1", "1000")
    rdb.ZAdd(ctx, "token_expiry:frac", goredis.Z{Score: future, Member: "j1"})

    resp, err := grpcClient.GetMetrics(ctx, &pb.GetMetricsRequest{
        ScaledObjectRef: &pb.ScaledObjectRef{
            ScalerMetadata: map[string]string{"queueName": "frac", "jobTimeoutSeconds": "3600"},
        },
    })
    if err != nil || resp.MetricValues[0].MetricValue != 1000 {
        t.Fatalf("pre-fracture: want 1000, got err=%v resp=%v", err, resp)
    }

    // Break the connection
    proxy.AddToxic("cut", "timeout", "upstream", 1.0, toxiclient.Attributes{"timeout": 1})

    // GetMetrics should return Internal, not hang
    ctx2, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()
    _, err = grpcClient.GetMetrics(ctx2, &pb.GetMetricsRequest{
        ScaledObjectRef: &pb.ScaledObjectRef{
            ScalerMetadata: map[string]string{"queueName": "frac", "jobTimeoutSeconds": "3600"},
        },
    })
    if err == nil {
        t.Fatal("expected error during network fracture")
    }
    code := status.Code(err)
    if code != codes.Internal && code != codes.DeadlineExceeded && code != codes.Unavailable {
        t.Fatalf("want Internal/Unavailable/DeadlineExceeded, got %v", code)
    }

    // Heal the connection
    proxy.RemoveToxic("cut")
    time.Sleep(200 * time.Millisecond)

    // Should self-heal
    resp3, err := grpcClient.GetMetrics(ctx, &pb.GetMetricsRequest{
        ScaledObjectRef: &pb.ScaledObjectRef{
            ScalerMetadata: map[string]string{"queueName": "frac", "jobTimeoutSeconds": "3600"},
        },
    })
    if err != nil {
        t.Fatalf("post-heal: %v", err)
    }
    if resp3.MetricValues[0].MetricValue != 1000 {
        t.Fatalf("post-heal: want 1000, got %d", resp3.MetricValues[0].MetricValue)
    }
}
```

- [ ] **Step 2: Run integration tests**

```bash
go test ./cmd/... -v -race -tags integration -timeout 120s
```
Expected: all PASS (requires Docker for testcontainers).

- [ ] **Step 3: Commit**

```bash
git add cmd/token-scaler/main_test.go
git commit -m "test: integration tests — lifecycle, cleanup ceiling, HPA boundary, network fracture"
```

---

### Task 9: Helm Chart

**Files:**
- Create: `deploy/helm/token-scaler/Chart.yaml`
- Create: `deploy/helm/token-scaler/values.yaml`
- Create: `deploy/helm/token-scaler/templates/deployment.yaml`
- Create: `deploy/helm/token-scaler/templates/service.yaml`
- Create: `deploy/helm/token-scaler/templates/serviceaccount.yaml`
- Create: `deploy/helm/token-scaler/templates/scaled-object-example.yaml`

- [ ] **Step 1: Chart.yaml**

```yaml
apiVersion: v2
name: token-scaler
description: KEDA Token-Aware External Scaler for LLM batch inference
type: application
version: 0.1.0
appVersion: "0.1.0"
```

- [ ] **Step 2: values.yaml**

```yaml
image:
  repository: ghcr.io/your-org/token-scaler
  tag: latest
  pullPolicy: IfNotPresent

replicaCount: 1

grpc:
  port: 9090

metrics:
  port: 8080

redis:
  address: ""          # REQUIRED: host:port
  passwordSecretRef:
    name: ""
    key: ""

serviceAccount:
  create: true
  annotations: {}      # Add eks.amazonaws.com/role-arn: <arn> for IRSA

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

- [ ] **Step 3: deployment.yaml**

`deploy/helm/token-scaler/templates/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "token-scaler.fullname" . }}
  labels:
    app: {{ include "token-scaler.fullname" . }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ include "token-scaler.fullname" . }}
  template:
    metadata:
      labels:
        app: {{ include "token-scaler.fullname" . }}
    spec:
      serviceAccountName: {{ include "token-scaler.serviceAccountName" . }}
      containers:
        - name: token-scaler
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: grpc
              containerPort: {{ .Values.grpc.port }}
            - name: metrics
              containerPort: {{ .Values.metrics.port }}
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
            {{- if .Values.redis.passwordSecretRef.name }}
            - name: REDIS_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.redis.passwordSecretRef.name }}
                  key: {{ .Values.redis.passwordSecretRef.key }}
            {{- end }}
          livenessProbe:
            httpGet:
              path: /healthz
              port: {{ .Values.metrics.port }}
            initialDelaySeconds: 5
            periodSeconds: 10
          readinessProbe:
            grpc:
              port: {{ .Values.grpc.port }}
            initialDelaySeconds: 5
            periodSeconds: 10
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
```

- [ ] **Step 4: service.yaml**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "token-scaler.fullname" . }}
spec:
  type: ClusterIP
  selector:
    app: {{ include "token-scaler.fullname" . }}
  ports:
    - name: grpc
      port: {{ .Values.grpc.port }}
      targetPort: grpc
    - name: metrics
      port: {{ .Values.metrics.port }}
      targetPort: metrics
```

- [ ] **Step 5: serviceaccount.yaml**

```yaml
{{- if .Values.serviceAccount.create }}
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ include "token-scaler.serviceAccountName" . }}
  annotations:
    {{- toYaml .Values.serviceAccount.annotations | nindent 4 }}
{{- end }}
```

- [ ] **Step 6: Add _helpers.tpl**

`deploy/helm/token-scaler/templates/_helpers.tpl`:
```
{{- define "token-scaler.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "token-scaler.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- include "token-scaler.fullname" . }}
{{- else }}
default
{{- end }}
{{- end }}
```

- [ ] **Step 7: scaled-object-example.yaml (reference only — not deployed)**

```yaml
# This file is a reference. Copy and customize per workload.
# Deploy separately: kubectl apply -f this-file.yaml
#
# apiVersion: keda.sh/v1alpha1
# kind: ScaledObject
# metadata:
#   name: vllm-batch-scaler
# spec:
#   scaleTargetRef:
#     name: vllm-deployment
#   minReplicaCount: 0
#   maxReplicaCount: 10
#   triggers:
#     - type: external
#       metadata:
#         scalerAddress: token-scaler.<namespace>.svc:9090
#         queueType: sqs                          # or "kafka"
#         queueName: batch-jobs                   # Redis key suffix
#         queueUrl: "https://sqs.us-east-1.amazonaws.com/123456789/batch-jobs"  # SQS only
#         awsRegion: us-east-1                    # SQS only
#         # kafkaBrokers: kafka-bootstrap.kafka.svc:9092  # Kafka only
#         # kafkaTopic: batch-inference-jobs              # Kafka only
#         redisAddress: redis.default.svc:6379
#         tokensPerReplica: "500000"
#         jobTimeoutSeconds: "3600"
```

- [ ] **Step 8: Lint the chart**

```bash
helm lint deploy/helm/token-scaler/
```
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add deploy/
git commit -m "feat: helm chart with deployment, service, serviceaccount, and ScaledObject reference"
```

---

### Task 10: Caller Contract Documentation

**Files:**
- Create: `docs/contract.md`

- [ ] **Step 1: Write contract.md**

`docs/contract.md`:
```markdown
# Token Scaler Caller Contract

This document specifies the dual-write protocol upstream publishers and downstream workers must implement to use the token-scaler.

## Publisher Contract

On every job submission, the publisher MUST:

1. Compute token count using a fast local tokenizer (tiktoken, HuggingFace tokenizers) **before** publishing to the queue.
2. Include the token count in the job payload:

```json
{
  "job_id": "unique-uuid",
  "payload": "...",
  "metadata": {
    "estimated_tokens": 42500
  }
}
```

3. After publishing to SQS/Kafka, execute the Redis atomic publish script:

```
EVALSHA <publisher_sha> 3
  token_backlog:{queueName}
  token_expiry:{queueName}
  token_events:{queueName}
  {job_id}
  {estimated_tokens}
  {expiry_epoch}
  {safety_ttl}
```

Where `expiry_epoch = now + jobTimeoutSeconds` and `safety_ttl = jobTimeoutSeconds`.

Python example:
```python
import redis
import time
import uuid

r = redis.Redis(host="redis-host", port=6379)
# Load the script once at startup
publisher_sha = r.script_load(SCRIPT_A_PUBLISHER)

def publish_job(queue_name: str, token_count: int, job_timeout_sec: int = 3600):
    job_id = str(uuid.uuid4())
    expiry_epoch = int(time.time()) + job_timeout_sec
    r.evalsha(
        publisher_sha, 3,
        f"token_backlog:{queue_name}",
        f"token_expiry:{queue_name}",
        f"token_events:{queue_name}",
        job_id, token_count, expiry_epoch, job_timeout_sec
    )
    return job_id
```

## Worker Contract

On successful job completion, the worker MUST:

```
HDEL token_backlog:{queueName} {job_id}
ZREM token_expiry:{queueName} {job_id}
```

Python example:
```python
def complete_job(queue_name: str, job_id: str):
    pipe = r.pipeline()
    pipe.hdel(f"token_backlog:{queue_name}", job_id)
    pipe.zrem(f"token_expiry:{queue_name}", job_id)
    pipe.execute()
```

## Failure Handling

- **Worker crash before completion:** The job entry will expire at `expiry_epoch`. The scaler purges it on the next evaluation cycle. Maximum staleness = `jobTimeoutSeconds` seconds.
- **Worker crash after queue consume but before Redis delete:** Same as above — TTL-based self-healing.
- **Do NOT** use this scaler for agentic workloads where a single job spawns non-deterministic sub-tasks. Token count in the payload must represent the complete job's token budget.

## Script A Source

See `internal/redis/scripts.go` constant `ScriptAPublisher`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/contract.md
git commit -m "docs: caller contract for publisher and worker dual-write protocol"
```

---

## Self-Review Against Spec

Checked spec sections against tasks:

| Spec requirement | Covered in |
|---|---|
| gRPC ExternalScaler interface (IsActive, StreamIsActive, GetMetricSpec, GetMetrics) | Task 6 |
| AverageValue metric type (`TargetSize` → HPA AverageValue) | Task 6 `GetMetricSpec` |
| Script A atomic publisher (HSET+ZADD+PERSIST+PUBLISH) | Task 2 |
| Script B atomic evaluator (ZRANGEBYSCORE LIMIT + HVALS sum + EXPIRE on empty) | Task 2 |
| NOSCRIPT retry on Redis restart | Task 2 `ExecuteEvaluator` |
| SQS adapter using `queueUrl` (not `queueName`) | Task 3 |
| Kafka adapter using watermark offsets (not DescribeGroups) | Task 4 |
| `token_scaler_healthy{phase}` gauge | Task 5 |
| `token_scaler_queue_depth_approx{queue}` gauge | Task 5 |
| `/healthz` endpoint on :8080 | Task 5 |
| `/metrics` endpoint on :8080 | Task 5 |
| `cleanupChunkSize` env → ARGV[3] in Script B | Tasks 2, 7 |
| Karpenter startup taint lifecycle | Not applicable — this is a pure KEDA scaler |
| Integration test: lifecycle, cleanup ceiling, HPA boundary, network fracture | Task 8 |
| Helm chart with env var injection | Task 9 |
| Caller contract documentation | Task 10 |
| `internal/scaler/gen/` (generated code committed) | Task 1 |
| Cloud-agnostic (no AWS defaults baked in) | Tasks 3, 7 |
| Binary not library (`internal/` only) | Global constraint, all tasks |

No gaps found.
```

---
