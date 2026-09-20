//go:build integration

package main_test

import (
	"context"
	"fmt"
	"net"
	"strings"
	"testing"
	"time"

	toxiclient "github.com/Shopify/toxiproxy/v2/client"
	"github.com/prometheus/client_golang/prometheus"
	goredis "github.com/redis/go-redis/v9"
	tcredis "github.com/testcontainers/testcontainers-go/modules/redis"
	tctoxiproxy "github.com/testcontainers/testcontainers-go/modules/toxiproxy"
	"github.com/testcontainers/testcontainers-go/network"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	"token-scaler/internal/redis"
	"token-scaler/internal/scaler"
	pb "token-scaler/internal/scaler/gen"
)

// startIntegrationServer starts a gRPC scaler server backed by a real Redis instance
// and returns an ExternalScalerClient and a direct Redis client for seeding test data.
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

	lis, _ := net.Listen("tcp", "127.0.0.1:0")
	gs := grpc.NewServer()
	pb.RegisterExternalScalerServer(gs, srv)
	go gs.Serve(lis)
	t.Cleanup(func() { gs.Stop(); rc.Close() })

	conn, _ := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	t.Cleanup(func() { conn.Close() })

	rdb := goredis.NewClient(&goredis.Options{Addr: redisAddr})
	t.Cleanup(func() { rdb.Close() })
	return pb.NewExternalScalerClient(conn), rdb
}

// redisAddr strips the "redis://" scheme from a Redis ConnectionString result.
func redisAddr(connStr string) string {
	return strings.TrimPrefix(connStr, "redis://")
}

// getMetrics is a thin helper to reduce call-site noise.
func getMetrics(t *testing.T, client pb.ExternalScalerClient, queueName string) int64 {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	resp, err := client.GetMetrics(ctx, &pb.GetMetricsRequest{
		ScaledObjectRef: &pb.ScaledObjectRef{
			ScalerMetadata: map[string]string{
				"queueName":         queueName,
				"jobTimeoutSeconds": "3600",
			},
		},
	})
	if err != nil {
		t.Fatalf("GetMetrics(%s): %v", queueName, err)
	}
	return resp.MetricValues[0].MetricValue
}

// TestIntegration_FullLifecycle verifies that the scaler reflects real Redis state:
// job insertion raises the metric and job removal lowers it.
func TestIntegration_FullLifecycle(t *testing.T) {
	ctx := context.Background()

	redisContainer, err := tcredis.Run(ctx, "redis:7")
	if err != nil {
		t.Fatalf("start redis container: %v", err)
	}
	t.Cleanup(func() { redisContainer.Terminate(ctx) })

	connStr, err := redisContainer.ConnectionString(ctx)
	if err != nil {
		t.Fatalf("redis connection string: %v", err)
	}
	addr := redisAddr(connStr)

	client, rdb := startIntegrationServer(t, addr)

	future := time.Now().Add(1 * time.Hour).Unix()
	rdb.HSet(ctx, "token_backlog:q1", "j1", "100000", "j2", "200000", "j3", "50000")
	rdb.ZAdd(ctx, "token_expiry:q1",
		goredis.Z{Score: float64(future), Member: "j1"},
		goredis.Z{Score: float64(future), Member: "j2"},
		goredis.Z{Score: float64(future), Member: "j3"},
	)

	if got := getMetrics(t, client, "q1"); got != 350000 {
		t.Fatalf("want 350000, got %d", got)
	}

	// Simulate job completion by removing j2 from both structures.
	rdb.HDel(ctx, "token_backlog:q1", "j2")
	rdb.ZRem(ctx, "token_expiry:q1", "j2")

	if got := getMetrics(t, client, "q1"); got != 150000 {
		t.Fatalf("want 150000 after j2 removal, got %d", got)
	}

	// Complete remaining jobs — metric must reach zero.
	rdb.HDel(ctx, "token_backlog:q1", "j1", "j3")
	rdb.ZRem(ctx, "token_expiry:q1", "j1", "j3")

	if got := getMetrics(t, client, "q1"); got != 0 {
		t.Fatalf("want 0 after all jobs removed, got %d", got)
	}
}

// TestIntegration_CleanupCeiling verifies that the Lua evaluator processes expired
// entries in bounded chunks (chunkSize=100). With 105 expired entries:
//   - cycle 1 purges 100, returning the sum of the 5 remaining (500)
//   - cycle 2 purges the last 5, returning 0
func TestIntegration_CleanupCeiling(t *testing.T) {
	ctx := context.Background()

	redisContainer, err := tcredis.Run(ctx, "redis:7")
	if err != nil {
		t.Fatalf("start redis container: %v", err)
	}
	t.Cleanup(func() { redisContainer.Terminate(ctx) })

	connStr, err := redisContainer.ConnectionString(ctx)
	if err != nil {
		t.Fatalf("redis connection string: %v", err)
	}
	addr := redisAddr(connStr)

	client, rdb := startIntegrationServer(t, addr)

	past := time.Now().Add(-1 * time.Hour).Unix()
	for i := 0; i < 105; i++ {
		id := fmt.Sprintf("j%d", i)
		rdb.HSet(ctx, "token_backlog:q-ceil", id, "100")
		rdb.ZAdd(ctx, "token_expiry:q-ceil", goredis.Z{Score: float64(past), Member: id})
	}

	// Cycle 1: 100 entries purged, 5 remain with 100 tokens each → sum = 500
	if got := getMetrics(t, client, "q-ceil"); got != 500 {
		t.Fatalf("cycle 1: want 500, got %d", got)
	}

	// Cycle 2: remaining 5 purged → sum = 0
	if got := getMetrics(t, client, "q-ceil"); got != 0 {
		t.Fatalf("cycle 2: want 0, got %d", got)
	}
}

// TestIntegration_HPABoundaryMath verifies that large token counts above the
// typical HPA boundary (500k tokens/replica) are returned without truncation.
func TestIntegration_HPABoundaryMath(t *testing.T) {
	ctx := context.Background()

	redisContainer, err := tcredis.Run(ctx, "redis:7")
	if err != nil {
		t.Fatalf("start redis container: %v", err)
	}
	t.Cleanup(func() { redisContainer.Terminate(ctx) })

	connStr, err := redisContainer.ConnectionString(ctx)
	if err != nil {
		t.Fatalf("redis connection string: %v", err)
	}
	addr := redisAddr(connStr)

	client, rdb := startIntegrationServer(t, addr)
	future := time.Now().Add(1 * time.Hour).Unix()

	rdb.HSet(ctx, "token_backlog:q-hpa1", "j1", "510000")
	rdb.ZAdd(ctx, "token_expiry:q-hpa1", goredis.Z{Score: float64(future), Member: "j1"})

	if got := getMetrics(t, client, "q-hpa1"); got != 510000 {
		t.Fatalf("q-hpa1: want 510000, got %d", got)
	}

	rdb.HSet(ctx, "token_backlog:q-hpa2", "j1", "560000")
	rdb.ZAdd(ctx, "token_expiry:q-hpa2", goredis.Z{Score: float64(future), Member: "j1"})

	if got := getMetrics(t, client, "q-hpa2"); got != 560000 {
		t.Fatalf("q-hpa2: want 560000, got %d", got)
	}

	// Boundary: one token above the 500k/replica target. The scaler must return 500001
	// exactly so the HPA ceil-divides to 2 replicas, not 1.
	rdb.HSet(ctx, "token_backlog:q-hpa3", "j1", "500001")
	rdb.ZAdd(ctx, "token_expiry:q-hpa3", goredis.Z{Score: float64(future), Member: "j1"})

	if got := getMetrics(t, client, "q-hpa3"); got != 500001 {
		t.Fatalf("q-hpa3: want 500001 (one above boundary, HPA must scale to 2 replicas), got %d", got)
	}
}

// TestIntegration_NetworkFracture verifies scaler resilience to Redis network
// failures by injecting a timeout toxic via Toxiproxy and asserting that
// GetMetrics returns an error rather than hanging, then recovers after
// the toxic is removed.
func TestIntegration_NetworkFracture(t *testing.T) {
	ctx := context.Background()

	// Shared Docker network so toxiproxy can resolve the "redis" alias.
	nw, err := network.New(ctx)
	if err != nil {
		t.Fatalf("create docker network: %v", err)
	}
	t.Cleanup(func() { nw.Remove(ctx) })

	// Redis container on the shared network, reachable inside Docker as "redis:6379".
	redisContainer, err := tcredis.Run(ctx, "redis:7",
		network.WithNetwork([]string{"redis"}, nw),
	)
	if err != nil {
		t.Fatalf("start redis container: %v", err)
	}
	t.Cleanup(func() { redisContainer.Terminate(ctx) })

	// Toxiproxy container proxying internal "redis:6379" on port 8666.
	toxiContainer, err := tctoxiproxy.Run(ctx, "ghcr.io/shopify/toxiproxy:2.12.0",
		tctoxiproxy.WithProxy("redis", "redis:6379"),
		network.WithNetwork([]string{"toxiproxy"}, nw),
	)
	if err != nil {
		t.Fatalf("start toxiproxy container: %v", err)
	}
	t.Cleanup(func() { toxiContainer.Terminate(ctx) })

	// ProxiedEndpoint(8666) returns the host-mapped address for the redis proxy.
	proxyHost, proxyPort, err := toxiContainer.ProxiedEndpoint(8666)
	if err != nil {
		t.Fatalf("proxied endpoint: %v", err)
	}
	proxiedAddr := net.JoinHostPort(proxyHost, proxyPort)

	client, rdb := startIntegrationServer(t, proxiedAddr)

	// Seed data through the (currently healthy) proxy.
	future := time.Now().Add(1 * time.Hour).Unix()
	rdb.HSet(ctx, "token_backlog:q-net", "j1", "100000")
	rdb.ZAdd(ctx, "token_expiry:q-net", goredis.Z{Score: float64(future), Member: "j1"})

	// Baseline: GetMetrics works normally.
	if got := getMetrics(t, client, "q-net"); got != 100000 {
		t.Fatalf("baseline: want 100000, got %d", got)
	}

	// Obtain the toxiproxy HTTP control endpoint and create a proxy client.
	toxiURI, err := toxiContainer.URI(ctx)
	if err != nil {
		t.Fatalf("toxiproxy URI: %v", err)
	}
	tc := toxiclient.NewClient(toxiURI)

	proxy, err := tc.Proxy("redis")
	if err != nil {
		t.Fatalf("get redis proxy: %v", err)
	}

	// Add a timeout toxic: stops all data and closes the connection after 100ms.
	_, err = proxy.AddToxic("break", "timeout", "upstream", 1.0, toxiclient.Attributes{"timeout": 100})
	if err != nil {
		t.Fatalf("add timeout toxic: %v", err)
	}

	// GetMetrics must fail — not hang.
	callCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	_, callErr := client.GetMetrics(callCtx, &pb.GetMetricsRequest{
		ScaledObjectRef: &pb.ScaledObjectRef{
			ScalerMetadata: map[string]string{
				"queueName":         "q-net",
				"jobTimeoutSeconds": "3600",
			},
		},
	})
	if callErr == nil {
		t.Fatal("expected GetMetrics to fail with broken network, got nil")
	}
	code := status.Code(callErr)
	if code != codes.Internal && code != codes.Unavailable && code != codes.DeadlineExceeded {
		t.Fatalf("expected Internal/Unavailable/DeadlineExceeded, got %v (%v)", code, callErr)
	}

	// Remove the toxic and allow connections to recover.
	if err := proxy.RemoveToxic("break"); err != nil {
		t.Fatalf("remove toxic: %v", err)
	}
	time.Sleep(200 * time.Millisecond)

	// GetMetrics must succeed again with the original value.
	if got := getMetrics(t, client, "q-net"); got != 100000 {
		t.Fatalf("post-recovery: want 100000, got %d", got)
	}
}
