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

func (n *noopAdapter) HealthCheck(_ context.Context, _ map[string]string) error { return nil }
func (n *noopAdapter) FetchTelemetry(_ context.Context, _ map[string]string) (adapter.QueueTelemetry, error) {
	return adapter.QueueTelemetry{IsHealthy: true}, nil
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
	t.Cleanup(func() { rc.Close() })

	conn, err := grpc.NewClient(lis.Addr().String(),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	return pb.NewExternalScalerClient(conn)
}

func TestIsActive_True(t *testing.T) {
	mr, _ := miniredis.Run()
	defer mr.Close()
	mr.HSet("token_backlog:q1", "job1", "1000")

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
		t.Fatalf("want 1 spec, got %d", len(resp.MetricSpecs))
	}
	if resp.MetricSpecs[0].MetricName != "token_backlog" {
		t.Fatalf("want token_backlog, got %s", resp.MetricSpecs[0].MetricName)
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
	future := time.Now().Add(1 * time.Hour).Unix()
	rdb := goredis.NewClient(&goredis.Options{Addr: mr.Addr()})
	rdb.HSet(context.Background(), "token_backlog:q1", "j1", "100000", "j2", "200000")
	rdb.ZAdd(context.Background(), "token_expiry:q1",
		goredis.Z{Score: float64(future), Member: "j1"},
		goredis.Z{Score: float64(future), Member: "j2"},
	)
	rdb.Close()

	client := startTestServer(t, mr)
	resp, err := client.GetMetrics(context.Background(), &pb.GetMetricsRequest{
		ScaledObjectRef: &pb.ScaledObjectRef{
			ScalerMetadata: map[string]string{
				"queueName":         "q1",
				"jobTimeoutSeconds": "3600",
			},
		},
	})
	if err != nil {
		t.Fatalf("GetMetrics: %v", err)
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
