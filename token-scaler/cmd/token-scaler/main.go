package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"

	"github.com/prometheus/client_golang/prometheus"
	"google.golang.org/grpc"

	"token-scaler/internal/adapter"
	"token-scaler/internal/adapter/kafka"
	"token-scaler/internal/adapter/sqs"
	"token-scaler/internal/redis"
	"token-scaler/internal/scaler"
	pb "token-scaler/internal/scaler/gen"
)

func main() {
	redisAddr   := requireEnv("REDIS_ADDRESS")
	redisPass   := os.Getenv("REDIS_PASSWORD")
	grpcPort    := envOr("GRPC_PORT", "9090")
	metricsPort := envOr("METRICS_PORT", "8080")
	chunkSize   := envOr("CLEANUP_CHUNK_SIZE", "100")
	queueType   := os.Getenv("QUEUE_TYPE")

	rc, err := redis.NewClient(redisAddr, redisPass)
	if err != nil {
		log.Fatalf("redis connect: %v", err)
	}
	defer rc.Close()

	ctx := context.Background()
	if _, err := rc.LoadScripts(ctx); err != nil {
		log.Fatalf("load Lua scripts: %v", err)
	}

	reg := prometheus.NewRegistry()
	m := scaler.NewMetrics(reg)

	var a adapter.QueueAdapter
	switch queueType {
	case "sqs":
		a = sqs.New()
	case "kafka":
		a = kafka.New()
	}

	srv := scaler.NewExternalScalerServer(rc, m, a).WithChunkSize(chunkSize)

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
