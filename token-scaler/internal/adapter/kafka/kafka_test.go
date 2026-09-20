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
