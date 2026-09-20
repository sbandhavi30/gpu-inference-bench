package sqs_test

import (
	"context"
	"testing"

	"token-scaler/internal/adapter/sqs"
)

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
