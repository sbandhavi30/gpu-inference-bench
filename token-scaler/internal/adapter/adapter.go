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
