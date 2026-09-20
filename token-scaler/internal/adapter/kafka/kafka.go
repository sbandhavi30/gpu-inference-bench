package kafka

import (
	"context"
	"fmt"
	"strings"

	"github.com/IBM/sarama"
	"token-scaler/internal/adapter"
)

type Adapter struct{}

func New() *Adapter {
	return &Adapter{}
}

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
// Uses GetOffset(OffsetNewest) — does NOT call DescribeGroups (avoids controller stress).
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
