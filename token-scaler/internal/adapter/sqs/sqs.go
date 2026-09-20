package sqs

import (
	"context"
	"fmt"
	"strconv"

	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	awssqs "github.com/aws/aws-sdk-go-v2/service/sqs"
	"github.com/aws/aws-sdk-go-v2/service/sqs/types"
	"token-scaler/internal/adapter"
)

type Adapter struct{}

func New() *Adapter {
	return &Adapter{}
}

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
