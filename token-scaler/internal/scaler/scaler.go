package scaler

import (
	"context"
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
	rc                *redis.Client
	metrics           *Metrics
	adapter           adapter.QueueAdapter
	chunkSize         string
	defaultJobTimeout string
}

func NewExternalScalerServer(rc *redis.Client, m *Metrics, a adapter.QueueAdapter) *ExternalScalerServer {
	return &ExternalScalerServer{rc: rc, metrics: m, adapter: a}
}

func (s *ExternalScalerServer) WithChunkSize(size string) *ExternalScalerServer {
	s.chunkSize = size
	return s
}

func (s *ExternalScalerServer) WithDefaultJobTimeout(t string) *ExternalScalerServer {
	s.defaultJobTimeout = t
	return s
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

func (s *ExternalScalerServer) GetMetricSpec(_ context.Context, in *pb.ScaledObjectRef) (*pb.GetMetricSpecResponse, error) {
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
	if in.GetScaledObjectRef() == nil {
		return nil, status.Error(codes.InvalidArgument, "scaledObjectRef required")
	}
	meta := in.GetScaledObjectRef().GetScalerMetadata()
	queueName := meta["queueName"]
	if queueName == "" {
		return nil, status.Error(codes.InvalidArgument, "queueName metadata required")
	}

	safetyTTL := s.defaultJobTimeout
	if safetyTTL == "" {
		safetyTTL = "86400"
	}
	if v := meta["jobTimeoutSeconds"]; v != "" {
		safetyTTL = v
	}
	chunkSize := s.chunkSize
	if chunkSize == "" {
		chunkSize = "100"
	}

	total, err := s.rc.ExecuteEvaluator(ctx, queueName, time.Now().Unix(), safetyTTL, chunkSize)
	if err != nil {
		s.metrics.SetHealthy("get_metrics", false)
		return nil, status.Errorf(codes.Internal, "evaluator: %v", err)
	}
	s.metrics.SetHealthy("get_metrics", true)

	if s.adapter != nil {
		go func() {
			tCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
			defer cancel()
			tel, err := s.adapter.FetchTelemetry(tCtx, meta)
			if err == nil {
				s.metrics.RecordQueueDrift(queueName, tel.ApproximateDepth)
			}
		}()
	}

	return &pb.GetMetricsResponse{
		MetricValues: []*pb.MetricValue{{MetricName: "token_backlog", MetricValue: total}},
	}, nil
}
