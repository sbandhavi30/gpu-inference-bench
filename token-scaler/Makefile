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
