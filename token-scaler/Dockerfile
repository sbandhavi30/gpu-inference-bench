FROM golang:1.23-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /token-scaler ./cmd/token-scaler

FROM alpine:3.19
RUN addgroup -S scaler && adduser -S -u 10001 -G scaler scaler
COPY --from=builder /token-scaler /token-scaler
USER 10001
EXPOSE 9090 8080
ENTRYPOINT ["/token-scaler"]
