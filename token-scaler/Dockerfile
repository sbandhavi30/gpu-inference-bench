FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o /token-scaler ./cmd/token-scaler

FROM alpine:3.19
RUN addgroup -S scaler && adduser -S scaler -G scaler
COPY --from=builder /token-scaler /token-scaler
USER scaler
EXPOSE 9090 8080
ENTRYPOINT ["/token-scaler"]
