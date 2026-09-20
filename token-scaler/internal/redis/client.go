package redis

import (
	"context"
	"fmt"
	"strings"

	goredis "github.com/redis/go-redis/v9"
)

// Client wraps go-redis with script management and NOSCRIPT retry.
type Client struct {
	rdb *goredis.Client
	sm  *ScriptManager
}

func NewClient(addr, password string) (*Client, error) {
	rdb := goredis.NewClient(&goredis.Options{
		Addr:     addr,
		Password: password,
	})
	if err := rdb.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("redis ping: %w", err)
	}
	return &Client{rdb: rdb}, nil
}

func (c *Client) GetHashLength(ctx context.Context, key string) (int64, error) {
	return c.rdb.HLen(ctx, key).Result()
}

func (c *Client) Subscribe(ctx context.Context, channel string) *goredis.PubSub {
	return c.rdb.Subscribe(ctx, channel)
}

func (c *Client) Close() error {
	return c.rdb.Close()
}

// ExecuteEvaluator runs Script B atomically. Retries once on NOSCRIPT (Redis restart).
func (c *Client) ExecuteEvaluator(ctx context.Context, queueName string, nowEpoch int64, safetyTTL, chunkSize string) (int64, error) {
	if c.sm == nil {
		return 0, fmt.Errorf("scripts not loaded: call LoadScripts before ExecuteEvaluator")
	}
	backlogKey := "token_backlog:" + queueName
	expiryKey := "token_expiry:" + queueName
	keys := []string{backlogKey, expiryKey}

	result, err := c.rdb.EvalSha(ctx, c.sm.EvaluatorSHA, keys, nowEpoch, safetyTTL, chunkSize).Int64()
	if err != nil && strings.Contains(err.Error(), "NOSCRIPT") {
		if _, reloadErr := c.LoadScripts(ctx); reloadErr != nil {
			return 0, fmt.Errorf("reload after NOSCRIPT: %w", reloadErr)
		}
		result, err = c.rdb.EvalSha(ctx, c.sm.EvaluatorSHA, keys, nowEpoch, safetyTTL, chunkSize).Int64()
	}
	return result, err
}
