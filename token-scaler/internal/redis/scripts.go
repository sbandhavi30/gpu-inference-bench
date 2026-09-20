package redis

import (
	"context"
	"fmt"
)

const ScriptAPublisher = `
local was_empty = redis.call("HLEN", KEYS[1]) == 0
redis.call("HSET", KEYS[1], ARGV[1], ARGV[2])
redis.call("ZADD", KEYS[2], ARGV[3], ARGV[1])
redis.call("PERSIST", KEYS[1])
redis.call("PERSIST", KEYS[2])
if was_empty then
    redis.call("PUBLISH", KEYS[3], "active")
end
return 1
`

const ScriptBEvaluator = `
local expired_ids = redis.call("ZRANGEBYSCORE", KEYS[2], 0, ARGV[1], "LIMIT", 0, tonumber(ARGV[3]))
if #expired_ids > 0 then
    for i = 1, #expired_ids do
        redis.call("HDEL", KEYS[1], expired_ids[i])
        redis.call("ZREM", KEYS[2], expired_ids[i])
    end
end
local total_tokens = 0
for _, val in ipairs(redis.call("HVALS", KEYS[1])) do
    total_tokens = total_tokens + tonumber(val)
end
if #expired_ids > 0 and redis.call("HLEN", KEYS[1]) == 0 then
    redis.call("SET", KEYS[1], "1", "EX", tonumber(ARGV[2]))
    redis.call("EXPIRE", KEYS[2], tonumber(ARGV[2]))
end
return total_tokens
`

// ScriptManager holds SHAs after scripts are loaded into Redis.
type ScriptManager struct {
	PublisherSHA string
	EvaluatorSHA string
}

func (c *Client) LoadScripts(ctx context.Context) (*ScriptManager, error) {
	pubSHA, err := c.rdb.ScriptLoad(ctx, ScriptAPublisher).Result()
	if err != nil {
		return nil, fmt.Errorf("load publisher script: %w", err)
	}
	evalSHA, err := c.rdb.ScriptLoad(ctx, ScriptBEvaluator).Result()
	if err != nil {
		return nil, fmt.Errorf("load evaluator script: %w", err)
	}
	sm := &ScriptManager{PublisherSHA: pubSHA, EvaluatorSHA: evalSHA}
	c.sm = sm
	return sm, nil
}
