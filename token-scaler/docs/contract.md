# Token Scaler Caller Contract

This document specifies the dual-write protocol upstream publishers and downstream workers must implement to use the token-scaler.

## Publisher Contract

On every job submission, the publisher MUST:

1. Compute token count using a fast local tokenizer (tiktoken, HuggingFace tokenizers) **before** publishing to the queue.
2. Include the token count in the job payload:

```json
{
  "job_id": "unique-uuid",
  "payload": "...",
  "metadata": {
    "estimated_tokens": 42500
  }
}
```

3. After publishing to SQS/Kafka, execute the Redis atomic publish script:

```
EVALSHA <publisher_sha> 3
  token_backlog:{queueName}
  token_expiry:{queueName}
  token_events:{queueName}
  {job_id}
  {estimated_tokens}
  {expiry_epoch}
  {safety_ttl}
```

Where `expiry_epoch = now + jobTimeoutSeconds` and `safety_ttl = jobTimeoutSeconds`.

Python example:
```python
import redis
import time
import uuid

r = redis.Redis(host="redis-host", port=6379)
# Load the script once at startup
publisher_sha = r.script_load(SCRIPT_A_PUBLISHER)

def publish_job(queue_name: str, token_count: int, job_timeout_sec: int = 3600):
    job_id = str(uuid.uuid4())
    expiry_epoch = int(time.time()) + job_timeout_sec
    r.evalsha(
        publisher_sha, 3,
        f"token_backlog:{queue_name}",
        f"token_expiry:{queue_name}",
        f"token_events:{queue_name}",
        job_id, token_count, expiry_epoch, job_timeout_sec
    )
    return job_id
```

## Worker Contract

On successful job completion, the worker MUST:

```
HDEL token_backlog:{queueName} {job_id}
ZREM token_expiry:{queueName} {job_id}
```

Python example:
```python
def complete_job(queue_name: str, job_id: str):
    pipe = r.pipeline()
    pipe.hdel(f"token_backlog:{queue_name}", job_id)
    pipe.zrem(f"token_expiry:{queue_name}", job_id)
    pipe.execute()
```

## Failure Handling

- **Worker crash before completion:** The job entry will expire at `expiry_epoch`. The scaler purges it on the next evaluation cycle. Maximum staleness = `jobTimeoutSeconds` seconds.
- **Worker crash after queue consume but before Redis delete:** Same as above — TTL-based self-healing.
- **Do NOT** use this scaler for agentic workloads where a single job spawns non-deterministic sub-tasks. Token count in the payload must represent the complete job's token budget.

## Script A Source

See `internal/redis/scripts.go` constant `ScriptAPublisher`.
