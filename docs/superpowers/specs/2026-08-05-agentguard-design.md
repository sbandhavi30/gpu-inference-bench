# AgentGuard — Durable Execution + Checkpointing for AI Agents

**Date:** 2026-08-05  
**Status:** Design approved, ready for implementation  
**Scope:** v1 — durable execution + checkpointing only (HITL, rollback, behavioral testing are v2+)

---

## Problem

AI agent frameworks have solved orchestration. Nobody has solved reliability.

Agent crashes at step 31 of 47 — all state lost, full restart required. No way to resume from last known good state. Destructive actions mid-crash leave unknown partial state. Found out via billing alert or user complaint.

Three failure modes:

| Failure | Mechanism |
|---|---|
| Crash-and-restart | In-memory state lost on OOM/network blip/deploy restart |
| Destructive mid-crash | `delete_bucket()` called, crashes mid-call — did it complete? |
| No visibility | No trace of which step failed or what was in-flight |

Current harnesses (E2B, Modal, Daytona) are execution layers only — spin up sandbox, run agent, tear down. No reliability layer exists between "spawn" and "hope it finishes."

---

## Solution

**AgentGuard** — standalone OSS Python library. Reliability + checkpointing layer that sits on top of any agent harness.

- Checkpoints after every tool call, on token pressure, before destructive actions
- Pluggable storage: disk (dev) → Redis (staging) → Postgres (prod)
- Framework adapters: LangGraph, LangChain, raw Anthropic SDK
- Opaque state: framework serializes its own state bytes, AgentGuard stores + restores
- Resume: `app.resume("run_id")` continues from last checkpoint

---

## Architecture

### Approach: Sidecar Store + Framework Adapters

AgentGuard = storage + restore protocol. Framework adapters handle serialization.

```
LangGraph adapter    ──┐
LangChain adapter    ──┤──► AgentGuard Core (checkpoint engine + store) ──► pluggable backend
Anthropic SDK adapter──┘
```

Core stays byte-opaque. Adapters own framework-specific serialization. This prevents dependency hell — core installs without requiring any framework binary.

### Project Structure

```
agentguard/
├── core/
│   ├── types.py           # CheckpointMeta, RestoredState dataclasses
│   ├── store.py           # Abstract StorageBackend interface
│   ├── engine.py          # CheckpointEngine: checkpoint() + restore()
│   ├── triggers.py        # TriggerPolicy + TriggerMeta (byte-opaque, metadata-driven)
│   └── exceptions.py      # AgentGuardError hierarchy
├── stores/
│   ├── disk.py            # Local disk: meta.jsonl + step_{n}.bin files
│   ├── redis.py           # Redis: sorted sets + SETEX TTL
│   └── postgres.py        # Postgres: BYTEA + compound primary key
├── adapters/
│   ├── langgraph.py       # DurableGraph wrapper
│   ├── langchain.py       # DurableExecutor wrapper
│   └── anthropic.py       # DurableAgentLoop wrapper
└── cli.py                 # agentguard list/inspect/resume/prune/delete/stats
```

### Optional Install Extras

```toml
[project.optional-dependencies]
langgraph = ["langgraph>=0.1.0"]
langchain = ["langchain-core>=0.2.0"]
anthropic = ["anthropic>=0.30.0"]
all = ["agentguard[langgraph]", "agentguard[langchain]", "agentguard[anthropic]"]
```

---

## Core Types

```python
# core/types.py
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class CheckpointMeta:
    run_id: str
    step: int
    trigger: str
    timestamp: datetime
    framework: str        # set by adapter — used by CLI for resume snippet detection
    token_count: int = 0
    cost_usd: float = 0.0

@dataclass(frozen=True)
class RestoredState:
    step: int
    state: bytes
```

---

## Core Interface

```python
# core/store.py
class StorageBackend(ABC):
    async def save(self, run_id: str, step: int, state: bytes, meta: CheckpointMeta) -> None
    async def load_latest(self, run_id: str) -> RestoredState | None
    async def list(self, run_id: str) -> list[CheckpointMeta]
    async def delete(self, run_id: str) -> None
    async def prune(self, run_id: str, keep_last: int = 3) -> None

# core/engine.py
class CheckpointEngine:
    async def checkpoint(self, run_id: str, step: int, state: bytes, meta: CheckpointMeta) -> None
    async def restore(self, run_id: str) -> RestoredState  # raises RestoreError if not found
```

---

## Smart Triggers

Trigger detection is metadata-driven. Core is byte-opaque — `triggers.py` never inspects state bytes. Adapters pass `TriggerMeta` with contextual markers.

```python
# core/triggers.py
import fnmatch
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class TriggerMeta:
    tool_name: Optional[str] = None
    token_count: int = 0
    total_budget: int = 0       # model context window limit (e.g. 200000 for Claude)
    is_destructive: bool = False
    step: int = 0

class TriggerPolicy:
    def __init__(
        self,
        destructive_tools: list[str] | None = None,
        token_pressure_threshold: float = 0.80,
    ):
        self.destructive_tools = destructive_tools or []
        self.token_pressure_threshold = token_pressure_threshold

    def should_checkpoint(self, meta: TriggerMeta) -> tuple[bool, str]:
        # 1. Adapter explicitly marked as destructive
        if meta.is_destructive:
            return True, "destructive_action"

        # 2. Wildcard match against configured destructive tool patterns
        if meta.tool_name:
            for pattern in self.destructive_tools:
                if fnmatch.fnmatch(meta.tool_name, pattern):
                    return True, "destructive_action"

        # 3. Token pressure (guard against zero budget)
        if meta.total_budget > 0:
            if meta.token_count / meta.total_budget >= self.token_pressure_threshold:
                return True, "token_pressure"

        # 4. Baseline: always checkpoint on tool call
        return True, "tool_call"
```

---

## Storage Backends

### Disk

```
.agentguard/
  {run_id}/
    meta.jsonl              # append-only CheckpointMeta log (one JSON per line)
    step_00001.bin          # raw state bytes per step
    step_00002.bin
    ...
```

Append-only `meta.jsonl` = fast writes, no locking, human-inspectable.

### Redis

```
agentguard:{run_id}:state:{step}   → bytes  (SETEX with TTL)
agentguard:{run_id}:meta:{step}    → JSON   (CheckpointMeta)
agentguard:{run_id}:latest         → int    (latest step — O(1) restore lookup)
agentguard:{run_id}:steps          → sorted set (for prune via ZRANGE + DEL)
```

TTL on state keys = automatic GC, no cron needed.

### Postgres

```sql
CREATE TABLE agentguard_checkpoints (
    run_id      TEXT        NOT NULL,
    step        INTEGER     NOT NULL,
    trigger     TEXT        NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT now(),
    framework   TEXT        NOT NULL,
    state       BYTEA       NOT NULL,
    token_count INTEGER,
    cost_usd    NUMERIC(10,6),
    PRIMARY KEY (run_id, step)
);

CREATE INDEX idx_agentguard_run_latest
    ON agentguard_checkpoints(run_id, step DESC);
```

`BYTEA` = native binary. Compound PK = natural dedup. Descending index = fast latest lookup. `cost_usd` + `token_count` = free audit log.

---

## Adapter Contract

Each adapter implements three methods:

```python
class AgentGuardAdapter:
    def serialize(self, framework_state: Any) -> bytes
    def deserialize(self, state_bytes: bytes) -> Any
    def extract_trigger_meta(self, event: Any) -> TriggerMeta
```

### Usage per framework

**LangGraph:**
```python
from agentguard.adapters.langgraph import DurableGraph

app = DurableGraph(
    workflow.compile(),
    store=RedisStore("redis://localhost"),
    policy=TriggerPolicy(destructive_tools=["delete_*", "deploy_production"])
)
result = await app.ainvoke({"task": "..."}, run_id="run_abc123")
# resume after crash:
result = await app.resume("run_abc123")
```

**LangChain:**
```python
from agentguard.adapters.langchain import DurableExecutor

chain = DurableExecutor(agent_executor, store=PostgresStore("postgresql://..."))
result = await chain.ainvoke({"input": "..."}, run_id="run_xyz")
result = await chain.resume("run_xyz")
```

**Raw Anthropic SDK:**
```python
from agentguard.adapters.anthropic import DurableAgentLoop

loop = DurableAgentLoop(client, store=DiskStore(".agentguard"))
result = await loop.run(messages, run_id="run_123")
result = await loop.resume("run_123")
```

---

## Error Handling

**Principle:** checkpoint failures never crash the agent. Restore failures always surface.

```python
# core/exceptions.py
class AgentGuardError(Exception): ...
class CheckpointWriteError(AgentGuardError): ...   # storage failed to save
class RestoreError(AgentGuardError): ...            # no checkpoint or corrupted bytes
class DeserializationError(AgentGuardError): ...   # adapter failed to deserialize
class BackendConnectionError(AgentGuardError): ... # backend unreachable
```

**Checkpoint — fail-safe:**
```python
try:
    await self.store.save(run_id, step, state_bytes, meta)
except Exception as e:
    self.logger.warning(f"AgentGuard checkpoint failed: run={run_id} step={step} error={e}")
    if self.on_checkpoint_failure:
        self.on_checkpoint_failure(run_id, step, e)
    # never re-raise — agent continues
```

**Restore — fail-loud:**
```python
result = await self.store.load_latest(run_id)
if result is None:
    raise RestoreError(f"No checkpoint found for run_id={run_id}")
if not result.state:
    raise RestoreError(f"Corrupted checkpoint: run={run_id} step={result.step}")
```

---

## CLI

```bash
agentguard list <run_id>              # list all checkpoints for a run
agentguard inspect <run_id> --step N  # inspect checkpoint metadata
agentguard resume <run_id>            # print resume code snippet (auto-detects framework)
agentguard prune <run_id> --keep 3    # keep last N checkpoints
agentguard delete <run_id>            # delete all checkpoints for a run
agentguard stats                      # aggregate stats across all runs
```

`resume` detects framework from `CheckpointMeta.framework` field and prints correct adapter snippet. No guessing.

---

## Testing Strategy

### Layer 1: Core unit tests (no framework, no storage)
- `TriggerPolicy` — wildcard matching, token pressure, zero budget guard
- `CheckpointEngine` — fail-safe write, fail-loud restore
- `CheckpointMeta` / `RestoredState` — serialization round-trips

### Layer 2: Storage backend tests (testcontainers — real Redis/Postgres in CI)
- Save + load_latest round-trip
- Prune keeps last N, drops older
- Concurrent writes to same run_id
- TTL expiry (Redis)
- Corrupted bytes handling

### Layer 3: Adapter integration tests (InMemoryStore — no backend needed)
- Checkpoints fire on correct triggers
- Serialize → deserialize round-trip per framework
- Resume hands correct state back to framework
- `resumed_from_step` matches last checkpoint step

### CI Matrix

```yaml
strategy:
  matrix:
    python: ["3.11", "3.12"]
    backend: ["disk", "redis", "postgres"]
    adapter: ["anthropic", "langgraph", "langchain"]
```

- Core tests: always run, no services
- Backend tests: testcontainers per job
- Adapter tests: InMemoryStore, no services
- Integration tests: real API — gated behind secret

---

## Out of Scope (v2+)

- HITL async approval flows
- Rollback / undo stack
- Behavioral testing / CI integration
- Policy enforcement layer
- Multi-agent coordination
- ContextStream integration (natural fit via `on_token_pressure` hook — future)
