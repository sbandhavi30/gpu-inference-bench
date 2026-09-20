# AgentGuard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build AgentGuard — a standalone OSS Python library that adds durable execution and checkpointing to AI agents across LangGraph, LangChain, and the raw Anthropic SDK.

**Architecture:** Sidecar store + framework adapters pattern. Core is byte-opaque: it stores and restores `bytes` without knowing what's inside. Framework adapters own serialization. Pluggable storage backends (disk, Redis, Postgres) swap via constructor injection.

**Tech Stack:** Python 3.11+, pytest, pytest-asyncio, testcontainers, typer (CLI), redis-py, asyncpg, pickle/msgpack (adapter serialization)

## Global Constraints

- Python >= 3.11 (uses `X | Y` union syntax, `match` is available but not required)
- All public async methods use `async def` — no sync wrappers
- Core package (`agentguard/core/`, `agentguard/stores/`) must install with zero framework dependencies
- Framework adapters are optional extras: `agentguard[langgraph]`, `agentguard[langchain]`, `agentguard[anthropic]`
- Checkpoint failures NEVER raise to the agent — log warning and continue
- Restore failures ALWAYS raise `RestoreError` — never silently return wrong state
- All dataclasses in `core/types.py` are `frozen=True`
- No placeholder comments, no TODOs in committed code

---

## File Map

| File | Responsibility |
|---|---|
| `agentguard/__init__.py` | Public re-exports |
| `agentguard/core/types.py` | `CheckpointMeta`, `RestoredState` frozen dataclasses |
| `agentguard/core/exceptions.py` | `AgentGuardError` hierarchy |
| `agentguard/core/store.py` | `StorageBackend` ABC |
| `agentguard/core/triggers.py` | `TriggerMeta`, `TriggerPolicy` |
| `agentguard/core/engine.py` | `CheckpointEngine` — orchestrates store + triggers |
| `agentguard/stores/memory.py` | `InMemoryStore` — test helper, no deps |
| `agentguard/stores/disk.py` | `DiskStore` — local `.agentguard/` dir |
| `agentguard/stores/redis.py` | `RedisStore` — redis-py async |
| `agentguard/stores/postgres.py` | `PostgresStore` — asyncpg |
| `agentguard/adapters/anthropic.py` | `DurableAgentLoop` — raw Anthropic SDK |
| `agentguard/adapters/langgraph.py` | `DurableGraph` — LangGraph wrapper |
| `agentguard/adapters/langchain.py` | `DurableExecutor` — LangChain wrapper |
| `agentguard/cli.py` | `typer` CLI — list/inspect/resume/prune/delete/stats |
| `pyproject.toml` | Package config, optional extras, entry points |
| `tests/core/test_types.py` | Types round-trip tests |
| `tests/core/test_exceptions.py` | Exception hierarchy tests |
| `tests/core/test_triggers.py` | TriggerPolicy unit tests |
| `tests/core/test_engine.py` | CheckpointEngine unit tests |
| `tests/stores/test_memory.py` | InMemoryStore tests |
| `tests/stores/test_disk.py` | DiskStore tests |
| `tests/stores/test_redis.py` | RedisStore tests (testcontainers) |
| `tests/stores/test_postgres.py` | PostgresStore tests (testcontainers) |
| `tests/adapters/test_anthropic.py` | DurableAgentLoop tests (InMemoryStore) |
| `tests/adapters/test_langgraph.py` | DurableGraph tests (InMemoryStore) |
| `tests/adapters/test_langchain.py` | DurableExecutor tests (InMemoryStore) |
| `tests/test_cli.py` | CLI command tests (typer test client) |

---

## Task 1: Project Scaffold + `pyproject.toml`

**Files:**
- Create: `agentguard/__init__.py`
- Create: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/core/__init__.py`
- Create: `tests/stores/__init__.py`
- Create: `tests/adapters/__init__.py`

**Interfaces:**
- Produces: installable package `agentguard` with extras `langgraph`, `langchain`, `anthropic`, `all`

- [ ] **Step 1: Create project root structure**

```bash
mkdir -p agentguard/core agentguard/stores agentguard/adapters
mkdir -p tests/core tests/stores tests/adapters
touch agentguard/__init__.py
touch agentguard/core/__init__.py
touch agentguard/stores/__init__.py
touch agentguard/adapters/__init__.py
touch tests/__init__.py tests/core/__init__.py tests/stores/__init__.py tests/adapters/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "agentguard"
version = "0.1.0"
description = "Durable execution and checkpointing for AI agents"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
langgraph = ["langgraph>=0.1.0"]
langchain = ["langchain-core>=0.2.0"]
anthropic = ["anthropic>=0.30.0"]
redis = ["redis>=5.0.0"]
postgres = ["asyncpg>=0.29.0"]
all = [
    "agentguard[langgraph]",
    "agentguard[langchain]",
    "agentguard[anthropic]",
    "agentguard[redis]",
    "agentguard[postgres]",
]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "testcontainers[redis,postgres]>=4.0.0",
    "typer>=0.12.0",
    "agentguard[all]",
]

[project.scripts]
agentguard = "agentguard.cli:app"

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 3: Write minimal `agentguard/__init__.py`**

```python
from agentguard.core.types import CheckpointMeta, RestoredState
from agentguard.core.exceptions import (
    AgentGuardError,
    CheckpointWriteError,
    RestoreError,
    DeserializationError,
    BackendConnectionError,
)
from agentguard.core.triggers import TriggerMeta, TriggerPolicy
from agentguard.core.engine import CheckpointEngine

__all__ = [
    "CheckpointMeta",
    "RestoredState",
    "AgentGuardError",
    "CheckpointWriteError",
    "RestoreError",
    "DeserializationError",
    "BackendConnectionError",
    "TriggerMeta",
    "TriggerPolicy",
    "CheckpointEngine",
]
```

- [ ] **Step 4: Install in dev mode**

```bash
pip install -e ".[dev]"
```

Expected: no errors, `agentguard` importable.

- [ ] **Step 5: Commit**

```bash
git add agentguard/ tests/ pyproject.toml
git commit -m "feat: scaffold agentguard package"
```

---

## Task 2: Core Types + Exceptions

**Files:**
- Create: `agentguard/core/types.py`
- Create: `agentguard/core/exceptions.py`
- Create: `tests/core/test_types.py`
- Create: `tests/core/test_exceptions.py`

**Interfaces:**
- Produces:
  - `CheckpointMeta(run_id: str, step: int, trigger: str, timestamp: datetime, framework: str, token_count: int = 0, cost_usd: float = 0.0)` — frozen dataclass
  - `RestoredState(step: int, state: bytes)` — frozen dataclass
  - `AgentGuardError`, `CheckpointWriteError`, `RestoreError`, `DeserializationError`, `BackendConnectionError` — exception classes

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_types.py
import json
from datetime import datetime, timezone
from agentguard.core.types import CheckpointMeta, RestoredState


def test_checkpoint_meta_frozen():
    meta = CheckpointMeta(
        run_id="run-1", step=5, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework="langgraph",
    )
    try:
        meta.step = 6  # type: ignore
        assert False, "should be immutable"
    except Exception:
        pass


def test_checkpoint_meta_defaults():
    meta = CheckpointMeta(
        run_id="r", step=1, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework="anthropic",
    )
    assert meta.token_count == 0
    assert meta.cost_usd == 0.0


def test_restored_state_frozen():
    rs = RestoredState(step=3, state=b"hello")
    try:
        rs.step = 4  # type: ignore
        assert False, "should be immutable"
    except Exception:
        pass


def test_restored_state_holds_bytes():
    payload = b"\x00\x01\x02\xff"
    rs = RestoredState(step=1, state=payload)
    assert rs.state == payload
```

```python
# tests/core/test_exceptions.py
from agentguard.core.exceptions import (
    AgentGuardError,
    CheckpointWriteError,
    RestoreError,
    DeserializationError,
    BackendConnectionError,
)


def test_all_inherit_base():
    for cls in [CheckpointWriteError, RestoreError, DeserializationError, BackendConnectionError]:
        assert issubclass(cls, AgentGuardError)


def test_raise_restore_error():
    try:
        raise RestoreError("no checkpoint found for run_id=xyz")
    except AgentGuardError as e:
        assert "xyz" in str(e)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/core/test_types.py tests/core/test_exceptions.py -v
```

Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement `agentguard/core/types.py`**

```python
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CheckpointMeta:
    run_id: str
    step: int
    trigger: str
    timestamp: datetime
    framework: str
    token_count: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class RestoredState:
    step: int
    state: bytes
```

- [ ] **Step 4: Implement `agentguard/core/exceptions.py`**

```python
class AgentGuardError(Exception):
    """Base class for all AgentGuard errors."""


class CheckpointWriteError(AgentGuardError):
    """Storage backend failed to save checkpoint."""


class RestoreError(AgentGuardError):
    """No checkpoint found or checkpoint bytes corrupted."""


class DeserializationError(AgentGuardError):
    """Adapter failed to deserialize state bytes back to framework state."""


class BackendConnectionError(AgentGuardError):
    """Storage backend unreachable."""
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/core/test_types.py tests/core/test_exceptions.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add agentguard/core/types.py agentguard/core/exceptions.py tests/core/
git commit -m "feat: add core types and exception hierarchy"
```

---

## Task 3: TriggerPolicy

**Files:**
- Create: `agentguard/core/triggers.py`
- Create: `tests/core/test_triggers.py`

**Interfaces:**
- Consumes: nothing (pure logic, no imports from other tasks)
- Produces:
  - `TriggerMeta(tool_name: str | None, token_count: int, total_budget: int, is_destructive: bool, step: int)` — frozen dataclass
  - `TriggerPolicy(destructive_tools: list[str] | None, token_pressure_threshold: float)` with method `should_checkpoint(meta: TriggerMeta) -> tuple[bool, str]`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_triggers.py
import pytest
from agentguard.core.triggers import TriggerMeta, TriggerPolicy


def make_meta(**kwargs):
    defaults = dict(
        tool_name=None, token_count=0, total_budget=0,
        is_destructive=False, step=0,
    )
    defaults.update(kwargs)
    return TriggerMeta(**defaults)


def test_explicit_destructive_flag():
    policy = TriggerPolicy()
    result, reason = policy.should_checkpoint(make_meta(is_destructive=True))
    assert result is True
    assert reason == "destructive_action"


def test_wildcard_destructive_match():
    policy = TriggerPolicy(destructive_tools=["delete_*", "drop_*"])
    result, reason = policy.should_checkpoint(make_meta(tool_name="delete_bucket"))
    assert result is True
    assert reason == "destructive_action"


def test_wildcard_no_match_falls_through():
    policy = TriggerPolicy(destructive_tools=["delete_*"])
    result, reason = policy.should_checkpoint(make_meta(tool_name="list_buckets"))
    assert result is True
    assert reason == "tool_call"


def test_token_pressure_triggers():
    policy = TriggerPolicy(token_pressure_threshold=0.80)
    result, reason = policy.should_checkpoint(
        make_meta(token_count=8500, total_budget=10000)
    )
    assert result is True
    assert reason == "token_pressure"


def test_token_pressure_below_threshold():
    policy = TriggerPolicy(token_pressure_threshold=0.80)
    result, reason = policy.should_checkpoint(
        make_meta(token_count=7000, total_budget=10000)
    )
    assert result is True
    assert reason == "tool_call"


def test_zero_budget_no_division_error():
    policy = TriggerPolicy(token_pressure_threshold=0.80)
    result, reason = policy.should_checkpoint(make_meta(total_budget=0, token_count=9999))
    assert result is True
    assert reason == "tool_call"


def test_destructive_flag_takes_priority_over_token_pressure():
    policy = TriggerPolicy(token_pressure_threshold=0.10)
    result, reason = policy.should_checkpoint(
        make_meta(is_destructive=True, token_count=9999, total_budget=10000)
    )
    assert reason == "destructive_action"


def test_trigger_meta_frozen():
    meta = make_meta(step=1)
    try:
        meta.step = 2  # type: ignore
        assert False, "should be frozen"
    except Exception:
        pass
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/core/test_triggers.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `agentguard/core/triggers.py`**

```python
import fnmatch
from dataclasses import dataclass


@dataclass(frozen=True)
class TriggerMeta:
    tool_name: str | None = None
    token_count: int = 0
    total_budget: int = 0
    is_destructive: bool = False
    step: int = 0


class TriggerPolicy:
    def __init__(
        self,
        destructive_tools: list[str] | None = None,
        token_pressure_threshold: float = 0.80,
    ) -> None:
        self.destructive_tools = destructive_tools or []
        self.token_pressure_threshold = token_pressure_threshold

    def should_checkpoint(self, meta: TriggerMeta) -> tuple[bool, str]:
        if meta.is_destructive:
            return True, "destructive_action"

        if meta.tool_name:
            for pattern in self.destructive_tools:
                if fnmatch.fnmatch(meta.tool_name, pattern):
                    return True, "destructive_action"

        if meta.total_budget > 0:
            if meta.token_count / meta.total_budget >= self.token_pressure_threshold:
                return True, "token_pressure"

        return True, "tool_call"
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/core/test_triggers.py -v
```

Expected: all 8 PASS

- [ ] **Step 5: Commit**

```bash
git add agentguard/core/triggers.py tests/core/test_triggers.py
git commit -m "feat: add TriggerPolicy with wildcard matching and token pressure"
```

---

## Task 4: StorageBackend ABC + InMemoryStore

**Files:**
- Create: `agentguard/core/store.py`
- Create: `agentguard/stores/memory.py`
- Create: `tests/stores/test_memory.py`

**Interfaces:**
- Consumes:
  - `CheckpointMeta(run_id, step, trigger, timestamp, framework, token_count, cost_usd)` from `agentguard.core.types`
  - `RestoredState(step, state)` from `agentguard.core.types`
- Produces:
  - `StorageBackend` ABC with methods: `save`, `load_latest`, `list`, `delete`, `prune`
  - `InMemoryStore(ttl_seconds: int | None = None)` — concrete implementation for tests

- [ ] **Step 1: Write failing tests**

```python
# tests/stores/test_memory.py
import pytest
from datetime import datetime, timezone
from agentguard.stores.memory import InMemoryStore
from agentguard.core.types import CheckpointMeta, RestoredState


def make_meta(run_id="run-1", step=1, trigger="tool_call", framework="anthropic"):
    return CheckpointMeta(
        run_id=run_id, step=step, trigger=trigger,
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework=framework,
    )


@pytest.mark.asyncio
async def test_save_and_load_latest():
    store = InMemoryStore()
    meta = make_meta(step=5)
    state = b"serialized-state"
    await store.save("run-1", 5, state, meta)
    restored = await store.load_latest("run-1")
    assert restored is not None
    assert restored.step == 5
    assert restored.state == state


@pytest.mark.asyncio
async def test_load_latest_returns_none_when_empty():
    store = InMemoryStore()
    result = await store.load_latest("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_load_latest_returns_highest_step():
    store = InMemoryStore()
    for step in [1, 3, 2]:
        await store.save("run-1", step, f"state-{step}".encode(), make_meta(step=step))
    restored = await store.load_latest("run-1")
    assert restored.step == 3


@pytest.mark.asyncio
async def test_list_returns_all_meta_desc():
    store = InMemoryStore()
    for step in [1, 2, 3]:
        await store.save("run-1", step, b"s", make_meta(step=step))
    metas = await store.list("run-1")
    assert [m.step for m in metas] == [3, 2, 1]


@pytest.mark.asyncio
async def test_delete_removes_run():
    store = InMemoryStore()
    await store.save("run-1", 1, b"s", make_meta())
    await store.delete("run-1")
    assert await store.load_latest("run-1") is None
    assert await store.list("run-1") == []


@pytest.mark.asyncio
async def test_prune_keeps_last_n():
    store = InMemoryStore()
    for step in range(10):
        await store.save("run-1", step, f"s{step}".encode(), make_meta(step=step))
    await store.prune("run-1", keep_last=3)
    metas = await store.list("run-1")
    assert len(metas) == 3
    assert metas[0].step == 9
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/stores/test_memory.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `agentguard/core/store.py`**

```python
from abc import ABC, abstractmethod
from agentguard.core.types import CheckpointMeta, RestoredState


class StorageBackend(ABC):
    @abstractmethod
    async def save(self, run_id: str, step: int, state: bytes, meta: CheckpointMeta) -> None: ...

    @abstractmethod
    async def load_latest(self, run_id: str) -> RestoredState | None: ...

    @abstractmethod
    async def list(self, run_id: str) -> list[CheckpointMeta]: ...

    @abstractmethod
    async def delete(self, run_id: str) -> None: ...

    @abstractmethod
    async def prune(self, run_id: str, keep_last: int = 3) -> None: ...
```

- [ ] **Step 4: Implement `agentguard/stores/memory.py`**

```python
from agentguard.core.store import StorageBackend
from agentguard.core.types import CheckpointMeta, RestoredState


class InMemoryStore(StorageBackend):
    def __init__(self) -> None:
        # {run_id: {step: (state_bytes, CheckpointMeta)}}
        self._data: dict[str, dict[int, tuple[bytes, CheckpointMeta]]] = {}

    async def save(self, run_id: str, step: int, state: bytes, meta: CheckpointMeta) -> None:
        if run_id not in self._data:
            self._data[run_id] = {}
        self._data[run_id][step] = (state, meta)

    async def load_latest(self, run_id: str) -> RestoredState | None:
        if run_id not in self._data or not self._data[run_id]:
            return None
        latest_step = max(self._data[run_id].keys())
        state, _ = self._data[run_id][latest_step]
        return RestoredState(step=latest_step, state=state)

    async def list(self, run_id: str) -> list[CheckpointMeta]:
        if run_id not in self._data:
            return []
        return [
            meta
            for _, meta in sorted(
                self._data[run_id].values(), key=lambda x: x[1].step, reverse=True
            )
        ]

    async def delete(self, run_id: str) -> None:
        self._data.pop(run_id, None)

    async def prune(self, run_id: str, keep_last: int = 3) -> None:
        if run_id not in self._data:
            return
        steps = sorted(self._data[run_id].keys(), reverse=True)
        for step in steps[keep_last:]:
            del self._data[run_id][step]
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/stores/test_memory.py -v
```

Expected: all 7 PASS

- [ ] **Step 6: Commit**

```bash
git add agentguard/core/store.py agentguard/stores/memory.py tests/stores/test_memory.py
git commit -m "feat: add StorageBackend ABC and InMemoryStore"
```

---

## Task 5: CheckpointEngine

**Files:**
- Create: `agentguard/core/engine.py`
- Create: `tests/core/test_engine.py`

**Interfaces:**
- Consumes:
  - `StorageBackend` from `agentguard.core.store`
  - `TriggerPolicy` from `agentguard.core.triggers`
  - `CheckpointMeta`, `RestoredState` from `agentguard.core.types`
  - `RestoreError` from `agentguard.core.exceptions`
- Produces:
  - `CheckpointEngine(store: StorageBackend, policy: TriggerPolicy | None, on_checkpoint_failure: Callable | None)` with methods:
    - `async checkpoint(run_id, step, state, meta) -> None` — fail-safe
    - `async restore(run_id) -> RestoredState` — fail-loud, raises `RestoreError`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_engine.py
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from agentguard.core.engine import CheckpointEngine
from agentguard.core.exceptions import RestoreError
from agentguard.core.types import CheckpointMeta, RestoredState
from agentguard.stores.memory import InMemoryStore


def make_meta(step=1):
    return CheckpointMeta(
        run_id="run-1", step=step, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework="anthropic",
    )


@pytest.mark.asyncio
async def test_checkpoint_saves_to_store():
    store = InMemoryStore()
    engine = CheckpointEngine(store=store)
    await engine.checkpoint("run-1", 1, b"state", make_meta(1))
    restored = await store.load_latest("run-1")
    assert restored is not None
    assert restored.state == b"state"


@pytest.mark.asyncio
async def test_checkpoint_failure_does_not_raise():
    store = InMemoryStore()
    store.save = AsyncMock(side_effect=Exception("disk full"))
    engine = CheckpointEngine(store=store)
    # Must NOT raise — agent continues
    await engine.checkpoint("run-1", 1, b"state", make_meta(1))


@pytest.mark.asyncio
async def test_checkpoint_failure_calls_callback():
    store = InMemoryStore()
    store.save = AsyncMock(side_effect=Exception("disk full"))
    callback = MagicMock()
    engine = CheckpointEngine(store=store, on_checkpoint_failure=callback)
    await engine.checkpoint("run-1", 1, b"state", make_meta(1))
    callback.assert_called_once()
    args = callback.call_args[0]
    assert args[0] == "run-1"
    assert args[1] == 1


@pytest.mark.asyncio
async def test_restore_returns_latest():
    store = InMemoryStore()
    engine = CheckpointEngine(store=store)
    await engine.checkpoint("run-1", 3, b"state-3", make_meta(3))
    restored = await engine.restore("run-1")
    assert restored.step == 3
    assert restored.state == b"state-3"


@pytest.mark.asyncio
async def test_restore_raises_when_no_checkpoint():
    store = InMemoryStore()
    engine = CheckpointEngine(store=store)
    with pytest.raises(RestoreError, match="run_id=missing-run"):
        await engine.restore("missing-run")


@pytest.mark.asyncio
async def test_restore_raises_on_empty_state():
    store = InMemoryStore()
    # Manually seed corrupted checkpoint
    from agentguard.core.types import CheckpointMeta
    from datetime import datetime, timezone
    meta = CheckpointMeta(
        run_id="run-1", step=1, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework="anthropic",
    )
    await store.save("run-1", 1, b"", meta)
    engine = CheckpointEngine(store=store)
    with pytest.raises(RestoreError, match="Corrupted"):
        await engine.restore("run-1")
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/core/test_engine.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `agentguard/core/engine.py`**

```python
import logging
from collections.abc import Callable
from agentguard.core.exceptions import RestoreError
from agentguard.core.store import StorageBackend
from agentguard.core.triggers import TriggerPolicy
from agentguard.core.types import CheckpointMeta, RestoredState

logger = logging.getLogger(__name__)


class CheckpointEngine:
    def __init__(
        self,
        store: StorageBackend,
        policy: TriggerPolicy | None = None,
        on_checkpoint_failure: Callable[[str, int, Exception], None] | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or TriggerPolicy()
        self.on_checkpoint_failure = on_checkpoint_failure

    async def checkpoint(
        self, run_id: str, step: int, state: bytes, meta: CheckpointMeta
    ) -> None:
        try:
            await self.store.save(run_id, step, state, meta)
        except Exception as exc:
            logger.warning(
                "AgentGuard checkpoint failed: run=%s step=%d error=%s",
                run_id, step, exc,
            )
            if self.on_checkpoint_failure:
                self.on_checkpoint_failure(run_id, step, exc)

    async def restore(self, run_id: str) -> RestoredState:
        result = await self.store.load_latest(run_id)
        if result is None:
            raise RestoreError(f"No checkpoint found for run_id={run_id}")
        if not result.state:
            raise RestoreError(
                f"Corrupted checkpoint: run={run_id} step={result.step}"
            )
        return result
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/core/test_engine.py -v
```

Expected: all 6 PASS

- [ ] **Step 5: Run all core tests together**

```bash
pytest tests/core/ -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add agentguard/core/engine.py tests/core/test_engine.py
git commit -m "feat: add CheckpointEngine with fail-safe write and fail-loud restore"
```

---

## Task 6: DiskStore

**Files:**
- Create: `agentguard/stores/disk.py`
- Create: `tests/stores/test_disk.py`

**Interfaces:**
- Consumes: `StorageBackend`, `CheckpointMeta`, `RestoredState`
- Produces: `DiskStore(base_dir: str | Path = ".agentguard")` — concrete `StorageBackend`

- [ ] **Step 1: Write failing tests**

```python
# tests/stores/test_disk.py
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from agentguard.stores.disk import DiskStore
from agentguard.core.types import CheckpointMeta, RestoredState


def make_meta(run_id="run-1", step=1, framework="anthropic"):
    return CheckpointMeta(
        run_id=run_id, step=step, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework=framework,
    )


@pytest.fixture
def tmp_store(tmp_path):
    return DiskStore(base_dir=tmp_path)


@pytest.mark.asyncio
async def test_save_creates_files(tmp_store, tmp_path):
    await tmp_store.save("run-1", 1, b"state", make_meta())
    assert (tmp_path / "run-1" / "step_00001.bin").exists()
    assert (tmp_path / "run-1" / "meta.jsonl").exists()


@pytest.mark.asyncio
async def test_save_and_load_latest(tmp_store):
    await tmp_store.save("run-1", 1, b"state-1", make_meta(step=1))
    await tmp_store.save("run-1", 2, b"state-2", make_meta(step=2))
    restored = await tmp_store.load_latest("run-1")
    assert restored.step == 2
    assert restored.state == b"state-2"


@pytest.mark.asyncio
async def test_load_latest_none_when_empty(tmp_store):
    result = await tmp_store.load_latest("no-such-run")
    assert result is None


@pytest.mark.asyncio
async def test_list_returns_desc(tmp_store):
    for step in [1, 2, 3]:
        await tmp_store.save("run-1", step, b"s", make_meta(step=step))
    metas = await tmp_store.list("run-1")
    assert [m.step for m in metas] == [3, 2, 1]


@pytest.mark.asyncio
async def test_delete_removes_directory(tmp_store, tmp_path):
    await tmp_store.save("run-1", 1, b"s", make_meta())
    await tmp_store.delete("run-1")
    assert not (tmp_path / "run-1").exists()


@pytest.mark.asyncio
async def test_prune_keeps_last_n(tmp_store, tmp_path):
    for step in range(5):
        await tmp_store.save("run-1", step, f"s{step}".encode(), make_meta(step=step))
    await tmp_store.prune("run-1", keep_last=2)
    metas = await tmp_store.list("run-1")
    assert len(metas) == 2
    assert metas[0].step == 4
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/stores/test_disk.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `agentguard/stores/disk.py`**

```python
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from agentguard.core.store import StorageBackend
from agentguard.core.types import CheckpointMeta, RestoredState


class DiskStore(StorageBackend):
    def __init__(self, base_dir: str | Path = ".agentguard") -> None:
        self._base = Path(base_dir)

    def _run_dir(self, run_id: str) -> Path:
        return self._base / run_id

    def _meta_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "meta.jsonl"

    def _step_path(self, run_id: str, step: int) -> Path:
        return self._run_dir(run_id) / f"step_{step:05d}.bin"

    async def save(self, run_id: str, step: int, state: bytes, meta: CheckpointMeta) -> None:
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._step_path(run_id, step).write_bytes(state)
        with self._meta_path(run_id).open("a") as f:
            f.write(json.dumps({
                "run_id": meta.run_id,
                "step": meta.step,
                "trigger": meta.trigger,
                "timestamp": meta.timestamp.isoformat(),
                "framework": meta.framework,
                "token_count": meta.token_count,
                "cost_usd": meta.cost_usd,
            }) + "\n")

    async def load_latest(self, run_id: str) -> RestoredState | None:
        metas = await self.list(run_id)
        if not metas:
            return None
        latest = metas[0]  # list() returns desc order
        state = self._step_path(run_id, latest.step).read_bytes()
        return RestoredState(step=latest.step, state=state)

    async def list(self, run_id: str) -> list[CheckpointMeta]:
        meta_path = self._meta_path(run_id)
        if not meta_path.exists():
            return []
        metas = []
        seen_steps: set[int] = set()
        for line in meta_path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            step = d["step"]
            if step in seen_steps:
                continue
            seen_steps.add(step)
            metas.append(CheckpointMeta(
                run_id=d["run_id"],
                step=step,
                trigger=d["trigger"],
                timestamp=datetime.fromisoformat(d["timestamp"]),
                framework=d["framework"],
                token_count=d.get("token_count", 0),
                cost_usd=d.get("cost_usd", 0.0),
            ))
        return sorted(metas, key=lambda m: m.step, reverse=True)

    async def delete(self, run_id: str) -> None:
        run_dir = self._run_dir(run_id)
        if run_dir.exists():
            shutil.rmtree(run_dir)

    async def prune(self, run_id: str, keep_last: int = 3) -> None:
        metas = await self.list(run_id)
        for meta in metas[keep_last:]:
            step_file = self._step_path(run_id, meta.step)
            if step_file.exists():
                step_file.unlink()
        # Rewrite meta.jsonl with only kept steps
        kept_steps = {m.step for m in metas[:keep_last]}
        kept_metas = [m for m in metas if m.step in kept_steps]
        meta_path = self._meta_path(run_id)
        with meta_path.open("w") as f:
            for m in sorted(kept_metas, key=lambda x: x.step):
                f.write(json.dumps({
                    "run_id": m.run_id, "step": m.step, "trigger": m.trigger,
                    "timestamp": m.timestamp.isoformat(), "framework": m.framework,
                    "token_count": m.token_count, "cost_usd": m.cost_usd,
                }) + "\n")
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/stores/test_disk.py -v
```

Expected: all 7 PASS

- [ ] **Step 5: Commit**

```bash
git add agentguard/stores/disk.py tests/stores/test_disk.py
git commit -m "feat: add DiskStore backend"
```

---

## Task 7: RedisStore

**Files:**
- Create: `agentguard/stores/redis.py`
- Create: `tests/stores/test_redis.py`

**Interfaces:**
- Consumes: `StorageBackend`, `CheckpointMeta`, `RestoredState`
- Produces: `RedisStore(url: str, ttl_seconds: int = 86400 * 7)` — concrete `StorageBackend`

- [ ] **Step 1: Write failing tests**

```python
# tests/stores/test_redis.py
import pytest
from datetime import datetime, timezone
from testcontainers.redis import RedisContainer
from agentguard.stores.redis import RedisStore
from agentguard.core.types import CheckpointMeta


def make_meta(run_id="run-1", step=1, framework="langgraph"):
    return CheckpointMeta(
        run_id=run_id, step=step, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework=framework,
    )


@pytest.fixture(scope="module")
def redis_url():
    with RedisContainer() as container:
        yield f"redis://{container.get_container_host_ip()}:{container.get_exposed_port(6379)}"


@pytest.fixture
async def store(redis_url):
    s = RedisStore(url=redis_url)
    yield s
    await s.aclose()


@pytest.mark.asyncio
async def test_save_and_load_latest(store):
    await store.save("run-r1", 3, b"state-3", make_meta(step=3))
    restored = await store.load_latest("run-r1")
    assert restored is not None
    assert restored.step == 3
    assert restored.state == b"state-3"


@pytest.mark.asyncio
async def test_load_latest_none_when_empty(store):
    result = await store.load_latest("run-nonexistent-xyz")
    assert result is None


@pytest.mark.asyncio
async def test_list_returns_desc(store):
    for step in [1, 2, 3]:
        await store.save("run-r2", step, b"s", make_meta(run_id="run-r2", step=step))
    metas = await store.list("run-r2")
    assert [m.step for m in metas] == [3, 2, 1]


@pytest.mark.asyncio
async def test_delete_clears_all_keys(store):
    await store.save("run-r3", 1, b"s", make_meta(run_id="run-r3"))
    await store.delete("run-r3")
    assert await store.load_latest("run-r3") is None


@pytest.mark.asyncio
async def test_prune_keeps_last_n(store):
    for step in range(6):
        await store.save("run-r4", step, f"s{step}".encode(), make_meta(run_id="run-r4", step=step))
    await store.prune("run-r4", keep_last=2)
    metas = await store.list("run-r4")
    assert len(metas) == 2
    assert metas[0].step == 5
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/stores/test_redis.py -v
```

Expected: `ImportError` or container not found

- [ ] **Step 3: Implement `agentguard/stores/redis.py`**

```python
import json
from datetime import datetime
from agentguard.core.store import StorageBackend
from agentguard.core.types import CheckpointMeta, RestoredState

try:
    import redis.asyncio as aioredis
except ImportError as exc:
    raise ImportError("Install agentguard[redis] to use RedisStore") from exc


class RedisStore(StorageBackend):
    def __init__(self, url: str, ttl_seconds: int = 86400 * 7) -> None:
        self._client = aioredis.from_url(url, decode_responses=False)
        self._ttl = ttl_seconds

    def _state_key(self, run_id: str, step: int) -> str:
        return f"agentguard:{run_id}:state:{step}"

    def _meta_key(self, run_id: str, step: int) -> str:
        return f"agentguard:{run_id}:meta:{step}"

    def _steps_key(self, run_id: str) -> str:
        return f"agentguard:{run_id}:steps"

    async def save(self, run_id: str, step: int, state: bytes, meta: CheckpointMeta) -> None:
        pipe = self._client.pipeline()
        pipe.setex(self._state_key(run_id, step), self._ttl, state)
        pipe.setex(
            self._meta_key(run_id, step),
            self._ttl,
            json.dumps({
                "run_id": meta.run_id, "step": meta.step, "trigger": meta.trigger,
                "timestamp": meta.timestamp.isoformat(), "framework": meta.framework,
                "token_count": meta.token_count, "cost_usd": meta.cost_usd,
            }).encode(),
        )
        pipe.zadd(self._steps_key(run_id), {str(step): step})
        pipe.expire(self._steps_key(run_id), self._ttl)
        await pipe.execute()

    async def load_latest(self, run_id: str) -> RestoredState | None:
        steps = await self._client.zrange(self._steps_key(run_id), -1, -1)
        if not steps:
            return None
        latest_step = int(steps[0])
        state = await self._client.get(self._state_key(run_id, latest_step))
        if state is None:
            return None
        return RestoredState(step=latest_step, state=state)

    async def list(self, run_id: str) -> list[CheckpointMeta]:
        steps = await self._client.zrange(self._steps_key(run_id), 0, -1, withscores=False, rev=True)
        metas = []
        for step_bytes in steps:
            step = int(step_bytes)
            raw = await self._client.get(self._meta_key(run_id, step))
            if raw is None:
                continue
            d = json.loads(raw)
            metas.append(CheckpointMeta(
                run_id=d["run_id"], step=d["step"], trigger=d["trigger"],
                timestamp=datetime.fromisoformat(d["timestamp"]),
                framework=d["framework"], token_count=d.get("token_count", 0),
                cost_usd=d.get("cost_usd", 0.0),
            ))
        return metas

    async def delete(self, run_id: str) -> None:
        steps = await self._client.zrange(self._steps_key(run_id), 0, -1)
        pipe = self._client.pipeline()
        for step_bytes in steps:
            step = int(step_bytes)
            pipe.delete(self._state_key(run_id, step))
            pipe.delete(self._meta_key(run_id, step))
        pipe.delete(self._steps_key(run_id))
        await pipe.execute()

    async def prune(self, run_id: str, keep_last: int = 3) -> None:
        steps = await self._client.zrange(self._steps_key(run_id), 0, -(keep_last + 1), rev=False)
        if not steps:
            return
        pipe = self._client.pipeline()
        for step_bytes in steps:
            step = int(step_bytes)
            pipe.delete(self._state_key(run_id, step))
            pipe.delete(self._meta_key(run_id, step))
            pipe.zrem(self._steps_key(run_id), str(step))
        await pipe.execute()

    async def aclose(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/stores/test_redis.py -v
```

Expected: all 5 PASS (requires Docker for testcontainers)

- [ ] **Step 5: Commit**

```bash
git add agentguard/stores/redis.py tests/stores/test_redis.py
git commit -m "feat: add RedisStore backend with TTL and sorted-set step index"
```

---

## Task 8: PostgresStore

**Files:**
- Create: `agentguard/stores/postgres.py`
- Create: `tests/stores/test_postgres.py`

**Interfaces:**
- Consumes: `StorageBackend`, `CheckpointMeta`, `RestoredState`
- Produces: `PostgresStore(dsn: str)` — concrete `StorageBackend`

- [ ] **Step 1: Write failing tests**

```python
# tests/stores/test_postgres.py
import pytest
from datetime import datetime, timezone
from testcontainers.postgres import PostgresContainer
from agentguard.stores.postgres import PostgresStore
from agentguard.core.types import CheckpointMeta


def make_meta(run_id="run-1", step=1, framework="langchain"):
    return CheckpointMeta(
        run_id=run_id, step=step, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework=framework,
    )


@pytest.fixture(scope="module")
def pg_dsn():
    with PostgresContainer("postgres:16") as pg:
        yield pg.get_connection_url().replace("psycopg2", "asyncpg").replace("+asyncpg", "")


@pytest.fixture
async def store(pg_dsn):
    s = PostgresStore(dsn=pg_dsn)
    await s.initialize()
    yield s
    await s.aclose()


@pytest.mark.asyncio
async def test_save_and_load_latest(store):
    await store.save("run-p1", 5, b"pg-state", make_meta(step=5))
    restored = await store.load_latest("run-p1")
    assert restored.step == 5
    assert restored.state == b"pg-state"


@pytest.mark.asyncio
async def test_load_latest_none_when_empty(store):
    result = await store.load_latest("run-pg-empty")
    assert result is None


@pytest.mark.asyncio
async def test_list_returns_desc(store):
    for step in [1, 2, 3]:
        await store.save("run-p2", step, b"s", make_meta(run_id="run-p2", step=step))
    metas = await store.list("run-p2")
    assert [m.step for m in metas] == [3, 2, 1]


@pytest.mark.asyncio
async def test_delete_removes_rows(store):
    await store.save("run-p3", 1, b"s", make_meta(run_id="run-p3"))
    await store.delete("run-p3")
    assert await store.load_latest("run-p3") is None


@pytest.mark.asyncio
async def test_prune_keeps_last_n(store):
    for step in range(5):
        await store.save("run-p4", step, f"s{step}".encode(), make_meta(run_id="run-p4", step=step))
    await store.prune("run-p4", keep_last=2)
    metas = await store.list("run-p4")
    assert len(metas) == 2
    assert metas[0].step == 4
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/stores/test_postgres.py -v
```

Expected: `ImportError` or container error

- [ ] **Step 3: Implement `agentguard/stores/postgres.py`**

```python
from datetime import datetime
from agentguard.core.store import StorageBackend
from agentguard.core.types import CheckpointMeta, RestoredState

try:
    import asyncpg
except ImportError as exc:
    raise ImportError("Install agentguard[postgres] to use PostgresStore") from exc

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS agentguard_checkpoints (
    run_id      TEXT        NOT NULL,
    step        INTEGER     NOT NULL,
    trigger     TEXT        NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT now(),
    framework   TEXT        NOT NULL,
    state       BYTEA       NOT NULL,
    token_count INTEGER     DEFAULT 0,
    cost_usd    NUMERIC(10,6) DEFAULT 0,
    PRIMARY KEY (run_id, step)
);
CREATE INDEX IF NOT EXISTS idx_agentguard_run_latest
    ON agentguard_checkpoints(run_id, step DESC);
"""


class PostgresStore(StorageBackend):
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE)

    async def _pool_or_raise(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Call await store.initialize() before use")
        return self._pool

    async def save(self, run_id: str, step: int, state: bytes, meta: CheckpointMeta) -> None:
        pool = await self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agentguard_checkpoints
                    (run_id, step, trigger, timestamp, framework, state, token_count, cost_usd)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                ON CONFLICT (run_id, step) DO UPDATE
                    SET state=EXCLUDED.state, trigger=EXCLUDED.trigger,
                        timestamp=EXCLUDED.timestamp, token_count=EXCLUDED.token_count,
                        cost_usd=EXCLUDED.cost_usd
                """,
                run_id, step, meta.trigger, meta.timestamp, meta.framework,
                state, meta.token_count, meta.cost_usd,
            )

    async def load_latest(self, run_id: str) -> RestoredState | None:
        pool = await self._pool_or_raise()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT step, state FROM agentguard_checkpoints "
                "WHERE run_id=$1 ORDER BY step DESC LIMIT 1",
                run_id,
            )
        if row is None:
            return None
        return RestoredState(step=row["step"], state=bytes(row["state"]))

    async def list(self, run_id: str) -> list[CheckpointMeta]:
        pool = await self._pool_or_raise()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT run_id,step,trigger,timestamp,framework,token_count,cost_usd "
                "FROM agentguard_checkpoints WHERE run_id=$1 ORDER BY step DESC",
                run_id,
            )
        return [
            CheckpointMeta(
                run_id=r["run_id"], step=r["step"], trigger=r["trigger"],
                timestamp=r["timestamp"], framework=r["framework"],
                token_count=r["token_count"] or 0, cost_usd=float(r["cost_usd"] or 0),
            )
            for r in rows
        ]

    async def delete(self, run_id: str) -> None:
        pool = await self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM agentguard_checkpoints WHERE run_id=$1", run_id
            )

    async def prune(self, run_id: str, keep_last: int = 3) -> None:
        pool = await self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                DELETE FROM agentguard_checkpoints
                WHERE run_id=$1 AND step NOT IN (
                    SELECT step FROM agentguard_checkpoints
                    WHERE run_id=$1 ORDER BY step DESC LIMIT $2
                )
                """,
                run_id, keep_last,
            )

    async def aclose(self) -> None:
        if self._pool:
            await self._pool.close()
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/stores/test_postgres.py -v
```

Expected: all 5 PASS (requires Docker)

- [ ] **Step 5: Commit**

```bash
git add agentguard/stores/postgres.py tests/stores/test_postgres.py
git commit -m "feat: add PostgresStore backend with BYTEA and compound primary key"
```

---

## Task 9: Anthropic SDK Adapter

**Files:**
- Create: `agentguard/adapters/anthropic.py`
- Create: `tests/adapters/test_anthropic.py`

**Interfaces:**
- Consumes: `CheckpointEngine`, `TriggerPolicy`, `TriggerMeta`, `InMemoryStore`, `CheckpointMeta`, `RestoredState`, `DeserializationError`
- Produces: `DurableAgentLoop(client, store, policy, framework_name="anthropic")` with:
  - `async run(messages, run_id, model, tools, **kwargs) -> Any`
  - `async resume(run_id, model, tools, **kwargs) -> Any`

- [ ] **Step 1: Write failing tests**

```python
# tests/adapters/test_anthropic.py
import pickle
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from agentguard.adapters.anthropic import DurableAgentLoop
from agentguard.stores.memory import InMemoryStore
from agentguard.core.triggers import TriggerPolicy
from agentguard.core.exceptions import RestoreError


def make_mock_client(responses):
    """Return an Anthropic client mock that yields responses in sequence."""
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=responses)
    return client


def end_response(text="done"):
    r = MagicMock()
    r.stop_reason = "end_turn"
    r.content = [MagicMock(type="text", text=text)]
    r.usage = MagicMock(input_tokens=100, output_tokens=50)
    return r


def tool_response(tool_name="list_files"):
    r = MagicMock()
    r.stop_reason = "tool_use"
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = "tool_abc"
    block.input = {}
    r.content = [block]
    r.usage = MagicMock(input_tokens=200, output_tokens=80)
    return r


@pytest.mark.asyncio
async def test_run_checkpoints_after_tool_call():
    store = InMemoryStore()
    client = make_mock_client([tool_response("list_files"), end_response()])
    loop = DurableAgentLoop(client=client, store=store)

    async def fake_execute(tool_name, tool_input):
        return [{"type": "tool_result", "tool_use_id": "tool_abc", "content": "[]"}]

    await loop.run(
        messages=[{"role": "user", "content": "list files"}],
        run_id="run-a1",
        model="claude-sonnet-4-6",
        tools=[],
        tool_executor=fake_execute,
    )
    metas = await store.list("run-a1")
    assert len(metas) >= 1


@pytest.mark.asyncio
async def test_resume_restores_from_checkpoint():
    store = InMemoryStore()
    # Pre-seed a checkpoint with serialized messages
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    state = pickle.dumps({"messages": messages, "step": 2})
    from agentguard.core.types import CheckpointMeta
    meta = CheckpointMeta(
        run_id="run-a2", step=2, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework="anthropic",
    )
    await store.save("run-a2", 2, state, meta)

    client = make_mock_client([end_response("resumed ok")])
    loop = DurableAgentLoop(client=client, store=store)

    result = await loop.resume(
        run_id="run-a2",
        model="claude-sonnet-4-6",
        tools=[],
    )
    assert result is not None


@pytest.mark.asyncio
async def test_resume_raises_when_no_checkpoint():
    store = InMemoryStore()
    client = MagicMock()
    loop = DurableAgentLoop(client=client, store=store)
    with pytest.raises(RestoreError):
        await loop.resume(run_id="missing-run", model="claude-sonnet-4-6", tools=[])
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/adapters/test_anthropic.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `agentguard/adapters/anthropic.py`**

```python
import logging
import pickle
from collections.abc import Callable, Awaitable
from datetime import datetime, timezone
from agentguard.core.engine import CheckpointEngine
from agentguard.core.exceptions import DeserializationError
from agentguard.core.store import StorageBackend
from agentguard.core.triggers import TriggerMeta, TriggerPolicy
from agentguard.core.types import CheckpointMeta

logger = logging.getLogger(__name__)

_TOTAL_BUDGET = 200_000  # Claude default context window


class DurableAgentLoop:
    def __init__(
        self,
        client,
        store: StorageBackend,
        policy: TriggerPolicy | None = None,
        framework_name: str = "anthropic",
        on_checkpoint_failure: Callable[[str, int, Exception], None] | None = None,
    ) -> None:
        self._client = client
        self._engine = CheckpointEngine(
            store=store,
            policy=policy or TriggerPolicy(),
            on_checkpoint_failure=on_checkpoint_failure,
        )
        self._framework = framework_name

    def _serialize(self, messages: list, step: int) -> bytes:
        return pickle.dumps({"messages": messages, "step": step})

    def _deserialize(self, state: bytes) -> dict:
        try:
            return pickle.loads(state)  # noqa: S301
        except Exception as exc:
            raise DeserializationError(f"Failed to deserialize Anthropic state: {exc}") from exc

    async def _make_meta(self, run_id: str, step: int, tool_name: str | None, token_count: int, trigger: str) -> CheckpointMeta:
        return CheckpointMeta(
            run_id=run_id, step=step, trigger=trigger,
            timestamp=datetime.now(tz=timezone.utc),
            framework=self._framework,
            token_count=token_count,
        )

    async def run(
        self,
        messages: list,
        run_id: str,
        model: str,
        tools: list,
        tool_executor: Callable[[str, dict], Awaitable[list]] | None = None,
        **kwargs,
    ):
        step = 0
        current_messages = list(messages)

        while True:
            response = await self._client.messages.create(
                model=model, messages=current_messages, tools=tools, **kwargs
            )
            token_count = getattr(response.usage, "input_tokens", 0)

            if response.stop_reason == "tool_use":
                tool_blocks = [b for b in response.content if b.type == "tool_use"]
                for block in tool_blocks:
                    trigger_meta = TriggerMeta(
                        tool_name=block.name,
                        token_count=token_count,
                        total_budget=_TOTAL_BUDGET,
                        step=step,
                    )
                    _, trigger_reason = self._engine.policy.should_checkpoint(trigger_meta)
                    state_bytes = self._serialize(current_messages, step)
                    meta = await self._make_meta(run_id, step, block.name, token_count, trigger_reason)
                    await self._engine.checkpoint(run_id, step, state_bytes, meta)

                    if tool_executor:
                        results = await tool_executor(block.name, block.input)
                        current_messages = current_messages + [
                            {"role": "assistant", "content": response.content},
                            {"role": "user", "content": results},
                        ]
                step += 1

            else:
                return response

    async def resume(self, run_id: str, model: str, tools: list, **kwargs):
        restored = await self._engine.restore(run_id)
        state = self._deserialize(restored.state)
        messages = state["messages"]
        return await self.run(
            messages=messages,
            run_id=run_id,
            model=model,
            tools=tools,
            **kwargs,
        )
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/adapters/test_anthropic.py -v
```

Expected: all 3 PASS

- [ ] **Step 5: Commit**

```bash
git add agentguard/adapters/anthropic.py tests/adapters/test_anthropic.py
git commit -m "feat: add DurableAgentLoop adapter for raw Anthropic SDK"
```

---

## Task 10: LangGraph Adapter

**Files:**
- Create: `agentguard/adapters/langgraph.py`
- Create: `tests/adapters/test_langgraph.py`

**Interfaces:**
- Consumes: `CheckpointEngine`, `TriggerPolicy`, `TriggerMeta`, `InMemoryStore`, `CheckpointMeta`, `DeserializationError`
- Produces: `DurableGraph(graph, store, policy, framework_name="langgraph")` with:
  - `async ainvoke(input_dict, run_id, **kwargs) -> Any`
  - `async resume(run_id, **kwargs) -> Any`

- [ ] **Step 1: Write failing tests**

```python
# tests/adapters/test_langgraph.py
import pickle
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from agentguard.adapters.langgraph import DurableGraph
from agentguard.stores.memory import InMemoryStore
from agentguard.core.exceptions import RestoreError
from agentguard.core.types import CheckpointMeta


def make_mock_graph(return_value=None):
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=return_value or {"output": "done"})
    return graph


@pytest.mark.asyncio
async def test_ainvoke_checkpoints_state():
    store = InMemoryStore()
    graph = make_mock_graph({"output": "result"})
    dg = DurableGraph(graph=graph, store=store)
    await dg.ainvoke({"task": "test"}, run_id="run-lg1")
    metas = await store.list("run-lg1")
    assert len(metas) >= 1
    assert metas[0].framework == "langgraph"


@pytest.mark.asyncio
async def test_ainvoke_returns_graph_result():
    store = InMemoryStore()
    graph = make_mock_graph({"output": "my result"})
    dg = DurableGraph(graph=graph, store=store)
    result = await dg.ainvoke({"task": "test"}, run_id="run-lg2")
    assert result == {"output": "my result"}


@pytest.mark.asyncio
async def test_resume_restores_and_reinvokes():
    store = InMemoryStore()
    state = pickle.dumps({"input": {"task": "resumed task"}, "step": 3})
    meta = CheckpointMeta(
        run_id="run-lg3", step=3, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework="langgraph",
    )
    await store.save("run-lg3", 3, state, meta)

    graph = make_mock_graph({"output": "resumed"})
    dg = DurableGraph(graph=graph, store=store)
    result = await dg.resume("run-lg3")
    assert result == {"output": "resumed"}
    graph.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_resume_raises_when_no_checkpoint():
    store = InMemoryStore()
    dg = DurableGraph(graph=MagicMock(), store=store)
    with pytest.raises(RestoreError):
        await dg.resume("missing-lg-run")
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/adapters/test_langgraph.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `agentguard/adapters/langgraph.py`**

```python
import pickle
from collections.abc import Callable
from datetime import datetime, timezone
from agentguard.core.engine import CheckpointEngine
from agentguard.core.exceptions import DeserializationError
from agentguard.core.store import StorageBackend
from agentguard.core.triggers import TriggerMeta, TriggerPolicy
from agentguard.core.types import CheckpointMeta


class DurableGraph:
    def __init__(
        self,
        graph,
        store: StorageBackend,
        policy: TriggerPolicy | None = None,
        framework_name: str = "langgraph",
        on_checkpoint_failure: Callable[[str, int, Exception], None] | None = None,
    ) -> None:
        self._graph = graph
        self._engine = CheckpointEngine(
            store=store,
            policy=policy or TriggerPolicy(),
            on_checkpoint_failure=on_checkpoint_failure,
        )
        self._framework = framework_name

    def _serialize(self, input_dict: dict, step: int) -> bytes:
        return pickle.dumps({"input": input_dict, "step": step})

    def _deserialize(self, state: bytes) -> dict:
        try:
            return pickle.loads(state)  # noqa: S301
        except Exception as exc:
            raise DeserializationError(f"Failed to deserialize LangGraph state: {exc}") from exc

    async def ainvoke(self, input_dict: dict, run_id: str, **kwargs):
        step = 0
        trigger_meta = TriggerMeta(tool_name=None, step=step)
        _, trigger_reason = self._engine.policy.should_checkpoint(trigger_meta)

        state_bytes = self._serialize(input_dict, step)
        meta = CheckpointMeta(
            run_id=run_id, step=step, trigger=trigger_reason,
            timestamp=datetime.now(tz=timezone.utc),
            framework=self._framework,
        )
        await self._engine.checkpoint(run_id, step, state_bytes, meta)

        result = await self._graph.ainvoke(input_dict, **kwargs)

        step += 1
        final_state_bytes = self._serialize({"result": result}, step)
        final_meta = CheckpointMeta(
            run_id=run_id, step=step, trigger="completed",
            timestamp=datetime.now(tz=timezone.utc),
            framework=self._framework,
        )
        await self._engine.checkpoint(run_id, step, final_state_bytes, final_meta)
        return result

    async def resume(self, run_id: str, **kwargs):
        restored = await self._engine.restore(run_id)
        state = self._deserialize(restored.state)
        input_dict = state.get("input", state)
        return await self.ainvoke(input_dict, run_id=run_id, **kwargs)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/adapters/test_langgraph.py -v
```

Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add agentguard/adapters/langgraph.py tests/adapters/test_langgraph.py
git commit -m "feat: add DurableGraph adapter for LangGraph"
```

---

## Task 11: LangChain Adapter

**Files:**
- Create: `agentguard/adapters/langchain.py`
- Create: `tests/adapters/test_langchain.py`

**Interfaces:**
- Consumes: `CheckpointEngine`, `TriggerPolicy`, `TriggerMeta`, `InMemoryStore`, `CheckpointMeta`, `DeserializationError`
- Produces: `DurableExecutor(executor, store, policy, framework_name="langchain")` with:
  - `async ainvoke(input_dict, run_id, **kwargs) -> Any`
  - `async resume(run_id, **kwargs) -> Any`

- [ ] **Step 1: Write failing tests**

```python
# tests/adapters/test_langchain.py
import pickle
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
from agentguard.adapters.langchain import DurableExecutor
from agentguard.stores.memory import InMemoryStore
from agentguard.core.exceptions import RestoreError
from agentguard.core.types import CheckpointMeta


def make_mock_executor(return_value=None):
    ex = MagicMock()
    ex.ainvoke = AsyncMock(return_value=return_value or {"output": "done"})
    return ex


@pytest.mark.asyncio
async def test_ainvoke_checkpoints_state():
    store = InMemoryStore()
    ex = make_mock_executor({"output": "lc result"})
    de = DurableExecutor(executor=ex, store=store)
    await de.ainvoke({"input": "test"}, run_id="run-lc1")
    metas = await store.list("run-lc1")
    assert len(metas) >= 1
    assert metas[0].framework == "langchain"


@pytest.mark.asyncio
async def test_ainvoke_returns_executor_result():
    store = InMemoryStore()
    ex = make_mock_executor({"output": "lc output"})
    de = DurableExecutor(executor=ex, store=store)
    result = await de.ainvoke({"input": "test"}, run_id="run-lc2")
    assert result == {"output": "lc output"}


@pytest.mark.asyncio
async def test_resume_restores_and_reinvokes():
    store = InMemoryStore()
    state = pickle.dumps({"input": {"input": "resumed"}, "step": 2})
    meta = CheckpointMeta(
        run_id="run-lc3", step=2, trigger="tool_call",
        timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
        framework="langchain",
    )
    await store.save("run-lc3", 2, state, meta)
    ex = make_mock_executor({"output": "resumed lc"})
    de = DurableExecutor(executor=ex, store=store)
    result = await de.resume("run-lc3")
    assert result == {"output": "resumed lc"}


@pytest.mark.asyncio
async def test_resume_raises_when_no_checkpoint():
    store = InMemoryStore()
    de = DurableExecutor(executor=MagicMock(), store=store)
    with pytest.raises(RestoreError):
        await de.resume("missing-lc-run")
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/adapters/test_langchain.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `agentguard/adapters/langchain.py`**

```python
import pickle
from collections.abc import Callable
from datetime import datetime, timezone
from agentguard.core.engine import CheckpointEngine
from agentguard.core.exceptions import DeserializationError
from agentguard.core.store import StorageBackend
from agentguard.core.triggers import TriggerMeta, TriggerPolicy
from agentguard.core.types import CheckpointMeta


class DurableExecutor:
    def __init__(
        self,
        executor,
        store: StorageBackend,
        policy: TriggerPolicy | None = None,
        framework_name: str = "langchain",
        on_checkpoint_failure: Callable[[str, int, Exception], None] | None = None,
    ) -> None:
        self._executor = executor
        self._engine = CheckpointEngine(
            store=store,
            policy=policy or TriggerPolicy(),
            on_checkpoint_failure=on_checkpoint_failure,
        )
        self._framework = framework_name

    def _serialize(self, input_dict: dict, step: int) -> bytes:
        return pickle.dumps({"input": input_dict, "step": step})

    def _deserialize(self, state: bytes) -> dict:
        try:
            return pickle.loads(state)  # noqa: S301
        except Exception as exc:
            raise DeserializationError(f"Failed to deserialize LangChain state: {exc}") from exc

    async def ainvoke(self, input_dict: dict, run_id: str, **kwargs):
        step = 0
        trigger_meta = TriggerMeta(tool_name=None, step=step)
        _, trigger_reason = self._engine.policy.should_checkpoint(trigger_meta)

        state_bytes = self._serialize(input_dict, step)
        meta = CheckpointMeta(
            run_id=run_id, step=step, trigger=trigger_reason,
            timestamp=datetime.now(tz=timezone.utc),
            framework=self._framework,
        )
        await self._engine.checkpoint(run_id, step, state_bytes, meta)

        result = await self._executor.ainvoke(input_dict, **kwargs)

        step += 1
        final_state_bytes = self._serialize({"result": result}, step)
        final_meta = CheckpointMeta(
            run_id=run_id, step=step, trigger="completed",
            timestamp=datetime.now(tz=timezone.utc),
            framework=self._framework,
        )
        await self._engine.checkpoint(run_id, step, final_state_bytes, final_meta)
        return result

    async def resume(self, run_id: str, **kwargs):
        restored = await self._engine.restore(run_id)
        state = self._deserialize(restored.state)
        input_dict = state.get("input", state)
        return await self.ainvoke(input_dict, run_id=run_id, **kwargs)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/adapters/test_langchain.py -v
```

Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add agentguard/adapters/langchain.py tests/adapters/test_langchain.py
git commit -m "feat: add DurableExecutor adapter for LangChain"
```

---

## Task 12: CLI

**Files:**
- Create: `agentguard/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `DiskStore`, `CheckpointMeta`, `RestoredState`
- Produces: `agentguard` CLI entry point with commands: `list`, `inspect`, `resume`, `prune`, `delete`, `stats`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli.py
import pytest
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typer.testing import CliRunner
from agentguard.cli import app
from agentguard.stores.disk import DiskStore
from agentguard.core.types import CheckpointMeta

runner = CliRunner()


async def seed_store(base_dir: Path, run_id: str, steps: int):
    store = DiskStore(base_dir=base_dir)
    for step in range(steps):
        meta = CheckpointMeta(
            run_id=run_id, step=step, trigger="tool_call",
            timestamp=datetime(2026, 8, 5, tzinfo=timezone.utc),
            framework="anthropic",
        )
        await store.save(run_id, step, f"state-{step}".encode(), meta)


def test_list_command(tmp_path):
    asyncio.run(seed_store(tmp_path, "run-cli1", 3))
    result = runner.invoke(app, ["list", "run-cli1", "--store", str(tmp_path)])
    assert result.exit_code == 0
    assert "run-cli1" in result.output
    assert "tool_call" in result.output


def test_list_empty_run(tmp_path):
    result = runner.invoke(app, ["list", "nonexistent-run", "--store", str(tmp_path)])
    assert result.exit_code == 0
    assert "No checkpoints" in result.output


def test_inspect_command(tmp_path):
    asyncio.run(seed_store(tmp_path, "run-cli2", 3))
    result = runner.invoke(app, ["inspect", "run-cli2", "--step", "1", "--store", str(tmp_path)])
    assert result.exit_code == 0
    assert "step" in result.output.lower()


def test_resume_command_prints_snippet(tmp_path):
    asyncio.run(seed_store(tmp_path, "run-cli3", 2))
    result = runner.invoke(app, ["resume", "run-cli3", "--store", str(tmp_path)])
    assert result.exit_code == 0
    assert "DurableAgentLoop" in result.output or "resume" in result.output.lower()


def test_prune_command(tmp_path):
    asyncio.run(seed_store(tmp_path, "run-cli4", 5))
    result = runner.invoke(app, ["prune", "run-cli4", "--keep", "2", "--store", str(tmp_path)])
    assert result.exit_code == 0
    store = DiskStore(base_dir=tmp_path)
    metas = asyncio.run(store.list("run-cli4"))
    assert len(metas) == 2


def test_delete_command(tmp_path):
    asyncio.run(seed_store(tmp_path, "run-cli5", 2))
    result = runner.invoke(app, ["delete", "run-cli5", "--store", str(tmp_path)])
    assert result.exit_code == 0
    store = DiskStore(base_dir=tmp_path)
    metas = asyncio.run(store.list("run-cli5"))
    assert metas == []


def test_stats_command(tmp_path):
    asyncio.run(seed_store(tmp_path, "run-cli6", 4))
    result = runner.invoke(app, ["stats", "--store", str(tmp_path)])
    assert result.exit_code == 0
    assert "total" in result.output.lower()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_cli.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `agentguard/cli.py`**

```python
import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

from agentguard.stores.disk import DiskStore

app = typer.Typer(help="AgentGuard — inspect and manage agent checkpoints")

_RESUME_SNIPPETS = {
    "anthropic": (
        "from agentguard.adapters.anthropic import DurableAgentLoop\n"
        'loop = DurableAgentLoop(client=client, store=store)\n'
        'result = await loop.resume("{run_id}", model=model, tools=tools)'
    ),
    "langgraph": (
        "from agentguard.adapters.langgraph import DurableGraph\n"
        'app = DurableGraph(graph=graph, store=store)\n'
        'result = await app.resume("{run_id}")'
    ),
    "langchain": (
        "from agentguard.adapters.langchain import DurableExecutor\n"
        'chain = DurableExecutor(executor=executor, store=store)\n'
        'result = await chain.resume("{run_id}")'
    ),
}


def _get_store(store_path: str) -> DiskStore:
    return DiskStore(base_dir=Path(store_path))


@app.command()
def list(
    run_id: str,
    store: str = typer.Option(".agentguard", help="Path to disk store"),
):
    """List all checkpoints for a run."""
    s = _get_store(store)
    metas = asyncio.run(s.list(run_id))
    if not metas:
        typer.echo(f"No checkpoints found for run_id={run_id}")
        return
    typer.echo(f"{'step':<8} {'trigger':<22} {'timestamp':<30} {'tokens':<8} {'cost'}")
    for m in metas:
        typer.echo(
            f"{m.step:<8} {m.trigger:<22} {m.timestamp.isoformat():<30} "
            f"{m.token_count:<8} ${m.cost_usd:.4f}"
        )


@app.command()
def inspect(
    run_id: str,
    step: int = typer.Option(..., help="Step number to inspect"),
    store: str = typer.Option(".agentguard", help="Path to disk store"),
):
    """Inspect metadata for a specific checkpoint."""
    s = _get_store(store)
    metas = asyncio.run(s.list(run_id))
    match = next((m for m in metas if m.step == step), None)
    if match is None:
        typer.echo(f"No checkpoint at step={step} for run_id={run_id}")
        raise typer.Exit(code=1)
    typer.echo(json.dumps({
        "run_id": match.run_id, "step": match.step, "trigger": match.trigger,
        "timestamp": match.timestamp.isoformat(), "framework": match.framework,
        "token_count": match.token_count, "cost_usd": match.cost_usd,
    }, indent=2))


@app.command()
def resume(
    run_id: str,
    store: str = typer.Option(".agentguard", help="Path to disk store"),
):
    """Print resume code snippet for a run (auto-detects framework)."""
    s = _get_store(store)
    metas = asyncio.run(s.list(run_id))
    if not metas:
        typer.echo(f"No checkpoints found for run_id={run_id}")
        raise typer.Exit(code=1)
    latest = metas[0]
    snippet = _RESUME_SNIPPETS.get(
        latest.framework,
        f"# Unknown framework: {latest.framework}\nawait adapter.resume('{run_id}')"
    ).format(run_id=run_id)
    typer.echo(f"Detected framework: {latest.framework}")
    typer.echo(f"Latest checkpoint: step {latest.step} ({latest.timestamp.isoformat()})")
    typer.echo("\nResume with:\n")
    typer.echo(snippet)


@app.command()
def prune(
    run_id: str,
    keep: int = typer.Option(3, help="Number of checkpoints to keep"),
    store: str = typer.Option(".agentguard", help="Path to disk store"),
):
    """Prune old checkpoints, keeping the last N."""
    s = _get_store(store)
    asyncio.run(s.prune(run_id, keep_last=keep))
    typer.echo(f"Pruned checkpoints for {run_id}, kept last {keep}.")


@app.command()
def delete(
    run_id: str,
    store: str = typer.Option(".agentguard", help="Path to disk store"),
):
    """Delete all checkpoints for a run."""
    s = _get_store(store)
    asyncio.run(s.delete(run_id))
    typer.echo(f"Deleted all checkpoints for run_id={run_id}")


@app.command()
def stats(
    store: str = typer.Option(".agentguard", help="Path to disk store"),
):
    """Show aggregate stats across all runs."""
    base = Path(store)
    if not base.exists():
        typer.echo("No checkpoint store found.")
        return
    total_runs = 0
    total_checkpoints = 0
    total_cost = 0.0
    triggers: dict[str, int] = {}
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        total_runs += 1
        s = _get_store(store)
        metas = asyncio.run(s.list(run_dir.name))
        total_checkpoints += len(metas)
        for m in metas:
            total_cost += m.cost_usd
            triggers[m.trigger] = triggers.get(m.trigger, 0) + 1
    typer.echo(f"Total runs:        {total_runs}")
    typer.echo(f"Total checkpoints: {total_checkpoints}")
    typer.echo(f"Total cost:        ${total_cost:.4f}")
    if triggers:
        most_common = max(triggers, key=lambda k: triggers[k])
        typer.echo(f"Most common trigger: {most_common} ({triggers[most_common]})")
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_cli.py -v
```

Expected: all 8 PASS

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/stores/test_redis.py --ignore=tests/stores/test_postgres.py
```

Expected: all PASS (Redis/Postgres require Docker — run separately)

- [ ] **Step 6: Commit**

```bash
git add agentguard/cli.py tests/test_cli.py
git commit -m "feat: add agentguard CLI with list/inspect/resume/prune/delete/stats"
```

---

## Task 13: GitHub Actions CI

**Files:**
- Create: `.github/workflows/test.yml`

**Interfaces:**
- Produces: CI that runs on push/PR, covers all Python versions and backends

- [ ] **Step 1: Create workflow**

```yaml
# .github/workflows/test.yml
name: test

on:
  push:
    branches: [main]
  pull_request:

jobs:
  core:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/core/ tests/stores/test_memory.py tests/stores/test_disk.py tests/adapters/ tests/test_cli.py -v

  redis:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/stores/test_redis.py -v

  postgres:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/stores/test_postgres.py -v
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "ci: add GitHub Actions matrix for core, redis, postgres backends"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `CheckpointMeta`, `RestoredState` frozen dataclasses | Task 2 |
| `AgentGuardError` hierarchy | Task 2 |
| `StorageBackend` ABC | Task 4 |
| `TriggerPolicy` with wildcard + token pressure | Task 3 |
| `CheckpointEngine` fail-safe write + fail-loud restore | Task 5 |
| `InMemoryStore` for testing | Task 4 |
| `DiskStore` — meta.jsonl + step bins | Task 6 |
| `RedisStore` — sorted set + TTL | Task 7 |
| `PostgresStore` — BYTEA + compound PK | Task 8 |
| `DurableAgentLoop` (Anthropic) | Task 9 |
| `DurableGraph` (LangGraph) | Task 10 |
| `DurableExecutor` (LangChain) | Task 11 |
| CLI: list/inspect/resume/prune/delete/stats | Task 12 |
| Optional install extras | Task 1 |
| CI matrix python 3.11/3.12 × backends | Task 13 |

All spec requirements covered. No gaps.
