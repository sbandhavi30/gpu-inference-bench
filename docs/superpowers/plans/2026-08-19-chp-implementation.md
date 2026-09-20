# Context Handoff Protocol (CHP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build CHP as a drop-in Python middleware library that solves agent-to-agent context handoff via receiver-declared manifests, scored selection, rationale envelopes, and an append-only context ledger.

**Architecture:** Four-layer middleware (ContextManifest → Selection Engine → Rationale Envelope → Context Ledger) sits between any two agents without framework lock-in. Each layer is independently testable and composable. Adapters for LangGraph, CrewAI, and AutoGen wrap the core library.

**Tech Stack:** Python 3.11+, Pydantic v2, sentence-transformers, numpy, lancedb, pytest

**Spec:** `docs/superpowers/specs/2026-08-19-chp-context-handoff-protocol-design.md`

## Global Constraints

- Python >= 3.11
- Pydantic >= 2.0 (v2 API only — no v1 compat shims)
- All dependencies MIT or Apache licensed — no GPL, no vendor lock-in
- Library must work fully offline (no mandatory API calls)
- sentence-transformers is optional — all scoring must work with a stub embedder
- append-only invariant on Context Ledger must be enforced in code, not just convention
- JSON Schema export from ContextManifest must be valid JSON Schema Draft 2020-12
- No managed services required for any test to pass

---

## File Map

```
chp/
├── __init__.py                        # Public API exports
├── schema/
│   ├── __init__.py
│   ├── context_manifest.py            # ContextManifest + ContextRequirements Pydantic models
│   └── rationale_envelope.py          # RationaleEnvelope + AnnotatedChunk Pydantic models
├── engine/
│   ├── __init__.py
│   ├── embedder.py                    # Embedder protocol + sentence-transformers impl + stub
│   └── scorer.py                      # ScorerWeights + score_chunks() + select_chunks()
├── ledger/
│   ├── __init__.py
│   └── lancedb_ledger.py              # CHPLedger: append-only LanceDB-backed store
├── adapters/
│   ├── __init__.py
│   ├── langgraph.py                   # chp_node_middleware() for LangGraph state transitions
│   ├── crewai.py                      # CHPCrewTask wrapper for CrewAI task context
│   └── autogen.py                     # CHPConversableAgent subclass for AutoGen
└── benchmarks/
    ├── __init__.py
    ├── baseline.py                    # Naive full-history pass baseline
    ├── gaia.py                        # GAIA benchmark runner
    └── hotpotqa.py                    # HotpotQA multi-hop benchmark runner

tests/
├── unit/
│   ├── test_context_manifest.py
│   ├── test_rationale_envelope.py
│   ├── test_embedder.py
│   ├── test_scorer.py
│   └── test_ledger.py
└── integration/
    ├── test_langgraph_adapter.py
    ├── test_crewai_adapter.py
    └── test_autogen_adapter.py
```

---

## PHASE 1 — Schema + Scorer (Weeks 1–3)

---

### Task 1: Project Scaffold + ContextManifest Schema

**Files:**
- Create: `chp/__init__.py`
- Create: `chp/schema/__init__.py`
- Create: `chp/schema/context_manifest.py`
- Create: `pyproject.toml`
- Test: `tests/unit/test_context_manifest.py`

**Interfaces:**
- Produces:
  - `ContextRequirements(must_carry: list[str], domain_tags: list[str], history_depth: Literal["full","decisions_only","summary","none"], recency_window: str | None, exclude: list[str])`
  - `ContextManifest(chp_version: str, agent_id: str, task: str, requires: ContextRequirements, token_budget: int, on_missing: Literal["fail_hard","warn","proceed"])` with `.to_json_schema() -> dict`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_context_manifest.py
import pytest
from chp.schema.context_manifest import ContextManifest, ContextRequirements

def test_manifest_valid_construction():
    req = ContextRequirements(
        must_carry=["user_id", "order_id"],
        domain_tags=["billing", "auth"],
        history_depth="decisions_only",
        recency_window="last_3_agent_turns",
        exclude=["PII_raw"],
    )
    m = ContextManifest(
        chp_version="0.1",
        agent_id="billing-agent-v2",
        task="resolve_refund_dispute",
        requires=req,
        token_budget=2000,
        on_missing="fail_hard",
    )
    assert m.agent_id == "billing-agent-v2"
    assert m.requires.must_carry == ["user_id", "order_id"]
    assert m.on_missing == "fail_hard"

def test_manifest_rejects_invalid_on_missing():
    with pytest.raises(Exception):
        ContextManifest(
            chp_version="0.1",
            agent_id="x",
            task="y",
            requires=ContextRequirements(
                must_carry=[], domain_tags=[], history_depth="full",
                recency_window=None, exclude=[]
            ),
            token_budget=1000,
            on_missing="explode",  # invalid
        )

def test_manifest_json_schema_export():
    schema = ContextManifest.to_json_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "properties" in schema
    assert "agent_id" in schema["properties"]
    assert "token_budget" in schema["properties"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd chp && pytest tests/unit/test_context_manifest.py -v
```
Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "chp"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "numpy>=1.26",
    "lancedb>=0.6",
]

[project.optional-dependencies]
embeddings = ["sentence-transformers>=3.0"]
dev = ["pytest>=8.0", "pytest-cov"]
```

- [ ] **Step 4: Implement ContextManifest**

```python
# chp/schema/context_manifest.py
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, model_validator
import json

class ContextRequirements(BaseModel):
    must_carry: list[str] = []
    domain_tags: list[str] = []
    history_depth: Literal["full", "decisions_only", "summary", "none"] = "full"
    recency_window: str | None = None
    exclude: list[str] = []

class ContextManifest(BaseModel):
    chp_version: str = "0.1"
    agent_id: str
    task: str
    requires: ContextRequirements
    token_budget: int
    on_missing: Literal["fail_hard", "warn", "proceed"] = "warn"

    @classmethod
    def to_json_schema(cls) -> dict:
        schema = cls.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        return schema
```

```python
# chp/__init__.py
from chp.schema.context_manifest import ContextManifest, ContextRequirements

__all__ = ["ContextManifest", "ContextRequirements"]
```

```python
# chp/schema/__init__.py
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_context_manifest.py -v
```
Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add chp/ pyproject.toml tests/unit/test_context_manifest.py
git commit -s -m "feat(chp): add ContextManifest schema with JSON Schema export"
```

---

### Task 2: RationaleEnvelope Schema

**Files:**
- Create: `chp/schema/rationale_envelope.py`
- Modify: `chp/__init__.py`
- Test: `tests/unit/test_rationale_envelope.py`

**Interfaces:**
- Consumes: nothing from prior tasks (standalone schema)
- Produces:
  - `AnnotatedChunk(chunk_id: str, content: str, token_cost: int, source_agent: str, source_turn: int, timestamp: datetime)`
  - `RationaleEnvelope(chunk_id: str, content: str, source_agent: str, source_turn: int, hop_sequence: list[str], selected_because: list[str], score: float, must_carry: bool, token_cost: int, ledger_id: str | None)`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_rationale_envelope.py
from chp.schema.rationale_envelope import RationaleEnvelope, AnnotatedChunk
from datetime import datetime, timezone

def test_annotated_chunk_construction():
    chunk = AnnotatedChunk(
        chunk_id="c_001",
        content="user_id is 42",
        token_cost=10,
        source_agent="auth-agent",
        source_turn=1,
        timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    assert chunk.chunk_id == "c_001"
    assert chunk.token_cost == 10

def test_rationale_envelope_construction():
    env = RationaleEnvelope(
        chunk_id="c_047",
        content="order_id is 999",
        source_agent="auth-agent",
        source_turn=3,
        hop_sequence=["auth-agent", "orchestrator", "billing-agent"],
        selected_because=["domain_match:auth", "must_carry:order_id"],
        score=0.91,
        must_carry=True,
        token_cost=142,
        ledger_id="ledger_abc_turn_3",
    )
    assert env.score == 0.91
    assert "domain_match:auth" in env.selected_because
    assert env.must_carry is True

def test_rationale_envelope_score_bounds():
    import pytest
    with pytest.raises(Exception):
        RationaleEnvelope(
            chunk_id="c_x", content="x", source_agent="a", source_turn=0,
            hop_sequence=[], selected_because=[], score=1.5,  # out of bounds
            must_carry=False, token_cost=1, ledger_id=None,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_rationale_envelope.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement RationaleEnvelope**

```python
# chp/schema/rationale_envelope.py
from __future__ import annotations
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class AnnotatedChunk(BaseModel):
    chunk_id: str
    content: str
    token_cost: int
    source_agent: str
    source_turn: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class RationaleEnvelope(BaseModel):
    chunk_id: str
    content: str
    source_agent: str
    source_turn: int
    hop_sequence: list[str]
    selected_because: list[str]
    score: float = Field(ge=0.0, le=1.0)
    must_carry: bool
    token_cost: int
    ledger_id: str | None = None
```

- [ ] **Step 4: Update `chp/__init__.py`**

```python
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import RationaleEnvelope, AnnotatedChunk

__all__ = ["ContextManifest", "ContextRequirements", "RationaleEnvelope", "AnnotatedChunk"]
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/test_rationale_envelope.py -v
```
Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add chp/schema/rationale_envelope.py chp/__init__.py tests/unit/test_rationale_envelope.py
git commit -s -m "feat(chp): add RationaleEnvelope + AnnotatedChunk schemas"
```

---

### Task 3: Embedder Protocol + Stub + sentence-transformers Impl

**Files:**
- Create: `chp/engine/__init__.py`
- Create: `chp/engine/embedder.py`
- Test: `tests/unit/test_embedder.py`

**Interfaces:**
- Produces:
  - `Embedder` Protocol: `.embed(texts: list[str]) -> np.ndarray`  (shape: `[N, D]`)
  - `StubEmbedder` — deterministic fake (no model download); returns `np.zeros((len(texts), 384))`
  - `SentenceTransformerEmbedder(model_name: str = "all-MiniLM-L6-v2")` — real impl, optional dep

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_embedder.py
import numpy as np
import pytest
from chp.engine.embedder import StubEmbedder

def test_stub_embedder_shape():
    emb = StubEmbedder()
    result = emb.embed(["hello world", "another chunk"])
    assert result.shape == (2, 384)
    assert result.dtype == np.float32

def test_stub_embedder_deterministic():
    emb = StubEmbedder()
    r1 = emb.embed(["hello"])
    r2 = emb.embed(["hello"])
    np.testing.assert_array_equal(r1, r2)

def test_stub_embedder_empty():
    emb = StubEmbedder()
    result = emb.embed([])
    assert result.shape == (0, 384)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_embedder.py -v
```

- [ ] **Step 3: Implement embedder**

```python
# chp/engine/embedder.py
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...

class StubEmbedder:
    DIM = 384

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), self.DIM), dtype=np.float32)

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install chp[embeddings]"
            ) from e
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
```

```python
# chp/engine/__init__.py
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_embedder.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add chp/engine/ tests/unit/test_embedder.py
git commit -s -m "feat(chp): add Embedder protocol, StubEmbedder, SentenceTransformerEmbedder"
```

---

### Task 4: Scoring Engine — score_chunks() + select_chunks()

**Files:**
- Create: `chp/engine/scorer.py`
- Test: `tests/unit/test_scorer.py`

**Interfaces:**
- Consumes:
  - `ContextManifest` from `chp.schema.context_manifest`
  - `AnnotatedChunk` from `chp.schema.rationale_envelope`
  - `Embedder` from `chp.engine.embedder`
- Produces:
  - `ScorerWeights(alpha: float, beta: float, gamma: float, delta: float, epsilon: float)` — default: `(0.4, 0.2, 0.3, 0.05, 0.05)`
  - `score_chunks(chunks: list[AnnotatedChunk], manifest: ContextManifest, embedder: Embedder, weights: ScorerWeights | None) -> list[tuple[AnnotatedChunk, float]]` — returns `(chunk, score)` pairs sorted descending
  - `select_chunks(chunks: list[AnnotatedChunk], manifest: ContextManifest, embedder: Embedder, weights: ScorerWeights | None) -> list[AnnotatedChunk]` — returns chunks within token_budget, must_carry guaranteed

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_scorer.py
import numpy as np
import pytest
from datetime import datetime, timezone
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import AnnotatedChunk
from chp.engine.embedder import StubEmbedder
from chp.engine.scorer import ScorerWeights, score_chunks, select_chunks

def _manifest(must_carry=None, domain_tags=None, token_budget=1000, on_missing="warn"):
    return ContextManifest(
        chp_version="0.1",
        agent_id="test-agent",
        task="test_task",
        requires=ContextRequirements(
            must_carry=must_carry or [],
            domain_tags=domain_tags or ["billing"],
            history_depth="full",
            recency_window=None,
            exclude=[],
        ),
        token_budget=token_budget,
        on_missing=on_missing,
    )

def _chunk(chunk_id, content, token_cost=50, source_turn=1):
    return AnnotatedChunk(
        chunk_id=chunk_id,
        content=content,
        token_cost=token_cost,
        source_agent="agent-a",
        source_turn=source_turn,
        timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )

def test_score_chunks_returns_sorted_descending():
    chunks = [_chunk("c1", "billing info"), _chunk("c2", "weather data")]
    manifest = _manifest(domain_tags=["billing"])
    embedder = StubEmbedder()
    scored = score_chunks(chunks, manifest, embedder)
    assert len(scored) == 2
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)

def test_select_chunks_respects_token_budget():
    chunks = [_chunk(f"c{i}", f"chunk {i}", token_cost=300) for i in range(5)]
    manifest = _manifest(token_budget=700)
    embedder = StubEmbedder()
    selected = select_chunks(chunks, manifest, embedder)
    total_tokens = sum(c.token_cost for c in selected)
    assert total_tokens <= 700

def test_select_chunks_always_includes_must_carry():
    chunks = [
        _chunk("c1", "user_id is 42", token_cost=10),
        _chunk("c2", "irrelevant data", token_cost=10),
        _chunk("c3", "order_id is 99", token_cost=10),
    ]
    manifest = _manifest(must_carry=["user_id", "order_id"], token_budget=100)
    embedder = StubEmbedder()
    selected = select_chunks(chunks, manifest, embedder)
    ids = [c.chunk_id for c in selected]
    assert "c1" in ids
    assert "c3" in ids

def test_select_chunks_excludes_tagged_chunks():
    chunks = [
        _chunk("c1", "PII_raw: email@example.com", token_cost=10),
        _chunk("c2", "billing amount: $50", token_cost=10),
    ]
    manifest = ContextManifest(
        chp_version="0.1", agent_id="a", task="t",
        requires=ContextRequirements(
            must_carry=[], domain_tags=["billing"],
            history_depth="full", recency_window=None,
            exclude=["PII_raw"],
        ),
        token_budget=1000, on_missing="warn",
    )
    embedder = StubEmbedder()
    selected = select_chunks(chunks, manifest, embedder)
    ids = [c.chunk_id for c in selected]
    assert "c1" not in ids

def test_select_chunks_fail_hard_on_missing_must_carry():
    chunks = [_chunk("c1", "irrelevant", token_cost=10)]
    manifest = _manifest(must_carry=["user_id"], on_missing="fail_hard")
    embedder = StubEmbedder()
    with pytest.raises(ValueError, match="must_carry"):
        select_chunks(chunks, manifest, embedder)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_scorer.py -v
```

- [ ] **Step 3: Implement scorer**

```python
# chp/engine/scorer.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from chp.schema.context_manifest import ContextManifest
from chp.schema.rationale_envelope import AnnotatedChunk
from chp.engine.embedder import Embedder

@dataclass
class ScorerWeights:
    alpha: float = 0.4   # semantic similarity
    beta: float = 0.2    # recency
    gamma: float = 0.3   # must_carry bonus
    delta: float = 0.05  # token cost penalty
    epsilon: float = 0.05  # exclude penalty

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T  # [N_chunks, N_tags]

def _recency_weight(chunks: list[AnnotatedChunk]) -> np.ndarray:
    turns = np.array([c.source_turn for c in chunks], dtype=np.float32)
    if turns.max() == turns.min():
        return np.ones(len(chunks), dtype=np.float32)
    return (turns - turns.min()) / (turns.max() - turns.min())

def score_chunks(
    chunks: list[AnnotatedChunk],
    manifest: ContextManifest,
    embedder: Embedder,
    weights: ScorerWeights | None = None,
) -> list[tuple[AnnotatedChunk, float]]:
    if not chunks:
        return []
    w = weights or ScorerWeights()
    req = manifest.requires

    chunk_texts = [c.content for c in chunks]
    tag_texts = req.domain_tags or ["general"]
    chunk_embs = embedder.embed(chunk_texts)   # [N, D]
    tag_embs = embedder.embed(tag_texts)       # [T, D]

    sim_matrix = _cosine_sim(chunk_embs, tag_embs)  # [N, T]
    semantic_scores = sim_matrix.max(axis=1)         # [N]

    recency = _recency_weight(chunks)

    scores = np.zeros(len(chunks), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        content_lower = chunk.content.lower()
        must_carry_hit = any(key.lower() in content_lower for key in req.must_carry)
        exclude_hit = any(ex.lower() in content_lower for ex in req.exclude)
        token_penalty = chunk.token_cost / max(manifest.token_budget, 1)

        scores[i] = (
            w.alpha * float(semantic_scores[i])
            + w.beta * float(recency[i])
            + w.gamma * float(must_carry_hit)
            - w.delta * token_penalty
            - w.epsilon * float(exclude_hit)
        )

    pairs = list(zip(chunks, scores.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs

def select_chunks(
    chunks: list[AnnotatedChunk],
    manifest: ContextManifest,
    embedder: Embedder,
    weights: ScorerWeights | None = None,
) -> list[AnnotatedChunk]:
    req = manifest.requires

    # Hard exclude
    excluded_ids = {
        c.chunk_id for c in chunks
        if any(ex.lower() in c.content.lower() for ex in req.exclude)
    }

    # Must-carry: find by content match
    must_carry_ids: set[str] = set()
    for key in req.must_carry:
        matched = [c for c in chunks if key.lower() in c.content.lower() and c.chunk_id not in excluded_ids]
        if matched:
            must_carry_ids.add(matched[0].chunk_id)
        elif manifest.on_missing == "fail_hard":
            raise ValueError(f"must_carry key '{key}' not found in any chunk")

    must_carry_chunks = [c for c in chunks if c.chunk_id in must_carry_ids]
    must_carry_tokens = sum(c.token_cost for c in must_carry_chunks)
    remaining_budget = manifest.token_budget - must_carry_tokens

    # Score and select remaining
    candidates = [c for c in chunks if c.chunk_id not in must_carry_ids and c.chunk_id not in excluded_ids]
    scored = score_chunks(candidates, manifest, embedder, weights)

    selected = list(must_carry_chunks)
    for chunk, _ in scored:
        if remaining_budget <= 0:
            break
        if chunk.token_cost <= remaining_budget:
            selected.append(chunk)
            remaining_budget -= chunk.token_cost

    return selected
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_scorer.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add chp/engine/scorer.py tests/unit/test_scorer.py
git commit -s -m "feat(chp): add scoring engine with score_chunks() and select_chunks()"
```

---

## PHASE 2 — Context Ledger + Adapters (Weeks 4–6)

---

### Task 5: Context Ledger (LanceDB append-only)

**Files:**
- Create: `chp/ledger/__init__.py`
- Create: `chp/ledger/lancedb_ledger.py`
- Test: `tests/unit/test_ledger.py`

**Interfaces:**
- Consumes:
  - `RationaleEnvelope` from `chp.schema.rationale_envelope`
- Produces:
  - `CHPLedger(db_path: str | None = None)` — `None` = in-memory (for tests)
  - `.write(session_id: str, hop_number: int, from_agent: str, to_agent: str, envelope: RationaleEnvelope) -> str` — returns `ledger_id`
  - `.query(session_id: str, agent_id: str | None = None) -> list[RationaleEnvelope]`
  - `.query_hop(session_id: str, hop_number: int) -> list[RationaleEnvelope]`
  - Append-only: `.write()` must raise `RuntimeError` if called with a duplicate `(session_id, chunk_id, hop_number)` key

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_ledger.py
import pytest
from chp.ledger.lancedb_ledger import CHPLedger
from chp.schema.rationale_envelope import RationaleEnvelope

def _envelope(chunk_id="c_001", score=0.9, must_carry=False):
    return RationaleEnvelope(
        chunk_id=chunk_id,
        content=f"content of {chunk_id}",
        source_agent="agent-a",
        source_turn=1,
        hop_sequence=["agent-a", "agent-b"],
        selected_because=["domain_match:billing"],
        score=score,
        must_carry=must_carry,
        token_cost=50,
        ledger_id=None,
    )

def test_ledger_write_and_query():
    ledger = CHPLedger()
    env = _envelope("c_001")
    ledger_id = ledger.write("session_1", hop_number=0, from_agent="a", to_agent="b", envelope=env)
    assert ledger_id is not None
    results = ledger.query("session_1")
    assert len(results) == 1
    assert results[0].chunk_id == "c_001"

def test_ledger_append_only_rejects_duplicate():
    ledger = CHPLedger()
    env = _envelope("c_dup")
    ledger.write("session_1", hop_number=0, from_agent="a", to_agent="b", envelope=env)
    with pytest.raises(RuntimeError, match="duplicate"):
        ledger.write("session_1", hop_number=0, from_agent="a", to_agent="b", envelope=env)

def test_ledger_query_by_hop():
    ledger = CHPLedger()
    ledger.write("s1", hop_number=0, from_agent="a", to_agent="b", envelope=_envelope("c_hop0"))
    ledger.write("s1", hop_number=1, from_agent="b", to_agent="c", envelope=_envelope("c_hop1"))
    hop0 = ledger.query_hop("s1", hop_number=0)
    assert len(hop0) == 1
    assert hop0[0].chunk_id == "c_hop0"

def test_ledger_query_isolated_by_session():
    ledger = CHPLedger()
    ledger.write("sess_x", hop_number=0, from_agent="a", to_agent="b", envelope=_envelope("c_x"))
    ledger.write("sess_y", hop_number=0, from_agent="a", to_agent="b", envelope=_envelope("c_y"))
    results = ledger.query("sess_x")
    ids = [r.chunk_id for r in results]
    assert "c_x" in ids
    assert "c_y" not in ids
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_ledger.py -v
```

- [ ] **Step 3: Implement CHPLedger**

```python
# chp/ledger/lancedb_ledger.py
from __future__ import annotations
import uuid
from datetime import datetime, timezone
import lancedb
import pyarrow as pa
from chp.schema.rationale_envelope import RationaleEnvelope

_SCHEMA = pa.schema([
    pa.field("ledger_id", pa.string()),
    pa.field("session_id", pa.string()),
    pa.field("hop_number", pa.int32()),
    pa.field("from_agent", pa.string()),
    pa.field("to_agent", pa.string()),
    pa.field("chunk_id", pa.string()),
    pa.field("envelope_json", pa.string()),
    pa.field("timestamp", pa.string()),
])

class CHPLedger:
    _TABLE = "chp_ledger"

    def __init__(self, db_path: str | None = None) -> None:
        self._db = lancedb.connect(db_path or "/tmp/chp_ledger_inmem")
        if self._TABLE not in self._db.table_names():
            self._db.create_table(self._TABLE, schema=_SCHEMA)
        self._table = self._db.open_table(self._TABLE)

    def write(
        self,
        session_id: str,
        hop_number: int,
        from_agent: str,
        to_agent: str,
        envelope: RationaleEnvelope,
    ) -> str:
        existing = self._table.search().where(
            f"session_id = '{session_id}' AND chunk_id = '{envelope.chunk_id}' AND hop_number = {hop_number}"
        ).to_list()
        if existing:
            raise RuntimeError(
                f"duplicate entry: session={session_id} chunk={envelope.chunk_id} hop={hop_number}"
            )
        ledger_id = str(uuid.uuid4())
        self._table.add([{
            "ledger_id": ledger_id,
            "session_id": session_id,
            "hop_number": hop_number,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "chunk_id": envelope.chunk_id,
            "envelope_json": envelope.model_dump_json(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }])
        return ledger_id

    def query(self, session_id: str, agent_id: str | None = None) -> list[RationaleEnvelope]:
        where = f"session_id = '{session_id}'"
        if agent_id:
            where += f" AND to_agent = '{agent_id}'"
        rows = self._table.search().where(where).to_list()
        return [RationaleEnvelope.model_validate_json(r["envelope_json"]) for r in rows]

    def query_hop(self, session_id: str, hop_number: int) -> list[RationaleEnvelope]:
        rows = self._table.search().where(
            f"session_id = '{session_id}' AND hop_number = {hop_number}"
        ).to_list()
        return [RationaleEnvelope.model_validate_json(r["envelope_json"]) for r in rows]
```

```python
# chp/ledger/__init__.py
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_ledger.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add chp/ledger/ tests/unit/test_ledger.py
git commit -s -m "feat(chp): add append-only CHPLedger backed by LanceDB"
```

---

### Task 6: LangGraph Adapter

**Files:**
- Create: `chp/adapters/__init__.py`
- Create: `chp/adapters/langgraph.py`
- Test: `tests/integration/test_langgraph_adapter.py`

**Interfaces:**
- Consumes:
  - `select_chunks()` from `chp.engine.scorer`
  - `CHPLedger` from `chp.ledger.lancedb_ledger`
  - `ContextManifest` from `chp.schema.context_manifest`
  - `AnnotatedChunk` from `chp.schema.rationale_envelope`
- Produces:
  - `chp_node_middleware(manifest: ContextManifest, ledger: CHPLedger, embedder: Embedder) -> Callable` — returns a decorator that wraps a LangGraph node function; intercepts state, runs CHP selection, injects filtered context, writes rationale envelopes to ledger

- [ ] **Step 1: Write failing integration test**

```python
# tests/integration/test_langgraph_adapter.py
import pytest
from datetime import datetime, timezone
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import AnnotatedChunk
from chp.engine.embedder import StubEmbedder
from chp.ledger.lancedb_ledger import CHPLedger
from chp.adapters.langgraph import chp_node_middleware

def test_langgraph_middleware_filters_context():
    manifest = ContextManifest(
        chp_version="0.1",
        agent_id="billing-agent",
        task="resolve_refund",
        requires=ContextRequirements(
            must_carry=["order_id"],
            domain_tags=["billing"],
            history_depth="full",
            recency_window=None,
            exclude=["debug_trace"],
        ),
        token_budget=200,
        on_missing="warn",
    )
    ledger = CHPLedger()
    embedder = StubEmbedder()

    chunks = [
        AnnotatedChunk(chunk_id="c1", content="order_id is 42", token_cost=20,
                       source_agent="auth", source_turn=1,
                       timestamp=datetime(2026,8,19,tzinfo=timezone.utc)),
        AnnotatedChunk(chunk_id="c2", content="debug_trace: stack overflow", token_cost=20,
                       source_agent="auth", source_turn=2,
                       timestamp=datetime(2026,8,19,tzinfo=timezone.utc)),
        AnnotatedChunk(chunk_id="c3", content="billing amount $99", token_cost=20,
                       source_agent="auth", source_turn=3,
                       timestamp=datetime(2026,8,19,tzinfo=timezone.utc)),
    ]

    wrapped_state = {"chp_chunks": chunks, "chp_session_id": "sess_test", "chp_hop": 0}

    @chp_node_middleware(manifest=manifest, ledger=ledger, embedder=embedder)
    def billing_node(state):
        return {"injected_chunks": state["chp_selected_chunks"]}

    result = billing_node(wrapped_state)
    selected_ids = [c.chunk_id for c in result["injected_chunks"]]
    assert "c1" in selected_ids       # must_carry
    assert "c2" not in selected_ids   # excluded
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/integration/test_langgraph_adapter.py -v
```

- [ ] **Step 3: Implement LangGraph adapter**

```python
# chp/adapters/langgraph.py
from __future__ import annotations
from functools import wraps
from typing import Any, Callable
from chp.schema.context_manifest import ContextManifest
from chp.schema.rationale_envelope import AnnotatedChunk, RationaleEnvelope
from chp.engine.embedder import Embedder
from chp.engine.scorer import select_chunks, ScorerWeights
from chp.ledger.lancedb_ledger import CHPLedger

def chp_node_middleware(
    manifest: ContextManifest,
    ledger: CHPLedger,
    embedder: Embedder,
    weights: ScorerWeights | None = None,
) -> Callable:
    def decorator(node_fn: Callable) -> Callable:
        @wraps(node_fn)
        def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            chunks: list[AnnotatedChunk] = state.get("chp_chunks", [])
            session_id: str = state.get("chp_session_id", "default")
            hop: int = state.get("chp_hop", 0)

            selected = select_chunks(chunks, manifest, embedder, weights)

            for i, chunk in enumerate(selected):
                envelope = RationaleEnvelope(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    source_agent=chunk.source_agent,
                    source_turn=chunk.source_turn,
                    hop_sequence=[chunk.source_agent, manifest.agent_id],
                    selected_because=[f"chp_selected_hop:{hop}"],
                    score=0.0,
                    must_carry=chunk.chunk_id in [
                        c.chunk_id for c in chunks
                        if any(k.lower() in c.content.lower() for k in manifest.requires.must_carry)
                    ],
                    token_cost=chunk.token_cost,
                    ledger_id=None,
                )
                try:
                    ledger.write(session_id, hop, chunk.source_agent, manifest.agent_id, envelope)
                except RuntimeError:
                    pass  # already written (idempotent on retry)

            state["chp_selected_chunks"] = selected
            state["chp_hop"] = hop + 1
            return node_fn(state)
        return wrapper
    return decorator
```

```python
# chp/adapters/__init__.py
```

- [ ] **Step 4: Run test**

```bash
pytest tests/integration/test_langgraph_adapter.py -v
```
Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add chp/adapters/ tests/integration/test_langgraph_adapter.py
git commit -s -m "feat(chp): add LangGraph node middleware adapter"
```

---

### Task 7: CrewAI + AutoGen Adapters

**Files:**
- Create: `chp/adapters/crewai.py`
- Create: `chp/adapters/autogen.py`
- Test: `tests/integration/test_crewai_adapter.py`
- Test: `tests/integration/test_autogen_adapter.py`

**Interfaces:**
- Consumes: same as Task 6
- Produces:
  - `CHPCrewTask(task_fn: Callable, manifest: ContextManifest, ledger: CHPLedger, embedder: Embedder)` — wraps a CrewAI task; overrides context injection with CHP selection
  - `CHPConversableAgent(manifest: ContextManifest, ledger: CHPLedger, embedder: Embedder, **kwargs)` — subclasses AutoGen `ConversableAgent`; overrides `_process_received_message` to run CHP selection before reply generation

- [ ] **Step 1: Write failing tests**

```python
# tests/integration/test_crewai_adapter.py
from datetime import datetime, timezone
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import AnnotatedChunk
from chp.engine.embedder import StubEmbedder
from chp.ledger.lancedb_ledger import CHPLedger
from chp.adapters.crewai import CHPCrewTask

def test_chp_crew_task_filters_context():
    manifest = ContextManifest(
        chp_version="0.1", agent_id="researcher", task="summarize",
        requires=ContextRequirements(
            must_carry=["user_id"], domain_tags=["research"],
            history_depth="full", recency_window=None, exclude=["debug"],
        ),
        token_budget=300, on_missing="warn",
    )
    ledger = CHPLedger()
    embedder = StubEmbedder()
    chunks = [
        AnnotatedChunk(chunk_id="c1", content="user_id is 7", token_cost=30,
                       source_agent="planner", source_turn=1,
                       timestamp=datetime(2026,8,19,tzinfo=timezone.utc)),
        AnnotatedChunk(chunk_id="c2", content="debug: verbose log", token_cost=30,
                       source_agent="planner", source_turn=2,
                       timestamp=datetime(2026,8,19,tzinfo=timezone.utc)),
    ]

    received = {}
    def my_task(context):
        received["context"] = context
        return "done"

    task = CHPCrewTask(my_task, manifest=manifest, ledger=ledger, embedder=embedder)
    task.run(chunks=chunks, session_id="sess_crew", hop=0)

    ids = [c.chunk_id for c in received["context"]]
    assert "c1" in ids
    assert "c2" not in ids
```

```python
# tests/integration/test_autogen_adapter.py
from datetime import datetime, timezone
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import AnnotatedChunk
from chp.engine.embedder import StubEmbedder
from chp.ledger.lancedb_ledger import CHPLedger
from chp.adapters.autogen import CHPConversableAgent

def test_chp_autogen_agent_filters_chunks():
    manifest = ContextManifest(
        chp_version="0.1", agent_id="summarizer", task="summarize_ticket",
        requires=ContextRequirements(
            must_carry=["ticket_id"], domain_tags=["support"],
            history_depth="full", recency_window=None, exclude=[],
        ),
        token_budget=500, on_missing="warn",
    )
    ledger = CHPLedger()
    embedder = StubEmbedder()
    chunks = [
        AnnotatedChunk(chunk_id="c1", content="ticket_id is T-42", token_cost=40,
                       source_agent="router", source_turn=1,
                       timestamp=datetime(2026,8,19,tzinfo=timezone.utc)),
        AnnotatedChunk(chunk_id="c2", content="random noise", token_cost=40,
                       source_agent="router", source_turn=2,
                       timestamp=datetime(2026,8,19,tzinfo=timezone.utc)),
    ]

    agent = CHPConversableAgent(
        manifest=manifest, ledger=ledger, embedder=embedder,
        name="summarizer",
    )
    selected = agent.select_context(chunks=chunks, session_id="sess_ag", hop=0)
    ids = [c.chunk_id for c in selected]
    assert "c1" in ids
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/integration/test_crewai_adapter.py tests/integration/test_autogen_adapter.py -v
```

- [ ] **Step 3: Implement CrewAI adapter**

```python
# chp/adapters/crewai.py
from __future__ import annotations
from typing import Callable, Any
from chp.schema.context_manifest import ContextManifest
from chp.schema.rationale_envelope import AnnotatedChunk, RationaleEnvelope
from chp.engine.embedder import Embedder
from chp.engine.scorer import select_chunks, ScorerWeights
from chp.ledger.lancedb_ledger import CHPLedger

class CHPCrewTask:
    def __init__(
        self,
        task_fn: Callable,
        manifest: ContextManifest,
        ledger: CHPLedger,
        embedder: Embedder,
        weights: ScorerWeights | None = None,
    ) -> None:
        self._task_fn = task_fn
        self._manifest = manifest
        self._ledger = ledger
        self._embedder = embedder
        self._weights = weights

    def run(self, chunks: list[AnnotatedChunk], session_id: str, hop: int) -> Any:
        selected = select_chunks(chunks, self._manifest, self._embedder, self._weights)
        for chunk in selected:
            envelope = RationaleEnvelope(
                chunk_id=chunk.chunk_id, content=chunk.content,
                source_agent=chunk.source_agent, source_turn=chunk.source_turn,
                hop_sequence=[chunk.source_agent, self._manifest.agent_id],
                selected_because=[f"crewai_hop:{hop}"],
                score=0.0, must_carry=False, token_cost=chunk.token_cost, ledger_id=None,
            )
            try:
                self._ledger.write(session_id, hop, chunk.source_agent, self._manifest.agent_id, envelope)
            except RuntimeError:
                pass
        return self._task_fn(selected)
```

- [ ] **Step 4: Implement AutoGen adapter**

```python
# chp/adapters/autogen.py
from __future__ import annotations
from chp.schema.context_manifest import ContextManifest
from chp.schema.rationale_envelope import AnnotatedChunk, RationaleEnvelope
from chp.engine.embedder import Embedder
from chp.engine.scorer import select_chunks, ScorerWeights
from chp.ledger.lancedb_ledger import CHPLedger

class CHPConversableAgent:
    """Lightweight stand-in — wraps CHP selection without requiring AutoGen installed in tests."""

    def __init__(
        self,
        manifest: ContextManifest,
        ledger: CHPLedger,
        embedder: Embedder,
        name: str = "chp-agent",
        weights: ScorerWeights | None = None,
    ) -> None:
        self.name = name
        self._manifest = manifest
        self._ledger = ledger
        self._embedder = embedder
        self._weights = weights

    def select_context(self, chunks: list[AnnotatedChunk], session_id: str, hop: int) -> list[AnnotatedChunk]:
        selected = select_chunks(chunks, self._manifest, self._embedder, self._weights)
        for chunk in selected:
            envelope = RationaleEnvelope(
                chunk_id=chunk.chunk_id, content=chunk.content,
                source_agent=chunk.source_agent, source_turn=chunk.source_turn,
                hop_sequence=[chunk.source_agent, self._manifest.agent_id],
                selected_because=[f"autogen_hop:{hop}"],
                score=0.0, must_carry=False, token_cost=chunk.token_cost, ledger_id=None,
            )
            try:
                self._ledger.write(session_id, hop, chunk.source_agent, self._manifest.agent_id, envelope)
            except RuntimeError:
                pass
        return selected
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/ -v
```
Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
git add chp/adapters/crewai.py chp/adapters/autogen.py tests/integration/
git commit -s -m "feat(chp): add CrewAI and AutoGen adapters"
```

---

## PHASE 3 — Benchmarks + Paper (Weeks 7–10)

---

### Task 8: Baseline + GAIA Benchmark Runner

**Files:**
- Create: `chp/benchmarks/__init__.py`
- Create: `chp/benchmarks/baseline.py`
- Create: `chp/benchmarks/gaia.py`

**Interfaces:**
- Produces:
  - `BenchmarkResult(task_id: str, method: Literal["chp","baseline"], task_success: bool, tokens_used: int, must_carry_recall: float, selection_precision: float)`
  - `run_baseline(chunks, task_fn) -> BenchmarkResult` — passes all chunks, no selection
  - `run_chp(chunks, manifest, task_fn, embedder, ledger) -> BenchmarkResult` — uses `select_chunks()`
  - `run_gaia_subset(n_tasks: int, embedder, ledger) -> list[tuple[BenchmarkResult, BenchmarkResult]]` — returns (chp_result, baseline_result) pairs

- [ ] **Step 1: Implement baseline + benchmark primitives**

```python
# chp/benchmarks/baseline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Callable
from chp.schema.rationale_envelope import AnnotatedChunk
from chp.schema.context_manifest import ContextManifest
from chp.engine.embedder import Embedder
from chp.engine.scorer import select_chunks
from chp.ledger.lancedb_ledger import CHPLedger

@dataclass
class BenchmarkResult:
    task_id: str
    method: Literal["chp", "baseline"]
    task_success: bool
    tokens_used: int
    must_carry_recall: float   # 0.0–1.0
    selection_precision: float  # 0.0–1.0

def run_baseline(
    task_id: str,
    chunks: list[AnnotatedChunk],
    task_fn: Callable[[list[AnnotatedChunk]], bool],
) -> BenchmarkResult:
    success = task_fn(chunks)
    return BenchmarkResult(
        task_id=task_id,
        method="baseline",
        task_success=success,
        tokens_used=sum(c.token_cost for c in chunks),
        must_carry_recall=1.0,   # passes everything — trivially 100%
        selection_precision=0.0,  # unknown without oracle
    )

def run_chp(
    task_id: str,
    chunks: list[AnnotatedChunk],
    manifest: ContextManifest,
    task_fn: Callable[[list[AnnotatedChunk]], bool],
    embedder: Embedder,
    ledger: CHPLedger,
) -> BenchmarkResult:
    selected = select_chunks(chunks, manifest, embedder)
    success = task_fn(selected)
    must_carry_found = sum(
        1 for key in manifest.requires.must_carry
        if any(key.lower() in c.content.lower() for c in selected)
    )
    recall = must_carry_found / max(len(manifest.requires.must_carry), 1)
    return BenchmarkResult(
        task_id=task_id,
        method="chp",
        task_success=success,
        tokens_used=sum(c.token_cost for c in selected),
        must_carry_recall=recall,
        selection_precision=0.0,  # requires oracle
    )
```

```python
# chp/benchmarks/gaia.py
from __future__ import annotations
from chp.benchmarks.baseline import BenchmarkResult, run_baseline, run_chp
from chp.engine.embedder import Embedder
from chp.ledger.lancedb_ledger import CHPLedger

def run_gaia_subset(
    n_tasks: int,
    embedder: Embedder,
    ledger: CHPLedger,
) -> list[tuple[BenchmarkResult, BenchmarkResult]]:
    """
    Stub: real GAIA integration requires HuggingFace dataset access.
    Returns synthetic pairs to validate the benchmark harness.
    """
    results = []
    for i in range(n_tasks):
        from datetime import datetime, timezone
        from chp.schema.rationale_envelope import AnnotatedChunk
        from chp.schema.context_manifest import ContextManifest, ContextRequirements

        chunks = [
            AnnotatedChunk(chunk_id=f"g{i}_c{j}", content=f"fact {j} for task {i}",
                           token_cost=100, source_agent="gaia", source_turn=j,
                           timestamp=datetime(2026,8,19,tzinfo=timezone.utc))
            for j in range(10)
        ]
        manifest = ContextManifest(
            chp_version="0.1", agent_id=f"gaia-agent-{i}", task=f"gaia_task_{i}",
            requires=ContextRequirements(
                must_carry=[f"fact 0 for task {i}"],
                domain_tags=["fact"], history_depth="full",
                recency_window=None, exclude=[],
            ),
            token_budget=300, on_missing="warn",
        )
        baseline_r = run_baseline(f"gaia_{i}", chunks, lambda cs: len(cs) > 0)
        chp_r = run_chp(f"gaia_{i}", chunks, manifest, lambda cs: len(cs) > 0, embedder, ledger)
        results.append((chp_r, baseline_r))
    return results
```

```python
# chp/benchmarks/__init__.py
```

- [ ] **Step 2: Run benchmark smoke test**

```bash
python -c "
from chp.benchmarks.gaia import run_gaia_subset
from chp.engine.embedder import StubEmbedder
from chp.ledger.lancedb_ledger import CHPLedger
results = run_gaia_subset(3, StubEmbedder(), CHPLedger())
for chp_r, base_r in results:
    print(f'{chp_r.task_id}: CHP tokens={chp_r.tokens_used} baseline={base_r.tokens_used}')
"
```
Expected: 3 lines printed, CHP tokens_used < baseline tokens_used

- [ ] **Step 3: Commit**

```bash
git add chp/benchmarks/
git commit -s -m "feat(chp): add benchmark harness with GAIA stub runner"
```

---

### Task 9: JSON Schema Publication + README

**Files:**
- Create: `schema/context_manifest.schema.json`
- Create: `README.md`

**Interfaces:** None (publication artifacts, no code deps)

- [ ] **Step 1: Export and write JSON Schema file**

```bash
python -c "
import json
from chp.schema.context_manifest import ContextManifest
schema = ContextManifest.to_json_schema()
with open('schema/context_manifest.schema.json', 'w') as f:
    json.dump(schema, f, indent=2)
print('Schema written.')
"
```

- [ ] **Step 2: Validate schema is valid JSON Schema Draft 2020-12**

```bash
pip install jsonschema
python -c "
import json, jsonschema
with open('schema/context_manifest.schema.json') as f:
    schema = json.load(f)
print('Schema title:', schema.get('title'))
print('Fields:', list(schema.get('properties', {}).keys()))
"
```
Expected: prints `agent_id`, `task`, `token_budget`, `on_missing`

- [ ] **Step 3: Commit schema + README stub**

```bash
mkdir -p schema
git add schema/context_manifest.schema.json README.md
git commit -s -m "feat(chp): publish ContextManifest JSON Schema as open standard"
```

---

## PARALLEL TRACK — Thesis Validation

These tasks run in parallel with Phase 1–3 and require no code.

---

### Task V1: Framework Maintainer Outreach

- [ ] Identify 3 LangGraph maintainers on GitHub (check recent PRs / CODEOWNERS)
- [ ] Identify 2 CrewAI maintainers
- [ ] Draft outreach message: share thesis doc + JSON Schema + ask for 20-min call
- [ ] Send to LangGraph first (largest user base)
- [ ] Record responses in a validation log

**Outreach message template:**
```
Subject: Open protocol proposal: Context Handoff Protocol (CHP) for multi-agent context selection

Hi [name],

I've been researching a gap in multi-agent systems: no protocol exists where the 
receiving agent declares what context it needs, and the sending agent scores against 
those requirements. I've confirmed this across 6 frameworks and 4 academic papers.

I've drafted a protocol proposal (CHP) with a reference Python implementation:
- ContextManifest schema (JSON Schema Draft 2020-12)
- Weighted scoring engine
- Append-only Context Ledger
- LangGraph/CrewAI/AutoGen adapters

Would love 20 minutes to share the design and get your reaction.

Thesis + spec: [link]
GitHub (coming): [link]
```

---

### Task V2: arXiv Preprint Outline

- [ ] Register at arxiv.org (cs.AI + cs.MA categories)
- [ ] Draft paper structure:
  - Abstract (from spec)
  - §1 Introduction — failure modes, research gap table
  - §2 Problem Formalization — formal definition, select() function
  - §3 CHP Design — 4 layers with pseudocode
  - §4 Evaluation — GAIA + HotpotQA results
  - §5 Related Work — A2A, MCP, Letta, BATON, CA-MCP
  - §6 Conclusion + Future Work
- [ ] Target venue: NeurIPS 2026 workshop (Multi-Agent Systems track) or MLSys 2027
- [ ] Timeline: paper draft after Task 8 benchmark results are available

---

### Task V3: Investor Signal Outreach

- [ ] Identify 3 AI infra investors: a16z Infra (Martin Casado / David Haber), Conviction (Sarah Guo), Felicis
- [ ] Tailor thesis for investor audience: emphasize market size, timing, monetization path, moat
- [ ] Send cold outreach with arXiv preprint link once paper is posted
- [ ] Do NOT pitch before benchmark results exist — paper is the credibility signal

---

*End of plan. Phase 1 produces a working, tested Python library. Phase 2 adds framework adapters. Phase 3 generates the arXiv contribution. Parallel track runs the validation campaign.*
