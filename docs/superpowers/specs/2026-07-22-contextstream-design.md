# ContextStream Design Spec
**Date:** 2026-07-22  
**Status:** Draft — awaiting user review  
**Author:** Principal AI Systems Architect session

---

## 1. Problem Statement

Every agent framework (LangChain, LangGraph, CrewAI, Autogen) forces developers to hand-write summarization logic, truncation policies, and RAG retrieval to prevent context overflow. After 15–20 tool calls, the context window fills with raw tool outputs — kubectl logs, API responses, CSV dumps — and model performance degrades badly.

Three failure modes:

| Failure | Mechanism |
|---|---|
| Lost-in-the-middle | Attention degrades for tokens buried in large context windows |
| Context rot | Accumulated raw tool output obscures signal with noise |
| Agent death | Context limit hit mid-task, agent cannot continue |

This is an infrastructure-layer problem being patched at the application layer. ContextStream solves it one layer down — transparent to application code.

**Key insight:** OS virtual memory solved the same finite-resource problem for RAM in 1962. Page out inactive data, page in what's needed, keep a lightweight page table. The mechanism maps directly to LLM context windows.

---

## 2. Prior Art and Validation

| Work | What it proves |
|---|---|
| MemGPT (Berkeley, 2023) | External memory + paging concept works for long conversations |
| "Is Progressive Disclosure All You Need?" | Selective context loading outperforms full-context stuffing on multi-hop tasks |
| Claude internal summarization | Anthropic compresses old turns natively — same principle |
| LangMem, Zep, MemoryOS | Market validating that memory management is unsolved |

**Gap:** None combine append-only paging + tool-aware sub-agent forking + structured lesson extraction. That is the gap ContextStream fills.

---

## 2.5 vLLM RFC Analysis (July 2026)

Community validation confirmed. vLLM is building exactly the infrastructure hooks ContextStream needs. Four RFCs map directly to ContextStream components.

### RFC #37168 — Active Coordination & Two-Zone KV Cache Scheduling
**Status:** Open  
**What it builds:** `POST /release_kv_cache` API — lets agents explicitly invalidate KV cache blocks by message index. Session-aware reference counting via `cache_salt`. Hyperbolic Two-Zone Scheduling (Aging Zone for hard invalidations, Fresh Zone for soft releases).

**ContextStream mapping:**
- `budget.py` calls `POST /release_kv_cache` when tombstone fired — explicitly evicts zombie cache blocks instead of waiting for LRU
- `fork_manager.py` uses `cache_salt` as session ID — sub-agent forks get isolated session scope, main agent's cache untouched
- Hard invalidate (AGING zone) = deduplicator tombstone. Soft release (FRESH zone) = low-confidence lesson candidate for re-paging

**Key insight:** vLLM's "zombie cache" problem (compressed context whose KV blocks linger) is exactly what ContextStream's append-only + explicit invalidation solves at the application layer. These two systems are complementary, not competing.

### RFC #48168 — Q3 2026 Roadmap
**Status:** Open (active)  
**Relevant Q3 items:**
- "Agent hints (Session-ID, Correlation-ID) for vLLM to understand multi-turn agents and their subagents" — direct hook for `fork_manager.py`
- "Prefix cache prefetching, eviction, selective offloading" — ContextStream's `page_table.py` drives the policy, vLLM executes it
- "Production-ready multi-tier KV cache offloading" — ContextStream's `eviction_store.py` maps to DRAM/disk tier

**Timeline:** Q3 2026 (current quarter). APIs arriving now. ContextStream v0.2 should land immediately after these merge.

### RFC #7697 — Memory Tiering (DRAM/Disk KV Offload)
**Status:** Closed (shipped)  
**What shipped:** KV cache eviction to CPU DRAM and disk. `block_manager` extended with CPU allocator. Swap-in/swap-out for prefill sequences. Proved: recompute cost exceeds swap-in cost from DRAM at >1024 tokens.

**ContextStream mapping:** `eviction_store.py` Redis/S3 backends map directly to this tiered storage. Raw tool output evicted to disk; vLLM KV blocks evicted to DRAM. Two-layer cold storage, same principle.

### RFC #5557 — Disaggregated Prefilling via KV Cache Transfer
**Status:** Closed (shipped)  
**What shipped:** `communicator` abstraction for KV cache transfer between vLLM instances. KV database with hash-keyed cache tensors. Enables: prefill on one instance, decode on another; on-demand KV loading for long shared documents.

**ContextStream mapping:** Sub-agent forks map to disaggregated prefill workers. Fork spawns a prefill-optimized vLLM instance for tool context processing, returns lesson to decode instance. `eviction_store` raw refs map to KV database keys.

### RFC-to-Component Mapping Summary

| vLLM RFC | ContextStream Component | Integration |
|---|---|---|
| #37168 `POST /release_kv_cache` | `budget.py` + `deduplicator.py` | Call on tombstone/pressure events |
| #37168 `cache_salt` session ID | `fork_manager.py` | Isolate sub-agent KV scope |
| #48168 Agent Session-ID hints | `fork_manager.py` | Correlation-ID per fork |
| #48168 Selective offloading policy | `page_table.py` | Drive eviction decisions |
| #7697 DRAM/disk tiering | `eviction_store.py` | Map to storage tier backend |
| #5557 KV transfer / disagg prefill | `fork_manager.py` | Sub-agent = prefill worker |

### Positioning clarification

ContextStream is **not** re-implementing what vLLM is building. Division of responsibility:

- **vLLM owns:** KV block lifecycle, cache invalidation mechanics, memory tiering, session reference counting
- **ContextStream owns:** semantic eviction policy (what to evict and when), tool-aware lesson extraction, conflict resolution, structured lesson schema, agent-level orchestration

vLLM handles the KV mechanics. ContextStream provides the intelligence layer that drives them.

---

## 3. Red-Team Findings and Architecture Pivots

### 3.1 KV-Cache Invalidation (Critical — design-breaking for naive VMM)

**Problem:** Injecting or removing tokens mid-context destroys prefix cache coherence. vLLM, SGLang, all modern serving engines cache KV by prefix hash. Any mutation = full recompute. Naive "page out verbose history, page in summary" costs more than no VMM.

**Pivot:** Ledger is append-only. Raw tool output never enters main context. Mutations are structurally impossible.

### 3.2 Loss of Latent Context (Valid, mitigated)

**Problem:** Freeform compression is lossy. Agent loses implicit hints, stylistic cues, causal nuance.

**Mitigation:** Structured extraction schemas per tool type (not freeform summarization). Typed fields: metric, cause, entity, confidence, timestamp. Raw data always retrievable via re-paging.

### 3.3 Compression Latency Loop (Valid, resolved by fork architecture)

**Problem:** Per-turn LLM summarization adds serial latency to every turn.

**Resolution:** Sub-agent forks run extraction in parallel to main agent's next turn. Compression does not block inference.

### 3.4 Architecture Commoditization Risk (Valid long-term, 18-month window)

**Problem:** Mamba, RingAttention, linear attention reduce attention degradation natively at the model layer.

**Assessment:** Token cost remains even with better architectures. Lost-in-middle persists empirically at 200k+ tokens. Build for the 18-month window; the eviction/lesson mechanism remains useful even as context windows grow.

---

## 4. Architecture: Approach C — Plugin Layer

ContextStream is a plugin/middleware that hooks into existing frameworks at well-defined extension points. Not a wrapper, not a runtime — a protocol.

**Rejected alternatives:**
- **Approach A (Interceptor/wrap-the-client):** Cannot intercept tool results before they hit context without app cooperation.
- **Approach B (Own the loop):** Competes with LangGraph/CrewAI, high build cost, "yet another framework" adoption problem.

**Why Approach C wins:** Frameworks expose stable hooks. Plugins enable strategy-swappability for benchmarking. Approach A can be built on top of C for raw SDK users.

---

## 5. Core Engine Design

### 5.1 Component Map

```
contextstream/core/
├── ledger.py          # append-only context ledger (KV-cache safe)
├── budget.py          # token budget tracker + pre-flight overflow check
├── page_table.py      # index: active lessons, confidence, raw data pointers
├── extractor.py       # mini-LLM engine (Haiku/GPT-4o-mini) → typed JSON lesson
├── deduplicator.py    # conflict resolver: tombstone/override, never mutates
├── fork_manager.py    # sub-agent lifecycle + dynamic dependency injection
└── eviction_store.py  # cold storage for raw dumps (Redis / S3 / disk)
```

### 5.2 Data Flow — Single Tool Call Cycle

```
Main agent reaches tool call
         ↓
1. budget.py pre-flight: compute token headroom, fire pressure event if needed
         ↓
2. fork_manager.py:
   └── queries page_table.py for tool-specific dependency context
       (targeted injection: "what state does this tool type need?")
   └── spawns sub-agent: tool schema + task + injected dependencies
   └── streams raw output directly into eviction_store.py
       → returns evicted_raw_ref (pointer only, never the data)
         ↓
3. extractor.py:
   └── receives evicted_raw_ref
   └── runs cheap LLM call (Haiku / GPT-4o-mini) with tool-specific schema
   └── output: typed Lesson {metric, cause, entity, confidence, timestamp, raw_ref}
         ↓
4. deduplicator.py:
   └── evaluates incoming lesson against page_table.py
   └── no conflict → pass through
   └── conflict/supersedes → append tombstone + override token
       "[Update Lesson 3]: User ID now 200, was 404"
   └── never mutates past entries
         ↓
5. ledger.py: appends lesson (append-only invariant)
         ↓
6. page_table.py: updates index (lesson_id, confidence, raw_ref, tombstone_of)
         ↓
fork_manager.py kills sub-agent
Main agent resumes — clean context
```

### 5.3 Key Invariants

| Invariant | Reason |
|---|---|
| Ledger is append-only | Prefix cache coherence — KV cache never invalidated |
| Raw tool output never enters main context | Prevents context rot at source |
| Every lesson has confidence score | Enables auto-recall if confidence < threshold |
| Evicted raw data always retrievable | Re-paging possible when agent signals uncertainty |
| Token counting happens before LLM call | Never overflow mid-generation |
| Deduplicator never mutates | Append-only invariant must hold through entire pipeline |

### 5.4 Plugin Hook Interface

```python
class VMMPlugin:
    def before_tool_call(
        self,
        tool_name: str,
        tool_input: dict
    ) -> None

    def after_tool_call(
        self,
        tool_name: str,
        raw_output_stream: Iterator[str]   # stream — not materialized string
    ) -> LessonReference                   # pointer — framework never holds raw data

    def on_token_pressure(
        self,
        level: float,                      # 0.0–1.0, fires at 0.7 / 0.9 / 0.95
        ledger: Ledger
    ) -> None

    def on_agent_uncertainty(
        self,
        query: str
    ) -> list[LessonReference]             # VMM handles re-paging, not the framework
```

---

## 6. Core Data Structures

### 6.1 Pydantic Models

```python
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class Lesson(BaseModel):
    """Structured output from extractor.py. Never enters main context raw."""
    lesson_id: str = Field(description="UUIDv4 or content-hashed string")
    tool_source: str = Field(description="Tool that generated the data: 'kubectl' | 'sql' | 'rest_api' | 'bash' | 'file'")
    metric_impact: Optional[str] = Field(None, description="Quantitative change observed: 'latency: +50ms'")
    root_cause: str = Field(description="Core synthesis of what happened or was found")
    entity_targets: List[str] = Field(default_factory=list, description="Affected entities: ['pod-a', 'user-123']")
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    raw_ref: str = Field(description="Cold storage pointer in eviction_store.py")

class PageTableEntry(BaseModel):
    """Active index entry in page_table.py tracking cognitive state."""
    lesson_id: str
    is_active: bool = True
    confidence: float
    raw_ref: str
    entity_targets: List[str] = Field(default_factory=list, description="Copied from Lesson — enables conflict detection without re-fetching cold storage")
    tombstone_of: Optional[str] = Field(None, description="lesson_id this entry overrides, if any")
    dependency_tags: List[str] = Field(default_factory=list, description="Tool-type tags for dynamic sub-agent dependency lookup")

class LedgerPayload(BaseModel):
    """Exact block appended to the prompt. Append-only — never modified after write."""
    sequence_id: int = Field(description="Monotonically increasing — guarantees chronological order")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload_type: str = Field(description="'LESSON' | 'TOMBSTONE' | 'SYSTEM_INSTRUCTION'")
    text_content: str = Field(description="Human-readable, token-optimized string injected into LLM context")

class LessonReference(BaseModel):
    """Lightweight pointer returned to framework. Framework never holds raw data."""
    lesson_id: str
    confidence: float
    tombstone_of: Optional[str] = None
```

### 6.2 Tool-Specific Extraction Schemas

Each tool type has a typed extraction schema. `extractor.py` uses the schema as a structured output prompt.

```python
# kubectl
KubectlLesson = {
    "resource_type": str,     # "pod" | "deployment" | "node"
    "resource_name": str,
    "condition": str,         # "OOMKilled" | "CrashLoopBackOff" | "Running"
    "metric": str | None,     # "memory=490Mi"
    "cause": str | None,
    "timestamp": datetime | None,
    "confidence": float
}

# SQL
SqlLesson = {
    "query_type": str,        # "SELECT" | "UPDATE" | "INSERT" | "DELETE"
    "tables_affected": list[str],
    "rows_affected": int | None,
    "result_summary": str,
    "error": str | None,
    "confidence": float
}

# REST API
RestLesson = {
    "endpoint": str,
    "status_code": int,
    "entity": str | None,
    "state_change": str | None,
    "error_message": str | None,
    "confidence": float
}
```

### 6.3 Bug Fix Note

`PageTableEntry` includes `entity_targets` (populated from `Lesson` during registration). `Deduplicator` reads entity targets directly from `PageTableEntry` — no cold storage fetch required for conflict detection.

---

## 7. Prototype Engine Loop

Runnable mock demonstrating the full tool call lifecycle and append-only invariant verification.

```python
import uuid
from datetime import datetime
from typing import Iterator, Dict, List

class EvictionStore:
    def __init__(self):
        self.store: Dict[str, str] = {}

    def save_stream(self, stream: Iterator[str]) -> str:
        ref_id = f"raw_ref_{uuid.uuid4().hex[:8]}"
        self.store[ref_id] = "".join(list(stream))
        return ref_id

    def fetch(self, ref_id: str) -> str:
        return self.store.get(ref_id, "")


class PageTable:
    def __init__(self):
        self.table: Dict[str, PageTableEntry] = {}

    def register_entry(self, entry: PageTableEntry):
        if entry.tombstone_of and entry.tombstone_of in self.table:
            self.table[entry.tombstone_of].is_active = False
        self.table[entry.lesson_id] = entry

    def get_dependencies_for_tool(self, tool_name: str) -> List[str]:
        return [
            f"[Active Context Pointer]: {entry.lesson_id}"
            for entry in self.table.values()
            if entry.is_active and tool_name in entry.dependency_tags
        ]


class Extractor:
    def __init__(self, store: EvictionStore):
        self.store = store

    def extract_lesson(self, tool_name: str, raw_ref: str) -> Lesson:
        raw_data = self.store.fetch(raw_ref)
        # Mocks cheap-LLM structured extraction
        if "404" in raw_data:
            return Lesson(
                lesson_id=f"les_{uuid.uuid4().hex[:6]}",
                tool_source=tool_name,
                root_cause="Database returned 404 — entity not found during user lookup",
                entity_targets=["user_service"],
                confidence=0.95,
                raw_ref=raw_ref
            )
        elif "200" in raw_data:
            return Lesson(
                lesson_id=f"les_{uuid.uuid4().hex[:6]}",
                tool_source=tool_name,
                root_cause="DB replication completed. User records recovered, status 200",
                entity_targets=["user_service"],
                confidence=0.99,
                raw_ref=raw_ref
            )
        return Lesson(
            lesson_id=f"les_{uuid.uuid4().hex[:6]}",
            tool_source=tool_name,
            root_cause="Diagnostic completed successfully",
            entity_targets=[],
            confidence=0.80,
            raw_ref=raw_ref
        )


class Deduplicator:
    def __init__(self, page_table: PageTable):
        self.page_table = page_table

    def process(self, incoming: Lesson) -> tuple[str, str | None]:
        for active_id, entry in self.page_table.table.items():
            if not entry.is_active:
                continue
            # Conflict: same entity targets, state has evolved (404 → 200)
            if (set(incoming.entity_targets) & set(entry.entity_targets)
                    and "200" in incoming.root_cause
                    and "404" in self.page_table.table[active_id].root_cause):
                text = f"[TOMBSTONE OVERRIDE - LESSON {active_id}]: State corrected. {incoming.root_cause}."
                return text, active_id
        text = f"[LOGGED LESSON {incoming.lesson_id}]: {incoming.root_cause}."
        return text, None


class Ledger:
    def __init__(self):
        self.history: List[LedgerPayload] = []

    def append(self, text: str, payload_type: str) -> int:
        seq = len(self.history)
        self.history.append(LedgerPayload(sequence_id=seq, payload_type=payload_type, text_content=text))
        return seq

    def render_prompt(self) -> str:
        prompt = "=== SYSTEM CONTEXT LEDGER ===\n"
        for block in self.history:
            prompt += f"[{block.timestamp.isoformat()}] {block.text_content}\n"
        prompt += "=============================\n"
        return prompt


class ContextStreamEngine:
    def __init__(self):
        self.store = EvictionStore()
        self.page_table = PageTable()
        self.extractor = Extractor(self.store)
        self.deduplicator = Deduplicator(self.page_table)
        self.ledger = Ledger()

    def handle_tool_lifecycle(self, tool_name: str, raw_stream: Iterator[str]):
        print(f"\n[VMM Engine] Intercept: {tool_name}")

        deps = self.page_table.get_dependencies_for_tool(tool_name)
        print(f"[Fork Manager] Injected prior state refs: {deps}")

        raw_ref = self.store.save_stream(raw_stream)
        print(f"[Eviction Store] Swallowed stream → pointer: {raw_ref}")

        lesson = self.extractor.extract_lesson(tool_name, raw_ref)
        print(f"[Extractor] Lesson {lesson.lesson_id} | raw size: {len(self.store.fetch(raw_ref))} chars")

        log_text, conflict_id = self.deduplicator.process(lesson)
        payload_type = "TOMBSTONE" if conflict_id else "LESSON"

        seq = self.ledger.append(log_text, payload_type)
        print(f"[Ledger] Appended at sequence {seq} as {payload_type}")

        entry = PageTableEntry(
            lesson_id=lesson.lesson_id,
            confidence=lesson.confidence,
            raw_ref=raw_ref,
            entity_targets=lesson.entity_targets,   # fix: populate for deduplicator
            tombstone_of=conflict_id,
            dependency_tags=[tool_name]
        )
        self.page_table.register_entry(entry)
        active = sum(1 for e in self.page_table.table.values() if e.is_active)
        print(f"[Page Table] Active entries: {active}")


if __name__ == "__main__":
    vmm = ContextStreamEngine()
    vmm.ledger.append("Main agent initialized: Diagnose cluster issues.", "SYSTEM_INSTRUCTION")

    # Turn 1: 404 error buried in log noise
    vmm.handle_tool_lifecycle("kubectl", iter([
        "Error line 1...", "Warning system failure...", "HTTP 404 User Not Found on node-2"
    ]))
    print("\n--- Prompt after Turn 1 ---")
    print(vmm.ledger.render_prompt())

    # Turn 2: corrective call yields 200 — triggers tombstone
    vmm.handle_tool_lifecycle("kubectl", iter([
        "Syncing shards...", "Re-checking connectivity...", "DB Status 200 Connection Restored"
    ]))
    print("\n--- Prompt after Turn 2 (append-only preserved) ---")
    print(vmm.ledger.render_prompt())
```

### Invariant verification from this output

1. **Raw output never in ledger** — only `text_content` (typed lesson string) appended
2. **Prefix unchanged between turns** — Turn 1 prompt is exact prefix of Turn 2 prompt → vLLM KV cache never invalidated
3. **Conflict resolved by append** — TOMBSTONE appended, old entry marked inactive, no mutation of past sequences

---

## 8. Monorepo Structure

```
contextstream/
├── core/              # engine: ledger, budget, page_table, extractor,
│                      #         deduplicator, fork_manager, eviction_store
├── sdk/               # Python wrappers + framework integrations
│   ├── langchain/     # VMMMemory drop-in
│   ├── langgraph/     # before_node / after_node hooks
│   ├── crewai/        # VMMAgent base class
│   ├── llamaindex/    # node postprocessor + query engine wrapper
│   └── raw/           # VMMOpenAI, VMMAnthropic, VMMGoogle client wrappers
├── proxy/             # HTTP proxy (OpenAI-compatible), vLLM plugin, LiteLLM middleware
├── bench/             # benchmarking harness, standard task suite, metrics
├── configs/           # tool schema registry (kubectl, SQL, REST, bash, file)
│                      # default YAML config (thresholds, compression strategy)
└── docs/
    ├── architecture.md
    ├── research-references.md
    └── examples/
```

---

## 9. Revised Roadmap (post-RFC analysis)

### v0.1 — Core Engine MVP (build now)

| Component | Notes |
|---|---|
| `core/budget.py` | Token tracking + pressure events |
| `core/ledger.py` | Append-only, KV-cache safe |
| `core/page_table.py` | Semantic index |
| `core/extractor.py` | 5 tool types: bash, SQL, REST, file, kubectl |
| `core/deduplicator.py` | Tombstone pattern |
| `core/fork_manager.py` | Sub-agent lifecycle, dependency injection |
| `core/eviction_store.py` | Local disk backend only |
| `sdk/langgraph/` | First framework integration |
| Dry-run mode | Critical for adoption — show decisions without modifying context |

### v0.2 — vLLM Native Integration (Q3 2026, after RFC #37168 + #48168 merge)

| Feature | vLLM hook used |
|---|---|
| Explicit KV invalidation on tombstone | `POST /release_kv_cache` (RFC #37168) |
| Sub-agent fork session isolation | `cache_salt` session ID (RFC #37168) |
| DRAM/disk eviction store backend | Memory tiering (RFC #7697, shipped) |
| Disaggregated prefill for forks | KV transfer communicator (RFC #5557, shipped) |
| Agent Session-ID / Correlation-ID | Q3 roadmap agent hints (RFC #48168) |

**Do not build v0.2 before RFC #37168 merges.** APIs are still in flux. Build v0.1 independently; v0.2 is a thin integration layer on top.

### v1.0 — Full Platform

| Feature | |
|---|---|
| HTTP proxy (OpenAI-compatible) | Zero code change deployment |
| LangChain / CrewAI / LlamaIndex SDKs | Broader framework coverage |
| Redis / S3 eviction backends | Production-scale cold storage |
| OpenTelemetry tracing | Full observability |
| Benchmarking harness | VMM vs RAG vs full-context comparison |
| Prometheus metrics | `contextstream_tokens_active`, `evictions_total`, `cache_hit_ratio` |

---

## 10. Feature Requests to Raise Against Upstream Projects

| Project | Feature Request |
|---|---|
| vLLM | Expose prefix cache hit/miss hooks for middleware layers |
| vLLM | Add session-scoped KV cache management API |
| LangChain | VMMMemory: append-only memory with structured eviction |
| LangGraph | Fork/join node type for isolated sub-agent tool execution |
| LiteLLM | Context rewrite middleware hook before token counting |

---

## 11. Open Questions

1. Should `extractor.py` support deterministic (regex/parser) extraction as a fallback for structured tool outputs (e.g., JSON APIs)? Or always LLM-backed?
2. What is the re-paging trigger threshold? Agent self-reports uncertainty, or VMM detects low-confidence lesson chain automatically?
3. For the benchmarking harness: which standard tasks ship in v0.1? Recommend: multi-hop QA, ops debugging (kubectl), RAG synthesis.
4. Compression model selection: user-configurable per tool type, or single global setting?

---

## 12. Success Criteria

| Metric | Target |
|---|---|
| Token reduction at synthesis step | ≥70% vs full-context baseline |
| Agent longevity | Unbounded vs ~15–20 tool calls without VMM |
| KV cache hit rate | No regression vs baseline (append-only guarantees this) |
| Lesson fidelity | Extractable ground-truth facts preserved ≥95% |
| TTFT | ≤ baseline (smaller context = faster, fork latency must not exceed savings) |
