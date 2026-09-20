# Context Handoff Protocol (CHP)
### Startup Thesis + Research Design
**Date:** 2026-08-19  
**Status:** Draft v0.1 — for validation with framework maintainers and investors

---

## Abstract

Multi-agent AI systems are becoming the dominant deployment pattern, yet no protocol exists for coordinating *what context* passes between agents at handoff time. Current systems either forward full conversation history (expensive, noisy) or truncate arbitrarily (lossy, undebuggable). We formalize this as a **requirements-matching problem** — the sender cannot know what to pass without knowing what the receiver needs — and propose the **Context Handoff Protocol (CHP)**: a four-layer open protocol where the receiving agent declares its context requirements, the sending agent scores and selects against them, every chunk carries a rationale envelope, and an append-only Context Ledger ensures nothing is permanently dropped across multi-hop chains. We confirm zero production solutions exist across six major frameworks and four academic papers (2024–2026). CHP is implementable as a drop-in Python middleware library under 1,000 lines, composable over LangGraph, CrewAI, and AutoGen without framework lock-in.

---

## 1. Problem Formalization

### 1.1 The Coordination Gap

Agent-to-agent context handoff is treated as a **data transfer problem** by every existing system. Pass the conversation history. Truncate if too long. Move on.

It is actually a **requirements-matching problem.**

The sender (Agent A) holds a context set `C_A = {c₁, c₂, ..., cₙ}`. The receiver (Agent B) needs a subset `R_B ⊆ C_A ∪ C_external` to complete its task `T_B`. Without B declaring `R_B`, A cannot select optimally. Without A knowing B's token budget `β`, A cannot respect constraints.

**Current systems solve:**
```
handoff(A, B) = pass(C_A)   # pass everything, or truncate(C_A, n)
```

**The correct problem:**
```
handoff(A, B) = select(C_A, R_B, β)
              = argmax_{S ⊆ C_A, tokens(S) ≤ β} relevance(S, R_B)
```

No framework, protocol, or production system implements this selection function with B's declared requirements as input.

### 1.2 Failure Modes

The gap produces four compounding failure modes:

1. **Context bloat** — full history injected regardless of relevance; 10-agent chain = 10x token cost
2. **Context loss** — truncation drops critical early facts; Agent E has no access to what Agent A established
3. **Silent failures** — Agent B produces wrong output from bad context; no trail to debug
4. **Multi-hop collapse** — in chains A→B→C→D→E, each hop drops context further; failures compound invisibly

### 1.3 Confirmed Research Gap

Systematic review of production systems and academic literature (2024–2025):

| System | Context Storage | Inter-Agent Selection | Receiver Requirements | Rationale |
|---|---|---|---|---|
| LangGraph | Shared state dict | Full state dump | None | None |
| CrewAI | Task output chaining | Manual `context=[]` | None | None |
| AutoGen | Conversation history | Full history forward | None | None |
| Semantic Kernel | ChatHistory | Full history forward | None | None |
| Google A2A | Task + message parts | Task output only | None | None |
| Anthropic MCP | Resources/tools | Not applicable | None | None |
| Mem0 | Per-user memories | Single-agent only | None | None |
| Zep | Knowledge graph | Single-agent only | None | None |
| Letta | Memory tiers | Single-agent only | None | None |

Academic literature confirms the gap explicitly:
- arXiv:2402.03578 — multi-agent survey lists context passing as unsolved
- arXiv:2404.13501 — memory survey identifies multi-agent handoff as "future research direction"
- BATON (arXiv:2608.16889) — nearest work; addresses robotics subtask transitions, not generalizable
- CA-MCP (arXiv:2601.11595) — shared context store extension; blackboard model, no selection coordination

**Finding: receiver-declared context requirements driving sender-side scoring with rationale propagation is unimplemented in any production system or academic prototype.**

---

## 2. Solution Design: Context Handoff Protocol (CHP)

### 2.1 Architecture Overview

CHP is a four-layer protocol sitting as middleware between any two agents:

```
┌──────────────────────────────────────────────────────┐
│  Layer 4: Context Ledger (append-only, cross-hop)    │
├──────────────────────────────────────────────────────┤
│  Layer 3: Rationale Envelope (per-chunk provenance)  │
├──────────────────────────────────────────────────────┤
│  Layer 2: Selection Engine (A scores against R_B)    │
├──────────────────────────────────────────────────────┤
│  Layer 1: ContextManifest (B declares requirements)  │
└──────────────────────────────────────────────────────┘
         ↑ middleware over existing frameworks ↑
    LangGraph | CrewAI | AutoGen | any A2A-compatible pair
```

### 2.2 Layer 1 — ContextManifest

Agent B publishes a manifest before or at handoff initiation. This is the core primitive — everything else derives from it.

```json
{
  "chp_version": "0.1",
  "agent_id": "billing-agent-v2",
  "task": "resolve_refund_dispute",
  "requires": {
    "must_carry": ["user_id", "order_id", "session_id"],
    "domain_tags": ["billing", "auth", "prior_resolution"],
    "history_depth": "decisions_only",
    "recency_window": "last_3_agent_turns"
  },
  "exclude": ["PII_raw", "debug_traces"],
  "token_budget": 2000,
  "on_missing": "fail_hard"
}
```

**Field semantics:**

| Field | Type | Semantics |
|---|---|---|
| `must_carry` | `list[str]` | Hard requirements; A must flag if absent |
| `domain_tags` | `list[str]` | Semantic filter; A scores chunks against these |
| `history_depth` | `enum` | `full` \| `decisions_only` \| `summary` \| `none` |
| `recency_window` | `string` | Recency constraint on chunk selection |
| `exclude` | `list[str]` | Hard exclusions; A must filter these out |
| `token_budget` | `int` | B's constraint; A selects within budget |
| `on_missing` | `enum` | `fail_hard` \| `warn` \| `proceed` — B controls failure mode |

The manifest is published alongside the agent's capability card (A2A-compatible extension point).

### 2.3 Layer 2 — Selection Engine

Agent A scores its context chunks against B's manifest:

```
score(chunk, manifest) =
    α · semantic_sim(chunk.embedding, manifest.domain_tag_embeddings)
  + β · recency_weight(chunk.timestamp)
  + γ · must_carry_bonus(chunk, manifest.must_carry)
  - δ · token_cost(chunk)
  - ε · exclude_penalty(chunk, manifest.exclude)
```

**Algorithm:**
1. Hard-include all chunks matching `must_carry` fields
2. Hard-exclude all chunks matching `exclude` fields
3. Score remaining chunks with the weighted function above
4. Select top-K within `token_budget`
5. If `must_carry` items absent: invoke `on_missing` behavior

Weights (α, β, γ, δ, ε) are tunable per deployment. CHP ships a default set calibrated on GAIA and HotpotQA benchmarks.

**Implementation:** `sentence-transformers/all-MiniLM-L6-v2` for local embeddings (CPU-friendly, 22M params). Optional: Cohere Rerank API or `cross-encoder/ms-marco-MiniLM-L-6-v2` for higher-fidelity reranking.

### 2.4 Layer 3 — Rationale Envelope

Every chunk passed in a CHP handoff carries provenance metadata:

```json
{
  "chunk_id": "c_047",
  "content": "...",
  "source_agent": "auth-agent",
  "source_turn": 3,
  "hop_sequence": ["auth-agent", "orchestrator", "billing-agent"],
  "selected_because": ["domain_match:auth", "recency:high", "must_carry:session_id"],
  "score": 0.91,
  "must_carry": true,
  "token_cost": 142,
  "ledger_id": "ledger_abc_turn_3"
}
```

**Enables:**
- B to weight or distrust context based on source and selection reason
- Post-hoc debugging of failed handoffs (which chunk caused the wrong answer?)
- Training data for scorer improvement over time (data flywheel)
- Compliance audit trails (who saw what, when, why)

### 2.5 Layer 4 — Context Ledger

The ledger solves the multi-hop collapse problem. It is append-only — nothing is ever deleted.

**Design principle:** Context is never *dropped*, only *not injected*. Any agent at any hop can recover any prior context if its manifest requests it.

```
Agent A → writes C_A to ledger [session_id, hop=0]
Agent B → queries ledger with manifest_B → gets scored slice → works
         → writes C_B to ledger [session_id, hop=1]
Agent C → queries ledger with manifest_C → scores over C_A ∪ C_B → works
         → writes C_C to ledger [session_id, hop=2]
...
Agent E → full lineage available; manifest-driven retrieval across all prior hops
```

**Ledger schema:**
```json
{
  "ledger_id": "uuid",
  "session_id": "session_xyz",
  "hop_number": 2,
  "from_agent": "auth-agent",
  "to_agent": "billing-agent",
  "chunk_id": "c_047",
  "rationale_envelope": { ... },
  "timestamp": "2026-08-19T10:00:00Z",
  "available_to": ["billing-agent", "orchestrator"]
}
```

**Multi-agent topology support:**

| Pattern | CHP behavior |
|---|---|
| Sequential A→B→C→D→E | Each agent queries ledger fresh; nothing permanently dropped |
| Orchestrator→subagents (fan-out) | Each subagent gets its own manifest slice from same ledger |
| Subagents→orchestrator (fan-in) | Orchestrator queries ledger for all subagent outputs + rationale envelopes; knows what each saw |
| Nested orchestrators | Scoped ledger per orchestration session; sub-orchestrators get sub-ledger views |

**Implementation:** LanceDB embedded (in-process, no server, automatic versioning, Apache Arrow). Alternative: SQLite with append-only enforcement (simpler, zero extra dependencies).

---

## 3. Evaluation Plan (arXiv Contribution)

### 3.1 Benchmarks

CHP will be evaluated against naive full-history pass (baseline) on:

| Benchmark | Why |
|---|---|
| **GAIA** | Multi-step reasoning requiring precise context management across agent hops |
| **HotpotQA multi-hop** | Multi-document reasoning; tests whether critical facts survive handoff |
| **Custom billing/auth scenario** | Real-world enterprise pattern with PII exclusion requirements and compliance constraints |

### 3.2 Metrics

| Metric | Definition | Hypothesis |
|---|---|---|
| **Selection precision** | % of passed chunks actually used by B in producing its output | CHP > baseline (less noise) |
| **Selection recall** | % of `must_carry` fields successfully delivered | CHP ≈ 100% vs. baseline's arbitrary truncation |
| **Task success rate** | B's output correctness vs. ground truth | CHP > baseline |
| **Token efficiency** | Tokens consumed at handoff vs. baseline | CHP significantly lower |
| **Failure debuggability** | Human-judged: given a failure, can you identify the cause? | CHP > baseline (rationale envelopes) |

### 3.3 Ablation Studies

1. Manifest vs. no manifest (full CHP vs. scorer only vs. full history)
2. Rationale envelope impact on downstream agent performance
3. Context Ledger recovery: does access to full ledger improve multi-hop task success?
4. Scorer weight sensitivity: how robust is performance to weight perturbations?

---

## 4. Implementation Plan

### 4.1 Dependency Footprint

```python
# Minimal CHP library dependencies
pydantic>=2.0           # ContextManifest + envelope schemas
sentence-transformers   # Local embedding (optional; API fallback supported)
numpy                   # Scoring arithmetic
lancedb                 # Context Ledger (embedded, no server)
```

All MIT/Apache licensed. No vendor lock-in. No managed services required.

### 4.2 Build Phases

**Phase 1 — Schema + Scorer (weeks 1-3)**
- `ContextManifest` Pydantic model + JSON Schema publication
- `RationaleEnvelope` Pydantic model
- Chunk scoring engine with default weights
- Unit tests + scorer benchmarks

**Phase 2 — Context Ledger + Adapters (weeks 4-6)**
- LanceDB-backed Context Ledger with session scoping
- LangGraph adapter (intercept node transition)
- CrewAI adapter (wrap task context injection)
- AutoGen adapter (subclass `ConversableAgent`)
- Integration tests on GAIA subset

**Phase 3 — Benchmark + Paper (weeks 7-10)**
- Full GAIA + HotpotQA evaluation
- Ablation studies
- arXiv paper draft
- GitHub repo with JSON Schema published as open standard

### 4.3 File Structure

```
chp/
├── schema/
│   ├── context_manifest.py       # Pydantic model + JSON Schema export
│   └── rationale_envelope.py     # Per-chunk provenance model
├── engine/
│   ├── scorer.py                 # Weighted scoring function
│   └── embedder.py               # sentence-transformers wrapper
├── ledger/
│   └── lancedb_ledger.py         # Append-only Context Ledger
├── adapters/
│   ├── langgraph.py              # LangGraph node transition hook
│   ├── crewai.py                 # CrewAI task context wrapper
│   └── autogen.py                # AutoGen ConversableAgent subclass
└── benchmarks/
    ├── gaia.py
    └── hotpotqa.py
```

---

## 5. Startup Thesis

### 5.1 Why Now

Three forces converging in 2025-2026:

1. **Multi-agent is the default deployment unit.** LangGraph, CrewAI, AutoGen all shipped production multi-agent support. Enterprises are moving from single-agent experiments to orchestrated pipelines.

2. **Token costs are production costs.** Full-history pass in a 10-agent chain multiplies token spend 10x. At GPT-4o pricing, this is measurable in quarterly bills.

3. **Agent failures are now compliance events.** When agents handle billing, support, and security — wrong output from bad context costs real money. Audit trails are becoming regulatory requirements.

### 5.2 Market Positioning

**Positioning:** CHP is the missing coordination layer for multi-agent AI — the handshake that agents never had. The same way TCP/IP standardized packet routing and MCP standardized tool access, CHP standardizes context negotiation.

**Comparable:** Anthropic's MCP went from publication to thousands of integrations in 9 months. A protocol wins fast when the pain is universal and the implementation cost is low.

**TAM:** Agent observability + orchestration tooling is a $2-4B serviceable market by 2027 (conservative), sitting inside the $50B+ AI infrastructure market.

### 5.3 Competitive Moat

| Moat | Mechanism |
|---|---|
| **Standard first-mover** | Whoever publishes spec + gets 2 framework adoptions owns the API surface; switching cost = re-implementing CHP manifests |
| **Context Ledger data flywheel** | Each handoff generates scorer training data; more usage → better selection → better results → more adoption |
| **Rationale envelope as audit layer** | Once CHP is the compliance audit trail, it is load-bearing infrastructure |
| **Network effects** | More agents publishing manifests → manifest registry → agent discovery layer → platform |

### 5.4 Monetization Path

```
Free (OSS)                        Paid (Enterprise)
──────────────────────────        ────────────────────────────────
ContextManifest JSON Schema       Hosted Context Ledger
Reference scorer (Python)         Managed manifest registry
Rationale envelope                Analytics + failure dashboard
LangGraph/CrewAI/AutoGen          Compliance audit exports (SOC2)
  adapters                        SLA + support
```

**Phase 1 (0-12 months):** OSS only. arXiv paper. Framework adoption.  
**Phase 2 (12-24 months):** Hosted Context Ledger. First enterprise revenue.  
**Phase 3 (24+ months):** Manifest registry. Agents discover each other's requirements. Platform.

### 5.5 Go-to-Market

Network in AI infra is the primary distribution mechanism:

1. **Framework maintainers first** — LangGraph, CrewAI, AutoGen PRs or direct outreach. One integration = access to millions of developers.
2. **arXiv + ML forums** — paper + GitHub drop simultaneously. NeurIPS/MLSys workshop submission. Visibility without ad spend.
3. **AI infra investors** — a16z Infra, Conviction, Felicis all active in agent infra. Paper + working OSS = credible cold outreach.
4. **Enterprise via framework channel** — once frameworks adopt, enterprises follow standard.

### 5.6 Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Google A2A adds manifest semantics | Medium (12-18 months) | Ship and get LangGraph adoption before they get there |
| LangChain builds this natively | Medium | If they adopt CHP schema, that's a win — open standard survives |
| Protocol adoption without distribution | High (early) | arXiv paper + network substitutes for Anthropic-scale push |
| Scorer quality degrades on vague manifests | Low-medium | Manifest linting tool; template library for common agent types |

---

## 6. Differentiation Summary

| Dimension | CHP | Every Existing System |
|---|---|---|
| Receiver declares requirements | Yes — ContextManifest | No |
| Sender scores against requirements | Yes — weighted scorer | No (pass-all or truncate) |
| Per-chunk rationale | Yes — RationaleEnvelope | No |
| Cross-hop context preservation | Yes — Context Ledger | No (dropped = gone) |
| Multi-topology support | Yes — sequential, fan-out, fan-in, nested | Partial (framework-specific) |
| Framework lock-in | None — middleware | Framework-specific |

---

## Next Steps

1. **Publish ContextManifest JSON Schema** on GitHub as open standard
2. **Build reference scorer** — Python library, MIT licensed
3. **Run GAIA benchmark** — establish baseline vs. CHP delta
4. **Submit arXiv preprint** — target MLSys 2027 or NeurIPS 2026 workshop track
5. **Open PRs** to LangGraph + CrewAI with CHP adapter + benchmark results
6. **Share thesis** with 3-5 AI infra investors and 2-3 framework maintainers for validation signal

---

*CHP is an open protocol proposal. Nothing in this document is proprietary. The goal is standardization, not lock-in.*
