# Phoenix SQL Agent Eval Pipeline — Design Spec

**Date:** 2026-08-18  
**Status:** Approved  

---

## Goal

Build a working Phoenix-based eval pipeline for a real LangGraph SQL agent. Demonstrates full eval suite (tool selection, SQL correctness, hallucination, faithfulness, reasoning path) with traces visible in Phoenix Cloud UI.

---

## Stack

| Component | Choice | Reason |
|---|---|---|
| Agent LLM | `claude-sonnet-4-6` | Capable, already available |
| Eval judge LLM | `claude-haiku-4-5-20251001` | Fast, cheap for batch evals |
| Agent framework | LangGraph | Best Phoenix instrumentation support |
| Database | Chinook SQLite | Zero infra, well-known schema |
| Tracing | Phoenix Cloud (`app.phoenix.arize.com`) | No local Docker needed |
| Instrumentation | `LangChainInstrumentor` | Auto-captures all LLM + tool spans |

---

## Architecture

```
Dataset (NL questions + expected answers)
        │
        ▼
run_experiment()                     Phoenix Cloud UI
        │                            ├── trace waterfall per run
        ▼                            ├── eval scores per question
LangGraph SQL Agent                  └── aggregate dashboard
├── list_tables (tool)
├── get_schema (tool)
├── execute_sql (tool)
└── claude-sonnet-4-6 (LLM)
        │
        ▼
5 Evaluators (judge: claude-haiku-4-5)
├── sql_correctness (code-based)
├── tool_selection (ToolSelectionEvaluator)
├── hallucination (HallucinationEvaluator)
├── faithfulness (FaithfulnessEvaluator)
└── reasoning_path (ClassificationEvaluator)
        │
        ▼
OTel spans → Phoenix Cloud
```

---

## Agent Internals

### State

```python
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

### Graph Flow

```
START
  → list_tables_tool      # discover available tables
  → get_schema_tool       # fetch CREATE statements for relevant tables
  → query_gen             # Claude generates SQL
  → should_continue?
      ├── tool call present → execute_query → back to query_gen
      ├── "Error:" in result → query_gen (retry, max 3)
      └── clean answer → END
```

### Tools

| Tool | Args | Returns |
|---|---|---|
| `list_tables` | none | `list[str]` of table names |
| `get_schema` | `table_names: list[str]` | `str` of CREATE TABLE statements |
| `execute_sql` | `query: str` | rows as JSON string or `"Error: ..."` |

---

## Dataset

10 NL→SQL questions against Chinook DB. Uploaded to Phoenix Cloud as named dataset.

| # | Question | Expected answer type |
|---|---|---|
| 1 | How many artists are there? | count (275) |
| 2 | Top 5 selling artists by invoice total | ranked list |
| 3 | Tracks longer than 5 minutes | track list |
| 4 | Total revenue by country | country + sum |
| 5 | Albums by AC/DC | album names |
| 6 | How many tracks in Rock genre? | count |
| 7 | Customers from Brazil | names |
| 8 | Most expensive track | name + price |
| 9 | Employees who are Sales Agents | names |
| 10 | Average invoice total | number |

---

## Eval Suite

### 1. `sql_correctness` — Code-based
- Extract numbers or sets from agent final answer
- Compare to expected answer
- Score: `1.0` (correct) / `0.0` (incorrect)
- No LLM needed — deterministic

### 2. `tool_selection` — `ToolSelectionEvaluator`
- Checks agent called `list_tables` then `get_schema` before `execute_sql`
- Labels: `correct` / `incorrect`
- Judge: `claude-haiku-4-5-20251001`

### 3. `hallucination` — `HallucinationEvaluator`
- Input: question, Output: final answer, Context: SQL execution result
- Labels: `hallucinated` / `factual`
- Judge: `claude-haiku-4-5-20251001`

### 4. `faithfulness` — `FaithfulnessEvaluator`
- Output faithful to SQL result context
- Labels: `faithful` / `unfaithful`
- Judge: `claude-haiku-4-5-20251001`

### 5. `reasoning_path` — `ClassificationEvaluator`
- Did agent inspect schema before writing SQL?
- Custom prompt checks trajectory for schema-first pattern
- Labels: `correct` / `incorrect`
- Judge: `claude-haiku-4-5-20251001`

---

## File Structure

```
phoenix-sql-eval/
├── agent.py            # LangGraph SQL agent definition
├── tools.py            # list_tables, get_schema, execute_sql
├── evals.py            # all 5 evaluator definitions
├── dataset.py          # upload dataset to Phoenix Cloud
├── run_pipeline.py     # orchestrates full experiment end-to-end
├── chinook.db          # SQLite DB (downloaded at setup)
└── requirements.txt    # all dependencies
```

---

## Data Flow (per experiment row)

```
1. dataset row → run_agent(question)
2. agent runs LangGraph → produces final_answer + trace_id
3. sql_correctness(final_answer, expected) → 0 or 1
4. tool_selection(spans from trace) → correct/incorrect
5. hallucination(answer, sql_result_context) → hallucinated/factual
6. faithfulness(answer, context) → faithful/unfaithful
7. reasoning_path(trajectory) → correct/incorrect
8. all scores attached to experiment row in Phoenix Cloud
```

---

## Environment Variables Required

```
ANTHROPIC_API_KEY=...
PHOENIX_API_KEY=...          # from app.phoenix.arize.com
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com
```

---

## Success Criteria

- Agent runs against all 10 dataset questions without crashing
- All 5 evaluators produce scores for every question
- Phoenix Cloud UI shows trace waterfall + eval scores per question
- Aggregate dashboard shows mean scores across experiment
- Two experiment runs (e.g. sonnet vs haiku as agent) comparable side-by-side
