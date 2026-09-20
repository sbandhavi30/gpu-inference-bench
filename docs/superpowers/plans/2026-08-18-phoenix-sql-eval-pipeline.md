# Phoenix SQL Agent Eval Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph SQL agent with full Phoenix Cloud eval suite covering tool selection, SQL correctness, hallucination, faithfulness, and reasoning path.

**Architecture:** LangGraph SQL agent queries a Chinook SQLite DB using 3 tools (list_tables, get_schema, execute_sql). Phoenix Cloud captures all traces via LangChainInstrumentor. Five evaluators score each agent run; results visible in Phoenix Cloud UI as experiments.

**Tech Stack:** `anthropic`, `langchain-anthropic`, `langgraph`, `arize-phoenix`, `openinference-instrumentation-langchain`, `sqlite3` (stdlib), `requests` (Chinook DB download)

**Spec:** `docs/superpowers/specs/2026-08-18-phoenix-sql-eval-pipeline-design.md`

## Global Constraints

- Agent LLM: `claude-sonnet-4-6` (exact model ID)
- Eval judge LLM: `claude-haiku-4-5-20251001` (exact model ID)
- Phoenix endpoint: `https://app.phoenix.arize.com`
- All code lives in `phoenix-sql-eval/` directory at repo root
- Python 3.11+
- No `.env` files — use `os.environ` reads with clear error if missing
- `chinook.db` downloaded at runtime if not present — never committed to git

---

## File Map

| File | Responsibility |
|---|---|
| `phoenix-sql-eval/requirements.txt` | All dependencies pinned |
| `phoenix-sql-eval/tools.py` | `list_tables()`, `get_schema()`, `execute_sql()` — pure SQLite functions |
| `phoenix-sql-eval/agent.py` | LangGraph graph definition, `run_agent(question) -> dict` |
| `phoenix-sql-eval/evals.py` | All 5 evaluator definitions, returns `list[Evaluator]` |
| `phoenix-sql-eval/dataset.py` | `upload_dataset() -> str` — uploads 10 questions to Phoenix Cloud, returns dataset ID |
| `phoenix-sql-eval/run_pipeline.py` | Entry point — wires tracing, uploads dataset, runs experiment, prints summary |

---

### Task 1: Project Scaffold + Dependencies

**Files:**
- Create: `phoenix-sql-eval/requirements.txt`
- Create: `phoenix-sql-eval/chinook_setup.py`

**Interfaces:**
- Produces: `get_db_path() -> str` — returns path to `chinook.db`, downloading if absent

- [ ] **Step 1: Create requirements.txt**

```
anthropic>=0.40.0
langchain-anthropic>=0.3.0
langgraph>=0.2.0
langchain-core>=0.3.0
arize-phoenix>=5.0.0
openinference-instrumentation-langchain>=0.1.0
opentelemetry-sdk>=1.25.0
opentelemetry-exporter-otlp>=1.25.0
requests>=2.31.0
```

- [ ] **Step 2: Create chinook_setup.py**

```python
import os
import requests

CHINOOK_URL = "https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
DB_PATH = os.path.join(os.path.dirname(__file__), "chinook.db")


def get_db_path() -> str:
    if not os.path.exists(DB_PATH):
        print("Downloading Chinook DB...")
        response = requests.get(CHINOOK_URL, timeout=30)
        response.raise_for_status()
        with open(DB_PATH, "wb") as f:
            f.write(response.content)
        print(f"Saved to {DB_PATH}")
    return DB_PATH
```

- [ ] **Step 3: Install dependencies**

```bash
cd phoenix-sql-eval
pip install -r requirements.txt
```

- [ ] **Step 4: Verify download works**

```bash
cd phoenix-sql-eval
python -c "from chinook_setup import get_db_path; print(get_db_path())"
```

Expected: prints path ending in `chinook.db`

- [ ] **Step 5: Commit**

```bash
git add phoenix-sql-eval/requirements.txt phoenix-sql-eval/chinook_setup.py
git commit -m "feat: scaffold phoenix-sql-eval project with deps and db setup"
```

---

### Task 2: SQLite Tools

**Files:**
- Create: `phoenix-sql-eval/tools.py`
- Create: `phoenix-sql-eval/tests/test_tools.py`

**Interfaces:**
- Consumes: `get_db_path() -> str` from `chinook_setup`
- Produces:
  - `list_tables(db_path: str) -> str` — JSON array of table names
  - `get_schema(db_path: str, table_names: list[str]) -> str` — CREATE TABLE statements joined by newline
  - `execute_sql(db_path: str, query: str) -> str` — JSON rows or `"Error: <msg>"`

- [ ] **Step 1: Write failing tests**

```python
# phoenix-sql-eval/tests/test_tools.py
import json
import pytest
from chinook_setup import get_db_path
from tools import list_tables, get_schema, execute_sql


@pytest.fixture
def db():
    return get_db_path()


def test_list_tables_returns_json_array(db):
    result = list_tables(db)
    tables = json.loads(result)
    assert isinstance(tables, list)
    assert "Artist" in tables
    assert "Album" in tables
    assert "Track" in tables


def test_get_schema_returns_create_statements(db):
    result = get_schema(db, ["Artist", "Album"])
    assert "CREATE TABLE" in result
    assert "Artist" in result
    assert "Album" in result


def test_execute_sql_returns_rows(db):
    result = execute_sql(db, "SELECT COUNT(*) as cnt FROM Artist")
    rows = json.loads(result)
    assert rows[0]["cnt"] == 275


def test_execute_sql_returns_error_string(db):
    result = execute_sql(db, "SELECT * FROM NonExistentTable")
    assert result.startswith("Error:")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd phoenix-sql-eval
python -m pytest tests/test_tools.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools'`

- [ ] **Step 3: Implement tools.py**

```python
import json
import sqlite3
from typing import Union


def list_tables(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        return json.dumps(tables)
    finally:
        conn.close()


def get_schema(db_path: str, table_names: list[str]) -> str:
    conn = sqlite3.connect(db_path)
    try:
        schemas = []
        for table in table_names:
            cursor = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            )
            row = cursor.fetchone()
            if row:
                schemas.append(row[0])
        return "\n\n".join(schemas)
    finally:
        conn.close()


def execute_sql(db_path: str, query: str) -> str:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(query)
        rows = [dict(row) for row in cursor.fetchall()]
        return json.dumps(rows)
    except sqlite3.Error as e:
        return f"Error: {e}"
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd phoenix-sql-eval
python -m pytest tests/test_tools.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add phoenix-sql-eval/tools.py phoenix-sql-eval/tests/test_tools.py
git commit -m "feat: add sqlite tools for list_tables, get_schema, execute_sql"
```

---

### Task 3: LangGraph SQL Agent

**Files:**
- Create: `phoenix-sql-eval/agent.py`
- Create: `phoenix-sql-eval/tests/test_agent.py`

**Interfaces:**
- Consumes:
  - `list_tables(db_path: str) -> str` from `tools`
  - `get_schema(db_path: str, table_names: list[str]) -> str` from `tools`
  - `execute_sql(db_path: str, query: str) -> str` from `tools`
  - `get_db_path() -> str` from `chinook_setup`
- Produces:
  - `run_agent(question: str, db_path: str) -> dict` — returns `{"answer": str, "messages": list, "sql_context": str}`

- [ ] **Step 1: Write failing test**

```python
# phoenix-sql-eval/tests/test_agent.py
import os
import pytest
from chinook_setup import get_db_path

# skip if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)


def test_run_agent_returns_answer():
    from agent import run_agent
    db_path = get_db_path()
    result = run_agent("How many artists are there?", db_path)
    assert "answer" in result
    assert "275" in result["answer"]


def test_run_agent_uses_sql_context():
    from agent import run_agent
    db_path = get_db_path()
    result = run_agent("How many artists are there?", db_path)
    assert "sql_context" in result
    assert len(result["sql_context"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd phoenix-sql-eval
python -m pytest tests/test_agent.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent'`

- [ ] **Step 3: Implement agent.py**

```python
import json
import os
from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict

import tools as db_tools


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    db_path: str
    sql_context: str
    retry_count: int


def build_graph(db_path: str):
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=_require_env("ANTHROPIC_API_KEY"),
        temperature=0,
    )

    @tool
    def list_tables_tool() -> str:
        """List all tables in the database."""
        return db_tools.list_tables(db_path)

    @tool
    def get_schema_tool(table_names: list[str]) -> str:
        """Get CREATE TABLE statements for the given tables."""
        return db_tools.get_schema(db_path, table_names)

    @tool
    def execute_sql_tool(query: str) -> str:
        """Execute a SQL query. Returns JSON rows or 'Error: <msg>'."""
        return db_tools.execute_sql(db_path, query)

    tool_list = [list_tables_tool, get_schema_tool, execute_sql_tool]
    llm_with_tools = llm.bind_tools(tool_list)

    tool_map = {t.name: t for t in tool_list}

    def call_list_tables(state: State) -> dict:
        result = db_tools.list_tables(db_path)
        msg = ToolMessage(content=result, tool_call_id="init_list", name="list_tables_tool")
        return {"messages": [msg]}

    def call_model(state: State) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def call_tools(state: State) -> dict:
        last = state["messages"][-1]
        results = []
        sql_context = state.get("sql_context", "")
        for tc in last.tool_calls:
            t = tool_map[tc["name"]]
            result = t.invoke(tc["args"])
            if tc["name"] == "execute_sql_tool":
                sql_context = result
            results.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])
            )
        return {"messages": results, "sql_context": sql_context}

    def should_continue(state: State) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "end"

    workflow = StateGraph(State)
    workflow.add_node("init_tables", call_list_tables)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)

    workflow.add_edge(START, "init_tables")
    workflow.add_edge("init_tables", "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def run_agent(question: str, db_path: str) -> dict:
    graph = build_graph(db_path)
    system_msg = HumanMessage(
        content=(
            "You are a SQL expert. Use the tools provided to answer questions about the database. "
            "Always call list_tables first (already done), then get_schema for relevant tables, "
            "then execute_sql to get the answer. Return a concise answer."
        )
    )
    user_msg = HumanMessage(content=question)
    result = graph.invoke(
        {
            "messages": [system_msg, user_msg],
            "db_path": db_path,
            "sql_context": "",
            "retry_count": 0,
        }
    )
    last_ai = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage) and not m.tool_calls),
        None,
    )
    return {
        "answer": last_ai.content if last_ai else "",
        "messages": result["messages"],
        "sql_context": result.get("sql_context", ""),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd phoenix-sql-eval
python -m pytest tests/test_agent.py -v
```

Expected: both tests PASS (requires `ANTHROPIC_API_KEY` set)

- [ ] **Step 5: Commit**

```bash
git add phoenix-sql-eval/agent.py phoenix-sql-eval/tests/test_agent.py
git commit -m "feat: add LangGraph SQL agent with Claude sonnet-4-6"
```

---

### Task 4: Phoenix Tracing Setup

**Files:**
- Create: `phoenix-sql-eval/tracing.py`
- Create: `phoenix-sql-eval/tests/test_tracing.py`

**Interfaces:**
- Consumes: `ANTHROPIC_API_KEY`, `PHOENIX_API_KEY`, `PHOENIX_COLLECTOR_ENDPOINT` env vars
- Produces:
  - `setup_tracing(project_name: str) -> None` — configures OTel + LangChainInstrumentor, must be called once before any agent run
  - `get_phoenix_client()` — returns configured `phoenix.Client` pointed at Phoenix Cloud

- [ ] **Step 1: Write failing test**

```python
# phoenix-sql-eval/tests/test_tracing.py
import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("PHOENIX_API_KEY"),
    reason="PHOENIX_API_KEY not set"
)


def test_setup_tracing_no_exception():
    from tracing import setup_tracing
    # should not raise
    setup_tracing("test-project")


def test_get_phoenix_client_returns_client():
    from tracing import get_phoenix_client
    import phoenix as px
    client = get_phoenix_client()
    assert client is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd phoenix-sql-eval
python -m pytest tests/test_tracing.py -v
```

Expected: `ModuleNotFoundError: No module named 'tracing'`

- [ ] **Step 3: Implement tracing.py**

```python
import os

import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val


def setup_tracing(project_name: str) -> None:
    api_key = _require_env("PHOENIX_API_KEY")
    endpoint = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com"
    )
    otlp_endpoint = f"{endpoint}/v1/traces"

    provider = TracerProvider()
    exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        headers={"api_key": api_key, "project_name": project_name},
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)

    LangChainInstrumentor().instrument()


def get_phoenix_client() -> px.Client:
    api_key = _require_env("PHOENIX_API_KEY")
    endpoint = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com"
    )
    return px.Client(endpoint=endpoint, api_key=api_key)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd phoenix-sql-eval
python -m pytest tests/test_tracing.py -v
```

Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add phoenix-sql-eval/tracing.py phoenix-sql-eval/tests/test_tracing.py
git commit -m "feat: add Phoenix Cloud tracing setup with OTel + LangChainInstrumentor"
```

---

### Task 5: Dataset Upload

**Files:**
- Create: `phoenix-sql-eval/dataset.py`
- Create: `phoenix-sql-eval/tests/test_dataset.py`

**Interfaces:**
- Consumes: `get_phoenix_client()` from `tracing`
- Produces:
  - `QUESTIONS: list[dict]` — 10 NL questions with expected answers
  - `upload_dataset(client: px.Client, name: str) -> str` — uploads dataset, returns dataset ID

- [ ] **Step 1: Write failing test**

```python
# phoenix-sql-eval/tests/test_dataset.py
import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("PHOENIX_API_KEY"),
    reason="PHOENIX_API_KEY not set"
)


def test_questions_has_10_items():
    from dataset import QUESTIONS
    assert len(QUESTIONS) == 10
    for q in QUESTIONS:
        assert "question" in q
        assert "expected" in q


def test_upload_dataset_returns_id():
    from dataset import upload_dataset, QUESTIONS
    from tracing import get_phoenix_client
    client = get_phoenix_client()
    dataset_id = upload_dataset(client, "test-chinook-dataset")
    assert isinstance(dataset_id, str)
    assert len(dataset_id) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd phoenix-sql-eval
python -m pytest tests/test_dataset.py::test_questions_has_10_items -v
```

Expected: `ModuleNotFoundError: No module named 'dataset'`

- [ ] **Step 3: Implement dataset.py**

```python
import phoenix as px

QUESTIONS = [
    {"question": "How many artists are there?", "expected": "275"},
    {
        "question": "Who are the top 5 selling artists by total invoice amount?",
        "expected": "ranked list of artists with totals",
    },
    {
        "question": "Which tracks are longer than 5 minutes?",
        "expected": "list of track names over 300000ms",
    },
    {
        "question": "What is the total revenue by country?",
        "expected": "country names with sum of invoice totals",
    },
    {"question": "What albums does AC/DC have?", "expected": "For Those About To Rock, Let There Be Rock"},
    {"question": "How many tracks are in the Rock genre?", "expected": "count of rock tracks"},
    {"question": "Which customers are from Brazil?", "expected": "list of customer names from Brazil"},
    {
        "question": "What is the most expensive track and its price?",
        "expected": "track name with unit price 1.99",
    },
    {
        "question": "Which employees are Sales Support Agents?",
        "expected": "list of employee names with Sales Support Agent title",
    },
    {"question": "What is the average invoice total?", "expected": "a decimal number around 5-6"},
]


def upload_dataset(client: px.Client, name: str) -> str:
    examples = [
        px.DatasetExample(
            input={"question": q["question"]},
            output={"expected": q["expected"]},
            metadata={},
        )
        for q in QUESTIONS
    ]
    dataset = client.upload_dataset(
        dataset_name=name,
        examples=examples,
    )
    return dataset.id
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd phoenix-sql-eval
python -m pytest tests/test_dataset.py -v
```

Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add phoenix-sql-eval/dataset.py phoenix-sql-eval/tests/test_dataset.py
git commit -m "feat: add 10-question Chinook dataset with Phoenix Cloud upload"
```

---

### Task 6: Eval Suite

**Files:**
- Create: `phoenix-sql-eval/evals.py`
- Create: `phoenix-sql-eval/tests/test_evals.py`

**Interfaces:**
- Consumes: `phoenix.evals` evaluator classes
- Produces:
  - `sql_correctness_eval(output: str, expected: str) -> dict` — code-based, returns `{"score": float, "label": str}`
  - `get_llm_evaluators() -> list` — returns `[HallucinationEvaluator, FaithfulnessEvaluator, ToolSelectionEvaluator, ClassificationEvaluator]` configured with haiku judge

- [ ] **Step 1: Write failing tests**

```python
# phoenix-sql-eval/tests/test_evals.py
import os
import pytest


def test_sql_correctness_exact_number_match():
    from evals import sql_correctness_eval
    result = sql_correctness_eval("There are 275 artists in the database.", "275")
    assert result["score"] == 1.0
    assert result["label"] == "correct"


def test_sql_correctness_wrong_number():
    from evals import sql_correctness_eval
    result = sql_correctness_eval("There are 100 artists.", "275")
    assert result["score"] == 0.0
    assert result["label"] == "incorrect"


def test_sql_correctness_no_number_found():
    from evals import sql_correctness_eval
    result = sql_correctness_eval("I cannot answer that question.", "275")
    assert result["score"] == 0.0


def test_get_llm_evaluators_returns_list():
    from evals import get_llm_evaluators
    evaluators = get_llm_evaluators()
    assert len(evaluators) == 4
    names = [e.name for e in evaluators]
    assert "hallucination" in names
    assert "faithfulness" in names
    assert "tool_selection" in names
    assert "reasoning_path" in names
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd phoenix-sql-eval
python -m pytest tests/test_evals.py -v
```

Expected: `ModuleNotFoundError: No module named 'evals'`

- [ ] **Step 3: Implement evals.py**

```python
import re

from phoenix.evals import (
    ClassificationEvaluator,
    FaithfulnessEvaluator,
    HallucinationEvaluator,
    LLM,
    ToolSelectionEvaluator,
)

REASONING_PATH_TEMPLATE = """You are evaluating an AI agent's reasoning trajectory for a SQL task.

Agent trajectory (messages):
{output}

Did the agent inspect the database schema (called get_schema or mentioned table structure) 
BEFORE writing and executing SQL? This is the correct pattern.

"correct" means the agent checked schema before writing SQL.
"incorrect" means the agent skipped schema inspection or wrote SQL blindly."""


def sql_correctness_eval(output: str, expected: str) -> dict:
    expected_nums = set(re.findall(r"\d+\.?\d*", expected))
    output_nums = set(re.findall(r"\d+\.?\d*", output))
    if not expected_nums:
        return {"score": 0.5, "label": "no_expected_number"}
    overlap = expected_nums & output_nums
    if overlap:
        return {"score": 1.0, "label": "correct"}
    return {"score": 0.0, "label": "incorrect"}


def get_llm_evaluators() -> list:
    llm = LLM(provider="anthropic", model="claude-haiku-4-5-20251001")

    hallucination = HallucinationEvaluator(llm=llm)
    hallucination.name = "hallucination"

    faithfulness = FaithfulnessEvaluator(llm=llm)
    faithfulness.name = "faithfulness"

    tool_sel = ToolSelectionEvaluator(llm=llm)
    tool_sel.name = "tool_selection"

    reasoning = ClassificationEvaluator(
        name="reasoning_path",
        llm=llm,
        prompt_template=REASONING_PATH_TEMPLATE,
        choices={"correct": 1.0, "incorrect": 0.0},
    )

    return [hallucination, faithfulness, tool_sel, reasoning]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd phoenix-sql-eval
python -m pytest tests/test_evals.py -v
```

Expected: all 4 tests PASS (first 3 are code-only, last requires import only)

- [ ] **Step 5: Commit**

```bash
git add phoenix-sql-eval/evals.py phoenix-sql-eval/tests/test_evals.py
git commit -m "feat: add 5-evaluator suite with sql_correctness and LLM judges"
```

---

### Task 7: Run Pipeline (End-to-End Orchestration)

**Files:**
- Create: `phoenix-sql-eval/run_pipeline.py`

**Interfaces:**
- Consumes:
  - `setup_tracing(project_name: str) -> None` from `tracing`
  - `get_phoenix_client()` from `tracing`
  - `upload_dataset(client, name) -> str` from `dataset`
  - `run_agent(question: str, db_path: str) -> dict` from `agent`
  - `sql_correctness_eval(output: str, expected: str) -> dict` from `evals`
  - `get_llm_evaluators() -> list` from `evals`
  - `get_db_path() -> str` from `chinook_setup`
- Produces: `main()` — entry point, prints summary table, experiment URL

- [ ] **Step 1: Implement run_pipeline.py**

```python
import os

import phoenix as px
from phoenix.evals import async_evaluate_dataframe
from phoenix.trace import suppress_tracing

from agent import run_agent
from chinook_setup import get_db_path
from dataset import upload_dataset
from evals import get_llm_evaluators, sql_correctness_eval
from tracing import get_phoenix_client, setup_tracing

PROJECT_NAME = "chinook-sql-agent-eval"
DATASET_NAME = "chinook-10-questions-v1"


def run_task(example: px.Example, db_path: str) -> dict:
    question = example.input["question"]
    expected = example.output["expected"]
    result = run_agent(question, db_path)
    return {
        "output": result["answer"],
        "expected": expected,
        "sql_context": result["sql_context"],
        "trajectory": str([m.content for m in result["messages"]]),
    }


def main():
    print("=== Phoenix SQL Agent Eval Pipeline ===\n")

    # 1. setup tracing → all LangGraph spans go to Phoenix Cloud
    setup_tracing(PROJECT_NAME)
    client = get_phoenix_client()
    db_path = get_db_path()

    # 2. upload dataset (idempotent name check)
    print(f"Uploading dataset '{DATASET_NAME}'...")
    dataset_id = upload_dataset(client, DATASET_NAME)
    print(f"Dataset ID: {dataset_id}\n")

    # 3. load dataset examples
    dataset = client.get_dataset(dataset_id)
    examples = list(dataset.examples)
    print(f"Running agent on {len(examples)} questions...\n")

    # 4. run agent on each example, collect outputs
    outputs = []
    for i, ex in enumerate(examples, 1):
        print(f"  [{i}/10] {ex.input['question']}")
        with suppress_tracing():  # don't double-trace eval calls
            out = run_task(ex, db_path)
        out["question"] = ex.input["question"]
        sql_score = sql_correctness_eval(out["output"], out["expected"])
        out["sql_correctness_score"] = sql_score["score"]
        out["sql_correctness_label"] = sql_score["label"]
        outputs.append(out)
        print(f"       → {out['output'][:80]}...")
        print(f"       sql_correctness: {sql_score['label']} ({sql_score['score']})\n")

    # 5. print summary
    scores = [o["sql_correctness_score"] for o in outputs]
    mean = sum(scores) / len(scores)
    print(f"\n=== SQL Correctness: {mean:.0%} ({sum(1 for s in scores if s == 1.0)}/10 correct) ===")
    print(f"\nView traces + full eval scores at:")
    print(f"  https://app.phoenix.arize.com  →  project: {PROJECT_NAME}")
    print(f"\nTo run LLM evals (hallucination, faithfulness, tool_selection, reasoning_path),")
    print(f"run: python run_llm_evals.py")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create run_llm_evals.py for the async LLM eval pass**

```python
# phoenix-sql-eval/run_llm_evals.py
# Run after run_pipeline.py — scores hallucination, faithfulness, tool_selection, reasoning_path
import asyncio
import pandas as pd
import phoenix as px
from phoenix.evals import async_evaluate_dataframe
from phoenix.trace import suppress_tracing

from evals import get_llm_evaluators
from tracing import get_phoenix_client

PROJECT_NAME = "chinook-sql-agent-eval"


async def run_evals():
    client = get_phoenix_client()

    # pull traces from Phoenix Cloud as dataframe
    spans_df = client.get_spans_dataframe(project_name=PROJECT_NAME)
    print(f"Found {len(spans_df)} spans in project '{PROJECT_NAME}'")

    # filter to root agent spans only
    agent_spans = spans_df[spans_df["span_kind"] == "AGENT"].copy()
    print(f"Evaluating {len(agent_spans)} agent spans...\n")

    evaluators = get_llm_evaluators()

    with suppress_tracing():
        results_df = await async_evaluate_dataframe(
            dataframe=agent_spans,
            evaluators=evaluators,
            concurrency=4,
        )

    print("\n=== LLM Eval Results ===")
    print(results_df[["hallucination", "faithfulness", "tool_selection", "reasoning_path"]].describe())

    # log evals back to Phoenix Cloud
    px.Client().log_evaluations(
        px.SpanEvaluations(
            eval_name="hallucination",
            dataframe=results_df[["context.span_id", "hallucination"]].rename(
                columns={"hallucination": "label"}
            ),
        )
    )
    print("\nEval scores logged to Phoenix Cloud.")
    print("View at: https://app.phoenix.arize.com")


if __name__ == "__main__":
    asyncio.run(run_evals())
```

- [ ] **Step 3: Run end-to-end (requires both API keys)**

```bash
cd phoenix-sql-eval
export ANTHROPIC_API_KEY=your_key
export PHOENIX_API_KEY=your_key
python run_pipeline.py
```

Expected output:
```
=== Phoenix SQL Agent Eval Pipeline ===
Uploading dataset 'chinook-10-questions-v1'...
Dataset ID: <uuid>
Running agent on 10 questions...
  [1/10] How many artists are there?
       → There are 275 artists in the database.
       sql_correctness: correct (1.0)
...
=== SQL Correctness: 80% (8/10 correct) ===
View traces + full eval scores at:
  https://app.phoenix.arize.com  →  project: chinook-sql-agent-eval
```

- [ ] **Step 4: Run LLM evals pass**

```bash
cd phoenix-sql-eval
python run_llm_evals.py
```

Expected: prints eval score summary table, logs scores to Phoenix Cloud

- [ ] **Step 5: Verify in Phoenix Cloud UI**

Open `https://app.phoenix.arize.com`:
- Project `chinook-sql-agent-eval` visible
- 10 traces, each with LLM + tool call spans in waterfall view
- Eval scores (hallucination, faithfulness, tool_selection, reasoning_path) attached to spans
- Aggregate dashboard shows mean scores

- [ ] **Step 6: Commit**

```bash
git add phoenix-sql-eval/run_pipeline.py phoenix-sql-eval/run_llm_evals.py
git commit -m "feat: add end-to-end pipeline orchestration and LLM eval pass"
```

---

### Task 8: Second Experiment Run (Agent Version Comparison)

**Files:**
- Modify: `phoenix-sql-eval/run_pipeline.py` — add `--model` CLI arg
- Create: `phoenix-sql-eval/run_comparison.sh`

**Interfaces:**
- Consumes: `run_agent()` with configurable model string
- Produces: two named experiments in Phoenix Cloud comparable side-by-side

- [ ] **Step 1: Add --model arg to run_pipeline.py**

Replace the `main()` call at bottom with:

```python
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Agent LLM model ID")
    parser.add_argument("--experiment", default="run-1", help="Experiment name suffix")
    args = parser.parse_args()
    main(model=args.model, experiment_suffix=args.experiment)
```

And update `main()` signature:

```python
def main(model: str = "claude-sonnet-4-6", experiment_suffix: str = "run-1"):
    ...
    PROJECT_NAME = f"chinook-sql-agent-eval-{experiment_suffix}"
    setup_tracing(PROJECT_NAME)
    ...
    # pass model to run_agent
    result = run_agent(question, db_path, model=model)
```

Update `agent.py`'s `run_agent` signature:

```python
def run_agent(question: str, db_path: str, model: str = "claude-sonnet-4-6") -> dict:
    graph = build_graph(db_path, model=model)
    ...
```

Update `build_graph` signature:

```python
def build_graph(db_path: str, model: str = "claude-sonnet-4-6"):
    llm = ChatAnthropic(model=model, ...)
```

- [ ] **Step 2: Create run_comparison.sh**

```bash
#!/bin/bash
set -e

echo "=== Run 1: claude-sonnet-4-6 ==="
python run_pipeline.py --model claude-sonnet-4-6 --experiment sonnet

echo ""
echo "=== Run 2: claude-haiku-4-5-20251001 ==="
python run_pipeline.py --model claude-haiku-4-5-20251001 --experiment haiku

echo ""
echo "Compare experiments at: https://app.phoenix.arize.com"
```

```bash
chmod +x run_comparison.sh
```

- [ ] **Step 3: Run comparison**

```bash
cd phoenix-sql-eval
./run_comparison.sh
```

Expected: two experiment projects in Phoenix Cloud UI, side-by-side comparison available

- [ ] **Step 4: Commit**

```bash
git add phoenix-sql-eval/run_pipeline.py phoenix-sql-eval/agent.py phoenix-sql-eval/run_comparison.sh
git commit -m "feat: add model comparison support for side-by-side experiment runs"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Agent LLM: `claude-sonnet-4-6` — Task 3
- ✅ Eval judge LLM: `claude-haiku-4-5-20251001` — Task 6
- ✅ LangGraph agent with 3 tools — Task 3
- ✅ Chinook SQLite DB — Task 1
- ✅ Phoenix Cloud tracing + LangChainInstrumentor — Task 4
- ✅ 10 NL questions dataset — Task 5
- ✅ sql_correctness (code-based) — Task 6
- ✅ ToolSelectionEvaluator — Task 6
- ✅ HallucinationEvaluator — Task 6
- ✅ FaithfulnessEvaluator — Task 6
- ✅ ClassificationEvaluator (reasoning_path) — Task 6
- ✅ run_experiment orchestration — Task 7
- ✅ Side-by-side experiment comparison — Task 8

**Type consistency:**
- `run_agent(question, db_path, model)` consistent across Tasks 3, 7, 8 ✅
- `build_graph(db_path, model)` consistent across Tasks 3, 8 ✅
- `sql_correctness_eval(output, expected) -> dict` consistent Tasks 6, 7 ✅
- `get_llm_evaluators() -> list` consistent Tasks 6, 7 ✅
- `upload_dataset(client, name) -> str` consistent Tasks 5, 7 ✅
