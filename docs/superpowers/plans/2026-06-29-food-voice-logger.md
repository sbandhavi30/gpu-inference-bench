# Food & Voice Logger API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI endpoint that accepts an image + optional voice transcript, runs both through an LLM (OpenAI GPT-4o Vision), applies voice-command overrides, and returns a structured food-log payload.

**Architecture:** Single FastAPI app under `AI/food-voice-logger/`. LLM interaction is encapsulated in `llm/analyzer.py` behind a clean interface; voice command parsing lives in `core/voice_commands.py`; Pydantic schemas are isolated in `schemas/food_log.py`. The endpoint wires these together and is responsible for nothing beyond HTTP orchestration.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, OpenAI SDK (`openai>=1.30`), Pillow (image validation), pytest, pytest-asyncio, httpx (test client)

## Global Constraints

- Python ≥ 3.11
- Pydantic v2 (`model_config` dict style, not `class Config`)
- OpenAI SDK `openai>=1.30` (new `client = AsyncOpenAI()` style — not `openai.ChatCompletion.create`)
- All LLM calls must be async
- `voice_transcript` is an **optional query parameter** (not form field)
- Image is **multipart/form-data**, field name `image`
- Endpoint path: `POST /api/v1/log/food`
- No database writes in this service (pure analysis + response)
- Response must include: `normalized_name`, `regional_name`, `portion_size`, `portion_unit`, `ui_anchor`, `confidence_score`, `overridden_by_voice` flag per food item
- Supported voice commands: `add`, `correct`, `change` — case-insensitive prefix match on transcript

---

## File Structure

```
AI/food-voice-logger/
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app factory + router registration
│   ├── api/
│   │   ├── __init__.py
│   │   └── food_logger.py        # POST /api/v1/log/food endpoint
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py             # Settings (API keys, model name, limits)
│   │   └── voice_commands.py     # Parse transcript → VoiceOverride list
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── analyzer.py           # Async LLM call, returns raw parsed food list
│   │   └── prompts.py            # System + user prompt templates
│   └── schemas/
│       ├── __init__.py
│       └── food_log.py           # Pydantic request/response models
├── tests/
│   ├── conftest.py               # Fixtures: test client, sample image bytes, mock LLM
│   ├── test_endpoint.py          # Integration tests against the HTTP endpoint
│   ├── test_voice_commands.py    # Unit tests for voice command parser
│   └── test_analyzer.py         # Unit tests for LLM analyzer (mocked OpenAI)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Task 1: Project Scaffold & Configuration

**Files:**
- Create: `AI/food-voice-logger/requirements.txt`
- Create: `AI/food-voice-logger/.env.example`
- Create: `AI/food-voice-logger/app/__init__.py`
- Create: `AI/food-voice-logger/app/core/__init__.py`
- Create: `AI/food-voice-logger/app/core/config.py`
- Create: `AI/food-voice-logger/app/api/__init__.py`
- Create: `AI/food-voice-logger/app/llm/__init__.py`
- Create: `AI/food-voice-logger/app/schemas/__init__.py`

**Interfaces:**
- Produces: `Settings` singleton importable as `from app.core.config import settings`
- `settings.openai_api_key: str`
- `settings.openai_model: str` (default `"gpt-4o"`)
- `settings.max_image_bytes: int` (default `10_485_760` — 10 MB)

- [ ] **Step 1: Write failing test for settings**

```python
# tests/test_config.py
import os
import pytest

def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    # re-import to pick up env
    import importlib
    import app.core.config as cfg
    importlib.reload(cfg)
    from app.core.config import settings
    assert settings.openai_api_key == "sk-test-key"
    assert settings.openai_model == "gpt-4o"
    assert settings.max_image_bytes == 10_485_760
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd AI/food-voice-logger
python -m pytest tests/test_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Create virtual env + install deps**

```bash
cd AI/food-voice-logger
python3.11 -m venv .venv
source .venv/bin/activate
```

Create `requirements.txt`:
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
openai>=1.30.0
Pillow>=10.3.0
pydantic>=2.7.0
pydantic-settings>=2.2.0
httpx>=0.27.0
pytest>=8.2.0
pytest-asyncio>=0.23.0
```

```bash
pip install -r requirements.txt
```

- [ ] **Step 4: Create package stubs**

```bash
touch app/__init__.py app/api/__init__.py app/core/__init__.py app/llm/__init__.py app/schemas/__init__.py
```

- [ ] **Step 5: Implement config**

`app/core/config.py`:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str = "sk-mock-key"
    openai_model: str = "gpt-4o"
    max_image_bytes: int = 10_485_760  # 10 MB


settings = Settings()
```

- [ ] **Step 6: Create .env.example**

`.env.example`:
```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
MAX_IMAGE_BYTES=10485760
```

- [ ] **Step 7: Run test — verify passes**

```bash
python -m pytest tests/test_config.py -v
```
Expected: `PASSED`

- [ ] **Step 8: Commit**

```bash
git add AI/food-voice-logger/
git commit -m "feat(food-logger): project scaffold and settings config"
```

---

## Task 2: Pydantic Schemas

**Files:**
- Create: `AI/food-voice-logger/app/schemas/food_log.py`
- Test: `AI/food-voice-logger/tests/test_schemas.py`

**Interfaces:**
- Produces:
  - `FoodItem` — single detected food with all required fields
  - `FoodLogResponse` — top-level response containing `items: list[FoodItem]`, `voice_command_applied: bool`, `raw_transcript: str | None`
  - `PortionUnit` — enum: `CUP | PIECE | GRAM | ML | TABLESPOON | TEASPOON | SERVING | BOWL`

- [ ] **Step 1: Write failing schema tests**

```python
# tests/test_schemas.py
from app.schemas.food_log import FoodItem, FoodLogResponse, PortionUnit


def test_food_item_required_fields():
    item = FoodItem(
        normalized_name="idli",
        regional_name="ఇడ్లి",
        portion_size=2.0,
        portion_unit=PortionUnit.PIECE,
        ui_anchor="idli_visual_0",
        confidence_score=0.92,
        overridden_by_voice=False,
    )
    assert item.normalized_name == "idli"
    assert item.portion_unit == PortionUnit.PIECE


def test_food_log_response_defaults():
    resp = FoodLogResponse(items=[], voice_command_applied=False, raw_transcript=None)
    assert resp.items == []
    assert resp.voice_command_applied is False


def test_confidence_score_validation():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        FoodItem(
            normalized_name="x",
            regional_name="x",
            portion_size=1.0,
            portion_unit=PortionUnit.CUP,
            ui_anchor="x",
            confidence_score=1.5,  # > 1.0 — invalid
            overridden_by_voice=False,
        )
```

- [ ] **Step 2: Run test — verify fails**

```bash
python -m pytest tests/test_schemas.py -v
```
Expected: `ModuleNotFoundError: No module named 'app.schemas.food_log'`

- [ ] **Step 3: Implement schemas**

`app/schemas/food_log.py`:
```python
from enum import Enum
from pydantic import BaseModel, Field


class PortionUnit(str, Enum):
    CUP = "cup"
    PIECE = "piece"
    GRAM = "gram"
    ML = "ml"
    TABLESPOON = "tablespoon"
    TEASPOON = "teaspoon"
    SERVING = "serving"
    BOWL = "bowl"


class FoodItem(BaseModel):
    normalized_name: str = Field(..., description="Standard English food name")
    regional_name: str = Field(..., description="Name in regional/local language")
    portion_size: float = Field(..., gt=0, description="Numeric quantity")
    portion_unit: PortionUnit
    ui_anchor: str = Field(
        ...,
        description="Stable string key for UI overlay binding, e.g. 'idli_visual_0'",
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    overridden_by_voice: bool = Field(
        default=False,
        description="True when voice transcript modified this item",
    )


class FoodLogResponse(BaseModel):
    items: list[FoodItem]
    voice_command_applied: bool
    raw_transcript: str | None = None
```

- [ ] **Step 4: Run tests — verify pass**

```bash
python -m pytest tests/test_schemas.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add app/schemas/food_log.py tests/test_schemas.py
git commit -m "feat(food-logger): Pydantic v2 response schemas"
```

---

## Task 3: Voice Command Parser

**Files:**
- Create: `AI/food-voice-logger/app/core/voice_commands.py`
- Test: `AI/food-voice-logger/tests/test_voice_commands.py`

**Interfaces:**
- Produces: `parse_voice_overrides(transcript: str) -> list[VoiceOverride]`
- `VoiceOverride` dataclass: `command: str`, `target_food: str`, `new_value: str | None`
- `apply_overrides(items: list[dict], overrides: list[VoiceOverride]) -> list[dict]`
  - Mutates a copy of `items`; returns modified list

Command patterns (case-insensitive):
- `"add <food>"` → appends new item stub
- `"correct <food> to <new_food>"` → renames matching `normalized_name`
- `"change <food> portion to <size> <unit>"` → updates `portion_size`/`portion_unit`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_voice_commands.py
from app.core.voice_commands import parse_voice_overrides, apply_overrides, VoiceOverride


def test_parse_add_command():
    overrides = parse_voice_overrides("add masala dosa")
    assert len(overrides) == 1
    assert overrides[0].command == "add"
    assert overrides[0].target_food == "masala dosa"


def test_parse_correct_command():
    overrides = parse_voice_overrides("correct idli to rava idli")
    assert overrides[0].command == "correct"
    assert overrides[0].target_food == "idli"
    assert overrides[0].new_value == "rava idli"


def test_parse_change_command():
    overrides = parse_voice_overrides("change sambar portion to 2 cup")
    assert overrides[0].command == "change"
    assert overrides[0].target_food == "sambar"
    assert overrides[0].new_value == "2 cup"


def test_parse_no_command_returns_empty():
    assert parse_voice_overrides("looks delicious") == []


def test_apply_correct_override():
    items = [{"normalized_name": "idli", "regional_name": "ఇడ్లి", "portion_size": 2.0,
               "portion_unit": "piece", "ui_anchor": "idli_visual_0",
               "confidence_score": 0.9, "overridden_by_voice": False}]
    overrides = [VoiceOverride(command="correct", target_food="idli", new_value="rava idli")]
    result = apply_overrides(items, overrides)
    assert result[0]["normalized_name"] == "rava idli"
    assert result[0]["overridden_by_voice"] is True


def test_apply_add_override():
    items = []
    overrides = [VoiceOverride(command="add", target_food="masala dosa", new_value=None)]
    result = apply_overrides(items, overrides)
    assert len(result) == 1
    assert result[0]["normalized_name"] == "masala dosa"
    assert result[0]["overridden_by_voice"] is True


def test_apply_change_portion():
    items = [{"normalized_name": "sambar", "regional_name": "సాంబార్",
               "portion_size": 1.0, "portion_unit": "bowl",
               "ui_anchor": "sambar_visual_0", "confidence_score": 0.85,
               "overridden_by_voice": False}]
    overrides = [VoiceOverride(command="change", target_food="sambar", new_value="2 cup")]
    result = apply_overrides(items, overrides)
    assert result[0]["portion_size"] == 2.0
    assert result[0]["portion_unit"] == "cup"
    assert result[0]["overridden_by_voice"] is True
```

- [ ] **Step 2: Run tests — verify fail**

```bash
python -m pytest tests/test_voice_commands.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement voice_commands.py**

`app/core/voice_commands.py`:
```python
import re
import copy
from dataclasses import dataclass


@dataclass
class VoiceOverride:
    command: str          # 'add' | 'correct' | 'change'
    target_food: str
    new_value: str | None


_ADD_RE     = re.compile(r"^add\s+(.+)$", re.IGNORECASE)
_CORRECT_RE = re.compile(r"^correct\s+(.+?)\s+to\s+(.+)$", re.IGNORECASE)
_CHANGE_RE  = re.compile(r"^change\s+(.+?)\s+portion\s+to\s+(.+)$", re.IGNORECASE)


def parse_voice_overrides(transcript: str) -> list[VoiceOverride]:
    transcript = transcript.strip()
    overrides: list[VoiceOverride] = []

    if m := _ADD_RE.match(transcript):
        overrides.append(VoiceOverride("add", m.group(1).strip(), None))
    elif m := _CORRECT_RE.match(transcript):
        overrides.append(VoiceOverride("correct", m.group(1).strip(), m.group(2).strip()))
    elif m := _CHANGE_RE.match(transcript):
        overrides.append(VoiceOverride("change", m.group(1).strip(), m.group(2).strip()))

    return overrides


def _new_item_stub(name: str) -> dict:
    return {
        "normalized_name": name,
        "regional_name": name,
        "portion_size": 1.0,
        "portion_unit": "serving",
        "ui_anchor": f"{name.replace(' ', '_')}_voice_added",
        "confidence_score": 1.0,
        "overridden_by_voice": True,
    }


def apply_overrides(items: list[dict], overrides: list[VoiceOverride]) -> list[dict]:
    result = copy.deepcopy(items)

    for override in overrides:
        if override.command == "add":
            result.append(_new_item_stub(override.target_food))

        elif override.command == "correct" and override.new_value:
            for item in result:
                if item["normalized_name"].lower() == override.target_food.lower():
                    item["normalized_name"] = override.new_value
                    item["overridden_by_voice"] = True

        elif override.command == "change" and override.new_value:
            # new_value format: "<size> <unit>"
            parts = override.new_value.split(maxsplit=1)
            if len(parts) == 2:
                try:
                    size = float(parts[0])
                    unit = parts[1].lower()
                except ValueError:
                    continue
                for item in result:
                    if item["normalized_name"].lower() == override.target_food.lower():
                        item["portion_size"] = size
                        item["portion_unit"] = unit
                        item["overridden_by_voice"] = True

    return result
```

- [ ] **Step 4: Run tests — verify pass**

```bash
python -m pytest tests/test_voice_commands.py -v
```
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add app/core/voice_commands.py tests/test_voice_commands.py
git commit -m "feat(food-logger): voice command parser and override engine"
```

---

## Task 4: LLM Analyzer (OpenAI GPT-4o Vision)

**Files:**
- Create: `AI/food-voice-logger/app/llm/prompts.py`
- Create: `AI/food-voice-logger/app/llm/analyzer.py`
- Test: `AI/food-voice-logger/tests/test_analyzer.py`

**Interfaces:**
- Produces: `analyze_food_image(image_bytes: bytes, voice_transcript: str | None, client: AsyncOpenAI) -> list[dict]`
  - Returns list of raw dicts matching `FoodItem` field names (pre-validation)
- Consumes: `settings.openai_model`, `settings.openai_api_key`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_analyzer.py
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from app.llm.analyzer import analyze_food_image


MOCK_LLM_RESPONSE = json.dumps([
    {
        "normalized_name": "idli",
        "regional_name": "ఇడ్లి",
        "portion_size": 2.0,
        "portion_unit": "piece",
        "ui_anchor": "idli_visual_0",
        "confidence_score": 0.93,
        "overridden_by_voice": False,
    }
])


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = MOCK_LLM_RESPONSE
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_analyze_returns_food_items(mock_openai_client):
    fake_image = b"fakeimagebytes"
    result = await analyze_food_image(fake_image, None, mock_openai_client)
    assert len(result) == 1
    assert result[0]["normalized_name"] == "idli"


@pytest.mark.asyncio
async def test_analyze_with_transcript(mock_openai_client):
    fake_image = b"fakeimagebytes"
    result = await analyze_food_image(fake_image, "add masala dosa", mock_openai_client)
    # transcript passed — LLM should have been called with it
    call_args = mock_openai_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    # find the user message with text content
    user_msg = next(m for m in messages if m["role"] == "user")
    text_parts = [c for c in user_msg["content"] if c.get("type") == "text"]
    assert any("add masala dosa" in p["text"] for p in text_parts)


@pytest.mark.asyncio
async def test_analyze_invalid_json_raises(mock_openai_client):
    mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
        "not valid json"
    )
    with pytest.raises(ValueError, match="LLM returned non-JSON"):
        await analyze_food_image(b"img", None, mock_openai_client)
```

- [ ] **Step 2: Run tests — verify fail**

```bash
python -m pytest tests/test_analyzer.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement prompts.py**

`app/llm/prompts.py`:
```python
SYSTEM_PROMPT = """\
You are a precision Indian food recognition assistant.

Given an image of a meal (and optionally a voice note from the user), identify every distinct food item visible.

Return ONLY a valid JSON array. No prose, no markdown fences.

Each object in the array must have exactly these keys:
- normalized_name   (string)  Standard English name, lowercase
- regional_name     (string)  Name in the relevant Indian language (Unicode)
- portion_size      (number)  Estimated quantity, positive float
- portion_unit      (string)  One of: cup, piece, gram, ml, tablespoon, teaspoon, serving, bowl
- ui_anchor         (string)  Stable snake_case key for UI overlay, e.g. "idli_visual_0"
- confidence_score  (number)  0.0 – 1.0
- overridden_by_voice (boolean) Always false — the API layer handles voice overrides

Be precise with Indian portion conventions (e.g. idli = pieces, dal = bowl/cup).
"""


def build_user_message(image_b64: str, voice_transcript: str | None) -> dict:
    content: list[dict] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"},
        }
    ]
    if voice_transcript:
        content.append({
            "type": "text",
            "text": f"User voice note: {voice_transcript}",
        })
    return {"role": "user", "content": content}
```

- [ ] **Step 4: Implement analyzer.py**

`app/llm/analyzer.py`:
```python
import base64
import json
import logging
from openai import AsyncOpenAI

from app.core.config import settings
from app.llm.prompts import SYSTEM_PROMPT, build_user_message

logger = logging.getLogger(__name__)


async def analyze_food_image(
    image_bytes: bytes,
    voice_transcript: str | None,
    client: AsyncOpenAI,
) -> list[dict]:
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        build_user_message(image_b64, voice_transcript),
    ]

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )

    raw_content = response.choices[0].message.content.strip()
    logger.debug("LLM raw response: %s", raw_content)

    try:
        items: list[dict] = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON: {raw_content!r}") from exc

    if not isinstance(items, list):
        raise ValueError(f"LLM response was not a JSON array: {raw_content!r}")

    return items
```

- [ ] **Step 5: Run tests — verify pass**

```bash
python -m pytest tests/test_analyzer.py -v
```
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add app/llm/prompts.py app/llm/analyzer.py tests/test_analyzer.py
git commit -m "feat(food-logger): LLM analyzer with GPT-4o vision integration"
```

---

## Task 5: FastAPI Endpoint + App Wiring

**Files:**
- Create: `AI/food-voice-logger/app/api/food_logger.py`
- Create: `AI/food-voice-logger/app/main.py`
- Test: `AI/food-voice-logger/tests/conftest.py`
- Test: `AI/food-voice-logger/tests/test_endpoint.py`

**Interfaces:**
- Consumes:
  - `analyze_food_image(image_bytes, voice_transcript, client) -> list[dict]` from `app.llm.analyzer`
  - `parse_voice_overrides(transcript) -> list[VoiceOverride]` from `app.core.voice_commands`
  - `apply_overrides(items, overrides) -> list[dict]` from `app.core.voice_commands`
  - `FoodLogResponse`, `FoodItem` from `app.schemas.food_log`
  - `settings` from `app.core.config`
- Produces: `POST /api/v1/log/food` → `FoodLogResponse` (JSON)

Error responses:
- `400` — image exceeds `MAX_IMAGE_BYTES` or cannot be decoded
- `422` — FastAPI automatic (bad form/params)
- `502` — OpenAI call fails

- [ ] **Step 1: Write failing endpoint tests**

```python
# tests/conftest.py
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image


MOCK_ITEMS = [
    {
        "normalized_name": "idli",
        "regional_name": "ఇడ్లి",
        "portion_size": 2.0,
        "portion_unit": "piece",
        "ui_anchor": "idli_visual_0",
        "confidence_score": 0.93,
        "overridden_by_voice": False,
    }
]


@pytest.fixture
def sample_image_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=(200, 150, 100))
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_analyzer():
    with patch("app.api.food_logger.analyze_food_image", new_callable=AsyncMock) as m:
        m.return_value = MOCK_ITEMS
        yield m


@pytest.fixture
def client(mock_analyzer):
    from app.main import create_app
    app = create_app()
    return TestClient(app)
```

```python
# tests/test_endpoint.py
import pytest


def test_food_log_no_transcript(client, sample_image_bytes):
    response = client.post(
        "/api/v1/log/food",
        files={"image": ("meal.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "items" in body
    assert body["voice_command_applied"] is False
    assert body["raw_transcript"] is None
    assert body["items"][0]["normalized_name"] == "idli"


def test_food_log_with_voice_add(client, sample_image_bytes, mock_analyzer):
    response = client.post(
        "/api/v1/log/food",
        files={"image": ("meal.jpg", sample_image_bytes, "image/jpeg")},
        params={"voice_transcript": "add masala dosa"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["voice_command_applied"] is True
    names = [i["normalized_name"] for i in body["items"]]
    assert "masala dosa" in names


def test_food_log_with_voice_correct(client, sample_image_bytes):
    response = client.post(
        "/api/v1/log/food",
        files={"image": ("meal.jpg", sample_image_bytes, "image/jpeg")},
        params={"voice_transcript": "correct idli to rava idli"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["voice_command_applied"] is True
    assert body["items"][0]["normalized_name"] == "rava idli"
    assert body["items"][0]["overridden_by_voice"] is True


def test_food_log_image_too_large(client):
    big_image = b"x" * (10_485_760 + 1)
    response = client.post(
        "/api/v1/log/food",
        files={"image": ("huge.jpg", big_image, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "exceeds" in response.json()["detail"].lower()


def test_food_log_llm_error(client, sample_image_bytes, mock_analyzer):
    mock_analyzer.side_effect = ValueError("LLM returned non-JSON")
    response = client.post(
        "/api/v1/log/food",
        files={"image": ("meal.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 502
```

- [ ] **Step 2: Run tests — verify fail**

```bash
python -m pytest tests/test_endpoint.py -v
```
Expected: `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Implement food_logger.py endpoint**

`app/api/food_logger.py`:
```python
import logging
from openai import AsyncOpenAI, APIError
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.config import settings
from app.core.voice_commands import apply_overrides, parse_voice_overrides
from app.llm.analyzer import analyze_food_image
from app.schemas.food_log import FoodItem, FoodLogResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/log", tags=["food-logger"])

_openai_client = AsyncOpenAI(api_key=settings.openai_api_key)


@router.post("/food", response_model=FoodLogResponse)
async def log_food(
    image: UploadFile = File(..., description="Meal photo (JPEG/PNG)"),
    voice_transcript: str | None = Query(
        default=None,
        description="Optional voice command or description",
    ),
) -> FoodLogResponse:
    image_bytes = await image.read()

    if len(image_bytes) > settings.max_image_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Image exceeds {settings.max_image_bytes} byte limit.",
        )

    try:
        raw_items = await analyze_food_image(image_bytes, voice_transcript, _openai_client)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except APIError as exc:
        logger.error("OpenAI API error: %s", exc)
        raise HTTPException(status_code=502, detail="LLM service unavailable.") from exc

    overrides = parse_voice_overrides(voice_transcript) if voice_transcript else []
    voice_applied = bool(overrides)

    if voice_applied:
        raw_items = apply_overrides(raw_items, overrides)

    items = [FoodItem.model_validate(item) for item in raw_items]

    return FoodLogResponse(
        items=items,
        voice_command_applied=voice_applied,
        raw_transcript=voice_transcript,
    )
```

- [ ] **Step 4: Implement main.py**

`app/main.py`:
```python
import logging
from fastapi import FastAPI

from app.api.food_logger import router as food_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Multi-Modal Food & Voice Logger",
        version="1.0.0",
        description="Image + voice food logging API for Indian metabolic health app",
    )
    app.include_router(food_router)
    return app


logging.basicConfig(level=logging.INFO)
app = create_app()
```

- [ ] **Step 5: Run tests — verify all pass**

```bash
python -m pytest tests/ -v
```
Expected: all tests pass (12+)

- [ ] **Step 6: Smoke test with uvicorn**

```bash
OPENAI_API_KEY=sk-test uvicorn app.main:app --reload --port 8001
# In another terminal:
curl -s http://localhost:8001/docs   # Should show Swagger UI HTML
```

- [ ] **Step 7: Commit**

```bash
git add app/api/food_logger.py app/main.py tests/conftest.py tests/test_endpoint.py
git commit -m "feat(food-logger): FastAPI endpoint with LLM + voice override wiring"
```

---

## Task 6: pytest.ini + README

**Files:**
- Create: `AI/food-voice-logger/pytest.ini`
- Create: `AI/food-voice-logger/README.md`

- [ ] **Step 1: Add pytest.ini**

`pytest.ini`:
```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 2: Run full suite with ini**

```bash
python -m pytest -v
```
Expected: all tests green, no asyncio warnings.

- [ ] **Step 3: Add README**

`README.md`:
```markdown
# food-voice-logger

Multi-modal food logging API. Accepts a meal photo + optional voice command. Returns structured food data with regional names, portion sizes, and UI anchors.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

## Test

```bash
pytest -v
```

## Endpoint

`POST /api/v1/log/food`

| Param | Type | Required |
|---|---|---|
| `image` | multipart file | yes |
| `voice_transcript` | query string | no |

Voice commands: `add <food>`, `correct <food> to <new>`, `change <food> portion to <size> <unit>`
```

- [ ] **Step 4: Final commit**

```bash
git add pytest.ini README.md
git commit -m "chore(food-logger): pytest config and README"
```

---

## Self-Review Checklist

### Spec Coverage

| Requirement | Task |
|---|---|
| Multipart image upload | Task 5 — `UploadFile = File(...)` |
| Optional `voice_transcript` query param | Task 5 — `Query(default=None)` |
| LLM integration (OpenAI Vision) | Task 4 — `analyzer.py` + `prompts.py` |
| Voice command dict override (add/correct/change) | Task 3 — `voice_commands.py` |
| `normalized_name`, `regional_name` in response | Task 2 — `FoodItem` schema |
| `portion_size`, `portion_unit` (cups, pieces) | Task 2 — `FoodItem` + `PortionUnit` enum |
| `ui_anchor` string | Task 2 — `FoodItem.ui_anchor` |
| `overridden_by_voice` flag | Task 2 — `FoodItem.overridden_by_voice` |
| Pydantic v2 validation | Task 2 — `model_validate`, `model_config` style |
| Error handling (400/502) | Task 5 — `HTTPException` blocks |
| Modular, production-ready structure | All tasks — clear file separation |

### Placeholder Scan
None found. All steps contain complete code.

### Type Consistency
- `analyze_food_image` signature consistent across Task 4 definition and Task 5 consumption
- `VoiceOverride` dataclass used consistently between Task 3 definition and Task 5 import
- `FoodItem.model_validate` (Pydantic v2) used correctly in Task 5
