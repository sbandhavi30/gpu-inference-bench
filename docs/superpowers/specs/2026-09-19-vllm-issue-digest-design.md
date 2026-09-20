# vLLM Issue Digest — Design Spec

**Date:** 2026-09-19  
**Goal:** Weekly email digest of new vLLM GitHub issues filtered for inference/KV cache relevance, with plain-English AI summaries and a "recommended to solve" ranking.

---

## Overview

A Python script runs weekly via GitHub Actions. It fetches vLLM issues filed in the last 7 days, filters by topic relevance, uses Claude (Haiku) to summarize each issue and score its contribution worthiness, then sends a formatted HTML email to bs258q@gmail.com via Gmail SMTP.

---

## Components

### 1. `fetch_issues.py`
Calls GitHub REST API (`/repos/vllm-project/vllm/issues`) with:
- `state=open`
- `since=<7 days ago ISO8601>`
- `per_page=100`
- Paginates if needed

Returns raw list of issue dicts (number, title, body, labels, comments, assignees, html_url, created_at).

### 2. `filter.py`
Drops issues that don't match. Two-pass filter:

**Pass 1 — label allow-list** (keep if any label matches):
```
bug, performance, help wanted, good first issue, feature request,
scheduler, kv-cache, inference, torch.compile, quantization
```

**Pass 2 — keyword scan** (title + body, case-insensitive):
```
kv cache, kv-cache, prefix cache, paged attention, pageattention,
scheduler, inference, throughput, latency, ttft, speculative,
memory, oom, block manager, flash attention, flashinfer,
disaggregat, prefill, decode, tensor parallel, lora
```

Issue kept if it matches either pass. Issues labeled `stale` AND filed >30 days ago are dropped.

### 3. `summarize.py`
For each filtered issue, calls Claude `claude-haiku-4-5-20251001` with:

```
System: You are an expert in LLM inference systems. Be concise and technical.

User: Summarize this vLLM GitHub issue for an SRE engineer learning AI infrastructure.

Title: {title}
Labels: {labels}
Body (first 1500 chars): {body}

Return JSON:
{
  "what": "1-2 sentence plain English: what is broken or requested",
  "why": "1 sentence: why this matters for production inference",
  "difficulty": "Easy | Medium | Hard",
  "skills_needed": "brief list e.g. Python, KV cache internals",
  "recommend": "YES | MAYBE | NO",
  "recommend_reason": "1 sentence why"
}
```

**Scoring for `recommend`:**
- `YES`: unassigned + (labeled bug OR kv-cache/scheduler/inference topic) + <10 comments + not stale
- `MAYBE`: unassigned + feature request OR moderate comment count
- `NO`: already has assignee OR labeled stale OR needs CUDA kernel expertise

### 4. `email_digest.py`
Builds HTML email with two sections:

**Section 1 — ⭐ Recommended to Solve** (all `recommend=YES` issues, sorted by: bug > feature, 0 comments first)

**Section 2 — 📋 Full Week's Issues** (all filtered issues, grouped by label type)

Each issue card shows:
```
[🔴 BUG | 🟡 FEATURE | 📄 DOCS]  #number — title
What: ...
Why it matters: ...
Difficulty: Easy  |  Skills: Python, block_manager.py
→ https://github.com/vllm-project/vllm/issues/NUMBER
```

Sends via `smtplib.SMTP_SSL("smtp.gmail.com", 465)` using Gmail app password.

### 5. `main.py`
Orchestrates: fetch → filter → summarize → email. Exits 0 on success, 1 on any error (so Actions marks run as failed).

### 6. `.github/workflows/weekly_digest.yml`
```yaml
on:
  schedule:
    - cron: '0 9 * * 1'   # Monday 9am UTC
  workflow_dispatch:        # manual trigger for testing
```

Secrets used:
- `ANTHROPIC_API_KEY`
- `GMAIL_USER`
- `GMAIL_APP_PASSWORD`
- `DIGEST_TO_EMAIL`

GitHub's built-in `GITHUB_TOKEN` used for API calls (no extra setup).

---

## File Structure

```
vllm-digest/
├── main.py
├── fetch_issues.py
├── filter.py
├── summarize.py
├── email_digest.py
├── requirements.txt          # requests, anthropic
├── .env.example
└── .github/
    └── workflows/
        └── weekly_digest.yml
```

Standalone repo — not inside solarwinds-emu monorepo.

---

## Secrets Setup (one-time)

1. Gmail: enable 2FA → generate App Password → save as `GMAIL_APP_PASSWORD`
2. Anthropic: create API key at console.anthropic.com → save as `ANTHROPIC_API_KEY`
3. Add all secrets to GitHub repo Settings → Secrets → Actions

---

## Cost

Claude Haiku: ~$0.001 per issue summary. At 20 filtered issues/week = ~$0.02/week. Negligible.

---

## Error Handling

- GitHub API rate limit: log and exit 1 (Actions will show failure)
- Claude API error on one issue: skip that issue, log warning, continue
- Email send failure: exit 1 (Actions notifies via email natively)
- Zero issues after filtering: send short email "No relevant issues this week" — don't skip silently

---

## Not In Scope

- Tracking which issues were already seen (dedup across weeks) — v2
- Slack delivery — v2
- Auto-commenting on issues to claim them — v2
