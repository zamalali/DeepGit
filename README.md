<h1 align="center">
  <img src="https://img.icons8.com/?size=100&id=118557&format=png&color=000000" width="72" style="vertical-align: middle;"/> DeepGit
</h1>

<p align="center">
  <b><i>Find gold in the GitHub haystack.</i></b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg">
  <img alt="MCP" src="https://img.shields.io/badge/MCP-ready-8A2BE2.svg">
  <img alt="PRs welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg">
  <a href="https://github.com/zamalali/DeepGit"><img alt="GitHub stars" src="https://img.shields.io/github/stars/zamalali/DeepGit?style=social"></a>
</p>

**DeepGit** is an agentic research engine for GitHub. Instead of collapsing your
question into a handful of keywords, it keeps the full intent, searches across
keyword, topic, and semantic angles, judges every candidate on real evidence, and
reads the actual source of the close calls before it answers. You get the repo you
actually meant, including the hidden gems a plain search would bury.

> **What's new:** a ground-up rebuild. A genuinely agentic pipeline that spends
> effort only where it is uncertain (typically **2 LLM calls**, escalating to real
> code-reading only for contested picks), persistent **semantic recall** that gets
> smarter the more you use it, and a first-class **MCP server** so your AI
> assistant can call DeepGit directly.

---

## Why DeepGit

| | |
|---|---|
| **Intent-first, not keyword-first** | Your full request is preserved as hard requirements, soft preferences, and anti-patterns the judge checks against. No lossy tag reduction. |
| **Agentic and adaptive** | A zero-cost confidence gate decides whether the fast answer is trustworthy. Only when it is not does DeepGit read real code, re-judge, and reflect. Effort matches uncertainty. |
| **Semantic recall that compounds** | Every repo it sees is embedded into a local vector index (LanceDB + fastembed, CPU-only). Flagship repos whose descriptions omit the obvious words (e.g. `apache/kafka`) still surface, and recall improves with every search you run. |
| **Real constraints, enforced** | Name a language (say "a C++ JSON library") or imply a hardware limit, and DeepGit treats it as a hard constraint. A C library never wins a C++ query, and a GPU-only tool never tops a plain "fast X" ask. |
| **MCP-native** | One command turns DeepGit into an MCP server for Claude Desktop, Cursor, or VS Code. [Jump to setup](#use-deepgit-as-an-mcp-server) |
| **Finds hidden gems** | Ranking is by genuine fit, not stars. A small, well-matched repo can and does win. |

---

## Quickstart

```bash
git clone https://github.com/zamalali/DeepGit.git
cd DeepGit
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -e ".[semantic]"                        # the semantic extra is optional but recommended

export GITHUB_API_KEY=ghp_xxx                        # a PAT with public-repo read access
```

Search from the terminal:

```bash
deepgit search "a fast, embeddable key-value store in Go"
```

Or from Python:

```python
from deepgit import search, search_structured

# Markdown report
print(search("type-safe form validation for React with great TypeScript support"))

# Structured result you can program against
result = search_structured("a library to parse and diff PDFs in Python")
print(result.top.repo, result.top.fit_score)   # e.g. "pymupdf/PyMuPDF" 0.92
```

---

## How it works

DeepGit is an adaptive agent, not a fixed pipeline. It does the cheap thing first
and only escalates when the evidence is genuinely contested.

1. **Understand the intent.** One LLM call turns your request into a plan: search angles, GitHub topics, hard *must-haves*, soft *nice-to-haves*, and *anti-patterns*.
2. **Gather widely (0 tokens).** Batched GitHub GraphQL search across keyword queries **and** star-sorted topic sweeps, so flagship repos that keyword search misses still show up.
3. **Semantic recall (0 tokens).** Everything seen is upserted into a local vector index, and near-neighbours from the accumulated corpus are pulled in. Recall compounds across every search.
4. **Prefilter on free signals.** Activity, health, license, tests/CI, and relevance narrow the field to a shortlist at zero LLM cost.
5. **Judge the shortlist (1 LLM call).** One batched, evidence-grounded ranking over compact cards (README + signals), enforcing the language and hardware hard constraints.
6. **Escalate only if uncertain.** A confidence gate decides. If the top picks are contested, DeepGit reads their **real source, tests, and manifests**, re-judges from that code, and a skeptical **reflection critic** reviews the final order.

Typical query: **2 LLM calls**. Hard query: it spends more, because it needs to.

---

## Ways to run

| Command | What it does |
|---|---|
| `deepgit search "..."` | One-off search from the terminal |
| `deepgit mcp` | Run the MCP server (see below) |
| `from deepgit import search` | Use it as a library |

---

## Benchmarks

DeepGit ships a reproducible benchmark suite under [`bench/`](bench). The numbers
below are the held-out set: 12 unseen queries with the adaptive controller on.

| Metric | Value |
|---|---|
| Hit@1 | 75% |
| Hit@3 | 92% |
| MRR | 0.833 |
| Wrong pick at #1 (lower is better) | 0% |
| Avg LLM calls per query | 2.0 |

Run it yourself:

```bash
python -m bench.run_bench --heldout      # held-out set (unseen queries)
python -m bench.run_bench                # core set
python -m bench.run_bench --edge         # robustness / edge cases
```

Each run writes a full per-case report to `_bench_report.md`.

---

## Installation and setup

**Requirements:** Python 3.11+ and a GitHub token.

```bash
git clone https://github.com/zamalali/DeepGit.git
cd DeepGit
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -e ".[semantic]"
```

Configure your GitHub token and an LLM provider via a `.env` file (or environment
variables):

```bash
GITHUB_API_KEY=ghp_xxx

# Pick one LLM provider:
LLM_PROVIDER=groq            # groq | openai | anthropic | vertex_ai
GROQ_API_KEY=...
# For Vertex AI instead: set VERTEX_PROJECT and GOOGLE_APPLICATION_CREDENTIALS
```

> The `[semantic]` extra (LanceDB + fastembed) is optional. Without it DeepGit still
> runs on keyword + topic search. With it you get compounding semantic recall.

---

## Use DeepGit as an MCP server

DeepGit ships a first-class **[Model Context Protocol](https://modelcontextprotocol.io) server**, so you can call it directly from Claude Desktop, Cursor, VS Code, Windsurf, or any MCP-compatible client. Your assistant gets a repository-research superpower with zero glue code.

### Tools exposed

| Tool | Returns | Use it for |
|---|---|---|
| `check_setup()` | Config status JSON | **Call first.** Instant check that your token, LLM creds, and semantic layer are wired up (runs no search). |
| `find_repositories(intent, limit=5)` | Ranked Markdown report | Everyday "find me the best repo for X". Fast, reads code only when uncertain. |
| `find_repositories_json(intent, limit=8)` | Structured JSON (repo, url, fit_score, verdict, why, risks + telemetry) | Programmatic use and building on top of the results. |
| `deep_research(intent)` | Markdown report | High-stakes comparisons. Always reads real code and reflects. |

### Install

```bash
pip install -e .            # installs the `deepgit-mcp` entry point
# optional: persistent semantic recall (LanceDB + fastembed)
pip install -e ".[semantic]"
```

Set a GitHub token (a classic or fine-grained PAT with public-repo read access):

```bash
export GITHUB_API_KEY=ghp_xxx        # Windows PowerShell: $env:GITHUB_API_KEY="ghp_xxx"
```

### Run it

```bash
deepgit-mcp                          # stdio transport (for local clients)
# or over HTTP:
MCP_TRANSPORT=streamable-http PORT=8080 deepgit-mcp
# equivalently:
deepgit mcp --transport streamable-http --port 8080
```

### Wire it into a client

Add DeepGit to your client's MCP config (example for Claude Desktop,
`claude_desktop_config.json`, or VS Code / Cursor `mcp.json`):

```jsonc
{
  "mcpServers": {
    "deepgit": {
      "command": "deepgit-mcp",
      "env": {
        "GITHUB_API_KEY": "ghp_xxx"
      }
    }
  }
}
```

That is it. Ask your assistant *"use DeepGit to find a high-performance message
broker for pub/sub in Go"* and it will call the tool and return a ranked,
evidence-backed answer.

---

## Contributing

Issues and pull requests are welcome. If DeepGit surfaced the wrong repo for a
query, that is a great bug report. The query and the expected repo are exactly what
make the benchmark stronger.

## License

Released under the [MIT License](LICENSE).
