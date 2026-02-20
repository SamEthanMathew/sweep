# Sweep AI — Improved Context Retrieval

> Fork of the [Sweep AI](https://github.com/sweepai/sweep) codebase with a rebuilt context retrieval pipeline that delivers **60x more efficient** context to the LLM.

## What We Changed

Sweep's original retrieval dumps ~15 raw files (~30K tokens) into the LLM's context window and hopes for the best. We replaced that with a four-stage pipeline that returns **structured, intent-aware definitions** in ~2.5K tokens.

### The Pipeline

```
Query → Intent Detection → Symbol Resolution → Context Assembly → LLM
```

| Stage | What It Does | Latency |
|-------|-------------|:-------:|
| **Intent Detection** | Classifies query into 6 intents (bug fix, refactor, test writing, etc.) with per-intent token budgets | <1ms |
| **Symbol Resolution** | Finds actual definitions via tree-sitter AST parsing, repo-wide symbol index, NL-to-identifier fuzzy matching | 2-25ms |
| **Context Assembly** | Packs DefinitionCards + code into structured XML sections within an 8K token budget | <1ms |
| **Score Boosting** | Promotes definition-containing files in Sweep's existing hybrid search | <1ms |

### Benchmark Results (30 queries, same candidate files)

| Metric | Sweep Baseline | Ours | Delta |
|--------|:-:|:-:|:-:|
| Symbol Recall@K | 78.3% | 80.6% | **+2.3 pp** |
| Wrong-File Rate | 94.0% | 63.3% | **-30.7 pp** |
| Context Efficiency | 0.006 | 0.372 | **60x** |
| Latency (p50 / p95) | 24ms / 31ms | 7ms / 26ms | **faster** |

Same recall. 60x more signal per token. 30 percentage points less noise.

## Key Concepts

**Definitions over usages** — When you search for `ChatGPT`, the baseline returns 30+ files that *mention* it. We return the one file where it's *defined*, with its signature, docstring, and members as a compressed DefinitionCard (~50 tokens).

**Intent-aware budgets** — A bug fix query allocates tokens differently than a "write tests" query:

| Intent | Definitions | Snippets | Tests | Callers |
|--------|:-:|:-:|:-:|:-:|
| BUG_FIX | 30% | 25% | 20% | 15% |
| TEST_WRITING | 40% | 15% | 40% | 0% |
| REFACTOR | 25% | 20% | 15% | 30% |

**NL-to-identifier bridging** — Query "make the code review process faster" has no backtick symbols. Our system generates candidates (`review_process`, `review_pr`, `ReviewProcess`...) and fuzzy-matches them against the repo's 1,007 indexed symbols to find `review_pr`.

## Files

### New Modules

| File | Purpose |
|------|---------|
| `sweepai/core/symbol_resolver.py` | Core engine: DefinitionCard, repo-wide index, NL candidates, fuzzy matching, neighbor expansion, safety net |
| `sweepai/core/intent_detector.py` | Rule-based intent classifier (6 intents, regex patterns, <1ms) |
| `sweepai/core/context_assembler.py` | Token-budgeted assembly with ContextBucket and budget redistribution |
| `sweepai/core/eval_context_retrieval.py` | 30-case evaluation framework with ground-truth test cases |
| `benchmark_comparison.py` | Head-to-head benchmark: baseline vs. ours |
| `test_full_pipeline.py` | Standalone functional tests for all modules |

### Modified Modules

| File | What Changed |
|------|-------------|
| `sweepai/utils/code_validators.py` | Rewrote `extract_definitions` to return structured `DefinitionInfo` objects; added 8-language support |
| `sweepai/core/context_pruning.py` | Rewired `get_relevant_context` to use our pipeline instead of the disabled LLM-DFS |
| `sweepai/utils/ticket_utils.py` | Added 1.5x definition-aware score boost in hybrid search |

### Documentation

| File | Purpose |
|------|---------|
| `REPORT.md` | Full technical report: strategies, architecture, metrics, assumptions, future work |
| `UPDATE.md` | Summary of all changes and why they make ours better |

## Running the Benchmark

```bash
# Install dependencies
pip install tree-sitter tree-sitter-python tree-sitter-javascript rapidfuzz loguru

# Set dummy API keys (needed for module imports, not used at runtime)
set OPENAI_API_KEY=test-key
set COHERE_API_KEY=test-key

# Run benchmark
python benchmark_comparison.py . benchmark_results.json

# Run functional tests
python test_full_pipeline.py
```

## How It Works (Quick Version)

1. User submits an issue: *"fix the bug in context_pruning.py where get_relevant_context returns early"*
2. **Intent detector** classifies this as `BUG_FIX` (confidence 0.90)
3. **Symbol resolver** extracts `get_relevant_context` from backticks, looks it up in the repo index, finds its definition in `context_pruning.py:639`, and pulls neighbor definitions (`RepoContextManager`, `context_dfs`) from the same file
4. **Context assembler** allocates 30% of 8K tokens to definitions, 25% to code snippets, 20% to tests, renders DefinitionCards in compact form, and emits structured XML
5. The LLM receives ~2,500 tokens of targeted context instead of ~30,000 tokens of raw file dumps

## Future Work

- **Identifier prediction model** — Replace rule-based NL candidate generation with a small trained model (CodeT5-small) to close the remaining 3 recall losses
- **Incremental index updates** — File-change-driven index rebuilds instead of full repo scans
- **Vector search integration** — Async embedding results injected into the snippets bucket
- **Background LLM refinement** — Idle-time "name hypothesis" generation for harder queries

---

*Built for the ProdHacks Machine Learning Track, February 2026.*

---

<details>
<summary>Original README</summary>

Hi everyone!

Thank you for all of the support on Sweep.
We're now building an AI coding assistant for JetBrains which is available here:
[https://plugins.jetbrains.com/plugin/26275-sweep-ai](https://plugins.jetbrains.com/plugin/26860-sweep-ai)

</details>
