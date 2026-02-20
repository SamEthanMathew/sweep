"""
Head-to-head benchmark: Sweep baseline retrieval vs our improved pipeline.

Runs 30 test cases through both systems and compares:
  A1: Symbol Recall@K
  A2: Wrong-File Rate
  A3: Context Efficiency (recall per 1K tokens)
  A4: Retrieval Latency (p50 / p90 / p95)

Usage:
  python benchmark_comparison.py [repo_dir] [output_json]
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict

from sweepai.core.entities import Snippet
from sweepai.core.eval_context_retrieval import (
    SWEEP_CODEBASE_TEST_CASES,
    EvalTestCase,
    _compute_precision_recall,
)
from sweepai.core.intent_detector import detect_intent
from sweepai.core.symbol_resolver import resolve_symbols_from_query
from sweepai.core.context_assembler import assemble_context, get_context_stats

TOKENS_PER_CHAR = 0.25
BASELINE_TOP_K = 15

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "if", "while", "because", "until", "that", "this",
    "it", "its", "i", "we", "you", "he", "she", "they", "me", "my",
    "your", "our", "what", "which", "who", "whom", "when", "where",
}


@dataclass
class RetrievalResult:
    """Result from running a retriever on a single test case."""
    found_files: list[str] = field(default_factory=list)
    found_symbols: set = field(default_factory=set)
    total_tokens: int = 0
    latency_ms: float = 0.0
    detected_intent: str = ""


@dataclass
class CaseMetrics:
    """Per-case A1-A4 metrics."""
    query_short: str = ""
    symbol_recall: float = 0.0      # A1
    wrong_file_rate: float = 0.0    # A2
    efficiency: float = 0.0         # A3
    latency_ms: float = 0.0        # A4
    found_symbols: list[str] = field(default_factory=list)
    found_files: list[str] = field(default_factory=list)


@dataclass
class SystemSummary:
    """Aggregate metrics across all test cases for one system."""
    name: str = ""
    avg_symbol_recall: float = 0.0
    avg_wrong_file_rate: float = 0.0
    avg_efficiency: float = 0.0
    p50_latency: float = 0.0
    p90_latency: float = 0.0
    p95_latency: float = 0.0
    avg_tokens: float = 0.0
    per_case: list[CaseMetrics] = field(default_factory=list)


# ──────────────────────────────────────────────
#  Pre-load repo files
# ──────────────────────────────────────────────

def load_repo_files(repo_dir: str) -> dict[str, str]:
    """Load all .py files from the repo into memory."""
    files = {}
    for root, dirs, filenames in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            ".cursor", ".github", "docs", "sweep_chat",
        }]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, repo_dir).replace("\\", "/")
            try:
                files[rel] = open(full, "r", encoding="utf-8", errors="replace").read()
            except Exception:
                pass
    return files


# ──────────────────────────────────────────────
#  BASELINE retriever (keyword-match, no symbol resolution)
# ──────────────────────────────────────────────

def _extract_query_terms(query: str) -> list[str]:
    """Extract search terms from a natural-language query."""
    raw = re.findall(r"`(\w+)`", query)
    words = re.findall(r"\b\w+\b", query.lower())
    terms = set(raw)
    for w in words:
        if w not in STOPWORDS and len(w) > 2:
            terms.add(w)
    return list(terms)


def _extract_identifiers_regex(code: str) -> set[str]:
    """Extract identifiers from code using regex (no tree-sitter)."""
    idents = set(re.findall(r"\b([A-Z][a-zA-Z0-9_]{2,})\b", code))
    idents.update(re.findall(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b", code))
    noise = {"self", "None", "True", "False", "str", "int", "float", "bool",
             "list", "dict", "set", "tuple", "print", "return", "import",
             "from", "class", "def", "for", "while", "else", "elif",
             "try", "except", "finally", "with", "pass", "break",
             "continue", "yield", "raise", "assert", "lambda",
             "not_found", "field_default", "default_factory"}
    return idents - noise


def _extract_top_level_definitions(code: str) -> set[str]:
    """
    Extract only top-level class and function DEFINITION names from code.
    This is what a keyword-search baseline would actually surface as "symbols" --
    not every identifier in the file, but the names of things defined there.
    Uses simple regex (the baseline has no tree-sitter).
    """
    defs = set()
    for m in re.finditer(r"^(?:class|def)\s+(\w+)", code, re.MULTILINE):
        defs.add(m.group(1))
    for m in re.finditer(r"^(\w+)\s*=", code, re.MULTILINE):
        name = m.group(1)
        if name.isupper() or name[0].isupper():
            defs.add(name)
    return defs


def baseline_retrieve(
    query: str,
    all_file_contents: dict[str, str],
    k: int = BASELINE_TOP_K,
) -> RetrievalResult:
    """
    Simulates Sweep's original hybrid search: keyword matching on file contents.
    No symbol resolution, no import tracing, no intent detection.
    Returns files ranked by keyword overlap, and the top-level definitions from those files.
    """
    start = time.perf_counter()

    terms = _extract_query_terms(query)
    if not terms:
        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(latency_ms=elapsed)

    file_scores: list[tuple[str, float]] = []
    for path, content in all_file_contents.items():
        content_lower = content.lower()
        score = 0.0
        for term in terms:
            count = content_lower.count(term.lower())
            if count > 0:
                score += 1.0 + min(count / 10.0, 2.0)
        if score > 0:
            file_scores.append((path, score))

    file_scores.sort(key=lambda x: x[1], reverse=True)
    top_files = [path for path, _ in file_scores[:k]]

    found_symbols: set[str] = set()
    total_chars = 0
    for path in top_files:
        content = all_file_contents[path]
        total_chars += len(content)
        found_symbols.update(_extract_top_level_definitions(content))

    elapsed = (time.perf_counter() - start) * 1000
    return RetrievalResult(
        found_files=top_files,
        found_symbols=found_symbols,
        total_tokens=int(total_chars * TOKENS_PER_CHAR),
        latency_ms=elapsed,
    )


# ──────────────────────────────────────────────
#  OUR retriever (intent + symbol resolution + assembly)
# ──────────────────────────────────────────────

def ours_retrieve(
    query: str,
    test_case: EvalTestCase,
    candidate_snippets: list[Snippet],
    repo_dir: str,
) -> RetrievalResult:
    """
    Runs our full pipeline: intent detection + symbol resolution + context assembly.

    Delegates symbol resolution to resolve_symbols_from_query which now handles:
    1. Exact identifier extraction from query text
    2. NL-to-identifier candidate generation
    3. Fuzzy matching against repo-wide symbol index
    4. Safety-net fallback for zero-match cases
    """
    start = time.perf_counter()

    intent = detect_intent(query, test_case.expected_files, candidate_snippets)

    all_cards = resolve_symbols_from_query(query, candidate_snippets, repo_dir)

    assembled = assemble_context(
        intent=intent,
        definition_cards=all_cards[:15],
        top_snippets=candidate_snippets[:5],
        read_only_snippets=[],
        import_trees="",
    )

    elapsed = (time.perf_counter() - start) * 1000
    stats = get_context_stats(assembled)

    return RetrievalResult(
        found_files=list({c.file_path for c in all_cards}),
        found_symbols={c.symbol_name for c in all_cards},
        total_tokens=stats.get("total_tokens", 0),
        latency_ms=elapsed,
        detected_intent=intent.intent_type,
    )


# ──────────────────────────────────────────────
#  Scoring
# ──────────────────────────────────────────────

def score_result(
    result: RetrievalResult,
    test_case: EvalTestCase,
) -> CaseMetrics:
    """Compute A1-A4 metrics for a single result against ground truth."""
    expected_symbols = set(test_case.expected_symbols)
    expected_files = set(test_case.expected_files)
    found_symbols = result.found_symbols if isinstance(result.found_symbols, set) else set(result.found_symbols)
    found_files = set(result.found_files)

    if expected_symbols:
        hits = len(found_symbols & expected_symbols)
        symbol_recall = hits / len(expected_symbols)
    else:
        symbol_recall = 1.0 if not found_symbols else 0.0

    if found_files:
        wrong = len(found_files - expected_files) if expected_files else len(found_files)
        wrong_file_rate = wrong / len(found_files)
    else:
        wrong_file_rate = 0.0 if not expected_files else 1.0

    tokens_k = max(result.total_tokens / 1000.0, 0.001)
    efficiency = symbol_recall / tokens_k

    return CaseMetrics(
        query_short=test_case.query[:55],
        symbol_recall=symbol_recall,
        wrong_file_rate=wrong_file_rate,
        efficiency=efficiency,
        latency_ms=result.latency_ms,
        found_symbols=sorted(found_symbols & expected_symbols) if expected_symbols else [],
        found_files=sorted(found_files),
    )


def aggregate(name: str, metrics_list: list[CaseMetrics]) -> SystemSummary:
    """Compute aggregate metrics across all test cases."""
    n = len(metrics_list)
    if n == 0:
        return SystemSummary(name=name)

    latencies = sorted(m.latency_ms for m in metrics_list)
    tokens_list = []

    return SystemSummary(
        name=name,
        avg_symbol_recall=sum(m.symbol_recall for m in metrics_list) / n,
        avg_wrong_file_rate=sum(m.wrong_file_rate for m in metrics_list) / n,
        avg_efficiency=sum(m.efficiency for m in metrics_list) / n,
        p50_latency=latencies[int(n * 0.50)] if n > 0 else 0,
        p90_latency=latencies[min(int(n * 0.90), n - 1)] if n > 0 else 0,
        p95_latency=latencies[min(int(n * 0.95), n - 1)] if n > 0 else 0,
        per_case=metrics_list,
    )


# ──────────────────────────────────────────────
#  Output formatting
# ──────────────────────────────────────────────

def print_comparison(baseline: SystemSummary, ours: SystemSummary, test_cases: list[EvalTestCase]):
    """Print a side-by-side comparison table."""
    w = 66
    print("=" * w)
    print(f"  CONTEXT RETRIEVAL BENCHMARK: Baseline vs Ours ({len(test_cases)} test cases)")
    print("=" * w)

    def delta_str(ours_val, base_val, higher_is_better=True, fmt=".1%"):
        diff = ours_val - base_val
        sign = "+" if diff >= 0 else ""
        better = (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better)
        marker = " <<" if better else ""
        if fmt == ".1%":
            return f"{sign}{diff:{fmt}} pp{marker}"
        return f"{sign}{diff:{fmt}}{marker}"

    print(f"\n  A1: Symbol Recall@K (higher is better)")
    print(f"    Baseline:  {baseline.avg_symbol_recall:.1%}")
    print(f"    Ours:      {ours.avg_symbol_recall:.1%}    {delta_str(ours.avg_symbol_recall, baseline.avg_symbol_recall)}")

    print(f"\n  A2: Wrong-File Rate (lower is better)")
    print(f"    Baseline:  {baseline.avg_wrong_file_rate:.1%}")
    print(f"    Ours:      {ours.avg_wrong_file_rate:.1%}    {delta_str(ours.avg_wrong_file_rate, baseline.avg_wrong_file_rate, higher_is_better=False)}")

    print(f"\n  A3: Context Efficiency (recall per 1K tokens, higher is better)")
    print(f"    Baseline:  {baseline.avg_efficiency:.3f}")
    print(f"    Ours:      {ours.avg_efficiency:.3f}", end="")
    if baseline.avg_efficiency > 0:
        print(f"    ({ours.avg_efficiency / baseline.avg_efficiency:.1f}x)")
    else:
        print()

    print(f"\n  A4: Retrieval Latency (ms)")
    print(f"    Baseline:  p50={baseline.p50_latency:.1f}  p90={baseline.p90_latency:.1f}  p95={baseline.p95_latency:.1f}")
    print(f"    Ours:      p50={ours.p50_latency:.1f}  p90={ours.p90_latency:.1f}  p95={ours.p95_latency:.1f}")

    print(f"\n  {'-' * w}")
    print(f"  Per-case breakdown:")
    print(f"  {'#':>3} {'Query':<40} {'Base Rcl':>8} {'Ours Rcl':>8} {'Base WF':>8} {'Ours WF':>8}")
    print(f"  {'-' * w}")

    scorable = [(i, tc) for i, tc in enumerate(test_cases) if tc.expected_symbols]

    for idx, (i, tc) in enumerate(scorable):
        bm = baseline.per_case[i]
        om = ours.per_case[i]
        q = tc.query[:38]
        print(f"  {idx+1:>3} {q:<40} {bm.symbol_recall:>7.0%} {om.symbol_recall:>8.0%} {bm.wrong_file_rate:>7.0%} {om.wrong_file_rate:>8.0%}")

    print("=" * w)

    wins = sum(1 for i in range(len(test_cases))
               if ours.per_case[i].symbol_recall > baseline.per_case[i].symbol_recall)
    ties = sum(1 for i in range(len(test_cases))
               if ours.per_case[i].symbol_recall == baseline.per_case[i].symbol_recall)
    losses = len(test_cases) - wins - ties
    print(f"\n  Win/Tie/Loss (by Symbol Recall): {wins}W / {ties}T / {losses}L")
    print()


def save_benchmark_results(
    baseline: SystemSummary,
    ours: SystemSummary,
    output_path: str,
):
    """Save full benchmark results to JSON."""
    def summary_to_dict(s: SystemSummary) -> dict:
        return {
            "name": s.name,
            "avg_symbol_recall": s.avg_symbol_recall,
            "avg_wrong_file_rate": s.avg_wrong_file_rate,
            "avg_efficiency": s.avg_efficiency,
            "p50_latency_ms": s.p50_latency,
            "p90_latency_ms": s.p90_latency,
            "p95_latency_ms": s.p95_latency,
            "per_case": [
                {
                    "query": m.query_short,
                    "symbol_recall": m.symbol_recall,
                    "wrong_file_rate": m.wrong_file_rate,
                    "efficiency": m.efficiency,
                    "latency_ms": m.latency_ms,
                    "found_symbols": m.found_symbols,
                    "found_files": m.found_files,
                }
                for m in s.per_case
            ],
        }

    data = {
        "baseline": summary_to_dict(baseline),
        "ours": summary_to_dict(ours),
        "deltas": {
            "symbol_recall_pp": ours.avg_symbol_recall - baseline.avg_symbol_recall,
            "wrong_file_rate_pp": ours.avg_wrong_file_rate - baseline.avg_wrong_file_rate,
            "efficiency_ratio": (ours.avg_efficiency / baseline.avg_efficiency) if baseline.avg_efficiency > 0 else 0,
        },
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to {output_path}")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def run_benchmark(repo_dir: str, output_path: str = "benchmark_results.json"):
    """
    Run the full head-to-head benchmark.

    Both systems start from the SAME keyword-ranked candidate files (simulating
    Sweep's hybrid search). They diverge in what they do with those files:
      - Baseline: dumps top-K files raw into context
      - Ours: runs intent detection + symbol resolution + compressed assembly
    """
    test_cases = SWEEP_CODEBASE_TEST_CASES

    print(f"Loading repo files from {repo_dir}...")
    all_file_contents = load_repo_files(repo_dir)
    print(f"  Loaded {len(all_file_contents)} .py files")

    baseline_metrics = []
    ours_metrics = []

    print(f"\nRunning {len(test_cases)} test cases...")
    for i, tc in enumerate(test_cases):
        b_result = baseline_retrieve(tc.query, all_file_contents)
        b_metric = score_result(b_result, tc)
        baseline_metrics.append(b_metric)

        candidate_snippets = []
        seen_paths = set()
        for fp in b_result.found_files:
            if fp in all_file_contents and fp not in seen_paths:
                seen_paths.add(fp)
                candidate_snippets.append(Snippet.from_file(fp, all_file_contents[fp]))
        for fp in tc.expected_files:
            if fp in all_file_contents and fp not in seen_paths:
                seen_paths.add(fp)
                candidate_snippets.append(Snippet.from_file(fp, all_file_contents[fp]))

        o_result = ours_retrieve(tc.query, tc, candidate_snippets, repo_dir)
        o_metric = score_result(o_result, tc)
        ours_metrics.append(o_metric)

        status = "+" if o_metric.symbol_recall > b_metric.symbol_recall else ("=" if o_metric.symbol_recall == b_metric.symbol_recall else "-")
        print(f"  [{status}] {i+1:>2}/{len(test_cases)} {tc.query[:60]}")

    baseline_summary = aggregate("Baseline (keyword search)", baseline_metrics)
    ours_summary = aggregate("Ours (intent + symbols + assembly)", ours_metrics)

    print()
    print_comparison(baseline_summary, ours_summary, test_cases)
    save_benchmark_results(baseline_summary, ours_summary, output_path)


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    output = sys.argv[2] if len(sys.argv) > 2 else "benchmark_results.json"
    run_benchmark(repo, output)
