"""
Evaluation framework for context retrieval quality.

Measures precision, recall, definition hit rate, and latency of the
new intent-aware symbol resolution pipeline vs the old hybrid-only approach.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from loguru import logger

from sweepai.core.entities import Snippet
from sweepai.core.intent_detector import detect_intent, RetrievalIntent
from sweepai.core.symbol_resolver import (
    DefinitionCard,
    resolve_symbols_from_query,
    get_definition_cards_for_file,
)
from sweepai.core.context_assembler import assemble_context, get_context_stats


@dataclass
class EvalTestCase:
    """A single evaluation test case."""
    query: str
    expected_files: list[str]
    expected_symbols: list[str]
    expected_intent: Optional[str] = None
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "EvalTestCase":
        return cls(
            query=d["query"],
            expected_files=d.get("expected_files", []),
            expected_symbols=d.get("expected_symbols", []),
            expected_intent=d.get("expected_intent"),
            description=d.get("description", ""),
        )


@dataclass
class EvalResult:
    """Results from evaluating a single test case."""
    test_case: EvalTestCase
    detected_intent: str = ""
    intent_correct: bool = False
    intent_confidence: float = 0.0

    definition_cards_found: list[str] = field(default_factory=list)
    definition_hit_rate: float = 0.0

    files_in_context: list[str] = field(default_factory=list)
    context_precision: float = 0.0
    context_recall: float = 0.0
    context_f1: float = 0.0

    symbols_in_context: list[str] = field(default_factory=list)
    symbol_precision: float = 0.0
    symbol_recall: float = 0.0

    retrieval_latency_ms: float = 0.0
    total_context_tokens: int = 0
    error: str = ""


@dataclass
class EvalSummary:
    """Aggregate evaluation metrics across all test cases."""
    total_cases: int = 0
    intent_accuracy: float = 0.0
    avg_definition_hit_rate: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_context_f1: float = 0.0
    avg_symbol_precision: float = 0.0
    avg_symbol_recall: float = 0.0
    avg_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    avg_context_tokens: float = 0.0
    errors: int = 0


def _compute_precision_recall(predicted: set, expected: set) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 between two sets."""
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    if not expected:
        return 0.0, 1.0, 0.0

    true_positives = len(predicted & expected)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_single(
    test_case: EvalTestCase,
    snippets: list[Snippet],
    repo_dir: str,
    import_trees: str = "",
) -> EvalResult:
    """Evaluate the new retrieval pipeline on a single test case."""
    result = EvalResult(test_case=test_case)

    try:
        start = time.perf_counter()

        intent = detect_intent(
            test_case.query,
            test_case.expected_files,
            snippets,
        )
        result.detected_intent = intent.intent_type
        result.intent_confidence = intent.confidence
        if test_case.expected_intent:
            result.intent_correct = intent.intent_type == test_case.expected_intent

        definition_cards = resolve_symbols_from_query(
            test_case.query,
            snippets,
            repo_dir,
        )
        result.definition_cards_found = [c.symbol_name for c in definition_cards]

        assembled = assemble_context(
            intent=intent,
            definition_cards=definition_cards,
            top_snippets=snippets,
            read_only_snippets=[],
            import_trees=import_trees,
        )

        elapsed = time.perf_counter() - start
        result.retrieval_latency_ms = elapsed * 1000

        stats = get_context_stats(assembled)
        result.total_context_tokens = stats.get("total_tokens", 0)

        result.files_in_context = list({c.file_path for c in definition_cards})
        expected_files_set = set(test_case.expected_files)
        found_files_set = set(result.files_in_context)
        p, r, f1 = _compute_precision_recall(found_files_set, expected_files_set)
        result.context_precision = p
        result.context_recall = r
        result.context_f1 = f1

        result.symbols_in_context = [c.symbol_name for c in definition_cards]
        expected_symbols_set = set(test_case.expected_symbols)
        found_symbols_set = set(result.symbols_in_context)
        sp, sr, _ = _compute_precision_recall(found_symbols_set, expected_symbols_set)
        result.symbol_precision = sp
        result.symbol_recall = sr

        if test_case.expected_symbols:
            hits = sum(1 for s in test_case.expected_symbols if s in found_symbols_set)
            result.definition_hit_rate = hits / len(test_case.expected_symbols)
        else:
            result.definition_hit_rate = 1.0

    except Exception as e:
        result.error = str(e)
        logger.error(f"Eval error for '{test_case.query[:50]}': {e}")

    return result


def evaluate_batch(
    test_cases: list[EvalTestCase],
    snippets: list[Snippet],
    repo_dir: str,
    import_trees: str = "",
) -> tuple[list[EvalResult], EvalSummary]:
    """Run evaluation across all test cases and compute aggregate metrics."""
    results = []
    for tc in test_cases:
        result = evaluate_single(tc, snippets, repo_dir, import_trees)
        results.append(result)

    summary = EvalSummary(total_cases=len(results))
    if not results:
        return results, summary

    intent_cases = [r for r in results if r.test_case.expected_intent]
    if intent_cases:
        summary.intent_accuracy = sum(1 for r in intent_cases if r.intent_correct) / len(intent_cases)

    valid = [r for r in results if not r.error]
    summary.errors = len(results) - len(valid)

    if valid:
        summary.avg_definition_hit_rate = sum(r.definition_hit_rate for r in valid) / len(valid)
        summary.avg_context_precision = sum(r.context_precision for r in valid) / len(valid)
        summary.avg_context_recall = sum(r.context_recall for r in valid) / len(valid)
        summary.avg_context_f1 = sum(r.context_f1 for r in valid) / len(valid)
        summary.avg_symbol_precision = sum(r.symbol_precision for r in valid) / len(valid)
        summary.avg_symbol_recall = sum(r.symbol_recall for r in valid) / len(valid)
        summary.avg_latency_ms = sum(r.retrieval_latency_ms for r in valid) / len(valid)
        summary.avg_context_tokens = sum(r.total_context_tokens for r in valid) / len(valid)

        latencies = sorted(r.retrieval_latency_ms for r in valid)
        p90_idx = int(len(latencies) * 0.9)
        summary.p90_latency_ms = latencies[min(p90_idx, len(latencies) - 1)]

    return results, summary


def format_summary(summary: EvalSummary) -> str:
    """Format evaluation summary as a readable string."""
    lines = [
        "=" * 60,
        "  Context Retrieval Evaluation Summary",
        "=" * 60,
        f"  Total test cases:       {summary.total_cases}",
        f"  Errors:                 {summary.errors}",
        "",
        "  Intent Detection:",
        f"    Accuracy:             {summary.intent_accuracy:.1%}",
        "",
        "  Definition Resolution:",
        f"    Avg hit rate:         {summary.avg_definition_hit_rate:.1%}",
        f"    Avg symbol precision: {summary.avg_symbol_precision:.1%}",
        f"    Avg symbol recall:    {summary.avg_symbol_recall:.1%}",
        "",
        "  Context Quality:",
        f"    Avg file precision:   {summary.avg_context_precision:.1%}",
        f"    Avg file recall:      {summary.avg_context_recall:.1%}",
        f"    Avg F1:               {summary.avg_context_f1:.1%}",
        "",
        "  Performance:",
        f"    Avg latency:          {summary.avg_latency_ms:.1f} ms",
        f"    p90 latency:          {summary.p90_latency_ms:.1f} ms",
        f"    Avg context tokens:   {summary.avg_context_tokens:.0f}",
        "=" * 60,
    ]
    return "\n".join(lines)


def save_results(
    results: list[EvalResult],
    summary: EvalSummary,
    output_path: str,
) -> None:
    """Save evaluation results to a JSON file."""
    data = {
        "summary": asdict(summary),
        "results": [
            {
                "query": r.test_case.query,
                "description": r.test_case.description,
                "detected_intent": r.detected_intent,
                "intent_correct": r.intent_correct,
                "definition_cards_found": r.definition_cards_found,
                "definition_hit_rate": r.definition_hit_rate,
                "context_precision": r.context_precision,
                "context_recall": r.context_recall,
                "context_f1": r.context_f1,
                "symbol_precision": r.symbol_precision,
                "symbol_recall": r.symbol_recall,
                "retrieval_latency_ms": r.retrieval_latency_ms,
                "total_context_tokens": r.total_context_tokens,
                "error": r.error,
            }
            for r in results
        ],
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved to {output_path}")


SWEEP_CODEBASE_TEST_CASES = [
    EvalTestCase(
        query="fix the bug in context_pruning.py where get_relevant_context returns early without running the DFS",
        expected_files=["sweepai/core/context_pruning.py"],
        expected_symbols=["get_relevant_context", "context_dfs", "RepoContextManager"],
        expected_intent="BUG_FIX",
        description="Bug fix targeting the disabled DFS in context pruning",
    ),
    EvalTestCase(
        query="add unit tests for the `ChatGPT` class in sweep",
        expected_files=["sweepai/core/chat.py"],
        expected_symbols=["ChatGPT"],
        expected_intent="TEST_WRITING",
        description="Test writing for the ChatGPT wrapper class",
    ),
    EvalTestCase(
        query="refactor extract_definitions to return structured data instead of printing",
        expected_files=["sweepai/utils/code_validators.py"],
        expected_symbols=["extract_definitions"],
        expected_intent="REFACTOR",
        description="Refactoring the definition extraction function",
    ),
    EvalTestCase(
        query="implement a new symbol resolver that uses tree-sitter to find definitions",
        expected_files=["sweepai/utils/code_validators.py"],
        expected_symbols=["get_parser", "extract_definitions"],
        expected_intent="IMPLEMENTATION",
        description="New feature implementation using existing tree-sitter infrastructure",
    ),
    EvalTestCase(
        query="how is the Snippet class used across the codebase?",
        expected_files=["sweepai/core/entities.py"],
        expected_symbols=["Snippet"],
        expected_intent="USAGE_EXPLORATION",
        description="Exploring usages of the Snippet data model",
    ),
    EvalTestCase(
        query="where is `RepoContextManager` defined and what fields does it have?",
        expected_files=["sweepai/core/context_pruning.py"],
        expected_symbols=["RepoContextManager"],
        expected_intent="DEFINITION_LOOKUP",
        description="Looking up a specific class definition",
    ),
    EvalTestCase(
        query="fix TypeError in multi_get_top_k_snippets when snippets list is empty",
        expected_files=["sweepai/utils/ticket_utils.py"],
        expected_symbols=["multi_get_top_k_snippets"],
        expected_intent="BUG_FIX",
        description="Bug fix with a specific function name mentioned",
    ),
    EvalTestCase(
        query="add caching to the vector search pipeline using the existing file_cache decorator",
        expected_files=["sweepai/core/vector_db.py", "sweepai/logn/cache.py"],
        expected_symbols=["file_cache", "compute_vector_search_scores"],
        expected_intent="IMPLEMENTATION",
        description="Adding a feature that spans two modules",
    ),
    EvalTestCase(
        query="rename the chunk_code function to chunk_file_into_snippets for clarity",
        expected_files=["sweepai/utils/code_validators.py"],
        expected_symbols=["chunk_code"],
        expected_intent="REFACTOR",
        description="Simple rename refactoring",
    ),
    EvalTestCase(
        query="write tests for the import tree building logic in context_pruning",
        expected_files=["sweepai/core/context_pruning.py"],
        expected_symbols=["build_import_trees", "build_full_hierarchy"],
        expected_intent="TEST_WRITING",
        description="Test writing for graph-related functions",
    ),
    EvalTestCase(
        query="implement intent detection for context retrieval queries",
        expected_files=[],
        expected_symbols=[],
        expected_intent="IMPLEMENTATION",
        description="New module creation (no expected files in current codebase)",
    ),
    EvalTestCase(
        query="the lexical search returns wrong results when query contains special characters. "
              "File \"sweepai/core/lexical_search.py\", line 170, in search_index",
        expected_files=["sweepai/core/lexical_search.py"],
        expected_symbols=["search_index", "CustomIndex"],
        expected_intent="BUG_FIX",
        description="Bug fix with a stack trace included",
    ),
    EvalTestCase(
        query="find all callers of get_relevant_context across the codebase",
        expected_files=["sweepai/core/context_pruning.py"],
        expected_symbols=["get_relevant_context"],
        expected_intent="USAGE_EXPLORATION",
        description="Searching for callers of a specific function",
    ),
    EvalTestCase(
        query="what is the SweepConfig class and where is it defined?",
        expected_files=["sweepai/config/client.py"],
        expected_symbols=["SweepConfig"],
        expected_intent="DEFINITION_LOOKUP",
        description="Definition lookup for a config class",
    ),
    EvalTestCase(
        query="add a new endpoint to handle webhook events for pull request reviews",
        expected_files=[],
        expected_symbols=[],
        expected_intent="IMPLEMENTATION",
        description="Feature implementation with no specific existing symbols",
    ),
    # --- 15 NEW TEST CASES for benchmark comparison ---
    EvalTestCase(
        query="fix the ClonedRepo class so it handles permission errors when cloning private repos",
        expected_files=["sweepai/utils/github_utils.py"],
        expected_symbols=["ClonedRepo"],
        expected_intent="BUG_FIX",
        description="Multi-file: ClonedRepo is defined in github_utils but imported in 25+ files",
    ),
    EvalTestCase(
        query="the `file_cache` decorator doesn't invalidate when function source code changes",
        expected_files=["sweepai/logn/cache.py"],
        expected_symbols=["file_cache", "recursive_hash"],
        expected_intent="BUG_FIX",
        description="Multi-file: file_cache is defined in cache.py but re-exported via __init__ and used in 18+ files",
    ),
    EvalTestCase(
        query="refactor the ChatGPT class to support streaming responses natively",
        expected_files=["sweepai/core/chat.py"],
        expected_symbols=["ChatGPT", "continuous_llm_calls"],
        expected_intent="REFACTOR",
        description="Multi-file: ChatGPT is defined in chat.py but imported in 30+ files",
    ),
    EvalTestCase(
        query="the on_ticket handler is failing because fetch_relevant_files returns None",
        expected_files=["sweepai/handlers/on_ticket.py", "sweepai/utils/ticket_utils.py"],
        expected_symbols=["fetch_relevant_files"],
        expected_intent="BUG_FIX",
        description="Cross-module: handler calls utility function across module boundary",
    ),
    EvalTestCase(
        query="modify the search agent to use a different strategy for finding code snippets",
        expected_files=["sweepai/agents/search_agent.py"],
        expected_symbols=["search"],
        expected_intent="REFACTOR",
        description="Ambiguous: 'search' appears as a function in 8+ files but we need the one in search_agent",
    ),
    EvalTestCase(
        query="the `parse_function_calls` in the chat module returns incorrect results for nested JSON",
        expected_files=["sweepai/core/chat.py"],
        expected_symbols=["parse_function_calls"],
        expected_intent="BUG_FIX",
        description="Ambiguous: parse_function_calls exists in both chat.py and agent_utils.py",
    ),
    EvalTestCase(
        query="add validation for file paths in the modify agent before applying changes",
        expected_files=["sweepai/agents/modify_utils.py", "sweepai/agents/modify.py"],
        expected_symbols=["validate_and_parse_function_call", "handle_function_call"],
        expected_intent="IMPLEMENTATION",
        description="Ambiguous: validate_and_parse_function_call exists in both modify_utils.py and agent_utils.py",
    ),
    EvalTestCase(
        query="improve the `check_syntax` function to support more languages",
        expected_files=["sweepai/utils/code_validators.py"],
        expected_symbols=["check_syntax", "get_parser"],
        expected_intent="IMPLEMENTATION",
        description="Cross-module: check_syntax uses get_parser which depends on tree-sitter",
    ),
    EvalTestCase(
        query="the sweep bot is not correctly parsing file change requests from LLM output",
        expected_files=["sweepai/core/sweep_bot.py", "sweepai/core/entities.py"],
        expected_symbols=["FileChangeRequest", "parse_fcr"],
        expected_intent="BUG_FIX",
        description="Deep chain: sweep_bot imports from entities which imports from diff which imports from search_and_replace",
    ),
    EvalTestCase(
        query="write tests for the `compute_vector_search_scores` function in the lexical search module",
        expected_files=["sweepai/core/lexical_search.py"],
        expected_symbols=["compute_vector_search_scores"],
        expected_intent="TEST_WRITING",
        description="Ambiguous: the function is in lexical_search.py but also re-used in ticket_utils.py",
    ),
    EvalTestCase(
        query="how does the modify agent decide which files to edit and in what order?",
        expected_files=["sweepai/agents/modify.py"],
        expected_symbols=["modify"],
        expected_intent="DEFINITION_LOOKUP",
        description="Cross-module: understanding the modify agent requires following its imports to modify_utils",
    ),
    EvalTestCase(
        query="the DirectoryTree class is rendering incorrectly for deeply nested directories",
        expected_files=["sweepai/utils/tree_utils.py"],
        expected_symbols=["DirectoryTree"],
        expected_intent="BUG_FIX",
        description="Symbol defined in tree_utils, imported in context_pruning and elsewhere",
    ),
    EvalTestCase(
        query="make the code review process faster by parallelizing the annotation step",
        expected_files=["sweepai/core/review_utils.py"],
        expected_symbols=["review_pr"],
        expected_intent="IMPLEMENTATION",
        description="No backtick symbols: requires understanding 'code review' maps to review_utils",
    ),
    EvalTestCase(
        query="users are reporting that the GitHub Actions log parsing misses error lines",
        expected_files=["sweepai/handlers/on_failing_github_actions.py"],
        expected_symbols=[],
        expected_intent="BUG_FIX",
        description="No symbols: pure natural language, file must be inferred from 'GitHub Actions'",
    ),
    EvalTestCase(
        query="update the tokenize_code function to handle multi-line strings properly",
        expected_files=["sweepai/core/lexical_search.py"],
        expected_symbols=["tokenize_code"],
        expected_intent="BUG_FIX",
        description="Specific function: tokenize_code is only defined in lexical_search but the name is generic",
    ),
]


def run_self_eval(repo_dir: str, output_path: str = "eval_results.json") -> EvalSummary:
    """
    Run the evaluation suite against the Sweep codebase itself.
    This uses the test cases defined above to measure retrieval quality.
    """
    snippets = []
    for tc in SWEEP_CODEBASE_TEST_CASES:
        for fp in tc.expected_files:
            full = os.path.join(repo_dir, fp)
            if os.path.isfile(full):
                try:
                    content = open(full, "r", encoding="utf-8", errors="replace").read()
                    snippets.append(Snippet.from_file(fp, content))
                except Exception:
                    pass

    seen = set()
    unique_snippets = []
    for s in snippets:
        if s.file_path not in seen:
            seen.add(s.file_path)
            unique_snippets.append(s)

    results, summary = evaluate_batch(
        SWEEP_CODEBASE_TEST_CASES,
        unique_snippets,
        repo_dir,
    )

    print(format_summary(summary))
    save_results(results, summary, output_path)
    return summary


if __name__ == "__main__":
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    output = sys.argv[2] if len(sys.argv) > 2 else "eval_results.json"
    run_self_eval(repo, output)
