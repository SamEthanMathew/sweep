"""
Rule-based intent detection for context retrieval.

Classifies user queries into retrieval intents that determine what kind of
context to prioritize (definitions, usages, tests, callers, etc.).
Designed to be deterministic and fast (< 1ms) -- no LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from sweepai.core.entities import Snippet


INTENT_DEFINITION_LOOKUP = "DEFINITION_LOOKUP"
INTENT_USAGE_EXPLORATION = "USAGE_EXPLORATION"
INTENT_IMPLEMENTATION = "IMPLEMENTATION"
INTENT_BUG_FIX = "BUG_FIX"
INTENT_REFACTOR = "REFACTOR"
INTENT_TEST_WRITING = "TEST_WRITING"

BUDGET_PROFILES: dict[str, dict[str, float]] = {
    INTENT_DEFINITION_LOOKUP: {
        "definitions": 0.50, "snippets": 0.30, "imports": 0.10, "tests": 0.10, "callers": 0.00,
    },
    INTENT_USAGE_EXPLORATION: {
        "definitions": 0.20, "snippets": 0.10, "imports": 0.05, "tests": 0.00, "callers": 0.65,
    },
    INTENT_IMPLEMENTATION: {
        "definitions": 0.40, "snippets": 0.35, "imports": 0.15, "tests": 0.10, "callers": 0.00,
    },
    INTENT_BUG_FIX: {
        "definitions": 0.30, "snippets": 0.25, "imports": 0.10, "tests": 0.20, "callers": 0.15,
    },
    INTENT_REFACTOR: {
        "definitions": 0.25, "snippets": 0.20, "imports": 0.10, "tests": 0.15, "callers": 0.30,
    },
    INTENT_TEST_WRITING: {
        "definitions": 0.40, "snippets": 0.15, "imports": 0.05, "tests": 0.40, "callers": 0.00,
    },
}


@dataclass
class RetrievalIntent:
    """The classified intent of a user query for context retrieval."""
    intent_type: str
    target_symbols: list[str] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    confidence: float = 0.0
    is_test_context: bool = False

    @property
    def budget_profile(self) -> dict[str, float]:
        return BUDGET_PROFILES.get(self.intent_type, BUDGET_PROFILES[INTENT_IMPLEMENTATION])


_BUG_FIX_PATTERNS = [
    re.compile(r"\b(fix|bug|error|issue|broken|crash|fail|exception|traceback|stacktrace)\b", re.I),
    re.compile(r"(TypeError|ValueError|KeyError|AttributeError|ImportError|RuntimeError|IndexError)", re.I),
    re.compile(r"File \"[^\"]+\", line \d+"),  # Python traceback
    re.compile(r"at .+\(.+:\d+:\d+\)"),  # JS/TS stack trace
]

_REFACTOR_PATTERNS = [
    re.compile(r"\b(refactor|rename|move|restructure|reorganize|extract|consolidate|simplify|clean\s*up)\b", re.I),
    re.compile(r"\b(replace|migrate|deprecat|remov)\b.*\b(with|to|from)\b", re.I),
]

_TEST_PATTERNS = [
    re.compile(r"\b(test|tests|testing|spec|unittest|pytest|jest|mocha)\b", re.I),
    re.compile(r"\bwrite\s+(unit\s+)?tests?\b", re.I),
    re.compile(r"\badd\s+(unit\s+)?tests?\b", re.I),
]

_IMPLEMENTATION_PATTERNS = [
    re.compile(r"\b(implement|create|build|introduce|set\s*up|develop)\b", re.I),
    re.compile(r"\b(add|make|write)\s+(a\s+|new\s+)?(feature|endpoint|handler|service|component|page|route|api|module|layer)\b", re.I),
]

_USAGE_PATTERNS = [
    re.compile(r"\bhow\s+(is|are|does|do)\s+.{1,40}\s+(used|called|invoked|referenced)\b", re.I),
    re.compile(r"\b(find|show|list)\s+(all\s+)?(usages?|callers?|references?|occurrences?)\b", re.I),
    re.compile(r"\bwhere\s+(is|are|does|do)\s+.{1,40}\s+(used|called)\b", re.I),
    re.compile(r"\b(who|what)\s+(calls?|uses?|invokes?)\b", re.I),
    re.compile(r"\bused\s+(across|throughout|in)\b", re.I),
]

_DEFINITION_PATTERNS = [
    re.compile(r"\b(what|where)\s+(is|are)\s+(the\s+)?(definition|implementation|source|code)\s+(of|for)\b", re.I),
    re.compile(r"\b(find|show|look\s*up|locate)\s+(the\s+)?(definition|implementation|class|function)\b", re.I),
    re.compile(r"\bunderstand\s+(how|what)\b", re.I),
    re.compile(r"\bwhere\s+(is|are)\s+\w+\s+defined\b", re.I),
    re.compile(r"\b(what|where)\s+(is|are)\s+\w+\b(?!.*\b(used|called))", re.I),
]

TEST_FILE_PATTERNS = [
    re.compile(r"(^|/)tests?/"),
    re.compile(r"(^|/)test_\w+"),
    re.compile(r"_test\.\w+$"),
    re.compile(r"\.test\.\w+$"),
    re.compile(r"\.spec\.\w+$"),
    re.compile(r"(^|/)__tests__/"),
    re.compile(r"(^|/)specs?/"),
]


def _extract_symbols_from_query(query: str) -> list[str]:
    """Extract likely symbol names from a query."""
    symbols = []
    symbols.extend(re.findall(r"`(\w+)`", query))
    symbols.extend(re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", query))
    symbols.extend(re.findall(r"\b([A-Z][a-zA-Z0-9]{2,})\b", query))
    symbols.extend(re.findall(r"\b([a-z]\w*_\w+)\b", query))

    noise = {
        "None", "True", "False", "The", "This", "That", "What", "Where",
        "How", "When", "Which", "File", "Line", "Code", "Error", "Bug",
        "Fix", "Add", "New", "All", "Any", "Set", "Get", "Let", "Var",
    }
    seen = set()
    result = []
    for s in symbols:
        if s not in seen and s not in noise and len(s) > 1:
            seen.add(s)
            result.append(s)
    return result


def _is_test_file(path: str) -> bool:
    """Check if a file path looks like a test file."""
    return any(p.search(path) for p in TEST_FILE_PATTERNS)


def _score_patterns(query: str, patterns: list[re.Pattern]) -> float:
    """Score how well a query matches a set of patterns (0.0 - 1.0)."""
    matches = sum(1 for p in patterns if p.search(query))
    return min(matches / max(len(patterns) * 0.3, 1), 1.0)


def detect_intent(
    query: str,
    relevant_file_paths: list[str],
    top_snippets: list[Snippet],
) -> RetrievalIntent:
    """
    Classify a user query into a retrieval intent using rule-based analysis.

    Returns a RetrievalIntent with the detected intent type, target symbols,
    confidence score, and whether the context is test-related.
    """
    target_symbols = _extract_symbols_from_query(query)
    is_test_context = any(_is_test_file(p) for p in relevant_file_paths)

    scores = {
        INTENT_BUG_FIX: _score_patterns(query, _BUG_FIX_PATTERNS),
        INTENT_REFACTOR: _score_patterns(query, _REFACTOR_PATTERNS),
        INTENT_TEST_WRITING: _score_patterns(query, _TEST_PATTERNS),
        INTENT_USAGE_EXPLORATION: _score_patterns(query, _USAGE_PATTERNS),
        INTENT_DEFINITION_LOOKUP: _score_patterns(query, _DEFINITION_PATTERNS),
        INTENT_IMPLEMENTATION: _score_patterns(query, _IMPLEMENTATION_PATTERNS),
    }

    if is_test_context and scores[INTENT_TEST_WRITING] < 0.3:
        scores[INTENT_TEST_WRITING] += 0.3

    has_stack_trace = bool(
        re.search(r"File \"[^\"]+\", line \d+", query)
        or re.search(r"at .+\(.+:\d+:\d+\)", query)
        or re.search(r"Traceback \(most recent call last\)", query)
    )
    if has_stack_trace:
        scores[INTENT_BUG_FIX] = max(scores[INTENT_BUG_FIX], 0.9)

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    if best_score < 0.15:
        best_intent = INTENT_IMPLEMENTATION
        best_score = 0.3

    return RetrievalIntent(
        intent_type=best_intent,
        target_symbols=target_symbols,
        target_files=relevant_file_paths,
        confidence=best_score,
        is_test_context=is_test_context,
    )
