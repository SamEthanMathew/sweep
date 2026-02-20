"""
Token-budgeted context assembler.

Merges hybrid search results with definition cards based on the detected
intent, respecting a hard token cap and using compressed representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sweepai.core.entities import Snippet
from sweepai.core.intent_detector import BUDGET_PROFILES, INTENT_IMPLEMENTATION, RetrievalIntent
from sweepai.core.symbol_resolver import DefinitionCard

MAX_CONTEXT_TOKENS = 8000
TOKENS_PER_CHAR = 0.25  # rough estimate: 1 token ~ 4 chars


def _estimate_tokens(text: str) -> int:
    return int(len(text) * TOKENS_PER_CHAR)


@dataclass
class ContextBucket:
    """A category of context with a token budget and items."""
    name: str
    budget_tokens: int
    items: list[str] = field(default_factory=list)
    used_tokens: int = 0

    def try_add(self, text: str) -> bool:
        """Add text if it fits within budget. Returns True if added."""
        tokens = _estimate_tokens(text)
        if self.used_tokens + tokens <= self.budget_tokens:
            self.items.append(text)
            self.used_tokens += tokens
            return True
        return False

    @property
    def remaining(self) -> int:
        return max(0, self.budget_tokens - self.used_tokens)

    def render(self) -> str:
        if not self.items:
            return ""
        return "\n".join(self.items)


def _format_definition_card(card: DefinitionCard, use_standard: bool = False) -> str:
    """Format a definition card for inclusion in context."""
    if use_standard:
        return card.to_standard()
    return card.to_compact()


def _format_snippet_for_context(snippet: Snippet, max_lines: int = 50) -> str:
    """Format a snippet for context inclusion, trimming if needed."""
    lines = snippet.content.splitlines()
    start = max(0, snippet.start - 1)
    end = min(len(lines), snippet.end)

    if end - start > max_lines:
        end = start + max_lines

    body = "\n".join(lines[start:end])
    return f"<file path=\"{snippet.file_path}\" lines=\"{snippet.start}-{snippet.end}\">\n{body}\n</file>"


def _categorize_snippets(
    snippets: list[Snippet],
) -> dict[str, list[Snippet]]:
    """Split snippets into categories based on their type_name and file path."""
    categories: dict[str, list[Snippet]] = {
        "snippets": [],
        "tests": [],
        "callers": [],
    }
    for s in snippets:
        if s.type_name == "tests":
            categories["tests"].append(s)
        else:
            categories["snippets"].append(s)
    return categories


def assemble_context(
    intent: RetrievalIntent,
    definition_cards: list[DefinitionCard],
    top_snippets: list[Snippet],
    read_only_snippets: list[Snippet],
    import_trees: str,
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> str:
    """
    Assemble context from multiple sources respecting token budgets.

    The budget profile from the intent determines how tokens are allocated
    across categories (definitions, snippets, imports, tests, callers).
    Within each category, items are added greedily by relevance.
    """
    profile = intent.budget_profile

    buckets: dict[str, ContextBucket] = {}
    for category, fraction in profile.items():
        budget = int(max_tokens * fraction)
        buckets[category] = ContextBucket(name=category, budget_tokens=budget)

    if "definitions" not in buckets:
        buckets["definitions"] = ContextBucket(name="definitions", budget_tokens=0)
    if "snippets" not in buckets:
        buckets["snippets"] = ContextBucket(name="snippets", budget_tokens=0)
    if "imports" not in buckets:
        buckets["imports"] = ContextBucket(name="imports", budget_tokens=0)
    if "tests" not in buckets:
        buckets["tests"] = ContextBucket(name="tests", budget_tokens=0)

    sorted_cards = sorted(definition_cards, key=lambda c: c.relevance_score, reverse=True)
    def_bucket = buckets["definitions"]
    for card in sorted_cards:
        use_standard = def_bucket.remaining > 200
        text = _format_definition_card(card, use_standard=use_standard)
        if not def_bucket.try_add(text):
            compact = _format_definition_card(card, use_standard=False)
            def_bucket.try_add(compact)

    if import_trees and import_trees.strip():
        import_bucket = buckets["imports"]
        import_bucket.try_add(import_trees.strip())

    snippet_categories = _categorize_snippets(top_snippets + read_only_snippets)

    seen_paths: set[str] = set()
    snippet_bucket = buckets["snippets"]
    for snippet in snippet_categories["snippets"]:
        if snippet.file_path in seen_paths:
            continue
        seen_paths.add(snippet.file_path)
        text = _format_snippet_for_context(snippet)
        snippet_bucket.try_add(text)

    test_bucket = buckets["tests"]
    for snippet in snippet_categories["tests"]:
        if snippet.file_path in seen_paths:
            continue
        seen_paths.add(snippet.file_path)
        text = _format_snippet_for_context(snippet)
        test_bucket.try_add(text)

    _redistribute_unused_budget(buckets, sorted_cards, snippet_categories, seen_paths)

    sections = []

    if def_bucket.items:
        sections.append(
            "<definitions>\n" + def_bucket.render() + "\n</definitions>"
        )

    if buckets["imports"].items:
        sections.append(
            "<import_context>\n" + buckets["imports"].render() + "\n</import_context>"
        )

    if snippet_bucket.items:
        sections.append(
            "<relevant_code>\n" + snippet_bucket.render() + "\n</relevant_code>"
        )

    if test_bucket.items:
        sections.append(
            "<test_context>\n" + test_bucket.render() + "\n</test_context>"
        )

    if "callers" in buckets and buckets["callers"].items:
        sections.append(
            "<caller_context>\n" + buckets["callers"].render() + "\n</caller_context>"
        )

    header = f"<!-- intent: {intent.intent_type}, confidence: {intent.confidence:.2f} -->"
    return header + "\n" + "\n\n".join(sections)


def _redistribute_unused_budget(
    buckets: dict[str, ContextBucket],
    sorted_cards: list[DefinitionCard],
    snippet_categories: dict[str, list[Snippet]],
    seen_paths: set[str],
) -> None:
    """
    Redistribute unused budget from exhausted categories to others.
    Definitions overflow goes to snippets, and vice versa.
    """
    total_remaining = sum(b.remaining for b in buckets.values())
    if total_remaining < 100:
        return

    def_bucket = buckets["definitions"]
    snippet_bucket = buckets["snippets"]

    if def_bucket.remaining > 0 and snippet_bucket.remaining < 50:
        extra = def_bucket.remaining
        for snippet in snippet_categories["snippets"]:
            if snippet.file_path in seen_paths:
                continue
            text = _format_snippet_for_context(snippet)
            tokens = _estimate_tokens(text)
            if tokens <= extra:
                snippet_bucket.items.append(text)
                snippet_bucket.used_tokens += tokens
                seen_paths.add(snippet.file_path)
                extra -= tokens
            if extra < 50:
                break

    if snippet_bucket.remaining > 0 and def_bucket.remaining < 50:
        extra = snippet_bucket.remaining
        for card in sorted_cards:
            if any(card.to_compact() in item for item in def_bucket.items):
                continue
            text = card.to_standard()
            tokens = _estimate_tokens(text)
            if tokens <= extra:
                def_bucket.items.append(text)
                def_bucket.used_tokens += tokens
                extra -= tokens
            if extra < 50:
                break


def get_context_stats(assembled_context: str) -> dict:
    """Return statistics about an assembled context string for evaluation."""
    import re
    stats = {
        "total_tokens": _estimate_tokens(assembled_context),
        "total_chars": len(assembled_context),
        "num_definitions": len(re.findall(r"\[(?:class|function|method|variable)\]", assembled_context)),
        "num_files": len(re.findall(r'<file path="', assembled_context)),
        "has_tests": "<test_context>" in assembled_context,
        "has_callers": "<caller_context>" in assembled_context,
        "has_imports": "<import_context>" in assembled_context,
    }
    intent_match = re.search(r"intent: (\w+)", assembled_context)
    if intent_match:
        stats["intent"] = intent_match.group(1)
    return stats
