"""End-to-end test of the new context retrieval pipeline."""
import os
import time

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


# ============================================================
print("=" * 60)
print("  TEST 1: extract_definitions (tree-sitter parsing)")
print("=" * 60)
from sweepai.utils.code_validators import extract_definitions, DefinitionInfo

py_code = '''
class DatabaseClient:
    """A client for the database."""
    def connect(self, url: str) -> None:
        pass

    def query(self, sql: str) -> list:
        """Execute a SQL query."""
        return []

def helper_function(x, y):
    return x + y

API_KEY = "secret"
'''

defs = extract_definitions("test.py", py_code)
check("Returns list of DefinitionInfo", isinstance(defs, list) and all(isinstance(d, DefinitionInfo) for d in defs))
names = [d.name for d in defs]
check("Finds DatabaseClient class", "DatabaseClient" in names, f"got {names}")
check("Finds helper_function", "helper_function" in names, f"got {names}")
check("Finds connect method", "connect" in names, f"got {names}")
check("Finds query method", "query" in names, f"got {names}")

db_def = next((d for d in defs if d.name == "DatabaseClient"), None)
if db_def:
    check("DatabaseClient is kind=class", db_def.kind == "class", f"got {db_def.kind}")
    check("DatabaseClient has docstring", "client" in db_def.doc_summary.lower(), f"got '{db_def.doc_summary}'")
    check("DatabaseClient has members", len(db_def.members) > 0, f"got {db_def.members}")

helper_def = next((d for d in defs if d.name == "helper_function"), None)
if helper_def:
    check("helper_function is kind=function", helper_def.kind == "function", f"got {helper_def.kind}")
    check("helper_function has signature", "helper_function" in helper_def.signature, f"got '{helper_def.signature}'")

# Test on real codebase file
real_file = os.path.join(REPO_DIR, "sweepai", "core", "entities.py")
if os.path.isfile(real_file):
    real_code = open(real_file, "r", encoding="utf-8").read()
    real_defs = extract_definitions("sweepai/core/entities.py", real_code)
    real_names = [d.name for d in real_defs]
    check("Real file: finds Snippet class", "Snippet" in real_names, f"got {real_names[:10]}")
    check("Real file: finds FileChangeRequest", "FileChangeRequest" in real_names, f"got {real_names[:10]}")
    print(f"  (Found {len(real_defs)} definitions in entities.py)")

# ============================================================
print()
print("=" * 60)
print("  TEST 2: Intent Detection")
print("=" * 60)
from sweepai.core.intent_detector import detect_intent, INTENT_BUG_FIX, INTENT_TEST_WRITING, INTENT_REFACTOR, INTENT_IMPLEMENTATION, INTENT_USAGE_EXPLORATION, INTENT_DEFINITION_LOOKUP

cases = [
    ("fix the bug in parser.py where it crashes on empty input", INTENT_BUG_FIX),
    ("add unit tests for the UserService class", INTENT_TEST_WRITING),
    ("refactor the database module to use connection pooling", INTENT_REFACTOR),
    ("implement a new caching layer for API responses", INTENT_IMPLEMENTATION),
    ("how is the Snippet class used across the codebase?", INTENT_USAGE_EXPLORATION),
    ("where is RepoContextManager defined?", INTENT_DEFINITION_LOOKUP),
    ("File \"app.py\", line 42, in process\n  TypeError: expected str", INTENT_BUG_FIX),
    ("write tests for the search_index function", INTENT_TEST_WRITING),
    ("rename chunk_code to parse_file_chunks", INTENT_REFACTOR),
]

for query, expected in cases:
    intent = detect_intent(query, [], [])
    check(
        f"'{query[:50]}...' -> {expected}",
        intent.intent_type == expected,
        f"got {intent.intent_type} (conf={intent.confidence:.2f})"
    )

# Test symbol extraction
intent = detect_intent("fix bug in `DatabaseClient` and UserService", [], [])
check("Extracts backtick symbols", "DatabaseClient" in intent.target_symbols, f"got {intent.target_symbols}")
check("Extracts CamelCase symbols", "UserService" in intent.target_symbols, f"got {intent.target_symbols}")

# ============================================================
print()
print("=" * 60)
print("  TEST 3: Symbol Resolver")
print("=" * 60)
from sweepai.core.symbol_resolver import (
    _extract_symbol_names_from_query,
    _parse_python_imports,
    get_definition_cards_for_file,
    DefinitionCard,
)

# Test query symbol extraction
syms = _extract_symbol_names_from_query("fix the bug in `DatabaseClient` and check UserService")
check("Extracts DatabaseClient", "DatabaseClient" in syms, f"got {syms}")
check("Extracts UserService", "UserService" in syms, f"got {syms}")

# Test Python import parsing
import_code = """
from sweepai.core.entities import Snippet, Message
from sweepai.logn.cache import file_cache
import os
from . import sibling_module
from ..parent import ParentClass as PC
"""
imports = _parse_python_imports(import_code)
check("Parses 'from X import Y'", "Snippet" in imports, f"got {list(imports.keys())}")
check("Parses 'from X import Y' (Message)", "Message" in imports, f"got {list(imports.keys())}")
check("Parses 'import os'", "os" in imports, f"got {list(imports.keys())}")
check("Parses 'from . import'", "sibling_module" in imports, f"got {list(imports.keys())}")
check("Parses 'as' alias", "PC" in imports, f"got {list(imports.keys())}")

# Test definition cards from real file
cards = get_definition_cards_for_file("sweepai/core/entities.py", REPO_DIR)
card_names = [c.symbol_name for c in cards]
check("Gets cards from entities.py", len(cards) > 0, f"got {len(cards)} cards")
check("Cards include Snippet", "Snippet" in card_names, f"got {card_names[:10]}")

if cards:
    snippet_card = next((c for c in cards if c.symbol_name == "Snippet"), None)
    if snippet_card:
        check("Snippet card has fqn", "entities.py:Snippet" in snippet_card.fqn)
        check("Snippet card kind is class", snippet_card.kind == "class", f"got {snippet_card.kind}")
        compact = snippet_card.to_compact()
        check("to_compact() produces text", len(compact) > 20, f"got {len(compact)} chars")
        standard = snippet_card.to_standard()
        check("to_standard() is longer than compact", len(standard) >= len(compact))
        print(f"\n  --- Compact card for Snippet ---")
        print(f"  {compact.replace(chr(10), chr(10) + '  ')}")
        print()

# ============================================================
print()
print("=" * 60)
print("  TEST 4: Context Assembly (end-to-end)")
print("=" * 60)
from sweepai.core.context_assembler import assemble_context, get_context_stats
from sweepai.core.entities import Snippet

snippet = Snippet.from_file(
    "sweepai/core/entities.py",
    open(os.path.join(REPO_DIR, "sweepai", "core", "entities.py"), "r").read()
)

intent = detect_intent(
    "fix the bug in Snippet class where denotation property fails",
    ["sweepai/core/entities.py"],
    [snippet],
)

t0 = time.perf_counter()
ctx = assemble_context(
    intent=intent,
    definition_cards=cards[:5],
    top_snippets=[snippet],
    read_only_snippets=[],
    import_trees="entities.py -> diff.py -> search_and_replace.py",
)
elapsed_ms = (time.perf_counter() - t0) * 1000

stats = get_context_stats(ctx)
check("Context is non-empty", len(ctx) > 100, f"got {len(ctx)} chars")
check("Context has definitions section", "<definitions>" in ctx)
check("Context has relevant_code section", "<relevant_code>" in ctx)
check("Context has intent annotation", "BUG_FIX" in ctx)
check("Stats has total_tokens", stats["total_tokens"] > 0, f"got {stats}")
check("Stats has num_definitions", stats["num_definitions"] > 0, f"got {stats}")
check(f"Assembly took <100ms", elapsed_ms < 100, f"took {elapsed_ms:.1f}ms")
print(f"  Context stats: {stats}")
print(f"  Assembly latency: {elapsed_ms:.1f}ms")

# ============================================================
print()
print("=" * 60)
print("  TEST 5: Full Evaluation Framework")
print("=" * 60)
from sweepai.core.eval_context_retrieval import (
    evaluate_single, EvalTestCase, format_summary, evaluate_batch
)

tc = EvalTestCase(
    query="where is Snippet defined?",
    expected_files=["sweepai/core/entities.py"],
    expected_symbols=["Snippet"],
    expected_intent="DEFINITION_LOOKUP",
)
result = evaluate_single(tc, [snippet], REPO_DIR)
check("Eval detects intent", result.detected_intent != "")
check("Eval finds definition cards", len(result.definition_cards_found) > 0, f"got {result.definition_cards_found}")
check("Eval measures latency", result.retrieval_latency_ms > 0, f"got {result.retrieval_latency_ms:.1f}ms")
check("Eval computes recall", result.definition_hit_rate > 0, f"got {result.definition_hit_rate:.1%}")
check("No eval errors", result.error == "", f"error: {result.error}")

# ============================================================
print()
print("=" * 60)
if FAIL == 0:
    print(f"  ALL {PASS} TESTS PASSED")
else:
    print(f"  {PASS} passed, {FAIL} FAILED")
print("=" * 60)
