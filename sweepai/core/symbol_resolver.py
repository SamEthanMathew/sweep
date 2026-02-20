"""
Tree-sitter based symbol resolution engine.

Resolves identifiers in code to their definition sites by:
1. Parsing source files with tree-sitter to extract identifiers
2. Tracing import statements to find source files
3. Locating definition nodes in target files
4. Building structured DefinitionCard objects
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from sweepai.core.entities import Snippet
from sweepai.logn.cache import file_cache
from sweepai.utils.code_validators import (
    DefinitionInfo,
    extension_to_language,
    extract_definitions,
    get_parser,
)

import hashlib

from rapidfuzz import fuzz, process as rfprocess

MAX_CARDS_PER_QUERY = 20
MAX_MEMBERS_IN_CARD = 5
COMPACT_TOKEN_BUDGET = 60
STANDARD_TOKEN_BUDGET = 150
FUZZY_MATCH_THRESHOLD = 78
MAX_FUZZY_MATCHES = 3
MAX_FILE_SIZE_BYTES = 100_000
SAFETY_NET_MAX_CARDS = 5
SAFETY_NET_MAX_FILES = 3

_NL_STOPWORDS = {
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
    "your", "our", "what", "which", "who", "whom", "fix", "add", "new",
    "use", "using", "make", "across", "correctly", "properly", "currently",
    "implement", "improve", "support", "handle",
    "module", "code", "logic", "step", "process",
}

_VERB_STEMS = {
    "build", "get", "set", "create", "parse", "make", "find", "compute",
    "check", "handle", "update", "validate", "fetch", "resolve", "detect",
    "extract", "format", "render", "review", "modify", "search", "add",
    "remove", "delete", "write", "read", "load", "save", "send", "run",
    "tokenize", "serialize", "deserialize", "encode", "decode", "convert",
    "transform", "process", "execute", "apply", "merge", "split", "join",
    "filter", "sort", "map", "reduce", "scan", "walk", "traverse", "visit",
}

_GERUND_TO_STEM: dict[str, str] = {}
for _v in _VERB_STEMS:
    if _v.endswith("e"):
        _GERUND_TO_STEM[_v[:-1] + "ing"] = _v
    elif len(_v) > 2 and _v[-1] not in "aeiou" and _v[-2] in "aeiou":
        _GERUND_TO_STEM[_v + _v[-1] + "ing"] = _v
    else:
        _GERUND_TO_STEM[_v + "ing"] = _v


@dataclass
class DefinitionCard:
    """Structured, compressed representation of a code symbol's definition."""
    symbol_name: str
    fqn: str
    kind: str  # "class" | "function" | "method" | "variable"
    signature: str
    file_path: str
    start_line: int
    end_line: int
    doc_summary: str = ""
    parent_class: Optional[str] = None
    members: list[str] = field(default_factory=list)
    relevance_score: float = 0.0

    @property
    def token_count_compact(self) -> int:
        text = self.to_compact()
        return len(text.split()) + len(text) // 4

    def to_compact(self) -> str:
        """~50 token representation: signature + kind + file + doc."""
        parts = [f"[{self.kind}] {self.signature}"]
        parts.append(f"  file: {self.file_path}:{self.start_line}")
        if self.doc_summary:
            parts.append(f"  doc: {self.doc_summary}")
        return "\n".join(parts)

    def to_standard(self) -> str:
        """~120 token representation: compact + members + parent info."""
        parts = [self.to_compact()]
        if self.parent_class:
            parts.append(f"  parent: {self.parent_class}")
        if self.members:
            parts.append("  members:")
            for m in self.members[:MAX_MEMBERS_IN_CARD]:
                parts.append(f"    - {m}")
        return "\n".join(parts)

    def to_full(self, file_contents: str) -> str:
        """Full representation including the actual code body."""
        lines = file_contents.splitlines()
        start = max(0, self.start_line - 1)
        end = min(len(lines), self.end_line)
        body = "\n".join(lines[start:end])
        return f"[{self.kind}] {self.file_path}:{self.start_line}-{self.end_line}\n{body}"


_definitions_cache: dict[str, list[DefinitionInfo]] = {}
_import_map_cache: dict[str, dict] = {}
_repo_symbol_index: dict[str, list[tuple[str, DefinitionInfo]]] = {}
_repo_symbol_index_dir: str = ""

_EXCLUDED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".cursor", ".github", "docs", "sweep_chat",
}


def _build_repo_symbol_index(
    repo_dir: str,
) -> dict[str, list[tuple[str, DefinitionInfo]]]:
    """
    Build a repo-wide mapping of symbol name -> [(file_path, DefinitionInfo)].

    Walks all .py files once, caches the result at module level,
    and invalidates when repo_dir changes.
    """
    global _repo_symbol_index, _repo_symbol_index_dir
    if _repo_symbol_index and _repo_symbol_index_dir == repo_dir:
        return _repo_symbol_index

    index: dict[str, list[tuple[str, DefinitionInfo]]] = {}
    for root, dirs, filenames in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in _EXCLUDED_DIRS]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            full = os.path.join(root, fname)
            try:
                size = os.path.getsize(full)
                if size > MAX_FILE_SIZE_BYTES:
                    continue
                code = open(full, "r", encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            rel = os.path.relpath(full, repo_dir).replace("\\", "/")
            defs = _cached_extract_definitions(rel, code)
            for d in defs:
                index.setdefault(d.name, []).append((rel, d))

    _repo_symbol_index = index
    _repo_symbol_index_dir = repo_dir
    logger.info(
        f"Built repo symbol index: {len(index)} unique symbols "
        f"across {sum(len(v) for v in index.values())} definitions"
    )
    return index


def _cached_extract_definitions(file_path: str, code: str) -> list[DefinitionInfo]:
    """Cache extract_definitions by file_path + content hash."""
    key = file_path + ":" + hashlib.md5(code.encode()).hexdigest()
    if key in _definitions_cache:
        return _definitions_cache[key]
    result = extract_definitions(file_path, code)
    _definitions_cache[key] = result
    if len(_definitions_cache) > 2000:
        oldest = next(iter(_definitions_cache))
        del _definitions_cache[oldest]
    return result


def _cached_parse_imports(file_path: str, code: str, language: str) -> dict:
    """Cache import map by file_path + content hash."""
    key = file_path + ":" + language + ":" + hashlib.md5(code.encode()).hexdigest()
    if key in _import_map_cache:
        return _import_map_cache[key]
    if language == "python":
        result = _parse_python_imports(code)
    elif language in ("tsx", "typescript", "javascript"):
        result = _parse_ts_imports(code)
    else:
        result = {}
    _import_map_cache[key] = result
    if len(_import_map_cache) > 2000:
        oldest = next(iter(_import_map_cache))
        del _import_map_cache[oldest]
    return result


def _parse_python_imports(code: str) -> dict[str, tuple[str, Optional[str]]]:
    """
    Parse Python import statements and return a mapping of
    local_name -> (module_path, original_name_or_None).

    Handles:
      - from foo.bar import Baz          -> {"Baz": ("foo.bar", "Baz")}
      - from foo.bar import Baz as B     -> {"B": ("foo.bar", "Baz")}
      - import foo.bar                   -> {"foo": ("foo.bar", None)}
      - from . import sibling            -> {"sibling": (".", "sibling")}
      - from ..parent import X           -> {"X": ("..parent", "X")}
    """
    imports: dict[str, tuple[str, Optional[str]]] = {}
    for line in code.splitlines():
        line = line.strip()
        m = re.match(r"^from\s+([\w.]+)\s+import\s+(.+)$", line)
        if m:
            module = m.group(1)
            names_str = m.group(2)
            if names_str.strip() == "*":
                continue
            for part in names_str.split(","):
                part = part.strip()
                if not part:
                    continue
                alias_match = re.match(r"(\w+)\s+as\s+(\w+)", part)
                if alias_match:
                    original, alias = alias_match.group(1), alias_match.group(2)
                    imports[alias] = (module, original)
                elif re.match(r"^\w+$", part):
                    imports[part] = (module, part)
            continue

        m = re.match(r"^import\s+([\w.]+)(?:\s+as\s+(\w+))?$", line)
        if m:
            module = m.group(1)
            alias = m.group(2) or module.split(".")[0]
            imports[alias] = (module, None)
    return imports


def _parse_ts_imports(code: str) -> dict[str, tuple[str, Optional[str]]]:
    """
    Parse TypeScript/JavaScript import statements.

    Handles:
      - import { Foo, Bar as B } from './module'
      - import Foo from './module'
      - const { X } = require('./module')
    """
    imports: dict[str, tuple[str, Optional[str]]] = {}
    for m in re.finditer(
        r"""import\s+\{([^}]+)\}\s+from\s+['"]([^'"]+)['"]""", code
    ):
        names_str, module = m.group(1), m.group(2)
        for part in names_str.split(","):
            part = part.strip()
            alias_match = re.match(r"(\w+)\s+as\s+(\w+)", part)
            if alias_match:
                original, alias = alias_match.group(1), alias_match.group(2)
                imports[alias] = (module, original)
            elif re.match(r"^\w+$", part):
                imports[part] = (module, part)

    for m in re.finditer(
        r"""import\s+(\w+)\s+from\s+['"]([^'"]+)['"]""", code
    ):
        name, module = m.group(1), m.group(2)
        imports[name] = (module, None)

    for m in re.finditer(
        r"""(?:const|let|var)\s+\{([^}]+)\}\s*=\s*require\s*\(\s*['"]([^'"]+)['"]\s*\)""",
        code,
    ):
        names_str, module = m.group(1), m.group(2)
        for part in names_str.split(","):
            part = part.strip()
            if re.match(r"^\w+$", part):
                imports[part] = (module, part)

    return imports


def _resolve_python_module_to_path(
    module: str,
    original_name: Optional[str],
    current_file: str,
    repo_dir: str,
) -> Optional[str]:
    """Convert a Python module path to a file path within the repo."""
    if module.startswith("."):
        current_dir = os.path.dirname(current_file)
        dots = len(module) - len(module.lstrip("."))
        for _ in range(dots - 1):
            current_dir = os.path.dirname(current_dir)
        remainder = module.lstrip(".")
        if remainder:
            rel = os.path.join(current_dir, remainder.replace(".", os.sep))
        else:
            rel = current_dir
    else:
        rel = module.replace(".", os.sep)

    candidates = [
        os.path.join(repo_dir, rel + ".py"),
        os.path.join(repo_dir, rel, "__init__.py"),
    ]
    if original_name:
        candidates.insert(0, os.path.join(repo_dir, rel, original_name + ".py"))

    for path in candidates:
        if os.path.isfile(path):
            return os.path.relpath(path, repo_dir).replace("\\", "/")
    return None


def _resolve_ts_module_to_path(
    module: str,
    current_file: str,
    repo_dir: str,
) -> Optional[str]:
    """Convert a TS/JS import path to a file path within the repo."""
    if module.startswith("."):
        current_dir = os.path.dirname(os.path.join(repo_dir, current_file))
        base = os.path.join(current_dir, module)
    else:
        base = os.path.join(repo_dir, "node_modules", module)
        if not os.path.exists(base):
            base = os.path.join(repo_dir, module)

    extensions = [".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.tsx", "/index.js"]
    for ext in extensions:
        candidate = base + ext
        if os.path.isfile(candidate):
            return os.path.relpath(candidate, repo_dir).replace("\\", "/")
    if os.path.isfile(base):
        return os.path.relpath(base, repo_dir).replace("\\", "/")
    return None


def _find_definition_in_file(
    symbol_name: str,
    file_path: str,
    repo_dir: str,
    file_contents_cache: dict[str, str],
) -> Optional[DefinitionCard]:
    """Look up a symbol's definition in a specific file."""
    full_path = os.path.join(repo_dir, file_path)
    if file_path in file_contents_cache:
        code = file_contents_cache[file_path]
    elif os.path.isfile(full_path):
        try:
            code = open(full_path, "r", encoding="utf-8", errors="replace").read()
            file_contents_cache[file_path] = code
        except Exception:
            return None
    else:
        return None

    defs = _cached_extract_definitions(file_path, code)
    for d in defs:
        if d.name == symbol_name:
            return DefinitionCard(
                symbol_name=d.name,
                fqn=f"{file_path}:{d.name}",
                kind=d.kind,
                signature=d.signature,
                file_path=file_path,
                start_line=d.start_line,
                end_line=d.end_line,
                doc_summary=d.doc_summary,
                parent_class=d.parent_name,
                members=d.members[:MAX_MEMBERS_IN_CARD],
            )
    return None


def _extract_identifiers_from_code(code: str, language: str) -> list[str]:
    """Use tree-sitter to extract unique identifiers from code."""
    try:
        parser = get_parser(language)
    except Exception:
        return _extract_identifiers_regex(code)
    tree = parser.parse(bytes(code, "utf8"))
    identifiers = set()

    def walk(node):
        if node.type == "identifier":
            text = node.text.decode("utf8")
            if len(text) > 1 and not text.startswith("_"):
                identifiers.add(text)
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return list(identifiers)


def _extract_identifiers_regex(code: str) -> list[str]:
    """Fallback identifier extraction using regex."""
    tokens = set(re.findall(r"\b([A-Z]\w{2,}|[a-z]\w{2,}_\w+)\b", code))
    builtins = {
        "self", "None", "True", "False", "str", "int", "float", "bool",
        "list", "dict", "set", "tuple", "print", "return", "import",
        "from", "class", "def", "for", "while", "if", "else", "elif",
        "try", "except", "finally", "with", "as", "pass", "break",
        "continue", "and", "or", "not", "in", "is", "lambda", "yield",
        "raise", "assert", "del", "global", "nonlocal", "async", "await",
        "const", "let", "var", "function", "new", "this", "typeof",
        "instanceof", "void", "null", "undefined", "export", "default",
    }
    return [t for t in tokens if t not in builtins]


def _extract_symbol_names_from_query(query: str) -> list[str]:
    """Extract likely symbol names from a natural language query."""
    symbols = []
    camel_case = re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", query)
    symbols.extend(camel_case)
    snake_case = re.findall(r"\b([a-z]\w*_\w+)\b", query)
    symbols.extend(snake_case)
    backtick = re.findall(r"`(\w+)`", query)
    symbols.extend(backtick)
    upper_start = re.findall(r"\b([A-Z][a-zA-Z0-9]{2,})\b", query)
    symbols.extend(upper_start)
    seen = set()
    result = []
    for s in symbols:
        if s not in seen and len(s) > 1:
            seen.add(s)
            result.append(s)
    return result


def _fuzzy_match_candidates(
    candidates: list[str],
    index: dict[str, list[tuple[str, DefinitionInfo]]],
    threshold: int = FUZZY_MATCH_THRESHOLD,
) -> list[tuple[str, float]]:
    """
    Fuzzy-match generated NL candidates against real symbol names in the index.

    Uses rapidfuzz.process.extract for batched C-level matching.
    Returns (real_symbol_name, score) pairs above threshold,
    capped at MAX_FUZZY_MATCHES.
    """
    index_keys = list(index.keys())
    if not index_keys or not candidates:
        return []

    matches: list[tuple[str, float]] = []
    seen: set[str] = set()

    for candidate in candidates:
        if candidate in index and candidate not in seen:
            seen.add(candidate)
            matches.append((candidate, 100.0))

    if len(matches) >= MAX_FUZZY_MATCHES:
        return matches[:MAX_FUZZY_MATCHES]

    remaining = [c for c in candidates if c not in seen and c not in index]
    for candidate in remaining:
        if len(matches) >= MAX_FUZZY_MATCHES:
            break
        hits = rfprocess.extract(
            candidate.lower(),
            {k: k.lower() for k in index_keys},
            scorer=fuzz.ratio,
            score_cutoff=threshold,
            limit=3,
        )
        for real_name, score, _key in hits:
            actual_name = _key
            if actual_name not in seen:
                seen.add(actual_name)
                matches.append((actual_name, score))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:MAX_FUZZY_MATCHES]


def _casing_variants(*parts: str) -> list[str]:
    """Generate snake_case, camelCase, and PascalCase from word parts."""
    return [
        "_".join(parts),
        parts[0] + "".join(p.title() for p in parts[1:]),
        "".join(p.title() for p in parts),
    ]


def _generate_identifier_candidates(query: str) -> list[str]:
    """
    Generate plausible code identifier candidates from natural language.

    Takes NL words, strips stopwords, and produces snake_case, camelCase,
    and PascalCase combinations. Also generates verb-object inversions
    (e.g. "tree building" -> "build_tree").
    """
    raw_words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
    words = []
    for w in raw_words:
        stem = _GERUND_TO_STEM.get(w)
        if stem:
            words.append(stem)
        elif w not in _NL_STOPWORDS:
            if w.endswith("s") and len(w) > 4 and not w.endswith("ss"):
                words.append(w[:-1])
            else:
                words.append(w)

    if not words:
        return []

    candidates: set[str] = set()

    for i in range(len(words)):
        for j in range(i + 1, min(i + 4, len(words) + 1)):
            chunk = words[i:j]
            candidates.add("_".join(chunk))
            candidates.add(chunk[0] + "".join(w.title() for w in chunk[1:]))
            candidates.add("".join(w.title() for w in chunk))

    verbs_in_query = [w for w in words if w in _VERB_STEMS]
    nouns_in_query = [w for w in words if w not in _VERB_STEMS]

    for verb in verbs_in_query:
        for i in range(len(nouns_in_query)):
            n1 = nouns_in_query[i]
            for style in _casing_variants(verb, n1):
                candidates.add(style)
            if not n1.endswith("s"):
                candidates.add(f"{verb}_{n1}s")
            for j in range(i + 1, min(i + 3, len(nouns_in_query))):
                n2 = nouns_in_query[j]
                for style in _casing_variants(verb, n1, n2):
                    candidates.add(style)
                if not n2.endswith("s"):
                    candidates.add(f"{verb}_{n1}_{n2}s")
                for style in _casing_variants(verb, n2, n1):
                    candidates.add(style)

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        stem = _GERUND_TO_STEM.get(w2, w2)
        if stem in _VERB_STEMS:
            candidates.add(f"{stem}_{w1}")
            candidates.add(f"{stem}_{w1}s")
            candidates.add(f"{stem}{w1.title()}")
            candidates.add(f"{stem}{w1.title()}s")
            candidates.add(f"{stem.title()}{w1.title()}")

    candidates.discard("")
    return [c for c in candidates if len(c) > 3]


def _get_language_for_file(file_path: str) -> Optional[str]:
    ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
    return extension_to_language.get(ext)


def resolve_symbols_in_snippet(
    snippet: Snippet,
    repo_dir: str,
    file_contents_cache: dict[str, str],
    max_cards: int = 10,
) -> list[DefinitionCard]:
    """
    Given a code snippet, extract identifiers and resolve them to their
    definition sites by following import statements.
    """
    language = _get_language_for_file(snippet.file_path)
    if not language:
        return []

    full_path = os.path.join(repo_dir, snippet.file_path)
    if snippet.file_path in file_contents_cache:
        full_code = file_contents_cache[snippet.file_path]
    elif os.path.isfile(full_path):
        try:
            full_code = open(full_path, "r", encoding="utf-8", errors="replace").read()
            file_contents_cache[snippet.file_path] = full_code
        except Exception:
            return []
    else:
        return []

    snippet_text = snippet.get_snippet(add_ellipsis=False, add_lines=False)
    identifiers = _extract_identifiers_from_code(snippet_text, language)

    import_map = _cached_parse_imports(snippet.file_path, full_code, language)

    local_defs = _cached_extract_definitions(snippet.file_path, full_code)
    local_def_names = {d.name for d in local_defs}

    cards: list[DefinitionCard] = []
    seen_symbols: set[str] = set()

    for ident in identifiers:
        if len(cards) >= max_cards:
            break
        if ident in seen_symbols:
            continue
        seen_symbols.add(ident)

        if ident in local_def_names:
            for d in local_defs:
                if d.name == ident:
                    cards.append(DefinitionCard(
                        symbol_name=d.name,
                        fqn=f"{snippet.file_path}:{d.name}",
                        kind=d.kind,
                        signature=d.signature,
                        file_path=snippet.file_path,
                        start_line=d.start_line,
                        end_line=d.end_line,
                        doc_summary=d.doc_summary,
                        parent_class=d.parent_name,
                        members=d.members[:MAX_MEMBERS_IN_CARD],
                    ))
                    break
            continue

        if ident in import_map:
            module, original_name = import_map[ident]
            target_name = original_name or ident

            if language == "python":
                target_file = _resolve_python_module_to_path(
                    module, original_name, snippet.file_path, repo_dir
                )
            else:
                target_file = _resolve_ts_module_to_path(
                    module, snippet.file_path, repo_dir
                )

            if target_file:
                card = _find_definition_in_file(
                    target_name, target_file, repo_dir, file_contents_cache
                )
                if card:
                    cards.append(card)

    return cards


def _make_card(
    d: DefinitionInfo, file_path: str, relevance: float,
) -> DefinitionCard:
    return DefinitionCard(
        symbol_name=d.name,
        fqn=f"{file_path}:{d.name}",
        kind=d.kind,
        signature=d.signature,
        file_path=file_path,
        start_line=d.start_line,
        end_line=d.end_line,
        doc_summary=d.doc_summary,
        parent_class=d.parent_name,
        members=d.members[:MAX_MEMBERS_IN_CARD],
        relevance_score=relevance,
    )


def resolve_symbols_from_query(
    query: str,
    snippets: list[Snippet],
    repo_dir: str,
) -> list[DefinitionCard]:
    """
    Resolve symbols from a natural-language query using a three-stage pipeline:

    1. Extract exact identifiers from the query text (regex)
    2. Generate NL identifier candidates and fuzzy-match against repo index
    3. If still nothing found, fall back to top-level defs from top-ranked files
    """
    all_cards: list[DefinitionCard] = []
    seen_fqns: set[str] = set()
    exact_files: set[str] = set()
    fuzzy_files: set[str] = set()

    index = _build_repo_symbol_index(repo_dir)

    exact_names = _extract_symbol_names_from_query(query)
    for sym in exact_names:
        if sym in index:
            for file_path, d in index[sym]:
                fqn = f"{file_path}:{d.name}"
                if fqn not in seen_fqns:
                    seen_fqns.add(fqn)
                    all_cards.append(_make_card(d, file_path, relevance=1.0))
                    exact_files.add(file_path)

    exact_name_set = set(exact_names)
    nl_candidates = _generate_identifier_candidates(query)
    fuzzy_matches = _fuzzy_match_candidates(nl_candidates, index)
    for real_name, score in fuzzy_matches:
        if real_name in exact_name_set:
            continue
        relevance = 0.7 * (score / 100.0)
        if real_name in index:
            for file_path, d in index[real_name]:
                fqn = f"{file_path}:{d.name}"
                if fqn not in seen_fqns:
                    seen_fqns.add(fqn)
                    all_cards.append(_make_card(d, file_path, relevance=relevance))
                    fuzzy_files.add(file_path)

    if all_cards:
        for file_path in sorted(exact_files) + sorted(fuzzy_files - exact_files):
            full_path = os.path.join(repo_dir, file_path)
            if not os.path.isfile(full_path):
                continue
            try:
                code = open(full_path, "r", encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            neighbor_rel = 0.35 if file_path in exact_files else 0.2
            defs = _cached_extract_definitions(file_path, code)
            for d in defs:
                fqn = f"{file_path}:{d.name}"
                if fqn not in seen_fqns and d.kind in ("class", "function"):
                    seen_fqns.add(fqn)
                    all_cards.append(_make_card(d, file_path, relevance=neighbor_rel))
    elif snippets:
        all_cards = _recall_safety_net(snippets, repo_dir, seen_fqns)

    all_cards.sort(key=lambda c: c.relevance_score, reverse=True)
    return all_cards[:MAX_CARDS_PER_QUERY]


def _recall_safety_net(
    snippets: list[Snippet],
    repo_dir: str,
    seen_fqns: set[str],
) -> list[DefinitionCard]:
    """
    When no symbols were resolved, return top-level definitions from the
    highest-ranked snippet files as a bounded fallback (~500 tokens max).
    """
    cards: list[DefinitionCard] = []
    for snippet in snippets[:SAFETY_NET_MAX_FILES]:
        full_path = os.path.join(repo_dir, snippet.file_path)
        if not os.path.isfile(full_path):
            continue
        try:
            code = open(full_path, "r", encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        defs = _cached_extract_definitions(snippet.file_path, code)
        for d in defs:
            if d.kind in ("class", "function"):
                fqn = f"{snippet.file_path}:{d.name}"
                if fqn not in seen_fqns:
                    seen_fqns.add(fqn)
                    cards.append(_make_card(d, snippet.file_path, relevance=0.3))
                    if len(cards) >= SAFETY_NET_MAX_CARDS:
                        return cards
    return cards


@file_cache()
def get_definition_cards_for_file(
    file_path: str,
    repo_dir: str,
) -> list[DefinitionCard]:
    """Return all definition cards for a single file. Used for indexing/caching."""
    full_path = os.path.join(repo_dir, file_path)
    if not os.path.isfile(full_path):
        return []
    try:
        code = open(full_path, "r", encoding="utf-8", errors="replace").read()
    except Exception:
        return []

    defs = _cached_extract_definitions(file_path, code)
    return [
        DefinitionCard(
            symbol_name=d.name,
            fqn=f"{file_path}:{d.name}",
            kind=d.kind,
            signature=d.signature,
            file_path=file_path,
            start_line=d.start_line,
            end_line=d.end_line,
            doc_summary=d.doc_summary,
            parent_class=d.parent_name,
            members=d.members[:MAX_MEMBERS_IN_CARD],
        )
        for d in defs
    ]
