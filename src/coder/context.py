"""Incremental source graph and time-filtered lexical retrieval in SQLite."""

import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

from .common import commit, digest, git

EXTENSIONS = {".py": "python", ".js": "javascript", ".jsx": "javascript",
              ".mjs": "javascript", ".cjs": "javascript", ".ts": "typescript", ".tsx": "tsx"}
DEFINITIONS = {"function_definition", "class_definition", "function_declaration",
               "class_declaration", "method_definition", "arrow_function", "function_expression"}
IMPORTS = {"import_statement", "import_from_statement", "export_statement"}
CALLS = {"call", "call_expression", "new_expression"}


@lru_cache(maxsize=4)
def parser(language):
    from tree_sitter import Language, Parser
    import tree_sitter_javascript
    import tree_sitter_python
    import tree_sitter_typescript
    grammars = {
        "python": tree_sitter_python.language,
        "javascript": tree_sitter_javascript.language,
        "typescript": tree_sitter_typescript.language_typescript,
        "tsx": tree_sitter_typescript.language_tsx,
    }
    return Parser(Language(grammars[language]()))


def parse_source(path, source):
    language = EXTENSIONS.get(Path(path).suffix)
    if not language:
        return [], [], [], [], False
    source_parser = parser(language)
    tree = source_parser.parse(source)
    root = tree.root_node
    symbols, calls, imports, references = [], [], [], []

    def text(node):
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    stack = [(root, "<module>")]
    while stack:
        node, scope = stack.pop()
        name = node.child_by_field_name("name")
        if node.type in DEFINITIONS:
            if name is None and node.parent is not None:
                name = node.parent.child_by_field_name("name")
            if name is not None:
                simple = text(name)
                scope = simple if scope == "<module>" else f"{scope}.{simple}"
                symbols.append((scope, simple, node.type, node.start_point.row + 1,
                                node.end_point.row + 1, text(node)))
        if node.type in IMPORTS:
            imports.append((scope, text(node), node.start_point.row + 1))
        if node.type in CALLS:
            target = node.child_by_field_name("function") or node.child_by_field_name("constructor")
            if target is not None:
                calls.append((scope, text(target), node.start_point.row + 1))
        if node.type == "identifier":
            references.append((scope, text(node), node.start_point.row + 1))
        stack.extend((child, scope) for child in reversed(node.named_children))
    return symbols, calls, imports, references, root.has_error


class Graph:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, hash TEXT, language TEXT, parse_error INTEGER);
          CREATE TABLE IF NOT EXISTS symbols(path TEXT, qualified TEXT, name TEXT, kind TEXT, line INTEGER, end_line INTEGER, source TEXT);
          CREATE TABLE IF NOT EXISTS calls(path TEXT, caller TEXT, target TEXT, line INTEGER, resolved_path TEXT, resolved_symbol TEXT);
          CREATE TABLE IF NOT EXISTS imports(path TEXT, scope TEXT, source TEXT, line INTEGER);
          CREATE TABLE IF NOT EXISTS refs(path TEXT, scope TEXT, name TEXT, line INTEGER);
          CREATE TABLE IF NOT EXISTS dependencies(path TEXT, module TEXT, target_path TEXT, aliases TEXT);
          CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT);
        """)
        with self.db:
            self._resolve_edges()

    def close(self):
        self.db.close()

    def update(self, sources, revision="working-tree"):
        old = dict(self.db.execute("SELECT path,hash FROM files"))
        changed = 0
        with self.db:
            for path in set(old) - set(sources):
                self._remove(path)
            for path, source in sources.items():
                checksum = digest(source)
                if old.get(path) == checksum:
                    continue
                changed += 1
                self._remove(path)
                symbols, calls, imports, references, errors = parse_source(path, source)
                self.db.execute("INSERT INTO files VALUES(?,?,?,?)",
                                (path, checksum, EXTENSIONS.get(Path(path).suffix), errors))
                self.db.executemany("INSERT INTO symbols VALUES(?,?,?,?,?,?,?)", [(path, *s) for s in symbols])
                self.db.executemany("INSERT INTO calls VALUES(?,?,?,?,NULL,NULL)", [(path, *c) for c in calls])
                self.db.executemany("INSERT INTO imports VALUES(?,?,?,?)", [(path, *i) for i in imports])
                self.db.executemany("INSERT INTO refs VALUES(?,?,?,?)", [(path, *r) for r in references])
            self._resolve_edges()
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES('revision',?)", (revision,))
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES('parser_version','0.25.2')")
        return {"files": len(sources), "changed": changed,
                "supported_files": self.db.execute("SELECT count(*) FROM files WHERE language IS NOT NULL").fetchone()[0],
                "parse_errors": self.db.execute("SELECT count(*) FROM files WHERE parse_error=1").fetchone()[0],
                "unresolved_calls": self.db.execute("SELECT count(*) FROM calls WHERE resolved_path IS NULL").fetchone()[0]}

    def _resolve_edges(self):
        paths = {row[0] for row in self.db.execute("SELECT path FROM files")}
        self.db.execute("DELETE FROM dependencies")
        for path, source in self.db.execute("SELECT path,source FROM imports").fetchall():
            aliases, module = {}, None
            python = re.search(r"from\s+([\w.]+)\s+import\s+(.+)", source, re.S)
            javascript = re.search(r"(?:from\s*|^import\s*)['\"]([^'\"]+)['\"]", source)
            if python:
                module, names = python.groups()
                for imported in names.strip(" ()\n").split(","):
                    pair = imported.strip().split(" as ")
                    if pair[0] and pair[0] != "*":
                        aliases[pair[-1].strip()] = pair[0].strip()
            elif javascript:
                module = javascript.group(1)
                named = re.search(r"\{([^}]+)\}", source)
                if named:
                    for imported in named.group(1).split(","):
                        pair = imported.strip().split(" as ")
                        if pair[0]:
                            aliases[pair[-1].strip()] = pair[0].strip()
                namespace = re.search(r"\*\s+as\s+([\w$]+)", source)
                if namespace:
                    aliases[namespace.group(1)] = "*"
            elif Path(path).suffix == ".py":
                imported = re.match(r"import\s+([\w.]+)(?:\s+as\s+(\w+))?", source)
                if imported:
                    module = imported.group(1)
                    aliases[imported.group(2) or module.split(".")[0]] = "*"
            if module is None:
                continue
            if Path(path).suffix == ".py":
                dots = len(module) - len(module.lstrip("."))
                directory = Path(path).parent
                for _ in range(max(0, dots - 1)):
                    directory = directory.parent
                stem = str(directory / module.lstrip(".").replace(".", "/")) if dots else module.replace(".", "/")
                candidates = [stem + ".py", stem + "/__init__.py"]
            elif module.startswith("."):
                import posixpath
                stem = posixpath.normpath(str(Path(path).parent / module).replace("\\", "/"))
                candidates = [stem, *[stem + suffix for suffix in EXTENSIONS if suffix != ".py"],
                              *[stem + "/index" + suffix for suffix in (".ts", ".tsx", ".js")]]
            else:
                candidates = []
            resolved = [candidate.replace("\\", "/") for candidate in candidates if candidate.replace("\\", "/") in paths]
            target_path = resolved[0] if len(resolved) == 1 else None
            self.db.execute("INSERT INTO dependencies VALUES(?,?,?,?)", (path, module, target_path, json.dumps(aliases)))
        self.db.execute("UPDATE calls SET resolved_path=NULL,resolved_symbol=NULL")
        for rowid, path, caller, target in self.db.execute("SELECT rowid,path,caller,target FROM calls").fetchall():
            matches = []
            if re.fullmatch(r"[A-Za-z_$][\w$]*", target):
                matches = self.db.execute("SELECT path,qualified FROM symbols WHERE path=? AND qualified IN (?,?)",
                                          (path, target, caller + "." + target)).fetchall()
            if not matches:
                for imported_path, raw_aliases in self.db.execute("SELECT target_path,aliases FROM dependencies WHERE path=? AND target_path IS NOT NULL", (path,)):
                    aliases = json.loads(raw_aliases)
                    symbol = aliases.get(target)
                    if "." in target:
                        namespace, member = target.split(".", 1)
                        if aliases.get(namespace) == "*":
                            symbol = member
                    if symbol and symbol != "*":
                        matches.extend(self.db.execute("SELECT path,qualified FROM symbols WHERE path=? AND qualified=?", (imported_path, symbol)).fetchall())
            if len(matches) == 1:
                self.db.execute("UPDATE calls SET resolved_path=?,resolved_symbol=? WHERE rowid=?", (*matches[0], rowid))

    def _remove(self, path):
        for table in ("files", "symbols", "calls", "imports", "refs", "dependencies"):
            self.db.execute(f"DELETE FROM {table} WHERE path=?", (path,))

    def context(self, paths, limit=16):
        result = []
        for path in paths:
            dependencies = self.db.execute("SELECT module,target_path,aliases FROM dependencies WHERE path=?", (path,)).fetchall()
            imports = self.db.execute("SELECT source,line FROM imports WHERE path=?", (path,)).fetchall()
            if dependencies or imports:
                result.append({"path": path, "imports": [{"source": source, "line": line} for source, line in imports],
                               "dependencies": [{"module": module, "resolved_path": target, "aliases": json.loads(aliases)}
                                                for module, target, aliases in dependencies]})
            for _, target_path, _ in dependencies:
                if target_path:
                    for symbol, line, source in self.db.execute("SELECT qualified,line,source FROM symbols WHERE path=? ORDER BY line LIMIT 2", (target_path,)):
                        result.append({"path": target_path, "symbol": symbol, "line": line, "source": source[:4000], "dependency_of": path})
            for qualified, line, end_line, source in self.db.execute(
                "SELECT qualified,line,end_line,source FROM symbols WHERE path=? ORDER BY line", (path,)
            ):
                result.append({"path": path, "symbol": qualified, "line": line,
                               "end_line": end_line, "source": source[:8000]})
            for caller_path, caller, line in self.db.execute(
                "SELECT path,caller,line FROM calls WHERE resolved_path=?", (path,)
            ):
                result.append({"caller_path": caller_path, "caller": caller, "line": line})
        return result[:limit]


def snapshot(repo, revision, max_files=5000, max_bytes=1024 * 1024, sources_required=True):
    revision = commit(repo, revision)
    entries = git(repo, "ls-tree", "-r", "-z", "--long", revision, binary=True).split(b"\0")
    sources, excluded = {}, []
    standards = {}
    for entry in entries:
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        path = raw_path.decode("utf-8", errors="replace")
        mode, kind, oid, size = metadata.decode().split()
        if kind != "blob" or mode == "120000":
            excluded.append({"path": path, "reason": "non-regular-file"})
            continue
        is_standard = bool(re.search(r"(^|/)(CONTRIBUTING|AGENTS|STYLE|CODING|README|SECURITY)", path, re.I))
        supported = Path(path).suffix in EXTENSIONS
        if not supported and not is_standard:
            excluded.append({"path": path, "reason": "unsupported-language-or-non-source"})
            continue
        if int(size) > max_bytes or len(sources) >= max_files:
            excluded.append({"path": path, "reason": "snapshot-size-limit"})
            continue
        content = git(repo, "cat-file", "blob", oid, binary=True) if sources_required or is_standard else b""
        if supported and sources_required:
            sources[path] = content
        if is_standard:
            standards[path] = content.decode("utf-8", errors="replace")
    return revision, sources, standards, excluded


class Retrieval:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY, repo TEXT, timestamp TEXT, content TEXT, provenance TEXT);
          CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(id UNINDEXED, content);
        """)

    def add(self, identity, repo, timestamp, content, provenance):
        with self.db:
            self.db.execute("DELETE FROM search WHERE id=?", (identity,))
            self.db.execute("INSERT OR REPLACE INTO docs VALUES(?,?,?,?,?)", (identity, repo, timestamp, content, json.dumps(provenance)))
            self.db.execute("INSERT INTO search VALUES(?,?)", (identity, content))

    def retrieve(self, query, repo, before, limit=6, exclude_pr=None):
        terms = list(dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z_0-9]{2,}", query)))[:32]
        if not terms:
            return []
        match = " OR ".join('"' + term + '"' for term in terms)
        values = self.db.execute("""
          SELECT docs.id,docs.timestamp,docs.content,docs.provenance
          FROM search JOIN docs ON docs.id=search.id
          WHERE search MATCH ? AND docs.repo=? AND docs.timestamp < ?
          ORDER BY bm25(search),docs.id
        """, (match, repo, before)).fetchall()
        result = []
        for identity, timestamp, content, raw_provenance in values:
            provenance = json.loads(raw_provenance)
            if exclude_pr is not None and provenance.get("pr") == exclude_pr:
                continue
            result.append({"id": identity, "timestamp": timestamp, "content": content[:4000], "provenance": provenance})
            if len(result) == limit:
                break
        return result

    def close(self):
        self.db.close()
