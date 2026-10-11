from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from agent_capabilities import binding_references
from content_digests import canonical_json_sha256

TRIGGER_HARNESS_IDENTITY_VERSION = 4
_ROOTS = ("run_trigger_matrix", "run_pi_trigger_eval", "trigger_identity")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class _SourceDigest:
    path: PurePosixPath
    sha256: str


@dataclass(frozen=True)
class _Catalog:
    root: Path
    paths: Mapping[str, PurePosixPath]
    direct_source: bool = True

    def local(self, module: str) -> bool:
        first = module.split(".")[0]
        if any(name.split(".")[0] == first for name in self.paths):
            return True
        candidate = self.root / first
        if self.direct_source and (candidate.with_suffix(".py").is_file() or (candidate / "__init__.py").is_file()):
            raise RuntimeError(f"trigger dependency {module} is absent from the packaging declaration")
        return False

    def required(self, module: str) -> PurePosixPath:
        try:
            return self.paths[module]
        except KeyError as exc:
            raise RuntimeError(f"cannot identify required trigger owner {module}: missing packaged module") from exc


def _module_name(path: PurePosixPath) -> str:
    parts = path.with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _source_catalog() -> _Catalog:
    own_file = Path(__file__).resolve()
    root = own_file.parent
    declaration = root / "pyproject.toml"
    for distribution in (() if declaration.is_file() else importlib.metadata.distributions()):
        files = distribution.files or ()
        if not any(Path(str(distribution.locate_file(item))).resolve() == own_file for item in files):
            continue
        record = distribution.read_text("RECORD")
        if record is None:
            raise RuntimeError("owning trigger distribution has no RECORD source catalog")
        declared = [PurePosixPath(row[0]) for row in csv.reader(record.splitlines()) if row]
        paths = {
            _module_name(item): item
            for item in declared
            if item.suffix == ".py" and not item.is_absolute() and ".." not in item.parts
        }
        return _Catalog(root, paths, direct_source=False)
    try:
        text = declaration.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError("trigger identity has no owning distribution or direct-source packaging declaration") from exc
    table = re.search(r"(?ms)^\[tool\.setuptools\]\s*$\n(.*?)(?=^\[|\Z)", text)
    array = re.search(r"(?ms)^py-modules\s*=\s*(\[.*?\])", table.group(1)) if table else None
    names = ast.literal_eval(array.group(1)) if array else None
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_]\w*", name) for name in names):
        raise RuntimeError("direct-source trigger identity requires explicit setuptools py-modules")
    return _Catalog(root, {name: PurePosixPath(name + ".py") for name in names})


class _Dependencies:
    def __init__(self, catalog: _Catalog, module: str, path: PurePosixPath) -> None:
        self.catalog = catalog
        self.module = module
        self.package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        self.modules: set[str] = set()
        self.categories: set[str] = set()

    def add(self, name: str) -> None:
        if self.catalog.local(name):
            self.modules.add(name)

    def absolute(self, name: str, package: str | None = None) -> str:
        if not name.startswith("."):
            return name
        context = self.package if package is None else package
        level = len(name) - len(name.lstrip("."))
        parts = context.split(".") if context else []
        if level > len(parts):
            raise RuntimeError(f"{self.module}: relative import escapes its packaged parent")
        return ".".join([*parts[:len(parts) - level + 1], *name[level:].split(".")]).rstrip(".")

    @staticmethod
    def lookup(node: ast.AST, aliases: Mapping[str, str]) -> str | None:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, "builtins.__import__" if node.id == "__import__" else None)
        if isinstance(node, ast.Attribute):
            parent = _Dependencies.lookup(node.value, aliases)
            return parent + "." + node.attr if parent else None
        return None

    def calls(self, node: ast.AST, aliases: Mapping[str, str], scope: tuple[str, ...]) -> None:
        if isinstance(node, ast.Call):
            target = self.lookup(node.func, aliases)
            if target in {"importlib.import_module", "builtins.__import__"}:
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    package = None
                    if target == "importlib.import_module":
                        argument = next((item.value for item in node.keywords if item.arg == "package"), None)
                        if len(node.args) > 1:
                            argument = node.args[1]
                        if argument is not None:
                            if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                                raise RuntimeError(f"{self.module}:{node.lineno}: computed import package is unsupported")
                            package = argument.value
                    self.add(self.absolute(node.args[0].value, package))
                elif not (self.module == "agent_capabilities" and scope == ("ObjectRef", "resolve")):
                    raise RuntimeError(f"{self.module}:{node.lineno}: computed dependency cannot be identified before invocation")
            elif target == "agent_capabilities.surface_implementations":
                if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str) or node.args[0].value not in {"trigger", "answer", "judge"}:
                    raise RuntimeError(f"{self.module}:{node.lineno}: computed registry category is unsupported")
                self.categories.add(node.args[0].value)
            elif target == "agent_capabilities.trace_dialect_implementations":
                self.categories.add("trace")
            elif target == "agent_capabilities.workspace_builder_implementations":
                self.categories.add("workspace_builder")
        for child in ast.iter_child_nodes(node):
            self.calls(child, aliases, scope)

    @staticmethod
    def local_names(body: Sequence[ast.stmt]) -> set[str]:
        found: set[str] = set()
        globals_: set[str] = set()
        def visit(node: ast.AST) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.add(node.name)
                return
            if isinstance(node, ast.Global):
                globals_.update(node.names)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                found.add(node.id)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                found.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
            for child in ast.iter_child_nodes(node):
                visit(child)
        for statement in body:
            visit(statement)
        return found - globals_

    @staticmethod
    def join_aliases(aliases: dict[str, str], branches: Sequence[dict[str, str]]) -> None:
        aliases.clear()
        aliases.update({name: value for name, value in branches[0].items()
                        if all(branch.get(name) == value for branch in branches[1:])})
        for branch in branches:
            for name, value in branch.items():
                if value in {"importlib", "importlib.import_module", "builtins.__import__"}:
                    aliases.setdefault(name, value)

    def walk(self, body: Sequence[ast.stmt], aliases: dict[str, str], scope: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.add(alias.name)
                    aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                module = self.absolute("." * node.level + (node.module or ""))
                self.add(module)
                for alias in node.names:
                    child = module + "." + alias.name
                    if alias.name != "*" and child in self.catalog.paths:
                        self.add(child)
                    elif alias.name != "*" and module in self.catalog.paths:
                        candidate = self.catalog.root.joinpath(*child.split("."))
                        if candidate.with_suffix(".py").is_file() or (candidate / "__init__.py").is_file():
                            raise RuntimeError(f"trigger dependency {child} is absent from the packaging declaration")
                    aliases[alias.asname or alias.name] = child
            elif isinstance(node, ast.If):
                self.calls(node.test, aliases, scope)
                if self.lookup(node.test, aliases) == "typing.TYPE_CHECKING":
                    self.walk(node.orelse, aliases, scope)
                    continue
                left, right = dict(aliases), dict(aliases)
                self.walk(node.body, left, scope)
                self.walk(node.orelse, right, scope)
                self.join_aliases(aliases, (left, right))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for decorator in node.decorator_list:
                    self.calls(decorator, aliases, scope)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.calls(node.args, aliases, scope)
                    if node.returns is not None:
                        self.calls(node.returns, aliases, scope)
                else:
                    for base in node.bases:
                        self.calls(base, aliases, scope)
                nested = dict(aliases)
                shadowed = self.local_names(node.body)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    shadowed.update(argument.arg for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs))
                    shadowed.update(argument.arg for argument in (node.args.vararg, node.args.kwarg) if argument is not None)
                for name in shadowed:
                    nested[name] = ""
                self.walk(node.body, nested, (*scope, node.name))
                aliases[node.name] = ""
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                self.calls(node, aliases, scope)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = self.lookup(node.value, aliases) if node.value is not None else None
                for target in targets:
                    for part in ast.walk(target):
                        if isinstance(part, ast.Name):
                            aliases[part.id] = value or ""
            else:
                branches = [getattr(node, key, None) for key in ("body", "orelse", "finalbody")]
                if isinstance(node, ast.Try):
                    branches.extend(handler.body for handler in node.handlers)
                if any(isinstance(branch, list) for branch in branches):
                    for key, value in ast.iter_fields(node):
                        if key not in {"body", "orelse", "finalbody", "handlers"}:
                            for child in value if isinstance(value, list) else [value]:
                                if isinstance(child, ast.AST):
                                    self.calls(child, aliases, scope)
                    variants = []
                    for branch in branches:
                        if isinstance(branch, list):
                            branch_aliases = dict(aliases)
                            self.walk(branch, branch_aliases, scope)
                            variants.append(branch_aliases)
                    if variants:
                        self.join_aliases(aliases, variants)
                else:
                    self.calls(node, aliases, scope)


@lru_cache(maxsize=256)
def _source_ast(path: PurePosixPath, content: bytes) -> ast.Module:
    return ast.parse(content, filename=str(path))


def _collect_sources(catalog: _Catalog | None = None, roots: Sequence[str] = _ROOTS) -> tuple[_SourceDigest, ...]:
    catalog = _source_catalog() if catalog is None else catalog
    pending = list(roots)
    visited: set[str] = set()
    sources: dict[PurePosixPath, str] = {}
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        path = catalog.required(module)
        for index in range(1, len(module.split("."))):
            pending.append(".".join(module.split(".")[:index]))
        try:
            content = (catalog.root / path).read_bytes()
        except OSError as exc:
            raise RuntimeError(f"cannot identify required trigger owner {module} at {path}: {exc}") from exc
        sources[path] = "sha256:" + hashlib.sha256(content).hexdigest()
        try:
            tree = _source_ast(path, content)
        except SyntaxError as exc:
            raise RuntimeError(f"cannot identify required trigger owner {module} at {path}:{exc.lineno}: {exc.msg}") from exc
        scanner = _Dependencies(catalog, module, path)
        scanner.walk(tree.body, {})
        pending.extend(scanner.modules)
        for category in scanner.categories:
            pending.extend(reference.module for reference in binding_references(category))
    return tuple(_SourceDigest(path, sources[path]) for path in sorted(sources))


def trigger_harness_identity() -> dict[str, Any]:
    modules = {str(item.path): item.sha256 for item in _collect_sources()}
    payload = {"schema_version": TRIGGER_HARNESS_IDENTITY_VERSION, "modules": modules}
    return {**payload, "identity_sha256": canonical_json_sha256(payload)}


def validate_trigger_harness_identity(identity: Any, label: str, *, expected_modules: Sequence[str] | None = None) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise TypeError(f"{label} harness_identity must be an object")
    version = identity.get("schema_version")
    if type(version) is int and version != TRIGGER_HARNESS_IDENTITY_VERSION:
        raise ValueError(f"{label} was produced by trigger harness identity v{version}; this harness reads v4. Regenerate the report.")
    payload = {key: value for key, value in identity.items() if key != "identity_sha256"}
    if type(version) is not int or set(payload) != {"schema_version", "modules"} or canonical_json_sha256(payload) != identity.get("identity_sha256"):
        raise ValueError(f"{label} harness_identity does not match its identity_sha256")
    expected = tuple(str(item.path) for item in _collect_sources()) if expected_modules is None else tuple(expected_modules)
    modules = identity.get("modules")
    if not isinstance(modules, dict) or set(modules) != set(expected):
        raise ValueError(f"{label} harness_identity must identify exactly {list(expected)}")
    if any(not isinstance(digest, str) or not _DIGEST.fullmatch(digest) for digest in modules.values()):
        raise ValueError(f"{label} harness_identity contains an invalid module digest")
    return identity


def assert_trigger_sources_unchanged(identity: Mapping[str, Any]) -> None:
    if identity != trigger_harness_identity():
        raise RuntimeError("trigger source bytes changed during execution; settled spend and partial observations remain available")
