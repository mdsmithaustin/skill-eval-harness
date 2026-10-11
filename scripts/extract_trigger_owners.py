"""Move the accepted trigger owners only after checking their original source."""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import symtable
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = '65c049a515b35de9f985f678910db319659ffe58bb4724cfe4ff99c339e61b4e'
RECIPE: list[dict[str, Any]] = json.loads(Path(__file__).with_name("trigger_owner_recipe.json").read_text(encoding="utf-8"))
ALLOWANCES = {"_ToolReceiptJournal": "injected bridge journal protocol", "validate_subagent_response": "injected final validator"}


def names(node: ast.stmt) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return [part.id for target in targets for part in ast.walk(target)
                if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)]
    return []


def source_segment(lines: list[str], node: ast.stmt) -> str:
    decorators = getattr(node, "decorator_list", [])
    start = min([node.lineno, *(item.lineno for item in decorators)])
    return "".join(lines[start - 1:node.end_lineno])


def globals_of(source: str, defined: Iterable[str]) -> set[str]:
    found: set[str] = set()
    def visit(table: symtable.SymbolTable, *, top: bool = False) -> None:
        found.update(symbol.get_name() for symbol in table.get_symbols()
                     if symbol.is_referenced() and (top or symbol.is_global()))
        for child in table.get_children():
            visit(child)
    visit(symtable.symtable(source, "movement", "exec"), top=True)
    return found - set(defined) - {part.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.comprehension) for part in ast.walk(node.target) if isinstance(part, ast.Name)}



def ast_fingerprint(node: ast.AST) -> str:
    normalized = ast.dump(node, include_attributes=False).replace(", type_params=[]", "")
    return hashlib.sha256(normalized.encode()).hexdigest()

def movement(apply: bool, root: Path) -> dict[str, Any]:
    path = root / "skill_benchmark.py"
    source = path.read_text(encoding="utf-8")
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError("skill_benchmark.py differs from the accepted extraction source")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    definitions: dict[str, ast.stmt] = {}
    imported: dict[str, str] = {}
    for node in tree.body:
        for name in names(node):
            if name in definitions:
                raise ValueError(f"duplicate defining binding {name}")
            definitions[name] = node
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
                text = alias.name + (f" as {alias.asname}" if alias.asname else "")
                imported[name] = (f"import {text}" if isinstance(node, ast.Import)
                                  else f"from {node.module} import {text}")
    owners = {row["name"]: row["owner"] for row in RECIPE}
    grouped: dict[str, list[tuple[ast.stmt, str]]] = {}
    dependencies: dict[str, set[str]] = {}
    for row in RECIPE:
        name, owner = row["name"], row["owner"]
        node = definitions[name]
        text = source_segment(lines, node)
        if hashlib.sha256(text.encode()).hexdigest() != row["sha256"]:
            raise ValueError(f"source span changed for {name}")
        if ast_fingerprint(node) != row["ast_sha256"]:
            raise ValueError(f"AST changed for {name}")
        captured = globals_of(text, names(node))
        if captured != set(row["globals"]):
            raise ValueError(f"globals/defaults/decorators changed for {name}")
        unresolved = captured - set(owners) - set(imported) - set(dir(builtins)) - set(ALLOWANCES) - {"__file__", "__name__"}
        if unresolved:
            raise ValueError(f"unassigned globals for {name}: {sorted(unresolved)}")
        grouped.setdefault(owner, []).append((node, text))
        dependencies.setdefault(owner, set()).update(captured)
    outputs: dict[Path, str] = {}
    remove: set[int] = set()
    for owner, sections in grouped.items():
        statements = set()
        for dep in dependencies[owner]:
            if dep in owners and owners[dep] != owner:
                statements.add(f"from {owners[dep]} import {dep}")
            elif dep in imported and not imported[dep].startswith(f"from {owner} import "):
                statements.add(imported[dep])
        target = root / f"{owner}.py"
        if target.exists():
            existing = target.read_text(encoding="utf-8")
            existing_names = {name for node in ast.parse(existing).body for name in names(node)}
            duplicate = existing_names & {name for name, actual in owners.items() if actual == owner}
            if duplicate:
                raise ValueError(f"target already defines {sorted(duplicate)}")
            text = existing + "\n\n" + "\n".join(sorted(statements)) + "\n\n"
        else:
            text = f'"""Shared {owner.replace("_", " ")} definitions."""\nfrom __future__ import annotations\n\n' + "\n".join(sorted(statements)) + "\n\n"
        for node, section in sorted(sections, key=lambda item: item[0].lineno):
            start = min([node.lineno, *(item.lineno for item in getattr(node, "decorator_list", []))])
            if node.end_lineno is None:
                raise ValueError("source node has no end location")
            remove.update(range(start - 1, node.end_lineno))
            if owner == "process_runtime":
                section = section.replace("journal: _ToolReceiptJournal", "journal: _BridgeJournal").replace("    limits: _BridgeLimits\n", "    limits: _BridgeLimits\n    validate_final: Callable[[Any], dict[str, Any]]\n", 1) if names(node) == ["_BridgeSession"] else section
                if names(node) == ["_invoke_process"]:
                    section = section.replace("validate_subagent_response(response)", "session.validate_final(response)")
            text += section + "\n\n"
        if owner == "process_runtime":
            text = text.replace("\n\nPROCESS_LEADER", "\nfrom typing import Protocol\n\nclass _BridgeJournal(Protocol):\n    def offer_bridge(self, frame: Mapping[str, Any], frame_limit: int = 64 * 1024) -> dict[str, Any]: ...\n    def acknowledge_bridge(self, frame: Mapping[str, Any]) -> None: ...\n\nPROCESS_LEADER", 1)
        if owner == "skill_inputs":
            text += "UniqueKeySafeLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_yaml_mapping)\n"
            text = text.replace("from yaml.constructor import ConstructorError", "from yaml.constructor import ConstructorError\nfrom yaml.resolver import BaseResolver")
        outputs[target] = text
    loader_init = next(node for node in tree.body if isinstance(node, ast.Expr)
                       and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "UniqueKeySafeLoader.add_constructor")
    if loader_init.end_lineno is None:
        raise ValueError("loader initialization has no end location")
    remove.update(range(loader_init.lineno - 1, loader_init.end_lineno))
    reexports = [f"from {owner} import {name}" for name, owner in sorted(owners.items())]
    kept = "".join(line for index, line in enumerate(lines) if index not in remove)
    kept = kept.replace("import tool_fault_contracts as tool_fault_domain", "\n".join(reexports) + "\n\nimport tool_fault_contracts as tool_fault_domain")
    kept = kept.replace("_BridgeSession(self, turn, directory, backend.limits)", "_BridgeSession(self, turn, directory, backend.limits, validate_subagent_response)")
    outputs[path] = kept
    summary = {"source_sha256": SOURCE_SHA256, "owners": {owner: [name for name, actual in sorted(owners.items()) if actual == owner] for owner in grouped}, "bindings": list(RECIPE), "allowances": ALLOWANCES, "apply": apply}
    if apply:
        for target, content in outputs.items():
            target.write_text(content, encoding="utf-8")
    return summary



def verify_movement(root: Path) -> dict[str, Any]:
    owner_nodes = {}
    counts: dict[str, int] = {}
    for owner in {row["owner"] for row in RECIPE}:
        tree = ast.parse((root / f"{owner}.py").read_text(encoding="utf-8"))
        owner_nodes[owner] = {name: node for node in tree.body for name in names(node)}
        if any(isinstance(node, ast.ImportFrom) and node.module == "skill_benchmark" for node in ast.walk(tree)):
            raise ValueError(f"{owner} imports answer orchestration")
    monolith = ast.parse((root / "skill_benchmark.py").read_text(encoding="utf-8"))
    remaining = {name for node in monolith.body for name in names(node)}
    imports = {alias.asname or alias.name: node.module for node in monolith.body
               if isinstance(node, ast.ImportFrom) for alias in node.names}
    for row in RECIPE:
        name, owner = row["name"], row["owner"]
        node = owner_nodes[owner][name]
        actual = ast_fingerprint(node)
        if actual != row.get("comment_cleanup_ast_sha256", row["moved_ast_sha256"]):
            raise ValueError(f"moved AST changed for {owner}.{name}")
        if name in remaining or imports.get(name) != owner:
            raise ValueError(f"monolith does not import the canonical {owner}.{name}")
        counts[owner] = counts.get(owner, 0) + 1
    return {"mode": "verify", "verified_bindings": len(RECIPE), "owners": counts,
            "source_sha256": SOURCE_SHA256, "allowances": ALLOWANCES,
            "removed_docstrings": {row["name"]: row["removed_docstrings"] for row in RECIPE if "removed_docstrings" in row}}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = verify_movement(args.source_root.resolve()) if args.verify else movement(args.apply, args.source_root.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
