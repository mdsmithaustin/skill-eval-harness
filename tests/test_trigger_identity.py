import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock

import process_runtime
import run_trigger_matrix as tm
import skill_benchmark as sb
import skill_inputs
import trace_normalization
import trigger_identity as ti
from content_digests import canonical_json_sha256
from spend_contracts import SpendPolicy

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "examples/demo-skill/evals/shared-benchmark.json"


class TriggerSourceIdentityTests(unittest.TestCase):
    def sources(self, directory, files):
        root = Path(directory)
        for relative, source in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        return ti._Catalog(root, {ti._module_name(PurePosixPath(relative)): PurePosixPath(relative) for relative in files})

    def test_ordinary_conditional_function_local_and_literal_alias_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            files = {name + ".py": "" for name in ("ordinary", "conditional", "inner", "literal", "type_only", "runtime_else", "shadowed", "unused")}
            files["root.py"] = """import ordinary
import importlib as il
from typing import TYPE_CHECKING as tc
if tc:
    import type_only
else:
    import runtime_else
if condition:
    import conditional
def f():
    import inner
    loader = il.import_module
    loader('literal')
def g():
    if tc:
        import shadowed
    tc = False
loader = il.import_module
loader = lambda name: None
loader('unused')
"""
            catalog = self.sources(directory, files)
            result = ti._collect_sources(catalog, ("root",))
            self.assertEqual([str(item.path) for item in result], ["conditional.py", "inner.py", "literal.py", "ordinary.py", "root.py", "runtime_else.py", "shadowed.py"])

    def test_relative_imports_include_initializers_and_literal_dynamic_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = self.sources(directory, {
                "root.py": "from importlib import import_module as load\nload('.child', 'pkg')\n",
                "pkg/__init__.py": "from . import sibling\n",
                "pkg/child.py": "from .sibling import VALUE\n",
                "pkg/sibling.py": "VALUE = 1\n",
            })
            self.assertEqual([str(item.path) for item in ti._collect_sources(catalog, ("root",))], ["pkg/__init__.py", "pkg/child.py", "pkg/sibling.py", "root.py"])

    def test_type_guard_shadowing_runtime_else_and_builtin_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = self.sources(directory, {
                "root.py": "import typing as t\nif t.TYPE_CHECKING:\n import skipped\nelse:\n import runtime\nt = object()\nif t.TYPE_CHECKING:\n import shadowed\nload = __import__\nload('literal')\n",
                **{name + ".py": "" for name in ("skipped", "runtime", "shadowed", "literal")},
            })
            self.assertEqual([str(item.path) for item in ti._collect_sources(catalog, ("root",))], ["literal.py", "root.py", "runtime.py", "shadowed.py"])

    def test_conditional_dynamic_alias_is_conservatively_included(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = self.sources(directory, {"root.py": "if condition:\n from importlib import import_module as load\nload('child')\n", "child.py": ""})
            self.assertEqual([str(item.path) for item in ti._collect_sources(catalog, ("root",))], ["child.py", "root.py"])

    def test_literal_import_alias_inside_try_retains_its_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = self.sources(directory, {
                "root.py": "try:\n from importlib import import_module as load\nexcept ImportError:\n raise\nload('child')\n",
                "child.py": "",
            })
            self.assertEqual([str(item.path) for item in ti._collect_sources(catalog, ("root",))], ["child.py", "root.py"])

    def test_computed_dependency_refuses_before_adapter_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = self.sources(directory, {
                "run_trigger_matrix.py": "import importlib\ndef load(name):\n return importlib.import_module(name)\n",
                "run_pi_trigger_eval.py": "", "trigger_identity.py": "",
            })
            with mock.patch.object(ti, "_source_catalog", return_value=catalog), mock.patch.object(tm.StubAdapter, "invoke") as invoke:
                with self.assertRaisesRegex(RuntimeError, "run_trigger_matrix:3: computed dependency"):
                    tm.run_matrix(MANIFEST, [{"id": "q", "query": "inspect the code", "should_trigger": True}], ["stub"], None, 1, 10, 1)
                invoke.assert_not_called()

    def test_missing_required_bytes_and_undeclared_named_sibling_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = self.sources(directory, {"root.py": "import child\n", "child.py": ""})
            (catalog.root / "child.py").unlink()
            with self.assertRaisesRegex(RuntimeError, "required trigger owner child"):
                ti._collect_sources(catalog, ("root",))
            (catalog.root / "child.py").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "child is absent from the packaging declaration"):
                ti._collect_sources(ti._Catalog(catalog.root, {"root": PurePosixPath("root.py")}), ("root",))

    def test_relocation_and_required_bytes_preserve_and_change_identity(self):
        files = {"root.py": "import child\n", "child.py": "VALUE = 1\n"}
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a, b = self.sources(first, files), self.sources(second, files)
            self.assertEqual(ti._collect_sources(a, ("root",)), ti._collect_sources(b, ("root",)))
            (b.root / "child.py").write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(ti._collect_sources(a, ("root",)), ti._collect_sources(b, ("root",)))

    def test_installed_external_dependencies_are_outside_the_owning_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = self.sources(directory, {"root.py": "import external\n"})
            (catalog.root / "external.py").write_text("", encoding="utf-8")
            installed = ti._Catalog(catalog.root, catalog.paths, direct_source=False)
            self.assertEqual([str(item.path) for item in ti._collect_sources(installed, ("root",))], ["root.py"])
            with self.assertRaisesRegex(RuntimeError, "external is absent from the packaging declaration"):
                ti._collect_sources(catalog, ("root",))

    def test_repeated_source_reads_recheck_named_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = self.sources(directory, {"root.py": "import external\n"})
            self.assertEqual(len(ti._collect_sources(catalog, ("root",))), 1)
            (catalog.root / "external.py").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "external is absent from the packaging declaration"):
                ti._collect_sources(catalog, ("root",))

    def test_saved_identity_validates_its_own_digests_without_replacing_them(self):
        identity = ti.trigger_harness_identity()
        payload = {"schema_version": 4, "modules": {name: "sha256:" + "0" * 64 for name in identity["modules"]}}
        saved = {**payload, "identity_sha256": canonical_json_sha256(payload)}
        self.assertIs(ti.validate_trigger_harness_identity(saved, "saved"), saved)
        changed = {**saved, "identity_sha256": "sha256:" + "1" * 64}
        with self.assertRaisesRegex(ValueError, "does not match"):
            ti.validate_trigger_harness_identity(changed, "saved")
        with self.assertRaisesRegex(RuntimeError, "changed during execution"):
            ti.assert_trigger_sources_unchanged(saved)

    def test_supported_reexports_and_captured_callbacks_have_canonical_owners(self):
        self.assertIs(sb.invoke_argv_with_timeout, process_runtime.invoke_argv_with_timeout)
        self.assertIs(sb.AblationError, skill_inputs.AblationError)
        self.assertIs(sb.TraceDialect, trace_normalization.TraceDialect)
        self.assertIs(tm.PiStream, trace_normalization.PiStream)
        for dialect in trace_normalization.TRACE_DIALECTS.values():
            for attribute in ("flatten", "stream_semantics", "usage_and_cost", "protocol_error", "retries"):
                self.assertEqual(getattr(dialect, attribute).__module__, "trace_normalization")

    def test_trigger_imports_exclude_answer_owners_and_loaded_files_are_identified(self):
        program = """import json,sys
from pathlib import Path
import run_trigger_matrix,run_pi_trigger_eval,trigger_identity
root=Path(trigger_identity.__file__).resolve().parent
loaded={Path(value.__file__).resolve().relative_to(root).as_posix() for value in sys.modules.values() if getattr(value,'__file__',None) and Path(value.__file__).resolve().parent==root}
inventory=set(trigger_identity.trigger_harness_identity()['modules'])
assert loaded <= inventory, (loaded-inventory)
assert 'skill_benchmark' not in sys.modules
assert 'report_contracts.py' not in inventory
print(json.dumps({'loaded': sorted(loaded), 'inventory': sorted(inventory)}))
"""
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, capture_output=True, text=True, timeout=30, check=True)
        self.assertGreater(len(json.loads(result.stdout)["loaded"]), 0)

    def test_source_change_keeps_settled_spend_and_rows_but_blocks_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(tm, "assert_trigger_sources_unchanged", side_effect=RuntimeError("changed source")):
                with self.assertRaises(tm.MatrixRunStopped) as stopped:
                    tm.run_matrix(MANIFEST, [{"id": "q", "query": "inspect the code", "should_trigger": True}], ["stub"], None, 1, 10, 1,
                                  spend_policy=SpendPolicy.from_raw(1), spend_root=Path(directory))
            report = stopped.exception.report
            self.assertEqual(len(report["results"]), 1)
            self.assertEqual(report["spend"]["snapshot"], "persisted")
            self.assertEqual(report["evidence_class"], "incomplete_trigger_measurement")
            with mock.patch.object(sb, "die", side_effect=ValueError) as reject:
                with self.assertRaises(ValueError):
                    sb._trigger_report_rows(report, "partial")
                self.assertIn("not a skill-trigger-matrix report", reject.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
