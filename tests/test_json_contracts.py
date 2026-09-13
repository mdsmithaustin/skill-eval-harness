"""The two JSON boundaries in json_contracts: the strict artifact rule and the
external-CLI stream rule differ only in how a repeated object key is treated."""
from __future__ import annotations

import json
import unittest

from json_contracts import (
    StreamJSONParse,
    parse_stream_json,
    stream_json_loads,
    strict_json_loads,
)

# The shape `codex exec --json` emits when it repeats `id` inside an item.
CODEX_DUPLICATE_ID_LINE = (
    '{"type":"item.completed","item":{"id":"item_0","type":"command_execution",'
    '"command":"ls","aggregated_output":"README.md\\n","exit_code":0,'
    '"status":"completed","id":"item_0"}}'
)


class StrictArtifactRuleTests(unittest.TestCase):
    def test_duplicate_object_key_is_rejected(self):
        with self.assertRaises(json.JSONDecodeError) as ctx:
            strict_json_loads(CODEX_DUPLICATE_ID_LINE)
        self.assertIn("duplicate object key: 'id'", ctx.exception.msg)

    def test_non_finite_constant_is_rejected(self):
        with self.assertRaises(json.JSONDecodeError):
            strict_json_loads('{"value": NaN}')


class StreamRuleTests(unittest.TestCase):
    def test_duplicate_key_resolves_last_value_wins_and_is_reported(self):
        parsed = parse_stream_json('{"id": "first", "n": 1, "id": "last", "id": "final"}')
        self.assertIsInstance(parsed, StreamJSONParse)
        self.assertEqual(parsed.value, {"id": "final", "n": 1})   # same rule as json.loads
        self.assertEqual(parsed.duplicate_keys, ("id", "id"))    # one entry per repeat
        self.assertEqual(json.loads('{"id": "first", "id": "final"}'), {"id": "final"})

    def test_nested_duplicates_are_reported_in_document_order(self):
        parsed = parse_stream_json('{"a": {"k": 1, "k": 2}, "b": [{"z": 0, "z": 1}], "a": 3}')
        self.assertEqual(parsed.value, {"a": 3, "b": [{"z": 1}]})
        self.assertEqual(parsed.duplicate_keys, ("k", "z", "a"))

    def test_codex_duplicate_id_line_parses_without_raising(self):
        parsed = parse_stream_json(CODEX_DUPLICATE_ID_LINE)
        self.assertEqual(parsed.value["item"]["id"], "item_0")
        self.assertEqual(parsed.value["item"]["type"], "command_execution")
        self.assertEqual(parsed.duplicate_keys, ("id",))
        self.assertEqual(stream_json_loads(CODEX_DUPLICATE_ID_LINE), parsed.value)

    def test_clean_input_reports_no_duplicates(self):
        parsed = parse_stream_json('{"type": "turn.started"}')
        self.assertEqual(parsed.value, {"type": "turn.started"})
        self.assertEqual(parsed.duplicate_keys, ())

    def test_stream_rule_keeps_every_other_strict_check(self):
        # Only the duplicate-key rule is relaxed: the value must still survive
        # strict re-serialization into a harness artifact.
        for text in ('{"value": NaN}', '{"value": Infinity}', "not json", ""):
            with self.subTest(text=text), self.assertRaises(json.JSONDecodeError):
                stream_json_loads(text)
        with self.assertRaises(json.JSONDecodeError):
            strict_json_loads('{"value": NaN}')


if __name__ == "__main__":
    unittest.main()
