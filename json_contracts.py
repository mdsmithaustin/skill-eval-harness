"""Lossless, immutable JSON values for persisted evidence contracts."""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn


class FrozenJSONDict(dict[str, Any]):
    """A JSON-serializable mapping that cannot be mutated after validation."""

    def _immutable(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("frozen JSON mapping cannot be mutated")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _immutable


def _validate_json_string(value: str, label: str) -> None:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} contains a surrogate code point") from exc


def validate_json_text(value: str, label: str) -> None:
    """Validate free text destined for a strict UTF-8 JSON/text artifact."""
    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    _validate_json_string(value, label)


def validate_json_value(value: Any, label: str) -> None:
    """Prove a value can be emitted by strict UTF-8 JSON serialization."""
    _validate_json_value(value, label, depth=0, ancestors=set())


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise json.JSONDecodeError(f"duplicate object key: {key!r}", "", 0)
        result[key] = value
    return result


def _persistable_json_loads(
    value: str | bytes | bytearray,
    object_pairs_hook: Callable[[list[tuple[str, Any]]], dict[str, Any]],
    label: str,
) -> Any:
    def reject_constant(constant: str) -> Any:
        raise json.JSONDecodeError(
            f"non-finite numeric constant is not valid JSON: {constant}", "", 0)

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=object_pairs_hook,
            parse_constant=reject_constant,
        )
        validate_json_value(parsed, label)
        return parsed
    except json.JSONDecodeError:
        raise
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise json.JSONDecodeError(str(exc), "", 0) from exc


def strict_json_loads(value: str | bytes | bytearray) -> Any:
    """Parse strict, persistable JSON without last-key-wins shadowing.

    This is the boundary for artifacts the harness authors or validates
    (manifests, prepared tasks, run sidecars, judge rows, reports): a repeated
    object key there is a defect in the artifact and is rejected. Bytes an
    external agent CLI wrote go through `parse_stream_json` instead.
    """
    return _persistable_json_loads(value, unique_json_object, "strict JSON")


@dataclass(frozen=True)
class StreamJSONParse:
    """One parsed line of an external CLI stream plus the duplicate object
    keys the last-value-wins rule resolved, in document order."""

    value: Any
    duplicate_keys: tuple[str, ...]


def parse_stream_json(value: str | bytes | bytearray) -> StreamJSONParse:
    """Parse one JSON value an external agent CLI emitted on its live stream.

    The harness does not author those bytes, so a repeated object key is not
    an artifact defect to reject: `codex exec --json` repeats `id` on some
    event lines, and rejecting the line threw away the whole row. Duplicate
    keys resolve last-value-wins, the same rule as stdlib `json.loads`, and the
    repeated key names are returned so a row can record that the lenient rule
    was exercised. Everything else is as strict as `strict_json_loads`:
    non-finite constants and non-persistable values are still rejected because
    the parsed value is re-serialized into harness artifacts.
    """
    duplicates: list[str] = []

    def last_value_wins(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = item
        return result

    parsed = _persistable_json_loads(value, last_value_wins, "stream JSON")
    return StreamJSONParse(parsed, tuple(duplicates))


def stream_json_loads(value: str | bytes | bytearray) -> Any:
    """`parse_stream_json` for callers that only need the value."""
    return parse_stream_json(value).value


def _validate_json_value(
    value: Any, label: str, *, depth: int, ancestors: set[int],
) -> None:
    if depth > 100:
        raise ValueError(f"{label} exceeds the maximum JSON nesting depth")
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{label} contains a cyclic JSON object")
        ancestors.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{label} object keys must be strings")
                _validate_json_string(key, f"{label} object key")
                _validate_json_value(
                    item, f"{label}.{key}", depth=depth + 1,
                    ancestors=ancestors)
        finally:
            ancestors.remove(identity)
        return
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{label} contains a cyclic JSON array")
        ancestors.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_json_value(
                    item, f"{label}[{index}]", depth=depth + 1,
                    ancestors=ancestors)
        finally:
            ancestors.remove(identity)
        return
    if isinstance(value, str):
        _validate_json_string(value, label)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must not contain non-finite numbers")
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            json.dumps(value, allow_nan=False)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{label} integer is too large for JSON serialization") from exc
        return
    if value is None or isinstance(value, (float, bool)):
        return
    raise TypeError(f"{label} contains non-JSON value {type(value).__name__}")


def freeze_json_value(value: Any, label: str) -> Any:
    """Validate and recursively freeze one strict-JSON value.

    Object keys are never coerced: coercion can collapse distinct evidence such
    as ``1`` and ``"1"``. Non-finite floats are rejected at construction so a
    value accepted by the typed domain is guaranteed to survive strict JSON
    serialization later.
    """
    validate_json_value(value, label)
    return _freeze_valid_json_value(value, label)


def _freeze_valid_json_value(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            frozen[key] = _freeze_valid_json_value(item, f"{label}.{key}")
        return FrozenJSONDict(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_valid_json_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    return value


def freeze_json_mapping(
    value: Mapping[str, Any] | None,
    label: str,
) -> Mapping[str, Any]:
    if value is None:
        return FrozenJSONDict()
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen = freeze_json_value(value, label)
    if not isinstance(frozen, Mapping):  # pragma: no cover - established above
        raise TypeError(f"{label} must be a mapping")
    return frozen


def thaw_json_value(value: Any, label: str = "frozen JSON value") -> Any:
    """Return a detached mutable JSON wire shape from a validated frozen value."""
    validate_json_value(value, label)
    if isinstance(value, Mapping):
        return {
            key: thaw_json_value(item, f"{label}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            thaw_json_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


def strict_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int or int/float coercions."""
    left_frozen = freeze_json_value(left, "left JSON value")
    right_frozen = freeze_json_value(right, "right JSON value")
    left_wire = json.dumps(
        left_frozen, sort_keys=True, separators=(",", ":"), allow_nan=False)
    right_wire = json.dumps(
        right_frozen, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return left_wire == right_wire
