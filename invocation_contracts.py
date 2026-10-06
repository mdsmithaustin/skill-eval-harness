"""Provider-neutral request, process-plan, and process-result contracts."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from json_contracts import freeze_json_mapping, strict_json_loads, validate_json_text
from manifest_contracts import ModelId


class CheckpointMatch(str, Enum):
    BYTES = "bytes"
    JSON = "json"


@dataclass(frozen=True)
class RecoveryCase:
    checkpoint_path: str
    expected_content: bytes
    recovery_prompt: str
    refusal_prompt: str
    forbidden_path: str
    match: CheckpointMatch = CheckpointMatch.BYTES

    @classmethod
    def parse(cls, raw: Any) -> RecoveryCase:
        if not isinstance(raw, dict):
            raise ValueError("recovery must be an object")
        required = {"checkpoint_path", "expected_content", "recovery_prompt",
                    "refusal_prompt", "forbidden_path"}
        if set(raw) - required - {"match"} or required - set(raw):
            raise ValueError("recovery requires checkpoint_path, expected_content, "
                             "recovery_prompt, refusal_prompt and forbidden_path")
        for key in required:
            value = raw[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"recovery.{key} must be non-empty text")
            validate_json_text(value, f"recovery.{key}")
        for key in ("checkpoint_path", "forbidden_path"):
            path = Path(raw[key])
            if path.is_absolute() or path == Path(".") or ".." in path.parts or "\x00" in raw[key]:
                raise ValueError(f"recovery.{key} must be a safe non-root relative path")
        if Path(raw["checkpoint_path"]) == Path(raw["forbidden_path"]):
            raise ValueError("recovery paths must be distinct")
        try:
            match = CheckpointMatch(raw.get("match", "bytes"))
        except (ValueError, TypeError) as exc:
            raise ValueError("recovery.match must be bytes or json") from exc
        expected = raw["expected_content"].encode("utf-8")
        if match is CheckpointMatch.JSON:
            try:
                strict_json_loads(expected)
            except (ValueError, UnicodeError) as exc:
                raise ValueError("recovery.expected_content must be valid JSON for json matching") from exc
        return cls(raw["checkpoint_path"], expected, raw["recovery_prompt"],
                   raw["refusal_prompt"], raw["forbidden_path"], match)

    def as_dict(self) -> dict[str, str]:
        return {"checkpoint_path": self.checkpoint_path,
                "expected_content": self.expected_content.decode("utf-8"),
                "recovery_prompt": self.recovery_prompt,
                "refusal_prompt": self.refusal_prompt,
                "forbidden_path": self.forbidden_path, "match": self.match.value}

    def matches(self, content: bytes) -> bool:
        if self.match is CheckpointMatch.BYTES:
            return content == self.expected_content
        def equivalent(actual: Any, expected: Any) -> bool:
            if isinstance(actual, bool) != isinstance(expected, bool):
                return False
            if isinstance(actual, dict) and isinstance(expected, dict):
                return actual.keys() == expected.keys() and all(
                    equivalent(actual[key], expected[key]) for key in actual)
            if isinstance(actual, list) and isinstance(expected, list):
                return len(actual) == len(expected) and all(
                    equivalent(actual_item, expected_item)
                    for actual_item, expected_item in zip(actual, expected, strict=True))
            return actual == expected
        try:
            actual = strict_json_loads(content)
            expected = strict_json_loads(self.expected_content)
        except (ValueError, UnicodeError):
            return False
        return equivalent(actual, expected)


class RecoveryProcessState(str, Enum):
    CHECKPOINT_STOP = "checkpoint_stop"
    COMPLETE = "complete"
    DEADLINE = "deadline"
    NATURAL_COMPLETION = "natural_completion"
    CHECKPOINT_MISMATCH = "checkpoint_mismatch"
    OBSERVER_FAILED = "observer_failed"
    CAPTURE_FAILED = "capture_failed"
    SIGNAL_FAILED = "signal_failed"
    CLEANUP_FAILED = "cleanup_failed"
    SPAWN_FAILED = "spawn_failed"
    PROCESS_FAILED = "process_failed"


@dataclass
class RecoveryCapture:
    directory: Path
    checkpoint: RecoveryCase | None = None
    state: RecoveryProcessState | None = field(default=None, init=False)


class InvocationState(str, Enum):
    """Closed lifecycle vocabulary shared by process and semantic adapters.

    ``InvocationResult`` admits only the four process-boundary states. Provider
    and harness adapters may use the two semantic failure states without
    rewriting the subprocess return code that produced their evidence.
    """

    COMPLETE = "complete"
    TIMED_OUT = "timed_out"
    SPAWN_FAILED = "spawn_failed"
    PROCESS_FAILED = "process_failed"
    PROVIDER_FAILED = "provider_failed"
    HARNESS_FAILED = "harness_failed"

    @property
    def reached_exit(self) -> bool:
        """Whether a provider process was spawned and exited on its own, with
        success or failure; a timeout or spawn failure never observed an exit."""
        return self in {InvocationState.COMPLETE, InvocationState.PROCESS_FAILED,
                        InvocationState.PROVIDER_FAILED}


def validate_invocation_lifecycle(
    state: InvocationState,
    returncode: int | None,
    provider_error: str | None = None,
    *,
    allow_harness_failure: bool = False,
    allow_nonzero_completion: bool = False,
) -> None:
    """Validate the shared process/provider lifecycle discriminant.

    A provider failure is an exit-zero response/protocol failure.  A nonzero
    exit remains a process failure even when provider diagnostics explain it.
    """
    if not isinstance(state, InvocationState):
        raise TypeError("invocation state must be InvocationState")
    if returncode is not None and type(returncode) is not int:
        raise TypeError("invocation returncode must be an integer or None")
    if provider_error is not None and (
        not isinstance(provider_error, str) or not provider_error.strip()
    ):
        raise ValueError("provider_error must be a non-empty string")

    if state is InvocationState.COMPLETE:
        if returncode != 0 and not (
            allow_nonzero_completion
            and returncode is not None
            and returncode != 0
        ):
            raise ValueError("complete invocation requires returncode 0")
        if provider_error is not None:
            raise ValueError("complete invocation cannot carry a provider error")
    elif state is InvocationState.TIMED_OUT:
        if returncode != 124:
            raise ValueError("timed-out invocation requires returncode 124")
        if provider_error is not None:
            raise ValueError("timed-out invocation cannot carry a provider error")
    elif state is InvocationState.SPAWN_FAILED:
        if returncode != 127:
            raise ValueError("spawn-failed invocation requires returncode 127")
        if provider_error is not None:
            raise ValueError("spawn-failed invocation cannot carry a provider error")
    elif state is InvocationState.PROCESS_FAILED:
        if returncode in {None, 0}:
            raise ValueError("process-failed invocation requires nonzero returncode")
    elif state is InvocationState.PROVIDER_FAILED:
        if returncode != 0 or provider_error is None:
            raise ValueError(
                "provider-failed invocation requires returncode 0 and provider_error")
    elif state is InvocationState.HARNESS_FAILED:
        if not allow_harness_failure or returncode is not None:
            raise ValueError("process invocation cannot represent harness failure")
        if provider_error is not None:
            raise ValueError("harness failure cannot carry a provider error")


class TimeoutSeconds(int):
    """A positive provider invocation timeout."""

    def __new__(cls, value: int) -> TimeoutSeconds:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("invocation timeout must be an integer")
        if value < 1:
            raise ValueError("invocation timeout must be positive")
        return int.__new__(cls, value)

    @classmethod
    def parse(cls, value: object) -> TimeoutSeconds:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("invocation timeout must be an integer")
        return cls(value)


@dataclass(frozen=True)
class InvocationRequest:
    """Provider-neutral request handed to an answer backend."""

    prompt: str
    workspace: Path
    model: ModelId | None
    timeout_s: TimeoutSeconds
    # Requested reasoning effort, or None for the backend's default. A backend
    # applies it through its own control (a CLI flag or config override); the
    # runner refuses the request before any spend when a backend has none.
    effort: str | None = None
    recovery_capture: RecoveryCapture | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise TypeError("invocation prompt must be text")
        validate_json_text(self.prompt, "invocation prompt")
        if not isinstance(self.workspace, Path):
            raise TypeError("invocation workspace must be a Path")
        object.__setattr__(
            self,
            "model",
            None if self.model is None else ModelId.parse(self.model),
        )
        if self.model is not None:
            validate_json_text(self.model, "invocation model")
        object.__setattr__(
            self, "timeout_s", TimeoutSeconds.parse(self.timeout_s)
        )
        if self.effort is not None and (
                not isinstance(self.effort, str) or not self.effort.strip()):
            raise ValueError("invocation effort must be a non-empty string or None")

    @classmethod
    def parse(
        cls,
        *,
        prompt: object,
        workspace: object,
        model: object,
        timeout_s: object,
        effort: object = None,
        recovery_capture: RecoveryCapture | None = None,
    ) -> InvocationRequest:
        if not isinstance(prompt, str):
            raise ValueError("invocation prompt must be text")
        if not isinstance(workspace, Path):
            raise ValueError("invocation workspace must be a Path")
        if effort is not None and not isinstance(effort, str):
            raise ValueError("invocation effort must be a string or None")
        return cls(
            prompt=prompt,
            workspace=workspace,
            model=None if model is None else ModelId.parse(model),
            timeout_s=TimeoutSeconds.parse(timeout_s),
            effort=effort,
            recovery_capture=recovery_capture,
        )


@dataclass(frozen=True)
class ProcessInvocationPlan:
    """Everything the subprocess owner needs, validated before spawn."""

    argv: tuple[str, ...]
    input_text: str | None
    cwd: Path
    timeout_s: TimeoutSeconds
    environment: Mapping[str, str] | None = None
    # Applied to the whole captured stdout and stderr before the owner caps
    # them, so a redacted span can never be cut in half first.
    redact_output: Callable[[str], str] | None = field(default=None, repr=False)
    # False leaves stdout as captured when it is the run's evidence.
    redact_stdout: bool = True
    recovery_capture: RecoveryCapture | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.argv, tuple):
            raise TypeError("process argv must be a tuple")
        if not self.argv or not self.argv[0]:
            raise ValueError("process argv needs a non-empty executable")
        if any(not isinstance(value, str) for value in self.argv):
            raise TypeError("process argv values must be strings")
        for position, value in enumerate(self.argv):
            validate_json_text(value, f"process argv[{position}]")
            if "\x00" in value:
                raise ValueError("process argv values cannot contain NUL")
        if self.input_text is not None:
            if not isinstance(self.input_text, str):
                raise TypeError("process stdin must be text or None")
            validate_json_text(self.input_text, "process stdin")
        if not isinstance(self.cwd, Path):
            raise TypeError("process cwd must be a Path")
        object.__setattr__(
            self, "timeout_s", TimeoutSeconds.parse(self.timeout_s)
        )
        if self.environment is not None:
            if not isinstance(self.environment, Mapping):
                raise TypeError("process environment must be a mapping or None")
            copied: dict[str, str] = {}
            for key, value in self.environment.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise TypeError("process environment keys and values must be strings")
                if not key or "=" in key or "\x00" in key or "\x00" in value:
                    raise ValueError("process environment contains an invalid key or value")
                copied[key] = value
            object.__setattr__(self, "environment", MappingProxyType(copied))
        if self.redact_output is not None and not callable(self.redact_output):
            raise TypeError("process redact_output must be callable or None")
        if not isinstance(self.redact_stdout, bool):
            raise TypeError("process redact_stdout must be boolean")

    @classmethod
    def from_values(
        cls,
        argv: Sequence[str],
        *,
        input_text: str | None,
        cwd: Path | str,
        timeout_s: int,
        environment: Mapping[str, str] | None = None,
        redact_output: Callable[[str], str] | None = None,
        redact_stdout: bool = True,
        recovery_capture: RecoveryCapture | None = None,
    ) -> ProcessInvocationPlan:
        if isinstance(argv, (str, bytes)):
            raise TypeError("process argv must be a sequence of argument strings")
        return cls(
            argv=tuple(argv),
            input_text=input_text,
            cwd=Path(cwd),
            timeout_s=TimeoutSeconds(timeout_s),
            environment=environment,
            redact_output=redact_output,
            redact_stdout=redact_stdout,
            recovery_capture=recovery_capture,
        )


@dataclass(frozen=True)
class InvocationResult:
    stdout: str
    stderr: str
    returncode: int
    elapsed_ms: int
    invocation_state: InvocationState
    stdout_utf8_valid: bool
    stderr_utf8_valid: bool
    timed_out: bool = False
    adapter_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("invocation stdout and stderr must be text")
        validate_json_text(self.stdout, "invocation stdout")
        validate_json_text(self.stderr, "invocation stderr")
        if type(self.returncode) is not int:
            raise TypeError("invocation returncode must be an integer")
        if (isinstance(self.elapsed_ms, bool) or not isinstance(self.elapsed_ms, int)
                or self.elapsed_ms < 0 or self.elapsed_ms > 2**63 - 1):
            raise ValueError("invocation elapsed_ms must be a non-negative integer")
        if not isinstance(self.invocation_state, InvocationState):
            raise TypeError("invocation_state must be InvocationState")
        if not isinstance(self.stdout_utf8_valid, bool) or not isinstance(
                self.stderr_utf8_valid, bool):
            raise TypeError("invocation UTF-8 validity fields must be boolean")
        if not isinstance(self.timed_out, bool):
            raise TypeError("invocation timed_out must be boolean")
        if self.invocation_state not in {
            InvocationState.COMPLETE,
            InvocationState.TIMED_OUT,
            InvocationState.SPAWN_FAILED,
            InvocationState.PROCESS_FAILED,
        }:
            raise ValueError("InvocationResult requires a process-boundary state")
        validate_invocation_lifecycle(self.invocation_state, self.returncode)
        if self.timed_out is not (
            self.invocation_state is InvocationState.TIMED_OUT
        ):
            raise ValueError("invocation state contradicts timed_out")
        if self.adapter_metadata is not None:
            frozen_metadata = freeze_json_mapping(
                self.adapter_metadata, "invocation adapter metadata")
            expected_flags = {
                "stdout_utf8_valid": self.stdout_utf8_valid,
                "stderr_utf8_valid": self.stderr_utf8_valid,
            }
            for key, expected_value in expected_flags.items():
                if key in frozen_metadata and frozen_metadata[key] is not expected_value:
                    raise ValueError(
                        f"invocation adapter metadata contradicts {key}")
            object.__setattr__(self, "adapter_metadata", frozen_metadata)
