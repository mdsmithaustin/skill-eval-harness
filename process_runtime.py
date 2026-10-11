from __future__ import annotations

import errno
import functools
import hashlib
import json
import os
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass as _dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from invocation_contracts import (
    ProcessInvocationPlan,
    RawProcessFacts,
    RecoveryCapture,
    RecoveryProcessState,
)
from json_contracts import freeze_json_mapping, strict_json_loads, write_json
from trigger_contracts import InvocationOutcome


class _BridgeJournal(Protocol):
    def offer_bridge(self, frame: Mapping[str, Any], frame_limit: int = 64 * 1024) -> dict[str, Any]: ...
    def acknowledge_bridge(self, frame: Mapping[str, Any]) -> None: ...

PROCESS_LEADER_POLL_INTERVAL_S = 0.05


PROCESS_PIPE_DRAIN_GRACE_S = 0.25


AGENT_STOP_GRACE_S = 2.0


def recovery_file_bytes(workspace: Path, relative: str) -> bytes:
    path = workspace / relative
    resolved = path.resolve()
    if workspace.resolve() not in resolved.parents:
        raise ValueError(f"recovery path escapes workspace: {relative}")
    parts = Path(relative).parts
    directory_flag, nofollow_flag, nonblock_flag = (
        getattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"))
    directory_flags = os.O_RDONLY | directory_flag | nofollow_flag
    file_flags = os.O_RDONLY | nofollow_flag | nonblock_flag
    descriptors = [os.open(workspace, directory_flags)]
    try:
        for part in parts[:-1]:
            descriptors.append(os.open(part, directory_flags, dir_fd=descriptors[-1]))
        descriptor = os.open(parts[-1], file_flags, dir_fd=descriptors[-1])
        descriptors.append(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"recovery path is not a regular file: {relative}")
            return handle.read()
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def recovery_group_stopped(pgid: int) -> tuple[bool, str]:
    if Path("/proc").is_dir():
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                return False, "procfs_live_member"
        return True, "procfs_no_live_members"
    killpg = getattr(os, "killpg", None)
    if not callable(killpg):
        return False, "unsupported"
    try:
        killpg(pgid, 0)
    except ProcessLookupError:
        return True, "killpg_no_members"
    return False, "killpg_members_remain"


_stop_signal: int | None = None


def _stop_run(signum: int, frame: object) -> None:
    global _stop_signal
    if _stop_signal is None:
        _stop_signal = signum
        raise KeyboardInterrupt


def stops_on_signal(main: Callable[[], int]) -> Callable[[], int]:
    """Make SIGINT and SIGTERM stop a CLI entrypoint and every agent it runs.

    The entrypoint then prints one line and returns 128 + the signal number. A
    signal ignored at startup stays ignored."""
    @functools.wraps(main)
    def run() -> int:
        global _stop_signal
        _stop_signal = None
        previous = {signum: signal.signal(signum, _stop_run)
                    for signum in (signal.SIGINT, signal.SIGTERM)
                    if signal.getsignal(signum) not in (signal.SIG_IGN, None)}
        try:
            return main()
        except KeyboardInterrupt:
            if _stop_signal is None:
                raise
            print(f"stopped by {signal.Signals(_stop_signal).name}", file=sys.stderr)
            return 128 + _stop_signal
        finally:
            _stop_signal = None
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    return run


def skill_benchmark_stopped() -> bool:
    return _stop_signal is not None


@_dataclass(frozen=True)
class _BridgeLimits:
    frame_bytes: int = 64 * 1024
    stream_bytes: int = 8 * 1024 * 1024
    outbound_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        for value, maximum in ((self.frame_bytes, 64 * 1024),
                               (self.stream_bytes, 8 * 1024 * 1024),
                               (self.outbound_bytes, 8 * 1024 * 1024)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("tool bridge limits must be positive integers within the default bounds")


@_dataclass(frozen=True)
class _BridgeSession:
    journal: _BridgeJournal
    turn: int
    directory: Path
    limits: _BridgeLimits
    validate_final: Callable[[Any], dict[str, Any]]


@_dataclass(frozen=True)
class _BridgeProcessFacts:
    pid: int | None
    os_returncode: int | None
    natural_exit: bool
    group_stopped: bool
    group_observation: str
    drained: bool
    reaped: bool
    timed_out: bool
    forced_termination: bool
    stream_bytes: int
    cleanup_bytes: int

    @property
    def incomplete_billing(self) -> bool:
        return self.pid is not None and (self.forced_termination or not self.group_stopped or not self.drained or not self.reaped)

    def as_dict(self) -> dict[str, Any]:
        return {"pid": self.pid, "os_returncode": self.os_returncode,
                "natural_exit": self.natural_exit, "group_stopped": self.group_stopped,
                "group_observation": self.group_observation, "drained": self.drained,
                "reaped": self.reaped, "timed_out": self.timed_out,
                "forced_termination": self.forced_termination,
                "stream_bytes": self.stream_bytes, "cleanup_bytes": self.cleanup_bytes}


@_dataclass(frozen=True)
class _BridgeStoppedAttempt:
    outcome: InvocationOutcome
    final: Mapping[str, Any] | None
    error: str | None
    interruption: BaseException | None
    facts: _BridgeProcessFacts


@_dataclass(frozen=True)
class _ToolBridgeBackend:
    argv: tuple[str, ...]
    timeout: int
    limits: _BridgeLimits


def _invoke_tool_dialogue(plan: ProcessInvocationPlan, session: _BridgeSession) -> _BridgeStoppedAttempt:
    if not _bridge_supported():
        raise ValueError("tool bridge requires POSIX nonreaping waitid, process-group controls, and Linux procfs")
    result = _invoke_process(plan, session)
    if not isinstance(result, _BridgeStoppedAttempt):
        raise TypeError("dialogue invocation returned an ordinary outcome")
    return result


def _bridge_supported() -> bool:
    return sys.platform == "linux" and Path("/proc/self/stat").is_file() and all(callable(getattr(os, name, None)) for name in
        ("waitid", "killpg", "set_blocking")) and all(isinstance(getattr(os, name, None), int)
        for name in ("P_PID", "WEXITED", "WNOHANG", "WNOWAIT"))


def _bridge_frame_bytes(frame: Mapping[str, Any], limit: int) -> bytes:
    remaining = limit - 1
    pending: list[Iterator[Any]] = [iter((frame,))]
    while pending:
        try:
            value = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        if isinstance(value, str):
            remaining -= len(value) + 2
        elif isinstance(value, dict):
            remaining -= 2 + len(value) + max(0, len(value) - 1)
            pending.append(part for pair in value.items() for part in pair)
        elif isinstance(value, (list, tuple)):
            remaining -= 2 + max(0, len(value) - 1)
            pending.append(iter(value))
        elif isinstance(value, int) and not isinstance(value, bool):
            remaining -= max(1, value.bit_length() // 4) + (value < 0)
        else:
            remaining -= 1
        if remaining < 0:
            raise ValueError("tool bridge outbound frame exceeds configured bound")
    encoded = bytearray()
    for text in json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":")).iterencode(frame):
        for index in range(0, len(text), 4096):
            chunk = text[index:index + 4096].encode("utf-8", errors="strict")
            if len(encoded) + len(chunk) + 1 > limit:
                raise ValueError("tool bridge outbound frame exceeds configured bound")
            encoded.extend(chunk)
    encoded.extend(b"\n")
    return bytes(encoded)


def invoke_argv_with_timeout(plan: ProcessInvocationPlan) -> InvocationOutcome:
    """Typed subprocess owner for every spawned runner/adapter process.

    The owner accepts one fully validated plan, so internal callers cannot
    bypass cwd, timeout, argv, environment, or stdin construction. Completion
    state is classified once here; consumers cannot independently assemble
    contradictory returncode/timeout/completeness booleans."""
    result = _invoke_process(plan)
    if not isinstance(result, InvocationOutcome):
        raise TypeError("ordinary invocation returned a dialogue attempt")
    return result


def _invoke_process(plan: ProcessInvocationPlan, dialogue: _BridgeSession | None = None) -> InvocationOutcome | _BridgeStoppedAttempt:
    if not isinstance(plan, ProcessInvocationPlan):
        raise TypeError("invoke_argv_with_timeout requires a ProcessInvocationPlan")
    argv = list(plan.argv)
    cwd = plan.cwd
    env = None if plan.environment is None else dict(plan.environment)
    timeout = int(plan.timeout_s)
    input_text = plan.input_text
    redact_stderr = plan.redact_output or (lambda text: text)
    redact_stdout = redact_stderr if plan.redact_stdout else (lambda text: text)

    def retain_raw(out: bytes | str | None, err: bytes | str | None,
                   returncode: int | None, *, timed_out: bool,
                   communication_complete: bool = False) -> None:
        if plan.raw_capture is not None:
            plan.raw_capture.facts = RawProcessFacts(
                out, err, returncode, int((time.time() - start) * 1000), timed_out,
                communication_complete)

    def _wire_text(value: Any, redact: Callable[[str], str]) -> tuple[str, bool]:
        if value is None:
            return "", True
        if isinstance(value, bytes):
            try:
                return redact(value.decode("utf-8", errors="strict")), True
            except UnicodeDecodeError:
                return redact(value.decode("utf-8", errors="backslashreplace")), False
        return redact(str(value)), True

    def kill_process_group(pgid: int) -> dict[str, Any]:
        killpg = getattr(os, "killpg", None)
        sigkill = getattr(signal, "SIGKILL", None)
        if not callable(killpg) or not isinstance(sigkill, int):
            return {"status": "unsupported"}
        try:
            killpg(pgid, sigkill)
        except ProcessLookupError:
            return {"status": "not_needed"}
        except OSError as exc:
            return {"status": "warning", "signal": "SIGKILL",
                    "error": errno.errorcode.get(exc.errno, type(exc).__name__)}
        return {"status": "kill_sent", "signal": "SIGKILL"}

    def stop_session(proc: subprocess.Popen[bytes], signum: int) -> None:
        killpg = getattr(os, "killpg", None)
        sigkill = getattr(signal, "SIGKILL", None)
        if not callable(killpg) or not isinstance(sigkill, int):
            proc.kill()
            return
        try:
            killpg(proc.pid, signum)
            deadline = time.monotonic() + AGENT_STOP_GRACE_S
            while not process_leader_exited(proc) and time.monotonic() < deadline:
                time.sleep(PROCESS_LEADER_POLL_INTERVAL_S)
            killpg(proc.pid, sigkill)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=PROCESS_PIPE_DRAIN_GRACE_S)
        except subprocess.TimeoutExpired:
            pass

    def process_leader_exited(proc: subprocess.Popen[bytes]) -> bool:
        if proc.returncode is not None:
            return True
        waitid = getattr(os, "waitid", None)
        p_pid = getattr(os, "P_PID", None)
        wexited = getattr(os, "WEXITED", None)
        wnohang = getattr(os, "WNOHANG", None)
        wnowait = getattr(os, "WNOWAIT", None)
        if (hasattr(os, "killpg") and callable(waitid)
                and isinstance(p_pid, int)
                and isinstance(wexited, int)
                and isinstance(wnohang, int)
                and isinstance(wnowait, int)):
            try:
                status = waitid(p_pid, proc.pid,
                                wexited | wnohang | wnowait)
            except (ChildProcessError, OSError):
                if dialogue is not None:
                    return False
            else:
                return status is not None
        return proc.poll() is not None

    def observe_recovery_process(proc: subprocess.Popen[bytes], capture: RecoveryCapture,
                                 input_bytes: bytes | None, start: float) -> InvocationOutcome:
        deadline = time.monotonic() + timeout
        state = RecoveryProcessState.COMPLETE
        error: str | None = None
        observed: bytes | None = None
        out, err = b"", b""
        started = False
        drained = False
        signal_sent: int | None = None
        checkpoint_live = False
        interruption: BaseException | None = None
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    state = RecoveryProcessState.DEADLINE
                    break
                if process_leader_exited(proc):
                    state = (RecoveryProcessState.NATURAL_COMPLETION if capture.checkpoint
                             else RecoveryProcessState.COMPLETE)
                    break
                if capture.checkpoint is not None:
                    try:
                        candidate = recovery_file_bytes(cwd, capture.checkpoint.checkpoint_path)
                    except FileNotFoundError:
                        candidate = None
                    if candidate is not None:
                        if not capture.checkpoint.matches(candidate):
                            state = RecoveryProcessState.CHECKPOINT_MISMATCH
                            break
                        observed = candidate
                        try:
                            (capture.directory / "checkpoint-observed.bin").write_bytes(observed)
                        except OSError as exc:
                            state = RecoveryProcessState.CAPTURE_FAILED
                            error = f"{type(exc).__name__}: {exc}"
                            break
                        if time.monotonic() >= deadline:
                            state = RecoveryProcessState.DEADLINE
                            break
                        if process_leader_exited(proc):
                            state = RecoveryProcessState.NATURAL_COMPLETION
                            break
                        checkpoint_live = True
                        state = RecoveryProcessState.CHECKPOINT_STOP
                        break
                try:
                    out, err = proc.communicate(
                        input=input_bytes if not started else None,
                        timeout=min(PROCESS_LEADER_POLL_INTERVAL_S, remaining))
                except subprocess.TimeoutExpired as exc:
                    started = True
                    out, err = exc.output or b"", exc.stderr or b""
                except Exception as exc:
                    state = RecoveryProcessState.CAPTURE_FAILED
                    error = f"{type(exc).__name__}: {exc}"
                    break
                else:
                    drained = True
                    state = (RecoveryProcessState.NATURAL_COMPLETION if capture.checkpoint
                             else RecoveryProcessState.COMPLETE)
                    break
        except BaseException as exc:
            state = RecoveryProcessState.OBSERVER_FAILED
            error = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, Exception):
                interruption = exc
        deadline_observed = (state is RecoveryProcessState.DEADLINE
                             or (state is RecoveryProcessState.CHECKPOINT_STOP
                                 and time.monotonic() >= deadline))
        try:
            if state is RecoveryProcessState.CHECKPOINT_STOP:
                if deadline_observed:
                    state = RecoveryProcessState.DEADLINE
                elif process_leader_exited(proc):
                    state = RecoveryProcessState.NATURAL_COMPLETION
                else:
                    try:
                        killpg = getattr(os, "killpg", None)
                        if not callable(killpg):
                            raise OSError("intentional checkpoint stop requires POSIX process groups")
                        killpg(proc.pid, signal.SIGTERM)
                        signal_sent = int(signal.SIGTERM)
                    except OSError as exc:
                        if isinstance(exc, ProcessLookupError) or exc.errno == errno.ESRCH:
                            state = RecoveryProcessState.NATURAL_COMPLETION
                        else:
                            state = RecoveryProcessState.SIGNAL_FAILED
                            error = f"{type(exc).__name__}: {exc}"
            if state is RecoveryProcessState.CHECKPOINT_STOP:
                try:
                    out, err = proc.communicate(timeout=PROCESS_PIPE_DRAIN_GRACE_S)
                    drained = True
                except subprocess.TimeoutExpired as exc:
                    out, err = exc.output or out, exc.stderr or err
                except Exception as exc:
                    state = RecoveryProcessState.CAPTURE_FAILED
                    error = f"{type(exc).__name__}: {exc}"
            group_cleanup = kill_process_group(proc.pid)
            if group_cleanup["status"] in {"unsupported", "warning"}:
                if proc.poll() is None:
                    proc.kill()
                if state in {RecoveryProcessState.COMPLETE, RecoveryProcessState.CHECKPOINT_STOP}:
                    state = RecoveryProcessState.CLEANUP_FAILED
            if not drained:
                try:
                    out, err = proc.communicate(timeout=PROCESS_PIPE_DRAIN_GRACE_S)
                    drained = True
                except subprocess.TimeoutExpired as exc:
                    out, err = exc.output or out, exc.stderr or err
                    for pipe in (proc.stdout, proc.stderr):
                        if pipe is not None:
                            pipe.close()
                    if state in {RecoveryProcessState.COMPLETE, RecoveryProcessState.CHECKPOINT_STOP}:
                        state = RecoveryProcessState.CAPTURE_FAILED
                except Exception as exc:
                    state = RecoveryProcessState.CAPTURE_FAILED
                    error = f"{type(exc).__name__}: {exc}"
                    for pipe in (proc.stdout, proc.stderr):
                        if pipe is not None:
                            pipe.close()
            proc.wait(timeout=PROCESS_PIPE_DRAIN_GRACE_S)
            group_deadline = time.monotonic() + PROCESS_PIPE_DRAIN_GRACE_S
            while True:
                group_stopped, group_observation = recovery_group_stopped(proc.pid)
                if group_stopped or time.monotonic() >= group_deadline:
                    break
                time.sleep(PROCESS_LEADER_POLL_INTERVAL_S)
            if not group_stopped and state in {RecoveryProcessState.COMPLETE, RecoveryProcessState.CHECKPOINT_STOP}:
                state = RecoveryProcessState.CLEANUP_FAILED
            if state is RecoveryProcessState.CHECKPOINT_STOP:
                assert capture.checkpoint is not None
                sigkill = getattr(signal, "SIGKILL", None)
                signal_exits = {-int(signal.SIGTERM)}
                if isinstance(sigkill, int):
                    signal_exits.add(-sigkill)
                if proc.returncode not in signal_exits:
                    state = RecoveryProcessState.NATURAL_COMPLETION
                else:
                    try:
                        stable_checkpoint = recovery_file_bytes(cwd, capture.checkpoint.checkpoint_path)
                    except FileNotFoundError:
                        state = RecoveryProcessState.CHECKPOINT_MISMATCH
                    except Exception as exc:
                        state = RecoveryProcessState.OBSERVER_FAILED
                        error = f"{type(exc).__name__}: {exc}"
                    else:
                        if stable_checkpoint != observed:
                            state = RecoveryProcessState.CHECKPOINT_MISMATCH
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, Exception):
                interruption = exc
            if state in {RecoveryProcessState.COMPLETE, RecoveryProcessState.CHECKPOINT_STOP}:
                state = RecoveryProcessState.CLEANUP_FAILED
            group_cleanup = {"status": "warning", "error": error}
            group_stopped, group_observation = False, "cleanup_exception"
            try:
                kill_process_group(proc.pid)
            except Exception as exc:
                group_cleanup["retry_error"] = f"{type(exc).__name__}: {exc}"
            try:
                if proc.poll() is None:
                    proc.kill()
                try:
                    out, err = proc.communicate(timeout=PROCESS_PIPE_DRAIN_GRACE_S)
                    drained = True
                except subprocess.TimeoutExpired as exc:
                    out, err = exc.output or out, exc.stderr or err
                    for pipe in (proc.stdout, proc.stderr):
                        if pipe is not None:
                            pipe.close()
                proc.wait(timeout=PROCESS_PIPE_DRAIN_GRACE_S)
            except Exception as exc:
                group_cleanup["drain_error"] = f"{type(exc).__name__}: {exc}"
        if state is RecoveryProcessState.COMPLETE and proc.returncode != 0:
            state = RecoveryProcessState.PROCESS_FAILED
        facts = {"state": state.value, "pid": proc.pid, "process_group": proc.pid,
                 "os_returncode": proc.returncode, "signal_sent": signal_sent,
                 "leader_reaped": proc.returncode is not None, "pipes_drained": drained,
                 "group_stopped": group_stopped, "group_observation": group_observation,
                 "process_group_cleanup": group_cleanup,
                 "checkpoint_observed_live": checkpoint_live,
                 "checkpoint_sha256": hashlib.sha256(observed).hexdigest() if observed is not None else None,
                 "compatibility_returncode": 124 if deadline_observed else proc.returncode,
                 "error": error}
        for name, content in (("stdout.bin", out), ("stderr.bin", err)):
            try:
                (capture.directory / name).write_bytes(content)
            except Exception as exc:
                state = RecoveryProcessState.CAPTURE_FAILED
                facts["state"] = state.value
                facts["error"] = f"{type(exc).__name__}: {exc}"
        facts["raw_sha256"] = {"stdout.bin": hashlib.sha256(out).hexdigest(),
                               "stderr.bin": hashlib.sha256(err).hexdigest()}
        facts["raw_files_present"] = {
            name: (capture.directory / name).is_file() for name in ("stdout.bin", "stderr.bin")}
        try:
            write_json(capture.directory / "process.json", facts)
        except Exception as exc:
            state = RecoveryProcessState.CAPTURE_FAILED
            facts["state"] = state.value
            facts["error"] = f"{type(exc).__name__}: {exc}"
        capture.state = state
        if interruption is not None:
            raise interruption
        if proc.returncode is None:
            return InvocationOutcome.harness_failed("recovery process leader was not reaped", metadata=facts)
        stdout, stdout_valid = _wire_text(out, redact_stdout)
        stderr, stderr_valid = _wire_text(err, redact_stderr)
        metadata = {"recovery_process": facts, "process_group_cleanup": group_cleanup,
                    "stdout_utf8_valid": stdout_valid, "stderr_utf8_valid": stderr_valid}
        elapsed = int((time.time() - start) * 1000)
        if deadline_observed:
            return InvocationOutcome.from_timeout(stdout=stdout, stderr=stderr[:4000],
                                                  elapsed_ms=elapsed, metadata=metadata)
        return InvocationOutcome.from_process(stdout=stdout, stderr=stderr[:4000],
                                              returncode=proc.returncode,
                                              elapsed_ms=elapsed, metadata=metadata)

    def observe_tool_dialogue(proc: subprocess.Popen[bytes], session: _BridgeSession) -> _BridgeStoppedAttempt:
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            raise AssertionError("bridge lost its owned pipes")
        stdin, stdout_pipe, stderr_pipe = proc.stdin, proc.stdout, proc.stderr
        deadline = time.monotonic() + timeout
        streams = {"stdout": bytearray(), "stderr": bytearray()}
        buffer = bytearray()
        outbound = input_bytes or b""
        written = 0
        outbound_total = len(outbound)
        state: Literal["opening", "ready", "writing", "awaiting_ack", "final"] = "opening"
        final: Mapping[str, Any] | None = None
        error: str | None = None
        interruption: BaseException | None = None
        timed_out = False
        stream_total = 0
        cleanup_bytes = 0
        leader_seen_at: float | None = None
        raw_handles: dict[str, Any] = {}
        selector = selectors.DefaultSelector()

        def receive(frame: Any) -> None:
            nonlocal state, final, outbound, written, outbound_total, error
            if not isinstance(frame, dict) or type(frame.get("protocol_version")) is not int or frame["protocol_version"] != 1:
                raise ValueError("tool bridge requires a version 1 object frame")
            kind = frame.get("type")
            if kind == "final" and final is None:
                final = freeze_json_mapping(frame, "bridge final candidate")
            if kind == "call":
                if set(frame) != {"type", "protocol_version", "id", "tool", "input"} or state != "ready":
                    raise ValueError("tool bridge call requires ready state and exact fields")
                if type(frame["id"]) is not int or not 1 <= frame["id"] <= 2**63 - 1:
                    raise ValueError("tool bridge request id must be a positive bounded integer")
                response = session.journal.offer_bridge(frame, session.limits.frame_bytes)
                outbound = _bridge_frame_bytes(response, session.limits.frame_bytes)
                if outbound_total + len(outbound) > session.limits.outbound_bytes:
                    raise ValueError("tool bridge outbound stream exceeds configured bound")
                outbound_total += len(outbound)
                written = 0
                state = "writing"
                selector.register(stdin, selectors.EVENT_WRITE, "stdin")
            elif kind == "ack":
                if state != "awaiting_ack":
                    raise ValueError("tool bridge ACK requires a completely written response")
                session.journal.acknowledge_bridge(frame)
                state = "ready"
            elif kind == "final":
                if state != "ready":
                    raise ValueError("tool bridge final requires ready state")
                response = {key: value for key, value in frame.items() if key not in {"type", "protocol_version"}}
                try:
                    if set(frame) - {"type", "protocol_version", "answer", "trace", "usage", "telemetry_scope"} or "answer" not in frame:
                        raise ValueError("tool bridge final requires exact fields")
                    session.validate_final(response)
                    if not response["answer"].strip():
                        raise ValueError("tool bridge final answer must be nonblank")
                except (TypeError, ValueError) as exc:
                    error = f"tool bridge final rejected: {exc}"
                state = "final"
                stdin.close()
            else:
                raise ValueError("unexpected tool bridge frame")

        def pump(duration: float, *, cleanup: bool = False) -> None:
            nonlocal written, state, stream_total, cleanup_bytes
            for key, _mask in selector.select(duration):
                channel = key.data
                pipe = stdout_pipe if channel == "stdout" else stderr_pipe
                if channel == "stdin":
                    try:
                        count = os.write(stdin.fileno(), outbound[written:written + 16384])
                    except BlockingIOError:
                        continue
                    written += count
                    if written == len(outbound):
                        selector.unregister(stdin)
                        state = "ready" if state == "opening" else "awaiting_ack"
                    continue
                try:
                    data = os.read(pipe.fileno(), 16384)
                except BlockingIOError:
                    continue
                if not data:
                    selector.unregister(pipe)
                    pipe.close()
                    continue
                streams[channel].extend(data)
                if cleanup:
                    cleanup_bytes += len(data)
                    if cleanup_bytes > 64 * 1024:
                        raise ValueError("tool bridge cleanup drain exceeds bound")
                else:
                    stream_total += len(data)
                raw_handles[channel].write(data)
                raw_handles[channel].flush()
                os.fsync(raw_handles[channel].fileno())
                if cleanup:
                    continue
                if stream_total > session.limits.stream_bytes:
                    raise ValueError("tool bridge stdout/stderr exceeds configured bound")
                if channel == "stdout":
                    buffer.extend(data)
                    while b"\n" in buffer:
                        position = buffer.index(b"\n")
                        if position + 1 > session.limits.frame_bytes:
                            raise ValueError("tool bridge inbound frame exceeds configured bound")
                        line = bytes(buffer[:position])
                        del buffer[:position + 1]
                        receive(strict_json_loads(line.decode("utf-8", errors="strict")))
                    if len(buffer) >= session.limits.frame_bytes:
                        raise ValueError("tool bridge unterminated frame exceeds configured bound")

        try:
            for channel in streams:
                raw_handles[channel] = (session.directory / (channel + ".bin")).open("xb")
                raw_handles[channel].flush()
                os.fsync(raw_handles[channel].fileno())
            descriptor = os.open(session.directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if sys.platform == "linux":
                for pipe in (stdin, stdout_pipe, stderr_pipe):
                    os.set_blocking(pipe.fileno(), False)
            selector.register(stdin, selectors.EVENT_WRITE, "stdin")
            selector.register(stdout_pipe, selectors.EVENT_READ, "stdout")
            selector.register(stderr_pipe, selectors.EVENT_READ, "stderr")
            while True:
                if _stop_signal is not None:
                    raise KeyboardInterrupt
                now = time.monotonic()
                if process_leader_exited(proc):
                    leader_seen_at = leader_seen_at or now
                    if all(key.data == "stdin" for key in selector.get_map().values()):
                        break
                    if now - leader_seen_at >= PROCESS_PIPE_DRAIN_GRACE_S:
                        raise ValueError("tool bridge pipes remained open after leader exit")
                if now >= deadline:
                    timed_out = True
                    raise TimeoutError("tool bridge deadline expired")
                pump(min(PROCESS_LEADER_POLL_INTERVAL_S, deadline - now))
            if buffer:
                raise ValueError("tool bridge has an unterminated final stream tail")
            if state != "final":
                raise ValueError("tool bridge EOF without final response")
        except BaseException as exc:
            error = f"tool bridge failed: {type(exc).__name__}: {exc}"
            if not isinstance(exc, Exception):
                interruption = exc
        natural_exit = process_leader_exited(proc)
        group_was_stopped, _observation = recovery_group_stopped(proc.pid)
        forced = not natural_exit or not group_was_stopped
        cleanup = kill_process_group(proc.pid)
        if cleanup.get("status") in {"unsupported", "warning"} and not natural_exit:
            proc.kill()
        if not stdin.closed:
            if stdin in selector.get_map():
                selector.unregister(stdin)
            stdin.close()
        drain_deadline = time.monotonic() + PROCESS_PIPE_DRAIN_GRACE_S
        try:
            while selector.get_map() and time.monotonic() < drain_deadline:
                pump(min(PROCESS_LEADER_POLL_INTERVAL_S, max(0, drain_deadline - time.monotonic())), cleanup=True)
        except BaseException as exc:
            error = error or f"tool bridge drain failed: {exc}"
            if not isinstance(exc, Exception):
                interruption = interruption or exc
        drained = not selector.get_map()
        selector.close()
        for pipe in (stdout_pipe, stderr_pipe):
            if not pipe.closed:
                pipe.close()
        for handle in raw_handles.values():
            try:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            except OSError as exc:
                error = error or f"tool bridge raw capture failed: {exc}"
        group_stopped, group_observation = recovery_group_stopped(proc.pid)
        group_deadline = time.monotonic() + AGENT_STOP_GRACE_S
        while not group_stopped and time.monotonic() < group_deadline:
            time.sleep(PROCESS_LEADER_POLL_INTERVAL_S)
            group_stopped, group_observation = recovery_group_stopped(proc.pid)
        try:
            proc.wait(timeout=PROCESS_PIPE_DRAIN_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            forced = True
            try:
                proc.wait(timeout=PROCESS_PIPE_DRAIN_GRACE_S)
            except subprocess.TimeoutExpired:
                error = error or "tool bridge leader reap timed out"
        out, err = bytes(streams["stdout"]), bytes(streams["stderr"])
        if final is None:
            for line in out.split(b"\n")[:-1]:
                try:
                    if len(line) + 1 > session.limits.frame_bytes:
                        continue
                    candidate = strict_json_loads(line.decode("utf-8", errors="strict"))
                    if isinstance(candidate, dict) and candidate.get("type") == "final" and type(candidate.get("protocol_version")) is int and candidate["protocol_version"] == 1:
                        final = freeze_json_mapping(candidate, "bridge final candidate")
                        break
                except (TypeError, ValueError):
                    continue
        stdout, stdout_valid = _wire_text(out, redact_stdout)
        stderr, stderr_valid = _wire_text(err, redact_stderr)
        facts = _BridgeProcessFacts(proc.pid, proc.returncode, natural_exit, group_stopped,
            group_observation, drained, proc.returncode is not None, timed_out, forced, stream_total, cleanup_bytes)
        if not stdout_valid or not stderr_valid:
            error = error or "tool bridge streams are not valid UTF-8"
        if proc.returncode != 0 or not natural_exit or not group_stopped or not drained or forced:
            error = error or "tool bridge did not complete naturally with a stopped group and drained pipes"
        retain_raw(out, err, proc.returncode, timed_out=timed_out, communication_complete=drained)
        process_evidence = {**facts.as_dict(), "error": error, "process_group_cleanup": cleanup,
                            "raw_sha256": {name + ".bin": hashlib.sha256(data).hexdigest()
                                           for name, data in streams.items()},
                            "stdout_utf8_valid": stdout_valid, "stderr_utf8_valid": stderr_valid}
        try:
            with (session.directory / "process.json").open("x", encoding="utf-8") as handle:
                json.dump(process_evidence, handle, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            descriptor = os.open(session.directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException as exc:
            error = error or f"tool bridge process evidence failed: {exc}"
            if not isinstance(exc, Exception):
                interruption = interruption or exc
        metadata = {"tool_bridge_process": process_evidence, "stdout_utf8_valid": stdout_valid,
                    "stderr_utf8_valid": stderr_valid, "process_group_cleanup": cleanup}
        if timed_out:
            outcome = InvocationOutcome.from_timeout(stdout=stdout, stderr=stderr[:4000],
                elapsed_ms=int((time.time() - start) * 1000), metadata=metadata)
        elif proc.returncode is None:
            outcome = InvocationOutcome.harness_failed("tool bridge leader was not reaped", metadata=metadata)
        else:
            outcome = InvocationOutcome.from_process(stdout=stdout, stderr=stderr[:4000], returncode=proc.returncode,
                elapsed_ms=int((time.time() - start) * 1000), metadata=metadata)
        return _BridgeStoppedAttempt(outcome, final, error, interruption, facts)

    try:
        input_bytes = input_text.encode("utf-8") if input_text is not None else None
    except UnicodeEncodeError:
        return InvocationOutcome.spawn_failed(
            stderr="subprocess stdin is not valid UTF-8", elapsed_ms=0)
    start = time.time()
    capture = plan.recovery_capture
    if capture is not None:
        execution_environment = os.environ if env is None else env
        if os.path.dirname(argv[0]):
            executable_path = Path(argv[0])
            if not executable_path.is_absolute():
                executable_path = cwd / executable_path
            executable = shutil.which(str(executable_path))
        else:
            search_path = os.pathsep.join(
                str(Path(part) if Path(part).is_absolute() else cwd / part)
                for part in execution_environment.get("PATH", os.defpath).split(os.pathsep))
            executable = shutil.which(argv[0], path=search_path)
        wrapper = {"client_argv": argv, "cwd": str(cwd), "executable": executable,
                   "executable_observation": "resolved from client cwd and PATH before spawn",
                   "executable_sha256": None, "version": None,
                   "effective_exec_argv": None, "effective_exec_argv_source": None,
                   "execution_boundary": (
                       "existing backend client subprocess; original POSIX process group"
                       if callable(getattr(os, "killpg", None)) else
                       "existing backend client subprocess; process-group control unavailable"),
                   "evidence_directory": str(capture.directory.resolve()),
                   "evidence_write_protection": "not established by the runner"}
        if executable is not None:
            try:
                wrapper["executable_sha256"] = hashlib.sha256(Path(executable).read_bytes()).hexdigest()
            except OSError:
                pass
        write_json(capture.directory / "invocation.json", wrapper)
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        if capture is not None:
            capture.state = RecoveryProcessState.SPAWN_FAILED
            write_json(capture.directory / "process.json", {
                "state": capture.state.value, "os_returncode": None, "pid": None,
                "error": f"{type(exc).__name__}: {exc}",
                "raw_sha256": {"stdout.bin": hashlib.sha256(b"").hexdigest(),
                               "stderr.bin": hashlib.sha256(b"").hexdigest()}})
            (capture.directory / "stdout.bin").write_bytes(b"")
            (capture.directory / "stderr.bin").write_bytes(b"")
        outcome = InvocationOutcome.spawn_failed(
            stderr=redact_stderr(f"{type(exc).__name__}: {exc}")[:4000],
            elapsed_ms=int((time.time() - start) * 1000),
        )
        if dialogue is not None:
            return _BridgeStoppedAttempt(outcome, None, "bridge process failed to spawn", None,
                _BridgeProcessFacts(None, None, False, False, "not_spawned", False, False, False, False, 0, 0))
        return outcome
    if dialogue is not None:
        return observe_tool_dialogue(proc, dialogue)
    if capture is not None:
        return observe_recovery_process(proc, capture, input_bytes, start)
    deadline = time.monotonic() + timeout
    communication_started = False
    communication_complete = False
    communication_timeout: subprocess.TimeoutExpired | None = None
    leader_exited = False
    try:
        while True:
            if _stop_signal is not None:
                raise KeyboardInterrupt
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if communication_timeout is None:
                    communication_timeout = subprocess.TimeoutExpired(argv, timeout)
                leader_exited = process_leader_exited(proc)
                break
            try:
                out, err = proc.communicate(
                    input=(input_bytes
                           if input_bytes is not None and not communication_started
                           else None),
                    timeout=min(PROCESS_LEADER_POLL_INTERVAL_S, remaining),
                )
            except subprocess.TimeoutExpired as exc:
                communication_started = True
                communication_timeout = exc
                # ``communicate`` waits for pipe EOF as well as leader exit. Poll in
                # short slices so a successful leader is not charged the full task
                # timeout merely because one of its helpers inherited a capture fd.
                leader_exited = process_leader_exited(proc)
                if leader_exited:
                    break
            else:
                retain_raw(out, err, proc.returncode, timed_out=False, communication_complete=True)
                stdout, stdout_utf8_valid = _wire_text(out, redact_stdout)
                stderr, stderr_utf8_valid = _wire_text(err, redact_stderr)
                stderr, returncode, _timed_out = stderr[:4000], proc.returncode, False
                communication_complete = True
                break
    except KeyboardInterrupt:
        if communication_timeout is not None:
            retain_raw(communication_timeout.output, communication_timeout.stderr,
                       proc.returncode, timed_out=False)
        stop_session(proc, _stop_signal or signal.SIGINT)
        if communication_timeout is not None:
            retain_raw(communication_timeout.output, communication_timeout.stderr,
                       proc.returncode, timed_out=False)
        raise
    except BaseException:
        stop_session(proc, signal.SIGTERM)
        raise
    if not communication_complete:
        assert communication_timeout is not None
        exc = communication_timeout
        retain_raw(exc.output, exc.stderr, proc.returncode, timed_out=not leader_exited)
        try:
            group_cleanup = kill_process_group(proc.pid)
        except Exception as cleanup_exc:
            group_cleanup = {"status": "warning", "signal": None,
                             "error": errno.errorcode.get(getattr(cleanup_exc, "errno", None), type(cleanup_exc).__name__)}
        if not leader_exited and group_cleanup.get("status") in {"unsupported", "warning"}:
            proc.kill()
        drain_complete = False
        try:
            out, err = proc.communicate(timeout=PROCESS_PIPE_DRAIN_GRACE_S)
            drain_complete = True
        except subprocess.TimeoutExpired as drain_exc:
            # A detached child can outlive the original group and keep the pipe
            # write ends open forever. Preserve the partial capture, close our
            # read ends, and reap the group leader without waiting for that child.
            out, err = drain_exc.output or exc.output, drain_exc.stderr or exc.stderr
            posix_pipe_close = hasattr(os, "killpg")
            if posix_pipe_close:
                for pipe in (proc.stdout, proc.stderr):
                    if pipe is not None:
                        pipe.close()
            try:
                proc.wait(timeout=PROCESS_PIPE_DRAIN_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=PROCESS_PIPE_DRAIN_GRACE_S)
                except subprocess.TimeoutExpired:
                    group_cleanup = {**group_cleanup, "leader_reap": "timed_out"}
            pipe_action = "closed" if posix_pipe_close else "abandoned"
            pipe_warning = (
                "capture pipes remained open after process-group termination"
                if posix_pipe_close else
                "capture pipes remained open; non-POSIX reader threads were abandoned"
            )
            group_cleanup = {**group_cleanup, "pipe_drain": pipe_action, "warning": pipe_warning}
        retain_raw(out or exc.stdout, err if err is not None else exc.stderr,
                   proc.returncode, timed_out=not leader_exited,
                   communication_complete=drain_complete)
        stdout, stdout_utf8_valid = _wire_text(out or exc.stdout, redact_stdout)
        captured_stderr = err if err is not None else exc.stderr
        if not leader_exited:
            captured_stderr = captured_stderr or exc.stderr or str(exc)
        stderr, stderr_utf8_valid = _wire_text(
            captured_stderr, redact_stderr)
        stderr = stderr[:4000]
        returncode = proc.returncode if leader_exited and proc.returncode is not None else 124
    else:
        try:
            group_cleanup = kill_process_group(proc.pid)
        except Exception as exc:
            group_cleanup = {"status": "warning", "signal": None,
                             "error": errno.errorcode.get(getattr(exc, "errno", None), type(exc).__name__)}
    if group_cleanup.get("status") == "warning":
        detail = str(group_cleanup.get("error") or "unknown error")
        warning = f"process-group cleanup warning: {detail}"
        stderr = _stderr_with_warning(stderr, warning)
    if group_cleanup.get("warning"):
        stderr = _stderr_with_warning(stderr, str(group_cleanup["warning"]))
    outcome_metadata = {
        "process_group_cleanup": group_cleanup,
        "stdout_utf8_valid": stdout_utf8_valid,
        "stderr_utf8_valid": stderr_utf8_valid,
    }
    if not communication_complete and not leader_exited:
        return InvocationOutcome.from_timeout(
            stdout=stdout, stderr=stderr,
            elapsed_ms=int((time.time() - start) * 1000),
            metadata=outcome_metadata,
        )
    return InvocationOutcome.from_process(
        stdout=stdout, stderr=stderr, returncode=returncode,
        elapsed_ms=int((time.time() - start) * 1000),
        metadata=outcome_metadata,
    )


def run_argv_with_timeout(argv: list[str], *, cwd: Path | str | None = None,
                          env: dict[str, str] | None = None, timeout: int,
                          input_text: str | None = None) -> dict[str, Any]:
    """Legacy dictionary boundary for external callers; internal code uses the typed owner."""
    plan = ProcessInvocationPlan.from_values(
        argv,
        input_text=input_text,
        cwd=Path.cwd() if cwd is None else cwd,
        timeout_s=timeout,
        environment=env,
    )
    return invoke_argv_with_timeout(plan).as_legacy_dict()


def _stderr_with_warning(stderr: str, warning: str, *, limit: int = 4000) -> str:
    warning = warning.strip()
    if len(warning) >= limit:
        return warning[:limit]
    base = stderr.rstrip()
    if not base:
        return warning
    available = max(0, limit - len(warning) - 1)
    prefix = base[:available].rstrip()
    return f"{prefix}\n{warning}" if prefix else warning
