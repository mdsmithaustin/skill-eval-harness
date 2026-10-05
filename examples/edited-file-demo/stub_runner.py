from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def emit(value: dict) -> None:
    print(json.dumps(value), flush=True)


def main() -> int:
    sys.stdin.read()
    emit({"type": "thread.started", "thread_id": "offline-edited-file"})
    emit({"type": "turn.started"})
    skills = list(Path("skills").glob("*/SKILL.md"))
    if any('" ".join(value.split()).casefold()' in skill.read_text(encoding="utf-8")
           for skill in skills):
        Path("inputs/name_tools.py").write_text(
            'def normalize_name(value: str) -> str:\n    return " ".join(value.split()).casefold()\n',
            encoding="utf-8",
        )
        emit({"type": "item.completed", "item": {
            "id": "edit", "type": "file_change", "status": "completed",
            "changes": [{"path": "inputs/name_tools.py", "kind": "update"}],
        }})
    command = "python3 -B inputs/test_name_tools.py"
    emit({"type": "item.started", "item": {
        "id": "test", "type": "command_execution", "command": command,
        "status": "in_progress",
    }})
    result = subprocess.run(
        command.split(), capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    emit({"type": "item.completed", "item": {
        "id": "test", "type": "command_execution", "command": command,
        "status": "completed", "exit_code": result.returncode,
        "aggregated_output": result.stdout + result.stderr,
    }})
    answer = f"Fixture test command exited {result.returncode}."
    output = Path(sys.argv[sys.argv.index("--output-last-message") + 1])
    output.write_text(answer, encoding="utf-8")
    emit({"type": "item.completed", "item": {
        "id": "answer", "type": "agent_message", "text": answer,
    }})
    emit({"type": "turn.completed", "usage": {"input_tokens": 0, "output_tokens": 0}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
