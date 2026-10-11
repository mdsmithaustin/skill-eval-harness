#!/usr/bin/env python3
from __future__ import annotations

import json
import sys


def main() -> int:
    json.loads(sys.stdin.readline())
    print(json.dumps({"type": "call", "protocol_version": 1, "id": 1,
                      "tool": "search", "input": {"q": "alpha"}}), flush=True)
    response = json.loads(sys.stdin.readline())
    print(json.dumps({"type": "ack", "protocol_version": 1, "id": response["id"],
                      "turn_number": response["turn_number"],
                      "exchange_number": response["exchange_number"],
                      "payload_sha256": response["payload_sha256"]}), flush=True)
    error = response["error"]
    answer = f"{response['output']} error={error['availability']}:{error.get('value')}"
    print(json.dumps({"type": "final", "protocol_version": 1, "answer": answer}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
