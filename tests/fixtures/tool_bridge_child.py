import json
import sys


def send(frame):
    print(json.dumps({"protocol_version": 1, **frame}), flush=True)


opening = json.loads(sys.stdin.readline())
send({"type": "call", "id": 1, "tool": "search", "input": {"q": "alpha"}})
response = json.loads(sys.stdin.readline())
send({"type": "ack", "id": response["id"], "turn_number": response["turn_number"],
      "exchange_number": response["exchange_number"], "payload_sha256": response["payload_sha256"]})
error = response["error"]
send({"type": "final", "answer": f"{response['output']} {error['availability']} {error.get('value')}",
      "usage": {"cost_usd": 0.02}, "telemetry_scope": "turn_delta"})
