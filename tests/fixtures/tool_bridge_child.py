import json
import os
import subprocess
import sys
import time


def send(frame):
    print(json.dumps({"protocol_version": 1, **frame}), flush=True)


opening = json.loads(sys.stdin.readline())
mode = sys.argv[1]
if mode == "duplicate":
    print('{"type":"call","type":"final","protocol_version":1,"answer":"alpha"}', flush=True)
    raise SystemExit
if mode == "oversized":
    print("x" * 2048, flush=True)
    raise SystemExit
if mode == "stream":
    os.write(2, b"x" * 4096)
    time.sleep(5)
if mode == "partial":
    os.write(1, b'{"type":"call"')
    time.sleep(5)
if mode == "invalid_utf8":
    os.write(1, b'\xff\n')
    raise SystemExit
send({"type": "call", "id": 1, "tool": "search", "input": {"q": "alpha"}})
if mode == "blocked_response":
    time.sleep(5)
response = json.loads(sys.stdin.readline())
if mode == "offer_timeout":
    time.sleep(5)
send({"type": "ack", "id": response["id"], "turn_number": response["turn_number"],
      "exchange_number": response["exchange_number"], "payload_sha256":
      "sha256:" + "0" * 64 if mode == "bad_ack" else response["payload_sha256"]})
if mode == "prefix_timeout":
    send({"type": "call", "id": 2, "tool": "search", "input": {"q": "alpha"}})
    json.loads(sys.stdin.readline())
    time.sleep(5)
error = response["error"]
send({"type": "final", "answer": f"{response['output']} {error['availability']} {error.get('value')}",
      "usage": {"cost_usd": 0.02}, "telemetry_scope": "turn_delta",
      **({"trace": "invalid"} if mode == "bad_sibling" else {})})
if mode in {"final_hang", "interrupt"}:
    time.sleep(5)
if mode == "held_pipes":
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
if mode == "final_nonzero":
    raise SystemExit(7)
