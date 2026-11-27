"""The agent's two tools.

    python -m policy.commander.cli look
    python -m policy.commander.cli run_skill '{"skill": "open", "object": {"name": "drawer", "camera": "right", "uv": [190, 214]}}'
    python -m policy.commander.cli done
"""
import json
import os
import sys
import urllib.request


def call(path, body=None):
    port = os.environ.get("RUA_CMD_PORT", "8765")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/{path}", data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("look", "run_skill", "done"):
        print(__doc__)
        sys.exit(2)
    body = json.loads(sys.argv[2]) if sys.argv[1] == "run_skill" and len(sys.argv) > 2 else {}
    print(json.dumps(call(sys.argv[1], body), indent=1))


if __name__ == "__main__":
    main()
