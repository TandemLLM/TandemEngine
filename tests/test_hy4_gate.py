"""ops/hy4-gate.sh, the Hy4-flash pack's gate window, on stand-ins: no network, no engine, no GPU.

The plan of a dry run; the refusals (no --uploaded, an idle wait under 10 minutes, a window over 3 hours, a live
engine that is no release tree); and one whole pass on a compressed clock with a stand-in release tree whose
hold script runs the command, a stand-in interpreter and a stand-in /health: the idle wait restarts while a
request runs, the hold runs under the box lock with the window's minutes, and the record holds every step.
"""

from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(ROOT, "ops", "hy4-gate.sh")


def _run(args, env=None, timeout=60):
    e = dict(os.environ, **(env or {}))
    r = subprocess.run(["bash", GATE] + args, capture_output=True, text=True, timeout=timeout, env=e)
    return r.returncode, r.stdout + r.stderr


def test_a_dry_run_prints_the_plan_and_runs_nothing():
    with tempfile.TemporaryDirectory() as d:
        live = os.path.join(d, "releases", "abc", "src")
        rc, out = _run(["--dry-run", "--dir", os.path.join(d, "pack"), "--out", os.path.join(d, "out")], {"HY4_LIVE": live})
        assert rc == 0 and not os.path.exists(os.path.join(d, "out")) and not os.path.exists(os.path.join(d, "pack")), out
        assert "0xBakeer/TandemLLM-Hy4-flash-EXL3-3.0@main" in out and "600 s with no running or waiting request" in out
        assert f"bash {live}/ops/hold.sh 150 --" in out and "flock" in out and "--fit 262144" in out and "--dfly" in out
    return "the repository, the idle rule and the hold line"


def test_the_refusals():
    with tempfile.TemporaryDirectory() as d:
        live = os.path.join(d, "releases", "abc", "src")
        base = ["--dir", os.path.join(d, "pack"), "--out", os.path.join(d, "out")]
        for args, env, why in ((base, {"HY4_LIVE": live}, "--uploaded"),
                               (base + ["--uploaded", "--idle", "300"], {"HY4_LIVE": live}, "at least 600 s"),
                               (base + ["--uploaded", "--minutes", "240"], {"HY4_LIVE": live}, "at most 180 minutes"),
                               (base + ["--uploaded"], {"HY4_LIVE": os.path.join(d, "some", "tree")}, "no release tree"),
                               (base + ["--frobnicate"], {"HY4_LIVE": live}, "unknown argument")):
            rc, out = _run(args, env)
            assert rc != 0 and why in out, (args, out)
        assert not os.path.exists(os.path.join(d, "out"))
    return "5 refusals, nothing written"


class _Health(http.server.BaseHTTPRequestHandler):
    busy = [2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]          # what `running` answers, one a request
    seen = []

    def do_GET(self):
        n = self.busy.pop(0) if len(self.busy) > 1 else self.busy[0]
        self.seen.append(n)
        body = json.dumps({"inflight": {"running": n, "waiting": 0}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_a_whole_pass_on_stand_ins():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Health)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as d:
            live = os.path.join(d, "releases", "abc", "src")
            os.makedirs(os.path.join(live, "ops"))
            with open(os.path.join(live, "ops", "hold.sh"), "w") as f:
                f.write('#!/bin/bash\necho "[hold] minutes $1 reason ${HOLD_REASON:-none}" >> "$HOLD_LOG"\nshift; shift\n"$@"\n')
            py = os.path.join(d, "py")
            with open(py, "w") as f:
                f.write('#!/bin/bash\nif [ "$1" = "-" ]; then cat > /dev/null; echo "stand-in audit of $2"; exit 0; fi\n'
                        'echo "[hy4] stand-in check: $*"\necho \'{"check": "self_check", "ok": true}\'\n')
            os.chmod(py, 0o755)
            out = os.path.join(d, "out")
            rc, text = _run(["--uploaded", "--skip-download", "--idle", "2", "--minutes", "90", "--dir", os.path.join(d, "pack"), "--out", out],
                            {"HY4_LIVE": live, "HY4_PY": py, "HY4_GATE_IDLE_MIN": "1", "HY4_GATE_POLL": "1",
                             "HY4_HEALTH": f"http://127.0.0.1:{srv.server_address[1]}/health",
                             "HY4_BOX_LOCK": os.path.join(d, "lock"), "HOLD_LOG": os.path.join(d, "hold.log")}, timeout=120)
            assert rc == 0, text
            log = open(os.path.join(out, "gate.log")).read()
            for step in ("START tree", "audit: start", "audit: done", "idle: waiting for 2 s", "hold: start (budget 90 min)", "hold: done rc=0; live /health 200"):
                assert step in log, (step, log)
            order = [log.index(s) for s in ("audit: done", "idle: waiting", "hold: start", "hold: done")]
            assert order == sorted(order)
            hold = open(os.path.join(d, "hold.log")).read()
            assert "minutes 90" in hold and "Hy4-flash pack gates" in hold and "262144" in hold
            assert json.loads(open(os.path.join(out, "check.jsonl")).read().strip()) == {"check": "self_check", "ok": True}
            assert "tools/hy4_check.py --pack" in open(os.path.join(out, "hold.out")).read()
            # the wait began again after the request that ran in the middle of it
            assert _Health.seen[:4] == [2, 0, 0, 1] and len(_Health.seen) >= 7, _Health.seen
    finally:
        srv.shutdown()
    return f"audit, an idle wait that restarted once ({len(_Health.seen)} health reads), the hold under the lock, the record"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<56} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<56} FAIL {str(e)[:900]}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
