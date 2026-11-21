"""The round runner against an abandoned request's teardown, on the GPU (a box step; about 8 minutes a tree).

    python tools/abandon_race_check.py --base-url http://127.0.0.1:8160/v1 [--iterations 6] [--doc ids-32768
        --long-docs DIR --tokenizer TOK | --doc synthetic:32768] [--server-log LOG] [--json-out FILE]
    python tools/abandon_race_check.py --self-test

On 2026-11-21 a P6 soak on Hy4-flash 3.08 (8 lanes, chat mode) ended in an illegal memory access. A client left
while the round runner waited for the GPU turn behind another lane's prefill chunk; the runner had picked the
request's task already, the request's handler got the turn first, trimmed the lane's pool pages, and the runner's
next round drafted from rows on a page that was just unmapped.

One iteration sets that up against a running server (stdlib only; the long document needs numpy and transformers):

  B   a cold long document (a fresh nonce in its first line), streamed, a few tokens out: a long prefill in chunks;
  C   a short prompt with a long answer, 1 s later, read to its end: it keeps the runner in the turn's queue, so a
      new lane starts at the runner's virtual time;
  A1-A6  from 3 s on, every 6 s: a short prompt, thinking off, streamed; the client closes the socket with an RST
      (SO_LINGER 0) at its first content delta.

/health is read every --poll-s seconds. An iteration shows the race when:
  fault   /health fails or answers 5xx, the round runner's error count rises, or the server log has an illegal memory
          access after the iteration's start;
  stale   a side slot that is idle (busy false) in two samples with the same `served` count has more rows in the
          second (kv_length or committed_rows up) and is still idle after: a round ran on a lane nobody holds.
A guard error (CaptureGuardError) or another round error is reported as `error`, not as the race. An iteration
counts only when one sample shows B in phase prefill (4,096 rows or more) and two lanes in phase runner while C and
an A are open; a fault counts in any case.

Exit: 0 no race in --iterations counted iterations; 1 the race shown (fault or stale); 3 another error; 2 too few
counted iterations, or the server is not one this check can use. The run stops at the first race or error.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import secrets
import socket
import struct
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

A_PROMPTS = [
    "Write a long story about a lighthouse keeper and a storm.",
    "Describe, step by step, how a river shapes a valley over time.",
    "Tell a long tale about a cat who learns to sail a small boat.",
    "Explain in detail how bread rises, from flour to the oven.",
    "Write a long letter from a sailor to his sister at home.",
    "Describe a long walk through an old city at night.",
]
C_PROMPT = "Count from 1 to 400, one number per line, nothing else."
IMA = re.compile(r"illegal memory access|cudaErrorIllegalAddress", re.I)
REQ = re.compile(r"^\[req\] (\S+) stream prompt=(\d+) completion=(\d+) finish=(\S+) (\d+) ms .*?slot=(\d+)"
                 r"(?:.*?decode_ms=([\d.]+))?")


# ------------------------------------------------------------------ the client (raw socket, an RST at will)
class _Body:
    """An HTTP/1.1 response body, plain (to the close) or chunked, fed bytes as they come."""

    def __init__(self, chunked: bool):
        self.chunked, self.buf, self.left, self.done = chunked, b"", None, False

    def feed(self, data: bytes) -> bytes:
        if not self.chunked:
            return data
        self.buf += data
        out = b""
        while not self.done:
            if self.left is None:
                i = self.buf.find(b"\r\n")
                if i < 0:
                    break
                size = int(self.buf[:i].split(b";")[0].strip() or b"0", 16)
                self.buf = self.buf[i + 2:]
                if size == 0:
                    self.done = True
                    break
                self.left = size
            take = self.buf[:self.left]
            out += take
            self.buf = self.buf[len(take):]
            self.left -= len(take)
            if self.left:
                break
            if len(self.buf) < 2:
                self.left = 0          # the CRLF after the chunk is still to come
                break
            self.buf, self.left = self.buf[2:], None
        if self.left == 0 and len(self.buf) >= 2:
            self.buf, self.left = self.buf[2:], None
        return out


def stream(v1: str, body: dict, *, timeout_s: float, abort: threading.Event, close_burst: int = 0,
           burst_gap_s: float = 0.5, t_ref: float = 0.0) -> dict:
    """One streamed chat completion over a raw socket. `close_burst` N > 0: the client closes with an RST at the
    first content (or reasoning) delta of the N-th burst of deltas (a burst: deltas less than `burst_gap_s` apart).
    Ends at the stream's end, the close, `abort`, `timeout_s` or an error; never raises. Times are from `t_ref`."""
    u = urllib.parse.urlsplit(v1)
    path = u.path.rstrip("/") + "/chat/completions"
    data = json.dumps(body).encode()
    t0 = time.time()
    rec = {"t0": round(t0 - t_ref, 3), "status": None, "cid": None, "t_first": None, "t_close": None,
           "t_end": None, "deltas": 0, "chars": 0, "bursts": 0, "finish": None, "completion": None,
           "closed": False, "aborted": False, "error": None}
    s = None
    try:
        s = socket.create_connection((u.hostname, u.port or 80), timeout=10)
        s.settimeout(1.0)
        s.sendall((f"POST {path} HTTP/1.1\r\nHost: {u.hostname}:{u.port or 80}\r\nContent-Type: application/json\r\n"
                   f"Accept: text/event-stream\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n").encode()
                  + data)
        head, dec, text, err_body, last = b"", None, "", b"", None
        while True:
            if abort.is_set():
                rec["aborted"] = True
                break
            if time.time() - t0 > timeout_s:
                rec["error"] = f"no end after {timeout_s:.0f} s"
                break
            try:
                got = s.recv(65536)
            except socket.timeout:
                continue
            if not got:
                break
            if dec is None:
                head += got
                if b"\r\n\r\n" not in head:
                    continue
                hdr, got = head.split(b"\r\n\r\n", 1)
                lines = hdr.decode("latin-1").split("\r\n")
                rec["status"] = int(lines[0].split()[1])
                hd = {k.strip().lower(): v.strip() for k, v in (ln.split(":", 1) for ln in lines[1:] if ":" in ln)}
                dec = _Body("chunked" in hd.get("transfer-encoding", "").lower())
            piece = dec.feed(got)
            if rec["status"] != 200:
                err_body += piece
                continue
            text += piece.decode("utf-8", "replace")
            *done, text = text.split("\n")
            for ln in done:
                ln = ln.strip()
                if not ln.startswith("data: ") or ln == "data: [DONE]":
                    continue
                try:
                    d = json.loads(ln[6:])
                except ValueError:
                    continue
                rec["cid"] = rec["cid"] or d.get("id")
                if d.get("error"):
                    rec["error"] = str(d["error"])[:300]
                if d.get("usage"):
                    rec["completion"] = d["usage"].get("completion_tokens")
                for c in d.get("choices") or []:
                    if c.get("finish_reason"):
                        rec["finish"] = c["finish_reason"]
                    dl = c.get("delta") or {}
                    if not (dl.get("content") or dl.get("reasoning_content")):
                        continue
                    now = time.time()
                    if last is None or now - last > burst_gap_s:
                        rec["bursts"] += 1
                    last = now
                    rec["deltas"] += 1
                    rec["chars"] += len(dl.get("content") or dl.get("reasoning_content") or "")
                    if rec["t_first"] is None:
                        rec["t_first"] = round(now - t_ref, 3)
                    if close_burst and rec["bursts"] >= close_burst and not rec["closed"]:
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                        s.close()
                        rec["closed"], rec["t_close"] = True, round(time.time() - t_ref, 3)
                        break
                if rec["closed"]:
                    break
            if rec["closed"]:
                break
        if rec["status"] not in (None, 200):
            rec["error"] = err_body.decode("utf-8", "replace")[:300] or f"HTTP {rec['status']}"
    except OSError as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        if s is not None and not rec["closed"]:
            try:
                s.close()
            except OSError:
                pass
    rec["t_end"] = round(time.time() - t_ref, 3)
    return rec


# ------------------------------------------------------------------ /health
def get_health(base: str, timeout: float = 5.0) -> tuple:
    """(HTTP status or None, the JSON body or {"error": ...}); a 5xx body is read too."""
    try:
        with urllib.request.urlopen(base + "/health", timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except ValueError:
            return exc.code, {"error": f"HTTP {exc.code}"}
    except Exception as exc:                                         # noqa: BLE001
        return None, {"error": f"{type(exc).__name__}: {exc}"[:200]}


def compact(t: float, code, h: dict) -> dict:
    """What the rules read from one /health: each slot's state and rows, the turn, the runner's errors."""
    par = h.get("parallel") or {}
    pool = {s.get("slot"): s for s in ((h.get("kv_pool") or {}).get("slots") or [])}
    slots = []
    for s in par.get("slots") or []:
        p = pool.get(s.get("index")) or {}
        slots.append({"i": s.get("index"), "busy": s.get("busy"), "served": s.get("served"), "phase": s.get("phase"),
                      "owner": s.get("owner"), "kv": s.get("kv_length"), "pkv": p.get("kv_length"),
                      "rows": p.get("committed_rows")})
    turn = par.get("turn") or {}
    rr = par.get("round_runner") or {}
    return {"t": round(t, 3), "code": code, "status": h.get("status"),
            "failed": h.get("failed") or (h.get("gpu") or {}).get("error") or h.get("error"),
            "rr_errors": rr.get("errors"), "rounds": rr.get("rounds"),
            "turn": {"owner": turn.get("owner"), "waiting": turn.get("waiting"), "vtime": turn.get("vtime_s")},
            "slots": slots, "abandoned": (h.get("inflight") or {}).get("abandoned"),
            "reclaims": (h.get("kv_pool") or {}).get("reclaims")}


def fault_of(s: dict, errors0) -> str | None:
    """A sample's fault: the GPU failed, a 5xx, or the runner's error count above the iteration's start."""
    if s["code"] is not None and s["code"] >= 500:
        return f"/health {s['code']}: {s['failed'] or s['status']}"
    if s["status"] == "failed" or (s["failed"] and IMA.search(str(s["failed"]))):
        return f"/health status {s['status']}: {s['failed']}"
    if errors0 is not None and s["rr_errors"] is not None and s["rr_errors"] > errors0:
        return f"round_runner.errors {errors0} -> {s['rr_errors']}"
    return None


# ------------------------------------------------------------------ the rules
def idle_moves(samples: list[dict]) -> list[dict]:
    """Rows a side slot gained while nobody held it: idle (busy false) in two samples in a row with the same `served`
    count, and kv_length or committed_rows higher in the second. `busy` is the slot's lock (server/slots.py), taken at
    the grant before any engine work on the slot, so no request's own work moves an unlocked slot's rows. Slot 0 has
    no `served` count and is not judged. A drop is a trim or a reclaim and is never a move. `then_busy`: the slot was
    granted in the sample after (the stale round's rows then sit under the next request: design 3, last paragraph)."""
    seq = [s for s in samples if s.get("slots")]
    moves = []
    for k in range(1, len(seq)):
        prev = {x["i"]: x for x in seq[k - 1]["slots"]}
        nxt = {x["i"]: x for x in seq[k + 1]["slots"]} if k + 1 < len(seq) else {}
        for x in seq[k]["slots"]:
            y = prev.get(x["i"])
            if not x["i"] or y is None or x["busy"] is not False or y["busy"] is not False:
                continue
            if x["served"] is None or x["served"] != y["served"]:
                continue
            z = nxt.get(x["i"])
            for key in ("pkv", "kv", "rows"):
                a, b = y.get(key), x.get(key)
                if a is not None and b is not None and b > a:
                    moves.append({"slot": x["i"], "t": seq[k]["t"], "what": key, "from": a, "to": b,
                                  "served": x["served"], "then_busy": bool(z and z["busy"])})
    return moves


def overlap_at(s: dict, c: dict | None, a_recs: list[dict]) -> bool:
    """B in phase prefill (4,096 rows or more on its slot) and two lanes in phase runner, while C and an A are open."""
    sl = s.get("slots") or []
    b = any(x["phase"] == "prefill" and max(x["pkv"] or 0, x["kv"] or 0) >= 4096 for x in sl)
    runners = sum(1 for x in sl if x["phase"] == "runner")
    t = s["t"]

    def open_(r):
        return r is not None and r["t0"] <= t <= (r["t_close"] if r.get("t_close") is not None else r["t_end"] or 1e18)
    return b and runners >= 2 and open_(c) and any(open_(r) for r in a_recs)


def verdict_of(it: dict) -> str:
    if it["fault"] and (IMA.search(it["fault"]) or it["log"]["ima"]):
        return "fault"
    if it["log"]["ima"]:
        return "fault"
    if it["moves"]:
        return "stale"
    if it["fault"] or it["log"]["guard"] or it["log"]["round_failed"]:
        return "error"
    return "clean"


def unusable(knobs: dict, min_lanes: int) -> str | None:
    """Why this server cannot show the race, or None. The round runner: the [limits] line's `round_runner` (the
    setting) decides; without the line, /health's `parallel.round_runner` after the warm-up (the runner's report,
    null until a request has gone through it)."""
    lanes = knobs.get("lanes") or int(knobs.get("limits_lanes") or 0)
    lr = knobs.get("limits_round_runner")
    if lanes < min_lanes or not (lr == "on" or (lr is None and knobs.get("runner"))):
        return (f"needs {min_lanes} lanes and the round runner (lanes {lanes}; [limits] round_runner {lr}, /health "
                f"parallel.round_runner after the warm-up {'present' if knobs.get('runner') else 'null'})")
    return None


def limits_line(path: str | None) -> dict:
    """The key=value pairs of the server log's last `[limits]` line ({} without a log or a line)."""
    if not path or not os.path.exists(path):
        return {}
    last = None
    with open(path, errors="replace") as fh:
        for ln in fh:
            if ln.startswith("[limits]"):
                last = ln
    return dict(re.findall(r"(\w+)=(\S+)", last or ""))


def read_log(path: str | None, off: int | None) -> tuple:
    """(the server log's lines after `off`, the new offset)."""
    if not path or off is None or not os.path.exists(path):
        return [], off
    with open(path, "rb") as fh:
        fh.seek(off)
        data = fh.read()
    return data.decode("utf-8", "replace").splitlines(), off + len(data)


def log_facts(lines: list[str]) -> dict:
    out = {"ima": sum(1 for ln in lines if IMA.search(ln)), "round_failed": sum(1 for ln in lines if
                                                                                  "[runner] the round failed" in ln),
           "traceback": sum(1 for ln in lines if ln.startswith("Traceback")),
           "guard": sum(1 for ln in lines if "CaptureGuardError" in ln), "req": {}, "stack": []}
    for ln in lines:
        m = REQ.match(ln)
        if m:
            out["req"][m.group(1)] = {"prompt": int(m.group(2)), "completion": int(m.group(3)), "finish": m.group(4),
                                      "ms": int(m.group(5)), "slot": int(m.group(6)),
                                      "decode_ms": float(m.group(7)) if m.group(7) else None}
    for i, ln in enumerate(lines):
        if "[runner] the round failed" in ln or ln.startswith("Traceback"):
            out["stack"] = [x[:300] for x in lines[i:i + 60]]
            break
    return out


# ------------------------------------------------------------------ one iteration
def body_of(model: str, content: str, max_tokens: int, cid: str) -> dict:
    return {"model": model, "stream": True, "temperature": 0, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "conversation_id": cid, "messages": [{"role": "user", "content": content}]}


def iteration(a, k: int, doc: str, model: str, base: str, log=print) -> dict:
    nonce = secrets.token_hex(8)
    abort, stop_poll = threading.Event(), threading.Event()
    t_ref = time.time()
    samples: list[dict] = []
    code0, h0 = get_health(base)
    s0 = compact(0.0, code0, h0)
    errors0 = s0["rr_errors"]
    samples.append(s0)
    fault: list = []
    _, log_off = read_log(a.server_log, os.path.getsize(a.server_log) if a.server_log and
                          os.path.exists(a.server_log) else None)

    def poll():
        down = 0
        while not stop_poll.is_set():
            code, h = get_health(base)
            s = compact(time.time() - t_ref, code, h)
            samples.append(s)
            down = down + 1 if code is None else 0
            f = fault_of(s, errors0) or (f"/health unreachable {down} times in a row" if down >= 6 else None)
            if f and not fault:
                fault.append(f)
                abort.set()
            stop_poll.wait(a.poll_s)

    recs: dict = {}

    def go(name: str, delay: float, body: dict, close_burst: int, timeout: float):
        if abort.wait(max(0.0, t_ref + delay - time.time())):
            recs[name] = None
            return
        recs[name] = stream(a.base_url, body, timeout_s=timeout, abort=abort, close_burst=close_burst, t_ref=t_ref)

    b_text = (f"Reference {nonce}: read this document and then answer the question at the end.\n\n{doc}\n\n"
              "Question: summarise the document in three sentences.")
    plan = [("B", 0.0, body_of(model, b_text, a.b_max_tokens, f"ima-b-{nonce}"), 0, a.b_timeout),
            ("C", a.c_delay, body_of(model, C_PROMPT, a.c_max_tokens, f"ima-c-{nonce}"), 0, a.c_timeout)]
    bursts = [int(x) for x in a.a_close_bursts.split(",") if x.strip()] or [1]
    for j in range(a.a_count):
        plan.append((f"A{j + 1}", a.a_start + j * a.a_every,
                     body_of(model, A_PROMPTS[j % len(A_PROMPTS)], a.a_max_tokens, f"ima-a{j + 1}-{nonce}"),
                     bursts[j % len(bursts)], a.a_timeout))
    pt = threading.Thread(target=poll, name="health", daemon=True)
    pt.start()
    ths = [threading.Thread(target=go, args=p, name=p[0], daemon=True) for p in plan]
    for th in ths:
        th.start()
    for th in ths:
        th.join(timeout=max(a.b_timeout, a.c_timeout, a.a_timeout) + 30)
    if not abort.is_set():
        abort.wait(a.settle_s)
    stop_poll.set()
    pt.join(timeout=10)
    code, h = get_health(base)
    last = compact(time.time() - t_ref, code, h)
    samples.append(last)
    f = fault_of(last, errors0)
    if f and not fault:
        fault.append(f)
    lines, _ = read_log(a.server_log, log_off)
    lf = log_facts(lines)
    a_recs = [recs.get(f"A{j + 1}") for j in range(a.a_count)]
    a_recs = [r for r in a_recs if r is not None]
    ov = sum(1 for s in samples if s.get("slots") and overlap_at(s, recs.get("C"), a_recs))
    for name, r in recs.items():
        if r is not None and r.get("cid") in lf["req"]:
            r["server"] = lf["req"][r["cid"]]
    it = {"k": k, "nonce": nonce, "wall_s": round(time.time() - t_ref, 1), "recs": recs, "overlap_samples": ov,
          "samples": len(samples), "fault": fault[0] if fault else None, "moves": idle_moves(samples),
          "log": {x: lf[x] for x in ("ima", "round_failed", "traceback", "guard")}, "stack": lf["stack"],
          "turn": last["turn"], "slots_end": [{"i": x["i"], "busy": x["busy"], "kv": x["pkv"], "rows": x["rows"]}
                                              for x in last.get("slots") or []]}
    it["verdict"] = verdict_of(it)
    it["counted"] = it["verdict"] == "fault" or ov > 0
    ab = [r for r in a_recs if r.get("closed")]
    srv = [r["server"] for r in a_recs if r.get("server")]
    b, c = recs.get("B") or {}, recs.get("C") or {}
    gone = ",".join("%s:%s" % (x["slot"], x["ms"]) for x in srv)
    kv_end = " ".join("%s:%s/%s" % (x["i"], x["kv"], x["rows"]) for x in it["slots_end"])
    tail = "" if it["counted"] else " (no overlap: does not count)"
    log(f"[ima] iter {k}: {it['verdict']}{tail}; wall {it['wall_s']} s; B ttft {b.get('t_first')} s finish "
        f"{b.get('finish')} err {b.get('error')}; C deltas {c.get('deltas')} finish {c.get('finish')}; A closed "
        f"{len(ab)}/{len(a_recs)}, server abandons {sum(1 for x in srv if x['finish'] == 'abandoned')} (slot:ms {gone}); "
        f"overlap samples {ov}/{len(samples)}; fault {it['fault']}; idle-slot moves {it['moves'][:3]}; log ima "
        f"{lf['ima']} round_failed {lf['round_failed']} guard {lf['guard']}", flush=True)
    log(f"[ima] iter {k}: turn owner {last['turn']['owner']} waiting {last['turn']['waiting']} vtime_s "
        f"{last['turn']['vtime']}; slots kv/rows {kv_end}", flush=True)
    it["samples_kept"] = samples if (it["verdict"] != "clean") else samples[-3:]
    return it


# ------------------------------------------------------------------ documents
def load_doc(spec: str, long_docs: str | None, tokenizer: str | None, row: int) -> str:
    """`ids-L`: row `row` of tools/longprompts.py's set at length L, decoded with the set's tokenizer (tools/soak.py
    `load_docs`); `synthetic:N`: about N tokens of numbered sentences (no tokenizer needed)."""
    if spec.startswith("synthetic:"):
        n = int(spec.split(":", 1)[1])
        rng = random.Random(1121 + row)
        words = ("river stone harbour lantern window engine garden market letter winter signal valley copper "
                 "orchard bridge station archive meadow compass beacon").split()
        out, toks = [], 0
        while toks < n:
            s = f"{len(out) + 1}. " + " ".join(rng.choice(words) for _ in range(rng.randint(8, 16))) + "."
            out.append(s)
            toks += len(s.split()) + 4
        return "\n".join(out)
    import numpy as np
    from transformers import AutoTokenizer
    L = int(spec.split("-", 1)[1])
    ids = np.load(os.path.join(long_docs, f"ids-{L}.npy"))
    tok = AutoTokenizer.from_pretrained(tokenizer)
    return tok.decode(ids[row % len(ids)].tolist())


def doc_rows(spec: str, long_docs: str | None) -> int:
    if spec.startswith("synthetic:"):
        return 1
    import numpy as np
    return len(np.load(os.path.join(long_docs, f"ids-{int(spec.split('-', 1)[1])}.npy"), mmap_mode="r"))


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8160/v1")
    ap.add_argument("--model", default=None, help="default: the first of /v1/models")
    ap.add_argument("--iterations", type=int, default=6, help="counted iterations to run without the race")
    ap.add_argument("--max-attempts", type=int, default=None, help="default: iterations + 3")
    ap.add_argument("--budget-min", type=float, default=0.0, help="no new iteration after this many minutes (0: none)")
    ap.add_argument("--doc", default="ids-32768", help="ids-L (with --long-docs, --tokenizer) or synthetic:N")
    ap.add_argument("--long-docs", default=os.path.expanduser("~/pub-bench/longprompts"))
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--server-log", default=os.environ.get("ENGINE_LOG"))
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--b-max-tokens", type=int, default=32)
    ap.add_argument("--c-delay", type=float, default=1.0)
    ap.add_argument("--c-max-tokens", type=int, default=512,
                    help="C decodes one round a prefill chunk while B prefills: 512 outlasts the prefill and ends soon after")
    ap.add_argument("--a-count", type=int, default=6)
    ap.add_argument("--a-start", type=float, default=3.0)
    ap.add_argument("--a-every", type=float, default=6.0)
    ap.add_argument("--a-max-tokens", type=int, default=256)
    ap.add_argument("--a-close-bursts", default="1", help="the burst each A closes at, in turn (1: its first delta)")
    ap.add_argument("--poll-s", type=float, default=0.5)
    ap.add_argument("--settle-s", type=float, default=4.0)
    ap.add_argument("--gap-s", type=float, default=3.0)
    ap.add_argument("--b-timeout", type=float, default=900.0)
    ap.add_argument("--c-timeout", type=float, default=900.0)
    ap.add_argument("--a-timeout", type=float, default=600.0)
    ap.add_argument("--min-lanes", type=int, default=4)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    base = re.sub(r"/v1/?$", "", a.base_url.rstrip("/"))
    code, h = get_health(base)
    par = h.get("parallel") or {}
    knobs = {"status": h.get("status"), "lanes": par.get("n"), "max_running": par.get("max_running"),
             "turn_quantum_ms": (par.get("turn") or {}).get("quantum_ms"),
             "runner": par.get("round_runner") is not None, "end_reread": (h.get("cache") or {}).get("end_reread"),
             "ceiling_gb": (h.get("memory") or {}).get("ceiling_gb"), "peak_gb": (h.get("memory") or {}).get("peak_gb"),
             "max_queue": h.get("max_queue"), "version": h.get("version")}
    # the start's [limits] line: the round runner's setting (/health's `parallel.round_runner` is the runner's own
    # report, null until the first request through it makes the runner) and the lanes
    lim = limits_line(a.server_log)
    knobs.update(limits_round_runner=lim.get("round_runner"), limits_lanes=lim.get("lanes"),
                 limits_ceiling=lim.get("MEM_CEILING_GIB"), limits_peak=lim.get("MEM_PEAK_GIB"))
    out = {"knobs": knobs, "args": vars(a), "iterations": [], "counted": 0, "result": None}

    def finish(result: str, rc: int) -> int:
        out["result"] = result
        if a.json_out:
            with open(a.json_out, "w") as fh:
                json.dump(out, fh, indent=1, default=str)
        print(f"[ima] RESULT {result} ({out['counted']} counted of {len(out['iterations'])} iterations); exit {rc}",
              flush=True)
        return rc

    if code != 200 or knobs["status"] != "ok":
        print(f"[ima] server {base}: {json.dumps(knobs)}", flush=True)
        return finish(f"server not ok ({code}, {knobs['status']})", 2)
    model = a.model
    if not model:
        try:
            with urllib.request.urlopen(a.base_url.rstrip("/") + "/models", timeout=30) as r:
                model = (json.loads(r.read()).get("data") or [{}])[0].get("id") or "TandemLLM"
        except Exception:                                            # noqa: BLE001
            model = "TandemLLM"
    # one short request first: it makes the runner (a drafted lane decodes through it) and warms the lane
    w = stream(a.base_url, body_of(model, "Reply with the single word: ready", 8, f"ima-warm-{secrets.token_hex(4)}"),
               timeout_s=600, abort=threading.Event())
    code, h = get_health(base)
    knobs["runner"] = ((h.get("parallel") or {}).get("round_runner") is not None)
    knobs["warmup"] = {k: w.get(k) for k in ("status", "deltas", "finish", "error", "t_end")}
    print(f"[ima] server {base}: {json.dumps(knobs)}", flush=True)
    if code != 200 or w.get("status") != 200 or w.get("error"):
        return finish(f"the warm-up request failed (/health {code}, warm-up {knobs['warmup']})", 2)
    why = unusable(knobs, a.min_lanes)
    if why:
        return finish(why, 2)
    rows = doc_rows(a.doc, a.long_docs)
    t_start = time.time()
    attempts = a.max_attempts or a.iterations + 3
    for k in range(1, attempts + 1):
        if out["counted"] >= a.iterations:
            break
        if a.budget_min and time.time() - t_start > a.budget_min * 60:
            print(f"[ima] the budget of {a.budget_min:g} min is spent", flush=True)
            break
        doc = load_doc(a.doc, a.long_docs, a.tokenizer, k - 1)
        it = iteration(a, k, doc, model, base)
        out["iterations"].append(it)
        out["counted"] += int(it["counted"])
        if it["verdict"] in ("fault", "stale"):
            return finish(f"race shown: {it['verdict']} at iteration {k} ({it['fault'] or it['moves'][:2]})", 1)
        if it["verdict"] == "error":
            return finish(f"another error at iteration {k} ({it['fault']}; log {it['log']})", 3)
        time.sleep(a.gap_s)
    if out["counted"] >= a.iterations:
        return finish(f"no race in {out['counted']} counted iterations", 0)
    return finish(f"only {out['counted']} counted iterations of {a.iterations} (docs rows {rows})", 2)


# ------------------------------------------------------------------ self-test (no GPU, no engine)
def self_test() -> int:
    import http.server
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(f"[self-test] {'ok  ' if cond else 'FAIL'} {name}", flush=True)

    def sl(i, busy, served, kv, rows, phase=""):
        return {"i": i, "busy": busy, "served": served, "phase": phase, "owner": None, "kv": kv, "pkv": kv,
                "rows": rows}

    def smp(t, slots):
        return {"t": t, "slots": slots}
    # the idle-slot rule
    s = [smp(1, [sl(0, False, 0, 10, 10), sl(1, False, 4, 46, 538)]),
         smp(2, [sl(0, False, 0, 99, 99), sl(1, False, 4, 48, 2336)]),
         smp(3, [sl(0, False, 0, 99, 99), sl(1, False, 4, 48, 2336)])]
    mv = idle_moves(s)
    check("a stale round on an idle side slot is a move (rows and kv up, served the same)",
          {m["what"] for m in mv} == {"pkv", "kv", "rows"} and all(m["slot"] == 1 for m in mv))
    check("slot 0 is not judged", not any(m["slot"] == 0 for m in mv))
    check("a drop (a trim, a reclaim) is no move",
          not idle_moves([smp(1, [sl(1, False, 4, 46, 2336)]), smp(2, [sl(1, False, 4, 46, 538)])]))
    check("a new request in between (served up) is no move",
          not idle_moves([smp(1, [sl(1, False, 4, 46, 538)]), smp(2, [sl(1, False, 5, 300, 900)])]))
    check("a busy slot is no move",
          not idle_moves([smp(1, [sl(1, True, 4, 46, 538)]), smp(2, [sl(1, True, 4, 60, 2336)])]))
    mv = idle_moves([smp(1, [sl(1, False, 4, 46, 538)]), smp(2, [sl(1, False, 4, 46, 900)]),
                     smp(3, [sl(1, True, 4, 46, 900)])])
    check("rows up on an idle slot granted in the next sample are a move, marked then_busy",
          len(mv) == 1 and mv[0]["then_busy"])
    check("a move in the last sample counts (the server died after it)",
          bool(idle_moves([smp(1, [sl(2, False, 7, 46, 538)]), smp(2, [sl(2, False, 7, 49, 2336)])])))
    # the overlap rule
    c = {"t0": 1.0, "t_close": None, "t_end": 60.0}
    a1 = {"t0": 3.0, "t_close": 12.0, "t_end": 12.0}
    s_ok = smp(10, [sl(0, True, 0, 8192, 8500, "prefill"), sl(1, True, 3, 40, 2336, "runner"),
                    sl(2, True, 3, 30, 2336, "runner")])
    check("overlap: B prefilling and two runner lanes while C and an A are open", overlap_at(s_ok, c, [a1]))
    check("no overlap once the A closed", not overlap_at(dict(s_ok, t=13), c, [a1]))
    check("no overlap with B under 4,096 rows",
          not overlap_at(smp(10, [sl(0, True, 0, 0, 0, "prefill"), sl(1, True, 3, 40, 1, "runner"),
                                  sl(2, True, 3, 30, 1, "runner")]), c, [a1]))
    check("no overlap with one runner lane",
          not overlap_at(smp(10, [sl(0, True, 0, 8192, 1, "prefill"), sl(1, True, 3, 40, 1, "runner")]), c, [a1]))
    # verdicts and faults
    base_it = {"fault": None, "log": {"ima": 0, "guard": 0, "round_failed": 0}, "moves": []}
    check("an illegal memory access in the log is the fault",
          verdict_of(dict(base_it, log={"ima": 2, "guard": 0, "round_failed": 1})) == "fault")
    check("a /health 500 naming it is the fault",
          verdict_of(dict(base_it, fault="/health 500: CUDA error: an illegal memory access was encountered")) == "fault")
    check("an idle-slot move is stale", verdict_of(dict(base_it, moves=[{"slot": 1}])) == "stale")
    check("a guard error is an error, not the race",
          verdict_of(dict(base_it, log={"ima": 0, "guard": 1, "round_failed": 1})) == "error")
    check("nothing is clean", verdict_of(base_it) == "clean")
    check("errors up is a fault line", fault_of({"code": 200, "status": "ok", "failed": None, "rr_errors": 1}, 0))
    check("a 500 is a fault line", fault_of({"code": 500, "status": "failed", "failed": "x", "rr_errors": 0}, 0))
    check("an ok sample is none", fault_of({"code": 200, "status": "ok", "failed": None, "rr_errors": 0}, 0) is None)
    check("the [req] line parses", log_facts(["[req] chatcmpl-d286 stream prompt=43 completion=3 finish=abandoned "
                                              "8330 ms 0.24 tok/s slot=1 blocks=1 committed=2 decode_ms=3815.2 x"])
          ["req"].get("chatcmpl-d286", {}).get("slot") == 1)
    check("the runner on by [limits] with /health's runner null at idle is usable (the box, 2026-11-22)",
          unusable({"lanes": 8, "limits_round_runner": "on", "runner": False}, 4) is None)
    check("no [limits] line: the runner seen after the warm-up is usable",
          unusable({"lanes": 8, "limits_round_runner": None, "runner": True}, 4) is None)
    check("the runner off by [limits] is not", unusable({"lanes": 8, "limits_round_runner": "off", "runner": True}, 4))
    check("one lane is not", unusable({"lanes": 1, "limits_round_runner": "on"}, 4))
    import tempfile as _tf
    with _tf.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write("[server] x\n[limits] lanes=8 kv_pool=on MEM_CEILING_GIB=84.1 round_runner=on runner_pipe=off\n")
    lm = limits_line(fh.name)
    os.unlink(fh.name)
    check("the [limits] line parses", lm.get("round_runner") == "on" and lm.get("lanes") == "8")
    dec = _Body(True)
    got = b"".join(dec.feed(x) for x in (b"5\r\nhel", b"lo\r\n6\r\n world\r", b"\n0\r\n\r\n"))
    check("the chunked body decodes across reads", got == b"hello world" and dec.done)
    # the client against a fake server: the RST close, a hung stream, a 500
    seen: dict = {"write_error": None, "writes": 0}

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path.startswith("/fail"):
                b = json.dumps({"error": {"message": "the engine needs a restart"}}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Connection", "close")
            self.end_headers()
            if self.path.startswith("/hang"):
                time.sleep(6)
                return
            try:
                self.wfile.write(b'data: {"id":"x1","choices":[{"delta":{"role":"assistant","content":""}}]}\n\n')
                for i in range(60):
                    self.wfile.write(f'data: {{"id":"x1","choices":[{{"delta":{{"content":"t{i} "}}}}]}}\n\n'.encode())
                    self.wfile.flush()
                    seen["writes"] = i + 1
                    time.sleep(0.03)
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError) as exc:
                seen["write_error"] = type(exc).__name__

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ab = threading.Event()
        r = stream(f"http://127.0.0.1:{port}/v1", {"x": 1}, timeout_s=20, abort=ab, close_burst=1)
        time.sleep(1.0)
        check(f"the client closes at its first content delta (closed {r['closed']}, deltas {r['deltas']}, cid "
              f"{r['cid']})", r["closed"] and r["deltas"] == 1 and r["cid"] == "x1")
        check(f"the server's next writes fail at once (error {seen['write_error']} after {seen['writes']} writes)",
              seen["write_error"] is not None and seen["writes"] < 30)
        r = stream(f"http://127.0.0.1:{port}/v1", {"x": 1}, timeout_s=20, abort=ab)
        check(f"a whole stream reads to its end ({r['deltas']} deltas)", r["deltas"] == 60 and not r["closed"])
        ab2 = threading.Event()
        threading.Timer(1.0, ab2.set).start()
        t0 = time.time()
        r = stream(f"http://127.0.0.1:{port}/hang/v1", {"x": 1}, timeout_s=20, abort=ab2)
        check(f"a hung stream ends at the abort ({time.time() - t0:.1f} s)", r["aborted"] and time.time() - t0 < 3)
        r = stream(f"http://127.0.0.1:{port}/fail/v1", {"x": 1}, timeout_s=20, abort=threading.Event())
        check(f"a 500's body is read ({r['status']}, {r['error']})", r["status"] == 500 and "restart" in (r["error"] or ""))
    finally:
        srv.shutdown()
    n_bad = ok.count(False)
    print(f"[self-test] {len(ok) - n_bad}/{len(ok)} ok", flush=True)
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
