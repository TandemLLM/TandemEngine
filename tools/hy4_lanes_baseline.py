#!/usr/bin/env python3
"""ENG-485 step 0: one client's decode on today's one-lane build, the record the lanes work is held to.

With side slots on, every decode of a family moves to the round runner, one client too; step 3 of the lanes
work runs the same prompt through the runner and must come within 5 % of this record (decode tok/s and tokens
a drafted round). So this run is on the build as served today: one lane, the slot loop (no pool, so the
runner is never used), the profile's drafter, a cold read every time.

    python tools/hy4_lanes_baseline.py --label hy4-3.56 --tokenizer ~/models/hy4-flash-exl3-3.56 \\
        --env-file run/setup.env --server-log <the server's log> --out hy4-3.56-baseline.json

The server is started beforehand on the profile (ops/use-profile.sh hy4-3.56); the tool only sends requests
and reads /health and the server's log. It reuses tools/parallel_card.py's client: `/v1/completions` with the
prompt as text and `X-Tandem: raw` (no conductor or chat-layer step), the chat template's ids with thinking
off rendered to text and checked id for id against `/v1/tokenize`, `cache_prompt: false`.

WHAT IT RUNS: one fixed prose prompt of `--prompt-tokens` tokens (default 3,000: every decoded row reads
selected groups, past the sparse layers' 2,048 first positions), greedy, `--max-tokens` out (600), `--runs`
times (2) after `--warmups` (1). The prompt is the same text in every run and is written to `--prompt-out`,
so step 3 sends the same text (`--prompt-in`). Per run: decode tok/s = (completion tokens - 1) / (last token -
first token), the card's formula; tokens a drafted round = committed / blocks from the run's `[req]` line;
the text's sha256. Every run's text must be the first run's.

It refuses to measure unless the server serves one lane (/health `parallel` absent or `n` 1) and is idle.
Every limit knob the env file sets is recorded beside the rows. Pass line:

    hy4-<bpw>: slot loop c1 X tok/s, Y tokens a round (prompt P tokens, N out, R runs, text equal)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools import parallel_card as pc  # noqa: E402

#: the knobs a lanes row records (the benchmarking rules: every limit knob, defaults included)
KNOBS = ("QSE_WEIGHTS", "MAX_LEN", "QSE_PARALLEL", "QSE_SIDE_MAX_LEN", "QSE_KV_POOL", "QSE_ROUND_RUNNER",
         "QSE_RUNNER_PIPE", "QSE_PARALLEL_BATCH", "QSE_PARALLEL_FLOOR_GB", "QSE_KV_HARD_FLOOR_GB",
         "QSE_PREFILL_BURST", "QSE_DISK_CACHE", "QWEN38_HY4_GRAPHS", "QWEN38_HY4_DFLY", "QWEN38_HY4_DRAFT_FREEZE",
         "QWEN38_HY4_TABLE", "QWEN38_HY4_TABLE_CACHE_GB")

REQ = re.compile(r"\[req\] \S+ \S+ prompt=(\d+) completion=(\d+) .*?blocks=(\d+) committed=(\d+)")


def req_lines(log_path: str, start: int) -> list[tuple[int, int, int, int]]:
    """(prompt, completion, blocks, committed) of every `[req]` line written after byte `start`."""
    out = []
    with open(log_path, "rb") as f:
        f.seek(start)
        for raw in f:
            m = REQ.search(raw.decode("utf-8", "replace"))
            if m:
                out.append(tuple(int(x) for x in m.groups()))
    return out


def one_lane(h: dict) -> tuple[bool, str]:
    p = h.get("parallel")
    if not p:
        return True, "no pool"
    n = int(p.get("n") or 1)
    return n == 1, f"parallel.n {n}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="TandemLLM")
    ap.add_argument("--label", required=True, help="the profile, e.g. hy4-3.56")
    ap.add_argument("--tokenizer", required=True, help="the served weights' directory")
    ap.add_argument("--env-file", required=True, help="the server's run/setup.env (its knobs are recorded)")
    ap.add_argument("--server-log", required=True, help="the server's log, for the [req] lines")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=str(REPO / "bench" / "card-prompts-v1.json"))
    ap.add_argument("--prompt-tokens", type=int, default=3000)
    ap.add_argument("--prompt-in", default=None, help="send this saved prompt text instead of building one")
    ap.add_argument("--prompt-out", default=None, help="default: <out>.prompt.txt")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--allow-lanes", action="store_true",
                    help="step 3: measure a server with side slots (the runner), on the saved prompt")
    a = ap.parse_args(argv)

    log = lambda *x: print(f"[baseline {time.strftime('%H:%M:%S')}]", *x, flush=True)   # noqa: E731
    h0 = pc.health(a.base)
    if "error" in h0:
        print(f"no server at {a.base}: {h0['error']}", file=sys.stderr)
        return 2
    ok, why = one_lane(h0)
    if not ok and not a.allow_lanes:
        print(f"refused: the record is one lane on the slot loop, the server has {why}", file=sys.stderr)
        return 2
    if not pc.wait_idle(a.base, 300):
        print("refused: the server is not idle", file=sys.stderr)
        return 2
    env = pc.envfile(a.env_file)
    knobs = {k: env.get(k, os.environ.get(k)) for k in KNOBS}
    cl = pc.Client(a.base, a.model, {"temperature": 0.0}, {"X-Tandem": "raw"}, tok=pc.load_tokenizer(a.tokenizer))
    if a.prompt_in:
        text = Path(a.prompt_in).read_text()
        planned = None
    else:
        pj = pc.load_prompts(Path(a.prompts))
        offset = pc.tokenize_offset(cl, pj)
        head = "Baseline 0000-lanes. Read the text below, then summarise it in detail.\n"
        k, planned = pc.prefill_words(cl.base, pj, a.prompt_tokens, head, offset)
        text = cl.render([{"role": "user", "content": head + " ".join(pc.filler_words(pj, a.prompt_tokens)[:k])}])
    p_out = a.prompt_out or (a.out + ".prompt.txt")
    Path(p_out).write_text(text)
    prompt_sha = hashlib.sha256(text.encode()).hexdigest()[:16]
    log(f"{a.label}: {why}; prompt {planned or '?'} tokens planned, sha256 {prompt_sha}, written to {p_out}")
    for i in range(a.warmups):
        r = cl.stream(text, min(a.max_tokens, 64), 1)
        log(f"warm-up {i + 1}: {r['completion_tokens']} tokens, TTFT {r['ttft_s']} s")
    runs = []
    for i in range(a.runs):
        start = os.path.getsize(a.server_log)
        r = cl.stream(text, a.max_tokens, 1)
        time.sleep(1.0)                                   # the [req] line follows the stream's end
        lines = [x for x in req_lines(a.server_log, start)
                 if x[0] == r["prompt_tokens"] and x[1] == r["completion_tokens"]]
        if lines:
            _, _, blocks, committed = lines[-1]
            r["blocks"], r["committed"] = blocks, committed
            r["tokens_a_round"] = round(committed / blocks, 3) if blocks else None
        else:
            r["blocks"] = r["committed"] = r["tokens_a_round"] = None
        runs.append(r)
        log(f"run {i + 1}: {r['decode_tok_s']} tok/s, {r['tokens_a_round']} tokens a round, "
            f"{r['completion_tokens']} tokens, prompt {r['prompt_tokens']}, TTFT {r['ttft_s']} s, {r['finish_reason']}")
    h1 = pc.health(a.base)
    same = len({r["text_sha256"] for r in runs}) == 1
    rate = pc.med([r["decode_tok_s"] for r in runs])
    tpr = pc.med([r["tokens_a_round"] for r in runs])
    raw = {"tool": "tools/hy4_lanes_baseline.py", "label": a.label, "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "lanes": why, "knobs": knobs, "health_start": pc.health_record(h0), "health_end": pc.health_record(h1),
           "prompt": {"sha256": prompt_sha, "planned_tokens": planned, "file": os.path.basename(p_out)},
           "max_tokens": a.max_tokens, "runs": runs, "texts_equal": same,
           "decode_tok_s": rate, "tokens_a_round": tpr}
    with open(a.out, "w") as f:
        json.dump(raw, f, indent=1)
        f.write("\n")
    p_tok = runs[0]["prompt_tokens"] if runs else None
    n_out = runs[0]["completion_tokens"] if runs else None
    print(f"{a.label}: slot loop c1 {rate} tok/s, {tpr} tokens a round (prompt {p_tok} tokens, {n_out} out, "
          f"{len(runs)} runs, text {'equal' if same else 'DIFFERS'})")
    if not same or tpr is None or rate is None:
        print("FAIL: the runs' texts differ or a run has no [req] line or no decode window", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
