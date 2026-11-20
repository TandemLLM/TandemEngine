#!/usr/bin/env python3
"""Where a Hy4-flash prompt chunk spends its time, by part (SPD-317, the prefill design's S0.1 and S0.4).

    python tools/hy4_prefill_profile.py --pack ~/models/hy4-flash-exl3-3.56 --text docs.txt [--rows 8192]
        [--arms grouped,pair] [--reps 2] [--repeat prose.txt] > profile.jsonl

One load through the engine's own loader (cold tables, the hot-row cache of --table-cache-gb), one engine with a state
of --rows + 64 rows. The prompt is the first --rows token ids of --text, raw (no template), in chunks of the engine's
PREFILL_CHUNK; "chunk 2" is the last chunk. For each arm (`pair`: the routed experts on the pair kernels, `grouped`:
on the family's grouped kernels, `group`: on the shared prefill kernel, where the engine has it):

  1. a warm-up of 4,608 random ids (every path compiled and launched);
  2. a control pass: the table cache emptied, every chunk unsynced, a device sync at each chunk's end: seconds and
     tok/s a chunk, cold table reads a chunk, and the pass's peak allocation above what was held before it (the
     prompt workspace; the S1 line compares the `group` arm with `grouped`);
  3. --reps profiles of the last chunk: the cache emptied, the chunks before it plain, then the last chunk with every
     part wrapped (`tools/fn_prefill_profile.PartTimer`: a device sync on entry and exit, time exclusive of the
     wrapped parts it calls; the chunk's forward is the root, so the parts add up to the synced chunk).

One JSON line a part, arm and rep, then the totals. The parts are gate 0's (hy4-quant/prefill2000/gate0.md, part C);
its first run on the tree before the prefill steps reproduces gate 0 C on the 3.56 pack within 1.5x: the parts sum to
about 24.1 s with the experts about 8.1 s (grouped), 34.2 s and 18.1 s (pair).

--repeat FILE (S0.4): the same chat request (FILE's text and an essay ask, the pack's template, thinking off) three
times on one serving object after the server's load-time self-check, each from row 0: its prefill seconds, and before
and after each request every table's hits, misses and blocks read and the allocator's reserved bytes. It tells how
much of a first request's extra time is cold table rows.

A tool only: no engine file imports it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: gate 0 C's parts (hy4-quant/prefill2000/gate0.md), by the engine method that does each; a method missing from the
#: engine is skipped. Exclusive times: a part never counts the wrapped parts it calls.
PARTS = {
    "_experts_chunk": "routed experts", "expert_pairs": "routed experts", "combine": "routed experts",
    "_experts_grouped": "routed experts",
    "attend_lists": "sparse attention",
    "lin": "trunk projections",
    "attention_chunk": "attention glue",
    "_index_chunk": "indexer", "_sparse_positions_chunk": "indexer", "group_keys": "indexer",
    "hc": "hyper-connections", "inject": "hyper-connections",
    "rms": "norms",
    "engram_apply": "Engram decode, apply, hashing, copy", "engram_rows": "Engram decode, apply, hashing, copy",
    "_engram_take": "Engram row reads",
    "route": "router",
    "forward_chunk": "rest", "mlp_chunk": "rest", "dense": "rest", "final": "rest", "embed": "rest", "rope": "rest",
}
#: gate 0 C's measured parts of the last chunk on the 3.56 pack (ms, grouped mean of two reps and pair), for the
#: first run's check
GATE0_C = {"grouped": {"sum": 24132.0, "routed experts": 8100.0}, "pair": {"sum": 34158.0, "routed experts": 18074.0}}


def say(kind: str, **kw) -> None:
    print(json.dumps({"check": kind, **kw}), flush=True)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_tables(eng) -> None:
    """Every cold table's hot-row cache emptied (as at load): the next chunk reads its rows from the drive."""
    for t in eng.tables.values():
        d = getattr(t, "direct", None)
        if d is not None:
            with d._mu:
                d.keys.fill(-1)
                d.ref.fill(False)


def table_stats(eng) -> list:
    out = []
    for li in sorted(eng.tables):
        d = getattr(eng.tables[li], "direct", None)
        out.append(dict(layer=li, **(dict(d.stats) if d is not None else {})))
    return out


def cold_reads(before: list, after: list) -> int:
    return sum(a.get("blocks_read", 0) - b.get("blocks_read", 0) for a, b in zip(before, after))


def set_arm(eng, arm: str) -> None:
    if arm == "group" and not hasattr(eng, "_experts_chunk"):
        raise SystemExit("arm group: this engine has no shared prefill expert path (_experts_chunk)")
    eng.prefill_grouped = arm == "grouped"
    if hasattr(eng, "prefill_experts"):
        eng.prefill_experts = "group" if arm == "group" else "rows"


def instrument(eng):
    """The engine's parts wrapped on the instance (and the window and sparse kinds of `attend_chunk` apart)."""
    from engine.models.hy4_flash.config import SPARSE
    from tools.fn_prefill_profile import PartTimer
    T = PartTimer()
    kinds = eng.cfg.layer_types
    for name, lab in PARTS.items():
        if hasattr(eng, name):
            T.wrap(eng, name, lab)
    T.wrap(eng, "attend_chunk", lambda q, K, allowed, li, *rest: "sparse attention" if kinds[li] == SPARSE else "window attention")
    if not hasattr(eng, "_engram_take"):
        # the read inside the layer loop (the tree before S2): the tables' packed reads apart from the CPU decode
        for t in eng.tables.values():
            if hasattr(t, "gather_packed"):
                T.wrap(t, "gather_packed", "Engram row reads")
    for h in eng.hashes.values():
        T.wrap(h, "ids", "Engram decode, apply, hashing, copy")
    return T


def chunks(eng, ids: list, st, upto: int | None = None) -> list:
    """The prompt's chunks forwarded on `st` (the last one left out when `upto` says so): seconds a chunk."""
    from engine.models.hy4_flash import engine as E
    C = E.PREFILL_CHUNK
    out = []
    for i in range(0, len(ids) if upto is None else upto, C):
        t0 = time.perf_counter()
        eng.forward_chunk(torch.tensor(ids[i:i + C]), st)
        sync()
        out.append(time.perf_counter() - t0)
    return out


def text_ids(a) -> list:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.pack, "tokenizer.json"))
    ids = tok.encode(open(a.text).read(), add_special_tokens=False).ids[:a.rows]
    if len(ids) < a.rows:
        raise SystemExit(f"--text holds {len(ids)} tokens, --rows asks for {a.rows}")
    return ids


def repeat(a, eng, cfg) -> None:
    """S0.4: one chat request three times on one serving object, after the server's own load-time self-check."""
    from tokenizers import Tokenizer
    from engine.models.hy4_flash.serving import Hy4FlashServing
    from tools.hy4_check import chat_ids
    srv = Hy4FlashServing(eng, a.max_len)
    tok = Tokenizer.from_file(os.path.join(a.pack, "tokenizer.json"))
    # the self-check the server runs at load (engine/models/hy4_flash/serving.py `build`)
    text = tok.encode("Count from one to twenty in words, then name the days of the week. ", add_special_tokens=False).ids
    n = min(eng.check_rows(), a.max_len - 64)
    res = eng.self_check([int(text[i % len(text)]) for i in range(max(n, 8))], widths=(2, 4, 6), log=lambda s: None,
                         graphs=srv.graphs is not None)
    srv.restart()
    say("repeat_load_check", ok=res["ok"], rows=res["rows"])
    ids, _ = chat_ids(a.pack, open(a.repeat).read(), "Write a detailed essay of at least 700 words that explains the text "
                      "above to a newcomer. Use plain prose paragraphs only: no lists, no tables, no headings, no code.")
    say("repeat_prompt", rows=len(ids), ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    for run in range(3):
        srv.restart()
        b, rb = table_stats(eng), torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        sync()
        t0 = time.perf_counter()
        srv.forward(torch.tensor(ids), start=0)
        sync()
        dt = time.perf_counter() - t0
        after, ra = table_stats(eng), torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        say("repeat", run=run, rows=len(ids), prefill_s=round(dt, 3), tok_s=round(len(ids) / dt, 1),
            tables_before=b, tables_after=after, cold_reads=cold_reads(b, after),
            reserved_before_gib=round(rb / (1 << 30), 3), reserved_after_gib=round(ra / (1 << 30), 3))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--text", help="the prompt's text file (raw, no template)")
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--arms", default="grouped,pair", help="comma list of pair, grouped, group")
    ap.add_argument("--reps", type=int, default=2, help="profiles of the last chunk an arm (gate 0: 2 grouped, 1 pair)")
    ap.add_argument("--table-cache-gb", type=float, default=2.0)
    ap.add_argument("--max-len", type=int, default=0, help="the state's rows (default --rows + 64)")
    ap.add_argument("--repeat", help="S0.4: a chat text, sent three times on one serving object")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.max_len = a.max_len or a.rows + 64

    from engine.models.hy4_flash import engine as E
    from engine.models.hy4_flash.load import load
    say("start", pack=a.pack, rows=a.rows, chunk=E.PREFILL_CHUNK, arms=a.arms, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    pack, W, cuts, tables, banks, timing = load(a.pack, a.device, log=lambda s: print(s, flush=True), table_policy="cold",
                                                table_cache_bytes=int(a.table_cache_gb * (1 << 30)))
    cfg = pack.cfg
    eng = E.Hy4FlashEngine(cfg, W, tables, device=a.device, max_len=a.max_len, banks=banks, cuts=cuts)
    knobs = {k: getattr(E, k) for k in ("PREFILL_CHUNK", "SUB", "SUB_LIST", "EXPERT_SUB", "EXPERTS_MIN") if hasattr(E, k)}
    knobs.update(experts_min=getattr(eng, "experts_min", None), table_cache_gb=a.table_cache_gb,
                 env={k: v for k, v in sorted(os.environ.items()) if k.startswith("QWEN38_")})
    say("load", timing=timing, banks=len(banks), knobs=knobs)
    if a.repeat:
        repeat(a, eng, cfg)
        if not a.text:
            return 0
    ids = text_ids(a)
    say("prompt", rows=len(ids), ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    g = torch.Generator().manual_seed(7)
    C, last = E.PREFILL_CHUNK, (len(ids) - 1) // E.PREFILL_CHUNK * E.PREFILL_CHUNK
    for arm in a.arms.split(","):
        set_arm(eng, arm)
        warm = torch.randint(3, min(cfg.vocab_size, cfg.engram_lookup_table_size), (4608,), generator=g).tolist()
        st = eng.new_state(a.max_len)
        chunks(eng, warm, st)
        reset_tables(eng)
        st = eng.new_state(a.max_len)
        b = table_stats(eng)
        secs, reads, prev = [], [], b
        held = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for i in range(0, len(ids), C):
            t0 = time.perf_counter()
            eng.forward_chunk(torch.tensor(ids[i:i + C]), st)
            sync()
            secs.append(round(time.perf_counter() - t0, 3))
            now = table_stats(eng)
            reads.append(cold_reads(prev, now))
            prev = now
        say("control", arm=arm, chunk_s=secs, chunk_tok_s=[round(min(C, len(ids) - i * C) / s, 1) for i, s in enumerate(secs)],
            tok_s=round(len(ids) / sum(secs), 1), cold_reads=reads,
            peak_above_held_gib=round((torch.cuda.max_memory_allocated() - held) / (1 << 30), 3) if torch.cuda.is_available() else None)
        for rep in range(a.reps):
            reset_tables(eng)
            st = eng.new_state(a.max_len)
            chunks(eng, ids, st, upto=last)
            b = table_stats(eng)
            T = instrument(eng)
            try:
                sync()
                t0 = time.perf_counter()
                eng.forward_chunk(torch.tensor(ids[last:]), st)
                sync()
                wall = time.perf_counter() - t0
            finally:
                T.restore()
            after = table_stats(eng)
            total = sum(T.t.values())
            for part in sorted(T.t, key=lambda k: -T.t[k]):
                say("part", arm=arm, rep=rep, part=part, ms=round(1e3 * T.t[part], 1), calls=T.n[part],
                    share=round(T.t[part] / total, 4))
            say("profile", arm=arm, rep=rep, rows=len(ids) - last, sum_ms=round(1e3 * total, 1), synced_chunk_ms=round(1e3 * wall, 1),
                cold_reads=cold_reads(b, after), tables=after)
            ref = GATE0_C.get(arm)
            if ref is not None and rep == 0:
                exp = 1e3 * T.t.get("routed experts", 0.0)
                say("against_gate0", arm=arm, sum_ratio=round(1e3 * total / ref["sum"], 3),
                    experts_ratio=round(exp / ref["routed experts"], 3),
                    within_1_5x=bool(1 / 1.5 <= 1e3 * total / ref["sum"] <= 1.5 and 1 / 1.5 <= exp / ref["routed experts"] <= 1.5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
