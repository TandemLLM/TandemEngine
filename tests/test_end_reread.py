"""The end read again (`Capabilities.reread_end`, engine/cache.py `ResidentPrefix.reread_end`, QWEN38_END_REREAD).

On a family whose prompt pass is row-invariant and whose decode rows are not prompt rows (Hy4-flash), a resume from
rows decode steps and verify blocks wrote shifts every later log-probability (engine-arch/endanchor design 1.2). At a
turn's end the answer's rows are read again on the prompt pass, and only that read's state is kept as the end anchor
and the session snapshot, so a resume there is a cold read's state, bit for bit.

A CPU reference cannot fail before the fix (its row pass and prompt pass share their sums), so the model here is a
double: `Marked`, the made-up model of tests/test_cache_any_model.py (`Toy`: one token at a time, a prompt pass
row-invariant by construction) whose row pass marks its rows: a decode step and a verify block's commit add `MARK`
to every row they write and to the recurrent state. A decode-written row is then unequal to a prompt row, as on the
device.

  E1   a family without reread_end keeps its end as before: kind end, taint set, the decode-written state
                                                     test_a_family_without_reread_end_keeps_its_end_as_before
  E2   turn 1 with decode steps, a tree with a rejected branch and a chain cut by trim_pending: the end's sections
       and views [0, e) equal a cold prefill of the same ids, bit for bit
                                                     test_the_end_is_a_cold_reads_state
  E3   the next turn resumes at the end (kind end), forwards only its new rows, logits bit-equal to a cold read
                                                     test_the_next_turn_resumes_at_the_end_and_equals_a_cold_read
  E4   the rows cleared: the next turn resumes from the session snapshot (kind session), logits bit-equal
                                                     test_the_session_snapshot_is_the_reread_state
  E5   no prompt end and no anchor: the read starts at row 0 and the end is a cold read's; a read refused by memory
       and a guest: no end, no session put; the next turn equals a cold read
                                                     test_without_a_prompt_end_nothing_decode_written_is_offered
  E6   after the read the prompt end is where it was, its buffers the same bytes; a copy that parts there resumes
       kind pend and equals a cold read             test_the_prompt_end_stays_where_it_was
  E7   turn 2 resumes at the end, turn 3 parts at turn 2's prompt end: equal to a cold read, no taint
                                                     test_a_prompt_end_above_a_read_end_is_exact
  E8   a read that raises keeps nothing: the rows valid to the prompt end, no end, the next turn equals cold
                                                     test_a_read_that_raises_keeps_nothing
  E9   hidden spans are read with the answer and replayed: the served ids equal the rows', and a cold read of the
       same ids                                      test_hidden_spans_are_read_and_replayed
  E10  the park and the spill carry `exact`: both round trips offer the end; an end without the key is refused
                                                     test_the_park_and_the_spill_carry_exact
  E11  a side slot's carry is a read end or none    test_a_side_slots_carry_is_a_read_end_or_none
  E12  QWEN38_END_REREAD=0: no end, the next turn resumes at the prompt end and equals a cold read
                                                     test_end_reread_off_resumes_at_the_prompt_end
  E13  the client has the stream's last chunk, its usage and [DONE] before the read starts; the lane is still the
       request's                                     test_a_streamed_client_has_its_last_bytes_before_the_read
  E14  the same for a JSON response                  test_a_json_client_has_its_body_before_the_read

E1 and E6 pass before the fix and after; E2 to E14 fail before it. E13 and E14 run the handler on the tiny Hy4-flash
(tests/hy4_tiny.py), which sets the capability.

Run: python tests/test_end_reread.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import pairapp  # noqa: E402,F401  (a pair test: TandemApp from TANDEMAPP_SRC, ENG-413 WP14)
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)
import torch  # noqa: E402

from engine import cache, disk_cache, spill  # noqa: E402
from test_cache_any_model import Toy  # noqa: E402

CH = 16
BIG = 1 << 30
NEW = [41, 43, 47, 53, 59, 61, 67]          # turn 2's new message
NEW3 = [71, 73, 79, 83]


class Marked(Toy):
    """`Toy` whose row pass marks the rows it writes (a decode step, a verify block's commit): `MARK` on each row of
    every view and on the recurrent state. `reread`: the capabilities a family that reads its end again sets
    (`prompt_rows`, `reread_end`); False: a family that keeps its decode-written end (E1). `fail_at`: a prompt-pass
    forward at that start raises (E8)."""

    MARK = 0.5

    def __init__(self, reread: bool = True, max_len: int = 512):
        super().__init__(max_len=max_len)
        self.caps = types.SimpleNamespace(prompt_rows=reread, reread_end=reread, caches=True)
        self._chain = None
        self._tree = None
        self.fail_at = None
        self.passes: list = []

    # the loop's verbs
    def forward(self, ids, start, last_only=True, prompt=False):
        self.settle()
        n = int(ids.numel())
        decode = n == 1 and start > 0 and not prompt
        if not decode and self.fail_at is not None and start >= self.fail_at:
            raise RuntimeError(f"the prompt pass failed at {start}")
        self.passes.append(("decode" if decode else "prompt", int(start), n))
        lg = Toy.forward(self, ids, start, last_only)
        if decode:
            self._mark(start, 1)
        return lg

    def _mark(self, p: int, r: int) -> None:
        self.va[:, p:p + r] += self.MARK
        self.vb[p:p + r] += self.MARK
        self.vc[0, :, p:p + r] += self.MARK
        self.s1.add_(self.MARK)

    def _sections(self):
        return [t.clone() for t in self.cache_state().sections().values()]

    def _load(self, st) -> None:
        for t, x in zip(self.cache_state().sections().values(), st):
            t.copy_(x)

    def forward_tree(self, block, parents, start):
        """A verify block: every row's logits along its path from the anchor; nothing kept until `commit_tree`."""
        self.settle()
        toks = [int(t) for t in block.view(-1).tolist()]
        st, out = self._sections(), []
        for i in range(len(toks)):
            path, j = [], i
            while j >= 0:
                path.append(j)
                j = parents[j]
            self._load(st)
            self.kv.length = start
            lg = Toy.forward(self, torch.tensor([toks[k] for k in reversed(path)]), start)
            out.append(lg[0, 0])
        self._load(st)
        self.kv.length = start
        self._tree = (toks, int(start), list(parents))
        self.passes.append(("verify", int(start), len(toks)))
        return torch.stack(out)

    def commit_tree(self, path):
        toks, start, _ = self._tree
        self.kv.length = start
        Toy.forward(self, torch.tensor([toks[k] for k in path]), start)
        self._mark(start, len(path))

    def forward_block(self, block, start):
        lg = self.forward_tree(block, list(range(-1, int(block.numel()) - 1)), start)
        self._chain = int(block.numel())
        self.kv.length = start + self._chain
        return lg

    def settle(self):
        if self._chain is not None:
            n, self._chain = self._chain, None
            self.commit_tree(list(range(n)))

    def trim_pending(self, n):
        if self._chain is None:
            return n == self.kv.length
        start = self._tree[1]
        if not start <= n <= start + self._chain:
            return False
        self._chain = None
        self.kv.length = start
        if n > start:
            self.commit_tree(list(range(n - start)))
        return True

    def discard_pending(self):
        self._chain = None

    def resume_at(self, n):
        self._chain = None
        self.kv.length = n

    def reset(self):
        self._chain = None
        super().reset()


def tokens(n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, 90, (n,), generator=g).tolist()


def _top(lg) -> int:
    return int(lg.reshape(-1, lg.shape[-1])[-1].argmax())


def _prefix(**kw) -> cache.ResidentPrefix:
    """The resident prefix as the server builds it for a family that reads its end again. The flag is set after the
    constructor, so the same test runs on the tree before the fix (where it is only an attribute)."""
    reread = kw.pop("reread", True)
    kw.setdefault("stash_bytes", BIG)
    res = cache.ResidentPrefix(kw.pop("budget", BIG), CH, end=True, **kw)
    res.end_reread = reread
    return res


class _pend:
    """The resident prefix keeps the last prompt's end (SPD-215; a test of another file may have turned it off)."""

    def __init__(self, on: bool = True):
        self.on = on

    def __enter__(self):
        self.keep, cache.PROMPT_END = cache.PROMPT_END, self.on
        return self

    def __exit__(self, *exc):
        cache.PROMPT_END = self.keep
        return False


class _reread_knob:
    """QWEN38_END_REREAD as the test sets it (the module global engine/cache.py reads it into)."""

    def __init__(self, on: bool):
        self.on = on

    def __enter__(self):
        self.keep = getattr(cache, "END_REREAD", None)
        cache.END_REREAD = self.on
        return self

    def __exit__(self, *exc):
        if self.keep is None:
            del cache.END_REREAD
        else:
            cache.END_REREAD = self.keep
        return False


def _answer(eng, lg, n0: int, chain: bool = True) -> list[int]:
    """Turn 1's answer as the served loop writes it: three decode steps, a tree whose row 1 is a rejected branch
    (rows 0, 2 and 3 committed), two decode steps, then (`chain`) a chain block whose commit waits and whose stream
    stopped at its second row (the engine ahead of the committed tokens, as `trim_pending` cuts it). Without the chain
    the last token is decided and not forwarded (`kv.length == n0 + len(out) - 1`)."""
    out, pos = [_top(lg)], n0
    for _ in range(3):
        lg = eng.forward(torch.tensor([out[-1]]), start=pos)
        pos += 1
        out.append(_top(lg))
    blk = [out[-1], 11, 13, 17]
    lgt = eng.forward_tree(torch.tensor(blk), [-1, 0, 0, 2], start=pos)
    eng.commit_tree([0, 2, 3])
    pos += 3
    out += [13, 17, _top(lgt[3])]
    for _ in range(2):
        lg = eng.forward(torch.tensor([out[-1]]), start=pos)
        pos += 1
        out.append(_top(lg))
    if chain:
        eng.forward_block(torch.tensor([out[-1], 19, 23, 29]), start=pos)
        out.append(19)
        assert eng.kv.length == n0 + len(out) + 2
    else:
        assert eng.kv.length == n0 + len(out) - 1
    assert "decode" in {p[0] for p in eng.passes} and "verify" in {p[0] for p in eng.passes}
    return out


def _remember(eng, res, ctx, n_prompt: int, session="A", store=None, hidden=()) -> None:
    """server/app.py `_remember_turn` after a turn on this engine and prefix (its close runs in place)."""
    from server import app
    keep = dict(app.STATE)
    app.STATE.clear()
    app.STATE.update(engine=eng, drafter=None, resident=res, state_store=store, session_cache=store is not None,
                     last_ctx=list(ctx), last_prompt_n=n_prompt, hidden_spans=[], suffix_store=None, verbose=False)
    try:
        app._remember_turn(list(ctx[:n_prompt]), list(ctx[n_prompt:]), session, hidden=list(hidden))
    finally:
        app.STATE.clear()
        app.STATE.update(keep)


def _turn1(res, prompt, eng=None, store=None, chain=True, session="A", hidden=()):
    """Turn 1: the prompt read through the caches, the answer (`_answer`), `_remember_turn`. Returns the engine, the
    context (prompt and answer) and the end `keep_end` keeps (the rows forwarded, within the committed tokens)."""
    eng = eng if eng is not None else Marked()
    lg, _, _ = cache.prefill(eng, None, list(prompt), "cpu", chunk=CH, resident=res, conv_id=session, store=store,
                             checkpoint=store is not None)
    ctx = list(prompt) + _answer(eng, lg, len(prompt), chain=chain)
    e = min(int(eng.kv.length), len(ctx))
    _remember(eng, res, ctx, len(prompt), session, store, hidden)
    return eng, ctx, e


def _cold(ids, reread=True):
    eng = Marked(reread)
    lg, _, _ = cache.prefill(eng, None, list(ids), "cpu", chunk=CH)
    return eng, lg


def _next(eng, res, ids, session="A", store=None):
    info: dict = {}
    lg, reused, fwd = cache.prefill(eng, None, list(ids), "cpu", chunk=CH, resident=res, conv_id=session,
                                    store=store, checkpoint=store is not None, info=info)
    return lg, {"kind": info.get("kind"), "reused": int(reused), "forwarded": int(fwd)}


def _same_state(a, b, n: int) -> list[str]:
    """What differs between two engines' states at `n` rows: every section, every view's rows [0, n)."""
    bad = []
    for (k, x), y in zip(a.cache_state().sections().items(), b.cache_state().sections().values()):
        if not torch.equal(x, y):
            bad.append(k)
    for (k, x, ax), (_, y, _) in zip(a.cache_state().views(), b.cache_state().views()):
        if not torch.equal(x.narrow(ax, 0, n), y.narrow(ax, 0, n)):
            bad.append(k)
    return bad


def _end_state_equals(res, ref) -> bool:
    return all(torch.equal(x, y) for x, y in zip(res._end_bufs, ref.cache_state().sections().values()))


# ------------------------------------------------------------------ E1

def test_a_family_without_reread_end_keeps_its_end_as_before():
    """E1 (passes before and after): no capability, no flag on the prefix: the end is the decode-written state
    (`taint` at the prompt's end), resumed at kind end, as today."""
    with _pend():
        res = _prefix(reread=False)
        eng, ctx, e = _turn1(res, tokens(40, 1), eng=Marked(reread=False))
        assert res.end is not None and res.end["at"] == e == len(ctx), (res.end, e)
        assert "exact" not in res.end and res.taint == 40, (res.end, res.taint)
        # the state kept is the one the decode steps and blocks wrote: unequal to a cold read's
        ref, _ = _cold(ctx[:e], reread=False)
        assert not _end_state_equals(res, ref)
        _, did = _next(eng, res, ctx + NEW)
        assert did["kind"] == "end" and did["reused"] == e, did
    return f"kept at {e} with taint 40: the decode-written state, resumed at the end as before"


# ------------------------------------------------------------------ E2, E3, E4

def test_the_end_is_a_cold_reads_state():
    """E2: decode steps, a tree with a rejected branch, a chain cut back: the end kept is a cold read's state, and
    every row below it a cold read's bytes."""
    with _pend():
        res = _prefix()
        eng, ctx, e = _turn1(res, tokens(40, 2))
        assert res.end is not None and res.end["at"] == e == len(ctx), (res.end, e, len(ctx))
        assert res.end.get("exact") is True and res.taint is None, (res.end, res.taint)
        ref, _ = _cold(ctx[:e])
        assert _end_state_equals(res, ref), "the end's sections are not a cold read's"
        assert _same_state(eng, ref, e) == [], _same_state(eng, ref, e)
    return f"end at {e}: sections and views [0, {e}) equal a cold read, bit for bit"


def test_the_next_turn_resumes_at_the_end_and_equals_a_cold_read():
    """E3: the next turn resumes at the end and forwards only its new rows; its logits are a cold read's."""
    with _pend():
        res = _prefix()
        eng, ctx, e = _turn1(res, tokens(40, 3))
        p2 = ctx + NEW
        lg, did = _next(eng, res, p2)
        assert did["kind"] == "end" and did["reused"] == e and did["forwarded"] == len(p2) - e, did
        ref, ref_lg = _cold(p2)
        assert torch.equal(lg, ref_lg), (lg - ref_lg).abs().max()
        assert _same_state(eng, ref, len(p2)) == []
        assert res.taint is None, res.taint
    return f"resumed at {e}, {len(p2) - e} rows forwarded: logits and state equal a cold read"


def test_the_session_snapshot_is_the_reread_state():
    """E4: the rows cleared, the next turn resumes from the session snapshot taken after the read (kind session),
    with no session's taint; its logits are a cold read's."""
    with _pend():
        store = cache.StateStore(BIG, chunk=CH)
        res = _prefix()
        eng, ctx, e = _turn1(res, tokens(40, 4), store=store, chain=False)
        res.clear()                                # the rows are gone: the store alone holds the turn
        p2 = ctx + NEW
        lg, did = _next(eng, res, p2, store=store)
        assert did["kind"] == "session" and did["reused"] == e, did
        assert res.taint is None, res.taint
        ref, ref_lg = _cold(p2)
        assert torch.equal(lg, ref_lg), (lg - ref_lg).abs().max()
        assert _same_state(eng, ref, len(p2)) == []
    return f"resumed from the session snapshot at {e}: logits and state equal a cold read"


# ------------------------------------------------------------------ E5

class _NoAnchors:
    """A memory policy that refuses every anchor (and nothing else)."""

    on = True

    def admit(self, level, nbytes):
        return level != "anchors"


def test_without_a_prompt_end_nothing_decode_written_is_offered():
    """E5: nothing decode-written is kept, as an end or a session snapshot, and the next turn equals a cold read:
    (a) no prompt end and every anchor refused: the read starts at the grid's first line (row 0) and what is kept is
    a cold read's state; (b) the read refused by the memory gate: no end, no snapshot; (c) a guest's own generation:
    none of its own, the chat's end stays."""
    from server import app
    got = []
    # (a) the prompt end off, every anchor refused: the read from the grid's first line
    with _pend(False):
        store = cache.StateStore(BIG, chunk=CH)
        res = _prefix(mem=_NoAnchors())
        eng, ctx, e = _turn1(res, tokens(40, 5), store=store, chain=False)
        assert res.pend is None and not res.anchors, (res.pend, res.anchors)
        assert res.end is not None and res.end.get("exact") is True and res.taint is None, (res.end, res.taint)
        assert res.stats.get("end_reread_last", {}).get("from") == "cold", res.stats.get("end_reread_last")
        assert all(x.exact for x in store._d.values() if x.kind == "session"), "a session snapshot of decode rows"
        assert _end_state_equals(res, _cold(ctx[:e])[0])
        p2 = ctx + NEW
        lg, did = _next(eng, res, p2)
        assert did["kind"] == "end" and did["reused"] == e, did
        assert torch.equal(lg, _cold(p2)[1])
        got.append("no prompt end: read from row 0, exact")
    # (b) the memory gate refuses the read
    real = app._reread_gate
    app._reread_gate = lambda cid: ((lambda n, rows: False), [])
    try:
        with _pend():
            store = cache.StateStore(BIG, chunk=CH)
            res = _prefix()
            eng, ctx, e = _turn1(res, tokens(40, 6), store=store, chain=False)
            assert res.end is None and res.valid == 40, (res.end, res.valid)
            assert not any(x.kind == "session" for x in store._d.values())
            p2 = ctx + NEW
            lg, did = _next(eng, res, p2)
            assert did["kind"] == "pend" and did["reused"] == 40, did
            assert torch.equal(lg, _cold(p2)[1])
            got.append("refused by memory: none")
    finally:
        app._reread_gate = real
    # (c) a guest between two turns: it keeps no end and no snapshot of its own; the chat's end stays and serves
    with _pend():
        store = cache.StateStore(BIG, chunk=CH)
        res = _prefix()
        eng, ctx, e = _turn1(res, tokens(60, 7), store=store, chain=False)
        g = tokens(8, 8)
        lgg, _, _ = cache.prefill(eng, None, g, "cpu", chunk=CH, resident=res, conv_id="B", store=store,
                                  checkpoint=True)
        assert res._stash is not None, "the short request was no guest"
        gctx = g + _answer(eng, lgg, len(g), chain=False)
        _remember(eng, res, gctx, len(g), "B", store)
        assert not any(x.kind == "session" and x.conv_id == "B" for x in store._d.values())
        assert res.end is not None and res.end["at"] == e, res.end
        p2 = ctx + NEW
        lg, did = _next(eng, res, p2, store=store)
        assert did["kind"] == "end" and did["reused"] == e, did
        assert torch.equal(lg, _cold(p2)[1])
        got.append("a guest: none of its own")
    return "; ".join(got) + "; every next turn equals a cold read"


# ------------------------------------------------------------------ E6, E7

def test_the_prompt_end_stays_where_it_was():
    """E6 (passes before and after): after the turn's close the prompt end is at the prompt's end with the same
    bytes; a copy that parts right after the prompt resumes there (kind pend) and equals a cold read."""
    with _pend():
        res = _prefix()
        eng = Marked()
        prompt = tokens(40, 9)
        lg, _, _ = cache.prefill(eng, None, prompt, "cpu", chunk=CH, resident=res, conv_id="A")
        held = [x.clone() for x in res._pend_bufs]
        ctx = prompt + _answer(eng, lg, 40)
        _remember(eng, res, ctx, 40)
        assert res.pend["at"] == 40 and all(torch.equal(a, b) for a, b in zip(res._pend_bufs, held))
        p2 = prompt + [ctx[40] ^ 1] + NEW
        lg2, did = _next(eng, res, p2)
        assert did["kind"] == "pend" and did["reused"] == 40, did
        assert torch.equal(lg2, _cold(p2)[1])
    return "the prompt end at 40, its bytes as they were; a copy parting there resumed at it, equal to cold"


def test_a_prompt_end_above_a_read_end_is_exact():
    """E7: turn 2 resumes at turn 1's end, turn 3 parts right after turn 2's prompt: it resumes at turn 2's prompt
    end and equals a cold read, and no session's taint was set on the way."""
    with _pend():
        res = _prefix()
        eng, ctx, e = _turn1(res, tokens(40, 10))
        p2 = ctx + NEW
        lg2, did2 = _next(eng, res, p2)
        assert did2["kind"] == "end" and res.taint is None, (did2, res.taint)
        ctx2 = p2 + _answer(eng, lg2, len(p2), chain=False)
        _remember(eng, res, ctx2, len(p2))
        p3 = p2 + [ctx2[len(p2)] ^ 1] + NEW3
        lg3, did3 = _next(eng, res, p3)
        assert did3["kind"] == "pend" and did3["reused"] == len(p2), did3
        ref, ref_lg = _cold(p3)
        assert torch.equal(lg3, ref_lg) and _same_state(eng, ref, len(p3)) == []
        assert res.taint is None, res.taint
    return f"turn 3 resumed at turn 2's prompt end {len(p2)}: equal to a cold read, no taint"


# ------------------------------------------------------------------ E8

def test_a_read_that_raises_keeps_nothing():
    """E8: the prompt pass raises inside the read: the rows stay valid to the prompt's end, no end is kept, and the
    next turn (resumed at the prompt end) equals a cold read."""
    with _pend():
        res = _prefix()
        eng = Marked()
        prompt = tokens(40, 11)
        lg, _, _ = cache.prefill(eng, None, prompt, "cpu", chunk=CH, resident=res, conv_id="A")
        ctx = prompt + _answer(eng, lg, 40)
        eng.fail_at = 44
        _remember(eng, res, ctx, 40)
        eng.fail_at = None
        assert res.end is None and res.valid == 40 and res.tokens == prompt, (res.end, res.valid)
        assert res.stats.get("end_reread_error") == 1, res.stats
        p2 = ctx + NEW
        lg2, did = _next(eng, res, p2)
        assert did["kind"] == "pend" and did["reused"] == 40, did
        assert torch.equal(lg2, _cold(p2)[1])
    return "the read raised at 44: valid 40, no end; the next turn resumed at the prompt end, equal to cold"


# ------------------------------------------------------------------ E9

def test_hidden_spans_are_read_and_replayed():
    """E9: tokens the session monitor hid in the answer are read with it (they are the rows' tokens); the client's
    copy without them is replayed to the rows' ids, which resume at the end and equal a cold read of those ids."""
    with _pend():
        res = _prefix()
        eng = Marked()
        prompt = tokens(40, 12)
        lg, _, _ = cache.prefill(eng, None, prompt, "cpu", chunk=CH, resident=res, conv_id="A")
        ctx = prompt + _answer(eng, lg, 40)
        e = min(int(eng.kv.length), len(ctx))
        span = (43, ctx[43:46])                     # three answer tokens the client never received
        _remember(eng, res, ctx, 40, hidden=[span])
        assert res.end is not None and res.end["hidden"] == [span], res.end
        copy = ctx[:43] + ctx[46:] + NEW
        served, spans = res.replay(copy)
        assert spans == [span] and served == ctx + NEW, (spans, served[:60])
        lg2, did = _next(eng, res, served)
        assert did["kind"] == "end" and did["reused"] == e, did
        ref, ref_lg = _cold(served)
        assert torch.equal(lg2, ref_lg) and _same_state(eng, ref, len(served)) == []
    return f"3 hidden tokens read and put back: the served ids are the rows', equal to a cold read of them"


# ------------------------------------------------------------------ E10

def _tier(d):
    from disk_tier_util import waiting
    return waiting(disk_cache.DiskTier(d, chunk=CH, fingerprint="toy", settings={"admit_after": 1, "budget_gb": 1},
                                       free_bytes=lambda p: 10 ** 15, head_anchors=2, min_tokens=CH).open())


def test_the_park_and_the_spill_carry_exact():
    """E10: the park takes the read end with its rows and the swap brings it back marked `exact`, offered and equal to
    a cold read; the spill writes `exact` in its manifest and the restore offers it, equal to a cold read. An end
    without the key (a park or an entry written before) is not offered on such a family."""
    got = []
    with _pend():
        # the park: session A's end goes to the park when session B replaces the rows, and comes back
        for drop_key in (False, True):
            res = _prefix(park_bytes=BIG)
            eng, ctx, e = _turn1(res, tokens(50, 13))
            b = tokens(40, 14)
            cache.prefill(eng, None, b, "cpu", chunk=CH, resident=res, conv_id="B")
            assert res.park_end is not None and res.park_end.get("exact") is True, res.park_end
            if drop_key:
                del res.park_end["exact"]
            p2 = ctx + NEW
            lg, did = _next(eng, res, p2)
            ref, ref_lg = _cold(p2)
            if drop_key:
                assert did["kind"] != "end" and did["reused"] < e, did
            else:
                assert did["kind"] == "end" and did["reused"] == e, did
            assert torch.equal(lg, ref_lg)
        got.append("park")
        # the spill: the conversation written with its read end and read back in place
        d = tempfile.mkdtemp(prefix="reread-spill-")
        try:
            t = _tier(d)
            sp = spill.SessionSpill(os.path.join(d, "sessions"), chunk=CH, fingerprint="toy", tier=t,
                                    settings={"budget_gb": 1.0, "min_tokens": 32, "drain_min_tokens": CH,
                                              "codec": "off"},
                                    free_bytes=lambda p: 10 ** 15, floor_bytes=0).open()
            res = _prefix()
            eng, ctx, e = _turn1(res, tokens(50, 15))
            info = sp.save_resident(res, eng, None, "evict")
            assert info is not None, sp.stats
            entry = sp.entries[info["eid"]]
            assert entry.man["end"].get("exact") is True, entry.man["end"]
            p2 = ctx + NEW
            assert entry.offer(p2, len(p2) - 1, exact_only=True) == e
            res.clear()
            for _, v, _ in eng.cache_state().views():
                v.zero_()
            assert sp.restore(res, eng, None, entry) is not None, sp.stats
            assert res.end is not None and res.end["at"] == e and res.end.get("exact") is True, res.end
            lg, did = _next(eng, res, p2)
            assert did["kind"] == "end" and did["reused"] == e, did
            assert torch.equal(lg, _cold(p2)[1])
            # an entry whose end has no key: its anchors are offered, its end is not, nor restored
            del entry.man["end"]["exact"]
            assert entry.offer(p2, len(p2) - 1, exact_only=True) < e
            res.clear()
            assert sp.restore(res, eng, None, entry) is not None, sp.stats
            assert res.end is None, res.end
            t.close()
            got.append("spill")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return " and ".join(got) + ": the read end comes back exact and equal to a cold read; one without the key is refused"


# ------------------------------------------------------------------ E11

def test_a_side_slots_carry_is_a_read_end_or_none():
    """E11: on a side slot without a resident prefix, the turn's decode-written end is carried nowhere (no carry
    record); on one with a prefix, the carry record is its read end, marked `exact`, and slot 0 adopts it as such."""
    from server import app, slots as slots_mod
    keep = dict(app.STATE)
    slot = slots_mod.Slot(1, 512, values={k: None for k in slots_mod.OVERLAY})
    try:
        with _pend():
            # no prefix on the slot: no carry
            eng = Marked()
            prompt = tokens(40, 16)
            lg, _, _ = cache.prefill(eng, None, prompt, "cpu", chunk=CH)
            ctx = prompt + _answer(eng, lg, 40)
            app.STATE.clear()
            app.STATE.update(engine=Marked(), drafter=None, resident=None, state_store=None, session_cache=False,
                             suffix_store=None, verbose=False, prefix_chunk=CH)
            slot.values.update(engine=eng, drafter=None, resident=None, last_ctx=list(ctx), last_prompt_n=40,
                               hidden_spans=[])
            slots_mod.bind(slot)
            slot.carry = {"stale": True}
            app._remember_turn(list(ctx[:40]), list(ctx[40:]), "A")
            assert slot.carry is None, slot.carry
            # a prefix on the slot: its read end is the carry record
            res = _prefix()
            eng2 = Marked()
            lg, _, _ = cache.prefill(eng2, None, prompt, "cpu", chunk=CH, resident=res, conv_id="A")
            ctx2 = prompt + _answer(eng2, lg, 40)
            slot.values.update(engine=eng2, drafter=None, resident=res, last_ctx=list(ctx2), last_prompt_n=40,
                               hidden_spans=[])
            app._remember_turn(list(ctx2[:40]), list(ctx2[40:]), "A")
            assert slot.carry is not None and slot.carry.get("exact") is True, slot.carry
            assert res.end is not None and res.end.get("exact") is True
            # slot 0 adopts a carried end as exact, and refuses one without the key
            r0 = _prefix()
            r0.adopt(ctx2, slot.carry["at"], 0, "A")
            assert r0.adopt_end(Marked(), {"at": slot.carry["at"], "hidden": [], "mark": None}) is None
            assert r0.adopt_end(Marked(), {"at": slot.carry["at"], "hidden": [], "mark": None, "exact": True})
            assert r0.end.get("exact") is True
    finally:
        slots_mod.bind(None)
        app.STATE.clear()
        app.STATE.update(keep)
    return "no prefix: no carry; with one: the carry is the read end (exact), adopted as such"


# ------------------------------------------------------------------ E12

def test_end_reread_off_resumes_at_the_prompt_end():
    """E12: QWEN38_END_REREAD=0: no read, no end and no session snapshot; the next turn resumes at the prompt end
    and equals a cold read (the fallback, the box's A/B arm)."""
    with _pend(), _reread_knob(False):
        store = cache.StateStore(BIG, chunk=CH)
        res = _prefix()
        eng, ctx, e = _turn1(res, tokens(40, 17), store=store, chain=False)
        assert res.end is None and res.valid == 40, (res.end, res.valid)
        assert not any(x.kind == "session" for x in store._d.values())
        p2 = ctx + NEW
        lg, did = _next(eng, res, p2, store=store)
        assert did["kind"] == "pend" and did["reused"] == 40, did
        assert torch.equal(lg, _cold(p2)[1])
    return f"off: no end; the next turn resumed at the prompt end (40) and read {len(p2) - 40} rows, equal to cold"


# ------------------------------------------------------------------ E13, E14: through the handler

def _hy4_handler(stream: bool):
    """One request through the handler on the tiny Hy4-flash with its capabilities' prefix; the bytes the client had
    when the end read started, and whether the request still held its lane then."""
    from engine.penalty import PenaltySpec
    from engine.spec import Relax
    from server import app
    from test_app_loop import FakeTok, Req
    from test_hy4_serving import serving32
    saved = dict(app.STATE)
    srv, store = serving32(256), cache.StateStore(BIG, chunk=CH)
    res = cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, end=True, chat_first=True, guest_store=True)
    res.end_reread = bool(getattr(srv.caps, "reread_end", False))
    seen: dict = {}
    r = Req("/v1/completions", {"prompt": "the quick brown fox jumps over the lazy dog and runs far away",
                                "max_tokens": 9, "stream": stream, "temperature": 0})
    real = res.reread_end

    def spy(*a, **kw):
        seen["bytes"] = r.wfile.getvalue()
        seen["running"] = int(app.INFLIGHT.get("running", 0))
        got = real(*a, **kw)
        seen["how"] = got
        return got
    res.reread_end = spy
    app.STATE.clear()
    app.STATE.update(engine=srv, drafter=None, k=4, tree=False, sampled_tree="det", relax=Relax(1.0, 1),
                     verbose=False, state_store=store, prefix_cache=True, prefix_chunk=CH, session_cache=True,
                     tok=FakeTok(), device="cpu", max_len=srv.max_len, default_max_tokens=64, pen_spec=PenaltySpec(),
                     model="t", cfg_eos=None, think_budget=0, think_stall=False, reasoning_format="tags",
                     request_timeout=0.0, max_queue=8, queue_timeout=5.0, pattern_stop=None, response_cache=None,
                     suffix_store=None, resident=res)
    try:
        head, raw = r.response()
    finally:
        app.STATE.clear()
        app.STATE.update(saved)
    assert head.startswith("HTTP/1.1 200"), (head, raw[:300])
    return seen, res, r.wfile.getvalue()


def test_a_streamed_client_has_its_last_bytes_before_the_read():
    """E13: a streamed answer's finish chunk, its usage and [DONE] are written before the end read starts; the read
    runs while the request still holds its lane (it is counted running), and its end is kept exact."""
    seen, res, final = _hy4_handler(True)
    assert "bytes" in seen, "the end read did not run"
    before = seen["bytes"].decode()
    assert before.rstrip().endswith("data: [DONE]"), before[-300:]
    assert '"finish_reason": "' in before and '"usage": {' in before, before[-600:]
    assert seen["bytes"] == final, "bytes were written after the read started"
    assert seen["running"] >= 1, seen
    assert seen["how"] == "read" and res.end is not None and res.end.get("exact") is True, (seen, res.end)
    return f"{len(seen['bytes'])} bytes out, [DONE] last, before the read; the lane still the request's; end exact"


def test_a_json_client_has_its_body_before_the_read():
    """E14: a JSON response's whole body (its usage in it) is written before the end read starts."""
    seen, res, final = _hy4_handler(False)
    assert "bytes" in seen, "the end read did not run"
    head, _, body = seen["bytes"].partition(b"\r\n\r\n")
    n = int(next(ln.split(":", 1)[1] for ln in head.decode().split("\r\n") if ln.lower().startswith("content-length")))
    assert len(body) == n and "usage" in json.loads(body), (n, len(body))
    assert seen["bytes"] == final and seen["running"] >= 1, seen
    assert seen["how"] == "read" and res.end is not None and res.end.get("exact") is True, (seen, res.end)
    return f"the {n}-byte body out before the read; the lane still the request's; end exact"


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]:
        try:
            print(f"  {name:72s} ok   {fn() or ''}", flush=True)
            passed += 1
        except Exception as exc:                                   # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  {name:72s} FAIL {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            failed += 1
    print(f"{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
