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
  E15  the read passes the memory gate a prefill passes (`Budget.gate`, the rows it forwards priced by
       `workspace_rows`, a refusal not counted): admitted, its workspace leaves the policy after it; refused, no end
       and no session snapshot                      test_the_read_passes_the_memory_gate_a_prefill_passes
  E16  on a real socket, a streamed client that reads to the end of the connection has it ([DONE], then the end)
       while the read still runs                    test_a_client_reading_to_the_connections_end_does_not_wait_for_the_read
  E17  a follow-up sent while the read runs waits for the lane, then resumes at the end the read kept
                                                     test_a_follow_up_during_the_read_waits_and_resumes_at_its_end
  E18  a prompt with a head mark: the read cuts and takes its anchors on the prompt's own grid (the mark, then every
       chunk from it), and the next prompt with that head resumes at one of them, equal to a cold read
                                                     test_the_read_takes_the_prompts_grid

  E19  thinking on, no tools: the template drops the answer's reasoning, the next prompt parts at the prompt end, and
       the close reads 0 rows; no end, the prompt end exact, the next turn equal to cold
                                                     test_a_turn_whose_reasoning_the_template_drops_reads_nothing
  E20  thinking off: the next prompt holds the whole answer, read and kept as the end
                                                     test_a_thinking_off_turn_reads_its_answer
  E21  a tool chain: what the template keeps of a call's reasoning, as the client sends it back or not, is read
                                                     test_a_tool_chain_reads_what_the_template_keeps
  E22  one line a close: rows read again, milliseconds, the rows the next prompt holds
                                                     test_the_close_says_its_cost_on_one_line
  E23  a read to where the next prompt parts inside the answer keeps the end there, a cold read's
                                                     test_a_read_to_where_the_next_prompt_parts_keeps_the_end_there
  E24  the end alignment, the park's and the disk's aligned offers are not asked on an end read's end
                                                     test_an_end_reads_end_is_never_aligned
  E25  with HY4_TOKENIZER, the pack's own template: thinking on, off, a call with and without reasoning sent back, a
       call under `--reasoning-format tags`
                                                     test_the_real_template_decides_the_rows
  E26  a stream's keep-alive client: `Connection: close`, its follow-up on a new connection, served after the read
                                                     test_a_streams_keep_alive_client_reconnects_for_its_follow_up
  E27  a JSON keep-alive follow-up written on the same socket during the read waits and is served after it
                                                     test_a_json_keep_alive_follow_up_on_the_same_socket_waits_and_is_served
  E28  the server stopped (SIGTERM) during a read, in its own process: the read stops at its next chunk, the drain
       waits for the close, exit 0, no core               test_a_server_stopped_during_a_read_exits_cleanly

E1 and E6 pass before the fix and after; E2 to E18 fail before it. E19 to E25 fail on the tree before the rows the
next prompt holds were asked (f8f6a4c1); E25 says "skipped" without HY4_TOKENIZER. E13 and E14 run the handler on the
tiny Hy4-flash (tests/hy4_tiny.py), which sets the capability.

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


# ------------------------------------------------------------------ E15: the memory gate

def test_the_read_passes_the_memory_gate_a_prefill_passes():
    """E15: the read asks the policy as a prefill does (server/app.py `_reread_gate`: `Budget.gate` on the live reserve
    for the rows it forwards onto the end, `count` False). Admitted: the gate priced the read's own rows, and its
    workspace leaves the policy once it has run (`prefill_done`). Refused (the reserve at the peak): no read, no end, no
    session snapshot, nothing counted as a request's refusal, no workspace held."""
    from server import app
    from test_budget import FakeAlloc, budget
    got = []
    for label, alloc in (("admitted", 60.0), ("refused", 76.9)):
        with _pend():
            fa = FakeAlloc(alloc)
            mem = budget(fa)
            store = cache.StateStore(BIG, chunk=CH)
            res = _prefix()
            eng = Marked()
            prompt = tokens(40, 18)
            lg, _, _ = cache.prefill(eng, None, prompt, "cpu", chunk=CH, resident=res, conv_id="A", store=store,
                                     checkpoint=True)
            ctx = prompt + _answer(eng, lg, 40, chain=False)
            e = min(int(eng.kv.length), len(ctx))
            gates, refused = mem.stats["gates"], int(mem.stats.get("refused", 0))
            keep = dict(app.STATE)
            app.STATE.clear()
            app.STATE.update(engine=eng, drafter=None, resident=res, state_store=store, session_cache=True, mem=mem,
                             last_ctx=list(ctx), last_prompt_n=40, hidden_spans=[], suffix_store=None, verbose=False)
            try:
                app._remember_turn(list(ctx[:40]), list(ctx[40:]), "A")
            finally:
                app.STATE.clear()
                app.STATE.update(keep)
            g = mem.stats.get("last_gate") or {}
            assert mem.stats["gates"] == gates + 1 and g.get("n") == e and g.get("rows") == e - 40, (label, g, e)
            assert mem.ws_now == 0 and int(mem.stats.get("refused", 0)) == refused, (label, mem.ws_now, mem.stats)
            sessions = [x for x in store._d.values() if x.kind == "session"]
            if label == "admitted":
                assert res.end is not None and res.end.get("exact") is True, res.end
                assert len(sessions) == 1 and sessions[0].exact, [(x.kind, x.exact) for x in store._d.values()]
            else:
                assert res.end is None and not sessions and res.stats.get("end_reread_memory") == 1, (res.end, res.stats)
            got.append(f"{label} at {alloc:g} GiB of 77 (rows {g.get('rows')} onto {g.get('n')})")
    return "; ".join(got)


# ------------------------------------------------------------------ E18: the grid of a marked prompt

def test_the_read_takes_the_prompts_grid():
    """E18: a prompt cut at its head mark (SPD-249, `grid_mark` 21 on a 16-row grid: lines 16, 21, 37, 53, 69): the
    read cuts its rows and takes its anchors on that grid, at 53 and not at the 0-anchored 48 or 64, so a later prompt
    with that head that parts inside the answer resumes at the read's anchor and equals a cold read on that grid."""
    with _pend():
        res = _prefix()
        eng = Marked()
        prompt = tokens(40, 19)
        lg, _, _ = cache.prefill(eng, None, prompt, "cpu", chunk=CH, resident=res, conv_id="A", grid_mark=21)
        assert sorted(res.anchors) == [16, 21, 37] and res.pend.get("mark") == 21, (sorted(res.anchors), res.pend)
        ctx = prompt + _answer(eng, lg, 40, chain=False)
        pos = int(eng.kv.length)                   # the last decided token is not forwarded
        for t in tokens(23, 21):                   # a longer answer by decode steps: past the 53 and 69 lines
            eng.forward(torch.tensor([ctx[-1]]), start=pos)
            pos += 1
            ctx.append(t)                          # the next token decided (the double's text does not matter)
        assert eng.kv.length == len(ctx) - 1
        _remember(eng, res, ctx, 40)
        e = res.end["at"]
        assert e > 69 and {53, 69} <= set(res.anchors) and not {48, 64} & set(res.anchors), (e, sorted(res.anchors))
        p2 = ctx[:60] + [ctx[60] ^ 1] + NEW
        info: dict = {}
        lg2, reused, _ = cache.prefill(eng, None, p2, "cpu", chunk=CH, resident=res, conv_id="A", grid_mark=21,
                                       info=info)
        assert info["kind"] == "resident" and reused == 53, (info, reused)
        ref = Marked()
        ref_lg, _, _ = cache.prefill(ref, None, p2, "cpu", chunk=CH, grid_mark=21)
        assert torch.equal(lg2, ref_lg) and _same_state(eng, ref, len(p2)) == []
    return f"anchors {sorted(res.anchors)} after the read to {e}; a prompt parting at 60 resumed at 53, equal to cold"


# ------------------------------------------------------------------ E16, E17: a real socket

class _Served:
    """The handler on a real socket (`ThreadingHTTPServer`, as tests/test_app_loop.py runs it) over the tiny Hy4-flash
    with its capabilities' prefix, one lane, and a tokenizer whose text comes back as the same ids (so a client's copy
    of an answer is the rows' own tokens). `slow_s`: the end read starts that long after it is called (a long answer's
    read, stood in for)."""

    def __init__(self, slow_s: float = 1.0, serve: bool = True):
        import threading
        import time
        from http.server import ThreadingHTTPServer
        from engine.penalty import PenaltySpec
        from engine.spec import Relax
        from server import app
        from test_app_loop import FakeTok
        from test_hy4_serving import serving32

        class RoundTok(FakeTok):
            """Text -> ids and back: one character an id, ids 0..63 as the characters 28..91."""

            def __call__(self, text, add_special_tokens=False, return_tensors=None):
                if text in ("<think>", "</think>") or not all(28 <= ord(c) <= 91 for c in text):
                    return super().__call__(text, add_special_tokens, return_tensors)
                ids = [ord(c) - 28 for c in text]
                if return_tensors == "pt":
                    ids = torch.tensor([ids])
                return type("R", (), {"input_ids": ids})()

            def decode(self, seq, skip_special_tokens=True):
                return "".join(chr(int(t) + 28) for t in seq)
        self.app, self.time, self.seen = app, time, {}
        self.saved = dict(app.STATE)
        srv = serving32(256)
        self.res = cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, end=True, chat_first=True, guest_store=True)
        self.res.end_reread = True
        real = self.res.reread_end

        def spy(*a, **kw):
            t0 = time.monotonic()
            self.seen.setdefault("t0", t0)           # the first close's (a follow-up's close comes after it)
            hook = self.seen.pop("during", None)
            if hook is not None:
                hook()
            time.sleep(slow_s)
            got = real(*a, **kw)
            self.seen.setdefault("t1", time.monotonic())
            self.seen.setdefault("how", got)
            self.seen.setdefault("calls", []).append((t0, time.monotonic(), got))
            return got
        self.res.reread_end = spy
        app.STATE.clear()
        app.STATE.update(engine=srv, drafter=None, k=4, tree=False, sampled_tree="det", relax=Relax(1.0, 1),
                         verbose=False, state_store=cache.StateStore(BIG, chunk=CH), prefix_cache=True, prefix_chunk=CH,
                         session_cache=True, tok=RoundTok(), device="cpu", max_len=srv.max_len, default_max_tokens=64,
                         pen_spec=PenaltySpec(), model="t", cfg_eos=None, think_budget=0, think_stall=False,
                         reasoning_format="tags", request_timeout=0.0, max_queue=8, queue_timeout=30.0,
                         pattern_stop=None, response_cache=None, suffix_store=None, resident=self.res)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        self.port = self.httpd.server_address[1]
        if serve:
            threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def post(self, body: dict) -> tuple:
        """One request on its own connection, read to the end of the connection: (raw bytes, monotonic time of the
        end)."""
        import socket
        raw = json.dumps(body).encode()
        c = socket.create_connection(("127.0.0.1", self.port), timeout=60)
        c.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  + f"Content-Length: {len(raw)}\r\nConnection: close\r\n\r\n".encode() + raw)
        out = b""
        while True:
            got = c.recv(65536)
            if not got:
                break
            out += got
        t = self.time.monotonic()
        c.close()
        return out, t

    def close(self) -> None:
        """The server stopped as an in-process server stops: the turns' closes told to stop and waited for (a
        follow-up's own close may still read; the handler threads are daemons nothing joins), then the listener, then
        the state given back. Without the wait a test process could end under a read (on the box: a core)."""
        stop = getattr(self.app, "stop_closes", None)
        if stop is not None:
            assert stop(timeout=60, resume=True), "a turn's close still runs a minute after the stop"
        self.httpd.shutdown()
        self.httpd.server_close()
        self.app.STATE.clear()
        self.app.STATE.update(self.saved)


def _text_of_stream(raw: bytes) -> str:
    out = ""
    for ln in raw.decode().split("\r\n\r\n", 1)[1].splitlines():
        if ln.startswith("data: {"):
            for ch in json.loads(ln[6:]).get("choices") or []:
                out += ch.get("text") or ""
    return out


def test_a_client_reading_to_the_connections_end_does_not_wait_for_the_read():
    """E16 (ruling 3 on a real socket): a streamed response closes its connection after it (`Connection: close`), and
    a client that reads to the end of the connection (curl, hold 1's rows.py) gets that end once [DONE] is out: the
    handler shuts the write side before the end read (`_end_response`), so the client never waits for the read (made
    to take a second here), which then runs on the lane and keeps the end."""
    sv = _Served(slow_s=1.0)
    try:
        raw, t_end = sv.post({"prompt": "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", "max_tokens": 9, "stream": True,
                              "temperature": 0})
        for _ in range(100):
            if "t1" in sv.seen:
                break
            sv.time.sleep(0.05)
    finally:
        sv.close()
    assert raw.startswith(b"HTTP/1.1 200") and raw.rstrip().endswith(b"data: [DONE]"), raw[-200:]
    assert "t0" in sv.seen and "t1" in sv.seen and sv.seen["how"] == "read", sv.seen
    assert t_end < sv.seen["t1"] - 0.5, (t_end - sv.seen["t0"], sv.seen["t1"] - sv.seen["t0"])
    assert sv.res.end is not None and sv.res.end.get("exact") is True, sv.res.end
    return (f"the end of the connection {t_end - sv.seen['t0']:+.2f} s from the read's start, the read ended at "
            f"{sv.seen['t1'] - sv.seen['t0']:.2f} s; end exact")


def test_a_follow_up_during_the_read_waits_and_resumes_at_its_end():
    """E17: the follow-up a client sends the moment it has its answer arrives while the read runs: it waits for the lane
    (the engine lock, held through the close), is not served before the read ends, and then resumes at the end the read
    kept, forwarding only its new rows."""
    import threading
    sv = _Served(slow_s=1.0)
    first = {"prompt": "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", "max_tokens": 9, "stream": True, "temperature": 0}
    follow, waiting = {}, []

    def during():
        def go():
            follow["raw"], follow["t"] = sv.post(dict(first, prompt=follow["prompt"], max_tokens=4))
        threading.Thread(target=go, daemon=True).start()
        for _ in range(40):
            sv.time.sleep(0.025)
            if sv.app.INFLIGHT.get("waiting", 0) >= 1:
                waiting.append(True)
                break
    try:
        # the follow-up's prompt is the first prompt, its answer and a new line: known once the first answer is out,
        # so the hook takes it from the first request's context (the rows' tokens, which this tokenizer writes back)
        def set_prompt():
            follow["prompt"] = sv.app.STATE["tok"].decode(list(sv.app.STATE["last_ctx"])) + " AND THEN"
            during()
        sv.seen["during"] = set_prompt
        raw, t_end = sv.post(first)
        for _ in range(200):
            if "t" in follow:
                break
            sv.time.sleep(0.05)
        e = sv.res.end["at"] if sv.res.end is not None else None
        last = dict(sv.app.STATE.get("last_prefill") or {})
    finally:
        sv.close()
    assert raw.rstrip().endswith(b"data: [DONE]") and "t" in follow, (raw[-100:], follow.keys())
    assert waiting, "the follow-up never waited in the queue"
    assert follow["t"] > sv.seen["t1"], (follow["t"] - sv.seen["t0"], sv.seen["t1"] - sv.seen["t0"])
    assert last.get("kind") == "end" and last.get("reused") == e, (last, e)
    assert follow["raw"].rstrip().endswith(b"data: [DONE]"), follow["raw"][-200:]
    return (f"the follow-up waited ({sv.app.INFLIGHT.get('served')} served), then resumed at the end {e} and forwarded "
            f"{last.get('forwarded')} rows")



# ------------------------------------------------------------------ E19 to E25: the rows the next prompt holds

class TplTok:
    """A tokenizer with Hy4-flash's chat template rule (the pack's chat_template.jinja) over a small vocabulary: a
    character an id (32..127 as 4..99), each tag one id. An answer's reasoning is rendered when the effort is not
    no_think, the client sent it, and `preserved_thinking` (on when tools are passed) or the answer comes after the last
    user message; else the block is empty. Calls are `<tool_calls><tool_call>NAME<arg_key>K</arg_key><arg_value>V
    </arg_value></tool_call></tool_calls>`, the wrapper left out of the text as Hy4-flash's tokenizer leaves it."""

    TAGS = {"<think>": 100, "</think>": 101, "<|s|>": 102, "<|m|>": 103, "<|e|>": 104, "<tool_call>": 105,
            "</tool_call>": 106, "<arg_key>": 107, "</arg_key>": 108, "<arg_value>": 109, "</arg_value>": 110,
            "<tool_response>": 111, "</tool_response>": 112, "<tool_calls>": 113, "</tool_calls>": 114}
    SPECIAL = (102, 103, 104)
    eos_token_id = 104
    unk_token_id = -1

    def __init__(self):
        self.names = {v: k for k, v in self.TAGS.items()}

    def convert_tokens_to_ids(self, text):
        return self.TAGS.get(text, -1)

    def encode(self, text: str) -> list[int]:
        out, i = [], 0
        while i < len(text):
            for tag, t in self.TAGS.items():
                if text.startswith(tag, i):
                    out.append(t)
                    i += len(tag)
                    break
            else:
                out.append(ord(text[i]) - 28 if 32 <= ord(text[i]) < 128 else 4)
                i += 1
        return out

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self.encode(text)
        if return_tensors == "pt":
            ids = torch.tensor([ids])
        return types.SimpleNamespace(input_ids=ids)

    def decode(self, seq, skip_special_tokens=True):
        out = []
        for t in seq:
            t = int(t)
            if t in (113, 114) or (skip_special_tokens and t in self.SPECIAL):
                continue
            out.append(self.names[t] if t in self.names else chr(t + 28))
        return "".join(out)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, tools=None,
                            enable_thinking=True, reasoning_effort=None, preserved_thinking=None, return_tensors=None,
                            return_dict=False, **kw):
        effort = "no_think" if not enable_thinking else (reasoning_effort or "high")
        keep = bool(tools) if preserved_thinking is None else bool(preserved_thinking)
        last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
        text = "<|s|>system<|m|>" + ("TOOLS " if tools else "") + f"EFFORT:{effort}<|e|>"
        for i, m in enumerate(messages):
            if m["role"] == "assistant":
                r = m.get("reasoning") or m.get("reasoning_content") or ""
                body = ("<think>" + r + "</think>" if effort != "no_think" and r and (keep or i > last_user)
                        else "<think></think>") + (m.get("content") or "")
                if m.get("tool_calls"):
                    body += "<tool_calls>" + "".join(
                        "<tool_call>" + c["function"]["name"] + "".join(
                            f"<arg_key>{k}</arg_key><arg_value>{v if isinstance(v, str) else json.dumps(v)}</arg_value>"
                            for k, v in c["function"]["arguments"].items()) + "</tool_call>"
                        for c in m["tool_calls"]) + "</tool_calls>"
                text += "<|s|>assistant<|m|>" + body + "<|e|>"
            elif m["role"] == "tool":
                text += "<|s|>tool<|m|><tool_response>" + str(m.get("content") or "") + "</tool_response><|e|>"
            else:
                text += f"<|s|>{m['role']}<|m|>" + str(m.get("content") or "") + "<|e|>"
        if add_generation_prompt:
            text += "<|s|>assistant<|m|><think>" + ("</think>" if effort == "no_think" else "")
        if not tokenize:
            return text
        ids = self.encode(text)
        return {"input_ids": torch.tensor([ids])} if return_dict else ids


TOOLS = [{"type": "function", "function": {"name": "get_time", "description": "the time",
                                           "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}}}}]


class _chat:
    """The server's state for a chat on `Marked` with `TplTok` and Hy4-flash's serving spec (its tool form and
    efforts), the prefix `res`, the prompt end kept (SPD-215)."""

    def __init__(self, res, eng=None):
        from engine.models.hy4_flash.arch import ARCH
        from server import app
        self.app, self.res = app, res
        self.eng = eng if eng is not None else Marked()
        self.tok = TplTok()
        self.state = dict(engine=self.eng, drafter=None, resident=res, state_store=None, session_cache=False,
                          hidden_spans=[], suffix_store=None, verbose=False, tok=self.tok, spec=ARCH.spec,
                          cfg_eos=None, device="cpu", segments=None, reasoning_effort=None)

    def __enter__(self):
        self.keep = dict(self.app.STATE)
        self.app.STATE.clear()
        self.app.STATE.update(self.state)
        self.pend = _pend().__enter__()
        return self

    def __exit__(self, *exc):
        self.pend.__exit__()
        self.app.STATE.clear()
        self.app.STATE.update(self.keep)
        return False

    def prompt(self, body) -> list[int]:
        return self.app.build_prompt(body, self.app.thinking_mod.read(body, strict=False), count="ids")

    def turn(self, body, said: str) -> tuple[list[int], int, str]:
        """One turn: the prompt read through the caches, `said` written by decode steps (its last token decided and not
        forwarded, as the loop leaves it), the turn's close with this request. Returns (context, n, the close's line)."""
        import contextlib
        import io
        app = self.app
        prompt = self.prompt(body)
        cache.prefill(self.eng, None, list(prompt), "cpu", chunk=CH, resident=self.res, conv_id="A")
        gen = self.tok.encode(said) + [self.tok.eos_token_id]
        n = len(prompt)
        for i, t in enumerate(gen[:-1]):
            self.eng.forward(torch.tensor([t]), start=n + i)
        ctx = list(prompt) + gen
        app.STATE.update(last_ctx=list(ctx), last_prompt_n=n)
        turn = (body, app.thinking_mod.read(body, strict=False), 0)
        kw = {"turn": turn} if "turn" in app._remember_turn.__code__.co_varnames else {}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            app._remember_turn(list(prompt), gen, "A", **kw)
        lines = [ln for ln in buf.getvalue().splitlines() if "end re-read" in ln]
        return ctx, n, "\n".join(lines)

    def next_turn(self, ids):
        """The next prompt read through the caches, as a turn's prefill: (logits, where it resumed)."""
        return _next(self.eng, self.res, ids)


def _held(tok, msgs, ctx, n, **kw) -> int:
    """How many of the answer's tokens the template's rendering of `msgs` holds (the test's own count)."""
    cur = tok.apply_chat_template(msgs[:-2], add_generation_prompt=True, **kw)
    nxt = tok.apply_chat_template(msgs, add_generation_prompt=True, **kw)
    assert nxt[:len(cur)] == cur
    gen, rest, g = ctx[n:], nxt[len(cur):], 0
    while g < min(len(gen), len(rest)) and gen[g] == rest[g]:
        g += 1
    return g


def test_a_turn_whose_reasoning_the_template_drops_reads_nothing():
    """E19: thinking on (the default), no tools: the template renders this answer back with an empty block, whatever
    the client sends, so the next prompt parts from the rows at the prompt end. The close reads nothing (skip), keeps no
    end and no decode-written row; the prompt end stays exact, and the next turn resumes there, equal to a cold read."""
    res = _prefix()
    reason = "LET ME THINK ABOUT THE GREETING AND WHAT TO SAY BACK TO IT"
    body = {"messages": [{"role": "user", "content": "HELLO THERE"}]}
    with _chat(res) as ch:
        ctx, n, line = ch.turn(body, reason + "</think>HI.")
        rows = int((res.stats.get("end_reread_last") or {}).get("rows", -1))
        assert res.end is None and rows == 0, (res.end, res.stats.get("end_reread_last"))
        assert res.stats.get("end_reread_skip") == 1 and res.stats.get("end_rereads", 0) == 0, res.stats
        assert res.pend is not None and res.pend["at"] == n and res.pend.get("exact"), res.pend
        nxt = ch.prompt({"messages": body["messages"] + [
            {"role": "assistant", "content": "HI.", "reasoning_content": reason},
            {"role": "user", "content": "AND NOW"}]})
        assert ctx[n] != nxt[n] and nxt[:n] == ctx[:n], "the copy parts from the rows at the prompt end"
        lg, last = ch.next_turn(nxt)
    assert last["kind"] == "pend" and last["reused"] == n, last
    cold, ref_lg = _cold(nxt)
    assert torch.equal(lg, ref_lg) and _same_state(ch.eng, cold, len(nxt)) == []
    return f"{len(ctx) - n - 1} answer rows, {len(reason)} of them reasoning: 0 read; resumed at the prompt end {n}"


def test_a_thinking_off_turn_reads_its_answer():
    """E20: thinking off: the prompt closes the block and the template renders the answer back as it was written, so
    the next prompt holds every answer row: the close reads them all and keeps the end there, exact, and the next turn
    resumes at it, equal to a cold read."""
    res = _prefix()
    body = {"messages": [{"role": "user", "content": "HELLO THERE"}], "chat_template_kwargs": {"enable_thinking": False}}
    with _chat(res) as ch:
        ctx, n, line = ch.turn(body, "HI. WHAT CAN I DO FOR YOU TODAY?")
        e = len(ctx) - 1
        last = res.stats.get("end_reread_last") or {}
        assert res.end is not None and res.end["at"] == e and res.end.get("exact"), res.end
        assert last.get("answer_rows") == e - n and last.get("held") == e - n, last
        assert f"the next prompt holds {e - n} of the answer's {e - n} rows" in line, line
        nxt = ch.prompt({"messages": body["messages"] + [{"role": "assistant", "content": "HI. WHAT CAN I DO FOR YOU TODAY?"},
                                                         {"role": "user", "content": "AND NOW"}],
                         "chat_template_kwargs": {"enable_thinking": False}})
        assert nxt[:e] == ctx[:e]
        lg, got = ch.next_turn(nxt)
    assert got["kind"] == "end" and got["reused"] == e, got
    cold, ref_lg = _cold(nxt)
    assert torch.equal(lg, ref_lg) and _same_state(ch.eng, cold, len(nxt)) == []
    return f"{e - n} answer rows read from the {last.get('from')} at {last.get('at')}; the next turn resumed at the end {e}"


def test_a_tool_chain_reads_what_the_template_keeps():
    """E21: tools passed (`preserved_thinking` on) and an answer that calls one. The template keeps a chain's reasoning
    after the last user message when the client sends it back. (a) The chain's first call, no earlier answer to tell:
    the answer goes back as the server gave it, and the read covers what the template renders of it, the reasoning and
    the call. (b) The next call, the client's earlier answer without its reasoning: the template renders an empty
    block, the next prompt parts at the prompt end, and nothing is read. (c) The same with the earlier answer's
    reasoning sent back: read as (a). Each read is the count of the answer's tokens the template's own rendering of
    the next conversation holds, and the next turn equal to a cold read."""
    call = {"id": "c1", "type": "function", "function": {"name": "get_time", "arguments": {"tz": "UTC"}}}
    said = "I SHOULD ASK FOR THE TIME</think><tool_calls><tool_call>get_time<arg_key>tz</arg_key><arg_value>UTC" \
           "</arg_value></tool_call></tool_calls>"
    user = {"role": "user", "content": "WHAT TIME IS IT"}
    out = []
    for case, earlier in (("a", None), ("b", False), ("c", True)):
        res = _prefix()
        msgs = [user]
        if earlier is not None:
            first = {"role": "assistant", "content": "", "tool_calls": [call]}
            if earlier:
                first["reasoning_content"] = "I SHOULD ASK FOR THE TIME"
            msgs = [user, first, {"role": "tool", "tool_call_id": "c1", "content": "12:00"}]
        body = {"messages": msgs, "tools": TOOLS}
        with _chat(res) as ch:
            ctx, n, line = ch.turn(body, said)
            e = len(ctx) - 1
            back = {"role": "assistant", "content": "", "tool_calls": [dict(call, id="c2")]}
            if earlier is not False:
                back["reasoning_content"] = "I SHOULD ASK FOR THE TIME"
            nmsgs = msgs + [back, {"role": "tool", "tool_call_id": "c2", "content": "12:01"}]
            want = min(e, n + _held(ch.tok, nmsgs, ctx, n, tools=TOOLS)) - n
            last = res.stats.get("end_reread_last") or {}
            if case == "b":
                assert want == 0 and res.end is None and res.stats.get("end_reread_skip") == 1, (want, res.end, last)
            else:
                assert want == e - n and res.end is not None and res.end["at"] == n + want, (want, e - n, res.end)
                assert last.get("held") == want and res.end.get("exact"), last
            nxt = ch.prompt({"messages": nmsgs, "tools": TOOLS})
            lg, got = ch.next_turn(nxt)
        assert got["kind"] == ("pend" if case == "b" else "end") and got["reused"] == n + want, (case, got)
        cold, ref_lg = _cold(nxt)
        assert torch.equal(lg, ref_lg) and _same_state(ch.eng, cold, len(nxt)) == [], case
        out.append(f"({case}) {want} of {e - n} read")
    return "; ".join(out)


def test_the_close_says_its_cost_on_one_line():
    """E22: every turn's close is one log line: what it did, the rows read again and the read's own milliseconds, how
    many of the answer's rows the next prompt holds, why, and how long deciding that took, the end and the whole
    close's time."""
    import re
    lines = []
    for body, said in (({"messages": [{"role": "user", "content": "HI"}]}, "SOME REASONING</think>HELLO."),
                       ({"messages": [{"role": "user", "content": "HI"}],
                         "chat_template_kwargs": {"enable_thinking": False}}, "HELLO.")):
        res = _prefix()
        with _chat(res) as ch:
            ctx, n, line = ch.turn(body, said)
        lines.append(line)
    pat = (r"^\[cache\] \S+ end re-read: (\w+), (\d+) rows in ([0-9.]+) ms( from the \w+ at \d+)?; the next prompt holds "
           r"(\d+) of the answer's (\d+) rows \([^)]*; decided in ([0-9.]+) ms\); end \w+; close ([0-9.]+) ms$")
    got = [re.match(pat, ln) for ln in lines]
    assert all(ln.count("\n") == 0 for ln in lines) and all(got), lines
    assert got[0].group(1) == "skip" and got[0].group(2) == "0" and got[0].group(5) == "0", lines[0]
    assert got[1].group(1) == "read" and int(got[1].group(2)) > 0 and got[1].group(5) == got[1].group(6), lines[1]
    for m in got:     # the read's own ms and the decision's, each within the whole close
        assert float(m.group(3)) <= float(m.group(8)) and float(m.group(7)) <= float(m.group(8)), m.group(0)
    return " | ".join(lines)


def test_a_read_to_where_the_next_prompt_parts_keeps_the_end_there():
    """E23: a next prompt that parts from the answer's rows inside it (`upto`): the read stops there and the end is kept
    there, the engine cut back to it, a cold read's state; a copy that holds those rows resumes at that end, equal to a
    cold read. At or below the prompt's end nothing is read and nothing moves."""
    res = _prefix()
    prompt, said = tokens(40, seed=91), tokens(12, seed=92)
    n, eng = len(prompt), Marked()
    with _pend():
        cache.prefill(eng, None, list(prompt), "cpu", chunk=CH, resident=res, conv_id="A")
        for i, t in enumerate(said[:-1]):
            eng.forward(torch.tensor([t]), start=n + i)
        ctx, e = prompt + said, n + len(said) - 1
        assert res.reread_end(eng, None, ctx, n, device="cpu", upto=n) == "skip"
        assert res.valid == n and int(eng.kv.length) == e and res.stats.get("end_reread_skip") == 1
        c = n + 5
        assert res.reread_end(eng, None, ctx, n, device="cpu", upto=c) == "read"
        assert int(eng.kv.length) == c and res.keep_end(eng, None, ctx, n, reread=True) == "kept"
        assert res.end["at"] == c and res.end.get("exact") and res.taint is None, res.end
        cold, _ = _cold(ctx[:c])
        assert _same_state(eng, cold, c) == []
        nxt = list(ctx[:c]) + NEW
        lg, got = _next(eng, res, nxt)
    assert got["kind"] == "end" and got["reused"] == c and got["forwarded"] == len(NEW), got
    cold, ref_lg = _cold(nxt)
    assert torch.equal(lg, ref_lg) and _same_state(eng, cold, len(nxt)) == []
    return f"read to {c} of {e}: the end at {c}, the next turn resumed there, equal to cold"


def test_an_end_reads_end_is_never_aligned():
    """E24: the end alignment (server/endalign.py) serves a copy that differs from the rows (whitespace, a dropped
    reasoning, a rewritten call) as the rows; that is not a cold read of the copy. On a prefix that reads its end again
    it is not asked: the copy is served as sent. The park's and the disk's offers likewise."""
    from server import endalign
    NL = 95

    def tb(i):
        return b"\n" if int(i) == NL else b"<%d>" % int(i)
    out = {}
    for flag in (False, True):
        res = _prefix(reread=flag)
        prompt, ans = tokens(30, seed=5), tokens(20, seed=6)
        res.tokens, res.valid = prompt + ans, 50
        res.end = {"at": 50, "hidden": [], "mark": None}
        copy = prompt + ans[:8] + [NL] + ans[8:] + [41, 43]
        al = endalign.EndAlign(tb, on=True)
        al.kept(res, 30, "kept")
        ids, spans, aligned = al.replay(res, copy)
        out[flag] = (ids == copy, aligned)
        res.park_end, res.park_tokens, res.park_regions = dict(res.end, exact=True), list(res.tokens), []
        res.park_align = lambda toks, end, regions, keys: True
        out[flag] += (res._continues_park(copy),)
    assert out[False] == (False, 20, True), out
    assert out[True] == (True, 0, False), out
    ent = spill.Entry.__new__(spill.Entry)
    ent.tokens, ent.man = list(range(50)), {"end": {"at": 50, "hidden": [], "exact": True}, "anchors": [32]}
    probe = list(range(40)) + [7, 8]
    kw = dict(replay=True, max_len=99, glue=frozenset(), align=lambda *a: True)
    assert ent.offer(probe, **kw) == 50 and ent.offer(probe, exact_only=True, **kw) == 32
    return f"without the flag aligned {out[False][1]} tokens and the park continued; with it served as sent"



def _keepalive(json_first: bool) -> dict:
    """A keep-alive client (`http.client`, which keeps a connection open unless the response says `Connection:
    close`, and opens a new one when it does): turn 1, then turn 2 sent the moment turn 1 is read, while the end read
    runs (made to take a second). Returns what each saw."""
    import http.client
    sv = _Served(slow_s=1.0)
    got: dict = {}
    try:
        c = http.client.HTTPConnection("127.0.0.1", sv.port, timeout=60)
        first = {"prompt": "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", "max_tokens": 9, "temperature": 0,
                 "stream": not json_first}
        c.request("POST", "/v1/completions", json.dumps(first), {"Content-Type": "application/json",
                                                                  "Connection": "keep-alive"})
        r1 = c.getresponse()
        got["h1"], got["b1"], got["port1"] = dict(r1.getheaders()), r1.read(), c.sock.getsockname()[1] if c.sock else None
        got["t_read1"] = sv.time.monotonic()
        follow = dict(first, prompt=sv.app.STATE["tok"].decode(list(sv.app.STATE["last_ctx"])) + " AND THEN",
                      max_tokens=4, stream=False)
        got["t_send2"] = sv.time.monotonic()
        c.request("POST", "/v1/completions", json.dumps(follow), {"Content-Type": "application/json",
                                                                   "Connection": "keep-alive"})
        got["port2"] = c.sock.getsockname()[1]
        r2 = c.getresponse()
        got["status2"], got["b2"], got["t2"] = r2.status, r2.read(), sv.time.monotonic()
        got["e"] = sv.res.end["at"] if sv.res.end is not None else None
        got["last"] = dict(sv.app.STATE.get("last_prefill") or {})
        got["seen"] = dict(sv.seen)
        c.close()
    finally:
        sv.close()
    return got


def test_a_streams_keep_alive_client_reconnects_for_its_follow_up():
    """E26 (the half-close and keep-alive): a streamed response says `Connection: close` (both stream paths send it,
    as before this change), so the server never reads a second request on that socket; the half-close only moves the
    connection's end (FIN) from after the turn's close to before it. A keep-alive client that sends its follow-up the
    moment it has [DONE] opens a new connection for it; the follow-up waits for the lane while the read runs and is
    served after it, resuming at the end the read kept."""
    got = _keepalive(json_first=False)
    seen = got["seen"]
    assert got["h1"].get("Connection") == "close" and got["b1"].rstrip().endswith(b"data: [DONE]"), got["h1"]
    assert got["t_read1"] < seen["t1"] - 0.5 and got["t_send2"] < seen["t1"], (got["t_send2"] - seen["t0"],
                                                                             seen["t1"] - seen["t0"])
    assert got["port2"] != got["port1"], "a new connection for the follow-up"
    assert got["status2"] == 200 and got["t2"] > seen["t1"], (got["status2"], got["t2"] - seen["t1"])
    assert got["last"].get("kind") == "end" and got["last"].get("reused") == got["e"], (got["last"], got["e"])
    return (f"Connection: close; the follow-up sent {got['t_send2'] - seen['t0']:+.2f} s from the read's start on a new "
            f"connection, served {got['t2'] - seen['t1']:.2f} s after it ended, resumed at the end {got['e']}")


def test_a_json_keep_alive_follow_up_on_the_same_socket_waits_and_is_served():
    """E27: a JSON response keeps the connection (no `Connection: close`, its length in Content-Length). A follow-up the
    client writes on that same socket while the read runs is not dropped: the handler reads it once the turn's close is
    done, serves it, and it resumes at the end the read kept."""
    got = _keepalive(json_first=True)
    seen = got["seen"]
    assert "Connection" not in got["h1"] and json.loads(got["b1"])["choices"], got["h1"]
    assert got["t_send2"] < seen["t1"] - 0.5, (got["t_send2"] - seen["t0"], seen["t1"] - seen["t0"])
    assert got["port2"] == got["port1"], "the follow-up went on the same socket"
    assert got["status2"] == 200 and json.loads(got["b2"])["choices"] and got["t2"] > seen["t1"], got["status2"]
    assert got["last"].get("kind") == "end" and got["last"].get("reused") == got["e"], (got["last"], got["e"])
    return (f"the follow-up written {got['t_send2'] - seen['t0']:+.2f} s from the read's start on the same socket, "
            f"served {got['t2'] - seen['t1']:.2f} s after it ended, resumed at the end {got['e']}")


_STOP_RUN = r"""
import json, os, signal, sys, threading
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.dirname(sys.argv[1]))
import test_end_reread as T
from server import app
sv = T._Served(slow_s=0.3, serve=False)
# the server's own stop (SIGTERM, `serve_until_drained`) while the turn's end read is about to start
sv.seen["during"] = lambda: os.kill(os.getpid(), signal.SIGTERM)


def mark(line):
    # the markers on their own channel, a line one write: the server's threads print on stdout meanwhile
    fd = os.open(sys.argv[2], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (line + "\n").encode())
    finally:
        os.close(fd)


def client():
    try:
        raw, _ = sv.post({"prompt": "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", "max_tokens": 40, "stream": True,
                          "temperature": 0})
        mark(f"CLIENT {raw.rstrip().endswith(b'data: [DONE]')}")
    except Exception as exc:
        mark(f"CLIENT {type(exc).__name__} {exc}")


threading.Thread(target=client, daemon=True).start()
app.serve_until_drained(sv.httpd)
mark(f"CALLS {json.dumps([c[2] for c in sv.seen.get('calls') or []])} CLOSING {app._CLOSING['n']}")
"""

#: the close line of a read the stop ended (one print: its text is one write, whatever else prints beside it)
_STOPPED_CLOSE = (r"\[cache\] \S+ end re-read: stopped, 0 rows in [0-9.]+ ms; the next prompt holds \d+ of the answer's \d+ "
                  r"rows \([^)]*\); end reread; close [0-9.]+ ms")


def test_a_server_stopped_during_a_read_exits_cleanly():
    """E28: the server's own stop (SIGTERM, `serve_until_drained`) while a turn's end read runs, in a process of its
    own: the read stops at its next chunk and keeps nothing (`end re-read: stopped`), the drain waits for the close
    (`stop_closes`) before it says `[server] stopped.`, nothing is left running, and the process exits 0, with no
    "terminate called" and no core. Before the stop existed the read ran to its end under the drain. The child's
    markers go to a file of their own, a line one write; the server's lines are found by their text in its output, not
    by where a line starts (its threads print beside each other)."""
    import re
    import subprocess
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    fd, marks = tempfile.mkstemp(prefix="e28-", suffix=".marks")
    os.close(fd)
    try:
        p = subprocess.run([sys.executable, "-c", _STOP_RUN, HERE, marks], capture_output=True, text=True, timeout=300,
                           env=env, cwd=os.path.dirname(HERE))
        got = open(marks).read().splitlines()
    finally:
        os.unlink(marks)
    out = p.stdout + p.stderr
    assert p.returncode == 0, (p.returncode, out[-1500:])
    assert "terminate called" not in out and "core dumped" not in out, out[-1500:]
    m = re.search(_STOPPED_CLOSE, p.stdout)
    j = p.stdout.find("[server] stopped. ")
    assert m is not None and j > m.start(), out[-1500:]
    assert sorted(got) == sorted(["CLIENT True", 'CALLS ["error"] CLOSING 0']), (got, out[-1500:])
    return f"rc 0; {m.group(0)[len('[cache] '):][:120]}; then the drain's [server] stopped; markers {got}"


def test_the_real_template_decides_the_rows():
    """E25 (HY4_TOKENIZER: a directory with the pack's tokenizer.json, tokenizer_config.json and chat_template.jinja):
    `_next_rows` on Hy4-flash's own template and ids. Thinking on without tools: the next prompt parts at the prompt end
    (its answer's reasoning dropped). Thinking off: it holds the whole answer. A call with tools passed: the whole
    answer, reasoning and call, when the client sends reasoning back (a first turn, or an earlier answer that carries
    it); the prompt end when the client's earlier answer came back without it."""
    d = os.environ.get("HY4_TOKENIZER")
    if not d or not os.path.isfile(os.path.join(d, "tokenizer.json")):
        return "skipped: HY4_TOKENIZER names no tokenizer"
    from transformers import AutoTokenizer
    from engine.models.hy4_flash.arch import ARCH
    from engine.models.hy4_flash.tokenizer import Hy4Tokenizer
    from server import app
    tok = Hy4Tokenizer(AutoTokenizer.from_pretrained(d))
    S = ":opensource"
    end = tok._tok.convert_tokens_to_ids(f"<｜hy_end{S}｜>")
    call_text = (f"<tool_calls{S}><tool_call{S}>get_time<arg_key{S}>tz</arg_key{S}><arg_value{S}>UTC</arg_value{S}>"
                 f"</tool_call{S}></tool_calls{S}>")
    call = {"id": "c1", "type": "function", "function": {"name": "get_time", "arguments": {"tz": "UTC"}}}
    user = {"role": "user", "content": "What time is it?"}
    keep = dict(app.STATE)
    app.STATE.clear()
    app.STATE.update(tok=tok, spec=ARCH.spec, cfg_eos=None, device="cpu", segments=None, reasoning_effort=None)
    out = []
    try:
        cases = [("on", {"messages": [user]}, f"The user asks the time.</think{S}>I cannot know it.", 0),
                 ("off", {"messages": [user], "chat_template_kwargs": {"enable_thinking": False}},
                  "I cannot know it.", None),
                 ("call", {"messages": [user], "tools": TOOLS}, f"I should ask the tool.</think{S}>" + call_text, None),
                 ("call, no reasoning back", {"messages": [user, {"role": "assistant", "content": "", "tool_calls": [call]},
                                                          {"role": "tool", "tool_call_id": "c1", "content": "12:00"}],
                                              "tools": TOOLS}, f"Again.</think{S}>" + call_text, 0),
                 ("call, reasoning back", {"messages": [user, {"role": "assistant", "content": "", "tool_calls": [call],
                                                               "reasoning_content": "I should ask the tool."},
                                                       {"role": "tool", "tool_call_id": "c1", "content": "12:00"}],
                                           "tools": TOOLS}, f"Again.</think{S}>" + call_text, None),
                 # `--reasoning-format tags` (the holds' servers): the reasoning comes back inside the content, where
                 # the template writes it after an empty block, so the next prompt parts at the prompt end
                 ("call, tags", {"messages": [user], "tools": TOOLS}, f"I should ask the tool.</think{S}>" + call_text, 0)]
        for name, body, said, want in cases:
            client = app.thinking_mod.read(body, strict=False)
            prompt = app.build_prompt(body, client, count="ids")
            gen = list(tok._tok(said, add_special_tokens=False).input_ids) + [end]
            fmt = "tags" if name.endswith("tags") else "reasoning_content"
            got, why = app._next_rows((body, client, 0, fmt), prompt + gen, len(prompt))
            held = got - len(prompt)
            assert held == (len(gen) if want is None else want), (name, held, len(gen), why)
            out.append(f"{name}: {held} of {len(gen)}")
    finally:
        app.STATE.clear()
        app.STATE.update(keep)
    return "; ".join(out)


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
