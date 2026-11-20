"""The resident prefix and the state store, pinned byte for byte on the 27B and on Flash-Next.

The conversation cache of a family without side slots (SRV-905, SRV-907) changes the guest test, the
prefill's head mark and the store's guards in engine/cache.py and server/app.py, behind capabilities only
that family sets. This test is the proof that nothing moves for the families that do not set them.

WHAT IT PINS. One scripted sequence of requests through the server's own decode loop and turn end
(server/app.py `generate_stream`, `_remember`: the prefill through every cache tier in RAM, the greedy loop,
the session snapshot, the end anchor), on the 27B's served engine class on the random model and on
Flash-Next's serving adapter on its small host pack, without a drafter. After every request it records where
the prefill resumed and how much it forwarded, the tokens written, the resident prefix's report (the switch
log's clock fields left out), the store's report, sha-256 digests of the engine's state (its sections, and its
views up to `kv.length`), and every call `begin` made to a keeper or a spill. The sequence: a first
conversation read cold (the store keeps its lines); a chat of five turns, the second resumed at its end and
the third at its prompt end; a short guest; a longer request of another session that takes the lane (the park
holds the chat) and the chat back; a store hit of each kind, the prefix one by a guest; a prompt with a grid
mark twice; a worker's request (`stay`). A prefix shaped as a side slot's (every shorter request a guest, a
keeper, no park): a chat's guest that becomes its conversation (`promote_guest`, called as `_remember_turn`
does on a side slot), its next turn, a worker's request that calls the keeper. A prefix with a session spill
and no park: a conversation that another one sends to the spill.

WHAT IT DOES NOT PIN. The handler's routing (the queue, the slot choice, carries between slots) and a real
side slot; drafters; images; the conductor; the memory policy (none here); the disk tier; the spill's files
(a stand-in records `begin`'s calls to it and writes nothing); the keeper's NVMe write (a stand-in records the
call). The server's own question (`own`) did not exist on the tree this is recorded on: a separate test runs
the sequence with it and without it and asserts the records are equal (with the families' flags off, `own`
must change nothing).

The digests depend on the torch build and the CPU's GEMM kernels, so the fixture is keyed by host as in
tests/test_refactor_gate.py. It is recorded on the code before the change, then compared at every later
commit; a host without a recorded key prints SKIP and compares nothing:

    git checkout --detach a85182c7 && git checkout <head> -- tests/test_resident_pinned.py
    QSE_PIN_RECORD=1 python tests/test_resident_pinned.py     # the code before the change
    python tests/test_resident_pinned.py                      # every later commit: "N steps reproduce"

Re-recording is allowed only for a commit that is SUPPOSED to change the other families' caching, and its
message says so.

Run: python tests/test_resident_pinned.py
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)

import torch  # noqa: E402

from engine import cache  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "resident_pinned.json")
BIG = 1 << 30
#: the clock fields of the report: the switch log's `ts` and `ms`
CLOCK = ("last_switch", "switch_list")


def _toks(n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, 60, (n,), generator=g).tolist()


def _bytes(t) -> bytes:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy().tobytes()


def _digest(tensors) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(str(tuple(t.shape)).encode() + str(t.dtype).encode())
        h.update(_bytes(t))
    return h.hexdigest()


def _state(eng) -> str:
    n = int(eng.kv.length)
    return _digest(list(cache.state_now(eng)) + [t.narrow(ax, 0, n) for t, ax in cache.model_views(eng)])


def _json(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()


def _res_report(res) -> dict:
    return {k: v for k, v in res.report().items() if k not in CLOCK}


def _serve_state(eng, res, store, ch: int, max_len: int) -> None:
    """The server's state around `generate_stream` (tests/test_end_anchor.py `_state`), on this engine."""
    from engine.penalty import PenaltySpec
    from engine.spec import Relax
    from server import app
    from test_app_loop import FakeTok
    app.STATE.clear()
    app.STATE.update(engine=eng, drafter=None, k=4, tree=False, sampled_tree=False, relax=Relax(1.0, 1),
                     verbose=False, state_store=store, prefix_cache=store is not None, prefix_chunk=ch,
                     session_cache=store is not None, tok=FakeTok(), device="cpu", max_len=max_len,
                     default_max_tokens=32, pen_spec=PenaltySpec(), model="t", cfg_eos=None, think_budget=0,
                     think_stall=False, reasoning_format="tags", request_timeout=0.0, max_queue=8,
                     queue_timeout=5.0, pattern_stop=None, response_cache=None, suffix_store=None,
                     resident=res, disk_cache=None, mem=None)


class Play:
    """One engine, one resident prefix and one store; `step` runs a request as the server does and records it."""

    def __init__(self, eng, ch: int, res, store, max_len: int, own: bool = False):
        self.eng, self.ch, self.res, self.store, self.max_len = eng, ch, res, store, max_len
        self.own = own                     # pass `own` where the step says so (`test_own_changes_nothing`)
        self.calls: list = []              # the keeper's and the spill's calls from `begin`
        self.rows: list[dict] = []

    def run(self, ids, k: int, session, grid_mark=None, stay=False, own=False, remember=True) -> list[int]:
        from server import app
        _serve_state(self.eng, self.res, self.store, self.ch, self.max_len)
        kw = {}
        if grid_mark is not None:
            kw["grid_mark"] = grid_mark
        if stay:
            kw["stay"] = True
        if own and self.own:
            kw["own"] = True
        out = [int(t) for t in app.generate_stream(torch.tensor(ids), k, set(), None, session, **kw)]
        pre = dict(app.STATE["last_prefill"])
        if remember:
            app._remember(list(ids), out, session, **({"own": True} if own and self.own else {}))
        self.pre = {"kind": pre.get("kind"), "reused": int(pre.get("reused") or 0),
                    "forwarded": int(pre.get("forwarded") or 0)}
        return list(ids) + out

    def record(self, name: str, ctx=None, **more) -> None:
        rep = _res_report(self.res)
        row = {"step": name, **getattr(self, "pre", {}), "out": _json(ctx) if ctx is not None else None,
               "valid": rep["valid"], "anchors": rep["anchors"], "end_at": rep["end_at"], "taint": rep["taint"],
               "stash_pending": rep["stash_pending"], "owner": rep["owner"], "res": _json(rep),
               "store": _json(self.store.report()) if self.store is not None else None,
               "state": _state(self.eng), "calls": list(self.calls), **more}
        self.rows.append(row)

    def step(self, name: str, ids, k: int, session, **kw) -> list[int]:
        ctx = self.run(ids, k, session, **kw)
        self.record(name, ctx)
        return ctx


def play(make, ch: int, max_len: int, own: bool = False) -> list[dict]:
    """The sequence on engines from `make()` at chunk `ch` (their own prefill grid). `own`: the steps that are
    the server's own questions say so (a tree without `own` cannot)."""
    from server import app
    keep_pe, cache.PROMPT_END = cache.PROMPT_END, True
    keep_state = dict(app.STATE)
    try:
        # slot 0: a park, a stash, the end anchor, the store
        res = cache.ResidentPrefix(BIG, ch, stash_bytes=BIG, park_bytes=BIG, end=True)
        p = Play(make(), ch, res, cache.StateStore(BIG, chunk=ch), max_len, own)
        g0 = _toks(5 * ch + 3, 1)
        g0c = p.step("first conversation, cold", g0, 4, "g")
        c1 = p.step("chat turn 1 replaces it", _toks(6 * ch + 5, 2), 2 * ch, "A")
        c2 = p.step("chat turn 2 at its end", c1 + _toks(ch // 2 + 1, 3), 6, "A")
        c3 = p.step("chat turn 3 at its prompt end", c2[:-6] + _toks(4, 4) + _toks(5, 5), 4, "A")
        p.step("a short guest (the server's own question)", _toks(ch + 2, 6), 3, "q", own=True)
        c4 = p.step("chat turn 4 after the guest", c3 + _toks(3, 7), 3, "A")
        p.step("another session, too long for a guest", _toks(4 * ch + 1, 8), 3, "B", own=True)
        p.step("the chat back from the park", c4 + _toks(3, 9), 2, "A")
        p.step("a prefix hit by a guest", g0[:ch + 1] + _toks(3, 10), 2, "r")
        p.step("a session hit", g0c + _toks(3, 11), 2, "g")
        head = _toks(3 * ch + 5, 12)
        p.step("a prompt with a grid mark", head + _toks(4, 13), 3, "m", grid_mark=3 * ch + 5)
        p.step("the same head again", head + _toks(6, 14), 3, "n", grid_mark=3 * ch + 5)
        p.step("a worker's request (stay)", _toks(ch + 3, 19), 2, "w", stay=True)
        # a side slot's prefix (server/app.py `_side_resident`): every shorter request is a guest, a keeper writes
        # another session's rows to the NVMe before they go, a chat's guest becomes the slot's conversation
        sres = cache.ResidentPrefix(BIG, ch, stash_bytes=BIG, guest_frac=1.0, end=True)
        s = Play(make(), ch, sres, None, max_len, own)
        sres.keeper = lambda r, e, d, stay: s.calls.append(["keeper", int(r.valid), bool(stay)])
        x1 = s.step("side: a conversation", _toks(5 * ch + 2, 15), ch, "X")
        y1 = _toks(2 * ch + 3, 16)
        y1c = s.run(y1, 3 * ch, "Y", remember=False)
        committed = list(app.STATE["last_ctx"])
        promoted = sres.promote_guest(s.eng, None, committed, len(y1), "Y")
        kept = sres.keep_end(s.eng, None, committed, len(y1), [])
        s.record("side: a chat's guest promoted", y1c, promoted=promoted, keep_end=kept)
        s.step("side: the promoted chat's next turn", y1c + _toks(3, 17), 2, "Y")
        s.step("side: the old conversation's next turn", x1 + _toks(3, 18), 2, "X")
        s.step("side: a worker's request (stay), the keeper", _toks(ch + 1, 20), 2, "w", stay=True)
        # a session spill (a stand-in: `begin`'s calls only, nothing written) and no park
        pres = cache.ResidentPrefix(BIG, ch, stash_bytes=BIG, end=True)
        q = Play(make(), ch, pres, None, max_len, own)
        pres.spiller = types.SimpleNamespace(
            min_tokens=4 * ch, cfg={}, maybe_restore=lambda *a, **k: None,
            save_resident=lambda r, e, d, reason: q.calls.append(["spill", reason, int(r.valid)]) or {"eid": "pin"})
        q.step("spill: a long conversation", _toks(5 * ch, 21), 4, "S")
        q.step("spill: another one sends it to the spill", _toks(3 * ch, 22), 3, "T")
        q.step("spill: a short one is cut, not spilled", _toks(2 * ch, 23), 2, "U")
        return p.rows + s.rows + q.rows
    finally:
        cache.PROMPT_END = keep_pe
        app.STATE.clear()
        app.STATE.update(keep_state)


def host_key() -> str:
    return f"{platform.machine()}-{sys.platform}-torch{torch.__version__.split('+')[0]}"


def _load() -> dict:
    if os.path.isfile(FIXTURE):
        with open(FIXTURE) as f:
            return json.load(f)
    return {}


def _check(kind: str, run) -> str:
    got = run()
    again = run()
    assert got == again, (f"{kind}: two runs in one process differ, the pin cannot work: "
                          f"{[a['step'] for a, b in zip(got, again) if a != b]}")
    fx, key = _load(), host_key()
    if os.environ.get("QSE_PIN_RECORD") == "1":
        fx.setdefault(key, {})[kind] = got
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w") as f:
            json.dump(fx, f, indent=1, sort_keys=True)
            f.write("\n")
        return f"recorded {len(got)} steps for {key}"
    want = fx.get(key, {}).get(kind)
    if want is None:
        return f"SKIP compare: no steps recorded for {key} (known: {sorted(fx) or 'none'})"
    assert len(want) == len(got), (kind, len(want), len(got))
    bad = [(w["step"], sorted(f for f in w if w[f] != g.get(f))) for w, g in zip(want, got) if w != g]
    assert not bad, f"{kind}: the caching moved at {bad} (host {key})"
    return f"{len(want)} steps reproduce ({key}): " + ", ".join(f"{r.get('kind') or 'cold'} {r.get('reused')}"
                                                              for r in got)


def _27b():
    from test_cache import fresh
    return fresh()


def test_the_27bs_caching_is_pinned():
    """The 27B's served engine class (`Qwen38Engine`) on the random 4-layer model, a 16-row grid."""
    return _check("27b", lambda: play(_27b, 16, 512))


def test_flashnexts_caching_is_pinned(monkeypatch):
    """Flash-Next's serving adapter (`FlashNextServing`) on its small host pack, an 8-row grid (the stand-ins of
    tests/fn_host.py: deterministic functions of what each kernel reads)."""
    import test_fn_stage as S
    H, _ = S.setup(monkeypatch)
    from engine.models.flashnext.serving import FlashNextServing

    def one():
        eng, _ = H.whole(256)
        return FlashNextServing(eng, 256)
    return _check("flashnext", lambda: play(one, 8, 256))


def test_own_changes_nothing_without_the_flags():
    """The server's own question (`own` through `generate_stream` and `_remember`) on prefixes without the
    families' flags: the records equal the sequence's without it, step for step. Skipped on a tree without it."""
    if "own" not in inspect.signature(cache.prefill).parameters:
        return "SKIP: this tree has no `own`"
    plain, own = play(_27b, 16, 512), play(_27b, 16, 512, own=True)
    diff = [a["step"] for a, b in zip(plain, own) if a != b]
    assert len(plain) == len(own) and not diff, f"`own` moved the caching at {diff}"
    return f"{len(plain)} steps equal with and without `own`"


def test_the_pin_sees_a_moved_rule():
    """The pin is worth something only if a changed rule moves it: a guest test at half the rows in place."""
    base = play(_27b, 16, 512)
    real = cache.ResidentPrefix.__init__

    def half(self, *a, **kw):
        kw["guest_frac"] = 0.5 if kw.get("guest_frac", 0.25) == 0.25 else kw["guest_frac"]
        real(self, *a, **kw)
    cache.ResidentPrefix.__init__ = half
    try:
        moved = play(_27b, 16, 512)
    finally:
        cache.ResidentPrefix.__init__ = real
    diff = [a["step"] for a, b in zip(base, moved) if a != b]
    assert diff, "a guest rule at half the rows moved nothing"
    return f"{len(diff)} of {len(base)} steps moved: {diff[:3]}"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
