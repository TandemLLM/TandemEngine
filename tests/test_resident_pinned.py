"""The resident prefix and the state store, pinned byte for byte on the 27B and on Flash-Next.

The conversation cache of a family without side slots changes the guest test, the prefill's head mark and
the store's guards in engine/cache.py and server/app.py, behind capabilities only that family sets. This
test is the proof that nothing moves for the families that do not set them: one scripted sequence through
`cache.prefill` and the cache calls of the server's `_remember_turn` (the session snapshot, the end anchor,
a side slot's promotion), on the 27B's served engine class on the random model and on Flash-Next's serving
adapter on its small host pack, and after every step:

  -  where the prefill resumed (`info["kind"]`, rows reused and forwarded) and what `keep_end` said
  -  the resident prefix's report (the switch log's clock fields left out) and the store's report
  -  sha-256 digests of the engine's state (its sections, and its views up to `kv.length`) and the logits

The sequence: a first conversation read cold (the store keeps its lines); a chat of five turns, the second
resumed at its end and the third at its prompt end; a short request that is a guest, then the chat again;
a longer request of another session that replaces the chat (the park takes it), then the chat back; a
store hit of each kind, the prefix one by a guest; a prompt with a grid mark twice; a side slot's guest that
becomes its conversation (`promote_guest`), then its next turn.

The digests depend on the torch build and the CPU's GEMM kernels, so the fixture is keyed by host as in
tests/test_refactor_gate.py. It is recorded on the tree before the change and compared after it:

    QSE_PIN_RECORD=1 python tests/test_resident_pinned.py     # on the untouched tree
    python tests/test_resident_pinned.py                      # on every later commit

A host without a recorded key prints its digests' count and skips the comparison. Re-recording is
allowed only for a commit that is SUPPOSED to change the other families' caching, and its message says so.

Run: python tests/test_resident_pinned.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys

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


def _greedy(eng, lg, n0: int, k: int) -> list[int]:
    """`k` greedy tokens after a prompt of `n0` rows, as the served loop leaves them: every token but the last
    forwarded (`kv.length == n0 + k - 1`)."""
    out, pos = [], n0
    for i in range(k):
        t = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
        out.append(t)
        if i == k - 1:
            break
        lg = eng.forward(torch.tensor([t]), start=pos, last_only=True)
        pos += 1
    return out


def _remember(eng, res, store, committed: list[int], n_prompt: int, session) -> str:
    """The cache calls of server/app.py `_remember_turn` on a slot without a memory policy: the session
    snapshot into the store, a side slot's guest promoted, then the end anchor."""
    if store is not None and eng.kv.length and eng.kv.length <= len(committed):
        if store.reserve(cache._snapshot_estimate(eng, eng.kv.length, None)):
            snap = cache.capture(eng, None, max_bytes=store.max_entry)
            if snap is not None:
                store.put(committed, snap, session, kind="session")
    return res.keep_end(eng, None, committed, n_prompt, [])


class Play:
    """One engine, one resident prefix and one store; `step` runs a request and records it."""

    def __init__(self, eng, ch: int, res, store):
        self.eng, self.ch, self.res, self.store = eng, ch, res, store
        self.rows: list[dict] = []

    def step(self, name: str, ids: list[int], k: int, session, grid_mark=None, remember=True) -> list[int]:
        info: dict = {}
        lg, reused, fwd = cache.prefill(self.eng, None, list(ids), "cpu", store=self.store, chunk=self.ch,
                                        conv_id=session, checkpoint=self.store is not None, resident=self.res,
                                        info=info, grid_mark=grid_mark)
        row = {"step": name, "n": len(ids), "kind": info.get("kind"), "reused": int(reused),
               "forwarded": int(fwd), "logits": _digest([lg]), "after_prefill": _state(self.eng)}
        out = _greedy(self.eng, lg, len(ids), k)
        ctx = list(ids) + out
        row["out"] = _json(out)
        row["keep_end"] = _remember(self.eng, self.res, self.store, ctx, len(ids), session) if remember else None
        rep = _res_report(self.res)
        row.update(valid=rep["valid"], anchors=rep["anchors"], end_at=rep["end_at"], taint=rep["taint"],
                   stash_pending=rep["stash_pending"], owner=rep["owner"], res=_json(rep),
                   store=_json(self.store.report()) if self.store is not None else None, state=_state(self.eng))
        self.rows.append(row)
        return ctx


def play(eng, ch: int, side_eng=None) -> list[dict]:
    """The sequence on one engine at chunk `ch` (its own prefill grid); `side_eng` a second engine of the same
    weights for the side slot's part (a fresh one when None)."""
    keep, cache.PROMPT_END = cache.PROMPT_END, True
    try:
        res = cache.ResidentPrefix(BIG, ch, stash_bytes=BIG, park_bytes=BIG, end=True)
        store = cache.StateStore(BIG, chunk=ch)
        p = Play(eng, ch, res, store)
        g0 = _toks(5 * ch + 3, 1)
        g0c = p.step("first conversation, cold", g0, 4, "g")
        a1 = _toks(6 * ch + 5, 2)
        c1 = p.step("chat turn 1 replaces it", a1, 2 * ch, "A")
        c2 = p.step("chat turn 2 at its end", c1 + _toks(ch // 2 + 1, 3), 6, "A")
        c3 = p.step("chat turn 3 at its prompt end", c2[:-6] + _toks(4, 4) + _toks(5, 5), 4, "A")
        p.step("a short guest", _toks(ch + 2, 6), 3, "q")
        c4 = p.step("chat turn 4 after the guest", c3 + _toks(3, 7), 3, "A")
        p.step("another session, too long for a guest", _toks(4 * ch + 1, 8), 3, "B")
        p.step("the chat back from the park", c4 + _toks(3, 9), 2, "A")
        p.step("a prefix hit by a guest", g0[:ch + 1] + _toks(3, 10), 2, "r")
        p.step("a session hit", g0c + _toks(3, 11), 2, "g")
        head = _toks(3 * ch + 5, 12)
        p.step("a prompt with a grid mark", head + _toks(4, 13), 3, "m", grid_mark=3 * ch + 5)
        p.step("the same head again", head + _toks(6, 14), 3, "n", grid_mark=3 * ch + 5)
        # a side slot's prefix (server/app.py `_side_resident`): every short request is a guest, and a chat's
        # guest becomes the slot's conversation at its end (`promote_guest`)
        side = side_eng() if callable(side_eng) else side_eng
        sres = cache.ResidentPrefix(BIG, ch, stash_bytes=BIG, guest_frac=1.0, end=True)
        s = Play(side, ch, sres, None)
        x1 = s.step("side: a conversation", _toks(5 * ch + 2, 15), ch, "X")
        y1 = _toks(2 * ch + 3, 16)
        info: dict = {}
        lg, reused, fwd = cache.prefill(side, None, y1, "cpu", chunk=ch, conv_id="Y", resident=sres, info=info)
        out = _greedy(side, lg, len(y1), 3 * ch)
        y1c = y1 + out
        promoted = sres.promote_guest(side, None, y1c, len(y1), "Y")
        kept = sres.keep_end(side, None, y1c, len(y1), [])
        rep = _res_report(sres)
        s.rows.append({"step": "side: a chat's guest promoted", "kind": info.get("kind"), "reused": int(reused),
                       "forwarded": int(fwd), "promoted": promoted, "keep_end": kept, "valid": rep["valid"],
                       "anchors": rep["anchors"], "end_at": rep["end_at"], "res": _json(rep),
                       "state": _state(side)})
        s.step("side: the promoted chat's next turn", y1c + _toks(3, 17), 2, "Y")
        s.step("side: the old conversation's next turn", x1 + _toks(3, 18), 2, "X")
        return p.rows + s.rows
    finally:
        cache.PROMPT_END = keep


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
    return f"{len(want)} steps reproduce ({key}): " + ", ".join(f"{r['kind'] or 'cold'} {r['reused']}" for r in got)


def _27b():
    from test_cache import fresh
    return fresh()


def test_the_27bs_caching_is_pinned():
    """The 27B's served engine class (`Qwen38Engine`) on the random 4-layer model, a 16-row grid."""
    return _check("27b", lambda: play(_27b(), 16, side_eng=_27b))


def test_flashnexts_caching_is_pinned(monkeypatch):
    """Flash-Next's serving adapter (`FlashNextServing`) on its small host pack, an 8-row grid (the stand-ins of
    tests/fn_host.py: deterministic functions of what each kernel reads)."""
    import test_fn_stage as S
    H, _ = S.setup(monkeypatch)
    from engine.models.flashnext.serving import FlashNextServing

    def one():
        eng, _ = H.whole(256)
        return FlashNextServing(eng, 256)
    return _check("flashnext", lambda: play(one(), 8, side_eng=one))


def test_the_pin_sees_a_moved_rule():
    """The pin is worth something only if a changed rule moves it: a guest test at half the rows in place."""
    base = play(_27b(), 16, side_eng=_27b)
    real = cache.ResidentPrefix.__init__

    def half(self, *a, **kw):
        kw["guest_frac"] = 0.5 if kw.get("guest_frac", 0.25) == 0.25 else kw["guest_frac"]
        real(self, *a, **kw)
    cache.ResidentPrefix.__init__ = half
    try:
        moved = play(_27b(), 16, side_eng=_27b)
    finally:
        cache.ResidentPrefix.__init__ = real
    diff = [a["step"] for a, b in zip(base, moved) if a != b]
    assert diff, "a guest rule at half the rows moved nothing"
    return f"{len(diff)} of {len(base)} steps moved: {diff[:3]}"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
