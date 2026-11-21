"""A drafted block stops at the reasoning cap, for every family: drafted == plain under a cap.

THE MEASUREMENT (fncheck at the T5 head, both Hy4 packs, caches on and off identical, and failing
before T5 too): 10 of 20 (3.56) and 11 of 20 (3.08) cases differed, every fork in the reasoning at
the forced close: plain closed at the cap, drafted went on. Reasoning tokens [plain, drafted]
against the cap: [49, 54] cap 48, [151, 152] cap 150, [91, 93] and [91, 95] cap 90, [46, 47] cap
45; the plan's 1,024 cap [1026, 1030].

THE CAUSE: a verified block was cut only at its end token and the output budget (server/app.py
`_end_cut`), and ThinkBudget looked at the whole block after its commit, so every token of it past
the cap was committed before `_force_close` ran. The plain step decides after each token.

THE FIX (engine/spec.py `ThinkBudget.cut`, server/app.py `_think_cut`, `_draft_room`): a drafted
block keeps the tokens through the one at which a close is decided at a count (the budget, a
supervisor's cap in `watch`, a stall arm firing), as a shorter accept (the last kept token is the
next anchor), on the tree loop (both orders), the chain loop and the round runner
(tests/test_round_runner.py); a proposal is no longer than the reasoning left before the cap.

  1  greedy, the random 4-layer model: drafted == plain token for token, with the same context, KV
     length and reasoning count, for budgets that fall at every offset of a block, on the tree loop
     (the old order and LAUNCH_FIRST) and the chain loop      test_greedy_drafted_is_plain_under_a_budget
  2  a supervisor's cap (the routed cap, SP4's): SP1 is asked at the cap and closes there, as on
     the plain step                                           test_a_supervisors_cap_closes_where_plain_closes
  3  the cut without a close (a cap the supervisor declines): the text is plain's, every row once
                                                              test_a_cut_the_supervisor_declines_changes_nothing
  4  a stall arm that fires inside a block: SP1 asked at that token, the close where plain's is;
     and with SP1 passing, plain's text                       test_a_stall_arm_is_asked_where_plain_asks_it
  5  a seeded sampled request (draws by position): drafted == plain under a budget, chain and tree
                                                              test_a_seeded_sampled_request_is_plain_under_a_budget
  6  `cut` against `observe`: random streams and splits, a block that opens inside them, the
     literal close, budgets, caps and arms: the decision lands at the token-by-token decision with
     the same state                                           test_cut_is_observe_token_by_token
  7  the arms' dry run puts them back exactly                 test_a_dry_run_leaves_the_arms_as_they_were
  8  a proposal stops at the cap: at most one row a cut is left uncommitted, and `room` is never 0
                                                              test_a_proposal_stops_at_the_cap
  9  the served commit, folded into the next verify (QWEN38_COMMIT_IN_VERIFY): the shorter accept's
     cut back is the plain text too                           test_the_folded_commit_cuts_the_same

Not covered: the monitor's own counts that the engine is not handed (D2's backstop, a policy number):
a drafted block may pass it by up to a block less one token before SP1 asks.

CPU, the random model of tests/test_window_edge.py. Run: python tests/test_think_cut.py
"""

from __future__ import annotations

import os
import random
import sys
import types

for _k in ("NORM", "GDN", "HEAD", "ATTN", "GDNBLOCK", "GDNTREE", "GDNPRE"):
    os.environ.setdefault(f"QWEN38_FUSED_{_k}", "0")
os.environ.setdefault("QWEN38_TREE_CHAIN_DELEGATE", "0")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.spec import StallArms, ThinkBudget  # noqa: E402
from engine.tree import DraftTree  # noqa: E402
from server import app  # noqa: E402
from test_app_loop import CLOSE, CLOSING, OPEN, FakeTok, serve  # noqa: E402

PROMPT = [5, 6, 7, 8, 9, 10, 11, OPEN]
MAX_NEW = 48
#: (name, tree loop, LAUNCH_FIRST)
PATHS = (("tree", True, False), ("tree, launch first", True, True), ("chain", False, False))


class Oracle:
    """Proposes what the target writes: the first of `seqs` (whole contexts) the context is a prefix
    of, to a depth that moves with the position (1 to 6), never more than the count it is handed.
    With the run without a cap first, a block goes on past the cap the way the model would."""

    def __init__(self, seqs, counts=None):
        self.seqs, self.counts = [list(s) for s in seqs], counts

    def reset(self):
        pass

    def observe(self, tokens):
        pass

    def _toks(self, ctx, count):
        if self.counts is not None:
            self.counts.append((len(ctx), count))
        n = len(ctx)
        for s in self.seqs:
            if s[:n] == list(ctx):
                return s[n:n + min(count, 1 + (n * 5) % 6)]
        return []

    def propose(self, ctx, count):
        return self._toks(ctx, count)

    def propose_tree(self, ctx, count):
        toks = self._toks(ctx, count)
        return DraftTree.chain(ctx[-1], toks) if toks else None


class Watcher:
    """A supervisor that watches every block (SP1): it records the reasoning count each SP1 asks at,
    and closes the block (hidden, CLOSING) the first time the count reaches `close_at`, or on the
    first stall when `on_stall`; else it passes."""

    def __init__(self, close_at=None, on_stall=False):
        self.close_at, self.on_stall = close_at, on_stall
        self.watches_blocks, self.checked, self.gave_up, self.cuts = True, True, None, []
        self.req = types.SimpleNamespace(tools_offered=False, n=1, request_id="r", max_tokens=0)
        self.memory_notes, self.asked, self.closed = None, [], None

    def at_block(self, ev, room):
        self.asked.append(ev.reasoning_tokens)
        due = ((self.close_at is not None and ev.reasoning_tokens >= self.close_at)
               or (self.on_stall and ev.stall))
        if due and self.closed is None:
            self.closed = ev.reasoning_tokens
            return types.SimpleNamespace(kind="CLOSE_REASONING", ids=list(CLOSING),
                                         visible=[CLOSE], detail={}, reason="t",
                                         hidden=len(CLOSING) - 1)
        return None

    def at_finish(self, ev, room):
        return None

    def check_answer(self, items, final=False):
        return None

    def done(self, outcome, detail):
        return None


def sup_of(**kw):
    return app.ext_mod.supervising(Watcher(**kw))


def _gen(drafter, think, *, tree=False, launch_first=False, sup=None, sampler=None,
         prompt=PROMPT, max_new=MAX_NEW):
    """One request on the served loop: (tokens, published context, KV length)."""
    eng = serve(drafter, tree=tree, k=7)
    app.STATE["sampled_tree"] = sampler is not None and tree
    saved = app.LAUNCH_FIRST
    app.LAUNCH_FIRST = launch_first
    try:
        out = list(app.generate_stream(torch.tensor(prompt), max_new, set(), think, sup=sup,
                                       sampler=sampler))
    finally:
        app.LAUNCH_FIRST = saved
    return out, list(app.STATE["last_ctx"]), int(eng.kv.length)


def plain(think=None, **kw):
    return _gen(None, think, **kw)


def drafted(path, seqs, think, counts=None, **kw):
    _, tree, lf = path
    return _gen(Oracle(seqs, counts), think, tree=tree, launch_first=lf, **kw)


def same(got, want, *what):
    """The drafted run wrote the plain run's text and context. The rows: the random model's every later
    token reads all of them, and the KV is never ahead of the context nor more than its last token
    behind (a generation that ends inside a block keeps its last row, SRV-530; a plain step's last
    token is never forwarded, so the two KV lengths may differ by that one row)."""
    (out, ctx, kv), (ref, rctx, _) = got, want
    assert out == ref, (*what, "the text", ref, out)
    assert ctx == rctx, (*what, "the context")
    assert len(ctx) - 1 <= kv <= len(ctx), (*what, "the KV", kv, len(ctx))


def _cuts():
    return dict(getattr(app, "THINK_CUTS", {"blocks": 0, "rows": 0}))


# ------------------------------------------------------------------ 1. greedy, a budget
def test_greedy_drafted_is_plain_under_a_budget():
    free = plain()[0]
    assert CLOSE not in free[:20], "the random model's own text does not close the block early"
    c0, n = _cuts(), 0
    for budget in range(2, 16):
        th0 = ThinkBudget(FakeTok(), budget)
        ref, rctx, rkv = plain(th0)
        assert CLOSE in ref and th0.done, (budget, ref)
        for path in PATHS:
            th = ThinkBudget(FakeTok(), budget)
            same(drafted(path, [PROMPT + free, PROMPT + ref], th), (ref, rctx, rkv), path[0], budget)
            assert th.done and th.n == th0.n, (path[0], budget, th.n, th0.n)
            n += 1
    c1 = _cuts()
    assert c1["blocks"] > c0["blocks"], "a block crossed the cap and was cut"
    return (f"{n} runs (budgets 2-15 x tree, launch first, chain): the plain text, context and KV; "
            f"{c1['blocks'] - c0['blocks']} blocks cut at the cap")


# ------------------------------------------------------------------ 2, 3. a supervisor's cap
def test_a_supervisors_cap_closes_where_plain_closes():
    free = plain()[0]
    n = 0
    for cap in range(3, 15):
        w_ref = sup_of(close_at=cap)
        ref, rctx, rkv = plain(ThinkBudget(FakeTok(), 0, watch=(cap,)), sup=w_ref)
        assert w_ref._obj.closed == cap, (cap, w_ref._obj.closed)
        for path in PATHS:
            w = sup_of(close_at=cap)
            got = drafted(path, [PROMPT + free, rctx], ThinkBudget(FakeTok(), 0, watch=(cap,)), sup=w)
            assert w._obj.closed == cap, (path[0], cap, "SP1 asked at", w._obj.asked)
            same(got, (ref, rctx, rkv), path[0], cap)
            n += 1
    return f"{n} runs: SP1 asked at the cap and closed there, the plain text, context and KV"


def test_a_cut_the_supervisor_declines_changes_nothing():
    free, fctx, fkv = plain()
    c0, n = _cuts(), 0
    for cap in (3, 6, 9, 13):
        for path in PATHS:
            w = sup_of()                            # watches every block, never closes
            got = drafted(path, [PROMPT + free], ThinkBudget(FakeTok(), 0, watch=(cap,)), sup=w)
            assert cap in w._obj.asked, (path[0], cap, w._obj.asked)
            same(got, (free, fctx, fkv), path[0], cap)
            n += 1
    assert _cuts()["blocks"] > c0["blocks"]
    return f"{n} runs cut at a cap nobody closed at: the uncapped text, context and KV, each row once"


# ------------------------------------------------------------------ 4. a stall arm
def _novelty(at: int) -> dict:
    """An arm that fires at the first look from reasoning token `at` (novelty is never below 1.01)."""
    return {"novelty": {"window": 4, "gram": 2, "min_tokens": at, "below": 1.01}}


def test_a_stall_arm_is_asked_where_plain_asks_it():
    free, fctx, fkv = plain()
    n = 0
    for at in range(8, 17):
        w_ref = sup_of(on_stall=True)
        t_ref = ThinkBudget(FakeTok(), 0, arms=_novelty(at))
        ref, rctx, rkv = plain(t_ref, sup=w_ref)
        assert t_ref.reason and w_ref._obj.closed == at, (at, t_ref.reason, w_ref._obj.closed)
        for path in PATHS:
            w, th = sup_of(on_stall=True), ThinkBudget(FakeTok(), 0, arms=_novelty(at))
            got = drafted(path, [PROMPT + free, rctx], th, sup=w)
            assert (th.reason, w._obj.closed) == (t_ref.reason, at), (path[0], at, th.reason,
                                                                      w._obj.asked)
            same(got, (ref, rctx, rkv), path[0], at)
            # SP1 passing: the arm's cut alone, the uncapped text
            w2, th2 = sup_of(), ThinkBudget(FakeTok(), 0, arms=_novelty(at))
            got2 = drafted(path, [PROMPT + free], th2, sup=w2)
            assert th2.reason == t_ref.reason, (path[0], at, th2.reason)
            same(got2, (free, fctx, fkv), path[0], at, "passing")
            n += 2
    return f"{n} runs: the arm fired at the plain step's token, SP1 asked there; passing, the uncapped text"


# ------------------------------------------------------------------ 5. sampled
def test_a_seeded_sampled_request_is_plain_under_a_budget():
    def smp():
        return app.Sampler(temperature=0.9, top_p=0.95, seed=11)
    free = plain(sampler=smp())[0]
    n = 0
    for budget in (3, 5, 8, 11, 14):
        ref, rctx, rkv = plain(ThinkBudget(FakeTok(), budget), sampler=smp())
        for path in (PATHS[0], PATHS[2]):        # the sampled tree (`--sampled-tree`) and chain
            same(drafted(path, [PROMPT + free, PROMPT + ref], ThinkBudget(FakeTok(), budget),
                         sampler=smp()), (ref, rctx, rkv), path[0], budget)
            n += 1
    return f"{n} seeded sampled runs (draws by position): the plain text under each budget"


# ------------------------------------------------------------------ 6, 7. cut, observe, arms
ARMS = {"loop": {"min_size": 1, "max_size": 4, "min_count": 3, "min_span": 6},
        "repeat": {"gram": 2, "window": 8, "min_tokens": 10, "every": 2, "act": 0.5, "in_a_row": 2},
        "novelty": {"window": 6, "gram": 2, "min_tokens": 12, "below": 0.3}}
VOCAB = [1, 2, 3, 4, OPEN, CLOSE, 60, 61, 62, 63, 80]     # 60 61: "<think>", 62 63: "</think>"


def _state(t: ThinkBudget) -> tuple:
    return (t.n, t.inside, t.done, t.reason, t.share)


def test_cut_is_observe_token_by_token():
    rng = random.Random(7)
    tok = FakeTok()
    seen_kinds = {"budget": 0, "watch": 0, "arm": 0, "opened": 0, "none": 0}
    for case in range(600):
        stream = []
        while len(stream) < rng.randint(4, 70):
            if rng.random() < 0.5:
                stream += [rng.choice(VOCAB) for _ in range(rng.randint(1, 8))]
            else:                                  # a repeated run (the arms' loop and novelty)
                pat = [rng.choice([1, 2, 3, 4]) for _ in range(rng.randint(1, 3))]
                stream += pat * rng.randint(2, 6)
        budget = rng.choice([0, rng.randint(1, 40)])
        watch = tuple(rng.sample(range(1, 45), rng.randint(0, 2)))
        arms = ARMS if rng.random() < 0.5 else None
        prompt = [5, OPEN] if rng.random() < 0.6 else [5, 6]
        caps = ([budget] if budget else []) + list(watch)
        # the token-by-token reference: the first token after which a close is decided at a count
        a = ThinkBudget(tok, budget, arms=arms, watch=watch).start(prompt)
        states, decide = [], None
        for i, t in enumerate(stream):
            n0, r0, live = a.n, a.reason, a.inside and not a.done
            a.observe([t])
            states.append(_state(a))
            if decide is None and live and a.inside and not a.done and (
                    any(n0 < c <= a.n for c in caps) or (r0 is None and a.reason is not None)):
                decide = i
        # block by block, each block cut where `cut` says
        b = ThinkBudget(tok, budget, arms=arms, watch=watch).start(prompt)
        i, got = 0, None
        while i < len(stream):
            if b.inside and not b.done:
                r = b.room
                ahead = [c for c in caps if c > b.n]
                assert r == (min(ahead) - b.n if ahead else None) and (r is None or r > 0), (case, r)
            blk = stream[i:i + rng.randint(1, 8)]
            before = _state(b)
            c = b.cut(blk)
            assert _state(b) == before, (case, "cut observed nothing")
            if c is not None:
                b.observe(blk[:c])
                got = i + c - 1
                break
            b.observe(blk)
            i += len(blk)
        assert got == decide, (case, stream, budget, watch, bool(arms), got, decide)
        if decide is not None:
            assert _state(b) == states[decide], (case, _state(b), states[decide])
            prev = states[decide - 1][3] if decide > 0 else None
            kind = ("arm" if b.reason is not None and prev is None
                    else "budget" if budget and b.n >= budget else "watch")
            seen_kinds[kind] += 1
            if prompt[-1] != OPEN:
                seen_kinds["opened"] += 1
        else:
            assert _state(b) == states[-1], (case, _state(b), states[-1])
            seen_kinds["none"] += 1
    assert all(v > 10 for v in seen_kinds.values()), seen_kinds
    return f"600 random streams and splits: the decision at the token-by-token token ({seen_kinds})"


def test_a_dry_run_leaves_the_arms_as_they_were():
    rng = random.Random(3)
    fields = ("share", "_tail", "_runs", "_grams", "_last", "_seen", "_streak", "_recent", "_next")
    fired = 0
    for case in range(60):
        a, b = StallArms(ARMS), StallArms(ARMS)
        for n in range(1, 160):
            t = rng.choice([1, 2, 3, 4, 5, 6] if n % 50 < 30 else [1, 2])
            snap = b.snapshot()
            for j in range(rng.randint(1, 6)):        # a dry run over a block that is not kept, to its
                if b.feed(rng.choice([1, 2, 3, 7, 8]), n + j) is not None:     # first firing (`cut`)
                    break
            b.restore(snap)
            got_a, got_b = a.feed(t, n), b.feed(t, n)
            assert got_a == got_b, (case, n, got_a, got_b)
            for f in fields:
                va, vb = getattr(a, f, None), getattr(b, f, None)
                assert va == vb, (case, n, f)
            if got_a is not None:                     # ThinkBudget feeds no further after a reason
                fired += 1
                break
    assert fired > 10, fired
    return f"60 twins of 160 tokens with a dry run before each: the same arms at every token ({fired} fired)"


# ------------------------------------------------------------------ 9. the folded commit
class _folded:
    """The served commit, folded into the next verify (tests/test_end_anchor.py `folded`)."""

    def __enter__(self):
        import engine.model as M
        self.M, self.old = M, (M.FUSED_COMMIT, M.COMMIT_IN_VERIFY)
        M.FUSED_COMMIT, M.COMMIT_IN_VERIFY = True, True

    def __exit__(self, *exc):
        self.M.FUSED_COMMIT, self.M.COMMIT_IN_VERIFY = self.old
        return False


def test_the_folded_commit_cuts_the_same():
    c0, n = _cuts(), 0
    with _folded():
        free = plain()[0]
        for budget in range(2, 16):
            tp = ThinkBudget(FakeTok(), budget)
            ref, rctx, rkv = plain(tp)
            for path in PATHS:
                th = ThinkBudget(FakeTok(), budget)
                same(drafted(path, [PROMPT + free, PROMPT + ref], th), (ref, rctx, rkv), "folded",
                     path[0], budget)
                assert th.n == tp.n
                n += 1
    assert _cuts()["blocks"] > c0["blocks"]
    return f"{n} runs with the commit folded into the verify: the plain text, context and KV"


# ------------------------------------------------------------------ 8. the proposal
def test_a_proposal_stops_at_the_cap():
    free = plain()[0]
    c0 = _cuts()
    over = []
    for budget in range(2, 16):
        ref = plain(ThinkBudget(FakeTok(), budget))[0]
        for path in PATHS:
            counts = []
            th = ThinkBudget(FakeTok(), budget)
            drafted(path, [PROMPT + free, PROMPT + ref], th, counts=counts)
            # the reasoning count at each proposal before the close: the tokens written less the
            # first (the prompt opened the block, so the loop counts from the second token)
            for at, count in counts:
                written = at - len(PROMPT)
                if written - 1 < budget:
                    left = budget - (written - 1)
                    if count > left:
                        over.append((path[0], budget, at, count, left))
    c1 = _cuts()
    blocks, rows = c1["blocks"] - c0["blocks"], c1["rows"] - c0["rows"]
    assert not over, over[:5]
    assert blocks > 0 and rows <= blocks, (blocks, rows)
    return f"no proposal past the cap; {blocks} cuts left {rows} rows uncommitted (one at most each)"


if __name__ == "__main__":
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            got = fn()
            print(f"  {name:64s} ok   {got or ''}")
            passed += 1
    print(f"{passed} passed")
