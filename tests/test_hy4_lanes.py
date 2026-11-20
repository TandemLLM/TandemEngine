"""ENG-485: Hy4-flash on the engine's lanes, on a tiny random model, on the CPU.

Step 0, the byte gate (tests/test_refactor_gate.py's form): one fixed script through `Hy4FlashServing` -- a
120-row prompt in chunks of 16, six decode steps, a tree block whose moved path is committed, a chain block
rolled back, a chain block landed by `settle` -- and the sha256 of every logit and pick it gave and of the
state's own tensors after it: `lat_all` and `ik_all` up to the length, `win_all`, `hist`. Not of `views()`,
whose layout the lanes work changes. The digests are recorded once on the tree before the lanes work and
must reproduce after every step of it.

The digests depend on the torch build and the CPU's kernels, so the fixture keeps one table a host key; a
host without a recorded key prints its digests and skips the comparison. Record (only on the tree the gate
is meant to hold, the commit that added this file):

    QSE_PIN_RECORD=1 python tests/test_hy4_lanes.py

The lanes (ENG-485), as Flash-Next's: a side slot is a second `Hy4FlashServing` over the one engine with a state
of its own, paged through the KV pool when the server opened one; its lanes take turns. On the tiny model:

  * a side slot shares the engine (the weights, the Engram tables) and owns its state, rings, graph set and heads;
    a paged one takes no bytes a token                 test_a_side_slot_shares_the_engine_and_owns_its_state
  * two sequences switched after every chunk, block and commit (a chain block of one left pending while the other
    runs) give every logit, pick and state row their alone runs give
                                                       test_two_lanes_interleaved_equal_each_alone
  * a side slot's drafter (its own DFly head, slot 0's prices, freeze and corpus) writes the plain greedy text
    while slot 0 reads another prompt between its rounds
                                                       test_a_lanes_drafter_writes_the_plain_text
  * every pass commits its pages first, at least the indexer's read floor; a trim commits again
                                                       test_every_pass_commits_its_rows_first
  * the group-key mirror is rebuilt from the committed rows only, with the same bits
                                                       test_the_mirror_reads_committed_rows_only
  * the cache's views are the two whole tensors; the paged ones are not counted as memory; the build asks for the
    pool before the state; the slot's graphs are captured on its set and dropped at once
                                                       test_the_state_goes_through_the_pool_and_the_graph_set

Run: python tests/test_hy4_lanes.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)
import torch  # noqa: E402

from engine.models.hy4_flash.engine import Hy4FlashEngine  # noqa: E402
from engine.models.hy4_flash.serving import Hy4FlashServing  # noqa: E402
from tests.hy4_tiny import random_weights, tiny_config  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "hy4_state_bytes.json")
CH = 16


def digest(t: torch.Tensor) -> str:
    """sha256 of a tensor's bytes as stored, its shape included."""
    t = t.detach().contiguous().cpu()
    raw = t.numpy().tobytes() if t.dtype == torch.bool else t.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(repr(tuple(t.shape)).encode() + raw).hexdigest()


def host_key() -> str:
    return f"{platform.machine()}-{sys.platform}-torch{torch.__version__.split('+')[0]}"


def tokens(n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, 60, (n,), generator=g).tolist()


@torch.no_grad()
def gate_run() -> dict[str, str]:
    """The fixed script; each output is digested when it is made (a row pass hands out its row buffers,
    which the next pass overwrites)."""
    cfg = tiny_config()
    W, tables = random_weights(cfg, seed=1)
    eng = Hy4FlashEngine(cfg, W, tables, dtype=torch.bfloat16, max_len=256)
    s = Hy4FlashServing(eng, 256)
    out: dict[str, str] = {}
    ids = tokens(120, 11)
    for i in range(0, len(ids), CH):
        lg = s.forward(torch.tensor(ids[i:i + CH]), start=i)
        out[f"prefill.{i}"] = digest(lg)
    pos = len(ids)
    tok = int(lg.view(-1).argmax())
    for k in range(6):
        lg = s.forward(torch.tensor([tok]), start=pos)
        out[f"decode.{k}"] = digest(lg)
        pos += 1
        tok = int(lg.view(-1).argmax())
    # a tree: a branch off the anchor that is rejected, the spine's first three rows kept (a moved path)
    extra = tokens(8, 12)
    blk = [tok, extra[0], extra[1], extra[2], extra[3]]
    lg = s.forward_tree(torch.tensor(blk), [-1, 0, 0, 2, 3], start=pos)
    out["tree"], out["tree.picks"] = digest(lg), digest(s.picks)
    s.commit_tree([0, 2, 3])
    pos += 3
    # a chain block rolled back to its first two rows
    lg = s.forward_block(torch.tensor([int(s.picks[3])] + extra[4:8]), start=pos)
    out["block"] = digest(lg)
    s.rollback_to(2)
    pos += 2
    # a chain block whose commit waits, landed by `settle`
    lg = s.forward_block(torch.tensor(tokens(4, 13)), start=pos)
    out["chain"] = digest(lg)
    s.settle()
    pos += 4
    st = s.st
    assert st.length == pos and s.kv.length == pos, (st.length, pos)
    out["lat"] = digest(st.lat_all[:, :pos])
    out["ik"] = digest(st.ik_all[:, :pos])
    out["win"] = digest(st.win_all)
    out["hist"] = digest(st.hist)
    return out


def test_step0_the_state_bytes_reproduce():
    got = gate_run()
    again = gate_run()
    assert got == again, f"two runs in one process differ, the gate cannot work: {[k for k in got if got[k] != again[k]]}"
    fx = {}
    if os.path.isfile(FIXTURE):
        with open(FIXTURE) as f:
            fx = json.load(f)
    key = host_key()
    if os.environ.get("QSE_PIN_RECORD") == "1":
        fx.setdefault(key, {})["gate"] = got
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w") as f:
            json.dump(fx, f, indent=1, sort_keys=True)
            f.write("\n")
        return f"recorded {len(got)} digests for {key}"
    want = fx.get(key, {}).get("gate")
    if want is None:
        return f"SKIP compare: no digests recorded for {key} (known: {sorted(fx) or 'none'})"
    bad = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
    assert not bad, f"bytes moved in {bad} (host {key})"
    return f"{len(want)} digests reproduce ({key})"


def test_step0_the_gate_sees_one_changed_weight():
    """The gate is worth something only if one bf16 step in one weight moves a digest."""
    a = gate_run()
    cfg = tiny_config()
    W, _ = random_weights(cfg, seed=1)
    name = "layers.1.self_attn.o_proj.weight"
    real = random_weights

    def nudged(c, seed=1):
        W2, t2 = real(c, seed)
        flat = W2[name].view(-1)
        flat[0] = flat[0] * (1 + 2 ** -6)
        return W2, t2
    globals()["random_weights"] = nudged
    try:
        b = gate_run()
    finally:
        globals()["random_weights"] = real
    moved = [k for k in a if a[k] != b[k]]
    assert moved and name in W, f"nudging {name} moved nothing"
    return f"{len(moved)}/{len(a)} digests moved"


# ------------------------------------------------------------------ the lanes (ENG-485)
from engine.models.hy4_flash.engine import BLOCK  # noqa: E402

_E: list = []


def serving(max_len: int = 256) -> Hy4FlashServing:
    if not _E:
        cfg = tiny_config()
        W, tables = random_weights(cfg)
        _E.append(Hy4FlashEngine(cfg, W, tables, dtype=torch.bfloat16, max_len=max_len))
    return Hy4FlashServing(_E[0], max_len)


def test_a_side_slot_shares_the_engine_and_owns_its_state():
    s0 = serving()
    assert not s0.caps.joint_verify and not hasattr(s0, "forward_tree_multi")
    s1, d1 = s0.side_slot(128, None)
    assert type(s1) is Hy4FlashServing and s1.fn is s0.fn and d1 is None and s1.max_len == 128
    assert s1.st is not s0.st and s1.st.lat_all.data_ptr() != s0.st.lat_all.data_ptr()
    assert s1.st.win_all.data_ptr() != s0.st.win_all.data_ptr() and s1.st.hist.data_ptr() != s0.st.hist.data_ptr()
    assert s1.graph_set is not s0.graph_set and s1.st.graphs is s1.graph_set and s1.heads == {}
    cost = s0.cost()
    per, const = s0.side_slot_bytes(128, True)
    assert per == 0 and const > cost.section_bytes, (per, const)
    assert s0.side_slot_bytes(128, False) == (cost.view_bytes_per_token, const)
    return f"own state and graph set over one engine; a paged slot: 0 bytes a token, {const} constant"


def _lane_script(s, ids: list[int], seed: int) -> list:
    """One lane's work as steps, each returning the digests of what it made: the prompt in chunks of 16, three
    trees (a spine kept, then a moved path), a chain block left pending, four plain steps (the first lands it)."""
    x = tokens(40, seed)
    out = []
    for i in range(0, len(ids), CH):
        out.append(lambda i=i: [digest(s.forward(torch.tensor(ids[i:i + CH]), start=i))])
    for k in range(3):
        def tree(k=k):
            lg = s.forward_tree(torch.tensor(x[4 * k:4 * k + 4]), [-1, 0, 1, 0], start=s.kv.length)
            got = [digest(lg), digest(s.picks)]
            s.commit_tree([0, 1, 2] if k % 2 == 0 else [0, 3])
            return got
        out.append(tree)
    out.append(lambda: [digest(s.forward_block(torch.tensor(x[20:23]), start=s.kv.length))])
    for j in range(4):
        out.append(lambda j=j: [digest(s.forward(torch.tensor([x[30 + j]]), start=s.kv.length))])
    return out


def _state_of(s) -> list:
    st, L = s.st, s.kv.length
    return [L, digest(st.lat_all[:, :L]), digest(st.ik_all[:, :L]), digest(st.win_all), digest(st.hist)]


@torch.no_grad()
def test_two_lanes_interleaved_equal_each_alone():
    a_ids, b_ids = tokens(120, 21), tokens(90, 22)
    alone = []
    for ids, seed in ((a_ids, 31), (b_ids, 32)):
        s = serving()
        alone.append([d for step in _lane_script(s, ids, seed) for d in step()] + _state_of(s))
    s0 = serving()
    s1, _ = s0.side_slot(256, None)
    steps = [_lane_script(s0, a_ids, 31), _lane_script(s1, b_ids, 32)]
    got = [[], []]
    for k in range(max(len(x) for x in steps)):
        for lane in (0, 1):
            if k < len(steps[lane]):
                got[lane] += steps[lane][k]()
    got = [got[0] + _state_of(s0), got[1] + _state_of(s1)]
    for lane in (0, 1):
        bad = [i for i, (u, v) in enumerate(zip(got[lane], alone[lane])) if u != v]
        assert len(got[lane]) == len(alone[lane]) and not bad, (lane, bad[:8])
    return f"{len(got[0])} and {len(got[1])} digests each equal to the lane alone"


@torch.no_grad()
def test_a_lanes_drafter_writes_the_plain_text():
    from engine.drafters.ngram import NgramDrafter
    from engine.methods.mtp_drafter import MTPTreeDrafter
    from engine.methods.staircut import PriceModel
    from engine.models.hy4_flash.serving import priced_cut
    import test_hy4_dfly as D
    cfg, eng, s0, head0 = D.setup(torch.bfloat16)
    s0.heads["dfly"] = head0
    prices = PriceModel({"0": {"chain": {"1": 20.0, "4": 30.0}}, "4096": {"chain": {"1": 22.0}}}, 2.0)
    cut, freeze = priced_cut(prices, False, True)
    ng = NgramDrafter(corpus_path="", min_order=3, max_depth=16, min_corpus_order=8)
    d0 = MTPTreeDrafter(s0, head0, cut, ng, freeze=freeze)
    s1, d1 = s0.side_slot(256, d0)
    assert d1.s is s1 and d1.head is s1.heads["dfly"] and d1.head is not head0 and d1.head.srv is s1
    assert d1.cut is not d0.cut and d1.cut.prices.table == d0.cut.prices.table and d1.freeze and d1.cut.frozen
    assert d1.ngram.corpus is d0.ngram.corpus and d1.ngram.min_order == 3
    ids = D.tokens(22, 4)
    plain = Hy4FlashServing(eng, 256)
    lg = plain.forward(torch.tensor(ids), start=0)
    want = []
    for i in range(24):
        t = int(lg[0, 0].argmax())
        want.append(t)
        lg = plain.forward(torch.tensor([t]), start=22 + i)
    other = D.tokens(40, 5)
    lg = s1.forward(torch.tensor(ids), start=0)
    nxt, out, k = int(lg[0, 0].argmax()), [], 0
    while len(out) < 24:
        if k * CH < len(other):                          # slot 0 reads another prompt between the side slot's rounds
            s0.forward(torch.tensor(other[k * CH:(k + 1) * CH]), start=k * CH)
        k += 1
        d1.head.launch(s1.last_streams(), [nxt])
        tok, _ = d1.head.read()
        block = [nxt] + [t[0] for t in tok]
        s1.forward_tree(torch.tensor(block), list(range(-1, len(block) - 1)), start=s1.kv.length)
        picks = s1.picks.tolist()
        n = 1
        while n < len(block) and block[n] == picks[n - 1]:
            n += 1
        s1.commit_tree(list(range(n)))
        out += block[:n]
        nxt = picks[n - 1]
    assert out[:24] == want, (out, want)
    return "24 tokens drafted on a side slot equal plain greedy, slot 0 reading beside it"


class _Pages:
    """A PagedSet's bookkeeping: its committed rows, its margin, what `ensure` was asked."""

    def __init__(self, margin: int = 8):
        self.rows, self._m, self.asked = 0, margin, []

    def margin(self) -> int:
        return self._m

    def on_device(self) -> bool:
        return False

    def ensure(self, rows: int) -> None:
        self.asked.append(int(rows))
        self.rows = int(rows) + self._m

    def trim(self, rows: int = 0) -> None:
        self.rows = int(rows)


@torch.no_grad()
def test_every_pass_commits_its_rows_first():
    s = serving()
    floor = s.cfg.index_topk + BLOCK
    ps = s.st.pages = _Pages()
    assert s.kv.pages is ps
    ids = tokens(64, 41)
    s.forward(torch.tensor(ids[:16]), start=0)
    assert ps.asked == [max(floor, 16 + BLOCK)], ps.asked
    s.forward(torch.tensor(ids[16:48]), start=16)
    assert ps.asked[-1] == max(floor, 48 + BLOCK) and ps.rows >= 48 + BLOCK, ps.asked
    n = len(ps.asked)
    s.forward_tree(torch.tensor(ids[48:52]), [-1, 0, 1, 2], start=48)
    assert len(ps.asked) == n or ps.asked[-1] >= 48 + BLOCK, ps.asked     # committed already: one compare
    s.commit_tree([0, 1])
    ps.trim(0)                                                          # the server gave the pages back
    s.forward_block(torch.tensor(ids[50:53]), start=50)
    assert ps.asked[-1] == max(floor, 50 + BLOCK), ps.asked
    s.kv.ensure(200)                                                    # the round runner's ask
    assert ps.asked[-1] == 200, ps.asked
    s.kv.ensure(10_000)                                                 # never past the window
    assert ps.asked[-1] == s.max_len, ps.asked
    return f"asked {ps.asked} (floor {floor})"


@torch.no_grad()
def test_the_mirror_reads_committed_rows_only():
    ids, x = tokens(40, 51), tokens(4, 52)
    a, b = serving(), serving()
    for s in (a, b):
        s.forward(torch.tensor(ids), start=0)
    rr = a.cfg.index_kpool
    G = 40 // rr
    a.st.ik_all[:, 40:] = 7.0                                           # rows past the length: never read
    for s in (a, b):
        s.kv.length = 40                                                # a resume: the mirror is rebuilt
    la = a.forward_tree(torch.tensor(x), [-1, 0, 1, 2], start=40).clone()
    lb = b.forward_tree(torch.tensor(x), [-1, 0, 1, 2], start=40).clone()
    assert not bool((a.st.gk_all[:, G:a.st.Gc] == 7.0).any()), "the rebuild read past the committed rows"
    assert torch.equal(a.st.gk_all[:, :G], b.st.gk_all[:, :G]) and torch.equal(la, lb)
    return f"{G} groups rebuilt, none past the length; the same logits"


def test_the_state_goes_through_the_pool_and_the_graph_set():
    import inspect
    import types
    from engine.models.graph_set import GraphSet
    from engine.models.hy4_flash import engine as E
    from engine.models.hy4_flash import serving as S
    s = serving()
    v = s.cache_state().views()
    assert [(n, ax) for n, _, ax in v] == [("lat", 1), ("ik", 1)] and v[0][1] is s.st.lat_all and v[1][1] is s.st.ik_all
    assert s.memory_parts()["kv_pool"] == sum(t.numel() * t.element_size() for t in (s.st.lat_all, s.st.ik_all))
    # the pool is opened before the state that pages through it, and serve asks for it
    src = inspect.getsource(S.build)
    assert 0 < src.index("kv_pool()") < src.index("srv = Hy4FlashServing(eng, max_len)")
    assert "kv_pool=open_pool(a, ctx)" in inspect.getsource(S.serve)
    # captures on the slot's graph set: its stream and its one pool; none without a set (a self-check's state)
    gs = GraphSet()
    gs._stream = "stream"
    real = torch.cuda.graph_pool_handle
    torch.cuda.graph_pool_handle = lambda: "pool"
    try:
        assert E.capture_args(types.SimpleNamespace(graphs=gs)) == {"pool": "pool", "stream": "stream"}
    finally:
        torch.cuda.graph_pool_handle = real
    assert E.capture_args(types.SimpleNamespace(graphs=None)) == {} and E.capture_args(object()) == {}
    # a drop resets every graph of the slot at once and forgets them
    log = []

    class G:
        def __init__(self, name):
            self.name = name

        def reset(self):
            log.append(self.name)
    s.graphs = {(1, False): G("rows1"), (32, True): G("rows32")}
    s.heads["dfly"] = types.SimpleNamespace(graph=G("draft"))
    assert s.graph_count() == 3
    assert s.drop_graphs() == 3 and sorted(log) == ["draft", "rows1", "rows32"]
    assert s.graphs == {} and s.heads["dfly"].graph is None and s.graph_count() == 0 and s.graph_set.dropped
    return "whole views; paged memory not counted; the pool first; captures on the slot's set; one drop"



if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<56} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<56} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
