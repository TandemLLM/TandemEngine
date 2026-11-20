"""Hy4-flash in the server's engine interface on a tiny random model, on the CPU: the loop's verbs, the
cold verbs, and the cache tiers with this family's state (rings and last tokens as sections, latents and
indexer keys as views): a snapshot, the resident prefix, the disk tier after a restart and the session
spill each give back a state equal to a cold prefill, bit for bit."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)
import torch  # noqa: E402

from engine import cache, disk_cache, spill  # noqa: E402
from engine.models.hy4_flash.engine import Hy4FlashEngine  # noqa: E402
from engine.models.hy4_flash.serving import Hy4FlashServing, manifest  # noqa: E402
from tests.hy4_tiny import random_weights, tiny_config  # noqa: E402

CH = 16
BIG = 1 << 30
_E = []


def serving(max_len: int = 512) -> Hy4FlashServing:
    if not _E:
        cfg = tiny_config()
        W, tables = random_weights(cfg)
        _E.append(Hy4FlashEngine(cfg, W, tables, dtype=torch.bfloat16, max_len=max_len))
    return Hy4FlashServing(_E[0], max_len)


def tokens(n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, 60, (n,), generator=g).tolist()


def _raises(fn, *needles):
    try:
        fn()
    except ValueError as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return
    raise AssertionError(f"no ValueError (wanted {needles})")


def _cold(ids):
    eng = serving()
    lg, _, _ = cache.prefill(eng, None, ids, "cpu", chunk=CH)
    return eng, lg


def _same(eng, ref, n, lg=None, ref_lg=None):
    assert eng.kv.length == ref.kv.length == n
    for (k, a), b in zip(eng.cache_state().sections().items(), ref.cache_state().sections().values()):
        assert torch.equal(a, b), k
    for (k, a, ax), (_, b, _) in zip(eng.cache_state().views(), ref.cache_state().views()):
        assert torch.equal(a.narrow(ax, 0, n), b.narrow(ax, 0, n)), k
    if lg is not None:
        assert torch.equal(lg, ref_lg)


def test_the_loop_verbs():
    s, plain = serving(), serving()
    ids = tokens(40, 1)
    lg = s.forward(torch.tensor(ids[:20]), start=0)
    plain.forward(torch.tensor(ids[:20]), start=0)
    assert lg.shape == (1, 1, s.cfg.vocab_size) and s.kv.length == 20 and s.hidden_post_norm.shape == (1, 20, s.cfg.hidden_size)
    want = [plain.forward(torch.tensor([t]), start=20 + i).clone() for i, t in enumerate(ids[20:30])]
    _raises(lambda: s.forward(torch.tensor([1]), start=3), "forward at 3", "holds 20 rows")
    # a tree with a rejected branch, its moved path committed
    lg = s.forward_tree(torch.tensor([ids[20], 9, ids[21], ids[22]]), [-1, 0, 0, 2], start=20)
    assert lg.shape[0] == 4 and s.picks.tolist() == lg.argmax(-1).tolist() and s.kv.length == 20
    assert torch.equal(lg[0], want[0][0, 0]) and torch.equal(lg[2], want[1][0, 0]) and torch.equal(lg[3], want[2][0, 0])
    s.commit_tree([0, 2, 3])
    assert s.kv.length == 23
    # a chain block waits for its commit: the length counts it, the next forward lands it
    lg = s.forward_block(torch.tensor(ids[23:27]), start=23)
    assert s.kv.length == 27 and s.st.length == 23 and torch.equal(lg[3], want[6][0, 0])
    assert not s.trim_pending(28) and s.trim_pending(25) and s.st.length == 25
    lg = s.forward_block(torch.tensor(ids[25:29]), start=25)
    s.rollback_to(2)
    assert s.st.length == 27 and s.kv.length == 27
    a = s.forward(torch.tensor([ids[27]]), start=27)
    assert torch.equal(a, want[7])
    assert s.stats == {"prefill_rows": 20, "decode_steps": 1, "verify_blocks": 3, "verify_rows": 12}
    return "a prompt, a tree, a pending chain, a trim and a rollback leave the rows of the plain decode"


def test_kv_length_and_restart():
    s = serving(64)
    s.forward(torch.tensor(tokens(10, 2)), start=0)
    s.kv.length = 7
    assert s.st.length == 7 and s.kv.max_len == 64
    li = s.cfg.sparse_layers[0]
    kept = s.st.lat[li][:7].clone()
    s.kv.length = 0
    assert s.st.length == 0 and all(float(t.abs().sum()) == 0 for t in s.st.win.values())
    assert s.st.hist.tolist() == [s.cfg.engram_pad_id] * 2 and torch.equal(s.st.lat[li][:7], kept)
    _raises(lambda: setattr(s.kv, "length", 65), "capacity 64")
    return "zero restarts in place (rings and last tokens cleared, by-slot rows kept)"


def test_cost_and_cache_state():
    s = serving()
    c = s.cfg
    cost = s.cost()
    R = c.kv_lora_rank + c.sparse_attn.rope
    assert cost.view_bytes_per_token == len(c.sparse_layers) * (R + c.index_head_dim) * 2
    assert cost.section_bytes == len(c.window_layers) * c.sliding_window * R * 2 + 2 * 8
    secs = s.cache_state().sections()
    assert list(secs) == [f"win.{li}" for li in c.window_layers] + ["engram.hist"]
    assert all(secs[f"win.{li}"] is s.st.win[li] for li in c.window_layers)        # the live tensors
    assert [(n, ax) for n, _, ax in s.cache_state().views()] == [(f"{k}.{li}", 0) for li in c.sparse_layers for k in ("lat", "ik")]
    assert manifest().arch == "hy_v4_flash" and s.memory_parts()["weights"] > 0
    return f"{cost.view_bytes_per_token} bytes a token in {len(c.sparse_layers)} sparse layers, {cost.section_bytes} bytes of rings"


def test_a_snapshot_round_trips():
    eng, ref = serving(), serving()
    ids = tokens(70, 3)
    cache.prefill(eng, None, ids, "cpu", chunk=CH)
    snap = cache.capture(eng)
    cache.prefill(ref, None, ids, "cpu", chunk=CH)
    more = tokens(9, 4)
    eng.forward(torch.tensor(more), start=70)
    cache.restore(eng, snap)
    _same(eng, ref, 70)
    assert snap.nbytes == eng.cost().section_bytes + 70 * eng.cost().view_bytes_per_token
    a, b = eng.forward(torch.tensor(more), start=70), ref.forward(torch.tensor(more), start=70)
    assert torch.equal(a, b)
    return f"{snap.nbytes} bytes at 70 tokens; the restored state continues as the cold one"


def test_the_resident_prefix_resumes_exactly():
    eng, res = serving(), cache.ResidentPrefix(BIG, CH, stash_bytes=BIG)
    turn1 = tokens(100, 5)
    cache.prefill(eng, None, turn1, "cpu", chunk=CH, resident=res)
    assert sorted(res.anchors)[-1] == 96
    turn2 = turn1 + tokens(30, 6)
    lg, reused, _ = cache.prefill(eng, None, turn2, "cpu", chunk=CH, resident=res)
    assert reused == 96, reused
    ref, ref_lg = _cold(turn2)
    _same(eng, ref, len(turn2), lg, ref_lg)
    # a turn that departs inside the kept rows resumes at the anchor before the change
    turn3 = turn1[:50] + tokens(40, 7)
    lg, reused, _ = cache.prefill(eng, None, turn3, "cpu", chunk=CH, resident=res)
    ref, ref_lg = _cold(turn3)
    assert reused == 48, reused
    _same(eng, ref, len(turn3), lg, ref_lg)
    return "turn 2 resumed at 96 and a changed turn at 48, each equal to a cold run"


def _tier(d):
    from disk_tier_util import waiting
    return waiting(disk_cache.DiskTier(d, chunk=CH, fingerprint="hy4", settings={"admit_after": 1, "budget_gb": 1},
                                       free_bytes=lambda p: 10 ** 15, head_anchors=2, min_tokens=CH).open())


def test_the_disk_tier_round_trips_after_a_restart():
    ids = tokens(150, 8)
    with tempfile.TemporaryDirectory() as d:
        t = _tier(d)
        cache.prefill(serving(), None, ids, "cpu", chunk=CH, disk=t)
        assert t.flush()
        t.close()
        anc = os.listdir(os.path.join(d, "anchors"))
        meta = t._read_header(os.path.join(d, "anchors", anc[0]), "anchor")
        names = [s["name"] for s in meta["sections"]]
        assert meta["format"] == 2 and names[0] == "win.2" and "engram.hist" in names
        t2 = _tier(d)                                  # a restart
        prompt = ids + tokens(20, 9)
        eng, info = serving(), {}
        lg, reused, _ = cache.prefill(eng, None, prompt, "cpu", chunk=CH, disk=t2, info=info)
        assert (info.get("kind"), reused) == ("disk", 144), (info, reused)
        ref, ref_lg = _cold(prompt)
        _same(eng, ref, len(prompt), lg, ref_lg)
        t2.close()
    return "restored at 144 from the disk tier after a restart, equal to a cold run"


def test_the_session_spill_round_trips():
    d = tempfile.mkdtemp(prefix="hy4-spill-")
    try:
        t = _tier(d)
        sp = spill.SessionSpill(os.path.join(d, "sessions"), chunk=CH, fingerprint="hy4", tier=t,
                                settings={"budget_gb": 1.0, "min_tokens": 64, "drain_min_tokens": CH, "codec": "off"},
                                free_bytes=lambda p: 10 ** 15, floor_bytes=0).open()
        eng, res = serving(), cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, end=True)
        ids = tokens(130, 10)
        cache.prefill(eng, None, ids, "cpu", chunk=CH, resident=res)
        ref = serving()
        cache.prefill(ref, None, ids, "cpu", chunk=CH)
        anchors0 = {b: tuple(x.clone() for x in a) for b, a in res.anchors.items()}
        info = sp.save_resident(res, eng, None, "evict")
        assert info is not None, sp.stats
        res.clear()
        for _, v, _ in eng.cache_state().views():
            v.zero_()
        got = sp.restore(res, eng, None, sp.entries[info["eid"]])
        assert got is not None, sp.stats
        for (k, a, ax), (_, b, _) in zip(eng.cache_state().views(), ref.cache_state().views()):
            assert torch.equal(a.narrow(ax, 0, 130), b.narrow(ax, 0, 130)), k
        for b in sorted(anchors0)[-4:]:
            assert all(torch.equal(x, y) for x, y in zip(res.anchors[b], anchors0[b]))
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return "rows and the newest anchors back in place, bit for bit"


def test_the_price_class_follows_the_context():
    srv = serving()
    k = srv.cfg.index_topk
    got = [srv.price_class(n) for n in (0, k - 1, k, 4096, 4097, 60000, 262144, 10 ** 7)]
    want = [0, 0, 4096 if k <= 4096 else None, 4096, 8192, 65536, 262144, 262144]
    assert all(w is None or g == w for g, w in zip(got, want)), (got, want)
    # the frozen cut over the pack's price file has a draft cost and a block price in every class a round can be in
    from engine.methods.staircut import PriceModel
    from engine.models.hy4_flash.serving import PRICE_CLASSES, priced_cut
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pm = PriceModel.load(os.path.join(root, "ops", "prices", "hy4-flash", "hy4-3.08.dgx-spark.json"), log=lambda s: None)
    assert pm.table, "the 3.08 pack's price file is missing"
    assert {srv.price_class(n) for n in (0, 3000, 5000, 20000, 40000, 100000, 200000, 262144)} <= set(PRICE_CLASSES)
    cut, frozen = priced_cut(pm, siblings=False, freeze=True)
    assert frozen and all(cut.draft.get(c, 0) > 0 and pm.prior(6, "chain", c) > pm.prior(1, "chain", c) > 0
                          and cut.spec_ms(c) < cut.plain_ms(c) for c in PRICE_CLASSES), cut.draft
    assert priced_cut(PriceModel(), siblings=False, freeze=True)[1] is False          # no file: the cut learns
    # the profile that names the price file: the family's running one, else the pack's own bit rate, else its folder
    import json
    from engine.models.hy4_flash.serving import profile_of
    was = os.environ.pop("QSE_WEIGHTS", None)
    try:
        with tempfile.TemporaryDirectory() as d:
            pack = os.path.join(d, "somewhere")
            os.makedirs(pack)
            assert profile_of(pack) == "hy4-somewhere"
            with open(os.path.join(pack, "config.json"), "w") as f:
                json.dump({"quantization_config": {"bits": 3.08}}, f)
            assert profile_of(pack) == "hy4-3.08"
            os.environ["QSE_WEIGHTS"] = "flashnext-3.05"
            assert profile_of(pack) == "hy4-3.08"                      # another model's profile is not this pack's
            os.environ["QSE_WEIGHTS"] = "hy4-custom"
            assert profile_of(pack) == "hy4-custom"
    finally:
        os.environ.pop("QSE_WEIGHTS", None)
        if was is not None:
            os.environ["QSE_WEIGHTS"] = was
    assert os.path.isfile(os.path.join(root, "ops", "prices", "hy4-flash", "hy4-3.08.dgx-spark.json"))
    # the limits a start without its own takes: the 3.08 pack at 262,144 tokens holds 63.9 GiB, 8.66 more at 3.5 bits
    from engine.models.hy4_flash.serving import memory_limits
    assert memory_limits(63.9) == (76.0, 83.0) and memory_limits(63.9 + 8.66) == (85.0, 92.0)
    return f"dense below {k} rows: class 0; then the next 4,096 x 2^k, up to 262,144; the frozen cut is priced in all {len(PRICE_CLASSES)}"


def test_the_fingerprint_covers_the_kernels_the_family_compiles():
    # the packed value half is built on tools/fn_kernels' source (compiled in): a change there, in the EXL3
    # helpers it reads or in the family's own kernel moves this family's fingerprint
    import shutil
    import tempfile
    from engine.methods import manifest as mf
    from engine.models.hy4_flash.serving import manifest
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rels = {os.path.relpath(p, root) for p in mf.files(root, manifest())}
    need = ["engine/models/hy4_flash/native.py", "engine/models/hy4_flash/csrc/native/vdense.cu",
            "engine/models/hy4_flash/csrc/native/vdense.cpp", "tools/fn_kernels.py", "tools/csrc/fn_kernels/kernels.cu",
            "tools/csrc/fn_kernels/kernels.cpp", "tools/exl3_linear.py", "tools/csrc/exl3_linear/exl3.cu", "engine/native_build.py"]
    assert not [r for r in need if r not in rels], [r for r in need if r not in rels]
    with tempfile.TemporaryDirectory() as d:
        for sub in ("engine", "tools", "ops"):
            shutil.copytree(os.path.join(root, sub), os.path.join(d, sub), ignore=shutil.ignore_patterns("__pycache__", "*.npy"))
        base = mf.fingerprint(d, manifest(), {})
        assert base == mf.fingerprint(root, manifest(), {}), "the copy's fingerprint is not the tree's"
        moved = []
        for rel in ("tools/csrc/fn_kernels/kernels.cu", "tools/fn_kernels.py", "tools/csrc/exl3_linear/exl3.cu",
                    "engine/models/hy4_flash/csrc/native/vdense.cu"):
            p = os.path.join(d, rel)
            was = open(p, "rb").read()
            with open(p, "ab") as f:
                f.write(b"\n// one more line\n" if rel.endswith(".cu") else b"\n# one more line\n")
            moved.append(mf.fingerprint(d, manifest(), {}) != base)
            with open(p, "wb") as f:
                f.write(was)
        assert all(moved), moved
    return f"{len(rels)} files; a line more in the shared kernels, the EXL3 source or the family's kernel moves the fingerprint"


def test_the_caps_carry_the_conductor_fit():
    """SRV-899: the router's notes have no place in Hy4-flash's prompt (every route row said
    `no_insertion_point`): its caps say `notes: False`, the key Flint declares for the same reason."""
    from engine.models.flint.serving import FlintServing
    assert Hy4FlashServing.caps.conductor == {"notes": False} == FlintServing.caps.conductor
    assert not Hy4FlashServing.caps.side_slots and Hy4FlashServing.caps.caches


def test_the_packs_limits_reach_the_cache_policy():
    """A start on serve.env's 70 and 77 GiB takes the pack's limits (serving.memory_limits over what the load holds),
    and they stay the cache policy's after a request refreshes the residency limits (server/memlimits.py), which
    falls back to the flags as started. On the 3.56 pack in chat mode the refresh took the ceiling back to 70 GiB
    under 72.69 GiB allocated, and every save was refused. A ceiling the residency document sets still wins."""
    import types

    from engine.models.hy4_flash.serving import SERVER_LIMITS, memory_limits
    from server import app as A
    from server import ext as ext_mod
    from server import memlimits
    gib = 1 << 30
    held = 72.69                                         # allocated after the 3.56 pack's load at 262,144 tokens
    keys = ("serve_flags", "args", "mem", "limits_version", "limits_applied")
    keep = {k: A.STATE.get(k) for k in keys}
    doc = ext_mod.document

    def build(a, ctx):                                   # serve()'s rule, on the device's figure
        if (a.mem_ceiling_gib, a.mem_peak_gib) == SERVER_LIMITS:
            a.mem_ceiling_gib, a.mem_peak_gib = memory_limits(held)
        return {}

    def start(load):
        for k in keys:
            A.STATE.pop(k, None)
        a = types.SimpleNamespace(mem_ceiling_gib=SERVER_LIMITS[0], mem_peak_gib=SERVER_LIMITS[1],
                                  reserve_slack_gib=2.0)
        A.STATE["serve_flags"] = dict(vars(a))          # _limits_at_start, before the load
        load(a)
        A.STATE["args"] = vars(a)
        A.STATE["mem"] = cache.Budget(int(a.mem_ceiling_gib * gib), int(a.mem_peak_gib * gib),
                                      int(a.reserve_slack_gib * gib))
        return a

    try:
        ext_mod.document = lambda state, name: (7, {"version": 7})       # no engine section: the flags
        # the fault, on the bare builder call: the refresh puts the start's ceiling back under the load
        start(lambda a: build(a, None))
        memlimits.refresh(A.STATE)
        assert A.STATE["mem"].ceiling == 70 * gib, A.STATE["mem"].ceiling / gib
        # through the server's load: the pack's limits survive the refresh, above what the load holds
        a = start(lambda a: A._build_with_limits(build, a, None))
        assert (a.mem_ceiling_gib, a.mem_peak_gib) == (85.0, 92.0)
        memlimits.refresh(A.STATE)
        assert A.STATE["mem"].ceiling == 85 * gib and A.STATE["mem"].peak == 92 * gib, A.STATE["mem"].ceiling / gib
        assert A.STATE["mem"].ceiling > held * gib
        # a limit the dashboard set in the residency document still wins
        ext_mod.document = lambda state, name: (8, {"engine": {"MEM_CEILING_GIB": 90}})
        memlimits.refresh(A.STATE)
        assert A.STATE["mem"].ceiling == 90 * gib
    finally:
        ext_mod.document = doc
        for k, v in keep.items():
            if v is None:
                A.STATE.pop(k, None)
            else:
                A.STATE[k] = v


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
