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
    # ENG-485: two whole tensors, row axis 1 (a paged buffer is found by its base address)
    assert [(n, t.shape, ax) for n, t, ax in s.cache_state().views()] == [
        ("lat", (len(c.sparse_layers), s.max_len, R), 1), ("ik", (len(c.sparse_layers), s.max_len, c.index_head_dim), 1)]
    assert s.cache_state().views()[0][1] is s.st.lat_all and s.cache_state().views()[1][1] is s.st.ik_all
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


def test_the_caches_give_a_cold_read_on_the_v2_prompt_pass():
    """SPD-323, SPD-325: on the prompt pass on fixed-order kernels (HY4_PREFILL_V2) the resident prefix, the disk tier
    after a restart and the session spill each give back what a cold prompt read on that pass writes, bit for bit."""
    serving()
    _E[0].prefill_v2 = True
    try:
        return "; ".join(f() for f in (test_the_resident_prefix_resumes_exactly, test_the_disk_tier_round_trips_after_a_restart,
                                        test_the_session_spill_round_trips))
    finally:
        _E[0].prefill_v2 = False


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


def test_the_fingerprint_covers_every_file_the_family_uses():
    """ENG-512: every file of another package this family's entry `uses` is in its manifest, so a change there moves its
    fingerprint: table.py reads its rows through flashnext/ple.py's O_DIRECT reader (BLOCK, DirectRows), which the
    manifest left out."""
    from engine.models import REGISTRY
    from engine.models.hy4_flash.serving import MODULES
    uses = set(REGISTRY["hy_v4_flash"].uses)
    assert "engine/models/flashnext/ple.py" in uses and uses <= set(MODULES), sorted(uses - set(MODULES))
    for path in uses:
        mod = path[:-3].replace("/", ".")
        users = [f for f in MODULES if f.startswith("engine/models/hy4_flash/")
                 and mod in open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f)).read()]
        assert users, (path, "no file of the family imports it: leave it out of `uses`")


def test_the_prompt_pass_takes_the_grouped_kernels_by_default():
    """SPD-311: a prompt's routed experts run on the grouped kernels unless QWEN38_HY4_PREFILL_GROUPED=0 says the pair
    kernels; build() hands the setting to the engine. The two give the same bits: the kernels are the device's (the
    expert banks), so tests/gpu/test_hy4_prefill_grouped_gpu.py compares them there, and a CPU prompt pass reads
    neither."""
    import inspect

    from engine.models.hy4_flash import serving as S
    from engine.models.hy4_flash.knobs import KNOBS
    from engine.settings import SETTINGS
    assert KNOBS["HY4_PREFILL_GROUPED"][0] == "1"
    if "QWEN38_HY4_PREFILL_GROUPED" not in os.environ:
        assert SETTINGS.get("HY4_PREFILL_GROUPED") == "1"
    assert 'eng.prefill_grouped = _S.get("HY4_PREFILL_GROUPED") == "1"' in inspect.getsource(S.build)


def test_the_caps_carry_the_conductor_fit():
    """SRV-899: the router's notes have no place in Hy4-flash's prompt (every route row said
    `no_insertion_point`): its caps say `notes: False`, the key Flint declares for the same reason."""
    from engine.models.flint.serving import FlintServing
    assert Hy4FlashServing.caps.conductor == {"notes": False} == FlintServing.caps.conductor
    # ENG-485: side slots whose lanes take turns
    assert Hy4FlashServing.caps.side_slots and not Hy4FlashServing.caps.joint_verify and Hy4FlashServing.caps.caches


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


def test_the_store_entry_cap_is_the_familys_and_survives_a_refresh():
    """SRV-907 (the design's 4.6): a start on the server's default `--store-entry-gb` (1 GiB) takes this family's
    0.25 GiB, and a request's refresh of the residency limits keeps it: the builder's change becomes the flag
    the refresh falls back to (server/app.py `BUILDER_LIMITS`). Called bare, the refresh takes it back to 1 GiB,
    which is the fault the list prevents. A start that names its own cap keeps it."""
    import types

    from engine.models.hy4_flash.serving import STORE_ENTRY_GB, store_entry
    from server import app as A
    from server import ext as ext_mod
    from server import memlimits
    gib = 1 << 30
    keys = ("serve_flags", "args", "state_store", "limits_version", "limits_applied")
    keep = {k: A.STATE.get(k) for k in keys}
    doc = ext_mod.document

    def start(load, cap=1.0):
        for k in keys:
            A.STATE.pop(k, None)
        a = types.SimpleNamespace(store_entry_gb=cap)
        A.STATE["serve_flags"] = dict(vars(a))
        load(a)
        A.STATE["args"] = vars(a)
        A.STATE["state_store"] = cache.StateStore(2 * gib, max_entry_bytes=int(a.store_entry_gb * gib))
        memlimits.refresh(A.STATE)
        return a, A.STATE["state_store"].max_entry

    def build(a, ctx):
        store_entry(a)
        return {}
    try:
        ext_mod.document = lambda state, name: (7, {"version": 7})
        assert start(lambda a: build(a, None))[1] == 1 * gib                      # the fault, on a bare call
        a, cap = start(lambda a: A._build_with_limits(build, a, None))
        assert a.store_entry_gb == STORE_ENTRY_GB == 0.25 and cap == int(0.25 * gib), cap
        a, cap = start(lambda a: A._build_with_limits(build, a, None), cap=0.5)   # the start's own cap
        assert a.store_entry_gb == 0.5 and cap == int(0.5 * gib), cap
    finally:
        ext_mod.document = doc
        for k, v in keep.items():
            if v is None:
                A.STATE.pop(k, None)
            else:
                A.STATE[k] = v


# ------------------------------------------------------------------ the conversation cache (SRV-907)
# The tests below run the follow-up turns on the tiny model in fp32: "same tokens" compares the next greedy
# tokens of a resumed turn with a cold read's, and bf16 random weights would leave near-ties between two
# chunkings of the same rows (the gate a resume at an end or a prompt end is held to: tests/test_end_anchor.py).

_E32 = []
#: a family's capabilities -> the resident prefix's arguments, as server/app.py `_load_caches` passes them
CAPS_ARGS = (("chat_first", "chat_first"), ("head_mark", "guest_store"), ("reread_end", "end_reread"))


def serving32(max_len: int = 512) -> Hy4FlashServing:
    if not _E32:
        cfg = tiny_config()
        W, tables = random_weights(cfg)
        _E32.append(Hy4FlashEngine(cfg, W, tables, dtype=torch.float32, max_len=512))
    return Hy4FlashServing(_E32[0], max_len)


def _prefix(**kw) -> cache.ResidentPrefix:
    """The resident prefix as the server builds it for this family: its capabilities' cache flags, the end
    anchor on, a stash."""
    for flag, arg in CAPS_ARGS:
        if getattr(Hy4FlashServing.caps, flag, False):
            kw.setdefault(arg, True)
    kw.setdefault("stash_bytes", BIG)
    return cache.ResidentPrefix(BIG, kw.pop("chunk", CH), end=True, **kw)


class _prompt_end:
    """The resident prefix keeps the last prompt's end (SPD-215, on by default; a test of another file may have
    turned it off in this process)."""

    def __enter__(self):
        self.keep, cache.PROMPT_END = cache.PROMPT_END, True

    def __exit__(self, *exc):
        cache.PROMPT_END = self.keep
        return False


def _greedy(eng, lg, n0: int, k: int) -> list[int]:
    """`k` greedy tokens after a prompt of `n0` rows, as the served loop leaves them (the last one decided and
    not forwarded: `kv.length == n0 + k - 1`)."""
    out, pos = [], n0
    for i in range(k):
        t = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
        out.append(t)
        if i == k - 1:
            break
        lg = eng.forward(torch.tensor([t]), start=pos)
        pos += 1
    return out


def _remember(eng, res, ctx, n_prompt: int, session=None, own: bool = False, store=None) -> None:
    """server/app.py `_remember_turn` after a turn on this engine and prefix."""
    from server import app
    keep = dict(app.STATE)
    app.STATE.clear()
    app.STATE.update(engine=eng, drafter=None, resident=res, state_store=store, session_cache=store is not None,
                     last_ctx=list(ctx), last_prompt_n=n_prompt, hidden_spans=[], suffix_store=None, verbose=False)
    try:
        app._remember_turn(list(ctx[:n_prompt]), list(ctx[n_prompt:]), session, **({"own": True} if own else {}))
    finally:
        app.STATE.clear()
        app.STATE.update(keep)


def _turn(eng, res, ids, k: int, session=None, own: bool = False, store=None, chunk: int = CH, **kw):
    """One request as the server runs it: the prefill through the caches, `k` greedy tokens, `_remember_turn`.
    Returns (the context: prompt and tokens, what the prefill did)."""
    info: dict = {}
    if own:
        kw["own"] = True
    lg, reused, fwd = cache.prefill(eng, None, list(ids), "cpu", chunk=chunk, resident=res, conv_id=session,
                                    store=store, checkpoint=store is not None, info=info, **kw)
    did = {"kind": info.get("kind"), "reused": int(reused), "forwarded": int(fwd), "guest": res._stash is not None}
    ctx = list(ids) + _greedy(eng, lg, len(ids), k)
    _remember(eng, res, ctx, len(ids), session, own, store)
    return ctx, did


def _cold32(ids, chunk: int = CH, **kw):
    eng = serving32()
    lg, _, _ = cache.prefill(eng, None, list(ids), "cpu", chunk=chunk, **kw)
    return eng, lg


def _cold_greedy(ids, k: int, **kw) -> list[int]:
    eng, lg = _cold32(ids, **kw)
    return _greedy(eng, lg, len(ids), k)


def _other(tok: int) -> int:
    """A token that is not `tok`: where a copy parts from the rows."""
    return 5 if tok != 5 else 6


def _thinking_on(res, eng, session=None):
    """The rehearsal's shape (thinking on, no tools): turn 1 thinks long, and turn 2's prompt holds turn 1's
    prompt, then the answer without its reasoning and a new message, so it parts from the rows right after turn
    1's prompt and is shorter than a quarter of them (77 rows beside 336). Returns turn 1's and turn 2's
    (context, what the prefill did) and turn 2's prompt."""
    p1 = tokens(37, 20)
    c1, t1 = _turn(eng, res, p1, 300, session)
    p2 = p1 + [_other(c1[37])] + tokens(39, 21)
    c2, t2 = _turn(eng, res, p2, 12, session)
    return (c1, t1), (c2, t2), p2


def test_a_follow_up_with_thinking_on_resumes_at_the_prompt_end():
    """5.1 test 1. Before: turn 2 ran as a guest of its own conversation (it took the prompt end but kept no end
    and no prompt end), and turn 3 resumed at turn 1's prompt end again."""
    with _prompt_end():
        res, eng = _prefix(), serving32()
        (c1, _), (c2, t2), p2 = _thinking_on(res, eng)
        assert len(c1) - 1 == 336 and len(p2) == 77 < 0.25 * 336
        assert t2["kind"] == "pend" and t2["reused"] == 37 and not t2["guest"], t2
        assert res.end is not None and res.end["at"] == len(c2) - 1 and res.pend["at"] == 77, (res.end, res.pend)
        assert c2[77:] == _cold_greedy(p2, 12), "turn 2's greedy tokens are not a cold read's"
        p3 = p2 + [_other(c2[77])] + tokens(29, 22)
        c3, t3 = _turn(eng, res, p3, 12)
        assert t3["kind"] == "pend" and t3["reused"] == 77 and not t3["guest"], t3
        assert c3[len(p3):] == _cold_greedy(p3, 12), "turn 3's greedy tokens are not a cold read's"
    return f"turn 2 resumed at 37 and forwarded {t2['forwarded']}, turn 3 at 77 and forwarded {t3['forwarded']}"


def test_a_follow_up_without_a_session_takes_no_park_and_no_keeper():
    """5.1 test 1a: the follow-up of test 1 without a session id is the same conversation. Each of the three ways
    `begin` sets another session's rows aside stays unused: the park, a side slot's keeper (an NVMe write) and
    the session spill (here a stand-in that records its calls). A request of another session would take each."""
    import types
    for way in ("park", "keeper", "spill"):
        calls = []
        with _prompt_end():
            res, eng = _prefix(park_bytes=BIG if way == "park" else 0), serving32()
            if way == "keeper":
                res.keeper = lambda *a: calls.append("keeper")
            if way == "spill":
                res.spiller = types.SimpleNamespace(
                    min_tokens=1, cfg={}, maybe_restore=lambda *a, **k: None,
                    save_resident=lambda *a, **k: calls.append("spill") or {"eid": "stand-in"})
            _, (_, t2), _ = _thinking_on(res, eng)
            assert not t2["guest"] and t2["kind"] == "pend", (way, t2)
            assert calls == [] and res.stats["parks"] == 0 and res.stats["spills"] == 0, (way, calls, res.stats["parks"])
            _turn(eng, res, tokens(120, 23), 2, "another")          # another session: the same way is taken
        assert res.stats["parks"] == 1 if way == "park" else calls == [way], (way, calls, res.stats["parks"])


def test_the_conductors_question_is_a_guest_at_any_size():
    """5.1 test 1b: the conductor's question longer than a quarter of the chat in place runs as a guest; the
    chat's end anchor and prompt end are still there after it, and its next turn resumes at its end. Before:
    the question replaced the chat."""
    with _prompt_end():
        res, eng = _prefix(), serving32()
        c1, _ = _turn(eng, res, tokens(37, 30), 30, "chat")
        q = tokens(40, 31)
        assert len(q) > 0.25 * res.valid
        _, tq = _turn(eng, res, q, 3, "q", own=True)
        assert tq["guest"], tq
        assert res.end is not None and res.end["at"] == len(c1) - 1 and res.pend["at"] == 37, (res.end, res.pend)
        p2 = c1 + tokens(5, 32)
        want = _cold_greedy(p2, 8)
        c2, t2 = _turn(eng, res, p2, 8, "chat")
    assert t2["kind"] == "end" and t2["reused"] == len(c1) - 1, t2
    assert c2[len(p2):] == want, "the chat's next turn after the question is not a cold read's"
    return f"a {len(q)}-row question beside {len(c1) - 1} rows was a guest; the chat resumed at its end"


def test_a_question_that_holds_every_row_leaves_the_chat_intact():
    """The conductor's question whose prompt holds every row in place (the chat's whole context, then its own
    text) runs as a guest too. As no guest it went on with the chat's rows and made them its own, and the rows
    it then gave back were the chat's: its rows, anchors and end."""
    with _prompt_end():
        res, eng = _prefix(), serving32()
        c1, _ = _turn(eng, res, tokens(37, 110), 30, "chat")
        e = res.end["at"]
        q = c1 + tokens(40, 111)
        assert res.common(q) == res.valid == e
        _, tq = _turn(eng, res, q, 3, "q", own=True)
        assert tq["guest"] and tq["kind"] == "end", tq
        assert res.valid == e and res.end["at"] == e and res.pend["at"] == 37 and res.owner == "chat", res.report()
        p2 = c1 + tokens(5, 112)
        want = _cold_greedy(p2, 6)
        c2, t2 = _turn(eng, res, p2, 6, "chat")
    assert (t2["kind"], t2["reused"]) == ("end", e) and c2[len(p2):] == want, t2


def test_a_question_whose_client_left_leaves_nothing_in_place():
    """The conductor's first question after a start is no guest (nothing is in place); when its client leaves
    (the conductor's timeout is 8 s, the first read of its head about 19 s on the device), `_remember_turn`
    never runs. Its rows are given back at the next prefill, so the next short chat is no guest of them."""
    with _prompt_end():
        res, eng = _prefix(), serving32()
        cache.prefill(eng, None, tokens(300, 113), "cpu", chunk=CH, resident=res, own=True)   # no `_remember`
        assert res.own_from == 0 and res.valid == 300, (res.own_from, res.valid)
        p = tokens(37, 114)
        assert len(p) < 0.25 * res.valid
        c, t = _turn(eng, res, p, 10, "chat")
    assert not t["guest"] and res.owner == "chat" and res.pend["at"] == 37 and res.end["at"] == len(c) - 1, t
    assert c[37:] == _cold_greedy(p, 10)


def test_a_question_at_the_chats_prompt_end_stores_a_cold_reads_head():
    """A guest that resumed at the chat's prompt end. Before the end read again that prompt end counted as a
    session's rows and the store took no snapshot of the guest's prefill. On this family the prompt end is a cold
    read's state (`pend["exact"]`, the row-invariant prompt pass), so what the guest reads from there is a cold
    read's rows and the store keeps its grid lines (the ruling: `taint` clear where the rows are exact); the
    chat's prompt end and its end stay as they were."""
    store = cache.StateStore(BIG, chunk=CH)
    with _prompt_end():
        res, eng = _prefix(), serving32()
        p1 = tokens(37, 115)
        c1, _ = _turn(eng, res, p1, 30, "chat", store=store)
        e, puts = res.end["at"], store.stats["puts"]
        assert res.pend.get("exact") is True and res.taint is None, (res.pend, res.taint)
        q = p1 + [_other(c1[37])] + tokens(60, 116)
        _, tq = _turn(eng, res, q, 3, "q", own=True, store=store)
    assert tq["guest"] and (tq["kind"], tq["reused"]) == ("pend", 37), tq
    got = sorted(x.tokens.__len__() for x in store._d.values() if x.conv_id == "q" and x.kind == "prefix")
    assert store.stats["puts"] > puts and got == [48, 64, 80, 96], (store.stats["puts"], puts, got)
    assert res.end is not None and res.end["at"] == e and res.pend["at"] == 37, (res.end, res.pend)


def test_the_chat_after_the_conductors_question_owns_the_rows():
    """5.1 test 1c: the conductor's question first (nothing in place: it is no guest), then a chat. The question
    keeps no session snapshot and gives its rows back; the chat owns the rows, its prompt end and its end are
    kept. Before: the question stayed the resident prompt and the chat was its guest (`guest`, `not_resident`)."""
    store = cache.StateStore(BIG, chunk=CH)
    with _prompt_end():
        res, eng = _prefix(), serving32()
        _, tq = _turn(eng, res, tokens(120, 33), 3, "q", own=True, store=store)
        assert not tq["guest"] and res.valid == 0 and not res.anchors, (tq, res.valid)
        assert "session" not in {e.kind for e in store._d.values()}, "the question's session snapshot was kept"
        c, t = _turn(eng, res, tokens(37, 34), 10, "chat", store=store)
    assert not t["guest"] and res.pend["at"] == 37 and res.end is not None and res.end["at"] == len(c) - 1, (t, res.end)
    assert "session" in {e.kind for e in store._d.values()}


def test_a_follow_up_on_a_side_slots_prefix_is_no_guest():
    """5.1 test 2: the same follow-up on a prefix built as a side slot's (`guest_frac` 1: every prompt shorter
    than the rows that does not hold them all is a guest there)."""
    with _prompt_end():
        res, eng = _prefix(guest_frac=1.0), serving32()
        (c1, _), (c2, t2), _ = _thinking_on(res, eng, "chat")
    assert t2["kind"] == "pend" and not t2["guest"] and res.end["at"] == len(c2) - 1, (t2, res.end)


def test_the_last_prompt_comes_back_with_its_rows():
    """The length of the last prompt (`last_n`) travels with the conversation's rows: into the park and back,
    to the NVMe and back. After a park swap, a short request of another session that holds as many of the
    returned chat's tokens as the other conversation's last prompt had is a guest, as by size. The pin brings
    back a head below an anchor, never a whole last prompt: `last_n` is 0 there."""
    with _prompt_end():
        res, eng = _prefix(park_bytes=BIG), serving32()
        a = tokens(60, 120)
        _turn(eng, res, a, 200, "A")
        _turn(eng, res, tokens(20, 121), 3, "B", stay=True)      # a worker's request: A goes to the park
        assert res.owner == "B" and res.last_n == 20 and res.park_last_n == 60 and res.stats["parks"] == 1
        assert res.unpark(eng, None) and res.owner == "A" and res.last_n == 60 and res.park_last_n == 20
        q = a[:25] + [_other(a[25])] + tokens(5, 122)
        assert len(q) < 0.25 * res.valid
        _, tq = _turn(eng, res, q, 2, "C")
        assert tq["guest"], tq
        res, eng = _prefix(pin_bytes=BIG), serving32()
        _turn(eng, res, tokens(100, 123), 2, "A")
        _turn(eng, res, tokens(60, 124), 2, "B")                 # too long for a guest: A's head is pinned
        assert res.pin_rows > 0 and res.last_n == 60, (res.pin_rows, res.last_n)
        assert res.unpin(eng, None) and res.last_n == 0
    d = tempfile.mkdtemp(prefix="hy4-last-n-")
    try:
        t = _tier(d)
        sp = spill.SessionSpill(os.path.join(d, "sessions"), chunk=CH, fingerprint="hy4", tier=t,
                                settings={"budget_gb": 1.0, "min_tokens": 64, "drain_min_tokens": CH, "codec": "off"},
                                free_bytes=lambda p: 10 ** 15, floor_bytes=0).open()
        res, eng = _prefix(), serving32()
        _turn(eng, res, tokens(100, 125), 10, "A")
        info = sp.save_resident(res, eng, None, "evict")
        assert info is not None, sp.stats
        res.clear()
        assert res.last_n == 0
        assert sp.restore(res, eng, None, sp.entries[info["eid"]]) is not None, sp.stats
        assert res.last_n == 100 and res.owner == "A", (res.last_n, res.owner)
        t.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_promoted_guests_next_turn_is_no_guest():
    """5.1 test 3: a chat that ran as a guest on a side slot and became its conversation (`promote_guest`): its
    next turn holds its whole prompt and is no guest, though it is shorter than the rows; the turn after it
    resumes at that turn's prompt end."""
    with _prompt_end():
        res, eng = _prefix(guest_frac=1.0), serving32()
        _turn(eng, res, tokens(60, 40), 10, "X")
        y = tokens(30, 41)
        lg, _, _ = cache.prefill(eng, None, y, "cpu", chunk=CH, resident=res, conv_id="Y")
        assert res._stash is not None, "the second chat was no guest of the first"
        cy = y + _greedy(eng, lg, 30, 120)
        # the server's order at the close (`_close_read`): the promotion, the end read again, the end of that read
        assert res.promote_guest(eng, None, cy, 30, "Y") and res.reread_end(eng, None, cy, 30) == "read"
        assert res.keep_end(eng, None, cy, 30, [], reread=True) == "kept"
        y2 = y + [_other(cy[30])] + tokens(20, 42)
        assert len(y2) < res.valid
        c2, t2 = _turn(eng, res, y2, 6, "Y")
        assert not t2["guest"] and res.owner == "Y", t2
        y3 = y2 + [_other(c2[len(y2)])] + tokens(10, 43)
        _, t3 = _turn(eng, res, y3, 2, "Y")
    assert t3["kind"] == "pend" and t3["reused"] == len(y2), t3


def test_a_turn_that_does_not_hold_the_last_prompt_stays_a_guest_by_size():
    """5.1 test 4: the conversation's own session id is not enough: a request that holds only part of the last
    prompt is a guest by size, as before."""
    res, eng = _prefix(), serving32()
    p = tokens(60, 50)
    _turn(eng, res, p, 200, "A")
    q = p[:20] + [_other(p[20])] + tokens(20, 51)
    assert len(q) < 0.25 * res.valid
    _, t = _turn(eng, res, q, 2, "A")
    assert t["guest"], t


def _blocks(eng, lg, n0: int, chain: bool = True) -> list[int]:
    """A turn's answer as the served loop writes it on this engine: decode steps, a verify block with a rejected
    branch (rows 0, 2 and 3 committed), decode steps, then (`chain`) a chain block whose commit waits and whose
    stream stopped at its second row (the engine ahead of the committed tokens). Greedy where the loop decides."""
    out, pos = [int(lg.reshape(-1, lg.shape[-1])[-1].argmax())], n0
    for _ in range(3):
        lg = eng.forward(torch.tensor([out[-1]]), start=pos)
        pos += 1
        out.append(int(lg.reshape(-1, lg.shape[-1])[-1].argmax()))
    lgt = eng.forward_tree(torch.tensor([out[-1], 5, 7, 9]), [-1, 0, 0, 2], start=pos)
    eng.commit_tree([0, 2, 3])
    pos += 3
    out += [7, 9, int(lgt[3].argmax())]
    for _ in range(2):
        lg = eng.forward(torch.tensor([out[-1]]), start=pos)
        pos += 1
        out.append(int(lg.reshape(-1, lg.shape[-1])[-1].argmax()))
    if chain:
        eng.forward_block(torch.tensor([out[-1], 11, 13, 15]), start=pos)
        out.append(11)
    assert eng.stats["decode_steps"] >= 5 and eng.stats["verify_blocks"] >= 1, eng.stats
    return out


def test_the_end_anchor_and_the_session_snapshot_give_a_cold_reads_tokens():
    """5.1 test 8, made stronger with the end read again (`Capabilities.reread_end`): turn 1 writes its answer by
    decode steps and verify blocks (a tree with a rejected branch, a chain cut back); a turn resumed at the end anchor
    and one resumed from the session snapshot hold a cold read's whole state and logits, bit for bit, and write the
    next 12 greedy tokens of a cold read. On the bf16 model: the served dtype, whose prompt pass gives the same bits at
    every cut on the CPU (the fp32 one does not: its sums move by about 3e-7 with the chunk's rows)."""
    with _prompt_end():
        p1 = tokens(37, 60)
        res, eng = _prefix(), serving()
        lg, _, _ = cache.prefill(eng, None, p1, "cpu", chunk=CH, resident=res, conv_id="A")
        c1 = p1 + _blocks(eng, lg, len(p1))
        e = min(int(eng.kv.length), len(c1))
        _remember(eng, res, c1, len(p1), "A")
        assert res.end is not None and res.end["at"] == e, res.end
        p2 = c1 + tokens(6, 61)
        info: dict = {}
        lg2, reused, _ = cache.prefill(eng, None, p2, "cpu", chunk=CH, resident=res, conv_id="A", info=info)
        assert info["kind"] == "end" and reused == e, (info, reused)
        ref, ref_lg = _cold(p2)
        _same(eng, ref, len(p2), lg2, ref_lg)
        assert res.end.get("exact") is True and res.taint is None, (res.end, res.taint)
        want = _greedy(ref, ref_lg, len(p2), 12)
        assert _greedy(eng, lg2, len(p2), 12) == want
        store = cache.StateStore(BIG, chunk=CH)
        res, eng = _prefix(), serving()
        lg, _, _ = cache.prefill(eng, None, p1, "cpu", chunk=CH, resident=res, conv_id="A", store=store,
                                 checkpoint=True)
        c1 = p1 + _blocks(eng, lg, len(p1), chain=False)
        e = min(int(eng.kv.length), len(c1))
        _remember(eng, res, c1, len(p1), "A", store=store)
        res.clear()                                   # the rows are gone: the store alone holds the turn
        p2 = c1 + tokens(6, 61)
        info = {}
        lg2, reused, _ = cache.prefill(eng, None, p2, "cpu", chunk=CH, resident=res, conv_id="A", store=store,
                                       info=info)
        assert info["kind"] == "session" and reused == e, (info, reused)
        ref, ref_lg = _cold(p2)
        _same(eng, ref, len(p2), lg2, ref_lg)
        assert _greedy(eng, lg2, len(p2), 12) == _greedy(ref, ref_lg, len(p2), 12)
    return f"end and session resumes after verify blocks: the whole state and the logits equal a cold read's, 12 tokens"


def test_decode_written_rows_have_the_prompt_passes_layout():
    """The structure the end read relies on (engine-arch/endanchor design 1.3, passes before and after): rows written
    by decode steps, a tree with a rejected branch and a chain cut back by `trim_pending`, against a prompt read of the
    same ids, on the bf16 model. Exactly equal: the length, the token history, the group-key mirror against the
    indexer keys of the closed groups (`gk[:G] == ik[rr-1::rr][:G]`, as the passes wrote it and after `touched()` and
    the rebuild), and which rows hold a pooled key. Layer 0's latents and token keys of the tail rows (their inputs
    are the embeddings, the same on both passes): every element within 0.0078 (2^-7, one bf16 step in [1, 2): the
    largest difference Gate 0 a saw between two kernel classes). Deeper layers: finite, recorded."""
    from engine.models.hy4_flash.engine import _mirror
    s, ref = serving(), serving()
    ids, k = tokens(60, 81), 37
    s.forward(torch.tensor(ids[:k]), start=0)
    pos = k
    for _ in range(4):
        s.forward(torch.tensor([ids[pos]]), start=pos)
        pos += 1
    s.forward_tree(torch.tensor([ids[pos], _other(ids[pos + 1]), ids[pos + 1], ids[pos + 2]]), [-1, 0, 0, 2],
                   start=pos)
    s.commit_tree([0, 2, 3])
    pos += 3
    s.forward_block(torch.tensor(ids[pos:pos + 4]), start=pos)
    assert s.trim_pending(pos + 2)
    pos += 2
    for _ in range(3):
        s.forward(torch.tensor([ids[pos]]), start=pos)
        pos += 1
    e = pos
    ref.forward(torch.tensor(ids[:e]), start=0, prompt=True)
    cfg = s.cfg
    rr, G = cfg.index_kpool, e // cfg.index_kpool
    assert s.kv.length == ref.kv.length == s.st.length == e, (s.kv.length, ref.kv.length, e)
    assert torch.equal(s.st.hist, ref.st.hist), (s.st.hist, ref.st.hist)

    def pooled(st):
        return [p for p in range(G * rr) if torch.equal(st.ik_all[:, p], st.gk_all[:, p // rr])]
    for st in (s.st, ref.st):
        assert torch.equal(st.gk_all[:, :G], st.ik_all[:, rr - 1::rr][:, :G]), "the passes' mirror"
        st.touched()
        _mirror(st, rr)
        assert torch.equal(st.gk_all[:, :G], st.ik_all[:, rr - 1::rr][:, :G]), "the rebuilt mirror"
    want = list(range(rr - 1, G * rr, rr))
    assert pooled(s.st) == pooled(ref.st) == want, (pooled(s.st), pooled(ref.st))
    tail = range(k, e)
    tok_rows = [p for p in tail if p % rr != rr - 1]
    d_lat = float((s.st.lat_all[0, k:e].float() - ref.st.lat_all[0, k:e].float()).abs().max())
    d_ik = float((s.st.ik_all[0, tok_rows].float() - ref.st.ik_all[0, tok_rows].float()).abs().max())
    assert d_lat <= 2 ** -7 and d_ik <= 2 ** -7, (d_lat, d_ik)
    deeper = []
    for i, li in enumerate(cfg.sparse_layers[1:], start=1):
        a, b = s.st.lat_all[i, :e].float(), ref.st.lat_all[i, :e].float()
        assert torch.isfinite(a).all() and torch.isfinite(s.st.ik_all[i, :e].float()).all(), li
        deeper.append(f"layer {li} {float((a - b).abs().max()):.3g}")
    for li in cfg.window_layers:
        assert torch.isfinite(s.st.win[li].float()).all(), li
    return (f"{e} rows ({e - k} decode-written): length, history, mirror and pooled rows exact; layer 0 within "
            f"{max(d_lat, d_ik):.3g}; deeper latents differ by {', '.join(deeper)} (recorded)")


def test_a_guest_that_wraps_the_rings_leaves_the_chats_anchor_exact():
    """5.1 test 9 (passes before and after): a chat long enough to wrap the window layers' rings and to select
    groups in the sparse layers, a guest whose rows wrap the rings too, then the chat's next turn from an anchor:
    the whole state and the logits equal a cold read's, a selecting row included."""
    res, eng = _prefix(), serving32()
    p1 = tokens(100, 70)
    _turn(eng, res, p1, 10, "A")
    assert len(p1) > eng.cfg.index_topk and len(p1) > eng.cfg.sliding_window
    _, tg = _turn(eng, res, tokens(20, 71), 10, "B")
    assert tg["guest"], tg
    p2 = p1[:90] + [_other(p1[90])] + tokens(19, 72)
    info: dict = {}
    lg, reused, _ = cache.prefill(eng, None, p2, "cpu", chunk=CH, resident=res, conv_id="A", info=info)
    assert info["kind"] == "resident" and reused == 80, (info, reused)
    ref, ref_lg = _cold32(p2)
    _same(eng, ref, len(p2), lg, ref_lg)
    return "resumed at 80 after a guest that wrapped the rings: equal to a cold read"


def test_the_conductors_question_reaches_the_prefill_as_own():
    """Through the handler: a request with the conductor's User-Agent reaches `cache.prefill` with `own` on a
    resident prefix with `chat_first`, keeps no session snapshot and gives its rows back. On a prefix without
    the flag (every other family) nothing of that changes: `own` is not passed and the snapshot is kept."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from engine.penalty import PenaltySpec
    from engine.spec import Relax
    from server import app
    from test_app_loop import FakeTok, Req
    saved = dict(app.STATE)
    seen, real = [], cache.prefill

    def spy(*a, **kw):
        seen.append(bool(kw.get("own")))
        return real(*a, **kw)
    body = {"prompt": "the quick brown fox jumps over the lazy dog and runs", "max_tokens": 6, "stream": False,
            "temperature": 0}
    got = {}
    cache.prefill = spy
    try:
        for label, res in (("chat_first", _prefix()),
                           ("size rule", cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, end=True))):
            srv, store = serving32(256), cache.StateStore(BIG, chunk=CH)
            app.STATE.clear()
            app.STATE.update(engine=srv, drafter=None, k=4, tree=False, sampled_tree="det", relax=Relax(1.0, 1),
                             verbose=False, state_store=store, prefix_cache=True, prefix_chunk=CH, session_cache=True,
                             tok=FakeTok(), device="cpu", max_len=srv.max_len, default_max_tokens=64,
                             pen_spec=PenaltySpec(), model="t", cfg_eos=None, think_budget=0, think_stall=False,
                             reasoning_format="tags", request_timeout=0.0, max_queue=8, queue_timeout=5.0,
                             pattern_stop=None, response_cache=None, suffix_store=None, resident=res)
            del seen[:]
            r = Req("/v1/completions", body)
            r.headers["User-Agent"] = app.OWN_QUESTION_UA
            head, raw = r.response()
            assert head.startswith("HTTP/1.1 200"), (label, head, raw[:400])
            got[label] = (list(seen), sorted({e.kind for e in store._d.values()}), int(res.valid))
    finally:
        cache.prefill = real
        app.STATE.clear()
        app.STATE.update(saved)
    assert got["chat_first"] == ([True], ["prefix"], 0), got
    assert got["size rule"][0] == [False] and "session" in got["size rule"][1] and got["size rule"][2] > 0, got
    return f"{got}"


# ------------------------------------------------------------------ the head mark (SRV-905)
#: a grid wider than the prompts, as the family's 4,096-row grid is beside a 2,345-row question: no line
#: below the head but its mark
WIDE = 512
#: the template's marks on the tiny vocabulary (64 ids; `tokens` draws 4..59): the turn opener, `user`, the end
OPEN, USER, END, SYSTEM, ASSISTANT = 61, 62, 63, 60, 0


class _MarksTok:
    """A tokenizer that names its template's turn marks itself (`Hy4Tokenizer.template_marks`)."""

    def template_marks(self):
        return OPEN, [USER], END


def _question(head_seed: int, user_seed: int, head: int = 300, user: int = 24) -> list[int]:
    """The conductor's question in the template's turns: a system turn of `head` rows (its question and
    options), then a user turn (the state) and the generation prompt."""
    return ([OPEN, SYSTEM] + tokens(head, head_seed) + [END] + [OPEN, USER] + tokens(user, user_seed) + [END]
            + [OPEN, ASSISTANT])


class _marks:
    """The server's view of this family for `_grid_mark`: its engine (the caps) and a tokenizer with marks."""

    def __init__(self, eng, tok=None):
        self.eng, self.tok = eng, tok if tok is not None else _MarksTok()

    def __enter__(self):
        from server import app
        self.app, self.keep = app, dict(app.STATE)
        app.STATE.update(engine=self.eng, tok=self.tok)
        self.marks, app._CONV_MARKS = app._CONV_MARKS, None
        return app

    def __exit__(self, *exc):
        self.app.STATE.clear()
        self.app.STATE.update(self.keep)
        self.app._CONV_MARKS = self.marks
        return False


def test_a_repeated_head_is_read_once():
    """5.1 test 5: two prompts with the same system turn of at least 256 rows, through `_grid_mark` with the
    family's marks: the second resumes at the mark, from the anchor in place and from the store, equal to a
    cold read on the same grid. Before: no plan, no mark, and no line below the grid: `reused == 0`."""
    a, b = _question(80, 81), _question(80, 82)
    store = cache.StateStore(BIG, chunk=WIDE)
    with _marks(serving32()) as app:
        mark = app._grid_mark(None, torch.tensor(a), None)
        assert mark == 303 == app._grid_mark(None, torch.tensor(b), None), mark
        res, eng = _prefix(chunk=WIDE), serving32()
        _turn(eng, res, a, 3, "a", store=store, chunk=WIDE, grid_mark=mark)
        assert 303 in store.report()["boundaries"], store.report()
        info: dict = {}
        lg, reused, _ = cache.prefill(eng, None, b, "cpu", chunk=WIDE, resident=res, info=info, grid_mark=mark)
        assert (info["kind"], reused) == ("resident", 303), (info, reused)
        ref, ref_lg = _cold32(b, chunk=WIDE, grid_mark=mark)
        _same(eng, ref, len(b), lg, ref_lg)
        res, eng, info = _prefix(chunk=WIDE), serving32(), {}
        lg, reused, _ = cache.prefill(eng, None, b, "cpu", chunk=WIDE, resident=res, store=store, info=info,
                                      grid_mark=mark)
        assert (info["kind"], reused) == ("prefix", 303), (info, reused)
        _same(eng, ref, len(b), lg, ref_lg)
    return f"the second prompt forwarded {len(b) - 303} of {len(b)} rows from the anchor and from the store"


def test_the_conductors_question_beside_a_chat_stores_its_head_and_keeps_the_chats_end():
    """5.1 test 6: a chat with a kept end is in place and shares 200 rows (more than one 128-row chunk) with the
    head of the conductor's question, which runs twice as its guest. The first resumes at the chat's anchor and
    stores its head although the chat is tainted (its kept end). The second resumes from that head in the store:
    its stash starts at row 0, not at the chunk line below what it shares (128), so every row the restore writes
    goes back, and the chat keeps its end. The chat's next turn resumes there and writes a cold read's greedy
    tokens. Before: no head was stored, and once one was, the hit dropped the end. With the end read again the chat
    is no longer tainted by its end (a cold read's state), so its head is stored as any other prompt's."""
    ch = 128
    store = cache.StateStore(BIG, chunk=ch)
    q1, q2 = _question(90, 91), _question(90, 92)
    chat = q1[:200] + tokens(30, 93)
    with _marks(serving32()) as app, _prompt_end():
        mark = app._grid_mark(None, torch.tensor(q1), None)
        assert mark == 303 and app._grid_mark(None, torch.tensor(chat), None) is None
        res, eng = _prefix(chunk=ch), serving32()
        c1, _ = _turn(eng, res, chat, 20, "chat", store=store, chunk=ch)
        e = res.end["at"]
        assert res.taint is None and res.end.get("exact") is True and e == len(c1) - 1, (res.taint, res.end)
        _, t1 = _turn(eng, res, q1, 3, "q", own=True, store=store, chunk=ch, grid_mark=mark)
        assert t1["guest"] and (t1["kind"], t1["reused"]) == ("resident", ch), t1
        assert mark in store.report()["boundaries"], store.report()["boundaries"]
        _, t2 = _turn(eng, res, q2, 3, "q", own=True, store=store, chunk=ch, grid_mark=mark)
        assert t2["guest"] and (t2["kind"], t2["reused"]) == ("prefix", mark), t2
        assert res.common(q2) >= ch and res._stash["start"] == 0, (res.common(q2), res._stash["start"])
        assert res.end is not None and res.end["at"] == e, res.end
        p2 = c1 + tokens(5, 94)
        want = _cold_greedy(p2, 6, chunk=ch)
        c3, t3 = _turn(eng, res, p2, 6, "chat", store=store, chunk=ch)
    assert (t3["kind"], t3["reused"]) == ("end", e) and c3[len(p2):] == want, t3
    return f"the second question forwarded {t2['forwarded']} of {len(q2)} rows; the chat resumed at its end ({e})"


def test_two_heads_that_share_256_rows_are_one_prompts():
    """SPD-249's rule, which the head mark keeps as it is: a new head that shares at least 256 leading rows with
    an older one marks the older stale (the same prompt, its notes changed), and a stale head loses its place at
    the back of the store's eviction order. Two conductor questions with a common preamble of 256 tokens or more
    would mark each other stale in turn. The default class and package questions share the classifier prompt up
    to its class list (about 35 tokens with this template's turn marks), so each keeps its head."""
    store = cache.StateStore(BIG, chunk=WIDE)
    h = tokens(300, 140)

    def ask(head, seed):
        return [OPEN, SYSTEM] + head + [END, OPEN, USER] + tokens(10, seed) + [END, OPEN, ASSISTANT]
    first, far, near = ask(h, 143), ask(h[:100] + tokens(200, 142), 144), ask(h[:280] + tokens(20, 141), 145)
    with _marks(serving32()) as app:
        for ids in (first, far, near):
            mark = app._grid_mark(None, torch.tensor(ids), None)
            assert mark == 303, mark
            cache.prefill(serving32(), None, ids, "cpu", store=store, chunk=WIDE, checkpoint=True, grid_mark=mark)
    heads = {tuple(e.tokens): store._head(k, e) for k, e in store._d.items() if e.mark is not None and k[0] == e.mark}
    assert heads == {tuple(first[:303]): False, tuple(far[:303]): True, tuple(near[:303]): True}, heads


def test_the_marks_and_the_mark_of_every_other_family_are_as_before():
    """The marks lookup of a tokenizer without `template_marks` asks for Qwen's names, as before; `_grid_mark` on
    an engine without `head_mark` gives a plan's mark or none, as before; with it, the plan's mark still wins."""
    import types

    from engine.models.adapter import Capabilities
    from server import app

    class Qwen:
        unk_token_id = 0

        def convert_tokens_to_ids(self, t):
            return {"<|im_start|>": 10, "<|im_end|>": 11}.get(t, 0)

        def encode(self, text, add_special_tokens=False):
            return [12]

    class Plain(Qwen):
        def convert_tokens_to_ids(self, t):
            return 0

    assert app._marks_of(Qwen()) == (10, [12], 11) and app._marks_of(Plain()) is None
    assert app._marks_of(_MarksTok()) == (OPEN, [USER], END)
    a = torch.tensor(_question(95, 96))
    plan = types.SimpleNamespace(head_end=40, head_ids=a[:40].tolist())
    other = types.SimpleNamespace(head_end=40, head_ids=[1] * 40)
    for caps, want in ((Capabilities(caches=True, side_slots=True), (None, 40, None, None)),
                       (Hy4FlashServing.caps, (303, 40, 303, None))):
        with _marks(types.SimpleNamespace(caps=caps)):
            got = (app._grid_mark(None, a, None), app._grid_mark(plan, a, None), app._grid_mark(other, a, None),
                   app._grid_mark(None, a, object()))
        assert got == want, (caps, got, want)
    short = torch.tensor([OPEN, SYSTEM] + tokens(100, 97) + [END, OPEN, USER, 9, END])
    with _marks(types.SimpleNamespace(caps=Hy4FlashServing.caps)):
        assert app._grid_mark(None, short, None) is None          # a head under 256 rows: no mark


def _held(store: cache.StateStore, conv_id: str) -> bool:
    """Whether the store holds an entry of the conversation (read under its lock)."""
    with store.lock:
        return any(e.conv_id == conv_id for e in store._d.values())


def test_a_chat_too_short_for_the_park_keeps_its_prefix_beside_a_short_second_chat():
    """SRV-907, the size rule on one resident prefix (the interim one-lane rule that let a second chat the store
    remembers take it is gone with Hy4-flash's side slots, ENG-485). A chat under one chunk above what it shares
    with a short second chat keeps the prefix: the second chat runs as its guest, the chat's end stays, nothing is
    parked, and the chat's next turn resumes at its end with a cold read's tokens. A guest's own answer is never read
    again (its rows go back to the stash), so on this family it keeps no session snapshot: nothing decode-written."""
    store = cache.StateStore(BIG, chunk=64)
    with _prompt_end():
        res, eng = _prefix(park_bytes=BIG, chunk=64), serving32()
        ca1, _ = _turn(eng, res, tokens(40, 130), 10, "A", store=store, chunk=64)
        e = res.end["at"]
        assert e < res.chunk
        b1 = tokens(8, 131)
        cb1, tb1 = _turn(eng, res, b1, 3, "B", store=store, chunk=64)
        assert tb1["guest"] and not _held(store, "B"), tb1
        b2 = b1 + [_other(cb1[8])] + tokens(2, 132)
        assert len(b2) < 0.25 * res.valid
        _, tb2 = _turn(eng, res, b2, 2, "B", store=store, chunk=64)
        assert tb2["guest"] and res.owner == "A" and res.end["at"] == e and res.stats["parks"] == 0, tb2
        p2 = ca1 + tokens(3, 133)
        want = _cold_greedy(p2, 4, chunk=64)
        c2, ta2 = _turn(eng, res, p2, 4, "A", store=store, chunk=64)
    assert (ta2["kind"], ta2["reused"]) == ("end", e) and c2[len(p2):] == want, ta2


def test_a_second_chat_that_ran_as_a_guest_resumes_from_its_head():
    """SRV-907, with the end read again. A second chat that ran as a guest keeps no session snapshot on this family
    (its answer was never read again: decode-written rows are never resumed from), so its next turn does not resume
    from one. It resumes from the head its guest prefill stored (a prompt read's grid line), is a guest again, and
    writes a cold read's greedy tokens; the conversation in place stays, nothing is parked. Before, it resumed from
    the guest's decode-written session snapshot and took the prefix."""
    store = cache.StateStore(BIG, chunk=CH)
    with _prompt_end():
        res, eng = _prefix(park_bytes=BIG), serving32()
        _turn(eng, res, tokens(60, 134), 100, "A", store=store)
        b1 = tokens(20, 135)
        cb1, tb1 = _turn(eng, res, b1, 3, "B", store=store)
        assert tb1["guest"], tb1
        assert not any(x.kind == "session" and x.conv_id == "B" for x in store._d.values())
        b2 = cb1 + tokens(3, 136)
        want = _cold_greedy(b2, 2)
        cb2, tb2 = _turn(eng, res, b2, 2, "B", store=store)
    got = (tb2["kind"], tb2["reused"], tb2["guest"], res.owner, res.stats["parks"])
    assert got == ("prefix", 16, True, "A", 0) and cb2[len(b2):] == want, (got, cb2[len(b2):], want)


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
