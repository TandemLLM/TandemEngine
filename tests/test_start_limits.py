"""OPS-310: the engine names every limit it runs with, once, at the start (benchmarking rule 5: every limit knob in
a result's record, defaults included, so a configuration can never hide a cap).

The test matrix:

  1  the line: `[limits] KEY=value ...` with the head's keys first (lanes, windows, the pool), then every key of
     the limits table (server/memlimits.py LIMITS) as it runs: a live key as the residency document applied it,
     PREFIX_CHUNK the model's own grid, MEM_CEILING_GIB and MEM_PEAK_GIB the memory policy's own (the side slots'
     share included) beside the values the start set, `off` without a policy; then the tail; None is `-`, a bool
     on or off, no value carries a space                       test_the_line_names_every_limit_as_it_runs
  2  a model family's own knobs are the ones its package's code reads (Hy4-flash: the Engram tables' cold
     reader and its 2 GiB row cache), with the values in force; a family without a package has none
                                                               test_a_familys_own_knobs_are_its_packages
  3  a fake engine with three lanes, a shorter side window, an effort word and the round runner's pipe off
     prints the line once, before it listens, with those values and every table key; it names what runs, not
     what was asked (`--response-cache` builds no store on the fake engine: off)
                                                               test_a_fake_engine_names_its_limits_before_it_listens
  4  the response store: off by default in every model's registry entry, in ops/serve.env, in ops/start.sh and
     in docs/caches.md, and opt-in by RESPONSE_CACHE=on alone; Hy4-flash's profiles leave it to serve.env
                                                               test_the_response_store_is_off_by_default_everywhere
  5  OPS-308, one accounting rule, every figure a configured one: on both Hy4-flash packs (their profiles'
     85/92, 8 lanes and one lane with the pool off), the 27B and Flash-Next, the [kv-pool], [parallel], [limits]
     and residency lines and the policy name one ceiling and peak, a request's refresh cuts nothing again, a
     dashboard figure is enforced live as the next start enforces it, and the Hy4-flash loads' idle gate sheds
     nothing; Hy4-flash takes its limits from its profiles, its builder sets none
                                     test_one_accounting_rule_from_the_start_through_every_refresh
                                     test_hy4_flash_takes_its_limits_from_its_profiles
  6  the end read again: `end_reread` in the line and in /health's cache block, on, off (QWEN38_END_REREAD=0) or
     `-` for a family without `Capabilities.reread_end`         test_the_end_read_is_named_in_the_line_and_health

1, 2 and 4 need no torch; 3 starts server/app.py --fake-engine (CPU torch); 5 runs the server's start steps on CPU.
Run: python tests/test_start_limits.py
"""

from __future__ import annotations

import contextlib
import glob
import io
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from server import memlimits  # noqa: E402

GIB = 1 << 30
#: a `[limits]` line's pairs
PAIR = re.compile(r"(\S+?)=(\S+)")


def _pairs(line: str) -> dict:
    assert line.startswith("[limits] "), line
    return dict(PAIR.findall(line[len("[limits] "):]))


def test_the_line_names_every_limit_as_it_runs():
    args = {"mem_ceiling_gib": 85.0, "mem_peak_gib": 92.0, "reserve_slack_gib": 2.0, "resident_gb": 4.0,
            "resident_stash_gb": 1.0, "resident_guest_max": 16384, "resident_pin_gb": 2.0, "resident_park_gb": 8.0,
            "cache_budget_gb": 2.0, "store_entry_gb": 0.25, "image_cache_mb": 512, "suffix_store_mb": 256,
            "prefix_chunk": 2048, "narrow_arm": "off", "max_len": 262144}
    mem = types.SimpleNamespace(on=True, ceiling=int(88.1 * GIB), peak=int(95.1 * GIB))
    state = {"args": args, "mem": mem, "limits_applied": {"CACHE_GB": 3.0}, "anchor_grid": 4096}
    line = memlimits.start_line({"lanes": 8, "slot0_window": 262144, "side_window": 262144, "kv_pool": True},
                                state, {"response_store": False, "effort_default": "medium:8192", "drafter": None})
    got = _pairs(line)
    assert list(got)[:4] == ["lanes", "slot0_window", "side_window", "kv_pool"], list(got)
    assert set(memlimits.KEYS) <= set(got), sorted(set(memlimits.KEYS) - set(got))
    assert got["MEM_CEILING_GIB"] == "88.1" and got["MEM_PEAK_GIB"] == "95.1", got
    assert got["MEM_CEILING_GIB_SET"] == "85" and got["MEM_PEAK_GIB_SET"] == "92", got
    assert got["CACHE_GB"] == "3" and got["PREFIX_CHUNK"] == "4096" and got["NARROW_ARM"] == "0", got
    assert got["kv_pool"] == "on" and got["response_store"] == "off" and got["drafter"] == "-", got
    assert got["effort_default"] == "medium:8192" and got["MAX_LEN"] == "262144", got
    assert len(line.split()) == 1 + len(got), line              # one pair a key, no value with a space
    # no memory policy: the ceiling and the peak say off, the values the start set stay named
    off = _pairs(memlimits.start_line({}, {"args": args}, {}))
    assert off["MEM_CEILING_GIB"] == "off" and off["MEM_PEAK_GIB"] == "off" and off["MEM_CEILING_GIB_SET"] == "85"
    return f"{len(got)} pairs, the {len(memlimits.KEYS)} table keys among them"


def test_a_familys_own_knobs_are_its_packages():
    from engine.models import REGISTRY
    from engine.settings import SETTINGS
    described = SETTINGS.describe()
    hy4 = memlimits.family_knobs(described, REGISTRY["hy_v4_flash"].package)
    assert {"QWEN38_HY4_TABLE", "QWEN38_HY4_TABLE_CACHE_GB"} <= set(hy4), sorted(hy4)
    assert all(k.startswith("QWEN38_HY4_") for k in hy4), sorted(hy4)
    want = os.environ.get("QWEN38_HY4_TABLE_CACHE_GB", described["HY4_TABLE_CACHE_GB"]["default"])
    assert hy4["QWEN38_HY4_TABLE_CACHE_GB"] == want, hy4
    fn = memlimits.family_knobs(described, REGISTRY["qwen4_exp"].package)
    assert fn and not set(fn) & set(hy4), sorted(fn)[:5]
    assert memlimits.family_knobs(described, None) == {}
    return f"Hy4-flash {len(hy4)}, Flash-Next {len(fn)}, none without a package"


def _torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def test_a_fake_engine_names_its_limits_before_it_listens():
    if not _torch():
        return "skipped: no torch"
    home = tempfile.mkdtemp(prefix="limits-home-")
    launch = ("import sys; sys.modules['triton'] = None; sys.modules['triton.language'] = None;"
              "sys.argv = ['server/app.py'] + sys.argv[1:]; import runpy;"
              "runpy.run_path('server/app.py', run_name='__main__')")
    cmd = [sys.executable, "-c", launch, "--fake-engine", "--port", "0", "--host", "127.0.0.1",
           "--usage-ledger", os.path.join(home, "ledger.sqlite3"), "--parallel", "3", "--max-len", "65536",
           "--side-max-len", "32768", "--response-cache", "--reasoning-effort", "low", "--max-queue", "5"]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home, "PYTHONPATH": ROOT,
           "CUDA_VISIBLE_DEVICES": "", "LANG": "C.UTF-8", "QSE_RUNNER_PIPE": "0",
           "QSE_ADMIN_TOKEN": "limits-admin-token-0123456789abcdef"}
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    seen, end = [], time.time() + 180
    try:
        while time.time() < end:
            line = p.stdout.readline()
            if not line:
                break
            seen.append(line.rstrip("\n"))
            if line.startswith("[server] listening on "):
                break
    finally:
        p.send_signal(signal.SIGTERM)
        try:
            p.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate(timeout=10)
    assert seen and seen[-1].startswith("[server] listening on "), "\n".join(seen[-30:])
    limits = [ln for ln in seen if ln.startswith("[limits] ")]
    assert len(limits) == 1, limits
    got = _pairs(limits[0])
    assert seen.index(limits[0]) == len(seen) - 2, seen[-3:]       # the last line before it listens
    assert set(memlimits.KEYS) <= set(got), sorted(set(memlimits.KEYS) - set(got))
    # the line names what runs, not what was asked: the fake engine builds no response store under the flag
    want = {"lanes": "3", "slot0_window": "65536", "side_window": "32768", "MAX_LEN": "65536",
            "response_store": "off", "effort_default": "low", "effort_source": "server", "runner_pipe": "off",
            "max_queue": "5", "drafter_built": "off"}
    assert {k: got.get(k) for k in want} == want, {k: got.get(k) for k in want}
    for k in ("kv_pool", "MEM_CEILING_GIB", "MEM_PEAK_GIB", "MEM_CEILING_GIB_SET", "think_budget",
              "answer_reserve", "drafter", "round_runner", "family", "queue_timeout_s", "request_timeout_s",
              "end_reread"):
        assert k in got, (k, sorted(got))
    assert got["end_reread"] == "-", got["end_reread"]             # the fake engine keeps its end as before
    return f"{len(got)} pairs, the line before the listening one"


def test_the_end_read_is_named_in_the_line_and_health():
    """The end read again (`Capabilities.reread_end`, QWEN38_END_REREAD): the `[limits]` line and /health's cache
    block say `end_reread`: on for a family that reads it, off with the knob at 0, `-` (None) for one without it."""
    if not _torch():
        return "skipped: no torch"
    import contextlib as cl
    import io as io_
    from engine import cache
    from engine.models.adapter import Capabilities
    from server import app, fake_engine
    a = app.parser().parse_args(["--fake-engine", "--fake-tps", "200000", "--max-len", "65536"])
    saved = dict(app.STATE)
    keep = cache.END_REREAD
    app.STATE.clear()
    try:
        with cl.redirect_stdout(io_.StringIO()):
            fake_engine.load(app, a)
        eng = app.STATE["engine"]
        got = {}
        for label, caps, knob in (("none", None, True), ("on", True, True), ("off", True, False)):
            cache.END_REREAD = knob
            if caps is not None:
                eng.caps = Capabilities(caches=True, side_slots=False, prompt_rows=True, reread_end=True)
            line = _pairs(app._limits_line(a))
            got[label] = (line["end_reread"], app.cache_stats()["end_reread"])
        assert got == {"none": ("-", None), "on": ("on", True), "off": ("off", False)}, got
    finally:
        cache.END_REREAD = keep
        app.STATE.clear()
        app.STATE.update(saved)
    return "end_reread: '-' without the capability, on, off with the knob at 0; /health's cache block the same"


def test_the_response_store_is_off_by_default_everywhere():
    import yaml
    for path in sorted(glob.glob(os.path.join(ROOT, "models", "*", "model.yaml"))):
        with open(path) as fh:
            m = yaml.safe_load(fh)
        c = [x for x in m["components"] if x["id"] == "response_cache"]
        assert len(c) == 1 and c[0]["default"] is False, (path, c)
        assert c[0]["knobs"] == {"RESPONSE_CACHE": {"on": "on", "off": "off"}}, (path, c[0]["knobs"])
    with open(os.path.join(ROOT, "ops", "serve.env")) as fh:
        assert re.findall(r"^RESPONSE_CACHE=(\S*)", fh.read(), re.M) == ["off"]
    with open(os.path.join(ROOT, "ops", "start.sh")) as fh:
        assert '[ "${RESPONSE_CACHE:-off}" = "on" ] && MEM_FLAGS="$MEM_FLAGS --response-cache"' in fh.read()
    for path in glob.glob(os.path.join(ROOT, "ops", "profiles", "*hy4*.env")):
        with open(path) as fh:
            assert not re.search(r"^RESPONSE_CACHE=", fh.read(), re.M), path
    with open(os.path.join(ROOT, "docs", "caches.md")) as fh:
        assert "| response cache | the tokens of an identical greedy request | memory | off |" in fh.read()
    return "off in every model.yaml, serve.env, start.sh and the caches docs; on only by RESPONSE_CACHE=on"



#: slot 0's window as the pool reserves it (B's start on 70c250f6: "-4.12 GiB (slot 0's window is the pool's now)")
WINDOW = int(4.12 * GIB)


def _after_the_pool(ceiling: float, peak: float) -> tuple:
    """(ceiling, peak, fixed) in GiB of the memory policy after `_kv_after_load`, on a stand-in engine whose slot 0
    window is the KV pool's (WINDOW of addresses), for a start whose limits are `ceiling`/`peak`."""
    import test_forward_tree  # noqa: F401  (the CPU environment before the server imports)
    from engine import cache, kvpool
    from server import app
    a = types.SimpleNamespace(mem_ceiling_gib=ceiling, mem_peak_gib=peak, store_entry_gb=4.0)
    saved = {k: app.STATE.get(k) for k in ("mem", "engine")}
    pool0, reclaim0 = kvpool.POOL, app._add_kv_reclaimers
    try:
        mem = cache.Budget(int(a.mem_ceiling_gib * GIB), int(a.mem_peak_gib * GIB), 2 * GIB,
                           probe=lambda: (0, 0), empty=lambda: None)
        bufs = [types.SimpleNamespace(nbytes=WINDOW, page=2 << 20)]
        app.STATE.update(mem=mem, engine=types.SimpleNamespace(kv=types.SimpleNamespace(
            pages=types.SimpleNamespace(bufs=bufs), max_len=262144)))
        kvpool.POOL = types.SimpleNamespace(hard_floor=10 * GIB, margin_rows=256)
        app._add_kv_reclaimers = lambda pool: None
        with contextlib.redirect_stdout(io.StringIO()):
            app._kv_after_load(a)
        return round(mem.ceiling / GIB, 4), round(mem.peak / GIB, 4), round(mem.fixed / GIB, 4)
    finally:
        kvpool.POOL, app._add_kv_reclaimers = pool0, reclaim0
        for k, v in saved.items():
            if v is None:
                app.STATE.pop(k, None)
            else:
                app.STATE[k] = v


def _configured_figures() -> dict:
    """Every configured ceiling and peak: serve.env's (the 27B's and every profile that sets none) and each weight
    profile's own MEM_CEILING_GIB / MEM_PEAK_GIB."""
    def figures(path):
        text = open(path).read()
        c = re.search(r"(?m)^MEM_CEILING_GIB=([0-9.]+)", text)
        p_ = re.search(r"(?m)^MEM_PEAK_GIB=([0-9.]+)", text)
        return (float(c.group(1)), float(p_.group(1))) if c and p_ else None
    out = {"serve.env": figures(os.path.join(ROOT, "ops", "serve.env"))}
    for f in sorted(glob.glob(os.path.join(ROOT, "ops", "profiles", "*.env"))):
        got = figures(f)
        if got:
            out[os.path.basename(f)] = got
    return out


def test_a_configured_limit_loses_slot_0s_window_to_the_pool():
    """The KV pool holds slot 0's window outside the allocator, which the memory policy measures, so a configured
    ceiling and peak (serve.env's 70/77, Flash-Next 3.05's 66/77 with "slot 0's KV at 262,144 tokens (7.58)" in it,
    every profile's) go down by the window: the caches keep the room they were sized for. Recorded on 5c89764b:
    each figure less the window, the policy's fixed bytes less it too."""
    figs = _configured_figures()
    assert figs["serve.env"] == (70.0, 77.0) and figs["flashnext-3.05.env"] == (66.0, 77.0), figs
    w = round(WINDOW / GIB, 4)
    for name, (c, p_) in figs.items():
        assert _after_the_pool(c, p_) == (round(c - w, 4), round(p_ - w, 4), -w), name
    return f"{len(figs)} configured figures, each less the window ({w} GiB)"


def test_hy4_flash_takes_its_limits_from_its_profiles():
    """OPS-308: Hy4-flash's ceiling and peak are its weight profiles' (ops/profiles/hy4-<bpw>.env), as every
    family's: both packs 85 / 92 GiB (both reserve 72.50 GiB after an 8-lane load and 73.40 after a one-lane one with
    the pool off, hold 1b). Its builder sets no memory limit (server/app.py `BUILDER_LIMITS`: the state store's
    entry cap alone) and its serving module computes none: until then it set them from what its load held, and
    the limits moved three times in two days."""
    from engine.models.hy4_flash import serving
    from server import app
    figs = _configured_figures()
    assert figs.get("hy4-3.56.env") == figs.get("hy4-3.08.env") == (85.0, 92.0), figs
    assert app.BUILDER_LIMITS == ("store_entry_gb",), app.BUILDER_LIMITS
    gone = [n for n in ("memory_limits", "own_limits", "held_after_load", "SERVER_LIMITS") if hasattr(serving, n)]
    assert not gone, gone
    return "hy4-3.56 and hy4-3.08: 85 / 92 GiB from their profiles; the builder sets the store's entry cap alone"


#: OPS-308: the shapes the one accounting rule is locked on: the start's configured ceiling and peak (GiB; Hy4-flash's
#: from its profiles), slot 0's window the KV pool holds (GiB; None: the pool off), the side slots' share (GiB; 0: one
#: lane), and the allocator after the load (allocated, reserved GiB) where a log has it. Hy4-flash: hold 1b on
#: aa596e81 ("ceiling and peak +3.22 GiB for the side slots", 8 lanes "after the load: 62.66 GiB allocated, 72.50
#: GiB reserved" on hy4-3.08, 71.29 and 72.50 on hy4-3.56; one lane with the pool off 64.02 and 73.40, 72.67 and
#: 73.40). The 27B: serve.env's 70/77; Flash-Next 3.05: its profile's 66/77 with slot 0's 7.58 GiB of KV; their window
#: and side figures are stand-ins
SHAPES = {"hy4-3.08": dict(profile="hy4-3.08.env", window=4.12, side=3.22, load=(62.66, 72.50)),
          "hy4-3.56": dict(profile="hy4-3.56.env", window=4.12, side=3.22, load=(71.29, 72.50)),
          "hy4-3.08 one lane": dict(profile="hy4-3.08.env", window=None, side=0.0, load=(64.02, 73.40)),
          "hy4-3.56 one lane": dict(profile="hy4-3.56.env", window=None, side=0.0, load=(72.67, 73.40)),
          "27b": dict(profile="serve.env", window=5.0, side=1.5, load=None),
          "flashnext-3.05": dict(profile="flashnext-3.05.env", window=7.58, side=3.0, load=None)}
NUM = r"([0-9]+(?:\.[0-9]+)?)"


def _served(shape: dict, docs: tuple = (None,), at_start: dict | None = None) -> dict:
    """One server of `shape` through its start and requests, as server/app.py runs them: the start's limits (its
    profile's figures, `_configured_figures`), the residency document's figures over them before the load
    (`at_start`, `_limits_at_start`: `overrides_at_start`), the memory policy, the pool step (`_kv_after_load`) or the
    window step without the pool, the side slots (`_side_slots_policy`), the start's `[limits]` line, then one
    request's residency refresh a document of `docs` (each a document's `engine` section, None for none) and its gate
    on the load's allocator, the caches a few MB each. Returns every printed ceiling and peak, the policy's own after
    each refresh, and what the gates shed."""
    import test_forward_tree  # noqa: F401  (the CPU environment before the server imports)
    from engine import cache, kvpool
    from server import app
    from server import ext as ext_mod
    saved = {k: app.STATE.get(k) for k in ("mem", "engine", "serve_flags")}
    pool0, reclaim0, doc0 = kvpool.POOL, app._add_kv_reclaimers, ext_mod.document
    figs = _configured_figures().get(shape["profile"])
    assert figs, f"{shape['profile']} names no MEM_CEILING_GIB / MEM_PEAK_GIB"
    a = types.SimpleNamespace(mem_ceiling_gib=figs[0], mem_peak_gib=figs[1], store_entry_gb=1.0,
                              reserve_slack_gib=2.0, max_len=262144)
    alloc, res = [int(x * GIB) for x in (shape["load"] or (0.0, 0.0))]
    box = {"alloc": alloc, "res": res}
    shed, out = [], {"figures": figs}
    try:
        app.STATE["serve_flags"] = dict(vars(a))                  # `_limits_at_start`: the flags as given
        if at_start:
            memlimits.overrides_at_start(a, {"engine": dict(at_start)})
        with contextlib.redirect_stdout(io.StringIO()):
            app._build_with_limits(lambda a_, ctx: {}, a, None)
        mem = cache.Budget(int(a.mem_ceiling_gib * GIB), int(a.mem_peak_gib * GIB), int(a.reserve_slack_gib * GIB),
                           probe=lambda: (box["alloc"], box["res"]), empty=lambda: None)
        for lv in ("store", "anchors", "end", "stash"):
            def fn(safe, _lv=lv):
                shed.append(_lv)
                box["alloc"] -= 17 << 20       # a cache's bytes leave what is allocated; the reserve keeps them
                return 17 << 20
            mem.register(lv, fn)
        pages = (types.SimpleNamespace(bufs=[types.SimpleNamespace(nbytes=int(shape["window"] * GIB), page=2 << 20)])
                 if shape["window"] else None)
        app.STATE.update(mem=mem, engine=types.SimpleNamespace(kv=types.SimpleNamespace(pages=pages, max_len=262144),
                                                               cost=lambda: types.SimpleNamespace(
                                                                   view_bytes_per_token=16896)))
        kvpool.POOL = types.SimpleNamespace(hard_floor=10 * GIB, margin_rows=256) if shape["window"] else None
        app._add_kv_reclaimers = lambda pool: None
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            app._kv_after_load(a)
            if shape["side"]:
                app._side_slots_policy(int(shape["side"] * GIB) - (1 << 20), int(shape["side"] * GIB))
        text = log.getvalue()
        m = re.search(r"\[kv-pool\] memory policy: .*\): " + NUM + " / " + NUM + " GiB", text)
        out["kv_pool"] = tuple(map(float, m.groups())) if m else None
        m = re.search(r"\[parallel\] memory policy: ceiling and peak \+" + NUM + r" GiB for the side slots \("
                      + NUM + " / " + NUM + r" GiB\)", text)
        out["parallel"] = tuple(map(float, m.groups()[1:])) if m else None
        state = {"mem": mem, "serve_flags": app.STATE["serve_flags"], "args": dict(vars(a))}
        got = _pairs(memlimits.start_line({}, state, {}))
        out["limits"] = (float(got["MEM_CEILING_GIB"]), float(got["MEM_PEAK_GIB"]))
        out["start"] = (mem.ceiling, mem.peak)
        out["refresh"], out["residency"] = [], []
        for v, doc in enumerate(docs, start=6):
            ext_mod.document = lambda st, name, _v=v, _d=doc: ((_v, {"engine": dict(_d or {})})
                                                                if name == "residency" else None)
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                memlimits.refresh(state)
            m = re.search(r"the memory policy enforces " + NUM + " / " + NUM + " GiB", log.getvalue())
            out["residency"].append(tuple(map(float, m.groups())) if m else None)
            out["refresh"].append((mem.ceiling, mem.peak))
            if shape["load"]:
                mem.gate(38)
        out["shed"], out["over_ceiling"] = shed, mem.stats["over_ceiling"]
        return out
    finally:
        kvpool.POOL, app._add_kv_reclaimers, ext_mod.document = pool0, reclaim0, doc0
        for k, val in saved.items():
            if val is None:
                app.STATE.pop(k, None)
            else:
                app.STATE[k] = val


def test_one_accounting_rule_from_the_start_through_every_refresh():
    """OPS-308: one rule for the memory limits (server/memlimits.py `limit_of`): the policy enforces a figure plus
    its fixed bytes (slot 0's window off with the KV pool, the side slots' share on), at the start and at every
    refresh, and every line that names the ceiling and peak prints the policy's own. Locked on both Hy4-flash packs at
    8 lanes and at one lane with the pool off (their profiles' 85/92), the 27B and Flash-Next 3.05:

      * the [kv-pool] line plus the side slots' share, the [parallel] line, the [limits] line, the residency
        refresh's line and the policy after the refresh name one ceiling and one peak;
      * the first request's refresh (a document that sets no limit: every key falls back to its start flag) leaves
        the policy as it started: no figure is cut again;
      * a document's figure is enforced live as the next start enforces it (`overrides_at_start` makes it a flag),
        and a document that stops setting it gives back the start's figure;
      * on the Hy4-flash loads hold 1b read, the gate after the refresh sheds nothing (84.10 GiB enforced over the
        8-lane 72.50 reserved, 85 over the one-lane 73.40)."""
    gib2 = lambda b: round(b / GIB, 2)                        # noqa: E731
    for name, sh in SHAPES.items():
        w, s = sh["window"] or 0.0, sh["side"]
        got = _served(sh, docs=(None,))
        c, p = got["figures"]
        want = (round(c - w + s, 2), round(p - w + s, 2))
        enforced = tuple(gib2(x) for x in got["refresh"][0])
        assert got["refresh"][0] == got["start"], (name, "the refresh moved the policy", got)
        if sh["window"]:
            assert tuple(round(x + s, 2) for x in got["kv_pool"]) == want, (name, got)
        else:
            assert got["kv_pool"] is None, (name, got)
        assert (got["parallel"] or want) == got["limits"] == want, (name, got)
        assert got["residency"][0] == enforced == want, (name, got)
        if sh["load"]:
            assert got["shed"] == [] and got["over_ceiling"] == 0, (name, "the idle gate shed", got)
        # live against a restart with the same document, both keys set
        doc = {"MEM_CEILING_GIB": c - 1.5, "MEM_PEAK_GIB": p - 1.5}
        live = _served(sh, docs=(None, doc, None))
        again = _served(sh, docs=(doc,), at_start=doc)
        assert live["refresh"][1] == again["start"] == again["refresh"][0], (name, live, again)
        assert gib2(live["refresh"][1][0]) == round(c - 1.5 - w + s, 2), (name, live)
        assert live["refresh"][2] == live["start"], (name, "the document's figure stayed after it was dropped", live)
    return f"{len(SHAPES)} shapes: every printed ceiling and peak is the policy's, through the refresh and a restart"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
