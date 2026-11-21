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

1, 2 and 4 need no torch; 3 starts server/app.py --fake-engine (CPU torch).
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
              "answer_reserve", "drafter", "round_runner", "family", "queue_timeout_s", "request_timeout_s"):
        assert k in got, (k, sorted(got))
    return f"{len(got)} pairs, the line before the listening one"


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


def _after_the_pool(ceiling: float, peak: float, own: bool) -> tuple:
    """(ceiling, peak, fixed) in GiB of the memory policy after `_kv_after_load`, on a stand-in engine whose slot 0
    window is the KV pool's (WINDOW of addresses). The start's limits are `ceiling`/`peak`: a configured figure
    (serve.env, a weight profile), or with `own` the model's builder's own from its load (`_build_with_limits`, as
    Hy4-flash's serving.memory_limits sets them)."""
    import test_forward_tree  # noqa: F401  (the CPU environment before the server imports)
    from engine import cache, kvpool
    from server import app
    a = types.SimpleNamespace(mem_ceiling_gib=70.0, mem_peak_gib=77.0, store_entry_gb=4.0)
    saved = {k: app.STATE.get(k) for k in ("mem", "engine", "serve_flags", "limits_from_load")}
    pool0, reclaim0 = kvpool.POOL, app._add_kv_reclaimers
    try:
        app.STATE.pop("limits_from_load", None)
        app.STATE["serve_flags"] = {"mem_ceiling_gib": 70.0, "mem_peak_gib": 77.0, "store_entry_gb": 4.0}
        if own:
            def build(a, ctx):
                a.mem_ceiling_gib, a.mem_peak_gib = ceiling, peak
                return {}
            with contextlib.redirect_stdout(io.StringIO()):
                app._build_with_limits(build, a, None)
        else:
            a.mem_ceiling_gib, a.mem_peak_gib = ceiling, peak
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
        assert _after_the_pool(c, p_, own=False) == (round(c - w, 4), round(p_ - w, 4), -w), name
    return f"{len(figs)} configured figures, each less the window ({w} GiB)"


def test_a_models_own_limit_from_its_load_keeps_the_window_it_never_held():
    """Hy4-flash's builder sets its pack's own limits from what the allocator holds after its load, 12 GiB over it
    (serving.memory_limits), measured with slot 0's window already the pool's: B's start on 70c250f6 held 68.4 GiB
    on hy4-3.56 (72.6 before the lanes, less the 4.12 GiB window) and set 81/88. `_kv_after_load` took the window
    off again (76.88/83.88 before the side slots): the caches had about 7.9 GiB of the 12. A limit the model set from
    its load now keeps it (81/88, then the side slots' 3.09: 84.09/91.09); the policy's fixed bytes still count the
    window, for a figure the residency document sets later. A configured figure is as before (the test above)."""
    w = round(WINDOW / GIB, 4)
    assert _after_the_pool(81.0, 88.0, own=True) == (81.0, 88.0, -w), _after_the_pool(81.0, 88.0, own=True)
    assert _after_the_pool(72.0, 79.0, own=True) == (72.0, 79.0, -w)
    assert _after_the_pool(81.0, 88.0, own=False) == (round(81 - w, 4), round(88 - w, 4), -w)
    return "own 81/88 and 72/79 keep the window; the same figures configured lose it"

if __name__ == "__main__":
    from scriptmode import run
    run(globals())
