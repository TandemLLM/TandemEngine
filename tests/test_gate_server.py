"""tools/gate/server.py and the runner's plan, on a CPU with no server: the arm trees, the port rule,
the arm's setup.env and serve.env, the knobs a report records, the order of the arms, the packs.

Run: python tests/test_gate_server.py
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tools.gate import server  # noqa: E402
from tools.gate.hw.fake import Fake  # noqa: E402
from tools.gate.hw.generic import Generic  # noqa: E402
from tools.gate.hw.base import Fingerprint  # noqa: E402
from tools.gate.runner import arm_env, arms_of, load_packs, plan_of  # noqa: E402
from tools.row3 import code_hash  # noqa: E402

HEALTH = {
    "status": "ok", "version": "0.15.2+flashnext-3.05", "max_len": 262144, "max_queue": 8,
    "weights": {"name": "flashnext-3.05", "weights": "/home/x/flashnext-packs/3.05", "format": "exl3",
                "self_check": {"ok": True}, "corrector": {"loaded": True}},
    "memory": {"reserved_gb": 57.8, "ceiling_gb": 64.82, "peak_gb": 75.82},
    "cache": {"prefix_cache": True, "prefix_chunk": 4096, "suffix_store": None, "response_cache": None,
              "state_store": {"budget": 2147483648}},
    "parallel": {"n": 8, "requested": 8, "max_running": 8, "max_running_reason": None,
                 "slots": [{"max_len": 262144}] * 8, "admission": {"floor_gib": 10.0}},
}


def _git_repo() -> str | None:
    r = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def test_tree_is_the_commit_and_a_patch_moves_its_hash():
    sha = _git_repo()
    if not sha:
        print("skipped: not a git checkout")
        return
    d = Path(tempfile.mkdtemp(prefix="gate-tree-"))
    h = server.make_tree(ROOT, sha, d / "a")
    assert h == code_hash(d / "a") and (d / "a" / "tools" / "row3.py").exists()
    p = d / "x.patch"
    p.write_text("--- a/VERSION\n+++ b/VERSION\n@@ -1 +1 @@\n-"
                 + (d / "a" / "VERSION").read_text() + "+9.9.9\n")
    h2 = server.make_tree(ROOT, sha, d / "b", str(p))
    assert (d / "b" / "VERSION").read_text() == "9.9.9\n"
    assert h2 == h                       # VERSION is outside the code folders the hash covers
    src = (d / "a" / "tools/row3.py").read_text().splitlines(True)
    p.write_text("".join(difflib.unified_diff(src, ["# planted\n"] + src, "a/tools/row3.py", "b/tools/row3.py")))
    assert server.make_tree(ROOT, sha, d / "c", str(p)) != h


def test_ports_stay_at_or_above_the_floor_and_skip_busy_ones():
    try:
        server.free_port(8000, set())
        raise AssertionError("a port under 8100 was accepted")
    except SystemExit as exc:
        assert "never beside" in str(exc)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    busy = s.getsockname()[1]
    try:
        got = server.free_port(busy, set(), floor=1024)
        assert got != busy and got > busy
        assert server.free_port(busy, {busy + 1}, floor=1024) not in (busy, busy + 1)
    finally:
        s.close()


def test_the_arm_setup_env_and_serve_env():
    d = Path(tempfile.mkdtemp(prefix="gate-arm-"))
    tree = d / "tree"
    (tree / "ops").mkdir(parents=True)
    (tree / "ops" / "serve.env").write_text("HOME_DIR=${HOME}\nREPO=${HOME}/qwen38-spark-engine\nPORT=8000\n")
    box = d / "box-setup.env"
    box.write_text("QSE_PARALLEL='8'\nPORT='8000'\nRESPONSE_CACHE='on'\n")
    local = d / "local.env"
    local.write_text("PY=/opt/py\n")
    src = d / "corrector.bin"
    src.write_bytes(b"x")
    (d / "community").mkdir()
    (d / "community" / "a.env").write_text("A=1\n")
    cond = d / "conductor"
    cond.mkdir()
    (cond / "config.json").write_text("{}")
    (cond / "decisions-1.jsonl").write_text("secret")
    (cond / "x.token").write_text("secret")
    site = {"setup_env": str(box), "local_env": str(local), "conductor_from": str(cond),
            "copies": {"ops/correctors/c.bin": str(src), "run/community": str(d / "community")},
            "links": {"corpus": str(d)}}
    pack = {"launcher": "ops", "profile": "flashnext-3.05"}
    ol = server.OpsLauncher(Generic(Fingerprint(memory_gib=48)), pack, site)
    ol.prepare(tree)
    serve = (tree / "ops" / "serve.env").read_text()
    assert f"REPO={tree}\n" in serve and "qwen38-spark-engine" not in serve
    assert (tree / "ops" / "active-profile").read_text() == "flashnext-3.05\n"
    assert (tree / "run" / "local.env").read_text() == "PY=/opt/py\n"
    assert oct((tree / "run").stat().st_mode)[-3:] == "700"
    assert (tree / "ops/correctors/c.bin").read_bytes() == b"x" and (tree / "corpus").is_symlink()
    assert (tree / "run/community/a.env").read_text() == "A=1\n"
    arm = server.Arm("head", "plain", tree, 8105, d / "home",
                     {"QSE_PARALLEL": "4", "QSE_ENGINE_FLAGS": "--model ${QSE_WEIGHTS_DIR} --drafter none"})
    ol.arm_state(arm)
    got = sorted(os.listdir(d / "home" / "conductor"))
    assert got == ["config.json"], got
    assert (d / "home" / "secrets.env").read_text() == ""
    env = ol.setup_env(arm)
    assert env.index("RESPONSE_CACHE='on'") < env.index("RESPONSE_CACHE='off'"), env
    assert "PORT='8105'" in env and "CONDUCTOR='raw'" in env and "QSE_USAGE_LEDGER='off'" in env
    assert f"QSE_CONDUCTOR_CONFIG='{d / 'home' / 'conductor'}'" in env
    assert 'QSE_ENGINE_FLAGS="--model ${QSE_WEIGHTS_DIR} --drafter none"' in env
    assert env.rindex("QSE_PARALLEL='4'") > env.index("QSE_PARALLEL='8'")
    # the overlay, sourced by bash, gives the arm's values last
    out = subprocess.run(["bash", "-c", f"set -a; QSE_WEIGHTS_DIR=/w; . /dev/stdin; echo $PORT $QSE_PARALLEL "
                          "$RESPONSE_CACHE \"$QSE_ENGINE_FLAGS\""], input=env, capture_output=True, text=True)
    assert out.stdout.strip() == "8105 4 off --model /w --drafter none", out.stdout + out.stderr


def test_knobs_record_every_limit_and_no_paths():
    k = server.knobs(HEALTH, {"QSE_PARALLEL": "8", "QWEN38_FN_PREFILL_CHUNK": "16384"},
                     ["app.py", "--port", "8100", "--model", "/home/x/flashnext-packs/3.05", "--drafter", "mtp",
                      "--response-cache"])
    for key in ("lanes", "max_running", "max_queue", "max_len", "prefill_chunk", "prefix_cache",
                "prefix_chunk", "cache_gb", "ceiling_gb", "peak_gb", "suffix_store", "response_cache",
                "drafter", "admission_floor_gib"):
        assert key in k, key
    assert k["lanes"] == 8 and k["prefill_chunk"] == "16384" and k["cache_gb"] == 2.0
    assert k["drafter"] == "mtp" and k["flags"]["--model"] == "3.05" and k["flags"]["--response-cache"] is True
    assert "/home" not in repr(k)


def test_secret_named_env_is_never_a_knob():
    assert server.KNOB_ENV.match("QSE_ADMIN_TOKEN") and server.SECRET.search("QSE_ADMIN_TOKEN")
    assert server.SECRET.search("QSE_DASHBOARD_LOGIN") and not server.SECRET.search("QSE_PARALLEL")


def plan(suite, files, prof=None, pack=None, pairs=0):
    a = argparse.Namespace(suite=suite, only=None, pairs=pairs)
    return plan_of(a, prof or Fake(), pack or {"launcher": "fake", "id": "fake"}, files)


def test_arm_order_and_variants():
    p = plan("auto", ["engine/spec.py"])
    assert p.level == "quick" and [(s, v) for s, v, _ in arms_of(p)] == [("base", "main"), ("head", "main")]
    flash = dict(load_packs()["flashnext-3.05"], id="flashnext-3.05")
    g = Generic(Fingerprint(memory_gib=48))
    p = plan("auto", ["engine/spec.py"], g, flash)
    assert [(s, v) for s, v, _ in arms_of(p)][-1] == ("head", "plain")
    p = plan("auto", ["server/slots.py"], g, flash)
    assert p.level == "quick+long"
    assert [(s, v) for s, v, _ in arms_of(p)][-2:] == [("base", "wide"), ("head", "wide")]
    p = plan("full", [], g, flash)
    assert p.pairs == 2 and [s for s, v, _ in arms_of(p) if v == "main"] == ["base", "head", "base", "head"]
    assert "stability" in arms_of(p)[0][2]
    assert plan("auto", ["docs/a.md"]).level == "none"


def test_arm_env_takes_the_variant_and_the_planted_head_settings():
    flash = dict(load_packs()["flashnext-3.05"], id="flashnext-3.05")
    g = Generic(Fingerprint(memory_gib=48))
    plant = {"REP_PENALTY": "1.05"}
    assert arm_env(g, flash, "base", "main", plant) == g.arm_env()
    assert arm_env(g, flash, "head", "main", plant) == {**g.arm_env(), **plant}
    assert "--drafter none" in arm_env(g, flash, "head", "plain")["QSE_ENGINE_FLAGS"]
    assert arm_env(g, flash, "base", "wide")["QWEN38_FN_PREFILL_CHUNK"] == "16384"


#: OPS-314: every pack's arm environments and plans as they were before a pack could set its own memory limits or a
#: cap on requests at once (recorded on 7f2a41f5): a pack that names neither keeps them byte for byte
ARMS_PIN = "0545f6d1f68090dd09655edfd8bcff72db447e302f22881720fc6a3bc46df715"


def _profiles():
    from tools.gate.hw.ada import Ada
    from tools.gate.hw.gb10 import GB10
    return [GB10(Fingerprint(name="NVIDIA GB10", memory_gib=119.7, cc="12.1", unified=True)),
            Ada(Fingerprint(name="NVIDIA RTX 4090", memory_gib=24.0, cc="8.9")), Generic(Fingerprint(memory_gib=48)),
            Fake()]


def _arms_digest(packs: dict) -> str:
    import hashlib
    import json
    out = []
    for prof in _profiles():
        for pid in sorted(packs):
            pack = dict(packs[pid], id=pid)
            for level in ("quick", "quick+long", "full"):
                p = plan(level, ["engine/spec.py"], prof, pack)
                out.append([prof.id, pid, level, p.profile.describe(),
                            [(s, v, su, arm_env(p.profile, pack, s, v, {"X": "1"})) for s, v, su in arms_of(p)]])
    return hashlib.sha256(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()


def test_a_pack_without_limits_or_a_cap_keeps_its_arms():
    old = {k: v for k, v in load_packs().items() if not ({"limits", "at_once", "base"} & set(v))}
    assert set(PINNED_PACKS) <= set(old), sorted(old)
    got = _arms_digest({k: v for k, v in old.items() if k in PINNED_PACKS})
    assert got == ARMS_PIN, got
    return f"{len(PINNED_PACKS)} packs x {len(_profiles())} profiles x 3 levels: the same arms and environments"


#: the packs of 7f2a41f5's packs.yaml
PINNED_PACKS = ("fake", "flashnext-3.05", "lfm25-exl3-4.0", "lfm25-exl3-2.5", "nemotron-bolt-exl3-4.0", "flint-exl3-4.0")


def _profile_limits(profile: str) -> tuple:
    """MEM_CEILING_GIB and MEM_PEAK_GIB as the weight profile ops/profiles/<profile>.env sets them (OPS-308: every
    family's ceiling and peak are its profile's)."""
    text = open(os.path.join(ROOT, "ops", "profiles", f"{profile}.env")).read()
    return tuple(float(re.search(rf"(?m)^{k}=([0-9.]+)", text).group(1)) for k in ("MEM_CEILING_GIB", "MEM_PEAK_GIB"))


def _arm_limits(pack: dict, prof, variant: str = "main") -> tuple:
    """MEM_CEILING_GIB and MEM_PEAK_GIB as the arm's server start reads them: the tree's ops/serve.env sourced with
    the arm's profile and its run/setup.env, by bash, as ops/start.sh sources them."""
    d = Path(tempfile.mkdtemp(prefix="gate-limits-"))
    tree = d / "tree"
    shutil.copytree(os.path.join(ROOT, "ops"), tree / "ops", ignore=shutil.ignore_patterns("__pycache__"))
    ol = server.OpsLauncher(prof, pack, {})
    ol.prepare(tree)
    arm = server.Arm("head", variant, tree, 8105, d / "home", arm_env(prof, pack, "head", variant))
    (tree / "run" / "setup.env").write_text(ol.setup_env(arm))
    out = subprocess.run(["bash", "-c", 'set -a; . "$0/ops/serve.env"; echo "$MEM_CEILING_GIB $MEM_PEAK_GIB '
                          '$QSE_PARALLEL $QSE_ENGINE_FLAGS"', str(tree)],
                         capture_output=True, text=True, env={"HOME": str(d), "PATH": os.environ.get("PATH", "")})
    shutil.rmtree(d, ignore_errors=True)
    got = out.stdout.split()
    assert len(got) >= 4, out.stdout + out.stderr
    return float(got[0]), float(got[1]), int(got[2]), " ".join(got[3:])


def _arm_mode(pack: dict, prof, site: dict | None = None) -> tuple:
    """(QSE_MODE, QSE_EXT, CONDUCTOR, QSE_CONDUCTOR_CONFIG is off) as the arm's server start reads them: the arm's own
    state made (its conductor folder), its run/setup.env written, the tree's ops/serve.env sourced by bash."""
    d = Path(tempfile.mkdtemp(prefix="gate-mode-"))
    tree = d / "tree"
    shutil.copytree(os.path.join(ROOT, "ops"), tree / "ops", ignore=shutil.ignore_patterns("__pycache__"))
    ol = server.OpsLauncher(prof, pack, site or {})
    ol.prepare(tree)
    arm = server.Arm("head", "main", tree, 8105, d / "home", arm_env(prof, pack, "head", "main"))
    ol.arm_state(arm)
    (tree / "run" / "setup.env").write_text(ol.setup_env(arm))
    out = subprocess.run(["bash", "-c", 'set -a; . "$0/ops/serve.env"; echo "$QSE_MODE|$QSE_EXT|$CONDUCTOR|'
                          '$QSE_CONDUCTOR_CONFIG"', str(tree)],
                         capture_output=True, text=True, env={"HOME": str(d), "PATH": os.environ.get("PATH", "")})
    shutil.rmtree(d, ignore_errors=True)
    got = out.stdout.strip().split("|")
    assert len(got) == 4, out.stdout + out.stderr
    return got[0], got[1], got[2], got[3] == "off"


def _install(d: Path, local: str) -> Path:
    """An install's run/ as install.sh leaves it: run/ops/serve.env written by install.sh's own `render_env` (its
    text read from install.sh, with its `q` and `setv`) over this tree's template, with a machine's PY, PYTHONPATH and
    HF_HUB_CACHE as its keys, and run/local.env holding `local`. Returns run/local.env."""
    text = open(os.path.join(ROOT, "install.sh")).read()

    def block(start: str, end: str) -> str:
        i = text.index(start)
        return text[i:text.index(end, i) + len(end)]
    run = d / "run"
    (run / "ops").mkdir(parents=True)
    (run / "local.env").write_text(local)
    script = "\n".join([
        "set -e", block("q() {", "}    # single-quoted for the env file"), "ENV_KEYS=()", "ENV_ASSIGN=()",
        block("setv() {", "}"), 'setv REPO "$(q /box/install/src)"', 'setv PY "$(q /box/install/venv/bin/python)"',
        'setv PYTHONPATH ""', 'setv HF_HUB_CACHE "$(q /box/hub)"', f'RUN={run}', f'TEMPLATE={ROOT}/ops/serve.env',
        "TANDEM_REF=test", block("render_env() {", "\n}\n"), f'render_env > {run}/ops/serve.env'])
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0 and "local.env" in (run / "ops" / "serve.env").read_text(), r.stderr[-400:]
    return run / "local.env"


def _arm_reads(site: dict, keys: str) -> list:
    """`keys` as the arm's start reads them: its tree prepared with `site`, its setup.env written, the tree's
    ops/serve.env sourced by bash as ops/start.sh sources it."""
    from tools.gate.hw.gb10 import GB10
    g = GB10(Fingerprint(name="NVIDIA GB10", memory_gib=119.7, cc="12.1", unified=True))
    pack = dict(load_packs()["hy4-3.56"], id="hy4-3.56")
    d = Path(tempfile.mkdtemp(prefix="gate-install-"))
    tree = d / "tree"
    shutil.copytree(os.path.join(ROOT, "ops"), tree / "ops", ignore=shutil.ignore_patterns("__pycache__"))
    ol = server.OpsLauncher(g, pack, site)
    ol.prepare(tree)
    arm = server.Arm("head", "main", tree, 8105, d / "home", arm_env(g, pack, "head", "main"))
    ol.arm_state(arm)
    (tree / "run" / "setup.env").write_text(ol.setup_env(arm))
    echo = "|".join(f"${{{k}}}" for k in keys.split())
    out = subprocess.run(["bash", "-c", f'set -a; . "$0/ops/serve.env"; echo "{echo}"', str(tree)],
                         capture_output=True, text=True, env={"HOME": str(d), "PATH": os.environ.get("PATH", "")})
    shutil.rmtree(d, ignore_errors=True)
    return out.stdout.strip().split("|")


def test_an_arm_reads_the_installs_local_env_as_its_start_does():
    """An install's run/local.env is read by the serve.env install.sh generated beside it (render_env puts the line
    before the template's mode block); the repository's template reads none. The gate copied a site's local_env into
    the arm's tree and nothing read it, so an arm ran the template's default PY, not the install's. A local_env in an
    install's run/ now takes that install's generated serve.env as the arm's (as the `serve_env` site key does, REPO
    the arm's tree): the arm's PY is local.env's where local.env sets it, else the install's, and the install's paths
    come with it; the arm's port and mode stay the arm's. A local_env with no generated serve.env beside it (a
    checkout's) is copied as before."""
    d = Path(tempfile.mkdtemp(prefix="gate-inst-"))
    try:
        local = _install(d / "a", "PY=/box/local/venv/bin/python\n")
        py, hub, port, mode, repo = _arm_reads({"local_env": str(local)}, "PY HF_HUB_CACHE PORT QSE_MODE REPO")
        assert (py, hub, port, mode) == ("/box/local/venv/bin/python", "/box/hub", "8105", "engine"), (py, hub, port, mode)
        assert repo.endswith("/tree"), repo
        local = _install(d / "b", "# nothing of PY here\n")
        assert _arm_reads({"local_env": str(local)}, "PY")[0] == "/box/install/venv/bin/python"
        plain = d / "plain.env"
        plain.write_text("PY=/opt/py\n")
        tmpl = re.search(r"(?m)^PY=\$\{HOME\}(.*)$", open(os.path.join(ROOT, "ops", "serve.env")).read()).group(1)
        for site in ({}, {"local_env": str(plain)}):
            got = _arm_reads(site, "PY")[0]
            assert got.endswith(tmpl) and got != "/opt/py", (site, got, "a checkout's local.env as before")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return "local.env's PY, the install's paths; else the install's PY; a checkout's local.env unread as before"


def test_a_gate_arm_serves_the_engine_alone():
    """The gate measures the engine, not the chat layer (tools/gate/client.py: every request X-Tandem raw). An arm's
    state makes a conductor folder and its setup.env names it, and serve.env (OPS-260) starts chat mode for an existing
    conductor folder when nothing sets QSE_MODE, which loads TandemApp's extension (QSE_EXT tandemapp.engine_ext): an
    engine whose Python has no TandemApp refused to start ("[ext] REFUSING", the gate run on 70c250f6). Every arm is in
    engine mode, conductor raw; a site that names a conductor copy (`conductor_from`) asks for chat mode, as before."""
    from tools.gate.hw.gb10 import GB10
    g = GB10(Fingerprint(name="NVIDIA GB10", memory_gib=119.7, cc="12.1", unified=True))
    packs = load_packs()
    for pid in ("flashnext-3.05", "hy4-3.56", "flint-exl3-4.0"):
        pack = dict(packs[pid], id=pid)
        assert _arm_mode(pack, g) == ("engine", "", "raw", True), (pid, _arm_mode(pack, g))
    cond = Path(tempfile.mkdtemp(prefix="gate-cond-"))
    (cond / "config.json").write_text("{}")
    mode, ext, conductor, off = _arm_mode(dict(packs["flashnext-3.05"], id="flashnext-3.05"), g,
                                          {"conductor_from": str(cond)})
    shutil.rmtree(cond, ignore_errors=True)
    assert (mode, ext, off) == ("chat", "tandemapp.engine_ext", False), (mode, ext, conductor, off)
    return "three packs: engine mode, conductor raw, no extension; a site's conductor copy: chat"


def test_a_pack_with_profile_limits_runs_them_on_both_sides():
    """OPS-308: the gb10 profile's 66/77 GiB, written after the pack's profile, would run a Hy4-flash start under what
    its load reserves at 8 lanes: 72.50 GiB on both packs without expandable segments, 71.84 on hy4-3.56 with them
    (ENG-609; hy4-3.08 63.29). With `limits: profile` both sides' arms get the ceiling and peak the
    pack's weight profile serves it with (79/81 on both packs since ENG-609), read from this checkout's profile, so
    the base side's tree, which names none, runs them too; a pack without the key keeps the hardware profile's. Both
    Hy4-flash packs, both sides, both variants."""
    from tools.gate.hw.gb10 import GB10
    g = GB10(Fingerprint(name="NVIDIA GB10", memory_gib=119.7, cc="12.1", unified=True))
    packs = load_packs()
    for pid in ("hy4-3.56", "hy4-3.08"):
        pack = dict(packs[pid], id=pid)
        want = _profile_limits(pack["profile"])
        assert want == (79.0, 81.0), (pid, want)
        for side in ("base", "head"):
            env = arm_env(g, pack, side, "main")
            assert (float(env["MEM_CEILING_GIB"]), float(env["MEM_PEAK_GIB"])) == want, (pid, side, env)
            assert {k: v for k, v in env.items() if not k.startswith("MEM_")} == \
                {k: v for k, v in g.arm_env().items() if not k.startswith("MEM_")}, env
        for variant in ("main", "plain"):
            ceiling, peak, lanes, flags = _arm_limits(pack, g, variant)
            assert (ceiling, peak) == want and lanes == g.lanes(), (pid, variant, ceiling, peak, lanes)
            assert flags.endswith("--drafter " + ("mtp" if variant == "main" else "none")), (pid, variant, flags)
        before = {k: v for k, v in pack.items() if k != "limits"}
        assert _arm_limits(before, g)[:2] == (g.ceiling_gib(), g.peak_gib()) != want, "the bypass, without the key"
    flash = dict(packs["flashnext-3.05"], id="flashnext-3.05")
    assert _arm_limits(flash, g)[:2] == (g.ceiling_gib(), g.peak_gib())
    return f"both packs, both sides and variants: the profile's {want[0]:g}/{want[1]:g} GiB reach the start; without the key: 66/77"


def test_a_pack_capped_at_one_plans_c1_rows_only():
    """OPS-314: a Hy4-flash pack plans the decode suite's c1 shapes and a burst of one; the lanes stay the profile's
    (the family's own 8); the dry run says so and takes the pack's base."""
    from tools.gate.hw.gb10 import GB10
    from tools.gate.client import prompts
    from tools.gate.runner import AtOnce, pack_notes
    from tools.gate.suites import decode
    g = GB10(Fingerprint(name="NVIDIA GB10", memory_gib=119.7, cc="12.1", unified=True))
    for pid in ("hy4-3.56", "hy4-3.08"):
        pack = dict(load_packs()[pid], id=pid)
        p = plan("quick", ["engine/models/hy4_flash/serving.py"], g, pack)
        assert isinstance(p.profile, AtOnce) and p.profile.decode_levels() == [] and p.profile.burst() == (1, 2048)
        assert p.profile.lanes() == 8 and p.profile.arm_env(pack)["QSE_PARALLEL"] == "8"
        d = p.profile.describe()
        assert d["decode_levels"] == [] and d["burst"] == [1, 2048] and d["at_once"] == 1, d
        ctx = type("C", (), {"profile": p.profile, "pj": prompts()})()
        assert [s[0] for s in decode.shapes(ctx)] == ["c1", "c1-1k"]
        assert [(s, v) for s, v, _ in arms_of(p)] == [("base", "main"), ("head", "main"), ("head", "plain")]
        notes = pack_notes(pack, p.profile, True)
        assert len(notes) == 3 and "limits: profile" in notes[0] and "79/81 GiB" in notes[0], notes
        assert "none above c1" in notes[1], notes
        assert pack["base"] == "7f2a41f51b9793dcc8fddb30316d8a0161d10072"
    assert plan("quick", [], g, dict(load_packs()["flashnext-3.05"], id="flashnext-3.05")).profile is g
    return "c1 and c1-1k, a burst of one, 8 lanes, base and head and head plain"


def test_a_dry_run_takes_a_hy4_pack():
    """OPS-314: `python -m tools.gate run --pack hy4-3.56 --dry` is no longer refused with "no pack": it prints the plan
    with the pack's base and its lines. Needs the pack's base commit in this checkout."""
    if not _git_repo() or subprocess.run(["git", "-C", ROOT, "cat-file", "-e", load_packs()["hy4-3.56"]["base"]],
                                         capture_output=True).returncode != 0:
        return "skipped: no git checkout with the pack's base"
    for pid in ("hy4-3.56", "hy4-3.08"):
        r = subprocess.run([sys.executable, "-m", "tools.gate", "run", "--pack", pid, "--hw", "gb10", "--suite",
                            "quick", "--dry"], cwd=ROOT, capture_output=True, text=True, timeout=300)
        out = r.stdout + r.stderr
        assert r.returncode == 0 and "no pack" not in out and f"pack {pid}" in out, out[-1500:]
        assert "(7f2a41f51b9793dcc8fddb30316d8a0161d10072)" in out and "rows of at most 1 request(s)" in out, out[-1500:]
        assert "servers: base, head, head:plain" in out, out[-1500:]
    return "both packs: planned, base 7f2a41f5, c1 rows, servers base, head, head:plain"


def test_packs_name_real_profiles():
    for pid, p in load_packs().items():
        assert p["launcher"] in ("ops", "fake"), pid
        if p["launcher"] == "ops":
            assert os.path.isfile(os.path.join(ROOT, "ops", "profiles", p["profile"] + ".env")), pid

def test_ops_scripts_resolve_the_arm_port_whatever_the_caller_exports():
    """A caller's QSE_SETUP_ENV, REPO or PORT (a hold's shell exports them) pointed ops/stop.sh at
    another server's settings, so the arm's stop went to :8000. The ops env drops them and names the
    arm's own run/setup.env; serve.env's own chain then resolves the arm's port."""
    d = Path(tempfile.mkdtemp(prefix="gate-ops-env-"))
    tree = d / "tree"
    (tree / "ops").mkdir(parents=True)
    (tree / "run").mkdir()
    (tree / "ops" / "serve.env").write_text(
        f"REPO={tree}\nPORT=8000\nQSE_SETUP_ENV_FILE=${{QSE_SETUP_ENV:-${{REPO}}/run/setup.env}}\n"
        "if [ -f \"${QSE_SETUP_ENV_FILE}\" ]; then . \"${QSE_SETUP_ENV_FILE}\"; fi\n")
    (tree / "run" / "setup.env").write_text("PORT='8107'\n")
    other = d / "live-setup.env"
    other.write_text("PORT='8000'\n")
    arm = server.Arm("base", "main", tree, 8107, d / "home", {})
    saved = {k: os.environ.get(k) for k in ("QSE_SETUP_ENV", "REPO", "PORT")}
    os.environ.update(QSE_SETUP_ENV=str(other), REPO="/elsewhere", PORT="8000")
    try:
        env = server.OpsLauncher.ops_env(arm)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert env["QSE_SETUP_ENV"] == str(tree / "run" / "setup.env"), env.get("QSE_SETUP_ENV")
    assert "REPO" not in env and "PORT" not in env and env["QSE_STATUS_KEEP"] == "1"
    out = subprocess.run(["bash", "-c", "set -a; . ops/serve.env; echo $PORT"], cwd=tree, env=env,
                         capture_output=True, text=True)
    assert out.stdout.strip() == "8107", out.stdout + out.stderr


def test_pid_alive():
    assert server.pid_alive(os.getpid())
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert not server.pid_alive(p.pid)

def test_a_stopped_server_with_closed_connections_frees_its_port():
    """The server closes first, so its side of each connection sits in TIME_WAIT on the port; the
    port is free for the next server all the same (the GPU dry run's stops failed on this)."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen()
    cli = socket.create_connection(("127.0.0.1", port))
    conn, _ = srv.accept()
    assert server.port_busy(port)
    conn.close()                                   # the server's side closes first: TIME_WAIT here
    cli.recv(1)
    cli.close()
    srv.close()
    assert not server.port_busy(port)
    # a new server gets no port that is in TIME_WAIT: one that binds without SO_REUSEADDR would fail
    assert server.port_busy(port, strict=True)
    assert server.free_port(port, set(), floor=1024) != port


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
