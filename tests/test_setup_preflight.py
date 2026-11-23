"""The preflight of a Setup Guide install (OPS-169, QA OPS-170): everything the restart needs is checked
before anything stops (server/setup/apply.py `preflight`, the step between the download and the
configuration), on a CPU against tests/test_setup_apply.py's Box, never the box's own tree.

On 2026-10-17 an install stopped the running setup before it found out that start.sh could not run.

The test matrix, the Gherkin of OPS-169:

  1  a script that does not parse: the preflight step fails naming it, nothing was stopped (no
     stop.sh, no switch) and no configuration was written     test_a_script_that_does_not_parse
  2  incomplete weights (a shard the index names is missing; a file shorter than the hub manifest
     says) are named                                          test_incomplete_weights
  3  not enough memory: MemAvailable plus what the engine frees under the profile's need, both
     numbers named; setup mode counts nothing to free         test_not_enough_memory
  4  the new configuration does not source under set -u: the line's error named; the engine's
     interpreter missing: named                               test_the_configuration_and_the_interpreter
  5  all checks pass: the step is done and the job goes on     test_all_checks_pass

SRV-953 (QA SRV-954): what a stop frees is the engine's own sum, `/health` `memory.stop_frees_gb` (the
allocator's reserve, the KV pool's pages outside it, the process's memory), where the check took the largest
of the three. On 2026-11-23 it refused Flash-Next 3.05 -> Hy4 3.56 after a 139k-token chat, 0.9 GiB short.

  6  D15's numbers (MemAvailable 30.8, stop_frees_gb 84.07, need 101) pass; the same /health without the
     field: the largest reading and the refusal of that day, unchanged   test_a_switch_after_a_long_session
  7  a real shortfall (MemAvailable 10, 84 freed) refuses, the sum's parts named, nothing stopped; the
     pool inside the allocator names no pool part                        test_a_real_shortfall
  8  setup mode counts nothing to free (no /health read); a /health without `memory` counts 0; a
     /health that does not answer skips the check (use-profile.sh checks again after the stop)
                                                         test_setup_mode_a_missing_block_and_a_read_error
  9  on the two measured samples of 2026-11-23, every shipped profile's verdict is the one
     use-profile.sh's check after the stop gives (MemAvailable 117.68 with the engine down); the
     largest reading refused profiles that check allows        test_every_profile_agrees_with_the_check_after_the_stop

Run: python tests/test_setup_preflight.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_setup_apply as T  # noqa: E402
from server.setup import apply, device as device_mod  # noqa: E402


def _run(box, sel=None, http=None, setup_mode=False):
    rec = T.Recorder()
    jid = T._job(sel or T._sel(), setup_mode=setup_mode)
    rc = apply.Runner(jid, cmd=rec, http=http or T._smoke_ok(), ops=box.ops).main()
    j = apply.read_job(jid)
    seq = [" ".join(os.path.basename(a) for a in c["argv"][1:]) for c in rec.calls]
    return rc, j, seq


def _stopped_nothing(box, j, seq):
    st = {s["id"]: s["state"] for s in j["steps"]}
    assert j["state"] == "failed" and j["failed_step"] == "preflight", j
    assert st["write_config"] == "skipped" and st["switch"] == "skipped", st
    assert not any(s.startswith(("stop.sh", "start.sh")) or s == "use-profile.sh exl3-4.5" for s in seq), seq
    assert not os.path.exists(os.path.join(box.root, "run", "setup.env"))
    assert "Nothing was stopped or changed." in j["message"], j["message"]


def test_a_script_that_does_not_parse():
    with T.Box() as box:
        with open(os.path.join(box.ops, "start.sh"), "w") as fh:
            fh.write("#!/bin/bash\nif then fi (\n")
        rc, j, seq = _run(box)
        assert rc == 1
        _stopped_nothing(box, j, seq)
        assert "ops/start.sh has an error:" in j["message"], j["message"]
        os.unlink(os.path.join(box.ops, "stop.sh"))
        rc, j, seq = _run(box)
        assert "ops/stop.sh is missing on this box" in j["message"], j["message"]


def test_incomplete_weights():
    with T.Box() as box:
        d = os.path.join(box.home, "exl3-quants", "4.5")
        with open(os.path.join(d, "_hub_manifest.json"), "w") as fh:
            json.dump({"files": {"m-1.safetensors": {"size": 999, "sha256": "x"}}}, fh)
        device_mod.clear()
        rc, j, seq = _run(box)
        _stopped_nothing(box, j, seq)
        assert "A weight file is cut short: m-1.safetensors has 1 bytes, the hub lists 999." in j["message"], \
            j["message"]
    with T.Box() as box:
        # the plan saw the shard, then it went (a download that another program cut, a disk): named
        real = apply.plan

        def plan(*a, **k):
            p = real(*a, **k)
            os.unlink(os.path.join(box.home, "exl3-quants", "4.5", "m-1.safetensors"))
            device_mod.clear()
            apply.plan = real
            return p
        apply.plan = plan
        try:
            rc, j, seq = _run(box)
        finally:
            apply.plan = real
        _stopped_nothing(box, j, seq)
        assert "The weights of exl3-4.5 are not complete on this box" in j["message"], j["message"]


def test_not_enough_memory():
    with T.Box() as box:
        device_mod.meminfo = lambda: {"MemTotal": 121.6, "MemAvailable": 30.0}

        def http(url, body=None, timeout=10):
            if url.endswith("/health"):
                return {"status": "ok", "weights": {"name": "exl3-4.5"}, "memory": {"engine_reserved_gb": 20.0}}
            return {"choices": [{"message": {"content": "ready"}}]}
        rc, j, seq = _run(box, http=http)
        _stopped_nothing(box, j, seq)
        assert "Not enough memory for exl3-4.5: it needs 80 GiB free to start; 30.0 GiB are free now and " \
               "the running engine gives back about 20.0." in j["message"], j["message"]
        # with room: through
        device_mod.meminfo = lambda: {"MemTotal": 121.6, "MemAvailable": 62.0}
        rc, j, seq = _run(box, http=http)
        assert rc == 0 and j["state"] == "done", j["message"]


#: /health `memory` of Flash-Next 3.05 on 2026-11-23 after a 139k-token chat (D15): the reserve, the KV
#: pool's pages outside it (6.25 GiB, 4.0 of them slot 5's) and the process's memory, summed by the engine
D15 = {"engine_reserved_gb": 65.7, "reserved_gb": 67.30, "rss_gb": 10.52, "stop_frees_gb": 84.07}


def _need(box, gib: int) -> None:
    """The box's exl3-4.5 profile needs `gib` free to start, as hy4-3.56's does (101)."""
    with open(os.path.join(box.ops, "profiles", "exl3-4.5.env"), "a") as fh:
        fh.write(f"QSE_MEM_NEED_GIB={gib}\n")


def _health(memory, asked=None, fail=False):
    """An engine whose /health carries `memory` (None: no memory block); `fail`: the preflight's read
    (and the idle check's, both with a 5 s timeout) raises; `asked` collects those reads."""
    def http(url, body=None, timeout=10):
        if url.endswith("/health"):
            if timeout == 5 and asked is not None:
                asked.append(url)
            if timeout == 5 and fail:
                raise OSError("timed out")
            return dict({"status": "ok", "weights": {"name": "exl3-4.5"}},
                        **({"memory": memory} if memory is not None else {}))
        return {"choices": [{"message": {"content": "ready"}}]}
    return http


def _went_through(rc, j, seq):
    st = {s["id"]: s["state"] for s in j["steps"]}
    assert rc == 0 and j["state"] == "done" and st["preflight"] == "done", (st, j["message"])
    assert seq == ["use-profile.sh exl3-4.5"], seq


def test_a_switch_after_a_long_session():
    with T.Box() as box:
        _need(box, 101)
        device_mod.meminfo = lambda: {"MemTotal": 121.6, "MemAvailable": 30.8}
        # an engine before SRV-953: the largest reading, 30.8 + 67.3 + 2 < 101, as on 2026-11-23
        old = {k: v for k, v in D15.items() if k != "stop_frees_gb"}
        rc, j, seq = _run(box, http=_health(old))
        _stopped_nothing(box, j, seq)
        assert "Not enough memory for exl3-4.5: it needs 101 GiB free to start; 30.8 GiB are free now and " \
               "the running engine gives back about 67.3. Stop other programs" in j["message"], j["message"]
        # the engine's sum: 30.8 + 84.07 + 2 = 116.9, through
        _went_through(*_run(box, http=_health(D15)))


def test_a_real_shortfall():
    with T.Box() as box:
        _need(box, 101)
        device_mod.meminfo = lambda: {"MemTotal": 121.6, "MemAvailable": 10.0}
        rc, j, seq = _run(box, http=_health({"reserved_gb": 67.3, "rss_gb": 10.5, "stop_frees_gb": 84.0}))
        assert rc == 1
        _stopped_nothing(box, j, seq)
        assert "Not enough memory for exl3-4.5: it needs 101 GiB free to start; 10.0 GiB are free now and " \
               "the running engine gives back about 84.0 (the allocator's reserve 67.3, the KV pool's pages " \
               "6.2 and the process's memory 10.5). Stop other programs that use memory, then try again." \
               in j["message"], j["message"]
        # the pool inside the allocator: the reserve holds its pages, no part of its own
        rc, j, seq = _run(box, http=_health({"reserved_gb": 59.67, "rss_gb": 7.56, "stop_frees_gb": 67.23}))
        _stopped_nothing(box, j, seq)
        assert "gives back about 67.2 (the allocator's reserve 59.7 and the process's memory 7.6). Stop" \
               in j["message"], j["message"]


def test_setup_mode_a_missing_block_and_a_read_error():
    # setup mode: no engine runs, nothing is counted and /health is not read
    with T.Box() as box:
        _need(box, 101)
        device_mod.meminfo = lambda: {"MemTotal": 121.6, "MemAvailable": 30.8}
        asked = []
        rc, j, seq = _run(box, http=_health(D15, asked), setup_mode=True)
        _stopped_nothing(box, j, seq)
        assert "30.8 GiB are free now and the running engine gives back about 0.0. Stop" in j["message"], \
            j["message"]
        assert asked == [], asked
    # a /health without a memory block: 0 counted
    with T.Box() as box:
        _need(box, 101)
        device_mod.meminfo = lambda: {"MemTotal": 121.6, "MemAvailable": 30.8}
        rc, j, seq = _run(box, http=_health(None))
        _stopped_nothing(box, j, seq)
        assert "the running engine gives back about 0.0. Stop" in j["message"], j["message"]
    # a /health that does not answer: the check is skipped, use-profile.sh's after the stop decides
    with T.Box() as box:
        _need(box, 101)
        device_mod.meminfo = lambda: {"MemTotal": 121.6, "MemAvailable": 30.8}
        asked = []
        _went_through(*_run(box, http=_health(D15, asked, fail=True)))
        assert any(u.endswith(":8000/health") for u in asked), asked


#: MemAvailable with the engine down after the D15 engine's stop (2026-11-23): what use-profile.sh's check
#: after the stop compares with the need; and the two /health samples before a stop (MemAvailable, memory),
#: the second inferred
AFTER_STOP = 117.68
SAMPLES = {"after the 139k chat": (31.15, D15),
           "fresh": (47.62, {"engine_reserved_gb": 59.0, "reserved_gb": 59.67, "rss_gb": 7.56,
                             "stop_frees_gb": 59.67 + 1.07 + 7.56})}


def test_every_profile_agrees_with_the_check_after_the_stop():
    with T.Box() as box:
        needs = {}
        for f in sorted(os.listdir(os.path.join(box.ops, "profiles"))):
            if f.endswith(".env") and not f.startswith("_"):
                env, err = apply._sources(box.ops, f[:-4], None, ["QSE_MEM_NEED_GIB"])
                assert err is None, (f, err)
                needs[f[:-4]] = float(env.get("QSE_MEM_NEED_GIB") or 80)     # use-profile.sh's default
    assert len(needs) > 20 and needs["hy4-3.56"] == 101, needs
    old_wrong = []
    for name, (avail, mm) in SAMPLES.items():
        freed, _ = apply.stop_frees(mm)
        largest = max(mm["engine_reserved_gb"], mm["reserved_gb"], mm["rss_gb"])
        for p, need in needs.items():
            gate = AFTER_STOP >= need
            assert (avail + freed + 2 >= need) == gate, (name, p, avail, freed, need)
            if (avail + largest + 2 >= need) != gate:
                old_wrong.append(f"{p} ({name})")
    assert "hy4-3.56 (after the 139k chat)" in old_wrong, old_wrong
    return f"{len(needs)} profiles x {len(SAMPLES)} samples; the largest reading refused {', '.join(old_wrong)}"


def test_the_configuration_and_the_interpreter():
    with T.Box() as box:
        with open(os.path.join(box.ops, "serve.env"), "a") as fh:
            fh.write("BROKEN=$NOT_SET_ANYWHERE\n")
        rc, j, seq = _run(box)
        _stopped_nothing(box, j, seq)
        assert "serve.env does not load with the new configuration:" in j["message"] and \
            "NOT_SET_ANYWHERE" in j["message"], j["message"]
    with T.Box() as box:
        text = open(os.path.join(box.ops, "serve.env")).read().replace(f"PY={sys.executable}", "PY=/nowhere/python")
        with open(os.path.join(box.ops, "serve.env"), "w") as fh:
            fh.write(text)
        rc, j, seq = _run(box)
        _stopped_nothing(box, j, seq)
        assert "The engine's Python (/nowhere/python) does not exist." in j["message"], j["message"]


def test_all_checks_pass():
    with T.Box() as box:
        rc, j, seq = _run(box)
        st = {s["id"]: s["state"] for s in j["steps"]}
        assert rc == 0 and st["preflight"] == "done" and st["switch"] == "done", (st, j["message"])
        pf = next(s for s in j["steps"] if s["id"] == "preflight")
        assert pf["detail"].startswith("the scripts, the new configuration, the weights") and \
            seq == ["use-profile.sh exl3-4.5"], (pf, seq)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            note = fn()
            print(f"  {name:50s} ok ({time.time() - t0:.1f} s)" + (f": {note}" if note else ""))
            passed += 1
    print(f"{passed} passed")
