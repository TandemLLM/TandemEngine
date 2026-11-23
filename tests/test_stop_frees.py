"""What a stop of the engine gives back to the box, `/health` `memory.stop_frees_gb` (SRV-953, QA SRV-954;
server/app.py `_memory`, `_stop_frees`), on a CPU: the allocator's counters, the process's resident set and
the KV pool are stand-ins.

On 2026-11-23 the Setup Guide refused Flash-Next 3.05 -> Hy4 3.56 after a 139k-token chat: it counted the
largest of the engine's readings (the reserve, 67.3 GiB) as what the stop frees, where the stop frees the
reserve, the KV pool's pages mapped outside it (6.25) and the process's memory (10.52) together.

The test matrix:

  1  no KV pool: the reserve plus the resident set                  test_no_pool
  2  the pool outside the allocator (the driver maps its pages): its committed GiB added
                                                                     test_the_pool_outside_the_allocator
  3  the pool inside the allocator (no driver): not added, the reserve holds its pages
                                                                     test_the_pool_inside_the_allocator
  4  a part not known (the counters not read, no resident set, the pool's report failing): no field,
     and the preflight counts the largest reading as before          test_a_part_not_known
  5  the preflight's reading of the field (server/setup/apply.py `stop_frees`): the sum and its parts
                                                                     test_the_preflight_reads_it

Run: python tests/test_stop_frees.py
"""
from __future__ import annotations

import builtins
import contextlib
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import cache, kvpool  # noqa: E402
from server import app  # noqa: E402
from server.setup import apply  # noqa: E402

GiB = 1 << 30


class _Pool:
    """The KV pool's report as engine/kvpool.py gives it (`committed_gib`), or a failing one."""

    def __init__(self, committed_gib: float, fail: bool = False):
        self.committed_gib, self.fail = committed_gib, fail

    def report(self) -> dict:
        if self.fail:
            raise RuntimeError("the pool's report failed")
        return {"committed_gib": self.committed_gib, "by_owner_gib": {"slot0": 2.25, "slot5": 4.0}}


@contextlib.contextmanager
def _engine(reserved_gib=67.30, rss_gib=10.52, pool=None, driver=False, counters=True):
    """`_memory()` on this machine with the allocator's reserve, the process's VmRSS, the KV pool and its
    driver (True: the pages are mapped outside the allocator) as given; restored after."""
    stats = {"allocated_bytes.all.current": int(55.0 * GiB), "allocated_bytes.all.peak": int(60.0 * GiB),
             "reserved_bytes.all.current": int(reserved_gib * GiB)}

    def cuda_stats(*names):
        if not counters:
            raise RuntimeError("no CUDA")
        return tuple(stats.get(n, 0) for n in names)

    def fake_open(path, *a, **k):
        if path == "/proc/self/status":
            if rss_gib is None:
                raise FileNotFoundError(path)
            return io.StringIO(f"VmRSS:\t{int(rss_gib * (1 << 20))} kB\nVmSize:\t{400 << 20} kB\n")
        return builtins.open(path, *a, **k)

    keep = (cache.cuda_stats, kvpool.POOL, kvpool.Driver.__dict__["get"], app.STATE.get("mem", None),
            "mem" in app.STATE)
    cache.cuda_stats = cuda_stats
    kvpool.POOL = pool
    kvpool.Driver.get = staticmethod(lambda device=0: object() if driver else None)
    app.STATE.pop("mem", None)
    app.open = fake_open                     # the module's `open`: /proc/self/status is not on this machine
    try:
        yield
    finally:
        del app.open
        cache.cuda_stats, kvpool.POOL = keep[0], keep[1]
        kvpool.Driver.get = keep[2]
        if keep[4]:
            app.STATE["mem"] = keep[3]


def test_no_pool():
    with _engine():
        out = app._memory()
    assert out["reserved_gb"] == 67.3 and out["rss_gb"] == 10.52, out
    assert out["stop_frees_gb"] == round(67.3 + 10.52, 3), out


def test_the_pool_outside_the_allocator():
    with _engine(pool=_Pool(6.25), driver=True):
        out = app._memory()
    assert out["stop_frees_gb"] == 84.07, out                    # D15: 67.30 + 6.25 + 10.52


def test_the_pool_inside_the_allocator():
    with _engine(reserved_gib=59.67, rss_gib=7.56, pool=_Pool(1.07), driver=False):
        out = app._memory()
    assert out["stop_frees_gb"] == round(59.67 + 7.56, 3), out


def test_a_part_not_known():
    for kw in ({"counters": False}, {"rss_gib": None}, {"pool": _Pool(6.25, fail=True), "driver": True}):
        with _engine(**kw):
            out = app._memory()
        assert "stop_frees_gb" not in out, (kw, out)
        assert apply.stop_frees(out) == (max(out.get("reserved_gb", 0), out.get("rss_gb", 0)), ""), (kw, out)
    assert apply.stop_frees({}) == (0.0, "")
    # a bool is not a number of GiB
    assert apply.stop_frees({"stop_frees_gb": True, "reserved_gb": 3.0}) == (3.0, "")


def test_the_preflight_reads_it():
    with _engine(pool=_Pool(6.25), driver=True):
        out = app._memory()
    assert apply.stop_frees(out) == (84.07, " (the allocator's reserve 67.3, the KV pool's pages 6.2 and the "
                                            "process's memory 10.5)"), apply.stop_frees(out)
    with _engine(reserved_gib=59.67, rss_gib=7.56, pool=_Pool(1.07), driver=False):
        out = app._memory()
    assert apply.stop_frees(out) == (67.23, " (the allocator's reserve 59.7 and the process's memory 7.6)"), \
        apply.stop_frees(out)
    # the sum without its parts (a client that strips the block): the sum, no words
    assert apply.stop_frees({"stop_frees_gb": 84.07}) == (84.07, "")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            fn()
            print(f"  {name:45s} ok ({time.time() - t0:.1f} s)")
            passed += 1
    print(f"{passed} passed")
