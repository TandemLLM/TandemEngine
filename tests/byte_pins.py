"""Byte pins recorded once on a host and compared on every run after (tests/test_refactor_gate.py's form).

A pin is a sha256 of a tensor's bytes. The bytes depend on the torch build and the machine's kernels, so a
fixture keeps one table a host key; a host without a recorded key reports the digests and skips the
comparison. Recording is for the tree the pin is meant to hold (the code before a change that must not move
it), and only then:

    QSE_PIN_RECORD=1 python tests/<test>.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys

import torch

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def digest(t: torch.Tensor) -> str:
    """sha256 of a tensor's bytes as stored (any dtype), its shape included."""
    t = t.detach().contiguous().cpu()
    raw = t.numpy().tobytes() if t.dtype == torch.bool else t.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(repr(tuple(t.shape)).encode() + raw).hexdigest()


def host_key(device: str | None = None) -> str:
    """The machine, the torch build, and for a device pin the device's name."""
    key = f"{platform.machine()}-{sys.platform}-torch{torch.__version__.split('+')[0]}"
    if device is not None and str(device).startswith("cuda"):
        key += "-" + torch.cuda.get_device_name(torch.device(device)).replace(" ", "_")
    return key


def check(fixture: str, kind: str, got: dict, device: str | None = None) -> str:
    """`got` (name -> digest) against the fixture's table for this host and `kind`; with QSE_PIN_RECORD=1 the
    table is written instead. Raises AssertionError on a digest that moved."""
    path = os.path.join(FIXTURES, fixture)
    fx = {}
    if os.path.isfile(path):
        with open(path) as f:
            fx = json.load(f)
    key = host_key(device)
    if os.environ.get("QSE_PIN_RECORD") == "1":
        fx.setdefault(key, {})[kind] = got
        os.makedirs(FIXTURES, exist_ok=True)
        with open(path, "w") as f:
            json.dump(fx, f, indent=1, sort_keys=True)
            f.write("\n")
        return f"recorded {len(got)} digests for {key}"
    want = fx.get(key, {}).get(kind)
    if want is None:
        return f"SKIP compare: no {kind} digests recorded for {key} (known: {sorted(fx) or 'none'})"
    bad = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
    assert not bad, f"{kind}: bytes moved in {bad[:8]}{' ...' if len(bad) > 8 else ''} ({len(bad)} of {len(want)}, host {key})"
    return f"{len(want)} digests reproduce ({key})"
