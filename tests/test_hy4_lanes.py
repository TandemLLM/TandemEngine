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
