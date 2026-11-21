"""`Capabilities.prompt_rows` (R83, the rowinv design's T5b): the server's prompt read gives a one-row prompt chunk to
the family's prompt pass, not to the decode step a lone token at a start past 0 takes, so a prompt-end resume whose
tail is one row equals the cold read; a family without the capability gets the call it got before.

  1  engine/cache.py's prompt read calls `forward(blk, start=i, last_only=True)` for a family without the
     capability, every chunk and a one-row tail alike, and `forward(..., prompt=True)` for one with it
                                              test_a_family_without_prompt_rows_gets_no_prompt_argument
  2  Hy4-flash is the one registered family that sets it      test_only_hy4_flash_sets_prompt_rows
  3  Hy4-flash's `forward` with `prompt=True` reads one row on the prompt pass: the state and the logits of the
     prompt read as n - 1 rows and then one equal the cold read's; without it the row is a decode step
                                              test_hy4_one_row_prompt_chunk_is_the_prompt_pass
  4  (R83) the load binds every trunk projection of the prompt pass, and an unequal shape fails it
                                              test_hy4_binds_its_prompt_order_at_load

CPU. Run: python tests/test_prompt_rows.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import test_forward_tree as T  # noqa: E402,F401  (the CPU environment before the engine imports)
import torch  # noqa: E402

from engine import cache  # noqa: E402
from engine.models.adapter import Capabilities  # noqa: E402
from test_cache_any_model import Toy  # noqa: E402

CH = 16


class _Rec(Toy):
    """The test's made-up model, its `forward` calls recorded: (rows, start, the keyword arguments)."""

    def __init__(self, caps=None):
        super().__init__()
        self.calls = []
        if caps is not None:
            self.caps = caps

    def forward(self, ids, start, last_only=True, **kw):
        self.calls.append((int(ids.numel()), int(start), dict(kw, last_only=last_only)))
        return super().forward(ids, start, last_only=last_only)


def _ids(n: int) -> list:
    return [4 + (7 * i) % 50 for i in range(n)]


def test_a_family_without_prompt_rows_gets_no_prompt_argument():
    for caps, want in ((None, {}), (Capabilities(caches=True, side_slots=False), {}),
                       (Capabilities(caches=True, side_slots=False, prompt_rows=True), {"prompt": True})):
        e = _Rec(caps)
        # a prompt of two chunks and a one-row tail at a start past 0, on the grid
        cache.prefill(e, None, _ids(2 * CH + 1), "cpu", chunk=CH)
        assert [(r, s) for r, s, _ in e.calls] == [(CH, 0), (CH, CH), (1, 2 * CH)], e.calls
        for r, s, kw in e.calls:
            assert kw == dict(want, last_only=True), (caps, r, s, kw)
    return "no capability, or False: forward(blk, start=i, last_only=True) as before; True: prompt=True on every chunk"


def test_only_hy4_flash_sets_prompt_rows():
    from engine.models import REGISTRY
    got = {}
    for name, a in REGISTRY.items():
        m = a.manifest_of()                     # imports the family's serving module
        mod = sys.modules.get(f"engine.models.{a.package.split('/')[-1]}.serving") if a.package else None
        holders = [v for v in vars(mod).values() if isinstance(v, type) and isinstance(getattr(v, "caps", None), Capabilities)] \
            if mod else []
        if name == "qwen3_5":
            from engine.model import Qwen38Engine
            import engine.models.qwen3_5.adapter  # noqa: F401  (sets Qwen38Engine.caps)
            holders = [Qwen38Engine]
        assert holders, (name, m.arch)
        got[name] = sorted({bool(h.caps.prompt_rows) for h in holders})
    assert got.pop("hy_v4_flash") == [True], got
    assert all(v == [False] for v in got.values()), got
    return f"hy_v4_flash: True; {sorted(got)}: False"


def _sha(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def _state(srv) -> dict:
    out = {k: _sha(v) for k, v in srv.state.sections().items()}
    for name, t, ax in srv.state.views():
        out[name] = _sha(t.narrow(ax, 0, srv.st.length))
    return out


def test_hy4_one_row_prompt_chunk_is_the_prompt_pass():
    from engine.models.hy4_flash.engine import Hy4FlashEngine
    from engine.models.hy4_flash.serving import Hy4FlashServing
    from tests.hy4_tiny import random_weights, tiny_config
    cfg = tiny_config()
    W, tables = random_weights(cfg)
    eng = Hy4FlashEngine(cfg, W, tables, dtype=torch.bfloat16, max_len=160)
    eng.prefill_v2 = True
    srv = Hy4FlashServing(eng, 160)
    assert Hy4FlashServing.caps.prompt_rows and dataclasses.is_dataclass(Hy4FlashServing.caps)
    ids = _ids(41)
    n = len(ids)
    srv.restart()
    cold = srv.forward(torch.tensor(ids), start=0).float().clone()
    want = _state(srv)
    for prompt in (True, False):
        srv.restart()
        srv.forward(torch.tensor(ids[:n - 1]), start=0)
        steps = srv.stats["decode_steps"]
        lg = srv.forward(torch.tensor(ids[n - 1:]), start=n - 1, prompt=prompt).float().clone()
        took_step = srv.stats["decode_steps"] == steps + 1
        if prompt:
            assert not took_step and torch.equal(lg, cold) and _state(srv) == want, "a one-row prompt chunk is the cold read"
        else:
            assert took_step, "without `prompt` a lone token past 0 is a decode step"
    return f"{n} rows: n - 1 then one on the prompt pass equals the cold read (logits, every section and view)"


def test_hy4_binds_its_prompt_order_at_load():
    """R83: the load checks every trunk projection `lin_prompt` reads, once a shape, and a shape that is not one
    order's bits fails there (`build` calls this before the self-check); the experts' banks, the head and kv_b_proj's
    packed value half are not the prompt pass's and are left out."""
    from exl3_launches import Launches
    from engine.models.hy4_flash.engine import Hy4FlashEngine
    from tools import exl3_linear as X
    from tools import exl3_rows as R

    def blk(N, K, kb):
        return X.EXL3Block(torch.zeros(K // 16, N // 16, X.tile_width(kb), dtype=torch.int16),
                           torch.ones(K, dtype=torch.float16), torch.ones(N, dtype=torch.float16))
    e = object.__new__(Hy4FlashEngine)
    e.kernels, e.device = True, torch.device("cpu")
    e.W = {"layers.0.self_attn.q_a_proj.weight": blk(1024, 3072, 5), "layers.1.self_attn.q_a_proj.weight": blk(1024, 3072, 5),
           "layers.0.self_attn.compressed_indexer.query_projection.weight": blk(2048, 1024, 35),
           "layers.0.self_attn.kv_b_proj.packed": blk(4096, 512, 5), "lm_head.weight": blk(1024, 3072, 6),
           "layers.0.mlp.experts.0.gate_proj.weight": blk(512, 1024, 3), "layers.0.input_layernorm.weight": torch.ones(8)}
    with Launches():
        got = e.bind_prompt_order()
        assert got["weights"] == 3 and got["shapes"] == 2 and got["failed"] == 0, got
    with Launches(checks="fail"):
        try:
            e.bind_prompt_order()
        except R.RowOrderError:
            pass
        else:
            raise AssertionError("an unequal shape must fail the load")
    # a device whose profile builds no inline extension (here the CPU's, kernels claimed): `lin_prompt` runs
    # exl3_matmul there, so nothing is bound and nothing can fail
    assert "skipped" in e.bind_prompt_order()
    e.kernels = False
    assert "skipped" in e.bind_prompt_order()
    return "3 trunk projections, 2 shapes bound; an unequal one fails the load; no binding where the order does not run"


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
