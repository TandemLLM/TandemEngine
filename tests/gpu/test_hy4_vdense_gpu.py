"""On the board: Hy4-flash's packed value half (engine/models/hy4_flash/native.py `vdense`).

  * every head's input through the head's value columns of the packed kv_b_proj equals the BF16 copy's product
    within the packed arithmetic's rounding, on a sparse and a window layer (heads whose slices share a
    128-column block);
  * a row of an M-row call is its own one-row call bit for bit;
  * the engine's row pass with the packed value half: block rows equal the decode steps through the graphs.

Needs a pack: HY4_TINY_PACK. Run on the box: python tests/gpu/test_hy4_vdense_gpu.py   (under flock ~/.qwen38-box.flock)
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

PACK = os.environ.get("HY4_TINY_PACK", "")
SKIP = None if PACK and os.path.isdir(PACK) and torch.cuda.is_available() else "skipped: HY4_TINY_PACK names no pack, or no device"
_L = {}


def _eng():
    if "e" not in _L:
        from engine.models.hy4_flash.engine import Hy4FlashEngine
        from engine.models.hy4_flash.load import load
        pack, W, cuts, tables, banks, timing = load(PACK, "cuda", log=lambda s: None, table_policy="cold")
        _L["e"] = (pack.cfg, Hy4FlashEngine(pack.cfg, W, tables, device="cuda", max_len=1024, banks=banks, cuts=cuts))
    return _L["e"]


def test_the_packed_value_half_is_the_bf16_one():
    if SKIP:
        return SKIP
    from engine.models.hy4_flash import native as NV
    cfg, eng = _eng()
    assert len(eng.vpk) == cfg.num_hidden_layers, sorted(eng.vpk)
    g = torch.Generator().manual_seed(2)
    worst, kinds = 0.0, set()
    for li in range(cfg.num_hidden_layers):
        a, R = cfg.attn(li), cfg.kv_lora_rank
        x = torch.randn(6 * a.heads, R, generator=g).to("cuda", torch.bfloat16)
        y = NV.vdense(x, eng.vpk[li], a.heads, a.nope + a.v, a.nope, a.v)
        want = torch.einsum("br,brv->bv", x.float(), eng.wv[li].float().repeat(6, 1, 1))
        worst = max(worst, float((y.float() - want).norm() / want.norm()))
        one = torch.cat([NV.vdense(x[m * a.heads:(m + 1) * a.heads].contiguous(), eng.vpk[li], a.heads, a.nope + a.v, a.nope, a.v)
                         for m in range(6)])
        assert torch.equal(one, y), li
        kinds.add((a.heads, (a.nope + a.v) % 128 != 0))
    assert worst < 0.01, worst
    return f"{cfg.num_hidden_layers} layers ({sorted(kinds)}: heads, slices sharing a block): worst relative error {worst:.4f}; rows equal their own calls"


def test_the_row_pass_with_it():
    if SKIP:
        return SKIP
    cfg, eng = _eng()
    g = torch.Generator().manual_seed(4)
    ids = torch.randint(5, min(cfg.vocab_size, 900), (140,), generator=g).tolist()
    r = eng.self_check(ids, steps=8, log=lambda s: None, graphs=True)
    assert r["ok"], r["bad"][:3]
    # the switch off gives the BF16 half: another rounding, the same next token on most rows
    st = eng.new_state(160)
    eng.prefill(torch.tensor(ids[:120]), st, head=None)
    a = eng.rows(ids[120:124], [-1, 0, 1, 2], st)[0]
    eng.value_packed = False
    b = eng.rows(ids[120:124], [-1, 0, 1, 2], st)[0]
    eng.value_packed = True
    d = float((a - b).abs().mean() / b.abs().mean())
    assert 0 < d < 0.1, d
    return f"self-check {r['rows']} rows equal through {r['graphs']} graphs; against the BF16 half the logits move {d:.4f} of their mean size"


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(f"  {name:<44} ok   {fn() or ''}", flush=True)
                passed += 1
            except AssertionError as e:
                print(f"  {name:<44} FAIL {e}", flush=True)
                failed += 1
    print(f"{passed} passed" + (f", {failed} FAILED" if failed else ""))
    sys.exit(1 if failed else 0)
