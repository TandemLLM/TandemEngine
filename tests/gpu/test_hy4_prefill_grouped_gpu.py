"""On the board: Hy4-flash's prompt pass on the grouped expert kernels (the default, HY4_PREFILL_GROUPED=1) against
the pair kernels (=0), SPD-311.

  * the same prompt, prefilled once on each: every row's logits and the whole state after it (the sparse layers'
    latents and indexer keys, the window layers' rings, the last tokens) are equal bit for bit;
  * a step after each prompt pass gives equal logits.

The grouped kernels decode an expert's tiles once for up to 8 of its pairs; the pairs' sums are the pair kernels'.
They exist on the device only (the expert banks), so the comparison runs here, not on a CPU.

Needs a pack: HY4_TINY_PACK. Run on the box: python tests/gpu/test_hy4_prefill_grouped_gpu.py   (under the box lock)
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
        assert banks, "the pack's experts are no EXL3 banks: nothing to compare"
        _L["e"] = (pack.cfg, Hy4FlashEngine(pack.cfg, W, tables, device="cuda", max_len=2048, banks=banks, cuts=cuts))
    return _L["e"]


def _pass(eng, ids: list[int], grouped: bool):
    eng.prefill_grouped = grouped
    st = eng.new_state(len(ids) + 8)
    logits = eng.prefill(torch.tensor(ids), st, head="all")
    nxt = eng.rows([int(logits[-1].argmax())], [-1], st)[0]
    torch.cuda.synchronize()
    return logits, st, nxt


def test_the_grouped_prompt_pass_is_the_pair_ones_bit_for_bit():
    if SKIP:
        return SKIP
    cfg, eng = _eng()
    g = torch.Generator().manual_seed(11)
    # past the 256-row launches of the experts, so a chunk's pairs share experts across several launches
    ids = torch.randint(5, min(cfg.vocab_size, 900), (700,), generator=g).tolist()
    was = eng.prefill_grouped
    try:
        la, sa, na = _pass(eng, ids, grouped=False)
        lb, sb, nb = _pass(eng, ids, grouped=True)
    finally:
        eng.prefill_grouped = was
    assert torch.equal(la, lb), float((la - lb).abs().max())
    for f in ("lat_all", "ik_all", "win_all", "hist"):
        assert torch.equal(getattr(sa, f), getattr(sb, f)), f
    assert sa.length == sb.length == len(ids)
    assert torch.equal(na, nb), float((na - nb).abs().max())
    return f"{len(ids)} rows: every row's logits, the state and the next step equal"


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(f"  {name:<56} ok   {fn() or ''}", flush=True)
                passed += 1
            except AssertionError as e:
                print(f"  {name:<56} FAIL {e}", flush=True)
                failed += 1
    print(f"{passed} passed" + (f", {failed} FAILED" if failed else ""))
    sys.exit(1 if failed else 0)
