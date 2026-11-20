"""SPD-323 (the prefill design's F1): the prompt pass's attention, `mla_rows`, through its torch reference on the CPU.

  * `mla_rows_reference` with the engine's absorbed key and value halves is the reference forward's attention
    (`Hy4FlashRef.attend`, which expands every latent into keys and values) within fp32 rounding, for rows that read
    every row before them, a window of rows, and a list of rows;
  * a row's attention is its own: a row alone and the same row among 300 give the same bits.

The kernel itself runs on a device: tests/gpu/test_hy4_mla_rows_gpu.py holds it to this reference and to its digests.
Run: python tests/test_hy4_kernels.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.hy4_flash import engine as E  # noqa: E402
from engine.models.hy4_flash import kernels as K  # noqa: E402
from engine.models.hy4_flash.config import SPARSE  # noqa: E402
from engine.models.hy4_flash.ref import Hy4FlashRef  # noqa: E402
from tests.hy4_tiny import random_weights, tiny_config  # noqa: E402

_T = {}


def tiny():
    if not _T:
        cfg = tiny_config()
        W, tables = random_weights(cfg)
        _T["m"] = (cfg, Hy4FlashRef(cfg, W, tables, dtype=torch.float32), E.Hy4FlashEngine(cfg, W, tables, dtype=torch.float32, max_len=128))
    return _T["m"]


def _rows(N: int, T: int, Wn: int, g: torch.Generator) -> tuple:
    """T rows at positions in [N/2, N) over N latents: a third read every row up to theirs, a third a window of Wn
    rows, a third a sorted random list of rows up to theirs (their own last). -> positions, ctl, sel, allowed [T, N]."""
    pos = torch.randint(N // 2, N, (T,), generator=g)
    ctl = torch.zeros(T, 3, dtype=torch.int32)
    sel = torch.zeros(T, N, dtype=torch.int32)
    allowed = torch.zeros(T, N, dtype=torch.bool)
    for t in range(T):
        p = int(pos[t])
        kind = t % 3
        if kind == 0:
            ctl[t] = torch.tensor([0, p + 1, 0])
            allowed[t, :p + 1] = True
        elif kind == 1:
            n = min(Wn, p + 1)
            ctl[t] = torch.tensor([p - n + 1, n, 0])
            allowed[t, p - n + 1:p + 1] = True
        else:
            keep = torch.randperm(p, generator=g)[:max(1, p // 2)].sort().values.tolist() + [p]
            ctl[t] = torch.tensor([0, len(keep), 1])
            sel[t, :len(keep)] = torch.tensor(keep, dtype=torch.int32)
            allowed[t, keep] = True
    return pos, ctl, sel, allowed


def test_mla_reference_is_the_reference_attention():
    cfg, ref, eng = tiny()
    g = torch.Generator().manual_seed(5)
    out = []
    for li in (cfg.sparse_layers[0], cfg.window_layers[0]):
        a, R = cfg.attn(li), cfg.kv_lora_rank
        N, T = 40, 24
        lat = torch.randn(N, R + a.rope, generator=g)
        q = torch.randn(T, a.heads, a.qk, generator=g)
        pos, ctl, sel, allowed = _rows(N, T, cfg.sliding_window, g)
        want = ref.attend(q, lat[:, :R], lat[:, R:], allowed, li)
        qa = torch.einsum("thd,hdr->thr", q[..., :a.nope], eng.wk[li])
        mix = K.mla_rows_reference(qa, q[..., a.nope:].contiguous(), lat, ctl, sel, eng.sinks[li].reshape(-1), a.qk ** -0.5)
        got = torch.einsum("thr,hrv->thv", mix, eng.wv[li])
        err = (got - want).abs().max().item()
        assert err < 1e-4, (li, err)
        out.append(f"{'sparse' if cfg.layer_types[li] == SPARSE else 'window'} layer {li}: {err:.1e}")
    return "dense, window and list rows against the reference's attention, fp32: " + ", ".join(out)


def test_a_rows_attention_is_its_own():
    cfg, ref, eng = tiny()
    g = torch.Generator().manual_seed(7)
    li = cfg.sparse_layers[0]
    a, R = cfg.attn(li), cfg.kv_lora_rank
    N, T = 400, 300
    lat = torch.randn(N, R + a.rope, generator=g).to(torch.bfloat16)
    qa = torch.randn(T, a.heads, R, generator=g).to(torch.bfloat16)
    qr = torch.randn(T, a.heads, a.rope, generator=g).to(torch.bfloat16)
    pos, ctl, sel, _ = _rows(N, T, cfg.sliding_window, g)
    sink = eng.sinks[li].reshape(-1)
    full = K.mla_rows_reference(qa, qr, lat, ctl, sel, sink, a.qk ** -0.5)
    for j in (0, 1, 2, 149, 299):
        alone = K.mla_rows_reference(qa[j:j + 1], qr[j:j + 1], lat, ctl[j:j + 1], sel[j:j + 1], sink, a.qk ** -0.5)
        assert torch.equal(alone[0], full[j]), j
    return "5 rows alone equal the same rows among 300, bf16"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<52} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<52} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
