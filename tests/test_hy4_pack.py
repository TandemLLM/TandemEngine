"""Hy4-flash's pack reader, Engram table reader and loader.

Without a pack: a table file written here in the pack's format decodes to the rows a plain restatement of
the ring gives, through both read policies, and the hash multipliers equal the ones the converter recorded
for the released checkpoint. With HY4_TINY_PACK (an EXL3 pack of a tiny random model made by the converter)
and HY4_TINY_SRC (its BF16 source): the header audit, the tables against the source's rows, and the engine
on the pack's weights against the reference forward on the same weights.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.models.hy4_flash.engram import fp8_round_trip, hash_multipliers  # noqa: E402
from engine.models.hy4_flash.table import PREFIX, EngramTable, mul1_codebook  # noqa: E402

PACK, SRC = os.environ.get("HY4_TINY_PACK"), os.environ.get("HY4_TINY_SRC")
SKIP = None if PACK and os.path.isdir(PACK) else "skipped: HY4_TINY_PACK names no pack"


def _write_safetensors(path: str, tensors: dict, meta: dict) -> None:
    hdr, blobs, at = {"__metadata__": meta}, [], 0
    names = {torch.int16: "I16", torch.int64: "I64", torch.float16: "F16"}
    for k, t in tensors.items():
        b = t.contiguous().numpy().tobytes()
        hdr[k] = {"dtype": names[t.dtype], "shape": list(t.shape), "data_offsets": [at, at + len(b)]}
        blobs.append(b)
        at += len(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_a_table_file_decodes_to_the_ring_it_holds():
    g = torch.Generator().manual_seed(1)
    K, dim, sizes = 5, 96, [37, 41, 43, 47]
    rows = sum(sizes)
    sym = torch.randint(0, 1 << K, (rows, dim), generator=g)
    scale = (torch.rand(rows, generator=g) * 0.1 + 0.01).to(torch.float16)
    bias = (torch.randn(len(sizes), dim, generator=g) * 0.01).to(torch.float16)
    # the stream: position i's symbol at bits [K i, K i + K), little-endian 16-bit words
    bits = ((sym.unsqueeze(-1) >> torch.arange(K)) & 1).reshape(rows, dim * K)
    words = (bits.reshape(rows, -1, 16) << torch.arange(16)).sum(-1).to(torch.int32)
    packed = torch.cat((scale.view(torch.int16).view(-1, 1).to(torch.int32), words), dim=1)
    packed = torch.where(packed >= 1 << 15, packed - (1 << 16), packed).to(torch.int16)
    offsets = torch.tensor([0] + sizes[:-1]).cumsum(0)
    with tempfile.TemporaryDirectory() as d:
        path, p = os.path.join(d, "engram.layers.3.safetensors"), PREFIX.format(3)
        _write_safetensors(path, {p + "head_bias": bias, p + "head_offsets": offsets, p + "head_vocab_sizes": torch.tensor(sizes),
                                  p + "layer_multipliers": torch.tensor([3, 5, 7]), p + "trellis": packed},
                           {"format": "exl3_ngram_trellis", "version": "1", "K": str(K), "codebook": "mul1", "row_dim": str(dim),
                            "rows": str(rows), "target": "fp8_e4m3_row_round_trip", "layer_id": "4"})
        t = EngramTable(path, 3)
        assert (t.rows, t.K, t.dim, t.words, t.fp8_done) == (rows, K, dim, 31, True)
        ids = torch.randint(0, rows, (300,), generator=g)
        got = t.gather(ids, dtype=torch.float32)
        # a state is the last 16 bits of the symbols ending at its position: the newest symbol lowest
        from engine.models.flashnext.ple import mul1_codebook as fn_codebook
        cb = mul1_codebook().float()
        assert torch.equal(mul1_codebook(), fn_codebook())            # one codebook, two readers
        want = torch.empty(300, dim)
        for n, r in enumerate(ids.tolist()):
            head = max(h for h in range(len(sizes)) if offsets[h] <= r)
            for i in range(dim):
                state = 0
                for back in range(4):
                    state |= int(sym[r, (i - back) % dim]) << (K * back)
                want[n, i] = cb[state & 0xFFFF] * scale[r].float() + bias[head, i].float()
        assert torch.equal(got, want)
        t.set_policy("cold", 1 << 20)
        assert t.policy == "cold" and torch.equal(t.gather(ids, dtype=torch.float32), want)
        rep = t.report()
        assert rep["policy"] == "cold" and rep["rows"] == rows
        t.set_policy("warm")
        try:
            t.gather(torch.tensor([rows]))
            raise SystemExit("a row past the table was read")
        except IndexError:
            pass
    return f"{rows} rows of 96 at K 5 (31 words a row): 300 rows equal through the page cache and through the cold reader"


def test_the_released_multipliers():
    """The four tables' hash multipliers as the converter recorded them (lookup table onto 92,620 ids, seed 0)."""
    want = {2: [81853575799317, 39690085502883, 24229915018165], 12: [19457948045719, 44729556510419, 84779352922315],
            22: [72974230057307, 21530602121221, 55998057243097], 32: [29128509562787, 16745155128523, 22474588302733]}
    for lid, m in want.items():
        assert hash_multipliers(92620, 3, lid, 0) == m, lid
    return "layer ids 2, 12, 22 and 32 equal"


_L = {}


def _loaded():
    if "x" not in _L:
        from engine.models.hy4_flash.load import load
        _L["x"] = load(PACK, "cpu", log=lambda s: None, table_policy="warm")
    return _L["x"]


def test_the_tiny_pack_audits_and_loads():
    if SKIP:
        return SKIP
    pack, W, cuts, tables, banks, timing = _loaded()
    c = pack.cfg
    assert sorted(tables) == sorted(c.engram_layers) and all(t.fp8_done and t.K == 5 for t in tables.values())
    kva = W["layers.0.self_attn.kv_a_proj_with_mqa.weight"]
    assert cuts[id(kva)] == c.kv_lora_rank + c.sparse_attn.rope and kva.N % 128 == 0
    assert torch.is_tensor(W["layers.2.self_attn.kv_b_proj.weight"]) and W["layers.1.mlp.gate.weight"].dtype in (torch.bfloat16, torch.float16)      # read as stored
    assert W["layers.1.hc_mlp_layer.hc_pre.hc_fn"].dtype == torch.float32
    assert W["layers.1.mlp.experts.gate_proj"].weight(3, torch.float32).shape == (c.moe_intermediate_size, c.latent_size)
    # a missing tensor and a truncated shard stop the load by name
    from engine.models.hy4_flash.pack import Hy4Pack
    broken = Hy4Pack.open(PACK)
    del broken.tensors["layers.3.input_layernorm.weight"]
    try:
        broken.audit()
        raise SystemExit("a missing tensor passed the audit")
    except ValueError as e:
        assert "layers.3.input_layernorm.weight" in str(e)
    return f"{len(pack.tensors)} tensors; {'; '.join(timing['notes'])}; DFly layers: {pack.has_dfly}"


def test_the_splits_leave_other_entries_alone():
    import tempfile
    from tools import fn_kernels as FK
    from engine.models.hy4_flash import load as L
    before = dict(FK.DENSE_SPLIT)
    held = next(iter(before)) if before else (7, 7, 3)
    FK.DENSE_SPLIT.setdefault(held, (1, 4))
    was = FK.DENSE_SPLIT[held]
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "splits.json")
        with open(f, "w") as o:
            json.dump({"x".join(str(v) for v in held): {"S": 9, "warps": 9}, "11x13x5": {"S": 2, "warps": 8}}, o)
        try:
            assert L.apply_splits(f) == 1 and FK.DENSE_SPLIT[held] == was and FK.DENSE_SPLIT[(11, 13, 5)] == (2, 8)
            assert L.apply_splits(f) == 1                     # its own entries apply again
        finally:
            L._APPLIED.discard((11, 13, 5))
            FK.DENSE_SPLIT.clear()
            FK.DENSE_SPLIT.update(before)
    return "an entry another family holds keeps its split; a new shape takes the file's"


def test_the_tables_hold_the_sources_rows():
    if SKIP or not (SRC and os.path.isdir(SRC)):
        return SKIP or "skipped: HY4_TINY_SRC names no checkpoint"
    from safetensors import safe_open
    pack, W, cuts, tables, banks, timing = _loaded()
    with open(os.path.join(SRC, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    errs = []
    for li, t in tables.items():
        key = f"model.layers.{li}.engram_layer.multi_head_embedding.embedding.weight"
        with safe_open(os.path.join(SRC, wm[key]), "pt") as f:
            src = fp8_round_trip(f.get_tensor(key).to(torch.bfloat16)).float()
        got = t.gather(torch.arange(t.rows), dtype=torch.float32)
        assert got.shape == src.shape
        errs.append(((got - src).norm() / src.norm()).item())
    assert max(errs) < 0.08, errs
    return "relative error of the K 5 rows against the source's e4m3-rounded rows: " + ", ".join(f"{e:.4f}" for e in errs)


def test_the_engine_on_the_pack_matches_the_reference_on_it():
    if SKIP:
        return SKIP
    from engine.models.hy4_flash.engine import Hy4FlashEngine
    from engine.models.hy4_flash.ref import Hy4FlashRef, Hy4State
    pack, W, cuts, tables, banks, timing = _loaded()
    c = pack.cfg
    g = torch.Generator().manual_seed(3)
    toks = torch.randint(3, c.vocab_size, (120,), generator=g)
    out = {}
    for dtype in (torch.float32, torch.bfloat16):
        ref = Hy4FlashRef(c, W, tables, dtype=dtype, cuts=cuts)
        eng = Hy4FlashEngine(c, W, tables, dtype=dtype, max_len=256, cuts=cuts)
        want = ref.forward(toks, Hy4State())
        st = eng.new_state()
        got = torch.cat((eng.prefill(toks[:100], st, head="all"), torch.cat([eng.step(int(t), st) for t in toks[100:]])))
        out[dtype] = ((got - want).abs().max().item(), (got.argmax(-1) == want.argmax(-1)).float().mean().item())
        if dtype == torch.bfloat16:
            res = eng.self_check(toks[:70].tolist(), steps=12, widths=(2, 4, 12), log=lambda s: None)
            assert res["ok"], res
    assert out[torch.float32][0] < 5e-4 and out[torch.float32][1] == 1.0, out
    assert out[torch.bfloat16][1] >= 0.9, out
    return (f"120 rows (window {c.sliding_window}, selection past {c.index_topk}): fp32 max err {out[torch.float32][0]:.1e}; "
            f"bf16 next-token agreement {out[torch.bfloat16][1]:.2f}; bf16 block rows equal the decode steps'")


def test_the_server_loads_the_pack_and_completes():
    """The family's builder through the server's one loader, then a raw completion through the request loop,
    twice: the second answer comes over the caches and is the same text."""
    if SKIP:
        return SKIP
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_forward_tree as T  # noqa: F401  (the CPU environment before the server imports)
    from server import app
    from test_app_loop import Req
    saved = dict(app.STATE)
    try:
        app._load(app.parser().parse_args(["--model", PACK, "--max-len", "512", "--drafter", "none", "--prompt-token-cache", "0"]))
        info = app.STATE["weights_info"]
        assert info["format"] == "exl3" and info["self_check"]["ok"] and app.STATE["device"] == "cpu", info
        body = {"prompt": "".join(chr(0x61 + i % 26) for i in range(90)), "max_tokens": 12, "stream": False, "temperature": 0}
        out = []
        for _ in range(2):
            head, raw = Req("/v1/completions", body).response()
            assert head.startswith("HTTP/1.1 200"), (head, raw[:300])
            r = json.loads(raw)
            assert r["usage"]["completion_tokens"] == 12, r["usage"]
            out.append(r["choices"][0]["text"])
        assert out[0] == out[1], out
        eng = app.STATE["engine"]
    finally:
        app.STATE.clear()
        app.STATE.update(saved)
    return f"loaded through _load on the CPU, self-check {info['self_check']['rows']} rows equal, 12 tokens twice ({eng.stats})"


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
