"""Hy4-flash's weights from a pack into what the engine reads.

  W        name -> a tensor on the device (as stored: fp32 where the reference keeps fp32, the router's fp16,
           BF16 norms and small projections) or an EXL3 projection (`tools/exl3_linear.EXL3Block`)
  cuts     id(weight) -> its real output width, where the converter padded it to 128
  banks    layer -> the routed experts as the MoE kernels read them (a device), and in W each layer's
           `experts.{gate,up,down}_proj` as a bank the reference can dequantise an expert from
  tables   layer -> its Engram table (table.py), checked against the hash the config gives

`kv_b_proj` is dequantised once: the engine reads it as its two absorbed halves, never through a kernel.
"""

from __future__ import annotations

import json
import os
import time

import torch

from engine.models.hy4_flash.engram import EngramHash
from engine.models.hy4_flash.pack import Hy4Pack, plain_names, projections
from engine.models.hy4_flash.table import EngramTable, table_file

#: tensors the reference keeps in fp32 whatever the model's dtype
FP32 = ("hc_fn", "hc_base", "hc_scale", "hc_head_fn", "hc_head_base", "hc_head_scale", "learnable_sink_param",
        "e_score_correction_bias", "key_value_projection.weight", "key_gate_projection.weight", "intra_group_ape")
#: read as stored (the row kernels widen them as they load: half the bytes of an fp32 copy)
STORED = ("mlp.gate.weight",)


#: Hy4-flash's own splits of tools/fn_kernels.dense for its shapes (a split is a summation order, keyed by the shape)
DENSE_SPLITS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                            "ops", "hy4-dense-splits.json")
#: the other families' tables: a shape one of them holds keeps that family's split (their bits must not move)
OTHER_SPLITS = ("fn-dense-splits.json", "lfm25-dense-splits.json", "flint-dense-splits.json")


def apply_splits(path: str = DENSE_SPLITS) -> int:
    """The file's splits into tools/fn_kernels.DENSE_SPLIT, but for shapes no other family's table names; returns
    the entries applied."""
    if not os.path.isfile(path):
        return 0
    from tools import fn_kernels as FK
    taken = set()
    for name in OTHER_SPLITS:
        f = os.path.join(os.path.dirname(path), name)
        if os.path.isfile(f):
            taken |= {k for k in json.load(open(f)) if not k.startswith("_")}
    n = 0
    for k, v in json.load(open(path)).items():
        if k.startswith("_") or k in taken:
            continue
        N, Kin, kb = (int(x) for x in k.split("x"))
        FK.DENSE_SPLIT[(N, Kin, kb)] = (int(v["S"]), int(v["warps"]))
        n += 1
    return n


class PackExperts:
    """One projection of a layer's routed experts in a pack: `weight(e, dtype)` dequantises expert e."""

    dense = None

    def __init__(self, blocks: list):
        self.blocks = blocks

    def weight(self, e: int, dtype) -> torch.Tensor:
        return self.blocks[e].dequant().to(dtype)


def load(path: str, device="cuda", dtype=torch.bfloat16, log=print, table_policy: str = "cold",
         table_cache_bytes: int = 2 << 30, experts: bool = True, splits: bool = True):
    """(pack, W, cuts, tables, banks, timing) of the pack at `path`."""
    t0 = time.time()
    pack = Hy4Pack.open(path)
    notes = pack.audit()
    pack.check_mul1()
    c = pack.cfg
    timing = {"index_s": round(time.time() - t0, 2), "notes": notes}
    W, cuts, banks = {}, {}, {}
    on_gpu = str(device).startswith("cuda")
    if on_gpu and splits:
        timing["dense_splits"] = apply_splits()
    t1 = time.time()

    def proj(base: str, N, out_fp32: bool = False):
        if pack.formats[base] != "exl3":
            return pack.read(base + ".weight").to(device, dtype).contiguous()
        if base.endswith("kv_b_proj"):
            w = pack.exl3(base, device).dequant()
            return w[:N].to(dtype).contiguous() if N is not None else w.to(dtype)
        w = pack.exl3(base, device, out_fp32=out_fp32)
        if N is not None and w.N != N:
            cuts[id(w)] = N
        return w
    for base, N, _ in projections(c, experts=False, dfly_mlp=pack.dfly_mlp):
        W[base + ".weight"] = proj(base, N, out_fp32=base == "lm_head")
    for name, _ in plain_names(c, dfly=pack.has_dfly):
        t = pack.read(name)
        if t.is_floating_point() and not (name.endswith(STORED) and dtype == torch.bfloat16):
            t = t.to(torch.float32 if name.endswith(FP32) and not name.startswith("dfly_layers.") else dtype)
        W[name] = t.to(device).contiguous()
    timing["dense_s"] = round(time.time() - t1, 2)
    t1 = time.time()
    if experts:
        for li in range(c.num_hidden_layers):
            if c.is_dense_mlp(li):
                continue
            p = f"layers.{li}.mlp.experts."
            if pack.formats[p + "0.gate_proj"] == "exl3":
                blocks = [tuple(pack.exl3(f"{p}{e}.{w}", device) for w in ("gate_proj", "up_proj", "down_proj")) for e in range(c.num_experts)]
                for j, w in enumerate(("gate_proj", "up_proj", "down_proj")):
                    W[p + w] = PackExperts([b[j] for b in blocks])
                if on_gpu:
                    from tools.fn_kernels import MoEBank
                    banks[li] = MoEBank([tuple((b.t, b.suh[0], b.svh) for b in e) for e in blocks], device)
            else:
                from engine.models.flashnext.ref import ExpertBank
                for w in ("gate_proj", "up_proj", "down_proj"):
                    W[p + w] = ExpertBank(dense=torch.stack([pack.read(f"{p}{e}.{w}.weight") for e in range(c.num_experts)]).to(device, dtype))
    timing["experts_s"] = round(time.time() - t1, 2)
    tables = {}
    for li, lid in c.engram_layers.items():
        t = EngramTable(table_file(pack.path, li), li)
        t.check(EngramHash.from_config(c, lid, W[f"layers.{li}.engram_layer.lookup_table"]))
        t.set_policy(table_policy, max(table_cache_bytes // max(len(c.engram_layers), 1), 1 << 20))
        tables[li] = t
    pack.close()
    timing["total_s"] = round(time.time() - t0, 2)
    log(f"[hy4] pack {path}: {'; '.join(notes)}; {len(tables)} Engram tables ({table_policy}); loaded in {timing['total_s']} s")
    return pack, W, cuts, tables, banks, timing
