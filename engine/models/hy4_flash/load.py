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


_APPLIED: set = set()


def apply_splits(path: str = DENSE_SPLITS) -> int:
    """The file's splits into tools/fn_kernels.DENSE_SPLIT, but only for shapes no other family's table names and
    no entry already there holds (another family's split is its summation order: it never moves); returns the
    entries applied."""
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
        key = tuple(int(x) for x in k.split("x"))
        if key in FK.DENSE_SPLIT and key not in _APPLIED:
            continue
        FK.DENSE_SPLIT[key] = (int(v["S"]), int(v["warps"]))
        _APPLIED.add(key)
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
         table_cache_bytes: int = 2 << 30, experts: bool = True, splits: bool = True, skip_dfly: bool = False):
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
            blk = pack.exl3(base, device)
            if on_gpu and base.startswith("layers."):
                W[base + ".packed"] = blk               # the row path reads the value half packed (native.vdense)
            w = blk.dequant()
            return w[:N].to(dtype).contiguous() if N is not None else w.to(dtype)
        w = pack.exl3(base, device, out_fp32=out_fp32)
        if N is not None and w.N != N:
            cuts[id(w)] = N
        return w
    for base, N, _ in projections(c, experts=False, dfly_mlp=0 if skip_dfly else pack.dfly_mlp):
        W[base + ".weight"] = proj(base, N, out_fp32=base == "lm_head")
    for name, _ in plain_names(c, dfly=False if skip_dfly else pack.has_dfly):
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


DFLY_FP32 = ("learnable_sink_param", "expert_bias", "e_score_correction_bias",
             "block_sink", "block_rel_bias", "block_gate_bias", "mlp.gate.weight")


def _open_safetensors_index(snap: str) -> tuple[dict[str, str], dict]:
    from safetensors import safe_open
    idx_path = os.path.join(snap, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        with open(idx_path) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    elif os.path.isfile(os.path.join(snap, "model.safetensors")):
        files = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"{snap}: no model.safetensors or model.safetensors.index.json")
    handles = {fn: safe_open(os.path.join(snap, fn), framework="pt", device="cpu") for fn in files}
    keymap: dict[str, tuple[str, object]] = {}
    for fn, h in handles.items():
        for raw_k in h.keys():
            canon = raw_k[6:] if raw_k.startswith("model.") else raw_k
            keymap[canon] = (raw_k, h)
    return keymap, handles


def _read_exl3_from_map(keymap: dict, base: str, device, out_fp32: bool = False):
    from tools.exl3_linear import EXL3Block, unpack_signs

    def get(suffix: str):
        raw_k, h = keymap[f"{base}.{suffix}"]
        return h.get_tensor(raw_k).to(device)

    def sign(s: str, alt: str):
        if f"{base}.{s}" in keymap:
            return get(s)
        return unpack_signs(get(alt)).to(device)

    return EXL3Block(get("trellis"), sign("suh", "su"), sign("svh", "sv"), out_fp32=out_fp32)


def _dfly_expected_widths(d, target_cfg) -> dict[str, int]:
    """Expected unpadded output widths N for every external DFly2 projection."""
    H, R, qr, a = target_cfg.hidden_size, target_cfg.kv_lora_rank, target_cfg.q_lora_rank, target_cfg.sparse_attn
    Im, Lz, Is = d.moe_intermediate_size, d.latent_size, d.shared_intermediate_size
    hc_i = d.hidden_correction_intermediate_size or H
    out = {"mtp_fusion.eh_proj": H, "hidden_correction.gate_proj": hc_i,
           "hidden_correction.up_proj": hc_i, "hidden_correction.down_proj": H}
    for i in range(d.layers):
        s = f"layers.{i}.self_attn."
        out.update({s + "q_a_proj": qr, s + "q_b_proj": a.heads * a.qk, s + "kv_a_proj_with_mqa": R + a.rope,
                    s + "kv_b_proj": a.heads * (a.nope + a.v), s + "linear_gate": a.heads * a.v, s + "o_proj": H})
        m = f"layers.{i}.mlp."
        if d.mlp_type == "moe":
            out.update({m + "routed_expert_down_proj": Lz, m + "routed_expert_up_proj": H,
                        m + "shared_experts.gate_proj": Is, m + "shared_experts.up_proj": Is, m + "shared_experts.down_proj": H})
        else:
            mi = d.intermediate_size or Im
            out.update({m + "gate_proj": mi, m + "up_proj": mi, m + "down_proj": H})
    if d.attentive_correction:
        s = "hidden_correction.attention."
        out.update({s + "q_in": qr, s + "q_b_proj": a.heads * a.qk, s + "kv_in": R + a.rope,
                    s + "kv_b_proj": a.heads * (a.nope + a.v), s + "o_proj": H, s + "block_gate": a.heads})
    return out


def load_dfly_external(path: str, target_cfg, device="cuda", dtype=torch.bfloat16, log=print):
    """Load standalone `HY4DFlyDraftModel` (BF16 AngelSlim or EXL3 pack) into `(DFlyConfig, W, cuts, banks)`."""
    from engine.models.flashnext.ref import ExpertBank
    from engine.models.hy4_flash.config import load_dfly_config
    t0 = time.time()
    d = load_dfly_config(path, target_cfg)
    keymap, _ = _open_safetensors_index(d.path)
    W, cuts, banks = {}, {}, {}
    on_gpu = str(device).startswith("cuda")
    widths = _dfly_expected_widths(d, target_cfg)
    exl3_bases = {k[:-len(".trellis")] for k in keymap if k.endswith(".trellis")}

    def read(name: str):
        raw_k, h = keymap[name]
        return h.get_tensor(raw_k)

    for base in sorted(exl3_bases):
        if ".mlp.experts." in base:
            continue
        N = widths.get(base)
        if base.endswith("kv_b_proj"):
            w = _read_exl3_from_map(keymap, base, device).dequant()
            W[base + ".weight"] = (w[:N] if N is not None else w).to(dtype).contiguous()
        else:
            blk = _read_exl3_from_map(keymap, base, device)
            if N is not None and blk.N != N:
                cuts[id(blk)] = N
            W[base + ".weight"] = blk

    if d.mlp_type == "moe":
        for i in range(d.layers):
            p = f"layers.{i}.mlp.experts."
            if f"{p}0.gate_proj" in exl3_bases:
                blocks = [tuple(_read_exl3_from_map(keymap, f"{p}{e}.{w}", device)
                                for w in ("gate_proj", "up_proj", "down_proj"))
                          for e in range(d.num_experts)]
                for j, w in enumerate(("gate_proj", "up_proj", "down_proj")):
                    W[p + w] = PackExperts([b[j] for b in blocks])
                if on_gpu:
                    from tools.fn_kernels import MoEBank
                    banks[i] = MoEBank([tuple((b.t, b.suh[0], b.svh) for b in e) for e in blocks], device)
            elif f"{p}gate_up_proj" in keymap and f"{p}down_proj" in keymap:
                # AngelSlim HY4DFlyDraftModel stores [E, 2*Im, Lz] (gate first) and [E, Lz, Im] ([E, out, in]).
                gu = read(p + "gate_up_proj")
                dn = read(p + "down_proj")
                Im, Lz = d.moe_intermediate_size, d.latent_size
                if gu.shape != (d.num_experts, 2 * Im, Lz) or dn.shape != (d.num_experts, Lz, Im):
                    raise ValueError(
                        f"DFly2 layer {i} expert shapes mismatch: gate_up_proj={tuple(gu.shape)}, "
                        f"down_proj={tuple(dn.shape)}, expected ({d.num_experts}, {2 * Im}, {Lz}) and ({d.num_experts}, {Lz}, {Im})"
                    )
                W[p + "gate_proj"] = ExpertBank(dense=gu[:, :Im, :].to(device, dtype).contiguous())
                W[p + "up_proj"] = ExpertBank(dense=gu[:, Im:, :].to(device, dtype).contiguous())
                W[p + "down_proj"] = ExpertBank(dense=dn.to(device, dtype).contiguous())
            else:
                for w in ("gate_proj", "up_proj", "down_proj"):
                    W[p + w] = ExpertBank(dense=torch.stack([read(f"{p}{e}.{w}.weight") for e in range(d.num_experts)]).to(device, dtype))

    skip_suffixes = (".trellis", ".suh", ".svh", ".su", ".sv", ".mul1")
    for k in sorted(keymap):
        if k.endswith(skip_suffixes) or ".mlp.experts." in k or k + ".weight" in W:
            continue
        t = read(k)
        if t.is_floating_point():
            t = t.to(torch.float32 if k.endswith(DFLY_FP32) else dtype)
        W[k] = t.to(device).contiguous()

    dt = round(time.time() - t0, 2)
    fmt = "exl3" if exl3_bases else "bf16"
    log(f"[hy4] external DFly2 ({fmt}, {d.layers}L {d.mlp_type}) {d.path}: loaded in {dt} s")
    return d, W, cuts, banks


def load_draft_head_exl3(path: str, target_cfg, device="cuda", log=print):
    """Load an EXL3 `lm_head` block from a checkpoint directory or single `.safetensors` file."""
    from safetensors import safe_open
    from tools.exl3_linear import EXL3Block, unpack_signs
    p = os.path.expanduser(path)
    if os.path.isfile(p):
        files = [p]
    else:
        from engine.config import resolve_snapshot
        snap = resolve_snapshot(p)
        files = sorted(os.path.join(snap, f) for f in os.listdir(snap) if f.endswith(".safetensors"))
    for fp in files:
        with safe_open(fp, framework="pt", device=str(device)) as f:
            keys = set(f.keys())
            for base in ("lm_head", "model.lm_head"):
                if base + ".trellis" in keys:
                    def sign(s, alt):
                        return f.get_tensor(f"{base}.{s}") if f"{base}.{s}" in keys else unpack_signs(f.get_tensor(f"{base}.{alt}")).to(device)
                    blk = EXL3Block(f.get_tensor(base + ".trellis"), sign("suh", "su"), sign("svh", "sv"), out_fp32=True)
                    log(f"[hy4] draft head from {path}: K{blk.kb}, {blk.nbytes / 1e9:.2f} GB")
                    return blk, (target_cfg.vocab_size if blk.N != target_cfg.vocab_size else None)
    raise RuntimeError(f"no lm_head.trellis in {path}")
