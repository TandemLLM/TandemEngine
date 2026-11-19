"""The Hy4-flash weights on disk: an EXL3 pack (or a BF16 checkpoint with the same names), one index over
its shards.

`Hy4Pack.open(dir)` reads every safetensors header (nothing else) and names every tensor by its stored
name without the `model.` prefix. A projection is EXL3 (`X.trellis` int16 [Kin/16, N/16, 16 K], `X.suh`,
`X.svh` fp16, `X.mul1`) or plain (`X.weight`) per tensor, so a pack that keeps one unquantised is read as
it is. The converter pads a projection's sides to multiples of 128 (the 576-wide `kv_a_proj_with_mqa` is
stored 640 wide); `projections(cfg)` gives the real sizes and the loader cuts the output back.

The routed experts are one EXL3 group an expert and projection, `layers.L.mlp.experts.<e>.{gate_proj,
up_proj,down_proj}` (the checkpoint's fused `gate_up_proj` split: the gate first). The four Engram tables
are files of their own beside the shards (table.py).

`audit()` stops by name on a missing tensor, a wrong shape, a rate the kernels do not take, another
codebook and a truncated shard, before anything loads.
"""

from __future__ import annotations

import json
import os
import struct
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch

from engine.models.hy4_flash.config import Hy4FlashConfig, config_from_dict
from engine.models.hy4_flash.table import table_file

PREFIX = "model."
MUL1 = 0x83DCD12D
RATES = (1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I16": torch.int16,
      "I32": torch.int32, "I64": torch.int64, "U8": torch.uint8, "I8": torch.int8}
DSIZE = {"BF16": 2, "F16": 2, "F32": 4, "I16": 2, "I32": 4, "I64": 8, "U8": 1, "I8": 1}
#: tensors of the checkpoint this engine does not read
SKIP = ("vit.", "dfly_layers.confidence_head.")


def canonical(key: str) -> str:
    return key[len(PREFIX):] if key.startswith(PREFIX) else key


def pad128(n: int) -> int:
    return -(-n // 128) * 128


@dataclass
class TensorInfo:
    name: str
    file: str
    dtype: str
    shape: tuple
    offset: int
    nbytes: int


def attn_projections(cfg: Hy4FlashConfig, p: str, a) -> list[tuple[str, int, int]]:
    H, R, qr = cfg.hidden_size, cfg.kv_lora_rank, cfg.q_lora_rank
    return [(p + "q_a_proj", qr, H), (p + "q_b_proj", a.heads * a.qk, qr), (p + "kv_a_proj_with_mqa", R + a.rope, H),
            (p + "kv_b_proj", a.heads * (a.nope + a.v), R), (p + "linear_gate", a.heads * a.v, H), (p + "o_proj", H, a.heads * a.v)]


def projections(cfg: Hy4FlashConfig, experts: bool = True, dfly_mlp: int | None = None) -> list[tuple[str, int, int]]:
    """Every weight the model multiplies by, as (base name, N, Kin)."""
    H, Lz, Im, Is = cfg.hidden_size, cfg.latent_size, cfg.moe_intermediate_size, cfg.shared_intermediate_size
    out = [("lm_head", cfg.vocab_size, H)]
    for li in range(cfg.num_hidden_layers):
        p = f"layers.{li}."
        out += attn_projections(cfg, p + "self_attn.", cfg.attn(li))
        if cfg.is_sparse(li):
            out.append((p + "self_attn.compressed_indexer.query_projection", cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank))
        m = p + "mlp."
        if cfg.is_dense_mlp(li):
            I = cfg.intermediate_size
            out += [(m + "gate_proj", I, H), (m + "up_proj", I, H), (m + "down_proj", H, I)]
        else:
            out += [(m + "shared_experts.gate_proj", Is, H), (m + "shared_experts.up_proj", Is, H), (m + "shared_experts.down_proj", H, Is),
                    (m + "routed_expert_down_proj", Lz, H), (m + "routed_expert_up_proj", H, Lz)]
            if experts:
                for e in range(cfg.num_experts):
                    x = f"{m}experts.{e}."
                    out += [(x + "gate_proj", Im, Lz), (x + "up_proj", Im, Lz), (x + "down_proj", Lz, Im)]
    if cfg.dfly is not None and dfly_mlp:
        d = "dfly_layers."
        for i in range(cfg.dfly.layers):
            out += attn_projections(cfg, f"{d}{i}.self_attn.", cfg.sparse_attn)
            out += [(f"{d}{i}.mlp.gate_proj", dfly_mlp, H), (f"{d}{i}.mlp.up_proj", dfly_mlp, H), (f"{d}{i}.mlp.down_proj", H, dfly_mlp)]
        out += [(d + "mtp_fusion.eh_proj", H, 2 * H)]
        out += [(d + "hidden_correction.gate_proj", None, 2 * H), (d + "hidden_correction.up_proj", None, 2 * H),
                (d + "hidden_correction.down_proj", H, None)]
    return out


def plain_names(cfg: Hy4FlashConfig, dfly: bool = False) -> list[tuple[str, tuple]]:
    """Every tensor read as stored, with its shape."""
    H, S, R, qr, D, E = cfg.hidden_size, cfg.hc_mult, cfg.kv_lora_rank, cfg.q_lora_rank, cfg.index_head_dim, cfg.num_experts
    out = [("embed_tokens.weight", (cfg.vocab_size, H)), ("norm.weight", (H,)), ("hc_head.hc_head_fn", (S, S * H)),
           ("hc_head.hc_head_base", (S,)), ("hc_head.hc_head_scale", (1,))]
    for li in range(cfg.num_hidden_layers):
        p, a = f"layers.{li}.", cfg.attn(li)
        for hc in ("hc_attn_layer", "hc_mlp_layer"):
            out += [(f"{p}{hc}.hc_pre.hc_fn", (2 * S, S * H)), (f"{p}{hc}.hc_pre.hc_base", (2 * S,)), (f"{p}{hc}.hc_pre.hc_scale", (2,))]
        out += [(p + "input_layernorm.weight", (H,)), (p + "post_attention_layernorm.weight", (H,))]
        s = p + "self_attn."
        out += [(s + "q_a_layernorm.weight", (qr,)), (s + "kv_a_layernorm.weight", (R,)), (s + "post_proj_norm.weight", (H,)),
                (s + "learnable_sink_param", (a.heads,))]
        if cfg.is_sparse(li):
            i = s + "compressed_indexer."
            out += [(i + "head_weight_projection.weight", (cfg.index_n_heads, H)), (i + "token_key_projection.weight", (D, H)),
                    (i + "token_key_norm.weight", (D,)), (i + "key_value_projection.weight", (D, D)),
                    (i + "key_gate_projection.weight", (D, D)), (i + "key_norm.weight", (D,)), (i + "intra_group_ape", (cfg.index_kpool, D))]
        m = p + "mlp."
        if cfg.is_dense_mlp(li):
            out.append((m + "post_fc2_norm.weight", (H,)))
        else:
            out += [(m + "gate.weight", (E, H)), (m + "gate.e_score_correction_bias", (E,)), (m + "combine_output_norm.weight", (H,))]
        if li in cfg.engram_layers:
            e, es = p + "engram_layer.", cfg.engram_size
            out += [(e + "lookup_table", (cfg.engram_lookup_table_size,)), (e + "value_proj.weight", (H, es)), (e + "value_proj.bias", (H,)),
                    (e + "unified_pre_gamma", (H,))]
            out += [(f"{e}lookup_per_head_norms.{h}.weight", (cfg.engram_head_dim,)) for h in range(cfg.engram_heads)]
            for k in range(S):
                out += [(f"{e}key_projs.{k}.weight", (H, es)), (f"{e}key_projs.{k}.bias", (H,)), (f"{e}norm_key.{k}.weight", (H,)),
                        (f"{e}norm_query.{k}.weight", (H,)), (f"{e}unified_per_hc_scales.{k}", (H,))]
    if dfly and cfg.dfly is not None:
        d = "dfly_layers."
        out += [(d + "final_norm.weight", (H,)), (d + "mtp_fusion.enorm.weight", (H,)), (d + "mtp_fusion.hnorm.weight", (H,)),
                (d + "mtp_fusion.gate", (H,)), (d + "hidden_correction.hidden_norm.weight", (H,)), (d + "hidden_correction.embed_norm.weight", (H,))]
        for i in range(cfg.dfly.layers):
            s = f"{d}{i}.self_attn."
            out += [(f"{d}{i}.input_layernorm.weight", (H,)), (f"{d}{i}.post_attention_layernorm.weight", (H,)), (s + "q_a_layernorm.weight", (qr,)),
                    (s + "kv_a_layernorm.weight", (R,)), (s + "post_proj_norm.weight", (H,)), (s + "learnable_sink_param", (cfg.sparse_attn.heads,))]
    return out


@dataclass
class Hy4Pack:
    path: str
    cfg: Hy4FlashConfig
    tensors: dict[str, TensorInfo]
    files: dict[str, int]
    quant: dict = field(default_factory=dict)
    formats: dict = field(default_factory=dict)       # projection base -> "exl3" | "bf16" (after audit)
    has_dfly: bool = False
    dfly_mlp: int | None = None                       # the draft MLP's width, from the weights
    _maps: dict = field(default_factory=dict, repr=False)

    @classmethod
    def open(cls, path: str) -> "Hy4Pack":
        path = os.path.expanduser(path)
        with open(os.path.join(path, "config.json")) as f:
            raw = json.load(f)
        gen = None
        if os.path.isfile(os.path.join(path, "generation_config.json")):
            with open(os.path.join(path, "generation_config.json")) as f:
                gen = json.load(f)
        cfg = config_from_dict(raw, path, gen)
        ip = os.path.join(path, "model.safetensors.index.json")
        if os.path.isfile(ip):
            with open(ip) as f:
                wm = json.load(f)["weight_map"]
            names = sorted(set(wm.values()))
        elif os.path.isfile(os.path.join(path, "model.safetensors")):
            wm, names = None, ["model.safetensors"]
        else:
            raise FileNotFoundError(f"{path}: no model.safetensors.index.json and no model.safetensors")
        tensors, files = {}, {}
        for fn in names:
            fp = os.path.join(path, fn)
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"{fp}: the index names it, the pack does not have it")
            with open(fp, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            hdr.pop("__metadata__", None)
            start, end = 8 + n, 0
            files[fn] = os.path.getsize(fp)
            listed = {k for k, v in wm.items() if v == fn} if wm is not None else set(hdr)
            missing = sorted(listed - set(hdr))
            if missing:
                raise ValueError(f"{fn}: the index puts {missing[0]} here, the file's header does not have it")
            for k, t in hdr.items():
                b0, b1 = t["data_offsets"]
                end = max(end, b1)
                if k not in listed:
                    continue
                name = canonical(k)
                if name in tensors:
                    raise ValueError(f"{name} is in {tensors[name].file} and {fn}")
                info = TensorInfo(name, fp, t["dtype"], tuple(t["shape"]), start + b0, b1 - b0)
                if info.nbytes != int(np.prod(info.shape)) * DSIZE[info.dtype]:
                    raise ValueError(f"{fn}: {k} has {info.nbytes} bytes for {info.shape} {info.dtype}")
                tensors[name] = info
            if files[fn] != start + end:
                raise ValueError(f"{fp}: {files[fn]} bytes on disk, the header describes {start + end} (truncated or corrupt)")
        pack = cls(path, cfg, tensors, files, raw.get("quantization_config") or {})
        g = tensors.get("dfly_layers.0.mlp.gate_proj.svh") or tensors.get("dfly_layers.0.mlp.gate_proj.weight")
        if cfg.dfly is not None and g is not None:
            pack.has_dfly = True
            pack.dfly_mlp = cfg.dfly.intermediate_size or int(g.shape[0])
        return pack

    # --------------------------------------------------------------------------- checks
    def k_of(self, base: str) -> float | None:
        t = self.tensors.get(base + ".trellis")
        return None if t is None else t.shape[2] / 16 if t.shape[2] % 16 == 0 else (t.shape[2] - 8) / 16

    def audit(self) -> list[str]:
        """Every tensor the engine reads is there with its shape; returns notes, raises on the first problem."""
        cfg, T = self.cfg, self.tensors
        notes, used = [], set()
        for base, N, K in projections(cfg, dfly_mlp=self.dfly_mlp):
            if base + ".trellis" in T:
                t = T[base + ".trellis"]
                k = self.k_of(base)
                if t.dtype != "I16" or len(t.shape) != 3 or k not in RATES:
                    raise ValueError(f"{base}.trellis {t.shape} {t.dtype}: not a rate the kernels decode")
                kin, n = t.shape[0] * 16, t.shape[1] * 16
                if (K is not None and kin != pad128(K)) or (N is not None and n != pad128(N)):
                    raise ValueError(f"{base}: stored {n} x {kin}, the config gives {N} x {K} (padded to 128)")
                for s in ("suh", "svh"):
                    if base + "." + s not in T and base + "." + s[:2] not in T:
                        raise ValueError(f"{base}.{s} is missing")
                if base + ".mul1" not in T:
                    raise ValueError(f"{base}: no mul1 marker (another codebook)")
                used |= {f"{base}.{s}" for s in ("trellis", "suh", "svh", "su", "sv", "mul1")}
                self.formats[base] = "exl3"
            elif base + ".weight" in T:
                t = T[base + ".weight"]
                if (N is not None and t.shape[0] != N) or (K is not None and t.shape[1] != K):
                    raise ValueError(f"{base}.weight is {t.shape}, the config gives {(N, K)}")
                used.add(base + ".weight")
                self.formats[base] = "bf16"
            else:
                raise ValueError(f"{base}: neither {base}.trellis nor {base}.weight is in the pack")
        for name, shape in plain_names(cfg, dfly=self.has_dfly):
            if name not in T:
                raise ValueError(f"{name} is missing")
            if tuple(T[name].shape) != tuple(shape):
                raise ValueError(f"{name} is {T[name].shape}, the config gives {shape}")
            used.add(name)
        for li in cfg.engram_layers:
            if not os.path.isfile(table_file(self.path, li)):
                raise ValueError(f"{table_file(self.path, li)}: the Engram table of layer {li} is missing")
        extra = sorted(k for k in set(T) - used if not k.startswith(SKIP))
        if not self.has_dfly:
            extra = [k for k in extra if not k.startswith("dfly_layers.")]
        if extra:
            raise ValueError(f"{len(extra)} tensors this engine does not know, first {extra[0]}")
        hist = Counter(f"{self.formats[b]}{'' if self.k_of(b) is None else ' K' + format(self.k_of(b), 'g')}" for b in self.formats)
        notes.append("projections: " + ", ".join(f"{n} {k}" for k, n in sorted(hist.items())))
        if cfg.dfly is not None and not self.has_dfly:
            notes.append("no DFly draft layers in the pack: plain decode only")
        return notes

    def check_mul1(self) -> None:
        for name, t in self.tensors.items():
            if name.endswith(".mul1"):
                v = int(self.read(name).view(-1)[0].item()) & 0xFFFFFFFF
                if v != MUL1:
                    raise ValueError(f"{name}: codebook multiplier {v:#x}, the kernels decode {MUL1:#x}")
                return

    # --------------------------------------------------------------------------- reading
    def read(self, name: str, device=None) -> torch.Tensor:
        """One tensor as stored (a CPU copy, or on `device`). A shard is mapped once: the 62,976 expert
        projections are small reads of the same few files."""
        t = self.tensors[name]
        m = self._maps.get(t.file)
        if m is None:
            m = self._maps[t.file] = np.memmap(t.file, dtype=np.uint8, mode="r")
        arr = np.array(m[t.offset:t.offset + t.nbytes])
        out = torch.from_numpy(arr).view(DT[t.dtype]).reshape(t.shape)
        return out.to(device) if device is not None else out

    def close(self) -> None:
        self._maps.clear()

    def exl3(self, base: str, device=None, out_fp32: bool = False):
        from tools.exl3_linear import EXL3Block, unpack_signs

        def part(s, alt):
            if base + "." + s in self.tensors:
                return self.read(base + "." + s, device)
            return unpack_signs(self.read(base + "." + alt)).to(device)
        return EXL3Block(self.read(base + ".trellis", device), part("suh", "su"), part("svh", "sv"), out_fp32=out_fp32)

    def resident_bytes(self) -> int:
        return sum(t.nbytes for n, t in self.tensors.items() if not n.startswith(SKIP))
