"""The Hy4-flash configuration (`hy_v4_flash`), read from the checkpoint.

Hy4-flash is 42 layers of gated multi-head latent attention over four residual streams. Twelve layers
read a selection the compressed indexer makes (groups of four tokens, the best 511 closed groups and the
query's own), thirty read the last 512 tokens. Layer 0 has a dense MLP; the other 41 a mixture of 512
experts that run in a 1,536-wide latent space between two shared projections. Four layers add a hashed
n-gram memory (Engram) to the streams before their attention. The checkpoint also carries the DFly2
draft layers, described by the `dfly_*` fields at the top level.

Every field a module reads is resolved here. A field this reader does not know stops the load with its
name: a release that adds an arithmetic switch must not be served as the old one. Fields that matter
only for training or for other runtimes are in `IGNORED`; fields whose other values would change the
arithmetic are checked against the one value the engine implements (`FIXED`).

Two things the tests pin: `engram_layer_ids` are one-indexed (id 2 sits in front of layer index 1),
and the checkpoint's `deepseek_sparse_attention` is the reference's `indexed_attention` (here `sparse`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from engine.config import resolve_snapshot

FAMILY_TYPES = ("hy_v4_flash",)
SPARSE, WINDOW = "sparse", "window"
_LAYER_NAMES = {"deepseek_sparse_attention": SPARSE, "indexed_attention": SPARSE, "sliding_attention": WINDOW}

# read by config_from_dict (directly, through a default, or as a cross-check of a derived value)
KNOWN = {
    "model_type", "hidden_size", "num_hidden_layers", "vocab_size", "rms_norm_eps", "max_position_embeddings",
    "layer_types", "swa_pattern", "mlp_layer_types", "first_k_dense_replace", "intermediate_size",
    "moe_intermediate_size", "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
    "routed_scaling_factor", "norm_topk_prob", "routed_expert_hidden_size", "swiglu_limit",
    "q_lora_rank", "kv_lora_rank", "num_attention_heads", "num_key_value_heads", "head_dim",
    "qk_nope_head_dim", "qk_rope_head_dim", "qk_head_dim", "v_head_dim",
    "swa_num_attention_heads", "swa_qk_nope_head_dim", "swa_qk_rope_head_dim", "swa_qk_head_dim", "swa_v_head_dim",
    "sliding_window", "window_size", "index_topk", "index_head_dim", "index_n_heads", "index_kpool",
    "compressed_indexer_compress_ratio", "compressed_indexer_group_topk", "compressed_indexer_candidate_topk",
    "compressed_indexer_head_dim", "compressed_indexer_num_attention_heads",
    "hc_mult", "hc_eps", "hc_magnitude", "rope_parameters",
    "engram_layer_ids", "engram_max_ngram_size", "engram_n_head_per_ngram", "engram_n_embed_per_ngram",
    "engram_layer_head_vocab_sizes", "engram_pad_id", "engram_fp8_embedding", "engram_seed",
    "engram_lookup_table_size", "eos_token_id", "pad_token_id", "bos_token_id", "indexer_types",
}
# training, other runtimes' kernel switches, initialisers, or ids the tokenizer files own
IGNORED = {
    "architectures", "attention_dropout", "bitwise_backward_align", "dtype", "torch_dtype",
    "compressed_indexer_stage2_query_block_size", "compressed_indexer_teacher_pooling_type",
    "compressed_indexer_triton_pooling_min_groups", "compressed_indexer_use_triton_pooling",
    "fast_op_fp32_linear_gemm", "ihc_kernel_backend", "initializer_range", "moe_aux_loss_coeff",
    "output_router_logits", "return_bos_score", "return_max_logit", "use_cache", "use_grouped_mm",
    "transformers_version", "learnable_sink_init", "engram_embed_random_seed", "engram_vocab_size",
}
# fields whose value would change the arithmetic: the one value this engine implements
FIXED = {
    "attention_bias": False, "hidden_act": "silu", "n_group": 1, "topk_group": 1,
    "moe_router_use_sigmoid": True, "moe_router_enable_expert_bias": True,
    "gated_mla": True, "gating_type": "elementwise", "learnable_sink": True, "sandwich_norm": True,
    "use_mla": True, "use_dsa": True, "use_compressed_indexer": True, "qk_norm": True, "full_nope": False,
    "compressed_indexer_key_pooling": "gated", "compressed_indexer_skip_fine_select": True,
    "compressed_indexer_reuse_fine_indexer": False, "enable_ihc": True, "enable_lm_head_fp32": True,
    "gate_lora_op": "none", "o_lora_op": "none", "gate_groups": 1, "o_groups": 1, "gate_lora_rank": None,
    "o_lora_rank": None, "dsa_skip_attn_post_norm": False, "tie_word_embeddings": False,
    "num_nextn_predict_layers": 0, "engram_isolate_fffd_namespace": False, "engram_vocab_map_path": None,
    "engram_tokenizer_name_or_path": None,
}
# the top level of a vision-language config: what this reader takes, and what the tower's reader owns
TOP_KNOWN = {
    "text_config", "model_type", "architectures", "image_token_id", "video_token_id", "eos_token_id",
    "speculative_config", "dfly_mtp_fusion", "dfly_target_index_topk", "dfly_context_local_window",
    "dfly_enable_hidden_correction", "dfly_num_hidden_layers", "dfly_mask_token_id",
    "dfly_target_index_layer", "dfly_target_kv_layers", "dfly_intermediate_size",
}
TOP_IGNORED = {
    "dtype", "torch_dtype", "transformers_version", "processor_class", "vision_config", "spatial_merge_size",
    "image_start_token_id", "image_end_token_id", "image_newline_token_id", "vision_start_token_id",
    "video_start_token_id", "video_end_token_id", "tie_word_embeddings", "quantization_config",
}
TOP_FIXED = {"dfly_mtp_fusion": "slot0", "dfly_context_local_window": None, "dfly_enable_hidden_correction": True}


def is_hy4_flash(raw: dict) -> bool:
    """True for a Hy4-flash config (top level or its `text_config`)."""
    t = raw.get("text_config") or raw
    mt = str(t.get("model_type", raw.get("model_type", "")))
    return mt.startswith(FAMILY_TYPES)


@dataclass(frozen=True)
class AttnShape:
    """One attention layer's head layout (the window layers have more heads than the sparse ones)."""
    heads: int
    nope: int           # key and query channels without position
    rope: int           # key and query channels the rotary turns (the last ones)
    v: int              # value channels a head

    @property
    def qk(self) -> int:
        return self.nope + self.rope


@dataclass
class DFlyConfig:
    """The DFly2 draft layers the checkpoint carries (`model.dfly_layers.*`)."""
    layers: int                         # draft layers
    mask_token_id: int
    index_layers: list[int]             # target layer whose indexer picks draft layer i reads
    kv_layers: list[int]                # target layer whose latents draft layer i reads
    index_topk: int                     # context tokens a draft layer reads beside its block
    draft_tokens: int                   # tokens a round drafts (Tencent's serving configs: 5)
    intermediate_size: int | None = None    # the draft MLP's width; None: from the weights


@dataclass
class Hy4FlashConfig:
    path: str
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    max_position_embeddings: int
    layer_types: list[str]              # SPARSE | WINDOW
    mlp_layer_types: list[str]          # "dense" | "sparse"
    intermediate_size: int              # the dense MLP's width
    moe_intermediate_size: int          # an expert's inner width
    shared_intermediate_size: int       # the shared expert's inner width
    num_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    latent_size: int                    # the width the routed experts read and write
    swiglu_limit: float
    q_lora_rank: int
    kv_lora_rank: int
    sparse_attn: AttnShape
    window_attn: AttnShape
    sliding_window: int                 # keys a window layer reads, the query's own included
    index_topk: int                     # tokens a sparse layer reads at most; at or below it, all of them
    index_head_dim: int
    index_n_heads: int
    index_kpool: int                    # tokens a group
    hc_mult: int
    hc_eps: float
    hc_magnitude: float
    rope_theta: dict                    # SPARSE / WINDOW -> base
    mrope_section: dict                 # SPARSE / WINDOW -> [t, h, w] frequency counts
    engram_layer_ids: list[int]         # ONE-indexed, as in the checkpoint
    engram_max_ngram_size: int
    engram_n_head_per_ngram: int
    engram_n_embed_per_ngram: int
    engram_head_vocab_sizes: dict       # layer id (one-indexed) -> the heads' table sizes, n-gram major
    engram_pad_id: int
    engram_fp8_embedding: bool
    engram_seed: int
    engram_lookup_table_size: int
    image_token_id: int | None
    video_token_id: int | None
    eos_token_id: int
    eos_token_ids: list[int] = field(default_factory=list)   # generation stop ids
    dfly: DFlyConfig | None = None

    # --- derived ---
    def attn(self, layer: int) -> AttnShape:
        return self.sparse_attn if self.layer_types[layer] == SPARSE else self.window_attn

    def is_sparse(self, layer: int) -> bool:
        return self.layer_types[layer] == SPARSE

    def is_dense_mlp(self, layer: int) -> bool:
        return self.mlp_layer_types[layer] == "dense"

    @property
    def sparse_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == SPARSE]

    @property
    def window_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == WINDOW]

    @property
    def group_topk(self) -> int:
        """Groups a sparse layer reads: the best `group_topk - 1` closed ones and the query's own."""
        return self.index_topk // self.index_kpool

    @property
    def hc_hidden(self) -> int:
        return self.hc_mult * self.hidden_size

    @property
    def engram_layers(self) -> dict[int, int]:
        """layer index (0-based) -> Engram layer id (one-indexed, the key of its table sizes and its hash seed)."""
        return {lid - 1: lid for lid in self.engram_layer_ids}

    @property
    def engram_heads(self) -> int:
        return (self.engram_max_ngram_size - 1) * self.engram_n_head_per_ngram

    @property
    def engram_head_dim(self) -> int:
        return self.engram_n_embed_per_ngram // self.engram_n_head_per_ngram

    @property
    def engram_size(self) -> int:
        return (self.engram_max_ngram_size - 1) * self.engram_n_embed_per_ngram

    def engram_offsets(self, lid: int) -> list[int]:
        """The row at which each head's slice of layer `lid`'s table starts."""
        out, at = [], 0
        for s in self.engram_head_vocab_sizes[lid]:
            out.append(at)
            at += s
        return out

    def engram_rows(self, lid: int) -> int:
        return sum(self.engram_head_vocab_sizes[lid])

    @property
    def latent_bytes_per_token(self) -> int:
        """BF16 bytes a cached token holds in one layer: the latent and the turned key channels."""
        return 2 * (self.kv_lora_rank + self.sparse_attn.rope)


def _check(name: str, got, want) -> None:
    if got != want:
        raise ValueError(f"Hy4-flash config {name}={got!r}: this engine expects {want!r}")


def config_from_dict(raw: dict, path: str = "", gen: dict | None = None) -> Hy4FlashConfig:
    t = raw.get("text_config") or raw
    top = raw if "text_config" in raw else {}
    n = t["num_hidden_layers"]
    hidden = t["hidden_size"]

    lt = t.get("layer_types")
    if lt is None:
        lt = ["indexed_attention" if (i < 2 or (i - 1) % 4 == 0) else "sliding_attention" for i in range(n)]
    bad = sorted({x for x in lt if x not in _LAYER_NAMES})
    if bad:
        raise ValueError(f"Hy4-flash layer type(s) this engine does not know: {', '.join(bad)}")
    lt = [_LAYER_NAMES[x] for x in lt]
    mt = t.get("mlp_layer_types")
    if mt is None:
        k = t.get("first_k_dense_replace", 1)
        mt = ["dense"] * min(k, n) + ["sparse"] * max(n - k, 0)

    rope = t.get("rope_parameters") or {}
    rope = {_LAYER_NAMES.get(k, k): v for k, v in rope.items()}
    default_theta = {SPARSE: 1e7, WINDOW: 1e4}
    theta = {k: float((rope.get(k) or {}).get("rope_theta", default_theta[k])) for k in (SPARSE, WINDOW)}
    section = {k: list((rope.get(k) or {}).get("mrope_section", [12, 10, 10])) for k in (SPARSE, WINDOW)}
    for k in (SPARSE, WINDOW):
        r = rope.get(k) or {}
        if r.get("rope_type", "default") != "default" or not r.get("mrope_interleaved", True):
            raise ValueError(f"Hy4-flash rope_parameters[{k}]: this engine implements the default interleaved rotary only")

    nope, rdim, vdim = t.get("qk_nope_head_dim", 192), t.get("qk_rope_head_dim", 64), t.get("v_head_dim", 256)
    sparse_attn = AttnShape(t.get("num_attention_heads", 32), nope, rdim, vdim)
    window_attn = AttnShape(t.get("swa_num_attention_heads", 64), t.get("swa_qk_nope_head_dim", nope),
                            t.get("swa_qk_rope_head_dim", rdim), t.get("swa_v_head_dim", vdim))

    kpool = t.get("compressed_indexer_compress_ratio", t.get("index_kpool", 4))
    eos = t.get("eos_token_id", top.get("eos_token_id"))
    eos = eos[0] if isinstance(eos, list) else eos
    if eos is None:
        raise ValueError("Hy4-flash needs eos_token_id in its text config")
    gen_eos: list[int] = []
    if gen:
        e = gen.get("eos_token_id")
        gen_eos = e if isinstance(e, list) else ([e] if e is not None else [])

    ids = sorted(set(t.get("engram_layer_ids") or []))
    sizes = {int(k): [s for order in v for s in order] for k, v in (t.get("engram_layer_head_vocab_sizes") or {}).items()}

    dfly = None
    # the draft layers need their mask token and the target layers they read; a config that names only their count
    # (a converter's tiny rehearsal model) has none to serve
    if top.get("dfly_num_hidden_layers") and all(k in top for k in ("dfly_mask_token_id", "dfly_target_index_layer", "dfly_target_kv_layers")):
        nd = int(top["dfly_num_hidden_layers"])
        spec = top.get("speculative_config") or {}

        def per_layer(v):
            return [int(v)] * nd if isinstance(v, int) else [int(x) for x in v]
        dfly = DFlyConfig(
            layers=nd, mask_token_id=int(top["dfly_mask_token_id"]),
            index_layers=per_layer(top["dfly_target_index_layer"]),
            kv_layers=per_layer(top["dfly_target_kv_layers"]),
            index_topk=int(top.get("dfly_target_index_topk", t.get("index_topk", 2048))),
            draft_tokens=int(spec.get("num_speculative_tokens", 5)),
            intermediate_size=top.get("dfly_intermediate_size"),
        )

    cfg = Hy4FlashConfig(
        path=path, hidden_size=hidden, num_hidden_layers=n, vocab_size=t["vocab_size"],
        rms_norm_eps=t.get("rms_norm_eps", 1e-5),
        max_position_embeddings=t.get("max_position_embeddings", 1048576),
        layer_types=lt, mlp_layer_types=list(mt),
        intermediate_size=t.get("intermediate_size", 13824),
        moe_intermediate_size=t.get("moe_intermediate_size", 1536),
        shared_intermediate_size=t.get("moe_intermediate_size", 1536) * t.get("n_shared_experts", 1),
        num_experts=t.get("n_routed_experts", 512),
        num_experts_per_tok=t.get("num_experts_per_tok", 16),
        routed_scaling_factor=float(t.get("routed_scaling_factor", 3.998)),
        norm_topk_prob=bool(t.get("norm_topk_prob", True)),
        latent_size=t.get("routed_expert_hidden_size", 1536),
        swiglu_limit=float(t.get("swiglu_limit", 10.0)),
        q_lora_rank=t.get("q_lora_rank", 1024), kv_lora_rank=t.get("kv_lora_rank", 512),
        sparse_attn=sparse_attn, window_attn=window_attn,
        sliding_window=t.get("sliding_window", 512),
        index_topk=t.get("index_topk", 2048), index_head_dim=t.get("index_head_dim", 128),
        index_n_heads=t.get("index_n_heads", 16), index_kpool=kpool,
        hc_mult=t.get("hc_mult", 4), hc_eps=float(t.get("hc_eps", 1e-6)), hc_magnitude=float(t.get("hc_magnitude", 2.0)),
        rope_theta=theta, mrope_section=section,
        engram_layer_ids=ids,
        engram_max_ngram_size=t.get("engram_max_ngram_size", 3),
        engram_n_head_per_ngram=t.get("engram_n_head_per_ngram", 8),
        engram_n_embed_per_ngram=t.get("engram_n_embed_per_ngram", 768),
        engram_head_vocab_sizes=sizes,
        engram_pad_id=t.get("engram_pad_id", 120002),
        engram_fp8_embedding=bool(t.get("engram_fp8_embedding", True)),
        engram_seed=t.get("engram_seed", 0),
        engram_lookup_table_size=t.get("engram_lookup_table_size", 120818),
        image_token_id=top.get("image_token_id"), video_token_id=top.get("video_token_id"),
        eos_token_id=int(eos), eos_token_ids=gen_eos or [int(eos)], dfly=dfly,
    )
    validate(cfg, t, top)
    return cfg


def validate(cfg: Hy4FlashConfig, t: dict, top: dict) -> None:
    """The reference's invariants, plus the ones this engine relies on."""
    unknown = sorted(set(t) - KNOWN - IGNORED - set(FIXED))
    if unknown:
        raise ValueError(f"Hy4-flash config field(s) this engine does not know: {', '.join(unknown)}")
    unknown = sorted(set(top) - TOP_KNOWN - TOP_IGNORED)
    if unknown:
        raise ValueError(f"Hy4-flash top-level config field(s) this engine does not know: {', '.join(unknown)}")
    for src, fixed in ((t, FIXED), (top, TOP_FIXED)):
        for k, want in fixed.items():
            if k in src and src[k] != want:
                raise ValueError(f"Hy4-flash config {k}={src[k]!r}: this engine implements {want!r} only")
    n = cfg.num_hidden_layers
    if len(cfg.layer_types) != n or len(cfg.mlp_layer_types) != n:
        raise ValueError("layer_types and mlp_layer_types need one entry a layer")
    if "swa_pattern" in t:
        _check("swa_pattern", list(t["swa_pattern"]), [int(x == WINDOW) for x in cfg.layer_types])
    if "first_k_dense_replace" in t:
        _check("first_k_dense_replace", t["first_k_dense_replace"], sum(x == "dense" for x in cfg.mlp_layer_types))
    if any(x not in ("dense", "sparse") for x in cfg.mlp_layer_types):
        raise ValueError("mlp_layer_types must be dense or sparse")
    if "indexer_types" in t and any(x != "full" for x in t["indexer_types"]):
        raise ValueError("indexer_types: this engine implements 'full' only")
    for a, name in ((cfg.sparse_attn, ""), (cfg.window_attn, "swa_")):
        if f"{name}qk_head_dim" in t:
            _check(f"{name}qk_head_dim", t[f"{name}qk_head_dim"], a.qk)
        if a.rope % 2 or sum(cfg.mrope_section[SPARSE if not name else WINDOW]) != a.rope // 2:
            raise ValueError("mrope_section must have three entries summing to half the rotary width")
    if "num_key_value_heads" in t:
        _check("num_key_value_heads", t["num_key_value_heads"], cfg.sparse_attn.heads)
    if "head_dim" in t and t["head_dim"] not in (cfg.sparse_attn.rope, cfg.sparse_attn.qk):
        raise ValueError(f"Hy4-flash config head_dim={t['head_dim']!r}: expected the rotary or the key width")
    if cfg.window_attn.rope != cfg.sparse_attn.rope or cfg.window_attn.nope != cfg.sparse_attn.nope:
        raise ValueError("this engine keeps one latent layout: the window layers' key split must equal the sparse layers'")
    if "window_size" in t:
        _check("window_size", list(t["window_size"]), [cfg.sliding_window - 1, 0])
    if cfg.index_kpool < 1 or cfg.index_topk % cfg.index_kpool:
        raise ValueError("index_topk must be divisible by the indexer's group size")
    for k, want in (("compressed_indexer_group_topk", cfg.group_topk), ("compressed_indexer_candidate_topk", cfg.index_topk),
                    ("compressed_indexer_head_dim", cfg.index_head_dim),
                    ("compressed_indexer_num_attention_heads", cfg.index_n_heads)):
        if k in t:
            _check(k, t[k], want)
    if cfg.sparse_attn.rope > cfg.index_head_dim:
        raise ValueError("rotary dims must fit the indexer head")
    if cfg.hc_mult < 1:
        raise ValueError("Hy4-flash needs hc_mult >= 1")
    if not 0 < cfg.num_experts_per_tok <= cfg.num_experts:
        raise ValueError("num_experts_per_tok out of range")
    orders = cfg.engram_max_ngram_size - 1
    for lid in cfg.engram_layer_ids:
        if not 1 <= lid <= n:
            raise ValueError(f"Engram layer id {lid} must be a 1-indexed layer")
        sizes = cfg.engram_head_vocab_sizes.get(lid)
        if not sizes or len(sizes) != orders * cfg.engram_n_head_per_ngram:
            raise ValueError(f"engram_layer_head_vocab_sizes['{lid}'] must hold {orders} lists of "
                             f"{cfg.engram_n_head_per_ngram} sizes")
    if cfg.engram_layer_ids:
        if cfg.engram_n_embed_per_ngram % cfg.engram_n_head_per_ngram:
            raise ValueError("engram_n_embed_per_ngram must be divisible by engram_n_head_per_ngram")
        if not cfg.engram_pad_id < cfg.engram_lookup_table_size:
            raise ValueError("engram_lookup_table_size must cover engram_pad_id")
    d = cfg.dfly
    if d is not None:
        if len(d.index_layers) != d.layers or len(d.kv_layers) != d.layers:
            raise ValueError("DFly needs one target layer a draft layer")
        for li in d.index_layers + d.kv_layers:
            if not 1 <= li < n or cfg.layer_types[li] != SPARSE:
                raise ValueError(f"DFly target layer {li} must be a sparse attention layer")


def load_hy4_flash_config(path: str | None = None) -> Hy4FlashConfig:
    snap = resolve_snapshot(path)
    with open(os.path.join(snap, "config.json")) as f:
        raw = json.load(f)
    if not is_hy4_flash(raw):
        raise ValueError(f"{snap}: not a Hy4-flash config")
    gen = None
    gp = os.path.join(snap, "generation_config.json")
    if os.path.isfile(gp):
        with open(gp) as f:
            gen = json.load(f)
    return config_from_dict(raw, snap, gen)
