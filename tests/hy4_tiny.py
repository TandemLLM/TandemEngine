"""A tiny random Hy4-flash for the CPU tests: the config of tests/test_hy4_ref.py and weights under the
checkpoint's names, made without `transformers`."""

from __future__ import annotations

import torch

from engine.models.flashnext.ple import DenseTable
from engine.models.flashnext.ref import ExpertBank
from engine.models.hy4_flash.config import SPARSE, Hy4FlashConfig, config_from_dict

TINY = dict(
    vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16, num_hidden_layers=6,
    num_attention_heads=4, swa_num_attention_heads=8, q_lora_rank=16, kv_lora_rank=8,
    qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8, n_routed_experts=8, num_experts_per_tok=3,
    routed_expert_hidden_size=16, routed_scaling_factor=3.998, swiglu_limit=0.25, sliding_window=4,
    index_topk=8, index_head_dim=16, index_n_heads=2, compressed_indexer_compress_ratio=2, hc_mult=4,
    max_position_embeddings=256, rms_norm_eps=1e-5, pad_token_id=2, bos_token_id=1, eos_token_id=3,
    engram_layer_ids=[2, 4], engram_max_ngram_size=3, engram_n_head_per_ngram=2, engram_n_embed_per_ngram=8,
    engram_layer_head_vocab_sizes={"2": [[11, 13], [17, 19]], "4": [[23, 29], [31, 37]]},
    engram_pad_id=2, engram_lookup_table_size=60,
    rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 1, 1],
                                           "mrope_interleaved": True},
                     "deepseek_sparse_attention": {"rope_type": "default", "rope_theta": 1.0e7, "mrope_section": [2, 1, 1],
                                                   "mrope_interleaved": True}},
)


def tiny_config(**changes) -> Hy4FlashConfig:
    return config_from_dict(dict(TINY, **changes), "tiny")


def random_weights(cfg: Hy4FlashConfig, seed: int = 1) -> tuple[dict, dict]:
    """(W, tables): every tensor the reference reads, fp32, nothing left at an initialiser's zero or one."""
    g = torch.Generator().manual_seed(seed)

    def m(*shape):
        return torch.randn(*shape, generator=g) * (0.3 if len(shape) == 1 else 0.2)
    H, S, R, qr = cfg.hidden_size, cfg.hc_mult, cfg.kv_lora_rank, cfg.q_lora_rank
    E, Im, Lz, D, nh = cfg.num_experts, cfg.moe_intermediate_size, cfg.latent_size, cfg.index_head_dim, cfg.index_n_heads
    W: dict = {"embed_tokens.weight": m(cfg.vocab_size, H), "lm_head.weight": m(cfg.vocab_size, H), "norm.weight": m(H),
               "hc_head.hc_head_fn": m(S, S * H), "hc_head.hc_head_base": m(S), "hc_head.hc_head_scale": torch.rand(1, generator=g) + 0.5}
    tables = {}
    for li in range(cfg.num_hidden_layers):
        p, a = f"layers.{li}.", cfg.attn(li)
        for hc in ("hc_attn_layer", "hc_mlp_layer"):
            W[f"{p}{hc}.hc_pre.hc_fn"], W[f"{p}{hc}.hc_pre.hc_base"] = m(2 * S, S * H), m(2 * S)
            W[f"{p}{hc}.hc_pre.hc_scale"] = torch.rand(2, generator=g) + 0.5
        W[p + "input_layernorm.weight"], W[p + "post_attention_layernorm.weight"] = m(H), m(H)
        s = p + "self_attn."
        W[s + "linear_gate.weight"], W[s + "q_a_proj.weight"], W[s + "q_a_layernorm.weight"] = m(a.heads * a.v, H), m(qr, H), m(qr)
        W[s + "q_b_proj.weight"], W[s + "kv_a_proj_with_mqa.weight"] = m(a.heads * a.qk, qr), m(R + a.rope, H)
        W[s + "kv_a_layernorm.weight"], W[s + "kv_b_proj.weight"] = m(R), m(a.heads * (a.nope + a.v), R)
        W[s + "o_proj.weight"], W[s + "post_proj_norm.weight"], W[s + "learnable_sink_param"] = m(H, a.heads * a.v), m(H), m(a.heads)
        if cfg.layer_types[li] == SPARSE:
            i = s + "compressed_indexer."
            W[i + "query_projection.weight"], W[i + "head_weight_projection.weight"] = m(nh * D, qr), m(nh, H)
            W[i + "token_key_projection.weight"], W[i + "token_key_norm.weight"] = m(D, H), m(D)
            W[i + "key_value_projection.weight"], W[i + "key_gate_projection.weight"] = m(D, D), m(D, D)
            W[i + "key_norm.weight"], W[i + "intra_group_ape"] = m(D), m(cfg.index_kpool, D)
        f = p + "mlp."
        if cfg.is_dense_mlp(li):
            I = cfg.intermediate_size
            W[f + "gate_proj.weight"], W[f + "up_proj.weight"], W[f + "down_proj.weight"] = m(I, H), m(I, H), m(H, I)
            W[f + "post_fc2_norm.weight"] = m(H)
        else:
            Is = cfg.shared_intermediate_size
            W[f + "gate.weight"], W[f + "gate.e_score_correction_bias"] = m(E, H), m(E)
            W[f + "experts.gate_proj"], W[f + "experts.up_proj"] = ExpertBank(dense=m(E, Im, Lz)), ExpertBank(dense=m(E, Im, Lz))
            W[f + "experts.down_proj"] = ExpertBank(dense=m(E, Lz, Im))
            W[f + "routed_expert_down_proj.weight"], W[f + "routed_expert_up_proj.weight"] = m(Lz, H), m(H, Lz)
            W[f + "shared_experts.gate_proj.weight"], W[f + "shared_experts.up_proj.weight"] = m(Is, H), m(Is, H)
            W[f + "shared_experts.down_proj.weight"], W[f + "combine_output_norm.weight"] = m(H, Is), m(H)
        if li in cfg.engram_layers:
            lid, e = cfg.engram_layers[li], p + "engram_layer."
            W[e + "lookup_table"] = torch.randint(0, 40, (cfg.engram_lookup_table_size,), generator=g, dtype=torch.int32)
            for hd in range(cfg.engram_heads):
                W[f"{e}lookup_per_head_norms.{hd}.weight"] = m(cfg.engram_head_dim)
            for k in range(S):
                W[f"{e}key_projs.{k}.weight"], W[f"{e}key_projs.{k}.bias"] = m(H, cfg.engram_size), m(H)
                W[f"{e}norm_key.{k}.weight"], W[f"{e}norm_query.{k}.weight"] = m(H), m(H)
                W[f"{e}unified_per_hc_scales.{k}"] = m(H)
            W[e + "value_proj.weight"], W[e + "value_proj.bias"], W[e + "unified_pre_gamma"] = m(H, cfg.engram_size), m(H), m(H)
            tables[li] = DenseTable(m(cfg.engram_rows(lid), cfg.engram_head_dim))
    return W, tables
