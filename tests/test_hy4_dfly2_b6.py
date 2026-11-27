"""CPU parity & loader tests for standalone HY4DFlyDraftModel (DFly2-B6: MoE draft layers + AttentiveCorrection)."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.models.hy4_flash.config import SPARSE, config_from_dict, dfly_config_from_dict  # noqa: E402
from engine.models.hy4_flash.dfly import DFlyHead  # noqa: E402
from engine.models.hy4_flash.engine import Hy4FlashEngine  # noqa: E402
from engine.models.hy4_flash.load import load_dfly_external  # noqa: E402
from engine.models.hy4_flash.serving import Hy4FlashServing  # noqa: E402
from tests.hy4_tiny import TINY, random_weights  # noqa: E402
from tests.test_hy4_dfly import TOP, dfly_weights  # noqa: E402


def _angelslim_real_cfg_dict() -> dict:
    return {
        "architectures": ["HY4DFlyDraftModel"],
        "model_type": "hy4_angelspec_draft",
        "block_size": 6,
        "sample_from_anchor": False,
        "hidden_size": 3072,
        "num_hidden_layers": 3,
        "num_attention_heads": 32,
        "rms_norm_eps": 1e-5,
        "q_lora_rank": 1024,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "gated_mla": True,
        "attn_post_proj_norm": True,
        "draft_mlp_type": "moe",
        "num_experts": 128,
        "num_experts_per_tok": 3,
        "moe_intermediate_size": 1536,
        "routed_expert_hidden_size": 1536,
        "num_shared_experts": 1,
        "route_norm": True,
        "router_scaling_factor": 3.998,
        "swiglu_limit": 10.0,
        "sandwich_norm": True,
        "target_kv_layers": [33, 37, 41],
        "target_index_layer": [33, 37, 41],
        "target_index_topk": 2048,
        "mask_token_id": 120009,
        "mtp_fusion": "slot0",
        "enable_hidden_correction": True,
        "hidden_correction_intermediate_size": 3072,
        "attentive_correction": True,
        "attentive_correction_source_layer": -1,
        "attentive_correction_block_keys": True,
        "attentive_correction_block_mode": "gated",
        "attentive_correction_block_kv_in": "hidden_embed",
    }


def test_dfly2_b6_config_parsing_matches_angelslim():
    from engine.models.hy4_flash.config import AttnShape, Hy4FlashConfig
    target = Hy4FlashConfig(
        path=".", hidden_size=3072, num_hidden_layers=42, vocab_size=131072, rms_norm_eps=1e-5,
        max_position_embeddings=262144,
        layer_types=["sparse" if (i < 2 or (i - 1) % 4 == 0) else "window" for i in range(42)],
        mlp_layer_types=["dense"] + ["sparse"] * 41,
        intermediate_size=13824, moe_intermediate_size=1536, shared_intermediate_size=1536,
        num_experts=512, num_experts_per_tok=16, routed_scaling_factor=3.998, norm_topk_prob=True,
        latent_size=1536, swiglu_limit=10.0, q_lora_rank=1024, kv_lora_rank=512,
        sparse_attn=AttnShape(32, 192, 64, 256), window_attn=AttnShape(64, 192, 64, 256),
        sliding_window=512, index_topk=2048, index_head_dim=128, index_n_heads=16, index_kpool=4,
        hc_mult=4, hc_eps=1e-6, hc_magnitude=2.0,
        rope_theta={"sparse": 1e7, "window": 1e4}, mrope_section={"sparse": [12, 10, 10], "window": [12, 10, 10]},
        engram_layer_ids=[2], engram_max_ngram_size=3, engram_n_head_per_ngram=8, engram_n_embed_per_ngram=768,
        engram_head_vocab_sizes={2: [64] * 16}, engram_pad_id=120002, engram_fp8_embedding=False,
        engram_seed=0, engram_lookup_table_size=120818,
        image_token_id=None, video_token_id=None, eos_token_id=1, eos_token_ids=[1],
    )
    d = dfly_config_from_dict(_angelslim_real_cfg_dict(), target, path="/tmp/dfly2")
    assert d.layers == 3 and d.draft_tokens == 5
    assert d.index_layers == [33, 37, 41] and d.kv_layers == [33, 37, 41]
    assert d.mlp_type == "moe" and d.num_experts == 128 and d.num_experts_per_tok == 3
    assert d.attentive_correction and d.attentive_source_layer == -1
    assert d.prefix == "layers." and d.norm_name == "norm.weight"
    # window layer index rejects cleanly
    bad = _angelslim_real_cfg_dict()
    bad["target_kv_layers"] = [2, 37, 41]
    try:
        dfly_config_from_dict(bad, target)
    except ValueError:
        return "3L MoE + AttentiveCorrection config parsed & validated"
    raise AssertionError("window layer should fail target validation")


def test_split_rope_matches_interleaved_dot_product():
    from engine.models.hy4_flash.ref import rotate
    torch.manual_seed(7)
    T, H_q, R = 6, 4, 16
    q = torch.randn(T, H_q, R)
    k = torch.randn(T, R)
    freqs = torch.randn(T, R // 2)
    cos, sin = freqs.cos(), freqs.sin()

    q_int = rotate(q, cos[:, None], sin[:, None])
    k_int = rotate(k, cos, sin)
    dots_int = torch.einsum("qhr,kr->hqk", q_int, k_int)

    def to_split(x):
        return torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)

    def rotate_split(x, c, s):
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)

    q_sp = rotate_split(to_split(q), cos[:, None], sin[:, None])
    k_sp = rotate_split(to_split(k), cos, sin)
    dots_sp = torch.einsum("qhr,kr->hqk", q_sp, k_sp)
    assert torch.max(torch.abs(dots_int - dots_sp)).item() < 1e-5
    return "interleaved RoPE == AngelSpec split RoPE (<1e-5)"


def _make_dfly2_checkpoint(root_dir: str, cfg, seed: int = 123) -> tuple[dict, dict]:
    g = torch.Generator().manual_seed(seed)

    def w(shape, dt=torch.bfloat16, scale=0.04):
        return (torch.randn(shape, generator=g, dtype=torch.float32) * scale).to(dt)

    def norm(n):
        return (torch.ones(n, dtype=torch.bfloat16) + 0.05 * torch.randn(n, generator=g, dtype=torch.bfloat16))

    H, R, qr = cfg.hidden_size, cfg.kv_lora_rank, cfg.q_lora_rank
    a = cfg.sparse_attn
    Im, Lz, Is, E = 32, 24, 28, 8
    raw_cfg = {
        "architectures": ["HY4DFlyDraftModel"],
        "model_type": "hy4_angelspec_draft",
        "block_size": 4,
        "sample_from_anchor": False,
        "hidden_size": H,
        "num_hidden_layers": 2,
        "num_attention_heads": a.heads,
        "rms_norm_eps": cfg.rms_norm_eps,
        "q_lora_rank": qr,
        "kv_lora_rank": R,
        "qk_nope_head_dim": a.nope,
        "qk_rope_head_dim": a.rope,
        "v_head_dim": a.v,
        "gated_mla": True,
        "attn_post_proj_norm": True,
        "draft_mlp_type": "moe",
        "num_experts": E,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": Im,
        "routed_expert_hidden_size": Lz,
        "num_shared_experts": 1,
        "route_norm": True,
        "router_scaling_factor": 2.5,
        "swiglu_limit": 5.0,
        "sandwich_norm": True,
        "target_kv_layers": [1, 5],
        "target_index_layer": [1, 5],
        "target_index_topk": cfg.index_topk,
        "mask_token_id": cfg.dfly.mask_token_id,
        "mtp_fusion": "slot0",
        "enable_hidden_correction": True,
        "hidden_correction_intermediate_size": H,
        "attentive_correction": True,
        "attentive_correction_source_layer": -1,
        "attentive_correction_block_keys": True,
        "attentive_correction_block_mode": "gated",
        "attentive_correction_block_kv_in": "hidden_embed",
    }
    raw_tensors = {
        "model.norm.weight": norm(H),
        "model.mtp_fusion.enorm.weight": norm(H),
        "model.mtp_fusion.hnorm.weight": norm(H),
        "model.mtp_fusion.gate": torch.randn(H, generator=g, dtype=torch.bfloat16) * 0.2,
        "model.mtp_fusion.eh_proj.weight": w((H, 2 * H)),
        "model.hidden_correction.hidden_norm.weight": norm(H),
        "model.hidden_correction.embed_norm.weight": norm(H),
        "model.hidden_correction.gate_proj.weight": w((H, 2 * H)),
        "model.hidden_correction.up_proj.weight": w((H, 2 * H)),
        "model.hidden_correction.down_proj.weight": w((H, H)),
        "model.hidden_correction.attention.q_in.weight": w((qr, 2 * H)),
        "model.hidden_correction.attention.q_a_layernorm.weight": norm(qr),
        "model.hidden_correction.attention.q_b_proj.weight": w((a.heads * a.qk, qr)),
        "model.hidden_correction.attention.kv_in.weight": w((R + a.rope, 2 * H)),
        "model.hidden_correction.attention.kv_a_layernorm.weight": norm(R),
        "model.hidden_correction.attention.kv_b_proj.weight": w((a.heads * (a.nope + a.v), R)),
        "model.hidden_correction.attention.o_proj.weight": w((H, a.heads * a.v)),
        "model.hidden_correction.attention.learnable_sink_param": torch.randn(a.heads, generator=g, dtype=torch.float32) * 0.2,
        "model.hidden_correction.attention.block_sink": torch.randn(a.heads, generator=g, dtype=torch.float32) * 0.2,
        "model.hidden_correction.attention.block_rel_bias": torch.randn(a.heads, 8, generator=g, dtype=torch.float32) * 0.1,
        "model.hidden_correction.attention.block_gate.weight": w((a.heads, 2 * H)),
        "model.hidden_correction.attention.block_gate_bias": torch.randn(a.heads, 8, generator=g, dtype=torch.float32) * 0.1,
    }
    for i in range(2):
        p = f"model.layers.{i}."
        raw_tensors.update({
            p + "input_layernorm.weight": norm(H),
            p + "post_attention_layernorm.weight": norm(H),
            p + "self_attn.linear_gate.weight": w((a.heads * a.v, H)),
            p + "self_attn.q_a_proj.weight": w((qr, H)),
            p + "self_attn.q_a_layernorm.weight": norm(qr),
            p + "self_attn.q_b_proj.weight": w((a.heads * a.qk, qr)),
            p + "self_attn.kv_a_proj_with_mqa.weight": w((R + a.rope, H)),
            p + "self_attn.kv_a_layernorm.weight": norm(R),
            p + "self_attn.kv_b_proj.weight": w((a.heads * (a.nope + a.v), R)),
            p + "self_attn.o_proj.weight": w((H, a.heads * a.v)),
            p + "self_attn.post_proj_norm.weight": norm(H),
            p + "self_attn.learnable_sink_param": torch.randn(a.heads, generator=g, dtype=torch.float32) * 0.2,
            p + "mlp.gate.weight": w((E, H), dt=torch.float32),
            p + "mlp.expert_bias": torch.randn(E, generator=g, dtype=torch.float32) * 0.05,
            p + "mlp.routed_expert_down_proj.weight": w((Lz, H)),
            p + "mlp.routed_expert_up_proj.weight": w((H, Lz)),
            p + "mlp.combine_output_norm.weight": norm(H),
            # AngelSlim [E, 2*Im, Lz] and [E, Lz, Im] layout ([E, out, in], matching HYV4FlashExperts)
            p + "mlp.experts.gate_up_proj": w((E, 2 * Im, Lz)),
            p + "mlp.experts.down_proj": w((E, Lz, Im)),
            p + "mlp.shared_experts.gate_proj.weight": w((Is, H)),
            p + "mlp.shared_experts.up_proj.weight": w((Is, H)),
            p + "mlp.shared_experts.down_proj.weight": w((H, Is)),
        })
    with open(os.path.join(root_dir, "config.json"), "w") as f:
        json.dump(raw_cfg, f)
    save_file(raw_tensors, os.path.join(root_dir, "model.safetensors"))
    return raw_cfg, raw_tensors


def _angelspec_reference_run(raw_cfg: dict, raw_t: dict, eng: Hy4FlashEngine, st, anchor: int, th: torch.Tensor):
    """Independent AngelSpec-order reference in float32/bf16 for parity verification."""
    cfg = eng.cfg
    a, R, eps = cfg.sparse_attn, cfg.kv_lora_rank, raw_cfg["rms_norm_eps"]
    depth = raw_cfg["block_size"] - 1
    Q, Lt = depth + 1, st.length
    W = {k[6:]: v for k, v in raw_t.items()}

    def rms(x, wt):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * wt.to(x.dtype)

    from engine.models.hy4_flash.ref import rotate
    pos = torch.arange(Lt, Lt + Q) + st.delta
    cos, sin = eng.rope(eng.pos3(pos), SPARSE)

    inp_toks = torch.tensor([anchor] + [raw_cfg["mask_token_id"]] * depth, dtype=torch.int64)
    emb = eng.embed(inp_toks)
    f0 = torch.cat((rms(emb[:1], W["mtp_fusion.enorm.weight"]), rms(th.to(emb.dtype), W["mtp_fusion.hnorm.weight"])), dim=-1)
    h0 = emb[:1] + W["mtp_fusion.gate"] * F.linear(f0, W["mtp_fusion.eh_proj.weight"])
    h = torch.cat((h0, emb[1:]), dim=0)

    tril = torch.ones(Q, Q, dtype=torch.bool).tril()
    for i in range(raw_cfg["num_hidden_layers"]):
        p = f"layers.{i}."
        x = rms(h, W[p + "input_layernorm.weight"])
        gate = F.linear(x, W[p + "self_attn.linear_gate.weight"])
        q = F.linear(rms(F.linear(x, W[p + "self_attn.q_a_proj.weight"]), W[p + "self_attn.q_a_layernorm.weight"]),
                     W[p + "self_attn.q_b_proj.weight"]).view(Q, a.heads, a.qk)
        q = torch.cat((q[..., :a.nope], rotate(q[..., a.nope:], cos[:, None], sin[:, None])), dim=-1)
        ckv = F.linear(x, W[p + "self_attn.kv_a_proj_with_mqa.weight"])
        own = torch.cat((rms(ckv[:, :R], W[p + "self_attn.kv_a_layernorm.weight"]), rotate(ckv[:, R:], cos, sin)), dim=-1)

        pi = st.pick_layers.index(raw_cfg["target_index_layer"][i])
        pq = st.pick_q[pi]
        ok = st.pick_ok[pi] & (pq < Lt)
        lat = st.lat[raw_cfg["target_kv_layers"][i]]
        ctx = lat[pq.clamp(0, lat.shape[0] - 1)]
        kv_all = torch.cat((ctx, own), dim=0)
        kv_b = W[p + "self_attn.kv_b_proj.weight"].view(a.heads, a.nope + a.v, R)
        wk, wv = kv_b[:, :a.nope], kv_b[:, a.nope:].transpose(1, 2)
        qa = torch.einsum("qhd,hdr->qhr", q[..., :a.nope], wk).float()
        s = (qa @ kv_all[:, :R].float().T + q[..., a.nope:].float() @ kv_all[:, R:].float().T) * (a.qk ** -0.5)
        allowed = torch.cat((ok[None, :].expand(Q, -1), tril), dim=-1)
        s = s.masked_fill(~allowed[:, None, :], float("-inf"))
        s = torch.cat((s, W[p + "self_attn.learnable_sink_param"].view(1, a.heads, 1).expand(Q, -1, -1)), dim=-1)
        p_attn = F.softmax(s, dim=-1)[..., :-1]
        y = torch.einsum("qhr,hrv->qhv", (p_attn @ kv_all[:, :R].float()).to(h.dtype), wv)
        y = (y * torch.sigmoid(gate).view(Q, a.heads, a.v)).reshape(Q, a.heads * a.v)
        h = h + rms(F.linear(y, W[p + "self_attn.o_proj.weight"]), W[p + "self_attn.post_proj_norm.weight"])

        x = rms(h, W[p + "post_attention_layernorm.weight"])
        scores = F.linear(x.float(), W[p + "mlp.gate.weight"].float()).sigmoid()
        idx = torch.topk(scores + W[p + "mlp.expert_bias"], raw_cfg["num_experts_per_tok"], dim=-1).indices
        wts = scores.gather(1, idx)
        wts = (wts / (wts.sum(dim=-1, keepdim=True) + 1e-20)) * raw_cfg["router_scaling_factor"]
        z = F.linear(x, W[p + "mlp.routed_expert_down_proj.weight"])
        gu_w = W[p + "mlp.experts.gate_up_proj"]
        d_w = W[p + "mlp.experts.down_proj"]
        Im, lim = raw_cfg["moe_intermediate_size"], raw_cfg["swiglu_limit"]
        acc = torch.zeros_like(z, dtype=torch.float32)
        for r in range(Q):
            for slot in range(raw_cfg["num_experts_per_tok"]):
                e = int(idx[r, slot])
                gu = F.linear(z[r:r + 1], gu_w[e])
                g_val = gu[:, :Im].clamp(max=lim)
                u_val = gu[:, Im:].clamp(min=-lim, max=lim)
                out_e = F.linear(F.silu(g_val) * u_val, d_w[e])
                acc[r] += out_e[0].float() * wts[r, slot]
        routed = F.linear(acc.to(h.dtype), W[p + "mlp.routed_expert_up_proj.weight"])
        sh_g = F.silu(F.linear(x, W[p + "mlp.shared_experts.gate_proj.weight"]))
        shared = F.linear(sh_g * F.linear(x, W[p + "mlp.shared_experts.up_proj.weight"]),
                          W[p + "mlp.shared_experts.down_proj.weight"])
        h = h + rms(routed + shared, W[p + "mlp.combine_output_norm.weight"])

    h = rms(h, W["norm.weight"])

    # AttentiveCorrection across slots 0..depth
    hc_kv_layer = raw_cfg["target_kv_layers"][raw_cfg["attentive_correction_source_layer"]]
    hc_idx_layer = raw_cfg["target_index_layer"][raw_cfg["attentive_correction_source_layer"]]
    pi_hc = st.pick_layers.index(hc_idx_layer)
    pq_hc = st.pick_q[pi_hc]
    ok_hc = st.pick_ok[pi_hc] & (pq_hc < Lt)
    ctx_hc = st.lat[hc_kv_layer][pq_hc.clamp(0, st.lat[hc_kv_layer].shape[0] - 1)]
    hc_kv_b = W["hidden_correction.attention.kv_b_proj.weight"].view(a.heads, a.nope + a.v, R)
    hc_wk, hc_wv = hc_kv_b[:, :a.nope], hc_kv_b[:, a.nope:].transpose(1, 2)

    blk_keys = []
    out_toks, out_probs = [], []
    prev_tok = torch.tensor([anchor], dtype=torch.int64)
    for slot in range(Q):
        e_prev = eng.embed(prev_tok)
        inp = torch.cat((rms(h[slot:slot + 1], W["hidden_correction.hidden_norm.weight"]),
                         rms(e_prev, W["hidden_correction.embed_norm.weight"])), dim=-1)
        ckv = F.linear(inp, W["hidden_correction.attention.kv_in.weight"])
        blk_keys.append(torch.cat((rms(ckv[:, :R], W["hidden_correction.attention.kv_a_layernorm.weight"]),
                                   rotate(ckv[:, R:], cos[slot:slot + 1], sin[slot:slot + 1])), dim=-1))
        if slot == 0:
            continue
        q = F.linear(rms(F.linear(inp, W["hidden_correction.attention.q_in.weight"]),
                         W["hidden_correction.attention.q_a_layernorm.weight"]),
                     W["hidden_correction.attention.q_b_proj.weight"]).view(1, a.heads, a.qk)
        q = torch.cat((q[..., :a.nope], rotate(q[..., a.nope:], cos[slot:slot + 1, None], sin[slot:slot + 1, None])), dim=-1)
        qa = torch.einsum("thd,hdr->thr", q[..., :a.nope], hc_wk).float()
        ctx_s = (qa @ ctx_hc[:, :R].float().T + q[..., a.nope:].float() @ ctx_hc[:, R:].float().T) * (a.qk ** -0.5)
        ctx_s = ctx_s.masked_fill(~ok_hc[None, None, :], float("-inf"))
        ctx_s = torch.cat((ctx_s, W["hidden_correction.attention.learnable_sink_param"].view(1, a.heads, 1)), dim=-1)
        ctx_read = F.softmax(ctx_s, dim=-1)[..., :-1] @ ctx_hc[:, :R].float()

        kb = torch.cat(blk_keys, dim=0).float()
        back = torch.arange(slot, -1, -1)
        rel = W["hidden_correction.attention.block_rel_bias"][:, back].unsqueeze(0)
        bs = (qa @ kb[:, :R].T + q[..., a.nope:].float() @ kb[:, R:].T) * (a.qk ** -0.5) + rel
        btot = torch.logaddexp(torch.logsumexp(bs, dim=-1, keepdim=True),
                               W["hidden_correction.attention.block_sink"].view(1, a.heads, 1))
        blk_read = torch.exp(bs - btot) @ kb[:, :R]
        gate = torch.tanh(F.linear(inp, W["hidden_correction.attention.block_gate.weight"]).float()
                          + W["hidden_correction.attention.block_gate_bias"][:, slot].unsqueeze(0))
        mix = (ctx_read + blk_read * gate.unsqueeze(-1)).to(h.dtype)
        attn_out = F.linear(torch.einsum("thr,hrv->thv", mix, hc_wv).reshape(1, -1),
                            W["hidden_correction.attention.o_proj.weight"])
        g_m = F.silu(F.linear(inp, W["hidden_correction.gate_proj.weight"]))
        mlp_out = F.linear(g_m * F.linear(inp, W["hidden_correction.up_proj.weight"]),
                           W["hidden_correction.down_proj.weight"])
        c = h[slot:slot + 1] + attn_out + mlp_out
        top = torch.topk(torch.softmax(eng.head(c)[0].float(), dim=-1), 2)
        out_toks.append(top.indices.tolist())
        out_probs.append(top.values.tolist())
        prev_tok = top.indices[:1]
    return out_toks, out_probs


def test_dfly2_b6_external_load_and_parity():
    torch.manual_seed(0)
    cfg = config_from_dict({**TOP, "text_config": dict(TINY, model_type="hy_v4_flash")}, "tiny")
    Wt, tables = random_weights(cfg)
    Wt.update(dfly_weights(cfg))
    for k, v in list(Wt.items()):
        if torch.is_tensor(v) and v.is_floating_point() and not k.endswith(
            ("hc_fn", "hc_base", "hc_scale", "hc_head_fn", "hc_head_base", "hc_head_scale",
             "learnable_sink_param", "e_score_correction_bias", "key_value_projection.weight",
             "key_gate_projection.weight", "intra_group_ape")
        ):
            Wt[k] = v.to(torch.bfloat16)
        elif hasattr(v, "dense") and v.dense is not None:
            v.dense = v.dense.to(torch.bfloat16)
    with tempfile.TemporaryDirectory() as d:
        raw_cfg, raw_t = _make_dfly2_checkpoint(d, cfg)
        d_cfg, Wd, cuts, banks = load_dfly_external(d, cfg, device="cpu", dtype=torch.bfloat16)
        assert d_cfg.index_layers == [1, 5] and d_cfg.draft_tokens == 3
        cfg.dfly = d_cfg
        eng = Hy4FlashEngine(cfg, Wt, tables, dtype=torch.bfloat16, device="cpu", max_len=64, cuts=cuts)
        eng.dfly_cfg, eng.dfly_weights, eng.dfly_banks = d_cfg, Wd, banks
        assert eng.pick_layers == {1, 5}
        srv = Hy4FlashServing(eng, 64)
        head = DFlyHead(srv)
        assert head.depth == 3 and head.hc_kv_layer == 5 and "hc_blk" in head.buf

        # Case 1: populated context after prefill + commit
        lg = srv.forward(torch.tensor([4, 9, 13, 21, 31, 42]), start=0)
        anchor = int(lg[0, 0].argmax())
        th = srv.last_streams().clone()
        head.launch(th, [anchor])
        got_toks, got_probs = head.read()
        want_toks, want_probs = _angelspec_reference_run(raw_cfg, raw_t, eng, srv.st, anchor, th)
        assert got_toks == want_toks, (got_toks, want_toks)
        max_dp = max(abs(a - b) for r1, r2 in zip(got_probs, want_probs) for a, b in zip(r1, r2))
        assert max_dp < 1e-4, f"prob diff {max_dp}"

        # Case 2: zero valid context keys (sink stability, zero NaNs)
        srv.st.pick_ok.zero_()
        head.launch(th, [anchor])
        toks_no_ctx, probs_no_ctx = head.read()
        want_no_ctx, _ = _angelspec_reference_run(raw_cfg, raw_t, eng, srv.st, anchor, th)
        assert toks_no_ctx == want_no_ctx
        assert all(torch.isfinite(torch.tensor(r)).all() for r in probs_no_ctx)

        # Case 3: stale round after resume produces nothing
        srv.resume_at(2)
        head.launch(th, [anchor])
        assert head.read() == ([], []) and head.stats["stale"] == 1

        # Case 4: custom draft_head projection override
        head_custom = DFlyHead(srv, draft_head=Wt["lm_head.weight"] * 0.5)
        assert head_custom.draft_head is not None
    return "BF16 external load, [E,out,in] (Im!=Lz) unpack, MoE + AttentiveCorrection 3-step chain == AngelSpec reference"


@torch.no_grad()
def test_forward_tree_multi_matches_single_lane_verify():
    cfg = config_from_dict({**TOP, "text_config": dict(TINY, model_type="hy_v4_flash")}, "tiny")
    Wt, tables = random_weights(cfg)
    eng = Hy4FlashEngine(cfg, Wt, tables, dtype=torch.bfloat16, device="cpu", max_len=128)
    assert Hy4FlashServing.caps.joint_verify and Hy4FlashServing.caps.joint_plain

    # Lone baseline runs
    p0 = [5, 7, 11, 13, 19, 23]                   # below index_topk=8
    p1 = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]   # past index_topk=8 and sliding_window=4
    b0, par0 = torch.tensor([17, 29, 31, 41]), [-1, 0, 1, 0]
    b1, par1 = torch.tensor([18, 22, 26]), [-1, 0, 1]

    s0_ref = Hy4FlashServing(eng, 128)
    s1_ref, _ = s0_ref.side_slot(128, None)
    s0_ref.forward(torch.tensor(p0), start=0)
    s1_ref.forward(torch.tensor(p1), start=0)
    want0 = s0_ref.forward_tree(b0, par0, start=len(p0)).clone()
    want1 = s1_ref.forward_tree(b1, par1, start=len(p1)).clone()
    want0_picks, want1_picks = s0_ref.picks.clone(), s1_ref.picks.clone()

    # Joint pass across both lanes in one verify_multi call
    s0 = Hy4FlashServing(eng, 128)
    s1, _ = s0.side_slot(128, None)
    s0.forward(torch.tensor(p0), start=0)
    s1.forward(torch.tensor(p1), start=0)
    got0, got1 = Hy4FlashServing.forward_tree_multi([
        (s0, b0, par0, len(p0)),
        (s1, b1, par1, len(p1)),
    ])
    assert torch.equal(got0, want0) and torch.equal(s0.picks, want0_picks)
    assert torch.equal(got1, want1) and torch.equal(s1.picks, want1_picks)
    Hy4FlashServing.commit_tree_multi([(s0, [0, 1, 2]), (s1, [0, 1])])
    s0_ref.commit_tree([0, 1, 2])
    s1_ref.commit_tree([0, 1])

    # Plain 1-row decode_multi parity
    d0 = s0_ref.forward_tree(torch.tensor([9]), [-1], start=s0_ref.kv.length).clone()
    d1 = s1_ref.forward_tree(torch.tensor([12]), [-1], start=s1_ref.kv.length).clone()
    jg0, jg1 = Hy4FlashServing.forward_tree_multi([
        (s0, torch.tensor([9]), [-1], s0.kv.length),
        (s1, torch.tensor([12]), [-1], s1.kv.length),
    ])
    assert torch.equal(jg0, d0) and torch.equal(jg1, d1)
    return "2-lane mixed verify_multi & decode_multi match single-lane forward_tree bit-for-bit"


@torch.no_grad()
def test_dfly2_run_packed_and_pipe_pre_commit():
    from engine.methods import draft_meet
    from engine.methods.mtp_drafter import MTPTreeDrafter
    from engine.methods.staircut import PriceModel, StairCut

    cfg = config_from_dict({**TOP, "text_config": dict(TINY, model_type="hy_v4_flash")}, "tiny")
    Wt, tables = random_weights(cfg)
    with tempfile.TemporaryDirectory() as d:
        _make_dfly2_checkpoint(d, cfg)
        d_cfg, Wd, cuts, banks = load_dfly_external(d, cfg, device="cpu", dtype=torch.bfloat16)
        cfg.dfly = d_cfg
        eng = Hy4FlashEngine(cfg, Wt, tables, dtype=torch.bfloat16, device="cpu", max_len=64, cuts=cuts)
        eng.dfly_cfg, eng.dfly_weights, eng.dfly_banks = d_cfg, Wd, banks

        # 1. Batched run_packed vs individual runs across 3 lanes
        lanes = [Hy4FlashServing(eng, 64)]
        lanes.append(lanes[0].side_slot(64, None)[0])
        lanes.append(lanes[0].side_slot(64, None)[0])
        heads = [DFlyHead(s) for s in lanes]
        prompts = [[4, 9, 13, 21, 31], [7, 11, 15, 19, 23, 27, 35, 39], [6, 12, 18, 24]]
        anchors = [17, 29, 41]
        ths = []
        for s, h, p, anc in zip(lanes, heads, prompts, anchors):
            s.forward(torch.tensor(p), start=0)
            th = s.last_streams().clone()
            ths.append(th)
            h.stage(anc, th, s.st)
            h.run(s.st)
        want = [(h.buf["tok"].tolist(), h.buf["prob"].tolist()) for h in heads]

        for h in heads:
            h.buf["tok"].zero_()
            h.buf["prob"].zero_()
        DFlyHead.run_packed(heads)
        for idx, h in enumerate(heads):
            got_tok, got_prob = h.buf["tok"].tolist(), h.buf["prob"].tolist()
            assert got_tok == want[idx][0], (idx, got_tok, want[idx][0])
            dp = max(abs(a - b) for r1, r2 in zip(got_prob, want[idx][1]) for a, b in zip(r1, r2))
            assert dp < 1e-4, dp

        # 2. QSE_RUNNER_PIPE=1 ordering: sync(rows=path) -> launch() BEFORE commit_tree(path) -> commit_tree(path)
        s_nopipe = Hy4FlashServing(eng, 64)
        s_pipe = Hy4FlashServing(eng, 64)
        h_nopipe, h_pipe = DFlyHead(s_nopipe), DFlyHead(s_pipe)
        dr_nopipe = MTPTreeDrafter(s_nopipe, h_nopipe, StairCut(PriceModel({}, 1.0)), None)
        dr_pipe = MTPTreeDrafter(s_pipe, h_pipe, StairCut(PriceModel({}, 1.0)), None)
        dr_nopipe.wants_rows = dr_pipe.wants_rows = True

        prompt = [4, 9, 13, 21, 31, 42]
        for s, dr in ((s_nopipe, dr_nopipe), (s_pipe, dr_pipe)):
            s.forward(torch.tensor(prompt), start=0)
            dr.reset()
            dr.sync(prompt, s.hidden_post_norm[0].clone(), 0)
        blk = torch.tensor([17, 25, 33, 44])
        par, path = [-1, 0, 1, 2], [0, 1, 2]
        lg_np = s_nopipe.forward_tree(blk, par, start=len(prompt))
        hid_np = s_nopipe.hidden_post_norm[0, path].clone()
        lg_p = s_pipe.forward_tree(blk, par, start=len(prompt))
        hid_p = s_pipe.hidden_post_norm[0, path].clone()
        assert torch.equal(lg_np, lg_p)

        # Non-piped: commit_tree BEFORE launch
        s_nopipe.commit_tree(path)
        dr_nopipe.sync([17, 25, 33], hid_np, len(prompt), rows=path)
        h_nopipe.launch(hid_np[-1:], [33])
        ref_tok, ref_prob = h_nopipe.read()

        # Piped: sync(rows=path) -> launch() -> commit_tree(path)
        saved_meet = draft_meet.MEET
        draft_meet.MEET = None
        try:
            dr_pipe.sync([17, 25, 33], hid_p, len(prompt), rows=path)
            h_pipe.launch(hid_p[-1:], [33])
            pipe_tok, pipe_prob = h_pipe.read()
            s_pipe.commit_tree(path)
        finally:
            draft_meet.MEET = saved_meet
        assert h_pipe.stats["stale"] == 0 and s_pipe.kv.length == len(prompt) + len(path)
        assert pipe_tok == ref_tok and pipe_prob == ref_prob
    return "run_packed 3-lane parity & QSE_RUNNER_PIPE=1 pre_commit == QSE_RUNNER_PIPE=0 bit-for-bit"


@torch.no_grad()
def test_prefill_yields_between_layers_for_d12():
    cfg = config_from_dict({**TOP, "text_config": dict(TINY, model_type="hy_v4_flash")}, "tiny")
    Wt, tables = random_weights(cfg)
    eng = Hy4FlashEngine(cfg, Wt, tables, dtype=torch.bfloat16, device="cpu", max_len=64)

    # Lone references
    s_pref_ref = Hy4FlashServing(eng, 64)
    s_dec_ref, _ = s_pref_ref.side_slot(64, None)
    p0 = [5, 7, 11, 13, 19, 23]
    p1 = [6, 12, 18, 24]
    s_dec_ref.forward(torch.tensor(p1), start=0)
    want_pref = s_pref_ref.forward(torch.tensor(p0), start=0).clone()
    want_dec = s_dec_ref.forward_tree(torch.tensor([17, 29, 31]), [-1, 0, 1], start=len(p1)).clone()

    # Interleave side-slot decode in the middle of prefill's layer yield
    s_pref = Hy4FlashServing(eng, 64)
    s_dec, _ = s_pref.side_slot(64, None)
    s_dec.forward(torch.tensor(p1), start=0)
    yields = [0]
    got_dec = [None]

    def on_yield():
        yields[0] += 1
        if yields[0] == 2:
            got_dec[0] = s_dec.forward_tree(torch.tensor([17, 29, 31]), [-1, 0, 1], start=len(p1)).clone()

    eng.layer_yield = on_yield
    try:
        got_pref = s_pref.forward(torch.tensor(p0), start=0).clone()
    finally:
        eng.layer_yield = None

    assert yields[0] == cfg.num_hidden_layers - 1, (yields[0], cfg.num_hidden_layers)
    assert torch.equal(got_pref, want_pref) and torch.equal(got_dec[0], want_dec)
    return f"prefill yielded {yields[0]} times per chunk; interleaved side decode & prefill bit-equal"


@torch.no_grad()
def test_quant_capture_and_probe_restore_smoke():
    """Smoke-test tools/hy4_dfly2_quant.py cmd_capture + probe restore (`st.picks_at`) end-to-end on CPU."""
    from types import SimpleNamespace
    import engine.models.hy4_flash.load as hy4_load
    from tools import hy4_dfly2_quant as qmod

    cfg = config_from_dict({**TOP, "text_config": dict(TINY, model_type="hy_v4_flash")}, "tiny")
    Wt, tables = random_weights(cfg)
    for k, v in list(Wt.items()):
        if torch.is_tensor(v) and v.is_floating_point() and not k.endswith(
            ("hc_fn", "hc_base", "hc_scale", "hc_head_fn", "hc_head_base", "hc_head_scale",
             "learnable_sink_param", "e_score_correction_bias", "key_value_projection.weight",
             "key_gate_projection.weight", "intra_group_ape")
        ):
            Wt[k] = v.to(torch.bfloat16)
        elif hasattr(v, "dense") and v.dense is not None:
            v.dense = v.dense.to(torch.bfloat16)

    with tempfile.TemporaryDirectory() as d:
        dfly_dir = os.path.join(d, "dfly")
        hess_dir = os.path.join(d, "hess")
        calib_path = os.path.join(d, "calib.safetensors")
        os.makedirs(dfly_dir)
        _make_dfly2_checkpoint(dfly_dir, cfg)
        ids = torch.randint(0, cfg.vocab_size, (4, 32), dtype=torch.int64)
        lens = torch.full((4,), 32, dtype=torch.int64)
        save_file({"input_ids": ids, "lengths": lens}, calib_path)

        orig_load = hy4_load.load
        hy4_load.load = lambda *a, **kw: (SimpleNamespace(cfg=cfg), dict(Wt), {}, tables, {}, {})
        try:
            qmod.cmd_capture(SimpleNamespace(
                target="unused", dfly=dfly_dir, calib=calib_path, hess_dir=hess_dir,
                seqs=2, prompt=12, blocks=2, probes=1, device="cpu",
            ))
            for k in ("mtp_fusion.eh_proj", "l0.attn_in", "l0.e0.gu", "l0.e0.down", "hc.in", "hc.kv_b_proj", "lm_head"):
                assert os.path.isfile(os.path.join(hess_dir, f"{k}.pt")), f"missing Hessian {k}"
            g_h = torch.load(os.path.join(hess_dir, "l0.e0.gu.pt"), map_location="cpu")["H"]
            d_h = torch.load(os.path.join(hess_dir, "l0.e0.down.pt"), map_location="cpu")["H"]
            assert g_h.shape == (24, 24) and d_h.shape == (32, 32), (g_h.shape, d_h.shape)
            qmod.cmd_verify(SimpleNamespace(
                target="unused", dfly="", quant_dir=dfly_dir, hess_dir=hess_dir, head_file="", device="cpu",
            ))
            rep = json.load(open(os.path.join(dfly_dir, "verify_report.json")))
            assert rep["hidden_cos_mean"] > 0.9999 and rep["top1_match_full_head"] == 1.0, rep
        finally:
            hy4_load.load = orig_load
    return "cmd_capture + cmd_verify BF16 probe restore smoke passed (Im=32 != Lz=24, stale=0)"


if __name__ == "__main__":
    for fn in (test_dfly2_b6_config_parsing_matches_angelslim,
               test_split_rope_matches_interleaved_dot_product,
               test_dfly2_b6_external_load_and_parity,
               test_forward_tree_multi_matches_single_lane_verify,
               test_dfly2_run_packed_and_pipe_pre_commit,
               test_prefill_yields_between_layers_for_d12,
               test_quant_capture_and_probe_restore_smoke):
        print(f"PASS {fn.__name__}: {fn()}")
