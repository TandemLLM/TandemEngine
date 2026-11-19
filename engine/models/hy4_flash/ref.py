"""The Hy4-flash forward in plain torch: slow, exact, the reference every kernel is checked against.

It follows the published implementation (`transformers` pull request 50903 at b865850d,
`models/hy_v4_flash`) operation for operation, on the weights of the served pack: an EXL3 projection is
multiplied by its dequantised weight, an Engram row is what the table decodes. Activations stay in the
compute dtype (bf16 when serving; fp32 in the tight CPU test against `transformers`). The dtypes inside
follow the reference: RMS norms compute in fp32 and multiply by the weight after the cast back, the
hyper-connections, the router, the indexer's pooling and scores and the head run in fp32, the attention
softmax runs in fp32 because the fp32 sinks join its logits, and the Engram norms are torch's own
`rms_norm` in the compute dtype with its default epsilon.

One step of the model, per layer (S = 4 residual streams of `hidden`):

    h [T, S, H]
    h += engram(h, tokens)                      (4 layers: hashed n-gram rows, gated a stream)
    pre, post = hc_attn(h)                      fp32 gates from the flattened streams
    y = attention(norm(sum_s pre_s h_s))        gated latent attention with a sink, sandwich norm
    h = post (x) y + h                          inject into every stream with weight post_s
    pre, post = hc_mlp(h)
    y = mlp(norm(sum_s pre_s h_s))              layer 0: dense; else latent MoE with a shared expert
    h = post (x) y + h
    ...
    logits = head(norm(hc_head(h)))             fp32 head

Attention reads latents: a token keeps `kv_lora_rank` normed values and `rope` turned key channels in
every layer, and a layer expands the rows it reads into keys and values. A window layer reads the
last `sliding_window` rows. A sparse layer reads every row up to `index_topk` rows of context; past
that its indexer pools every `index_kpool` token keys into a group key, scores the closed groups and
keeps the best `group_topk - 1` of them beside the query's own group.

Differences from the reference, all deliberate and all value-preserving:
  * top-k ties (router, indexer groups) resolve to the lower index (a stable sort), so the choice
    never depends on the launch shape; the reference's `torch.topk` leaves ties unspecified.
  * batch size is 1 (the engine serves one sequence per state), and the state keeps every layer's
    rows by slot (the reference's window layers drop rows that left the window; nothing reads them).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from engine.models.flashnext.ref import ExpertBank, lin, topk_stable  # noqa: F401  (ExpertBank: the tests' banks)
from engine.models.hy4_flash.config import SPARSE, Hy4FlashConfig
from engine.models.hy4_flash.engram import EngramHash, fp8_round_trip, ngram_tokens


# ------------------------------------------------------------------------------ state
@dataclass
class Hy4State:
    """Everything a continuation needs, for batch size 1."""
    length: int = 0
    tokens: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.int64))   # [L] ids by slot
    pos: torch.Tensor = field(default_factory=lambda: torch.zeros(3, 0, dtype=torch.int64))   # [3, L] rotary coordinates
    kp: dict = field(default_factory=dict)       # layer -> [L, kv_lora_rank] normed latents
    kr: dict = field(default_factory=dict)       # layer -> [L, rope] turned key channels
    ik: dict = field(default_factory=dict)       # sparse layer -> [L, index_head_dim] token keys
    live: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.bool))      # [L] rows that own an n-gram
    # the rotary offset of the rows after the prompt (position = slot + delta; images move it)
    delta: int = 0

    def clone(self) -> "Hy4State":
        return Hy4State(self.length, self.tokens.clone(), self.pos.clone(), {k: v.clone() for k, v in self.kp.items()},
                        {k: v.clone() for k, v in self.kr.items()}, {k: v.clone() for k, v in self.ik.items()},
                        self.live.clone(), self.delta)

    def truncate(self, n: int) -> None:
        """Back to the first n rows (every part is kept by slot, so this is exact)."""
        if not 0 <= n <= self.length:
            raise ValueError(f"cannot truncate a state of {self.length} rows to {n}")
        self.length = n
        self.tokens, self.pos, self.live = self.tokens[:n], self.pos[:, :n], self.live[:n]
        for d in (self.kp, self.kr, self.ik):
            for k in d:
                d[k] = d[k][:n]


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The rotary on interleaved pairs of x's last axis, in fp32, back in x's dtype
    (`apply_rotary_pos_emb`); cos and sin [..., pairs] broadcast against x[..., 2 * pairs]."""
    c = cos.repeat_interleave(2, dim=-1)
    s = sin.repeat_interleave(2, dim=-1)
    half = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
    return (x.float() * c + half.float() * s).to(x.dtype)


class Hy4FlashRef:
    """`W` maps the checkpoint's tensor names without the `model.` prefix to a tensor, an EXL3 projection
    or (the routed experts' `gate_proj`, `up_proj`, `down_proj`) an `ExpertBank`. `tables` maps an Engram
    layer's index to its table (`gather(ids, dtype) -> [n, 96]`; `fp8_done` True when its rows already
    went through the e4m3 round trip)."""

    def __init__(self, cfg: Hy4FlashConfig, W: dict, tables: dict, dtype=torch.bfloat16, device="cpu", cuts: dict | None = None):
        self.cfg, self.W, self.tables, self.dtype, self.device = cfg, W, tables, dtype, torch.device(device)
        self.cuts = cuts or {}
        self.hashes = {li: EngramHash.from_config(cfg, lid, W[f"layers.{li}.engram_layer.lookup_table"])
                       for li, lid in cfg.engram_layers.items()}
        self.inv_freq = {}
        for kind in set(cfg.layer_types):
            r = (cfg.sparse_attn if kind == SPARSE else cfg.window_attn).rope
            self.inv_freq[kind] = (1.0 / (cfg.rope_theta[kind] ** (torch.arange(0, r, 2, dtype=torch.float) / r))).to(self.device)

    # ---------------------------------------------------------------- primitives
    def w(self, name: str):
        return self.W[name]

    def lin(self, x, w):
        """x @ w^T. A projection the converter padded to a multiple of 128 takes x padded with zeros and gives its
        real width (`cuts`: id(weight) -> width)."""
        k = getattr(w, "K", None)
        if k is not None and x.shape[-1] < k:
            x = F.pad(x, (0, k - x.shape[-1]))
        y = lin(x, w)
        n = self.cuts.get(id(w))
        return y if n is None else y[..., :n]

    def rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """`HYV4FlashRMSNorm`: the norm in fp32, cast back, then times the weight."""
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        return w.to(self.dtype) * h.to(x.dtype)

    def rope(self, pos: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
        """[3, n] coordinates (time, height, width; equal on a text row) -> cos, sin [n, rope / 2] in the
        compute dtype: the time plane with the height and width frequencies interleaved in."""
        freqs = pos.to(self.device)[..., None].float() * self.inv_freq[kind]
        sec = self.cfg.mrope_section[kind]
        out = []
        for f in (freqs.cos(), freqs.sin()):
            thw = f[0].clone()
            for dim, offset in ((1, 1), (2, 2)):
                idx = slice(offset, sec[dim] * 3, 3)
                thw[..., idx] = f[dim][..., idx]
            out.append(thw.to(self.dtype))
        return out[0], out[1]

    # ---------------------------------------------------------------- hyper-connections
    def hc(self, h: torch.Tensor, p: str) -> tuple[torch.Tensor, torch.Tensor]:
        """h [T, S, H] -> the streams' sum weighted by the pre gates [T, H] and the post gates [T, S] (fp32)."""
        S = self.cfg.hc_mult
        flat = h.flatten(1).float()
        mixes = F.linear(flat, self.w(p + "hc_fn").float())
        mixes = mixes * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        base, scale = self.w(p + "hc_base").float(), self.w(p + "hc_scale").float()
        pre = torch.sigmoid(mixes[:, :S] * scale[0] + base[:S]) + self.cfg.hc_eps
        post = self.cfg.hc_magnitude * torch.sigmoid(mixes[:, S:] * scale[1] + base[S:]) + self.cfg.hc_eps
        x = torch.sum(pre.unsqueeze(-1) * h, dim=1).to(h.dtype)
        return x, post

    def hc_head(self, h: torch.Tensor) -> torch.Tensor:
        flat = h.flatten(1).float()
        mixes = F.linear(flat, self.w("hc_head.hc_head_fn").float())
        mixes = mixes * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        pre = torch.sigmoid(mixes * self.w("hc_head.hc_head_scale").float() + self.w("hc_head.hc_head_base").float())
        return ((pre + self.cfg.hc_eps).unsqueeze(-1) * h).sum(dim=1).to(h.dtype)

    @staticmethod
    def inject(h: torch.Tensor, y: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        return (post.float().unsqueeze(-1) * y.float().unsqueeze(-2) + h.float()).to(h.dtype)

    # ---------------------------------------------------------------- Engram
    def engram_rows(self, li: int, ids: torch.Tensor) -> torch.Tensor:
        """Table rows [..., heads, head_dim] in the compute dtype, as the reference reads them."""
        t = self.tables[li]
        e = t.gather(ids.reshape(-1).cpu(), dtype=self.dtype).to(self.device).reshape(*ids.shape, -1)
        if self.cfg.engram_fp8_embedding and not getattr(t, "fp8_done", False):
            e = fp8_round_trip(e)
        return e

    def engram(self, h: torch.Tensor, grams: torch.Tensor, li: int, live: torch.Tensor) -> torch.Tensor:
        """The residual the memory adds to h [T, S, H]; `grams` [T, n] token ids, `live` [T] rows that own one."""
        return self.engram_apply(h, self.engram_rows(li, self.hashes[li].ids(grams)), li, live)

    def engram_apply(self, h: torch.Tensor, rows: torch.Tensor, li: int, live: torch.Tensor) -> torch.Tensor:
        cfg, p = self.cfg, f"layers.{li}.engram_layer."
        H, d = cfg.hidden_size, cfg.engram_head_dim
        emb = torch.stack([F.rms_norm(rows[:, hd], (d,), self.w(f"{p}lookup_per_head_norms.{hd}.weight").to(self.dtype))
                           for hd in range(cfg.engram_heads)], dim=1).flatten(1)
        gates = []
        for s in range(cfg.hc_mult):
            key = F.linear(emb, self.w(f"{p}key_projs.{s}.weight").to(self.dtype), self.w(f"{p}key_projs.{s}.bias").to(self.dtype))
            key = F.rms_norm(key, (H,), self.w(f"{p}norm_key.{s}.weight").to(self.dtype))
            query = F.rms_norm(h[:, s], (H,), self.w(f"{p}norm_query.{s}.weight").to(self.dtype))
            g = (key * query).sum(dim=-1) * H ** -0.5
            g = g.abs().clamp_min(1e-6).sqrt() * g.sign()
            gates.append(g.sigmoid())
        gates = torch.stack(gates, dim=1).unsqueeze(-1)
        value = F.linear(emb, self.w(p + "value_proj.weight").to(self.dtype), self.w(p + "value_proj.bias").to(self.dtype))
        main = self.w(p + "unified_pre_gamma").to(self.dtype) * F.rms_norm(value, (H,))
        branch = torch.stack([self.w(f"{p}unified_per_hc_scales.{s}").to(self.dtype) * main for s in range(cfg.hc_mult)], dim=1)
        out = gates * main.unsqueeze(1) + branch
        return out.masked_fill(~live.to(out.device)[:, None, None], 0.0)

    # ---------------------------------------------------------------- the indexer
    def group_keys(self, token_keys: torch.Tensor, li: int) -> torch.Tensor:
        """[G * r, D] token keys of closed groups -> [G, D] group keys (gated pooling in fp32, then the norm)."""
        cfg, p = self.cfg, f"layers.{li}.self_attn.compressed_indexer."
        r, D = cfg.index_kpool, cfg.index_head_dim
        g = token_keys.reshape(-1, r, D).float()
        values = F.linear(g, self.w(p + "key_value_projection.weight").float())
        gates = F.linear(g, self.w(p + "key_gate_projection.weight").float()) + self.w(p + "intra_group_ape").float()
        pooled = (values * gates.softmax(dim=1)).sum(dim=1)
        return self.rms(pooled, self.w(p + "key_norm.weight")).to(token_keys.dtype)

    def index_scores(self, x: torch.Tensor, q_resid: torch.Tensor, gk: torch.Tensor, pos: torch.Tensor,
                     gpos: torch.Tensor, li: int) -> torch.Tensor:
        """Every query's score of every group [T, G] (fp32): relu of query head times turned group key,
        summed over the heads with the query's own head weights."""
        cfg, p = self.cfg, f"layers.{li}.self_attn.compressed_indexer."
        nh, D, rd = cfg.index_n_heads, cfg.index_head_dim, cfg.sparse_attn.rope
        T = x.shape[0]
        q = self.lin(q_resid, self.w(p + "query_projection.weight")).view(T, nh, D)
        cos, sin = self.rope(pos, SPARSE)
        q = torch.cat((q[..., :D - rd], rotate(q[..., D - rd:], cos[:, None], sin[:, None])), dim=-1)
        cos, sin = self.rope(gpos, SPARSE)
        keys = torch.cat((gk[..., :D - rd], rotate(gk[..., D - rd:], cos, sin)), dim=-1)
        hw = self.lin(x, self.w(p + "head_weight_projection.weight")).float() * (nh ** -0.5 * D ** -0.5)
        scores = F.relu(torch.matmul(q.float(), keys.float().transpose(0, 1)))        # [T, nh, G]
        return torch.matmul(hw.unsqueeze(-2), scores).squeeze(-2)

    def index_groups(self, logits: torch.Tensor, qgroup: torch.Tensor) -> torch.Tensor:
        """[T, G] scores -> [T, group_topk] groups a query reads: the best `group_topk - 1` that closed
        before its own (ties to the lower index), then its own."""
        G = logits.shape[1]
        gids = torch.arange(G, device=logits.device)
        logits = logits.masked_fill(gids >= qgroup.unsqueeze(-1), float("-inf"))
        top = topk_stable(logits, min(self.cfg.group_topk - 1, G))[1]
        return torch.cat((top, qgroup.unsqueeze(-1)), dim=-1)

    def index_mask(self, x: torch.Tensor, q_resid: torch.Tensor, li: int, st: Hy4State, slots: torch.Tensor,
                   pos: torch.Tensor, L: int) -> torch.Tensor | None:
        """[T, L] rows a sparse layer's queries may read beside the causal rule, or None: all of them."""
        cfg = self.cfg
        if L <= cfg.index_topk:
            return None
        r = cfg.index_kpool
        G = L // r
        gk = self.group_keys(st.ik[li][:G * r], li)
        gpos = st.pos[:, :G * r:r]
        logits = self.index_scores(x, q_resid, gk, pos, gpos, li)
        groups = self.index_groups(logits, (slots // r).to(logits.device))
        rows = (groups.unsqueeze(-1) * r + torch.arange(r, device=groups.device)).flatten(-2).clamp(max=L - 1)
        allowed = torch.zeros(x.shape[0], L, dtype=torch.bool, device=groups.device)
        allowed.scatter_(-1, rows, True)
        return allowed

    # ---------------------------------------------------------------- attention
    def attention(self, x: torch.Tensor, li: int, st: Hy4State, slots: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x [T, H] at `slots` -> the layer's attention output [T, H]; appends the rows' latents to the state."""
        cfg = self.cfg
        a, kind, p = cfg.attn(li), cfg.layer_types[li], f"layers.{li}.self_attn."
        T, R = x.shape[0], cfg.kv_lora_rank
        gate = self.lin(x, self.w(p + "linear_gate.weight")).view(T, a.heads, a.v)
        q_resid = self.rms(self.lin(x, self.w(p + "q_a_proj.weight")), self.w(p + "q_a_layernorm.weight"))
        q = self.lin(q_resid, self.w(p + "q_b_proj.weight")).view(T, a.heads, a.qk)
        ckv = self.lin(x, self.w(p + "kv_a_proj_with_mqa.weight"))
        k_pass = self.rms(ckv[:, :R], self.w(p + "kv_a_layernorm.weight"))
        cos, sin = self.rope(pos, kind)
        q = torch.cat((q[..., :a.nope], rotate(q[..., a.nope:], cos[:, None], sin[:, None])), dim=-1)
        k_rot = rotate(ckv[:, R:], cos, sin)

        st.kp[li] = torch.cat((st.kp[li], k_pass)) if li in st.kp else k_pass
        st.kr[li] = torch.cat((st.kr[li], k_rot)) if li in st.kr else k_rot
        L = int(st.kp[li].shape[0])
        kslot = torch.arange(L, device=x.device)
        allowed = kslot[None, :] <= slots[:, None]
        lo = 0
        if kind == SPARSE:
            tk = self.rms(self.lin(x, self.w(p + "compressed_indexer.token_key_projection.weight")),
                          self.w(p + "compressed_indexer.token_key_norm.weight"))
            st.ik[li] = torch.cat((st.ik[li], tk)) if li in st.ik else tk
            picked = self.index_mask(x, q_resid, li, st, slots, pos, L)
            if picked is not None:
                allowed = allowed & picked
        else:
            allowed = allowed & (kslot[None, :] > slots[:, None] - cfg.sliding_window)
            lo = max(0, int(slots[0]) - cfg.sliding_window + 1)      # rows no query of this call reads
        y = self.attend(q, st.kp[li][lo:], st.kr[li][lo:], allowed[:, lo:], li)
        y = (y * torch.sigmoid(gate)).reshape(T, a.heads * a.v)
        return self.rms(self.lin(y, self.w(p + "o_proj.weight")), self.w(p + "post_proj_norm.weight"))

    def attend(self, q: torch.Tensor, kp: torch.Tensor, kr: torch.Tensor, allowed: torch.Tensor, li: int) -> torch.Tensor:
        """q [T, heads, qk] over latents kp [n, R], kr [n, rope] with `allowed` [T, n] -> [T, heads, v]:
        the rows expand to keys and values, the fp32 sink is one more logit, and its weight is dropped."""
        a, p = self.cfg.attn(li), f"layers.{li}.self_attn."
        n = kp.shape[0]
        kv = self.lin(kp, self.w(p + "kv_b_proj.weight")).view(n, a.heads, a.nope + a.v).transpose(0, 1)
        k = torch.cat((kv[..., :a.nope], kr.unsqueeze(0).expand(a.heads, -1, -1)), dim=-1)     # [heads, n, qk]
        v = kv[..., a.nope:]
        w = torch.matmul(q.transpose(0, 1), k.transpose(1, 2)) * (a.qk ** -0.5)                # [heads, T, n]
        w = w + torch.where(allowed, 0.0, torch.finfo(w.dtype).min).to(w.dtype)
        sinks = self.w(p + "learnable_sink_param").float().reshape(-1, 1, 1).expand(-1, w.shape[1], -1)
        w = torch.cat([w, sinks.to(w.device)], dim=-1)                                          # fp32 from here
        w = w - w.max(dim=-1, keepdim=True).values
        probs = F.softmax(w, dim=-1, dtype=w.dtype)[..., :-1].to(v.dtype)
        return torch.matmul(probs, v).transpose(0, 1)

    # ---------------------------------------------------------------- MLPs
    def route(self, x: torch.Tensor, li: int) -> tuple[torch.Tensor, torch.Tensor]:
        """x [T, H] -> the routed experts [T, k] (best first, ties to the lower index) and their weights (fp32)."""
        cfg, p = self.cfg, f"layers.{li}.mlp.gate."
        scores = F.linear(x.float(), self.w(p + "weight").float()).sigmoid()
        idx = topk_stable(scores + self.w(p + "e_score_correction_bias").float(), cfg.num_experts_per_tok)[1]
        wts = scores.gather(1, idx)
        if cfg.norm_topk_prob:
            wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-20)
        return idx, wts * cfg.routed_scaling_factor

    def experts(self, z: torch.Tensor, idx: torch.Tensor, wts: torch.Tensor, li: int) -> torch.Tensor:
        """z [T, latent] through each row's routed experts, weighted and summed in expert order."""
        p, lim = f"layers.{li}.mlp.experts.", self.cfg.swiglu_limit
        G, U, D = self.w(p + "gate_proj"), self.w(p + "up_proj"), self.w(p + "down_proj")
        out = torch.zeros_like(z)
        for e in torch.unique(idx).tolist():
            rows, slot = torch.where(idx == e)
            zi = z[rows]
            g = F.linear(zi, G.weight(e, z.dtype)).clamp(max=lim)
            u = F.linear(zi, U.weight(e, z.dtype)).clamp(min=-lim, max=lim)
            y = F.linear(F.silu(g) * u, D.weight(e, z.dtype)) * wts[rows, slot, None]
            out.index_add_(0, rows, y.to(out.dtype))
        return out

    def dense(self, x: torch.Tensor, p: str) -> torch.Tensor:
        g = F.silu(self.lin(x, self.w(p + "gate_proj.weight")))
        return self.lin(g * self.lin(x, self.w(p + "up_proj.weight")), self.w(p + "down_proj.weight"))

    def mlp(self, x: torch.Tensor, li: int) -> torch.Tensor:
        p = f"layers.{li}.mlp."
        if self.cfg.is_dense_mlp(li):
            return self.rms(self.dense(x, p), self.w(p + "post_fc2_norm.weight"))
        idx, wts = self.route(x, li)
        z = self.lin(x, self.w(p + "routed_expert_down_proj.weight"))
        routed = self.lin(self.experts(z, idx, wts, li), self.w(p + "routed_expert_up_proj.weight"))
        return self.rms(routed + self.dense(x, p + "shared_experts."), self.w(p + "combine_output_norm.weight"))

    # ---------------------------------------------------------------- the model
    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.w("embed_tokens.weight")[tokens.to(self.w("embed_tokens.weight").device)].to(self.device, self.dtype)

    def layer(self, h: torch.Tensor, li: int, st: Hy4State, slots: torch.Tensor, pos: torch.Tensor,
              grams: torch.Tensor | None, live: torch.Tensor) -> torch.Tensor:
        p = f"layers.{li}."
        if li in self.hashes and grams is not None:
            h = h + self.engram(h, grams, li, live)
        x, post = self.hc(h, p + "hc_attn_layer.hc_pre.")
        y = self.attention(self.rms(x, self.w(p + "input_layernorm.weight")), li, st, slots, pos)
        h = self.inject(h, y, post)
        x, post = self.hc(h, p + "hc_mlp_layer.hc_pre.")
        y = self.mlp(self.rms(x, self.w(p + "post_attention_layernorm.weight")), li)
        return self.inject(h, y, post)

    def final(self, h: torch.Tensor) -> torch.Tensor:
        """The streams after the last layer -> the hidden state the head reads [T, H]."""
        return self.rms(self.hc_head(h), self.w("norm.weight"))

    def head(self, n: torch.Tensor) -> torch.Tensor:
        return self.lin(n.float(), self.w("lm_head.weight"))

    def forward(self, tokens: torch.Tensor, st: Hy4State, positions: torch.Tensor | None = None,
                live: torch.Tensor | None = None, embeds: torch.Tensor | None = None, layers=None,
                trace: list | None = None, hidden: bool = False) -> torch.Tensor:
        """`tokens` [T] appended to the state -> fp32 logits [T, vocab] (`hidden`: the head's input instead).
        `positions` [3, T]: the rows' rotary coordinates (default: slot + st.delta on all three);
        `live` [T]: rows that own an n-gram (default: every row but the pad token); `embeds` [T, H]: the
        rows' inputs when some are not token embeddings (images)."""
        cfg = self.cfg
        tokens = tokens.to(torch.int64).reshape(-1).cpu()
        T, L0 = int(tokens.shape[0]), st.length
        slots = torch.arange(L0, L0 + T, device=self.device)
        if positions is None:
            positions = (slots.cpu() + st.delta).unsqueeze(0).expand(3, -1)
        positions = positions.to(torch.int64).cpu()
        if live is None:
            live = tokens != cfg.engram_pad_id
            for skip in (cfg.image_token_id, cfg.video_token_id):
                if skip is not None:
                    live = live & (tokens != skip)
        grams = None
        if self.hashes:
            grams = ngram_tokens(tokens, st.tokens, slots.cpu(), cfg.engram_max_ngram_size, cfg.engram_pad_id)
        st.tokens = torch.cat((st.tokens, tokens))
        st.pos = torch.cat((st.pos, positions), dim=1)
        st.live = torch.cat((st.live, live.cpu()))

        x = self.embed(tokens) if embeds is None else embeds.to(self.device, self.dtype)
        h = x.unsqueeze(1).expand(-1, cfg.hc_mult, -1).contiguous()
        for li in (range(cfg.num_hidden_layers) if layers is None else layers):
            h = self.layer(h, li, st, slots, positions, grams, live)
            if trace is not None:
                trace.append(h)
        st.length = L0 + T
        n = self.final(h)
        return n if hidden else self.head(n)
