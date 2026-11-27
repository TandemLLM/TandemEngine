"""DFly2: the five draft layers Hy4-flash ships (`model.dfly_layers.*`), as a draft head.

A round drafts `depth` tokens after the anchor in one pass. The block is the anchor token and `depth`
mask tokens at the positions after the last committed row. The draft layers keep no cache of their own:

  context     draft layer i reads the latents the target's layer `kv_layers[i]` cached, at the rows the
              target's layer `index_layers[i]` read for the last committed row (its indexer's picks), and
              the block's own rows up to the query (causal inside the block)
  fusion      the anchor's embedding takes the target's head input of the last committed row:
              e + gate * eh_proj([norm_e(e), norm_h(h)])
  layers      single-stream pre-norm layers: gated latent attention with a sink and a sandwich norm (the
              target's sparse layers' head layout and rotary), a dense MLP
  correction  the row after the anchor predicts the first drafted token, and so on; each prediction first
              corrects its hidden state with the embedding of the token before it (the anchor, then the
              tokens drafted so far): h + mlp([norm_h(h), norm_e(e)]), then the target's head

The arithmetic follows the serving implementation Tencent published (vLLM a87e3a65,
`vllm/models/hy_v4/nvidia/dfly*.py`; AngelSpec 9ee5a9de exports the weights in this layout). The
`transformers` reference does not run these layers. A draft changes which rows are verified, never the
text: the target's verify decides.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.models.hy4_flash.config import SPARSE, AttnShape
from engine.models.hy4_flash.ref import rotate

P = "dfly_layers."


class DFlyHead:
    """The draft layers over one serving state, with the interface the shared tree drafter asks of a head
    (engine/methods/mtp_drafter.py): `depth`, `reset`, `prime`, `launch`, `read`, `views`, `resume`."""

    captures = 0

    def __init__(self, srv, W: dict | None = None, graphs: bool = False, cfg=None, banks: dict | None = None,
                 draft_head=None):
        self.srv, self.fn, self.cfg = srv, srv.fn, srv.cfg
        d = cfg or getattr(self.fn, "dfly_cfg", None) or self.cfg.dfly
        if d is None:
            raise ValueError("this checkpoint's config names no DFly draft layers")
        self.d, self.depth = d, d.draft_tokens
        self.W = W if W is not None else getattr(self.fn, "dfly_weights", None) or self.fn.W
        self.banks = banks if banks is not None else getattr(self.fn, "dfly_banks", None) or {}
        self.draft_head = draft_head if draft_head is not None else getattr(self.fn, "draft_head", None)
        fn, c = self.fn, self.cfg
        self.wk, self.wv, self.sinks, self.shape = {}, {}, {}, {}
        for i in range(d.layers):
            sink = self.w(f"{i}.self_attn.learnable_sink_param").float().to(fn.device).reshape(-1, 1)
            heads = sink.shape[0]
            a = c.sparse_attn
            w = self.w(f"{i}.self_attn.kv_b_proj.weight")
            w = (w if torch.is_tensor(w) else w.bf16_cached())[:heads * (a.nope + a.v)].to(fn.device, fn.dtype)
            if w.shape != (heads * (a.nope + a.v), c.kv_lora_rank):
                raise ValueError(f"DFly layer {i}: kv_b_proj {tuple(w.shape)} does not fit {heads} heads of the target's layout")
            w = w.view(heads, a.nope + a.v, c.kv_lora_rank)
            self.wk[i], self.wv[i] = w[:, :a.nope].contiguous(), w[:, a.nope:].transpose(1, 2).contiguous()
            self.sinks[i], self.shape[i] = sink, AttnShape(heads, a.nope, a.rope, a.v)
        self.st = type("DFlyRows", (), {"length": 0})()        # rows the drafter fed (the head keeps none)
        dev = fn.device
        self.buf = {"ctl_h": torch.zeros(2, dtype=torch.int64), "ctl": torch.zeros(2, dtype=torch.int64, device=dev),
                    "th": torch.zeros(1, c.hidden_size, dtype=fn.dtype, device=dev),
                    "tok": torch.zeros(self.depth, 2, dtype=torch.int64, device=dev),
                    "prob": torch.zeros(self.depth, 2, dtype=torch.float32, device=dev)}
        if d.attentive_correction:
            src_pos = d.attentive_source_layer if d.attentive_source_layer >= 0 else d.layers + d.attentive_source_layer
            self.hc_idx_layer = d.index_layers[src_pos]
            self.hc_kv_layer = d.kv_layers[src_pos]
            s = "hidden_correction.attention."
            self.hc_sink = self.w(s + "learnable_sink_param").float().to(dev).reshape(-1, 1)
            hc_heads = self.hc_sink.shape[0]
            a = c.sparse_attn
            w = self.w(s + "kv_b_proj.weight")
            w = (w if torch.is_tensor(w) else w.bf16_cached())[:hc_heads * (a.nope + a.v)].to(dev, fn.dtype)
            w = w.view(hc_heads, a.nope + a.v, c.kv_lora_rank)
            self.hc_wk, self.hc_wv = w[:, :a.nope].contiguous(), w[:, a.nope:].transpose(1, 2).contiguous()
            self.hc_shape = AttnShape(hc_heads, a.nope, a.rope, a.v)
            self.buf["hc_blk"] = torch.zeros(self.depth + 1, c.kv_lora_rank + a.rope, dtype=fn.dtype, device=dev)
            self._hc_back_full = torch.arange(self.depth, -1, -1, device=dev)
        self._mask = torch.full((self.depth,), d.mask_token_id, dtype=torch.int64, device=dev)
        self._tril = torch.ones(self.depth + 1, self.depth + 1, dtype=torch.bool, device=dev).tril()
        self.graphs = graphs and dev.type == "cuda"
        self.graph = None
        self._ready = False
        self.stats = {"rounds": 0, "stale": 0}

    # --- the head interface ------------------------------------------------------------------------
    def reset(self) -> None:
        self.st.length = 0
        self._ready = False

    def prime(self, rows, nexts) -> None:
        self.st.length += len(nexts)

    def resume(self, n: int) -> None:
        self.st.length = int(n)

    def capture_all(self) -> None:
        pass

    def views(self) -> list:
        return []

    def stage(self, anchor: int, target_hidden: torch.Tensor, st) -> None:
        b = self.buf
        b["ctl_h"][0], b["ctl_h"][1] = int(anchor), st.length
        b["ctl"].copy_(b["ctl_h"])
        b["th"].copy_(target_hidden.reshape(1, -1))

    def launch(self, rows: torch.Tensor, nexts: list[int]) -> None:
        """`rows[-1]`: the last committed row's head input; `nexts[-1]`: the anchor token."""
        self.st.length += len(nexts)
        st = self.srv.st
        self._ready = False
        if st.picks_at != st.length or len(st.pick_layers) != len(self.fn.pick_layers):
            self.stats["stale"] += 1                         # after a restore: this round drafts nothing
            return
        self.stage(int(nexts[-1]), rows[-1:], st)
        if self.graphs:
            if self.graph is None:
                self.graph = self._capture(st)
                self.captures += 1
            self.graph.replay()
        else:
            self.run(st)
        self._ready = True
        self.stats["rounds"] += 1

    def read(self) -> tuple[list, list]:
        if not self._ready:
            return [], []
        self._ready = False
        return self.buf["tok"].tolist(), self.buf["prob"].tolist()

    def _capture(self, st):
        from engine import capture_lock
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.run(st)
        torch.cuda.current_stream().wait_stream(s)
        from engine.models.hy4_flash.engine import capture_args
        g = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(g, **capture_args(st)):     # ENG-485: the slot's set
            self.run(st)
        torch.cuda.synchronize()
        return g

    # --- the arithmetic ----------------------------------------------------------------------------
    def _key(self, name: str) -> str:
        if self.d.prefix == P:
            return P + name
        if name == "final_norm.weight":
            return self.d.norm_name
        return self.d.prefix + name if name[0].isdigit() else name

    def w(self, name: str):
        return self.W[self._key(name)]

    def fuse(self, e: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        fn = self.fn
        x = torch.cat((fn.rms(e, self.w("mtp_fusion.enorm.weight")), fn.rms(h.to(e.dtype), self.w("mtp_fusion.hnorm.weight"))), dim=-1)
        return e + self.w("mtp_fusion.gate").to(e.device, e.dtype) * fn.lin(x, self.w("mtp_fusion.eh_proj.weight"))

    def _corr_input(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        fn = self.fn
        return torch.cat((fn.rms(h, self.w("hidden_correction.hidden_norm.weight")),
                          fn.rms(e, self.w("hidden_correction.embed_norm.weight"))), dim=-1)

    def correct(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        fn = self.fn
        x = self._corr_input(h, e)
        g = F.silu(fn.lin(x, self.w("hidden_correction.gate_proj.weight")))
        return h + fn.lin(g * fn.lin(x, self.w("hidden_correction.up_proj.weight")), self.w("hidden_correction.down_proj.weight"))

    def _write_hc_key(self, slot: int, inp: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
        fn, R = self.fn, self.cfg.kv_lora_rank
        s = "hidden_correction.attention."
        ckv = fn.lin(inp, self.w(s + "kv_in.weight"))
        blk_k = torch.cat((fn.rms(ckv[:, :R], self.w(s + "kv_a_layernorm.weight")), rotate(ckv[:, R:], cos, sin)), dim=-1)
        self.buf["hc_blk"][slot:slot + 1].copy_(blk_k)

    def correct_attentive(self, slot: int, h: torch.Tensor, e: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                          ctx: torch.Tensor, ctx_ok: torch.Tensor) -> torch.Tensor:
        fn, a, R = self.fn, self.hc_shape, self.cfg.kv_lora_rank
        s = "hidden_correction.attention."
        inp = self._corr_input(h, e)
        self._write_hc_key(slot, inp, cos, sin)
        qres = fn.rms(fn.lin(inp, self.w(s + "q_in.weight")), self.w(s + "q_a_layernorm.weight"))
        q = fn.lin(qres, self.w(s + "q_b_proj.weight")).view(1, a.heads, a.qk)
        q = torch.cat((q[..., :a.nope], rotate(q[..., a.nope:], cos[:, None], sin[:, None])), dim=-1)
        ctx_read = fn.attend_chunk_latent(q, ctx, ctx_ok.unsqueeze(0), (self.hc_wk, self.hc_wv, self.hc_sink, a),
                                            return_fp32=True)
        qa = torch.einsum("thd,hdr->thr", q[..., :a.nope], self.hc_wk).float()
        blk_k = self.buf["hc_blk"][:slot + 1].float()
        back = self._hc_back_full[self.depth - slot:]
        rel = self.w(s + "block_rel_bias")[:, back]
        logits = (qa @ blk_k[:, :R].T + q[..., a.nope:].float() @ blk_k[:, R:].T) * (a.qk ** -0.5) + rel.unsqueeze(0)
        total = torch.logaddexp(torch.logsumexp(logits, dim=-1, keepdim=True), self.w(s + "block_sink")[None, :, None])
        blk_read = torch.exp(logits - total) @ blk_k[:, :R]
        gate = torch.tanh(fn.lin(inp, self.w(s + "block_gate.weight")).float() + self.w(s + "block_gate_bias")[:, slot][None, :])
        mix = (ctx_read + blk_read * gate.unsqueeze(-1)).to(fn.dtype)
        attn_out = fn.lin(torch.einsum("thr,hrv->thv", mix, self.hc_wv).reshape(1, -1), self.w(s + "o_proj.weight"))
        g = F.silu(fn.lin(inp, self.w("hidden_correction.gate_proj.weight")))
        mlp_out = fn.lin(g * fn.lin(inp, self.w("hidden_correction.up_proj.weight")), self.w("hidden_correction.down_proj.weight"))
        return h + attn_out + mlp_out

    def layer(self, i: int, h: torch.Tensor, ctx: torch.Tensor, ctx_ok: torch.Tensor, cos, sin) -> torch.Tensor:
        """h [Q, H] the block's rows; ctx [n, R + rope] the target's latents this layer reads, ctx_ok [n]."""
        fn, a, R = self.fn, self.shape[i], self.cfg.kv_lora_rank
        Q, s = h.shape[0], f"{i}.self_attn."
        x = fn.rms(h, self.w(f"{i}.input_layernorm.weight"))
        gate = fn.lin(x, self.w(s + "linear_gate.weight"))
        qres = fn.rms(fn.lin(x, self.w(s + "q_a_proj.weight")), self.w(s + "q_a_layernorm.weight"))
        q = fn.lin(qres, self.w(s + "q_b_proj.weight")).view(Q, a.heads, a.qk)
        q = torch.cat((q[..., :a.nope], rotate(q[..., a.nope:], cos[:, None], sin[:, None])), dim=-1)
        ckv = fn.lin(x, self.w(s + "kv_a_proj_with_mqa.weight"))
        own = torch.cat((fn.rms(ckv[:, :R], self.w(s + "kv_a_layernorm.weight")), rotate(ckv[:, R:], cos, sin)), dim=-1)
        allowed = torch.cat((ctx_ok.unsqueeze(0).expand(Q, -1), self._tril), dim=-1)
        y = fn.attend_chunk(q, torch.cat((ctx, own)), allowed, -1, w=(self.wk[i], self.wv[i], self.sinks[i], a))
        y = (y * torch.sigmoid(gate).view(Q, a.heads, a.v)).reshape(Q, a.heads * a.v)
        h = h + fn.rms(fn.lin(y, self.w(s + "o_proj.weight")), self.w(s + "post_proj_norm.weight"))
        x = fn.rms(h, self.w(f"{i}.post_attention_layernorm.weight"))
        if self.d.mlp_type == "dense":
            g = F.silu(fn.lin(x, self.w(f"{i}.mlp.gate_proj.weight")))
            return h + fn.lin(g * fn.lin(x, self.w(f"{i}.mlp.up_proj.weight")), self.w(f"{i}.mlp.down_proj.weight"))
        m = f"{i}.mlp."
        bias = self.W.get(self._key(m + "expert_bias"))
        if bias is None:
            bias = self.w(m + "gate.e_score_correction_bias")
        idx, wts = fn.route_weights(x, self.w(m + "gate.weight"), bias, self.d.num_experts_per_tok,
                                    self.d.routed_scaling_factor, self.d.norm_topk_prob,
                                    fused=fn.kernels and fn.dtype == torch.bfloat16)
        z = fn.lin(x, self.w(m + "routed_expert_down_proj.weight"))
        bank = self.banks.get(i)
        experts = None if bank is not None else (self.w(m + "experts.gate_proj"),
                                                 self.w(m + "experts.up_proj"),
                                                 self.w(m + "experts.down_proj"))
        yd = fn.expert_pairs_bank(z, idx, bank=bank, experts=experts, lim=self.d.swiglu_limit, grouped=False)
        routed = fn.lin(fn.combine(yd, idx, wts), self.w(m + "routed_expert_up_proj.weight"))
        sp = m + "shared_experts."
        g = F.silu(fn.lin(x, self.w(sp + "gate_proj.weight")))
        shared = fn.lin(g * fn.lin(x, self.w(sp + "up_proj.weight")), self.w(sp + "down_proj.weight"))
        return h + fn.rms(routed + shared, self.w(m + "combine_output_norm.weight"))

    def _hidden_and_rope(self, st) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The staged block's rows after the draft layers and their norm [depth + 1, H], plus block RoPE."""
        fn, d, b = self.fn, self.d, self.buf
        Q, Lt = self.depth + 1, b["ctl"][1]
        e = fn.embed(torch.cat((b["ctl"][:1], self._mask)))
        h = torch.cat((self.fuse(e[:1], b["th"]), e[1:]))
        cos, sin = fn.rope(fn.pos3(Lt + fn.ar(Q) + st.delta), SPARSE)
        for i in range(d.layers):
            pi = st.pick_layers.index(d.index_layers[i])
            pq = st.pick_q[pi]
            ok = st.pick_ok[pi] & (pq < Lt)
            lat = st.lat[d.kv_layers[i]]
            h = self.layer(i, h, lat[pq.clamp(0, lat.shape[0] - 1)], ok, cos, sin)
        return fn.rms(h, self.w("final_norm.weight")), cos, sin

    def _hidden(self, st) -> torch.Tensor:
        return self._hidden_and_rope(st)[0]

    @torch.no_grad()
    def run(self, st) -> None:
        """The staged round on the device: per drafted position its two best tokens and their probabilities,
        along the best tokens' chain, into the head's buffers. What the draft graph captures."""
        fn, b = self.fn, self.buf
        h, cos, sin = self._hidden_and_rope(st)
        prev = b["ctl"][:1]
        if self.d.attentive_correction:
            Lt = b["ctl"][1]
            pi = st.pick_layers.index(self.hc_idx_layer)
            pq = st.pick_q[pi]
            ok_hc = st.pick_ok[pi] & (pq < Lt)
            lat_hc = st.lat[self.hc_kv_layer]
            ctx_hc = lat_hc[pq.clamp(0, lat_hc.shape[0] - 1)]
            self._write_hc_key(0, self._corr_input(h[:1], fn.embed(prev)), cos[:1], sin[:1])
            for j in range(self.depth):
                slot = j + 1
                c = self.correct_attentive(slot, h[slot:slot + 1], fn.embed(prev),
                                           cos[slot:slot + 1], sin[slot:slot + 1], ctx_hc, ok_hc)
                top = torch.topk(torch.softmax(fn.head(c, w=self.draft_head)[0].float(), dim=-1), 2)
                b["tok"][j].copy_(top.indices)
                b["prob"][j].copy_(top.values)
                prev = top.indices[:1]
            return
        for j in range(self.depth):
            c = self.correct(h[j + 1:j + 2], fn.embed(prev))
            top = torch.topk(torch.softmax(fn.head(c, w=self.draft_head)[0].float(), dim=-1), 2)
            b["tok"][j].copy_(top.indices)
            b["prob"][j].copy_(top.values)
            prev = top.indices[:1]

    @torch.no_grad()
    def hidden(self, anchor: int, target_hidden: torch.Tensor, st) -> torch.Tensor:
        self.stage(anchor, target_hidden, st)
        return self._hidden(st)

    @torch.no_grad()
    def draft(self, anchor: int, target_hidden: torch.Tensor, st) -> tuple[list, list]:
        """Per drafted position its two best tokens and their probabilities (the plain pass, no graph)."""
        self.stage(anchor, target_hidden, st)
        self.run(st)
        return self.buf["tok"].tolist(), self.buf["prob"].tolist()
