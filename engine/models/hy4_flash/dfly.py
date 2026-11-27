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
#: SPD-349: packed-round graphs kept (all flushed together with a fresh pool past this, SPD-223)
PACK_GRAPHS_MAX = 48


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
        self.captures = 0
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

    def pre_commit(self, path: list[int]) -> None:
        """Under QSE_RUNNER_PIPE=1, commit accepted rows before drafting the next round so DFly picks/lat stay exact."""
        if hasattr(self.srv, "pre_commit_tree"):
            self.srv.pre_commit_tree(path)

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
        from engine.methods import draft_meet
        self.st.length += len(nexts)
        st = self.srv.st
        self._ready = False
        if st.picks_at != st.length or len(st.pick_layers) != len(self.fn.pick_layers):
            self.stats["stale"] += 1                         # after a restore: this round drafts nothing
            return
        self.stage(int(nexts[-1]), rows[-1:], st)
        if draft_meet.MEET is not None and draft_meet.MEET(self):
            self._ready = True
            self.stats["rounds"] += 1
            return
        if self.graphs:
            if self.graph is None:
                self.graph = self._capture(st)
                self.captures += 1
            self.graph.replay()
        else:
            self.run(st)
        self._ready = True
        self.stats["rounds"] += 1

    @staticmethod
    @torch.no_grad()
    def run_packed(heads: list) -> None:
        """SPD-343 (`MTP_PACK`): draft all active lanes together reading DFly weights, MoE experts, and `head` ONCE."""
        if not heads:
            return
        if heads[0]._fused():
            if len(heads) > 1 and DFlyHead._pack_graph(heads):
                return
            DFlyHead._run_lanes([(h, h.srv.st) for h in heads])
            return
        if len(heads) == 1:
            heads[0].run(heads[0].srv.st)
            return
        h0 = heads[0]
        fn, d, depth, Q, B = h0.fn, h0.d, h0.depth, h0.depth + 1, len(heads)
        sts = [h.srv.st for h in heads]
        anchors = torch.cat([h.buf["ctl"][:1] for h in heads])
        Lts = [h.buf["ctl"][1] for h in heads]
        th_cat = torch.cat([h.buf["th"] for h in heads])
        e_anc = fn.embed(anchors)
        h_anc = h0.fuse(e_anc, th_cat)
        e_mask = fn.embed(h0._mask)
        h = torch.cat([torch.cat((h_anc[b:b + 1], e_mask)) for b in range(B)], dim=0)
        cos = torch.cat([fn.rope(fn.pos3(Lts[b] + fn.ar(Q) + sts[b].delta), SPARSE)[0] for b in range(B)], dim=0)
        sin = torch.cat([fn.rope(fn.pos3(Lts[b] + fn.ar(Q) + sts[b].delta), SPARSE)[1] for b in range(B)], dim=0)
        R = fn.cfg.kv_lora_rank
        fused_moe = fn.kernels and fn.dtype == torch.bfloat16

        for i in range(d.layers):
            a, s = h0.shape[i], f"{i}.self_attn."
            x = fn.rms(h, h0.w(f"{i}.input_layernorm.weight"))
            gate = fn.lin(x, h0.w(s + "linear_gate.weight"))
            qres = fn.rms(fn.lin(x, h0.w(s + "q_a_proj.weight")), h0.w(s + "q_a_layernorm.weight"))
            q = fn.lin(qres, h0.w(s + "q_b_proj.weight")).view(B * Q, a.heads, a.qk)
            q = torch.cat((q[..., :a.nope], rotate(q[..., a.nope:], cos[:, None], sin[:, None])), dim=-1)
            ckv = fn.lin(x, h0.w(s + "kv_a_proj_with_mqa.weight"))
            own = torch.cat((fn.rms(ckv[:, :R], h0.w(s + "kv_a_layernorm.weight")), rotate(ckv[:, R:], cos, sin)), dim=-1)
            ys = []
            for b, st in enumerate(sts):
                sl = slice(b * Q, (b + 1) * Q)
                pi = st.pick_layers.index(d.index_layers[i])
                pq = st.pick_q[pi]
                ok = st.pick_ok[pi] & (pq < Lts[b])
                lat = st.lat[d.kv_layers[i]]
                ctx = lat[pq.clamp(0, lat.shape[0] - 1)]
                allowed = torch.cat((ok.unsqueeze(0).expand(Q, -1), h0._tril), dim=-1)
                ys.append(fn.attend_chunk(q[sl], torch.cat((ctx, own[sl])), allowed, -1,
                                          w=(h0.wk[i], h0.wv[i], h0.sinks[i], a)))
            y = (torch.cat(ys, dim=0) * torch.sigmoid(gate).view(B * Q, a.heads, a.v)).reshape(B * Q, a.heads * a.v)
            h = h + fn.rms(fn.lin(y, h0.w(s + "o_proj.weight")), h0.w(s + "post_proj_norm.weight"))
            x = fn.rms(h, h0.w(f"{i}.post_attention_layernorm.weight"))
            if d.mlp_type == "dense":
                g = F.silu(fn.lin(x, h0.w(f"{i}.mlp.gate_proj.weight")))
                h = h + fn.lin(g * fn.lin(x, h0.w(f"{i}.mlp.up_proj.weight")), h0.w(f"{i}.mlp.down_proj.weight"))
            else:
                m = f"{i}.mlp."
                bias = h0.W.get(h0._key(m + "expert_bias"))
                if bias is None:
                    bias = h0.w(m + "gate.e_score_correction_bias")
                idx, wts = fn.route_weights(x, h0.w(m + "gate.weight"), bias, d.num_experts_per_tok,
                                            d.routed_scaling_factor, d.norm_topk_prob, fused=fused_moe)
                z = fn.lin(x, h0.w(m + "routed_expert_down_proj.weight"))
                bank = h0.banks.get(i)
                experts = None if bank is not None else (h0.w(m + "experts.gate_proj"),
                                                         h0.w(m + "experts.up_proj"),
                                                         h0.w(m + "experts.down_proj"))
                yd = fn.expert_pairs_bank(z, idx, bank=bank, experts=experts, lim=d.swiglu_limit, grouped=True)
                routed = fn.lin(fn.combine(yd, idx, wts), h0.w(m + "routed_expert_up_proj.weight"))
                sp = m + "shared_experts."
                g = F.silu(fn.lin(x, h0.w(sp + "gate_proj.weight")))
                shared = fn.lin(g * fn.lin(x, h0.w(sp + "up_proj.weight")), h0.w(sp + "down_proj.weight"))
                h = h + fn.rms(routed + shared, h0.w(m + "combine_output_norm.weight"))

        h = fn.rms(h, h0.w("final_norm.weight")).view(B, Q, -1)
        prev = anchors
        if d.attentive_correction:
            a, sc_s = h0.hc_shape, "hidden_correction.attention."
            cos_q = cos.view(B, Q, -1)
            sin_q = sin.view(B, Q, -1)
            ctxs, oks = [], []
            for st, Lt in zip(sts, Lts):
                pi = st.pick_layers.index(h0.hc_idx_layer)
                pq = st.pick_q[pi]
                ok_b = st.pick_ok[pi] & (pq < Lt)
                lat_b = st.lat[h0.hc_kv_layer]
                ctxs.append(lat_b[pq.clamp(0, lat_b.shape[0] - 1)])
                oks.append(ok_b)
            inp0 = h0._corr_input(h[:, 0], fn.embed(prev))
            ckv0 = fn.lin(inp0, h0.w(sc_s + "kv_in.weight"))
            k0 = torch.cat((fn.rms(ckv0[:, :R], h0.w(sc_s + "kv_a_layernorm.weight")),
                            rotate(ckv0[:, R:], cos_q[:, 0], sin_q[:, 0])), dim=-1)
            hc_blk = torch.zeros(B, Q, R + a.rope, dtype=fn.dtype, device=h.device)
            hc_blk[:, 0].copy_(k0)
            for j in range(depth):
                slot = j + 1
                inp = h0._corr_input(h[:, slot], fn.embed(prev))
                ckv = fn.lin(inp, h0.w(sc_s + "kv_in.weight"))
                ks = torch.cat((fn.rms(ckv[:, :R], h0.w(sc_s + "kv_a_layernorm.weight")),
                                rotate(ckv[:, R:], cos_q[:, slot], sin_q[:, slot])), dim=-1)
                hc_blk[:, slot].copy_(ks)
                qres = fn.rms(fn.lin(inp, h0.w(sc_s + "q_in.weight")), h0.w(sc_s + "q_a_layernorm.weight"))
                qb = fn.lin(qres, h0.w(sc_s + "q_b_proj.weight")).view(B, a.heads, a.qk)
                qb = torch.cat((qb[..., :a.nope], rotate(qb[..., a.nope:], cos_q[:, slot, None], sin_q[:, slot, None])), dim=-1)
                ctx_read = torch.cat([fn.attend_chunk_latent(qb[b:b + 1], ctxs[b], oks[b].unsqueeze(0),
                                                             (h0.hc_wk, h0.hc_wv, h0.hc_sink, a), return_fp32=True)
                                      for b in range(B)], dim=0)
                qa = torch.einsum("bhd,hdr->bhr", qb[..., :a.nope], h0.hc_wk).float()
                kb = hc_blk[:, :slot + 1].float()
                back = h0._hc_back_full[depth - slot:]
                rel = h0.w(sc_s + "block_rel_bias")[:, back].unsqueeze(0)
                logits_blk = (torch.bmm(qa, kb[:, :, :R].transpose(1, 2))
                              + torch.bmm(qb[..., a.nope:].float(), kb[:, :, R:].transpose(1, 2))) * (a.qk ** -0.5) + rel
                tot = torch.logaddexp(torch.logsumexp(logits_blk, dim=-1, keepdim=True),
                                      h0.w(sc_s + "block_sink")[None, :, None])
                blk_read = torch.bmm(torch.exp(logits_blk - tot), kb[:, :, :R])
                gate_b = torch.tanh(fn.lin(inp, h0.w(sc_s + "block_gate.weight")).float()
                                    + h0.w(sc_s + "block_gate_bias")[:, slot][None, :])
                mix = (ctx_read + blk_read * gate_b.unsqueeze(-1)).to(fn.dtype)
                attn_out = fn.lin(torch.einsum("bhr,hrv->bhv", mix, h0.hc_wv).reshape(B, -1),
                                  h0.w(sc_s + "o_proj.weight"))
                gm = F.silu(fn.lin(inp, h0.w("hidden_correction.gate_proj.weight")))
                mlp_out = fn.lin(gm * fn.lin(inp, h0.w("hidden_correction.up_proj.weight")),
                                 h0.w("hidden_correction.down_proj.weight"))
                c = h[:, slot] + attn_out + mlp_out
                top = torch.topk(torch.softmax(fn.head(c, w=h0.draft_head).float(), dim=-1), 2, dim=-1)
                for b, hd in enumerate(heads):
                    hd.buf["tok"][j].copy_(top.indices[b])
                    hd.buf["prob"][j].copy_(top.values[b])
                prev = top.indices[:, 0]
            return

        for j in range(depth):
            c = h0.correct(h[:, j + 1], fn.embed(prev))
            top = torch.topk(torch.softmax(fn.head(c, w=h0.draft_head).float(), dim=-1), 2, dim=-1)
            for b, hd in enumerate(heads):
                hd.buf["tok"][j].copy_(top.indices[b])
                hd.buf["prob"][j].copy_(top.values[b])
            prev = top.indices[:, 0]

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

    # --- the arithmetic on the row kernels (SPD-345) -----------------------------------------------------------
    @staticmethod
    def _pack_graph(heads: list) -> bool:
        """SPD-349 (HY4_PACK_GRAPH): a packed round of these heads from its CUDA graph: every tensor `_run_lanes`
        reads or writes is a head's or its lane's own and lives as long as they do (the staged anchor, length and head
        input, the lane's picks and latents, the head's token and probability rows), so one capture a set of heads
        replays every later round of that set. Captured on the engine's leased packed-draft stream (`capture_lock`) and
        shared pool at a set's first round the server allows (`pack_capture_ok`); past `PACK_GRAPHS_MAX`, all kept
        graphs are flushed together with a fresh pool before capturing (SPD-223). False: run eager."""
        fn = heads[0].fn
        if not (getattr(fn, "pack_graph", False) and fn.device.type == "cuda" and all(h.graphs for h in heads)):
            return False
        pg = fn.__dict__.setdefault("_pack_graphs", {"graphs": {}, "stream": None, "pool": None,
                                                      "stats": {"captures": 0, "replays": 0, "flushes": 0}})
        key = tuple(id(h) for h in heads)
        g = pg["graphs"].get(key)
        if g is None:
            if not getattr(fn, "pack_capture_ok", getattr(fn, "joint_capture_ok", True)):
                return False
            from engine import capture_lock
            if pg["stream"] is None:
                pg["stream"] = capture_lock.lease_stream()
            if len(pg["graphs"]) >= PACK_GRAPHS_MAX:
                for old in pg["graphs"].values():
                    old.reset()
                pg["graphs"].clear()
                pg["pool"] = torch.cuda.graph_pool_handle()
                pg["stats"]["flushes"] = pg["stats"].get("flushes", 0) + 1
                capture_lock.graphs_reset()
            elif pg["pool"] is None:
                pg["pool"] = torch.cuda.graph_pool_handle()
            lanes = [(h, h.srv.st) for h in heads]
            s = pg["stream"]
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    DFlyHead._run_lanes(lanes)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with capture_lock.LOCK, torch.cuda.graph(g, pool=pg["pool"], stream=s):
                DFlyHead._run_lanes(lanes)
            torch.cuda.synchronize()
            pg["stats"]["captures"] += 1
            heads[0].captures += 1
            pg["graphs"][key] = g
        g.replay()
        pg["stats"]["replays"] += 1
        return True

    def _fused(self) -> bool:
        """HY4_DFLY_FUSED: the round on the row kernels (`_run_lanes`), one lane or many with the same bits a lane."""
        return bool(getattr(self.fn, "dfly_fused", False)) and self.fn.fused

    def _wb(self, name: str) -> torch.Tensor:
        """A plain weight in the compute dtype, contiguous (made once)."""
        cache = self.__dict__.setdefault("_wbc", {})
        t = cache.get(name)
        if t is None:
            t = cache[name] = self.w(name).to(self.fn.device, self.fn.dtype).contiguous()
        return t

    def _norm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        from engine.models.hy4_flash import kernels as K_
        return K_.norm(x.contiguous(), self._wb(name), self.cfg.rms_norm_eps)

    def _lin(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """A projection of rows x whose bits a row never depend on the others: an EXL3 one on `dense_rows` at every row
        count (`lin_any`), a plain weight on `mm` with the rows on its tile axis (the library GEMM picks its algorithm
        by the row count), rounded to the compute dtype as the library's bf16 output is."""
        w = self.w(name)
        fn = self.fn
        if hasattr(w, "bf16_cached"):
            return fn.lin_any(x.contiguous(), w)
        n = fn.cuts.get(id(w))
        if x.shape[-1] < w.shape[1]:
            x = F.pad(x, (0, w.shape[1] - x.shape[-1]))
        y = fn._rows_mm(x.contiguous(), w).to(fn.dtype)
        return y if n is None else y[:, :n]

    def _key_rows(self, x: torch.Tensor, lin: str, ln: str, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """A latent key of rows x: the projection, the norm of its latent half in place, the rotary on the rest."""
        from engine.models.hy4_flash import kernels as K_
        R = self.cfg.kv_lora_rank
        ckv = self._lin(x, lin).contiguous()
        K_.norm(ckv[:, :R], self._wb(ln), self.cfg.rms_norm_eps, out=ckv[:, :R])
        K_.rot(ckv.view(ckv.shape[0], 1, ckv.shape[1]), cos, sin, R)
        return ckv

    def _ctx(self, st, i_layer: int, kv_layer: int, Lt: torch.Tensor) -> tuple:
        pi = st.pick_layers.index(i_layer)
        pq = st.pick_q[pi]
        lat = st.lat[kv_layer]
        return lat[pq.clamp(0, lat.shape[0] - 1)], st.pick_ok[pi] & (pq < Lt)

    def _mlp_f(self, i: int, x: torch.Tensor) -> torch.Tensor:
        fn, m = self.fn, f"{i}.mlp."
        if self.d.mlp_type == "dense":
            g = F.silu(self._lin(x, m + "gate_proj.weight"))
            return self._lin(g * self._lin(x, m + "up_proj.weight"), m + "down_proj.weight")
        bias = self.W.get(self._key(m + "expert_bias"))
        if bias is None:
            bias = self.w(m + "gate.e_score_correction_bias")
        idx, wts = fn.route_weights(x, self.w(m + "gate.weight"), bias, self.d.num_experts_per_tok,
                                    self.d.routed_scaling_factor, self.d.norm_topk_prob, fused=True)
        z = self._lin(x, m + "routed_expert_down_proj.weight")
        bank = self.banks.get(i)
        experts = None if bank is not None else (self.w(m + "experts.gate_proj"), self.w(m + "experts.up_proj"),
                                                 self.w(m + "experts.down_proj"))
        yd = fn.expert_pairs_bank(z, idx, bank=bank, experts=experts, lim=self.d.swiglu_limit, grouped=True)
        routed = self._lin(fn.combine(yd, idx, wts), m + "routed_expert_up_proj.weight")
        sp = m + "shared_experts."
        g = F.silu(self._lin(x, sp + "gate_proj.weight"))
        shared = self._lin(g * self._lin(x, sp + "up_proj.weight"), sp + "down_proj.weight")
        return self._norm(routed + shared, m + "combine_output_norm.weight")

    @staticmethod
    @torch.no_grad()
    def _run_lanes(lanes: list) -> None:
        """SPD-345: the round of every lane [(head, state)] in one pass on the row kernels: the norms on `K_.norm`,
        the rotary in place, the projections on `lin1` (`dense_rows` in 16-row slices), the experts grouped, the
        attention on `attend_lanes` (`mm` and the fused softmax), the head on `dense_rows`. Every one of those gives a
        row the bits it has alone, so a lane's tokens and probabilities are the same in a packed round of any width
        as in its lone round (`run`, a graph's capture); the torch reductions left (softmax, top-2 over the vocabulary,
        the block's log-sum-exp as element-wise steps) run a row at a time in a launch whose shape is the row's."""
        from engine.models.hy4_flash import kernels as K_
        h0 = lanes[0][0]
        fn, d, c = h0.fn, h0.d, h0.cfg
        depth, Q, B = h0.depth, h0.depth + 1, len(lanes)
        R, dt = c.kv_lora_rank, fn.dtype
        ctl = torch.stack([hd.buf["ctl"] for hd, _ in lanes])                  # [B, 2]: anchor, length
        anchors, Lt = ctl[:, 0], ctl[:, 1]
        th = torch.cat([hd.buf["th"] for hd, _ in lanes])
        h = torch.cat((h0._fuse_f(fn.embed(anchors), th).unsqueeze(1), fn.embed(h0._mask).unsqueeze(0).expand(B, -1, -1)), dim=1)
        h = h.reshape(B * Q, -1).contiguous()
        pos = torch.cat([ctl[b, 1] + fn.ar(Q) + st.delta for b, (_, st) in enumerate(lanes)])
        cos, sin = fn.rope(fn.pos3(pos), SPARSE)
        cos, sin = cos.contiguous(), sin.contiguous()
        for i in range(d.layers):
            a, s = h0.shape[i], f"{i}.self_attn."
            ctx = [h0._ctx(st, d.index_layers[i], d.kv_layers[i], ctl[b, 1]) for b, (_, st) in enumerate(lanes)]
            x = h0._norm(h, f"{i}.input_layernorm.weight")
            gate = h0._lin(x, s + "linear_gate.weight")
            qres = h0._norm(h0._lin(x, s + "q_a_proj.weight"), s + "q_a_layernorm.weight")
            q = h0._lin(qres, s + "q_b_proj.weight").contiguous().view(B * Q, a.heads, a.qk)
            K_.rot(q, cos, sin, a.nope)
            own = h0._key_rows(x, s + "kv_a_proj_with_mqa.weight", s + "kv_a_layernorm.weight", cos, sin)
            n = ctx[0][0].shape[0]
            Kl = torch.cat((torch.stack([cx[0] for cx in ctx]), own.view(B, Q, -1)), dim=1)
            ok = torch.stack([cx[1] for cx in ctx])
            valid = torch.cat((ok[:, None, :].expand(B, Q, n), h0._tril[None].expand(B, Q, Q)), dim=2).reshape(B * Q, n + Q).contiguous()
            y = fn.attend_lanes(q, Kl, valid, h0.wk[i], h0.wv[i], h0.sinks[i], a, B, Q).reshape(B * Q, -1)
            y = K_.mul_sigmoid(y, gate) if fn._ms else y * torch.sigmoid(gate)
            h = h + h0._norm(h0._lin(y, s + "o_proj.weight"), s + "post_proj_norm.weight")
            x = h0._norm(h, f"{i}.post_attention_layernorm.weight")
            h = h + h0._mlp_f(i, x)
        hN = h0._norm(h, "final_norm.weight").view(B, Q, -1)
        prev = anchors
        toks, probs = [], []
        if d.attentive_correction:
            a, sc = h0.hc_shape, "hidden_correction.attention."
            cq, sq = cos.view(B, Q, -1), sin.view(B, Q, -1)
            hx = [h0._ctx(st, h0.hc_idx_layer, h0.hc_kv_layer, ctl[b, 1]) for b, (_, st) in enumerate(lanes)]
            ctx_hc, ok_hc = torch.stack([x_[0] for x_ in hx]), torch.stack([x_[1] for x_ in hx]).contiguous()
            blk = torch.empty(B, Q, R + a.rope, dtype=dt, device=h.device)
            blk[:, 0] = h0._key_rows(h0._corr_f(hN[:, 0], fn.embed(prev)), sc + "kv_in.weight", sc + "kv_a_layernorm.weight",
                                     cq[:, 0].contiguous(), sq[:, 0].contiguous())
            rel_all = h0.w(sc + "block_rel_bias").float()
            sink_b = h0.w(sc + "block_sink").float()
            gbias = h0.w(sc + "block_gate_bias").float()
            for j in range(depth):
                slot = j + 1
                cs, ss = cq[:, slot].contiguous(), sq[:, slot].contiguous()
                inp = h0._corr_f(hN[:, slot], fn.embed(prev))
                blk[:, slot] = h0._key_rows(inp, sc + "kv_in.weight", sc + "kv_a_layernorm.weight", cs, ss)
                qres = h0._norm(h0._lin(inp, sc + "q_in.weight"), sc + "q_a_layernorm.weight")
                qb = h0._lin(qres, sc + "q_b_proj.weight").contiguous().view(B, a.heads, a.qk)
                K_.rot(qb, cs, ss, a.nope)
                ctx_read = fn.attend_lanes(qb, ctx_hc, ok_hc, h0.hc_wk, None, h0.hc_sink, a, B, 1, latent=True)
                qa = fn.mm(qb[..., :a.nope].transpose(0, 1), h0.hc_wk).to(dt).transpose(0, 1).contiguous()     # [B, heads, R]
                kb = blk[:, :slot + 1]
                back = h0._hc_back_full[depth - slot:]
                lg = ((fn.mm(qa, kb[..., :R].transpose(1, 2)) + fn.mm(qb[..., a.nope:], kb[..., R:].transpose(1, 2))) * (a.qk ** -0.5)
                      + rel_all[:, back].unsqueeze(0))                                                            # [B, heads, slot + 1]
                # log(sum exp) over the block's keys and the sink, element-wise a key (no reduction kernel)
                mx = lg[..., 0]
                for k in range(1, slot + 1):
                    mx = torch.maximum(mx, lg[..., k])
                se = torch.exp(lg[..., 0] - mx)
                for k in range(1, slot + 1):
                    se = se + torch.exp(lg[..., k] - mx)
                tot = torch.logaddexp(mx + torch.log(se), sink_b[None, :])
                blk_read = fn.mm(torch.exp(lg - tot.unsqueeze(-1)), kb[..., :R].float())                         # [B, heads, R]
                gate_b = torch.tanh(h0._lin(inp, sc + "block_gate.weight").float() + gbias[:, slot][None, :])
                mix = (ctx_read + blk_read * gate_b.unsqueeze(-1)).to(dt)
                attn_out = h0._lin(fn.mm(mix.transpose(0, 1), h0.hc_wv).transpose(0, 1).to(dt).reshape(B, -1), sc + "o_proj.weight")
                gm = F.silu(h0._lin(inp, "hidden_correction.gate_proj.weight"))
                mlp_out = h0._lin(gm * h0._lin(inp, "hidden_correction.up_proj.weight"), "hidden_correction.down_proj.weight")
                cc = hN[:, slot] + attn_out + mlp_out
                top = torch.topk(torch.softmax(h0._head_f(cc), dim=-1), 2, dim=-1)
                toks.append(top.indices)
                probs.append(top.values)
                prev = top.indices[:, 0]
        else:
            for j in range(depth):
                x = h0._corr_f(hN[:, j + 1], fn.embed(prev))
                g = F.silu(h0._lin(x, "hidden_correction.gate_proj.weight"))
                cc = hN[:, j + 1] + h0._lin(g * h0._lin(x, "hidden_correction.up_proj.weight"), "hidden_correction.down_proj.weight")
                top = torch.topk(torch.softmax(h0._head_f(cc), dim=-1), 2, dim=-1)
                toks.append(top.indices)
                probs.append(top.values)
                prev = top.indices[:, 0]
        tk, pr = torch.stack(toks, dim=1), torch.stack(probs, dim=1)                       # [B, depth, 2]
        for b, (hd, _) in enumerate(lanes):
            hd.buf["tok"].copy_(tk[b])
            hd.buf["prob"].copy_(pr[b])

    def _head_f(self, c: torch.Tensor) -> torch.Tensor:
        """The draft head's logits fp32 [B, vocab] of rows c on `lin_any` (a lone row on `dense_rows` too)."""
        w = self.draft_head if self.draft_head is not None else self.fn.w("lm_head.weight")
        if not hasattr(w, "bf16_cached"):
            return self.fn.head(c, w=self.draft_head).float()
        return self.fn.lin_any(c.contiguous(), w).float()

    def _fuse_f(self, e: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self._norm(e, "mtp_fusion.enorm.weight"), self._norm(h.to(e.dtype), "mtp_fusion.hnorm.weight")), dim=-1)
        return e + self.w("mtp_fusion.gate").to(e.device, e.dtype) * self._lin(x, "mtp_fusion.eh_proj.weight")

    def _corr_f(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return torch.cat((self._norm(h, "hidden_correction.hidden_norm.weight"), self._norm(e, "hidden_correction.embed_norm.weight")), dim=-1)

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
        if self._fused():
            DFlyHead._run_lanes([(self, st)])
            return
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
