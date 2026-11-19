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

    def __init__(self, srv, W: dict | None = None):
        self.srv, self.fn, self.cfg = srv, srv.fn, srv.cfg
        d = self.cfg.dfly
        if d is None:
            raise ValueError("this checkpoint's config names no DFly draft layers")
        self.d, self.depth = d, d.draft_tokens
        self.W = W if W is not None else self.fn.W
        fn, c = self.fn, self.cfg
        self.wk, self.wv, self.sinks, self.shape = {}, {}, {}, {}
        for i in range(d.layers):
            sink = self.W[f"{P}{i}.self_attn.learnable_sink_param"].float().to(fn.device).reshape(-1, 1)
            heads = sink.shape[0]
            a = c.sparse_attn
            w = self.W[f"{P}{i}.self_attn.kv_b_proj.weight"]
            w = (w if torch.is_tensor(w) else w.bf16_cached()).to(fn.device, fn.dtype)
            if w.shape != (heads * (a.nope + a.v), c.kv_lora_rank):
                raise ValueError(f"DFly layer {i}: kv_b_proj {tuple(w.shape)} does not fit {heads} heads of the target's layout")
            w = w.view(heads, a.nope + a.v, c.kv_lora_rank)
            self.wk[i], self.wv[i] = w[:, :a.nope].contiguous(), w[:, a.nope:].transpose(1, 2).contiguous()
            self.sinks[i], self.shape[i] = sink, AttnShape(heads, a.nope, a.rope, a.v)
        self.st = type("DFlyRows", (), {"length": 0})()        # rows the drafter fed (the head keeps none)
        self._out = ([], [])
        self.stats = {"rounds": 0, "stale": 0}

    # --- the head interface ------------------------------------------------------------------------
    def reset(self) -> None:
        self.st.length = 0
        self._out = ([], [])

    def prime(self, rows, nexts) -> None:
        self.st.length += len(nexts)

    def resume(self, n: int) -> None:
        self.st.length = int(n)

    def capture_all(self) -> None:
        pass

    def views(self) -> list:
        return []

    def launch(self, rows: torch.Tensor, nexts: list[int]) -> None:
        """`rows[-1]`: the last committed row's head input; `nexts[-1]`: the anchor token."""
        self.st.length += len(nexts)
        st = self.srv.st
        if st.picks_at != st.length or len(st.picks) != len(self.fn.pick_layers):
            self.stats["stale"] += 1                         # after a restore: this round drafts nothing
            self._out = ([], [])
            return
        self._out = self.draft(int(nexts[-1]), rows[-1:].reshape(1, -1), st)
        self.stats["rounds"] += 1

    def read(self) -> tuple[list, list]:
        out, self._out = self._out, ([], [])
        return out

    # --- the arithmetic ----------------------------------------------------------------------------
    def w(self, name: str):
        return self.W[P + name]

    def fuse(self, e: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        fn = self.fn
        x = torch.cat((fn.rms(e, self.w("mtp_fusion.enorm.weight")), fn.rms(h.to(e.dtype), self.w("mtp_fusion.hnorm.weight"))), dim=-1)
        return e + self.w("mtp_fusion.gate").to(e.device, e.dtype) * fn.lin(x, self.w("mtp_fusion.eh_proj.weight"))

    def correct(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        fn = self.fn
        x = torch.cat((fn.rms(h, self.w("hidden_correction.hidden_norm.weight")), fn.rms(e, self.w("hidden_correction.embed_norm.weight"))), dim=-1)
        g = F.silu(fn.lin(x, self.w("hidden_correction.gate_proj.weight")))
        return h + fn.lin(g * fn.lin(x, self.w("hidden_correction.up_proj.weight")), self.w("hidden_correction.down_proj.weight"))

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
        allowed = torch.cat((ctx_ok.unsqueeze(0).expand(Q, -1), torch.ones(Q, Q, dtype=torch.bool, device=h.device).tril()), dim=-1)
        y = fn.attend_chunk(q, torch.cat((ctx, own)), allowed, -1, w=(self.wk[i], self.wv[i], self.sinks[i], a))
        y = (y * torch.sigmoid(gate).view(Q, a.heads, a.v)).reshape(Q, a.heads * a.v)
        h = h + fn.rms(fn.lin(y, self.w(s + "o_proj.weight")), self.w(s + "post_proj_norm.weight"))
        x = fn.rms(h, self.w(f"{i}.post_attention_layernorm.weight"))
        g = F.silu(fn.lin(x, self.w(f"{i}.mlp.gate_proj.weight")))
        return h + fn.lin(g * fn.lin(x, self.w(f"{i}.mlp.up_proj.weight")), self.w(f"{i}.mlp.down_proj.weight"))

    @torch.no_grad()
    def hidden(self, anchor: int, target_hidden: torch.Tensor, st) -> torch.Tensor:
        """The block's rows after the draft layers and their norm [depth + 1, H]."""
        fn, d = self.fn, self.d
        Q, L = self.depth + 1, st.length
        ids = torch.tensor([anchor] + [d.mask_token_id] * self.depth, dtype=torch.int64)
        e = fn.embed(ids)
        h = torch.cat((self.fuse(e[:1], target_hidden.to(fn.device)), e[1:]))
        cos, sin = fn.rope(fn.pos3(torch.arange(L, L + Q) + st.delta), SPARSE)
        for i in range(d.layers):
            pq, ok = st.picks[d.index_layers[i]]
            ok = ok & (pq < L)
            ctx = st.lat[d.kv_layers[i]][pq.clamp(0, max(L - 1, 0))]
            h = self.layer(i, h, ctx, ok, cos, sin)
        return fn.rms(h, self.w("final_norm.weight"))

    @torch.no_grad()
    def draft(self, anchor: int, target_hidden: torch.Tensor, st) -> tuple[list, list]:
        """Per drafted position its two best tokens and their probabilities, along the best tokens' chain."""
        fn = self.fn
        h = self.hidden(anchor, target_hidden, st)
        toks, probs, prev = [], [], anchor
        for j in range(self.depth):
            c = self.correct(h[j + 1:j + 2], fn.embed(torch.tensor([prev])))
            p = torch.softmax(fn.head(c)[0].float(), dim=-1)
            top = torch.topk(p, 2)
            toks.append(top.indices.tolist())
            probs.append(top.values.tolist())
            prev = toks[-1][0]
        return toks, probs
