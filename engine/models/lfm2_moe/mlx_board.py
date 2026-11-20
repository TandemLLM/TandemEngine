"""LFM2 MoE on Apple silicon: the reference forward (ref.py) in MLX, on the Mac's GPU.

`Lfm2MoeMLX` keeps `Lfm2MoeRef`'s block algebra and API (`new_state`, `rows(tokens, parents, st)`,
`commit(st, path)`, `prefill`, `decode`), so the drafting and verify loop above it is unchanged. Where
the reference loops over rows, this runs a block as one step, on one of two paths (`Geo`):

  row-invariant  the decode step and every block of up to mlx_kernels.ROWS_MAX rows (a draft tree in DFS
                 pre-order): projections and experts on the row kernels of mlx_kernels.py, attention on
                 `mlx_kernels.path_attention` (each row against the cache and its own path), the short
                 conv's window gathered by the row's path (`conv_taps`), the rest elementwise. A verified
                 row is the decode step's row bit for bit, so greedy speculation writes the plain greedy
                 text, as on the CUDA board (ENG-287)
  batched        a longer prompt chunk (a chain): batched kernels whose rows are not the row path's bits;
                 decode and verify both continue from the state it leaves

The weights are a dict of the reference's canonical names as `mx.array`, plus each MoE layer's experts
stacked as `experts.L.w13` [E, 2 I, H] and `experts.L.w2` [E, H, I] (`from_torch`, `load_checkpoint`); on
the GPU in BF16 they are repacked into the kernels' tile order (`fuse`).

What is not here yet: quantised weights, EXL3, the KV pool's pages.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from engine.models.lfm2_moe import mlx_kernels as K
from engine.models.lfm2_moe.ref import chain, tree_paths



@dataclass
class MlxState:
    """One sequence: conv state per conv layer [S, H], K and V per attention layer [kv heads, cap, D]."""
    cap: int
    length: int = 0
    conv: dict = field(default_factory=dict)
    k: dict = field(default_factory=dict)
    v: dict = field(default_factory=dict)

    def reset_inplace(self) -> None:
        self.length = 0
        for li in self.conv:
            self.conv[li] = mx.zeros_like(self.conv[li])


@dataclass
class Geo:
    """A block's geometry for `_forward`: `inv` the row-invariant path (with each row's conv taps, path,
    path length and rotary), else the batched chain path."""
    inv: bool
    paths: list
    taps: object = None
    anc: object = None
    dlen: object = None
    rope: object = None


def conv_taps(paths: list[list[int]], S: int) -> np.ndarray:
    """[M, S + 1] indices into concat(state [S rows], block rows [M]) of each row's conv window, oldest
    first: the last S + 1 entries of state followed by the block rows on the row's path."""
    M = len(paths)
    out = np.empty((M, S + 1), dtype=np.int32)
    for i, pa in enumerate(paths):
        d = len(pa)
        for t in range(S + 1):
            j = S - t                        # steps back from the row itself
            pos = S + d - 1 - j              # its place in concat(state, path rows)
            out[i, t] = S + pa[pos - S] if pos >= S else pos
    return out


class Lfm2MoeMLX:
    def __init__(self, cfg, W: dict, dtype=mx.bfloat16):
        self.cfg, self.W, self.dtype = cfg, W, dtype
        self._cc = None
        self._attn_cc: dict = {}
        self.fuse()
        D = cfg.head_dim
        self.inv_freq = 1.0 / (cfg.rope_theta ** (mx.arange(0, D, 2, dtype=mx.float32) / D))
        self._anc1 = mx.array([[0]], dtype=mx.int32)
        self._dlen1 = mx.array([1], dtype=mx.int32)
        mx.eval(self.inv_freq, self._anc1, self._dlen1)
        self.rec: dict = {}
        self._block = None
        self.hidden = None

    @staticmethod
    def _can_swiz(c, dtype) -> bool:
        if dtype != mx.bfloat16:
            return False
        if mx.default_device().type != mx.DeviceType.gpu:
            return False
        H, I, E = c.hidden_size, c.moe_intermediate_size, c.num_experts
        q_dim = c.num_attention_heads * c.head_dim
        qkv_dim = (c.num_attention_heads + 2 * c.num_key_value_heads) * c.head_dim
        return (H % 64 == 0 and I % 64 == 0 and E <= 128 and qkv_dim % 16 == 0 and q_dim % 64 == 0
                and c.intermediate_size % 64 == 0 and c.vocab_size % 16 == 0)

    # ------------------------------------------------------------------------ weights in
    @classmethod
    def from_torch(cls, cfg, W: dict, dtype=mx.float32):
        """From the reference's torch dict (tests: the same random weights on both sides)."""
        def a(t):
            return mx.array(t.detach().float().cpu().numpy()).astype(dtype)
        M: dict = {}
        for name, t in W.items():
            if ".experts." in name:
                continue
            M[name] = a(t) if not name.endswith("expert_bias") else mx.array(t.float().numpy())
        for li in cfg.moe_layers:
            f = f"layers.{li}.feed_forward.experts."
            E = cfg.num_experts
            M[f"experts.{li}.w13"] = mx.stack([mx.concatenate([a(W[f + f"{e}.w1.weight"]), a(W[f + f"{e}.w3.weight"])])
                                               for e in range(E)])
            M[f"experts.{li}.w2"] = mx.stack([a(W[f + f"{e}.w2.weight"]) for e in range(E)])
        return cls(cfg, M, dtype)

    @classmethod
    def load_checkpoint(cls, cfg, path: str, dtype=mx.bfloat16, log=print):
        """From a HF safetensors checkpoint (LiquidAI/LFM2.5-24B-A2B): per-expert w1/w3/w2 or stacked
        gate_up_proj/down_proj, `model.` prefix dropped."""
        raw: dict = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            raw.update(mx.load(f))
        raw = {k[len("model."):] if k.startswith("model.") else k: v for k, v in raw.items()}
        M: dict = {}
        I = cfg.moe_intermediate_size
        swiz = cls._can_swiz(cfg, dtype)
        for name, t in raw.items():
            if ".experts." in name:
                continue
            M[name] = t.astype(mx.float32) if name.endswith("expert_bias") else t.astype(dtype)
        for li in cfg.moe_layers:
            f = f"layers.{li}.feed_forward.experts."
            if f + "gate_up_proj" in raw:
                w13 = raw[f + "gate_up_proj"].astype(dtype)
                w2 = raw[f + "down_proj"].astype(dtype)
            else:
                E = cfg.num_experts
                w13 = mx.stack([mx.concatenate([raw[f + f"{e}.w1.weight"], raw[f + f"{e}.w3.weight"]])
                                for e in range(E)]).astype(dtype)
                w2 = mx.stack([raw[f + f"{e}.w2.weight"] for e in range(E)]).astype(dtype)
            assert w13.shape[1] == 2 * I
            if swiz:
                M[f"experts.{li}.w13"] = K.swizzle_moe_w13(w13)
                M[f"experts.{li}.w2"] = K.swizzle_moe_w2(w2)
                mx.eval(M[f"experts.{li}.w13"], M[f"experts.{li}.w2"])
            else:
                M[f"experts.{li}.w13"] = w13
                M[f"experts.{li}.w2"] = w2
                mx.eval(M[f"experts.{li}.w13"], M[f"experts.{li}.w2"])
            for e in range(cfg.num_experts):          # free the per-expert copies as we go
                for w in ("w1", "w3", "w2"):
                    raw.pop(f + f"{e}.{w}.weight", None)
            raw.pop(f + "gate_up_proj", None)
            raw.pop(f + "down_proj", None)
        mx.eval(M)
        board = cls(cfg, M, dtype)
        if log:
            log(f"[lfm-mlx] {path}: {sum(v.nbytes for v in board.W.values()) / 1e9:.2f} GB on the GPU")
        return board

    def fuse(self) -> None:
        """The row-invariant path's stacked weights: an attention layer's q, k, v (`qkv`), a dense MLP's w1, w3
        (`w13`), plus the 512B-coalesced weight swizzle on GPU bf16 models."""
        c = self.cfg
        self._swiz = self._can_swiz(c, self.dtype)
        for li in range(c.num_hidden_layers):
            p = f"layers.{li}."
            if c.is_attention(li):
                a = p + "self_attn."
                qkv = mx.concatenate([self.W[a + f"{x}_proj.weight"] for x in ("q", "k", "v")])
                self.W[a + "qkv"] = K.swizzle_dense(qkv) if self._swiz else qkv
                if self._swiz:
                    self.W[a + "out_proj.weight"] = K.swizzle_dense(self.W[a + "out_proj.weight"])
            if not c.is_moe(li):
                f = p + "feed_forward."
                w13 = mx.concatenate([self.W[f + "w1.weight"], self.W[f + "w3.weight"]])
                self.W[f + "w13"] = K.swizzle_moe_w13(w13[None]) if self._swiz else w13
                if self._swiz:
                    self.W[f + "w2.weight"] = K.swizzle_dense(self.W[f + "w2.weight"])
            elif self._swiz and self.W[f"experts.{li}.w13"].ndim == 3:
                self.W[f"experts.{li}.w13"] = K.swizzle_moe_w13(self.W[f"experts.{li}.w13"])
                self.W[f"experts.{li}.w2"] = K.swizzle_moe_w2(self.W[f"experts.{li}.w2"])
                mx.eval(self.W[f"experts.{li}.w13"], self.W[f"experts.{li}.w2"])
            if not c.is_attention(li):
                cv = p + "conv."
                self.W[cv + "conv2d"] = self.W[cv + "conv.weight"].reshape(c.hidden_size, -1)   # [H, taps]
                if self._swiz:
                    self.W[cv + "in_proj.weight"] = K.swizzle_dense(self.W[cv + "in_proj.weight"])
                    self.W[cv + "out_proj.weight"] = K.swizzle_dense(self.W[cv + "out_proj.weight"])
        if self._swiz:
            self.W["embed_tokens.weight"] = K.swizzle_dense(self.W["embed_tokens.weight"])
            if "lm_head.weight" in self.W:
                self.W["lm_head.weight"] = K.swizzle_dense(self.W["lm_head.weight"])
        mx.eval(self.W)

    # ------------------------------------------------------------------------ helpers
    def head_weight(self):
        return self.W.get("lm_head.weight", self.W["embed_tokens.weight"])

    def rms(self, x, w):
        # fused; normalises in fp32 like the reference, but multiplies by w before the cast back
        return mx.fast.rms_norm(x, w, self.cfg.rms_norm_eps)

    def rope(self, positions):
        f = positions.astype(mx.float32)[:, None] * self.inv_freq[None, :]
        emb = mx.concatenate([f, f], axis=-1)
        return mx.cos(emb).astype(self.dtype), mx.sin(emb).astype(self.dtype)

    @staticmethod
    def rotate_half(x):
        h = x.shape[-1] // 2
        return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)

    def new_state(self, cap: int) -> MlxState:
        c = self.cfg
        st = MlxState(cap=cap)
        for li in range(c.num_hidden_layers):
            if c.is_attention(li):
                st.k[li] = mx.zeros((c.num_key_value_heads, cap, c.head_dim), dtype=self.dtype)
                st.v[li] = mx.zeros_like(st.k[li])
            else:
                st.conv[li] = mx.zeros((c.conv_state_len, c.hidden_size), dtype=self.dtype)
        return st

    # ------------------------------------------------------------------------ the layers
    # The batched path: a prompt chunk of more than ROWS_MAX rows (a chain), on batched kernels and MLX's own
    # matmul and attention. Its rows are not the row path's bits; decode and verify both continue from the
    # state it leaves. The row-invariant path (the decode step and every verify block) is `_forward_inv`.
    def _lin(self, x, w, N: int, Kd: int, out_dtype=None):
        if self._swiz:
            return K.dense_prefill_swiz(x, w, N, Kd, out_dtype=out_dtype)
        out = x @ w.T
        return out.astype(out_dtype) if out_dtype is not None and out.dtype != out_dtype else out

    def conv(self, h, li: int, st: MlxState):
        c = self.cfg
        p = f"layers.{li}.conv."
        H = c.hidden_size
        bcx = self._lin(h, self.W[p + "in_proj.weight"], 3 * H, H)
        B, C, x = bcx[:, :H], bcx[:, H:2 * H], bcx[:, 2 * H:]
        bx = B * x
        self.rec[li] = bx
        wc = self.W[p + "conv.weight"].reshape(H, -1)               # [H, K]
        src = mx.concatenate([st.conv[li], bx], axis=0)              # [S + M, H]
        M, Kt = h.shape[0], wc.shape[1]
        win = [src[t:t + M] for t in range(Kt)]                      # a chain: each tap's window is a slice
        y = win[0] * wc[:, 0]
        for t in range(1, Kt):
            y = y + win[t] * wc[:, t]
        return self._lin(C * y.astype(h.dtype), self.W[p + "out_proj.weight"], H, H)

    def attention(self, h, li: int, st: MlxState):
        c = self.cfg
        p = f"layers.{li}.self_attn."
        M, D, Hq, Hk = h.shape[0], c.head_dim, c.num_attention_heads, c.num_key_value_heads
        L = st.length
        q, k, v = (h @ self.W[p + f"{x}_proj.weight"].T for x in ("q", "k", "v"))
        q = self.rms(q.reshape(M, Hq, D), self.W[p + "q_layernorm.weight"])
        k = self.rms(k.reshape(M, Hk, D), self.W[p + "k_layernorm.weight"])
        v = v.reshape(M, Hk, D)
        q, k = q.transpose(1, 0, 2)[None], k.transpose(1, 0, 2)          # [1, Hq, M, D], [Hk, M, D]
        q = mx.fast.rope(q, D, traditional=False, base=c.rope_theta, scale=1.0, offset=L)
        k = mx.fast.rope(k[None], D, traditional=False, base=c.rope_theta, scale=1.0, offset=L)[0]
        st.k[li][:, L:L + M] = k
        st.v[li][:, L:L + M] = v.transpose(1, 0, 2)
        ks, vs = st.k[li][None, :, :L + M], st.v[li][None, :, :L + M]   # [1, Hk, L + M, D]
        o = mx.fast.scaled_dot_product_attention(q, ks, vs, scale=D ** -0.5, mask="causal" if M > 1 else None)
        return self._lin(o[0].transpose(1, 0, 2).reshape(M, Hq * D), self.W[p + "out_proj.weight"],
                         c.hidden_size, Hq * D)

    def mlp(self, h, li: int):
        c = self.cfg
        p = f"layers.{li}.feed_forward."
        a, b = h @ self.W[p + "w1.weight"].T, h @ self.W[p + "w3.weight"].T
        return self._lin(mx.sigmoid(a) * a * b, self.W[p + "w2.weight"], c.hidden_size, c.intermediate_size)

    def route(self, h, li: int):
        """(ids [M, k] ascending, weights [M, k]): the top k by score + bias, renormalised by explicit
        adds in slot order (no reduction kernel whose order could follow the row count)."""
        c = self.cfg
        p = f"layers.{li}.feed_forward."
        logits = h @ self.W[p + "gate.weight"].T
        s = mx.sigmoid(logits)
        k = c.num_experts_per_tok
        choose = s + self.W[p + "expert_bias"] if c.use_expert_bias else s
        ids = mx.sort(mx.argpartition(-choose.astype(mx.float32), kth=k - 1, axis=-1)[:, :k], axis=-1)
        wts = mx.take_along_axis(s, ids, axis=-1)
        tot = wts[:, 0:1]
        for j in range(1, k):
            tot = tot + wts[:, j:j + 1]
        return ids.astype(mx.int32), wts / (tot + 1e-6) * c.routed_scaling_factor

    def moe(self, h, li: int):
        c = self.cfg
        ids, wts = self.route(h, li)                                 # [M, k]
        w13, w2 = self.W[f"experts.{li}.w13"], self.W[f"experts.{li}.w2"]
        I, M, k = c.moe_intermediate_size, h.shape[0], c.num_experts_per_tok
        if self._swiz:
            return K.moe_prefill_swiz(h, ids, wts, w13, w2, c.num_experts, I, k)
        # a prompt chunk: the (row, expert) pairs sorted by expert, so each expert is one matrix product
        # over its rows (`sorted_indices`), not a matrix-vector product a pair (800 tok/s against 7,000)
        flat = ids.reshape(-1)
        order = mx.argsort(flat)
        inv = mx.argsort(order)
        xs = h[order // k][:, None, :]                           # [M k, 1, H]
        es = flat[order]
        gu = mx.gather_mm(xs, w13.swapaxes(-1, -2), rhs_indices=es, sorted_indices=True)
        a, b = gu[..., :I], gu[..., I:]
        y = mx.gather_mm(mx.sigmoid(a) * a * b, w2.swapaxes(-1, -2), rhs_indices=es, sorted_indices=True)
        y = y[inv][:, 0, :].reshape(M, k, -1)
        wt = wts.astype(y.dtype)
        out = y[:, 0] * wt[:, 0:1]                                   # the experts added in ascending id
        for j in range(1, k):
            out = out + y[:, j] * wt[:, j:j + 1]
        return out.astype(h.dtype)

    # ------------------------------------------------------------------------ a block
    def _geo(self, parents: list[int], L: int, M: int) -> "Geo":
        """The block's geometry: the row-invariant path for up to ROWS_MAX rows (any tree), else the batched
        path (a chain only)."""
        paths = tree_paths(parents)
        if M > K.ROWS_MAX:
            if list(parents) != chain(M):
                raise ValueError(f"a draft tree of {M} rows exceeds {K.ROWS_MAX}")
            return Geo(inv=False, paths=paths)
        A = max(len(pa) for pa in paths)
        anc = np.zeros((M, A), dtype=np.int32)
        for i, pa in enumerate(paths):
            anc[i, :len(pa)] = pa
        cos, sin = self.rope(mx.array([L + len(pa) - 1 for pa in paths], dtype=mx.int32))
        # a chain's conv windows are slices (the same values a gather would move, so the same bits)
        taps = None if list(parents) == chain(M) else mx.array(conv_taps(paths, self.cfg.conv_state_len))
        return Geo(inv=True, paths=paths, taps=taps,
                   anc=mx.array(anc), dlen=mx.array([len(pa) for pa in paths], dtype=mx.int32),
                   rope=(cos[:, None, :], sin[:, None, :]))

    def rows(self, tokens, parents: list[int], st: MlxState, last_only: bool = False):
        """Logits [M, vocab] (or [1, vocab] with `last_only`) fp32 of a block at the state's length."""
        tokens = mx.array(np.asarray(tokens, dtype=np.int32).reshape(-1))
        M = tokens.size
        L = st.length
        if L + M > st.cap:
            raise ValueError(f"{L + M} rows exceed the state's capacity {st.cap}")
        g = self._geo(list(parents), L, M)
        self._block = (list(parents), g.paths)
        out = self._forward(tokens, st, g, last_only=last_only)
        mx.eval(out, *st.k.values(), *st.v.values(), *self.rec.values())
        return out

    def step(self, token, st: MlxState):
        """One decode token (an int32 `mx.array` of shape [1], possibly still lazy) committed at once:
        lazy logits [1, vocab], nothing evaluated, so the caller can queue the next step while this one
        runs (`mx.async_eval`). The row-invariant path: the row a verify block computes for this token."""
        L = st.length
        cc = self._compiled()
        lb = mx.array([L], dtype=mx.int32)
        cos, sin = cc["rope1"](lb)
        g = Geo(inv=True, paths=[[0]], taps=None, anc=self._anc1, dlen=self._dlen1, rope=(cos, sin))
        out = self._forward_inv(token, st, g, commit_conv=True, lb=lb)
        st.length = L + 1
        self.rec, self._block = {}, None
        return out

    # ------------------------------------------------------------------------ the compiled row-invariant layers
    def _compiled(self) -> dict:
        """The row-invariant path's layer parts as `mx.compile`d functions (traced once a row count and kind,
        then replayed: a decode step built ~400 ops in Python, 1.6 ms a token the server waited for). The
        decode step and every verify block call the same functions, so their rows stay bit for bit equal;
        the compiler's fused elementwise kernels compute each element alone."""
        if self._cc is not None:
            return self._cc
        c = self.cfg
        eps, k, E, I = c.rms_norm_eps, c.num_experts_per_tok, c.num_experts, c.moe_intermediate_size
        H, Hq, Hk, D = c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim
        rot = self.rotate_half

        swiz = self._swiz
        use_fast_moe = (self.dtype == mx.bfloat16 and H % 64 == 0 and I % 64 == 0 and E <= 128)
        use_fast_qkv = (D <= 64 and D % 2 == 0 and self.dtype == mx.bfloat16)
        use_fast_conv = (self.dtype == mx.bfloat16)

        def moe_res(h, nw, gw, bias, w13, w2):
            if swiz:
                return K.moe_fast_swiz(h, nw, gw, bias, w13, w2, I, eps, k, c.routed_scaling_factor, c.use_expert_bias)
            if use_fast_moe:
                return K.moe_fast(h, nw, gw, bias, w13, w2, eps, k, c.routed_scaling_factor, c.use_expert_bias)
            n = mx.fast.rms_norm(h, nw, eps)
            s = mx.sigmoid(K.dense(n, gw))
            choose = s + bias if c.use_expert_bias else s
            ids = mx.sort(mx.argpartition(-choose.astype(mx.float32), kth=k - 1, axis=-1)[:, :k], axis=-1)
            ids = ids.astype(mx.int32)
            wts = mx.take_along_axis(s, ids, axis=-1)
            tot = wts[:, 0:1]
            for j in range(1, k):
                tot = tot + wts[:, j:j + 1]
            wts = wts / (tot + 1e-6) * c.routed_scaling_factor
            M = n.shape[0]
            pexp = ids.reshape(-1)
            el = pexp if M == 1 else K.experts_used(pexp, E)
            gu = K.rowmm(n, w13, K.pair_rows(M, k), pexp, rows=M, elist=el)
            a, b = gu[:, :I], gu[:, I:]
            y = K.rowmm(mx.sigmoid(a) * a * b, w2, K._index(M * k)[0], pexp, rows=M, elist=el).reshape(M, k, -1)
            wt = wts.astype(y.dtype)
            out = y[:, 0] * wt[:, 0:1]
            for j in range(1, k):
                out = out + y[:, j] * wt[:, j:j + 1]
            return h + out.astype(h.dtype)

        def mlp_res(h, nw, w13, w2):
            n = mx.fast.rms_norm(h, nw, eps)
            if swiz:
                act = K.dense_swiglu_swiz(n, w13, c.intermediate_size, H)
                return K.dense_swiz(act, w2, H, c.intermediate_size, res=h)
            ab = K.dense(n, w13)
            a, b = ab[:, :ab.shape[1] // 2], ab[:, ab.shape[1] // 2:]
            return K.dense(mx.sigmoid(a) * a * b, w2, res=h)

        def conv_res(h, nw, inw, cw, outw, state, taps=None):
            if swiz:
                bcx = K.dense_swiz(h, inw, 3 * H, H, nw=nw, eps=eps)
                cy, bx = K.conv_gate(bcx, state, cw, taps)
                return K.dense_swiz(cy, outw, H, H, res=h), bx
            n = mx.fast.rms_norm(h, nw, eps)
            bcx = K.dense(n, inw)
            if use_fast_conv:
                cy, bx = K.conv_gate(bcx, state, cw, taps)
                return K.dense(cy, outw, res=h), bx
            B, C, x = bcx[:, :H], bcx[:, H:2 * H], bcx[:, 2 * H:]
            bx = B * x
            src = mx.concatenate([state, bx], axis=0)
            M, Kt = n.shape[0], cw.shape[1]
            win = [src[t:t + M] for t in range(Kt)] if taps is None else [src[taps[:, t]] for t in range(Kt)]
            y = win[0] * cw[:, 0]
            for t in range(1, Kt):
                y = y + win[t] * cw[:, t]
            return K.dense(C * y.astype(n.dtype), outw, res=h), bx

        def conv_step(h, nw, inw, cw, outw, state):
            if swiz:
                bcx = K.dense_swiz(h, inw, 3 * H, H, nw=nw, eps=eps)
                cy, nxt_st = K.conv_gate(bcx, state, cw, None, next_state=True)
                return K.dense_swiz(cy, outw, H, H, res=h), nxt_st
            n = mx.fast.rms_norm(h, nw, eps)
            bcx = K.dense(n, inw)
            if use_fast_conv:
                cy, nxt_st = K.conv_gate(bcx, state, cw, None, next_state=True)
                return K.dense(cy, outw, res=h), nxt_st
            B, C, x = bcx[:, :H], bcx[:, H:2 * H], bcx[:, 2 * H:]
            bx = B * x
            src = mx.concatenate([state, bx], axis=0)
            Kt = cw.shape[1]
            y = src[0:1] * cw[:, 0]
            for t in range(1, Kt):
                y = y + src[t:t + 1] * cw[:, t]
            return K.dense(C * y.astype(n.dtype), outw, res=h), src[-(Kt - 1):]

        def qkv_prep(h, nw, qkvw, qn, kn, cos, sin):
            if swiz:
                qkv = K.dense_swiz(h, qkvw, (Hq + 2 * Hk) * D, H, nw=nw, eps=eps)
            else:
                n = mx.fast.rms_norm(h, nw, eps)
                qkv = K.dense(n, qkvw)
            if use_fast_qkv:
                return K.qkv_norm_rope(qkv, qn, kn, cos, sin, Hq, Hk, D, eps)
            M = h.shape[0]
            q = mx.fast.rms_norm(qkv[:, :Hq * D].reshape(M, Hq, D), qn, eps)
            kk = mx.fast.rms_norm(qkv[:, Hq * D:(Hq + Hk) * D].reshape(M, Hk, D), kn, eps)
            v = qkv[:, (Hq + Hk) * D:].reshape(M, Hk, D)
            q = q * cos + rot(q) * sin
            kk = kk * cos + rot(kk) * sin
            return q, kk.transpose(1, 0, 2), v.transpose(1, 0, 2)

        def attn_out(h, o, ow):
            if swiz:
                return K.dense_swiz(o.reshape(o.shape[0], -1), ow, H, Hq * D, res=h)
            return K.dense(o.reshape(o.shape[0], -1), ow, res=h)

        def head(h, nw, hw):
            n = mx.fast.rms_norm(h, nw, eps)
            if swiz:
                return n, K.dense_swiz(n, hw, c.vocab_size, H, out_dtype=mx.float32)
            return n, K.dense(n, hw, out_dtype=mx.float32)

        inv_freq = self.inv_freq
        dt = self.dtype

        def rope1(pos):
            f = pos.astype(mx.float32)[:, None] * inv_freq[None, :]
            emb = mx.concatenate([f, f], axis=-1)
            return mx.cos(emb).astype(dt)[:, None, :], mx.sin(emb).astype(dt)[:, None, :]

        self._raw = {"qkv": qkv_prep, "attn_out": attn_out}
        self._cc = {name: mx.compile(f) for name, f in (("moe", moe_res), ("mlp", mlp_res), ("conv", conv_res),
                                                          ("conv_step", conv_step), ("qkv", qkv_prep),
                                                          ("attn_out", attn_out), ("head", head), ("rope1", rope1))}
        return self._cc

    def _compiled_attn(self, nchunks: int):
        fn = self._attn_cc.get(nchunks)
        if fn is not None:
            return fn
        c = self.cfg
        self._compiled()
        qkv_fn, out_fn = self._raw["qkv"], self._raw["attn_out"]
        gqa = c.gqa

        def attn_full(h, nw, qkvw, qn, kn, ow, kc, vc, cos, sin, anc, dlen, lb):
            q, k, v = qkv_fn(h, nw, qkvw, qn, kn, cos, sin)
            kc = mx.slice_update(kc, k, lb, axes=[1])
            vc = mx.slice_update(vc, v, lb, axes=[1])
            o = K.path_attention(q, kc, vc, anc, dlen, 0, gqa, lb=lb, nchunks=nchunks)
            return out_fn(h, o, ow), kc, vc

        fn = self._attn_cc[nchunks] = mx.compile(attn_full)
        return fn

    def _forward_inv(self, tokens, st: MlxState, g, commit_conv: bool = False, lb=None):
        """The row-invariant path through the compiled parts (`_compiled`)."""
        c, W, cc = self.cfg, self.W, self._compiled()
        L = st.length
        if lb is None:
            lb = mx.array([L], dtype=mx.int32)
        nchunks = (L + g.anc.shape[1] + K.ATTN_BK - 1) // K.ATTN_BK
        attn_fn = self._compiled_attn(nchunks) if (c.head_dim == 64 and c.gqa == 4 and self.dtype == mx.bfloat16) else None
        self.rec = {}
        h = (K.embed_swiz(tokens, W["embed_tokens.weight"], c.hidden_size) if self._swiz
             else W["embed_tokens.weight"][tokens].astype(self.dtype))
        for li in range(c.num_hidden_layers):
            p = f"layers.{li}."
            if c.is_attention(li):
                a = p + "self_attn."
                if attn_fn is not None:
                    h, st.k[li], st.v[li] = attn_fn(
                        h, W[p + "operator_norm.weight"], W[a + "qkv"], W[a + "q_layernorm.weight"],
                        W[a + "k_layernorm.weight"], W[a + "out_proj.weight"], st.k[li], st.v[li],
                        *g.rope, g.anc, g.dlen, lb)
                else:
                    q, k, v = cc["qkv"](h, W[p + "operator_norm.weight"], W[a + "qkv"], W[a + "q_layernorm.weight"],
                                        W[a + "k_layernorm.weight"], *g.rope)
                    M = q.shape[0]
                    st.k[li][:, L:L + M] = k
                    st.v[li][:, L:L + M] = v
                    o = K.path_attention(q, st.k[li], st.v[li], g.anc, g.dlen, L, c.gqa, lb=lb)
                    h = cc["attn_out"](h, o, W[a + "out_proj.weight"])
            else:
                cv = p + "conv."
                args = (h, W[p + "operator_norm.weight"], W[cv + "in_proj.weight"], W[cv + "conv2d"],
                        W[cv + "out_proj.weight"], st.conv[li])
                if commit_conv:
                    h, st.conv[li] = cc["conv_step"](*args)
                else:
                    h, self.rec[li] = cc["conv"](*args) if g.taps is None else cc["conv"](*args, g.taps)
            f = p + "feed_forward."
            if c.is_moe(li):
                h = cc["moe"](h, W[p + "ffn_norm.weight"], W[f + "gate.weight"], W[f + "expert_bias"],
                              W[f"experts.{li}.w13"], W[f"experts.{li}.w2"])
            else:
                h = cc["mlp"](h, W[p + "ffn_norm.weight"], W[f + "w13"], W[f + "w2.weight"])
        self.hidden, out = cc["head"](h, W["embedding_norm.weight"], self.head_weight())
        return out

    def _forward(self, tokens, st: MlxState, g, last_only: bool = False):
        if g.inv:
            out = self._forward_inv(tokens, st, g)
            return out[-1:] if last_only else out
        c = self.cfg
        self.rec = {}
        h = (K.embed_swiz(tokens, self.W["embed_tokens.weight"], c.hidden_size) if self._swiz
             else self.W["embed_tokens.weight"][tokens].astype(self.dtype))
        for li in range(c.num_hidden_layers):
            p = f"layers.{li}."
            n = self.rms(h, self.W[p + "operator_norm.weight"])
            h = h + (self.attention(n, li, st) if c.is_attention(li) else self.conv(n, li, st))
            n = self.rms(h, self.W[p + "ffn_norm.weight"])
            h = h + (self.moe(n, li) if c.is_moe(li) else self.mlp(n, li))
        h = self.rms(h, self.W["embedding_norm.weight"])
        self.hidden = h                    # the rows' final normed states [M, H] (a draft head reads them)
        if last_only:
            if self._swiz:
                return K.dense_swiz(h[-1:], self.head_weight(), c.vocab_size, c.hidden_size, out_dtype=mx.float32)
            return (h[-1:] @ self.head_weight().T).astype(mx.float32)
        return self._lin(h, self.head_weight(), c.vocab_size, c.hidden_size, out_dtype=mx.float32)

    def commit(self, st: MlxState, path: list[int]) -> None:
        parents, _ = self._block
        if path and (path[0] != 0 or any(parents[b] != a for a, b in zip(path, path[1:]))):
            raise ValueError(f"{path} is not a root path of the last block")
        L, n = st.length, len(path)
        idx = mx.array(path, dtype=mx.int32)
        S = self.cfg.conv_state_len
        for li, bx in self.rec.items():
            st.conv[li] = mx.concatenate([st.conv[li], bx[idx]], axis=0)[-S:]
        if any(r != j for j, r in enumerate(path)):
            for li in st.k:
                st.k[li][:, L:L + n] = st.k[li][:, L + idx]
                st.v[li][:, L:L + n] = st.v[li][:, L + idx]
        st.length = L + n
        mx.eval(*st.conv.values(), *st.k.values(), *st.v.values())
        self._block, self.rec = None, {}

    def prefill(self, tokens, st: MlxState, chunk: int = 2048, last_only: bool = False):
        """A prompt in chunks of `chunk` rows: logits [T, vocab] of every row, or [1, vocab] of the
        last with `last_only` (a long prompt's logits would not fit)."""
        toks = np.asarray(tokens, dtype=np.int32).reshape(-1)
        outs, hid = [], []
        for s in range(0, len(toks), chunk):
            part = toks[s:s + chunk]
            lg = self.rows(part, chain(len(part)), st, last_only=last_only)
            outs.append(lg[-1:] if last_only else lg)
            hid.append(self.hidden)
            self.commit(st, list(range(len(part))))
        # every prompt row's final normed state [T, H] (the drafter syncs on them)
        self.prompt_hidden = hid[0] if len(hid) == 1 else mx.concatenate(hid, axis=0)
        return outs[-1] if last_only else mx.concatenate(outs, axis=0)

    def decode(self, token: int, st: MlxState):
        if st.length + 1 > st.cap:
            raise ValueError(f"{st.length + 1} rows exceed the state's capacity {st.cap}")
        out = self.step(mx.array([int(token)], dtype=mx.int32), st)
        mx.eval(out, *st.conv.values(), *st.k.values(), *st.v.values())
        return out

    def generate(self, token: int, st: MlxState, n: int, stop=()):
        """Greedy tokens after `token`, up to `n`, one step queued ahead of the one read (the plain
        decode loop, no drafting). Yields ints."""
        y = mx.array([int(token)], dtype=mx.int32)
        nxt = mx.argmax(self.step(y, st), axis=-1).astype(mx.int32)
        mx.async_eval(nxt)
        for _ in range(n):
            cur = nxt
            if st.length < st.cap:
                nxt = mx.argmax(self.step(cur, st), axis=-1).astype(mx.int32)
                mx.async_eval(nxt)
            t = int(cur.item())
            yield t
            if t in stop or st.length >= st.cap:
                return


def load(path: str, dtype=mx.bfloat16, log=print):
    """(cfg, board) of an HF LFM2 MoE checkpoint directory."""
    from engine.models.lfm2_moe.config import config_from_dict
    with open(os.path.join(path, "config.json")) as f:
        raw = json.load(f)
    gen = None
    gp = os.path.join(path, "generation_config.json")
    if os.path.isfile(gp):
        with open(gp) as f:
            gen = json.load(f)
    cfg = config_from_dict(raw, path, gen)
    return cfg, Lfm2MoeMLX.load_checkpoint(cfg, path, dtype, log)
