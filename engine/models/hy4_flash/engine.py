"""Hy4-flash on the board: the reference forward's arithmetic arranged for serving.

`Hy4FlashRef` (ref.py) is the model, checked against `transformers`. This class keeps its parts and
changes how attention reads its rows, the way latent attention is served:

  absorbed keys   a layer never expands a cached latent into per-head keys and values. The query takes
                  the key half of `kv_b_proj` (qa_h = q_nope_h W_k,h), scores are qa_h . latent + q_rot_h .
                  turned channels, the weighted sum of the latents goes through the value half once
                  (out_h = W_v,h (sum_j p_hj latent_j)). Same values as the reference up to rounding, and
                  a token costs its 576 latent values a layer, nothing else.
  read lists      every row reads an explicit list of rows in ascending position: a window layer the
                  last `sliding_window`, a sparse layer everything up to `index_topk` rows of context
                  and past that the 4 rows of each selected group and its own open group.
  state           the 12 sparse layers keep their latents and one indexer key a row by slot (a row that
                  closes a group keeps the group's pooled, turned key, any other row its token key); the
                  30 window layers keep a ring of `sliding_window` latents; Engram keeps the last two
                  tokens. Rings and tokens are small and copied whole by the caches; the by-slot buffers
                  are sliced by length.

  rows(tokens, parents)   a decode token or a draft block of up to 32 rows in DFS pre-order. The
                  projections of the block run together on row-invariant kernels; every other
                  operation of a row runs on that row alone with shapes that do not depend on the
                  block, so a row of a block is the decode step's row at that position bit for bit.
                  Nothing but the block's scratch rows is written.
  commit(path)    the path's scratch rows move into the state, the length moves.
  prefill(tokens) a prompt in chunks on batched operations: the next token agrees with the row
                  path, the bits do not.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.models.flashnext.ref import topk_stable
from engine.models.hy4_flash.config import SPARSE, Hy4FlashConfig
from engine.models.hy4_flash.ref import Hy4FlashRef, rotate

#: rows a block may hold: the anchor and 31 drafted rows (the row kernels' limit)
BLOCK = 32
#: rows a prompt chunk takes (the caches' anchor grid)
PREFILL_CHUNK = 4096
#: rows of a prompt chunk the attention and the experts take at a time
SUB = 256
#: rows of a prompt chunk a sparse layer takes at a time once its rows read lists
SUB_LIST = 32


class Hy4EngState:
    """One sequence: the by-slot buffers, the rings, the last tokens, and a block's scratch rows."""

    def __init__(self, cfg: Hy4FlashConfig, cap: int, dtype, device):
        self.cfg, self.cap, self.length, self.delta = cfg, int(cap), 0, 0
        R, D = cfg.kv_lora_rank + cfg.sparse_attn.rope, cfg.index_head_dim

        def z(*shape):
            return torch.zeros(*shape, dtype=dtype, device=device)
        self.lat = {li: z(self.cap, R) for li in cfg.sparse_layers}
        self.ik = {li: z(self.cap, D) for li in cfg.sparse_layers}
        self.win = {li: z(cfg.sliding_window, R) for li in cfg.window_layers}
        self.hist = torch.full((max(cfg.engram_max_ngram_size - 1, 1),), cfg.engram_pad_id, dtype=torch.int64, device=device)
        # a block's rows, written by `rows` and read by `commit`
        self.blat = {li: z(BLOCK, R) for li in range(cfg.num_hidden_layers)}
        self.bik = {li: z(BLOCK, D) for li in cfg.sparse_layers}
        self.bgk = {li: z(BLOCK, D) for li in cfg.sparse_layers}
        self.btok: list[int] = []
        self.bparents: list[int] = []
        # the rows the last committed row read in the layers a draft head follows: layer -> (positions, mask),
        # good while `picks_at` is the length (a restore or a length set by hand leaves them stale)
        self.bpick: dict = {}
        self.picks: dict = {}
        self.picks_at = -1

    def reset_inplace(self) -> None:
        """A fresh sequence in the same tensors (the by-slot rows stay: nothing reads past the length)."""
        self.length, self.delta = 0, 0
        for t in self.win.values():
            t.zero_()
        self.hist.fill_(self.cfg.engram_pad_id)
        self.btok, self.bparents = [], []
        self.bpick, self.picks, self.picks_at = {}, {}, -1

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for d in (self.lat, self.ik, self.win) for t in d.values())


def _paths(parents: list[int]) -> tuple[list[int], list[list[int]]]:
    """A block in DFS pre-order -> every row's depth and its path from the anchor (itself last)."""
    depth, paths = [], []
    for r, p in enumerate(parents):
        if p < 0:
            if r != 0:
                raise ValueError("a block has one anchor, its row 0")
            depth.append(0)
            paths.append([0])
        else:
            if not 0 <= p < r:
                raise ValueError(f"row {r}: parent {p} must come before it")
            depth.append(depth[p] + 1)
            paths.append(paths[p] + [r])
    return depth, paths


class Hy4FlashEngine(Hy4FlashRef):
    """`W` as the reference's, with two additions the loader makes: `layers.N.self_attn.kv_b_proj.weight`
    is a plain tensor (the absorbed halves are read from it), and on a device the routed experts may be
    a kernel bank (`banks[layer]`, tools/fn_kernels.MoEBank) instead of `ExpertBank`s."""

    def __init__(self, cfg: Hy4FlashConfig, W: dict, tables: dict, dtype=torch.bfloat16, device="cpu",
                 max_len: int = 32768, banks: dict | None = None, cuts: dict | None = None):
        super().__init__(cfg, W, tables, dtype=dtype, device=device, cuts=cuts)
        self.max_len = int(max_len)
        self.banks = banks or {}
        self.kernels = self.device.type == "cuda"
        # the absorbed halves of kv_b_proj: [heads, nope, R] for the keys, [heads, R, v] for the values
        self.wk, self.wv = {}, {}
        for li in range(cfg.num_hidden_layers):
            a = cfg.attn(li)
            w = W[f"layers.{li}.self_attn.kv_b_proj.weight"]
            w = (w if torch.is_tensor(w) else w.bf16_cached()).to(self.device, dtype).view(a.heads, a.nope + a.v, cfg.kv_lora_rank)
            self.wk[li] = w[:, :a.nope].contiguous()
            self.wv[li] = w[:, a.nope:].transpose(1, 2).contiguous()
        self.sinks = {li: W[f"layers.{li}.self_attn.learnable_sink_param"].float().to(self.device).reshape(-1, 1)
                      for li in range(cfg.num_hidden_layers)}
        #: the sparse layers whose last row's read list is kept for a draft head
        self.pick_layers = set(cfg.dfly.index_layers) if cfg.dfly is not None else set()
        self.prompt_hidden = None
        self.stats = {"rows": 0, "blocks": 0, "prefill_rows": 0, "engram_rows": 0}

    def new_state(self, cap: int | None = None) -> Hy4EngState:
        return Hy4EngState(self.cfg, cap or self.max_len, self.dtype, self.device)

    # ------------------------------------------------------------------------- primitives
    def lin_rows(self, x: torch.Tensor, w) -> torch.Tensor:
        """x [M, K] @ w^T with every row's bits its own: the row-invariant EXL3 kernel on a device, else one
        launch a row (the same shape whatever the block)."""
        if x.shape[0] == 1:
            return self.lin1(x, w)
        if self.kernels and hasattr(w, "bf16_cached") and x.shape[0] <= BLOCK:
            return self.lin1(x, w)
        return torch.cat([self.lin1(x[r:r + 1], w) for r in range(x.shape[0])])

    def lin(self, x: torch.Tensor, w) -> torch.Tensor:
        """On a device an EXL3 projection never dequantises: up to BLOCK rows on the row kernel, more on the
        library's GEMM."""
        if not (self.kernels and hasattr(w, "bf16_cached") and x.dtype == torch.bfloat16):
            return super().lin(x, w)
        x2 = x.reshape(-1, x.shape[-1])
        if x2.shape[0] <= BLOCK:
            return self.lin1(x2, w).view(*x.shape[:-1], -1)
        from tools.exl3_linear import exl3_matmul
        if x2.shape[-1] < w.K:
            x2 = F.pad(x2, (0, w.K - x2.shape[-1]))
        y = exl3_matmul(x2.contiguous(), w)
        n = self.cuts.get(id(w))
        return (y if n is None else y[:, :n]).view(*x.shape[:-1], -1)

    def head(self, n: torch.Tensor) -> torch.Tensor:
        if self.kernels and hasattr(self.w("lm_head.weight"), "bf16_cached") and n.dtype == torch.bfloat16:
            return torch.cat([self.head_rows(n[i:i + BLOCK]) for i in range(0, n.shape[0], BLOCK)])
        return super().head(n)

    def lin1(self, x: torch.Tensor, w) -> torch.Tensor:
        if self.kernels and hasattr(w, "bf16_cached") and x.dtype == torch.bfloat16:
            from tools import fn_kernels as FK
            if x.shape[-1] < w.K:
                x = F.pad(x, (0, w.K - x.shape[-1]))
            y = FK.dense(x.contiguous(), w)
            n = self.cuts.get(id(w))
            return y if n is None else y[:, :n]
        return self.lin(x, w)

    def pos3(self, positions: torch.Tensor) -> torch.Tensor:
        return positions.reshape(1, -1).expand(3, -1)

    # ------------------------------------------------------------------------- attention, one row
    def attend_row(self, q: torch.Tensor, K: torch.Tensor, valid: torch.Tensor, li: int) -> torch.Tensor:
        """q [heads, qk] (turned) over the latents K [n, R + rope] it reads -> [heads, v] in the compute dtype."""
        cfg, a = self.cfg, self.cfg.attn(li)
        R = cfg.kv_lora_rank
        qa = torch.bmm(q[:, None, :a.nope], self.wk[li]).squeeze(1).float()                 # [heads, R]
        Kf = K.float()
        s = (qa @ Kf[:, :R].T + q[:, a.nope:].float() @ Kf[:, R:].T) * (a.qk ** -0.5)       # [heads, n]
        s = torch.cat((s.masked_fill(~valid, float("-inf")), self.sinks[li]), dim=-1)
        s = s - s.max(dim=-1, keepdim=True).values
        p = F.softmax(s, dim=-1)[:, :-1]
        mix = (p @ Kf[:, :R]).to(self.dtype)                                                # [heads, R]
        return torch.bmm(mix[:, None, :], self.wv[li]).squeeze(1)

    def _window_list(self, p: int, L: int, path: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Positions p - window + 1 .. p as rows of cat(ring, scratch), and which of them exist."""
        Wn = self.cfg.sliding_window
        q = torch.arange(p - Wn + 1, p + 1, device=self.device)
        blk = (q - L).clamp(min=0)
        src = torch.where(q < L, q % Wn, Wn + torch.tensor(path, device=self.device)[blk.clamp(max=len(path) - 1)])
        return src, q >= 0

    def _gather(self, by_slot: torch.Tensor, scratch: torch.Tensor, q: torch.Tensor, L: int, path: list[int]) -> torch.Tensor:
        """Rows at positions q: committed ones from the by-slot buffer, the block's from its scratch rows."""
        out = by_slot[q.clamp(0, max(L - 1, 0))]
        blk = q >= L
        if bool(blk.any()):
            pr = torch.tensor(path, device=self.device)[(q - L).clamp(0, len(path) - 1)]
            out = torch.where(blk.unsqueeze(-1), scratch[pr], out)
        return out

    def _sparse_positions(self, li: int, st: Hy4EngState, r: int, p: int, depth: list[int], path: list[int],
                          x: torch.Tensor, iq: torch.Tensor, hw: torch.Tensor, cos, sin) -> tuple[torch.Tensor, torch.Tensor]:
        """The positions row r of a sparse layer reads, ascending, padded to `index_topk`, and their mask."""
        cfg = self.cfg
        NS, rr, L = cfg.index_topk, cfg.index_kpool, st.length
        qg = p // rr
        if qg < cfg.group_topk:                           # everything so far fits: no selection
            q = torch.arange(NS, device=self.device)
            return q, q <= p
        nh, D, rd = cfg.index_n_heads, cfg.index_head_dim, cfg.sparse_attn.rope
        Gc = st.cap // rr
        closed = [a for a in path[:-1] if (L + depth[a]) % rr == rr - 1]         # ancestors that closed a group
        BG = BLOCK // rr + 1
        bkeys = torch.zeros(BG, D, dtype=self.dtype, device=self.device)
        if closed:
            bkeys[:len(closed)] = st.bgk[li][torch.tensor(closed, device=self.device)]
        keys = torch.cat((st.ik[li][rr - 1::rr][:Gc], bkeys))
        qi = iq.view(nh, D)
        qi = torch.cat((qi[:, :D - rd], rotate(qi[:, D - rd:], cos, sin)), dim=-1)
        logit = (hw.float() * (nh ** -0.5 * D ** -0.5)) @ F.relu(qi.float() @ keys.float().T)      # [Gc + BG]
        gid = torch.arange(Gc + BG, device=self.device)
        ok = (gid < min(qg, L // rr)) | ((gid >= Gc) & (gid < Gc + len(closed)))
        top = topk_stable(logit.masked_fill(~ok, float("-inf")).unsqueeze(0), cfg.group_topk - 1)[1][0]
        bg = torch.tensor([(L + depth[a]) // rr for a in closed] + [0] * (BG - len(closed)), device=self.device)
        groups = torch.where(top < Gc, top, bg[(top - Gc).clamp(min=0)]).sort().values
        off = torch.arange(rr, device=self.device)
        q = torch.cat(((groups.unsqueeze(-1) * rr + off).flatten(), qg * rr + off))
        return q, q <= p

    # ------------------------------------------------------------------------- a block's rows
    def attention_rows(self, x: torch.Tensor, li: int, st: Hy4EngState, depth: list[int], paths: list[list[int]]) -> torch.Tensor:
        cfg = self.cfg
        a, kind, p_ = cfg.attn(li), cfg.layer_types[li], f"layers.{li}.self_attn."
        M, R, L, rr = x.shape[0], cfg.kv_lora_rank, st.length, cfg.index_kpool
        gate = self.lin_rows(x, self.w(p_ + "linear_gate.weight"))
        qres = torch.cat([self.rms(t, self.w(p_ + "q_a_layernorm.weight"))
                          for t in self.lin_rows(x, self.w(p_ + "q_a_proj.weight")).split(1)])
        q = self.lin_rows(qres, self.w(p_ + "q_b_proj.weight"))
        ckv = self.lin_rows(x, self.w(p_ + "kv_a_proj_with_mqa.weight"))
        sparse = kind == SPARSE
        if sparse:
            ip = p_ + "compressed_indexer."
            tk = self.lin_rows(x, self.w(ip + "token_key_projection.weight"))
            iq = self.lin_rows(qres, self.w(ip + "query_projection.weight"))
            hw = self.lin_rows(x, self.w(ip + "head_weight_projection.weight"))
            D, rd = cfg.index_head_dim, a.rope
        out = []
        for r in range(M):
            pos = L + depth[r]
            cos, sin = self.rope(self.pos3(torch.tensor([pos + st.delta])), kind)
            qr = q[r].view(a.heads, a.qk)
            qr = torch.cat((qr[:, :a.nope], rotate(qr[:, a.nope:], cos, sin)), dim=-1)
            st.blat[li][r] = torch.cat((self.rms(ckv[r:r + 1, :R], self.w(p_ + "kv_a_layernorm.weight")),
                                        rotate(ckv[r:r + 1, R:], cos, sin)), dim=-1)[0]
            if sparse:
                st.bik[li][r] = self.rms(tk[r:r + 1], self.w(ip + "token_key_norm.weight"))[0]
                if pos % rr == rr - 1:                    # the row closes its group: pool it, turn it at its first row
                    g0 = pos - rr + 1
                    tks = self._gather(st.ik[li], st.bik[li], torch.arange(g0, pos + 1, device=self.device), L, paths[r])
                    gk = self.group_keys(tks, li)
                    c0, s0 = self.rope(self.pos3(torch.tensor([g0 + st.delta])), kind)
                    st.bgk[li][r] = torch.cat((gk[:, :D - rd], rotate(gk[:, D - rd:], c0, s0)), dim=-1)[0]
                pq, valid = self._sparse_positions(li, st, r, pos, depth, paths[r], x[r], iq[r], hw[r], cos, sin)
                if li in self.pick_layers:
                    st.bpick[(li, r)] = (pq, valid)
                K = self._gather(st.lat[li], st.blat[li], pq, L, paths[r])
            else:
                src, valid = self._window_list(pos, L, paths[r])
                K = torch.cat((st.win[li], st.blat[li]))[src]
            out.append(self.attend_row(qr, K, valid, li).reshape(1, -1))
        y = torch.cat(out) * torch.sigmoid(gate)
        o = self.lin_rows(y, self.w(p_ + "o_proj.weight"))
        return torch.cat([self.rms(t, self.w(p_ + "post_proj_norm.weight")) for t in o.split(1)])

    def expert_pairs(self, z: torch.Tensor, idx: torch.Tensor, li: int) -> torch.Tensor:
        """Every (row, slot) pair through its expert: [M * k, latent] fp32, unweighted."""
        lim = self.cfg.swiglu_limit
        M, k = idx.shape
        if li in self.banks:
            from tools import fn_kernels as FK
            bank, mod = self.banks[li], FK._module()
            flat = idx.reshape(-1).to(torch.int32).contiguous()
            order = FK.expert_order(idx.to(torch.int32)) if M > 1 else torch.empty(0, dtype=torch.int32, device=z.device)
            ygu = torch.empty(M * k, 2, bank.I, dtype=torch.bfloat16, device=z.device)
            yd = torch.empty(M * k, bank.H, dtype=torch.float32, device=z.device)
            mod.fn_moe_gu(z.contiguous(), flat, order, bank.tr, bank.kb, bank.su, bank.sv, ygu, bank.ks[-1])
            ygu[:, 0].clamp_(max=lim)
            ygu[:, 1].clamp_(min=-lim, max=lim)
            mod.fn_moe_down(ygu, flat, order, bank.tr, bank.kb, bank.su, bank.sv, yd, bank.ks[-1])
            return yd
        p = f"layers.{li}.mlp.experts."
        G, U, D = self.w(p + "gate_proj"), self.w(p + "up_proj"), self.w(p + "down_proj")
        out = []
        for r in range(M):
            for e in idx[r].tolist():
                g = F.linear(z[r:r + 1], G.weight(e, z.dtype)).clamp(max=lim)
                u = F.linear(z[r:r + 1], U.weight(e, z.dtype)).clamp(min=-lim, max=lim)
                out.append(F.linear(F.silu(g) * u, D.weight(e, z.dtype)).float())
        return torch.cat(out)

    def combine(self, yd: torch.Tensor, idx: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
        """A row's pairs weighted and added in ascending expert id, in fp32, rounded once."""
        M, k = idx.shape
        if self.kernels and self.dtype == torch.bfloat16:
            from tools import fn_kernels as FK
            return FK.moe_combine(yd, idx.to(torch.int32).contiguous(), wts.contiguous())
        order = idx.argsort(dim=1, stable=True)
        y = yd.view(M, k, -1)
        acc = torch.zeros(M, y.shape[-1], dtype=torch.float32, device=yd.device)
        rows = torch.arange(M, device=yd.device)
        for j in range(k):
            s = order[:, j]
            acc = acc + y[rows, s] * wts[rows, s, None]
        return acc.to(self.dtype)

    def mlp_rows(self, x: torch.Tensor, li: int) -> torch.Tensor:
        p = f"layers.{li}.mlp."
        if self.cfg.is_dense_mlp(li):
            g = F.silu(self.lin_rows(x, self.w(p + "gate_proj.weight")))
            y = self.lin_rows(g * self.lin_rows(x, self.w(p + "up_proj.weight")), self.w(p + "down_proj.weight"))
            return torch.cat([self.rms(t, self.w(p + "post_fc2_norm.weight")) for t in y.split(1)])
        routes = [self.route(x[r:r + 1], li) for r in range(x.shape[0])]
        idx, wts = torch.cat([i for i, _ in routes]), torch.cat([w for _, w in routes])
        z = self.lin_rows(x, self.w(p + "routed_expert_down_proj.weight"))
        routed = self.lin_rows(self.combine(self.expert_pairs(z, idx, li), idx, wts), self.w(p + "routed_expert_up_proj.weight"))
        sp = p + "shared_experts."
        g = F.silu(self.lin_rows(x, self.w(sp + "gate_proj.weight")))
        shared = self.lin_rows(g * self.lin_rows(x, self.w(sp + "up_proj.weight")), self.w(sp + "down_proj.weight"))
        return torch.cat([self.rms(t, self.w(p + "combine_output_norm.weight")) for t in (routed + shared).split(1)])

    def _row_grams(self, st: Hy4EngState, tokens: list[int], paths: list[list[int]]) -> torch.Tensor:
        """[M, n] each row's token and the n - 1 before it: its ancestors, then the state's last tokens."""
        n = self.cfg.engram_max_ngram_size
        hist = st.hist.tolist()
        out = []
        for path in paths:
            chain = hist + [tokens[a] for a in path]
            out.append([chain[-1 - j] for j in range(n)])
        return torch.tensor(out, dtype=torch.int64)

    @torch.no_grad()
    def rows(self, tokens: list[int], parents: list[int], st: Hy4EngState) -> tuple[torch.Tensor, torch.Tensor]:
        """A block in DFS pre-order (row 0 at the state's length) -> fp32 logits [M, vocab] and the head's
        input [M, H]. Only the block's scratch rows are written."""
        cfg = self.cfg
        M = len(tokens)
        if not 1 <= M <= BLOCK or len(parents) != M:
            raise ValueError(f"a block holds 1 to {BLOCK} rows")
        depth, paths = _paths([int(p) for p in parents])
        if st.length + max(depth) >= st.cap:
            raise ValueError(f"the state holds {st.length} of {st.cap} rows: no room for this block")
        tok = torch.tensor(tokens, dtype=torch.int64)
        st.btok, st.bparents = [int(t) for t in tokens], [int(p) for p in parents]
        live = tok != cfg.engram_pad_id
        for skip in (cfg.image_token_id, cfg.video_token_id):
            if skip is not None:
                live = live & (tok != skip)
        grams = self._row_grams(st, st.btok, paths) if self.hashes else None
        h = self.embed(tok).unsqueeze(1).expand(-1, cfg.hc_mult, -1).contiguous()
        for li in range(cfg.num_hidden_layers):
            p = f"layers.{li}."
            if li in self.hashes:
                rows = self.engram_rows(li, self.hashes[li].ids(grams))
                self.stats["engram_rows"] += rows.shape[0] * rows.shape[1]
                h = h + torch.cat([self.engram_apply(h[r:r + 1], rows[r:r + 1], li, live[r:r + 1]) for r in range(M)])
            xs, posts = zip(*[self.hc(h[r:r + 1], p + "hc_attn_layer.hc_pre.") for r in range(M)])
            x = torch.cat([self.rms(t, self.w(p + "input_layernorm.weight")) for t in xs])
            y = self.attention_rows(x, li, st, depth, paths)
            h = torch.cat([self.inject(h[r:r + 1], y[r:r + 1], posts[r]) for r in range(M)])
            xs, posts = zip(*[self.hc(h[r:r + 1], p + "hc_mlp_layer.hc_pre.") for r in range(M)])
            x = torch.cat([self.rms(t, self.w(p + "post_attention_layernorm.weight")) for t in xs])
            y = self.mlp_rows(x, li)
            h = torch.cat([self.inject(h[r:r + 1], y[r:r + 1], posts[r]) for r in range(M)])
        n = torch.cat([self.final(h[r:r + 1]) for r in range(M)])
        self.stats["rows"] += M
        self.stats["blocks"] += 1
        return self.head_rows(n), n

    def head_rows(self, n: torch.Tensor) -> torch.Tensor:
        w = self.w("lm_head.weight")
        if self.kernels and hasattr(w, "bf16_cached"):
            from tools import fn_kernels as FK
            y = FK.dense(n.contiguous(), w)
            y = y if y.dtype == torch.float32 else y.float()
            cut = self.cuts.get(id(w))
            return y if cut is None else y[:, :cut]
        return torch.cat([Hy4FlashRef.head(self, n[r:r + 1]) for r in range(n.shape[0])])

    @torch.no_grad()
    def commit(self, st: Hy4EngState, path: list[int]) -> None:
        """Keep the block rows `path` (a chain from the anchor): their scratch rows into the state."""
        cfg = self.cfg
        path = [int(r) for r in path]
        for i, r in enumerate(path):
            if r >= len(st.bparents) or st.bparents[r] != (path[i - 1] if i else -1):
                raise ValueError(f"{path} is not a chain of the last block from its anchor")
        L, rr, Wn = st.length, cfg.index_kpool, cfg.sliding_window
        if not path:
            return
        rows = torch.tensor(path, device=self.device)
        slots = torch.arange(L, L + len(path), device=self.device)
        closes = (slots % rr == rr - 1).unsqueeze(-1)
        for li in cfg.sparse_layers:
            st.lat[li][slots] = st.blat[li][rows]
            st.ik[li][slots] = torch.where(closes, st.bgk[li][rows], st.bik[li][rows])
        for li in cfg.window_layers:
            st.win[li][slots % Wn] = st.blat[li][rows]
        toks = st.hist.tolist() + [st.btok[r] for r in path]
        st.hist.copy_(torch.tensor(toks[-st.hist.shape[0]:], dtype=torch.int64))
        st.length = L + len(path)
        st.picks = {li: st.bpick[(li, path[-1])] for li in self.pick_layers if (li, path[-1]) in st.bpick}
        st.picks_at = st.length if len(st.picks) == len(self.pick_layers) else -1
        st.btok, st.bparents, st.bpick = [], [], {}

    @torch.no_grad()
    def step(self, token: int, st: Hy4EngState) -> torch.Tensor:
        """One decode token: its row, committed. Logits [1, vocab] fp32."""
        lg, _ = self.rows([int(token)], [-1], st)
        self.commit(st, [0])
        return lg

    # ------------------------------------------------------------------------- prefill
    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor, st: Hy4EngState, head: str = "last") -> torch.Tensor | None:
        """A prompt appended in chunks of PREFILL_CHUNK. Logits of the last row [1, vocab] (`head` "last"),
        of every row ("all"), or none; `prompt_hidden` keeps the last chunk's head inputs."""
        tokens = tokens.reshape(-1).to(torch.int64).cpu()
        out = []
        for i in range(0, tokens.shape[0], PREFILL_CHUNK):
            n = self.forward_chunk(tokens[i:i + PREFILL_CHUNK], st)
            if head == "all":
                out.append(self.head(n))
        if head == "all":
            return torch.cat(out)
        return self.head(self.prompt_hidden[-1:]) if head == "last" else None

    def forward_chunk(self, tokens: torch.Tensor, st: Hy4EngState) -> torch.Tensor:
        """`tokens` [T] appended to the state on batched operations -> the head's inputs [T, H]."""
        cfg = self.cfg
        T, L0 = int(tokens.shape[0]), st.length
        if L0 + T > st.cap:
            raise ValueError(f"the state holds {L0} of {st.cap} rows: no room for {T} more")
        slots = torch.arange(L0, L0 + T, device=self.device)
        live = tokens != cfg.engram_pad_id
        for skip in (cfg.image_token_id, cfg.video_token_id):
            if skip is not None:
                live = live & (tokens != skip)
        n_ctx = st.hist.shape[0]
        hist = st.hist.cpu()
        full = torch.cat((hist, tokens))
        grams = torch.stack([full[n_ctx - j: n_ctx - j + T] for j in range(cfg.engram_max_ngram_size)], dim=-1) if self.hashes else None
        h = self.embed(tokens).unsqueeze(1).expand(-1, cfg.hc_mult, -1).contiguous()
        for li in range(cfg.num_hidden_layers):
            p = f"layers.{li}."
            if li in self.hashes:
                rows = self.engram_rows(li, self.hashes[li].ids(grams))
                self.stats["engram_rows"] += rows.shape[0] * rows.shape[1]
                h = h + self.engram_apply(h, rows, li, live)
            x, post = self.hc(h, p + "hc_attn_layer.hc_pre.")
            y = self.attention_chunk(self.rms(x, self.w(p + "input_layernorm.weight")), li, st, slots)
            h = self.inject(h, y, post)
            x, post = self.hc(h, p + "hc_mlp_layer.hc_pre.")
            y = self.mlp_chunk(self.rms(x, self.w(p + "post_attention_layernorm.weight")), li)
            h = self.inject(h, y, post)
        st.hist.copy_(full[-n_ctx:])
        st.length = L0 + T
        st.picks_at = st.length if self.pick_layers else -1
        st.btok, st.bparents, st.bpick = [], [], {}
        self.prompt_hidden = self.final(h)
        self.stats["prefill_rows"] += T
        return self.prompt_hidden

    def attention_chunk(self, x: torch.Tensor, li: int, st: Hy4EngState, slots: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        a, kind, p_ = cfg.attn(li), cfg.layer_types[li], f"layers.{li}.self_attn."
        T, R, L0, rr, Wn = x.shape[0], cfg.kv_lora_rank, st.length, cfg.index_kpool, cfg.sliding_window
        gate = self.lin(x, self.w(p_ + "linear_gate.weight"))
        qres = self.rms(self.lin(x, self.w(p_ + "q_a_proj.weight")), self.w(p_ + "q_a_layernorm.weight"))
        q = self.lin(qres, self.w(p_ + "q_b_proj.weight")).view(T, a.heads, a.qk)
        ckv = self.lin(x, self.w(p_ + "kv_a_proj_with_mqa.weight"))
        cos, sin = self.rope(self.pos3(slots.cpu() + st.delta), kind)
        q = torch.cat((q[..., :a.nope], rotate(q[..., a.nope:], cos[:, None], sin[:, None])), dim=-1)
        lat = torch.cat((self.rms(ckv[:, :R], self.w(p_ + "kv_a_layernorm.weight")), rotate(ckv[:, R:], cos, sin)), dim=-1)
        out = torch.empty(T, a.heads, a.v, dtype=self.dtype, device=self.device)
        if kind == SPARSE:
            st.lat[li][slots] = lat
            self._index_chunk(x, qres, li, st, slots)
            L = L0 + T
            if (L - 1) // rr < cfg.group_topk:                    # every row still reads everything before it
                if li in self.pick_layers:
                    pq = torch.arange(cfg.index_topk, device=self.device)
                    st.picks[li] = (pq, pq <= L - 1)
                for i in range(0, T, SUB):
                    sl = slots[i:i + SUB]
                    n = int(sl[-1]) + 1
                    allowed = torch.arange(n, device=self.device)[None, :] <= sl[:, None]
                    out[i:i + SUB] = self.attend_chunk(q[i:i + SUB], st.lat[li][:n], allowed, li)
            else:
                iq = self.lin(qres, self.w(p_ + "compressed_indexer.query_projection.weight"))
                hw = self.lin(x, self.w(p_ + "compressed_indexer.head_weight_projection.weight"))
                for i in range(0, T, SUB_LIST):
                    sl = slots[i:i + SUB_LIST]
                    pq, valid = self._sparse_positions_chunk(li, st, sl, iq[i:i + SUB_LIST], hw[i:i + SUB_LIST],
                                                             cos[i:i + SUB_LIST], sin[i:i + SUB_LIST])
                    out[i:i + SUB_LIST] = self.attend_lists(q[i:i + SUB_LIST], st.lat[li][pq.clamp(max=L - 1)], valid, li)
                if li in self.pick_layers:
                    st.picks[li] = (pq[-1], valid[-1])
        else:
            # the ring's rows before the chunk in position order, then the chunk's own
            before = torch.arange(L0 - Wn + 1, L0, device=self.device)
            keys = torch.cat((st.win[li][before % Wn], lat))
            kpos = torch.cat((before, slots))
            for i in range(0, T, SUB):
                sl = slots[i:i + SUB]
                lo, hi = i, i + sl.shape[0] + Wn - 1              # keys at positions sl[0] - Wn + 1 .. sl[-1]
                kp = kpos[lo:hi]
                allowed = (kp[None, :] <= sl[:, None]) & (kp[None, :] > sl[:, None] - Wn) & (kp[None, :] >= 0)
                out[i:i + SUB] = self.attend_chunk(q[i:i + SUB], keys[lo:hi], allowed, li)
            st.win[li][slots[-Wn:] % Wn] = lat[-Wn:]
        y = (out * torch.sigmoid(gate).view(T, a.heads, a.v)).reshape(T, a.heads * a.v)
        return self.rms(self.lin(y, self.w(p_ + "o_proj.weight")), self.w(p_ + "post_proj_norm.weight"))

    def attend_chunk(self, q: torch.Tensor, K: torch.Tensor, allowed: torch.Tensor, li: int, w: tuple | None = None) -> torch.Tensor:
        """q [t, heads, qk] over shared latents K [n, R + rope] with `allowed` [t, n] -> [t, heads, v]. `w`: another
        layer's absorbed halves, sinks and head shape (a draft layer reading this model's latents)."""
        wk, wv, sinks, a = w if w is not None else (self.wk[li], self.wv[li], self.sinks[li], self.cfg.attn(li))
        R = self.cfg.kv_lora_rank
        qa = torch.einsum("thd,hdr->thr", q[..., :a.nope], wk).float()
        Kf = K.float()
        s = (qa @ Kf[:, :R].T + q[..., a.nope:].float() @ Kf[:, R:].T) * (a.qk ** -0.5)        # [t, heads, n]
        s = s.masked_fill(~allowed[:, None, :], float("-inf"))
        s = torch.cat((s, sinks.expand(s.shape[0], -1, -1)), dim=-1)
        s = s - s.max(dim=-1, keepdim=True).values
        p = F.softmax(s, dim=-1)[..., :-1]
        mix = (p @ Kf[:, :R]).to(self.dtype)
        return torch.einsum("thr,hrv->thv", mix, wv)

    def attend_lists(self, q: torch.Tensor, K: torch.Tensor, valid: torch.Tensor, li: int) -> torch.Tensor:
        """q [t, heads, qk], each row over its own latents K [t, n, R + rope] with `valid` [t, n]."""
        cfg, a = self.cfg, self.cfg.attn(li)
        R = cfg.kv_lora_rank
        qa = torch.einsum("thd,hdr->thr", q[..., :a.nope], self.wk[li]).float()
        Kf = K.float()
        s = (torch.bmm(qa, Kf[..., :R].transpose(1, 2)) + torch.bmm(q[..., a.nope:].float(), Kf[..., R:].transpose(1, 2))) * (a.qk ** -0.5)
        s = s.masked_fill(~valid[:, None, :], float("-inf"))
        s = torch.cat((s, self.sinks[li].expand(s.shape[0], -1, -1)), dim=-1)
        s = s - s.max(dim=-1, keepdim=True).values
        p = F.softmax(s, dim=-1)[..., :-1]
        mix = torch.bmm(p, Kf[..., :R]).to(self.dtype)
        return torch.einsum("thr,hrv->thv", mix, self.wv[li])

    def _index_chunk(self, x: torch.Tensor, qres: torch.Tensor, li: int, st: Hy4EngState, slots: torch.Tensor) -> None:
        """The chunk's token keys into the state, then the pooled, turned key of every group it closes."""
        cfg, ip = self.cfg, f"layers.{li}.self_attn.compressed_indexer."
        rr, D, rd = cfg.index_kpool, cfg.index_head_dim, cfg.sparse_attn.rope
        st.ik[li][slots] = self.rms(self.lin(x, self.w(ip + "token_key_projection.weight")), self.w(ip + "token_key_norm.weight"))
        L0, L = int(slots[0]), int(slots[-1]) + 1
        g0, g1 = L0 // rr, L // rr                          # the groups whose last row is in the chunk
        if g1 <= g0:
            return
        gk = self.group_keys(st.ik[li][g0 * rr:g1 * rr], li)
        first = torch.arange(g0, g1) * rr
        cos, sin = self.rope(self.pos3(first + st.delta), SPARSE)
        st.ik[li][first.to(self.device) + rr - 1] = torch.cat((gk[:, :D - rd], rotate(gk[:, D - rd:], cos, sin)), dim=-1)

    def _sparse_positions_chunk(self, li: int, st: Hy4EngState, slots: torch.Tensor, iq: torch.Tensor, hw: torch.Tensor,
                                cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The rows each of a chunk's rows reads in a sparse layer [t, index_topk], ascending, and their mask."""
        cfg = self.cfg
        NS, rr = cfg.index_topk, cfg.index_kpool
        nh, D, rd = cfg.index_n_heads, cfg.index_head_dim, cfg.sparse_attn.rope
        t = slots.shape[0]
        qg = slots // rr
        G = int(qg.max())                                  # groups closed before the last row's own
        keys = st.ik[li][rr - 1::rr][:G]
        qi = iq.view(t, nh, D)
        qi = torch.cat((qi[..., :D - rd], rotate(qi[..., D - rd:], cos[:, None], sin[:, None])), dim=-1)
        scores = F.relu(torch.matmul(qi.float(), keys.float().T))
        logit = torch.matmul((hw.float() * (nh ** -0.5 * D ** -0.5)).unsqueeze(-2), scores).squeeze(-2)       # [t, G]
        gid = torch.arange(G, device=self.device)
        logit = logit.masked_fill(gid[None, :] >= qg[:, None], float("-inf"))
        off = torch.arange(rr, device=self.device)
        dense = torch.arange(NS, device=self.device).expand(t, -1)
        if G >= cfg.group_topk - 1:
            top = topk_stable(logit, cfg.group_topk - 1)[1].sort(dim=-1).values
            picked = torch.cat(((top.unsqueeze(-1) * rr + off).flatten(1), qg[:, None] * rr + off), dim=-1)
        else:
            picked = dense
        pq = torch.where((qg < cfg.group_topk)[:, None], dense, picked)
        return pq, pq <= slots[:, None]

    def mlp_chunk(self, x: torch.Tensor, li: int) -> torch.Tensor:
        p = f"layers.{li}.mlp."
        if self.cfg.is_dense_mlp(li):
            return self.rms(self.dense(x, p), self.w(p + "post_fc2_norm.weight"))
        idx, wts = self.route(x, li)
        z = self.lin(x, self.w(p + "routed_expert_down_proj.weight"))
        if li in self.banks:
            routed = torch.cat([self.combine(self.expert_pairs(z[i:i + BLOCK], idx[i:i + BLOCK], li), idx[i:i + BLOCK], wts[i:i + BLOCK])
                                for i in range(0, x.shape[0], BLOCK)])
        else:
            routed = self._experts_grouped(z, idx, wts, li)
        routed = self.lin(routed, self.w(p + "routed_expert_up_proj.weight"))
        return self.rms(routed + self.dense(x, p + "shared_experts."), self.w(p + "combine_output_norm.weight"))

    def _experts_grouped(self, z: torch.Tensor, idx: torch.Tensor, wts: torch.Tensor, li: int) -> torch.Tensor:
        """A chunk's routed experts, an expert's rows together, the pairs added in fp32 and rounded once."""
        p, lim = f"layers.{li}.mlp.experts.", self.cfg.swiglu_limit
        G, U, D = self.w(p + "gate_proj"), self.w(p + "up_proj"), self.w(p + "down_proj")
        acc = torch.zeros(z.shape[0], z.shape[1], dtype=torch.float32, device=z.device)
        for e in torch.unique(idx).tolist():
            rows, slot = torch.where(idx == e)
            zi = z[rows]
            g = F.linear(zi, G.weight(e, z.dtype)).clamp(max=lim)
            u = F.linear(zi, U.weight(e, z.dtype)).clamp(min=-lim, max=lim)
            acc.index_add_(0, rows, F.linear(F.silu(g) * u, D.weight(e, z.dtype)).float() * wts[rows, slot, None])
        return acc.to(self.dtype)

    # ------------------------------------------------------------------------- the load-time check
    @torch.no_grad()
    def self_check(self, prompt: list[int], steps: int = 12, widths=(2, 4, 8), log=print) -> dict:
        """A prompt, then `steps` greedy decode steps recorded; the same tokens replayed as chain blocks of
        every width and as a tree with a wrong branch: every row must be the decode step's bit for bit."""
        st = self.new_state(len(prompt) + steps + BLOCK + 8)
        lg = self.prefill(torch.tensor(prompt), st)
        base, toks, want = st.length, [], []
        nxt = int(lg[0].argmax())
        for _ in range(steps):
            toks.append(nxt)
            lg = self.step(nxt, st)
            want.append(lg[0].clone())
            nxt = int(lg[0].argmax())
        rows, bad = 0, []

        def rewind():
            st2 = self.new_state(st.cap)
            self.prefill(torch.tensor(prompt), st2)
            return st2
        for wdt in widths:
            st2 = rewind()
            for i in range(0, steps, wdt):
                blk = toks[i:i + wdt]
                got, _ = self.rows(blk, list(range(-1, len(blk) - 1)), st2)
                for j in range(len(blk)):
                    rows += 1
                    if not torch.equal(got[j], want[i + j]):
                        bad.append(("chain", wdt, i + j))
                self.commit(st2, list(range(len(blk))))
        # a tree: after the anchor a wrong child with its own child, then the right branch
        st2 = rewind()
        i = 0
        while i + 3 <= steps:
            wrong = (toks[i + 1] + 1) % self.cfg.vocab_size
            got, _ = self.rows([toks[i], wrong, toks[i + 2], toks[i + 1], toks[i + 2]], [-1, 0, 1, 0, 3], st2)
            for j, r in ((0, 0), (1, 3), (2, 4)):
                rows += 1
                if not torch.equal(got[r], want[i + j]):
                    bad.append(("tree", 5, i + j))
            self.commit(st2, [0, 3, 4])
            i += 3
        ok = not bad
        log(f"[hy4] self-check: {rows} rows against {steps} decode steps after {base} prompt rows: "
            + ("every row equal" if ok else f"{len(bad)} rows differ, first {bad[0]}"))
        return {"ok": ok, "rows": rows, "bad": bad[:8], "steps": steps}
