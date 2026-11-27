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

The row pass is the family's one path for a decode token, a draft block or tree and the DFly2 rounds'
verify: `stage` (the host's part: positions, ancestors, the Engram rows) and `run` (the device's, what a
graph captures). A row's bits never depend on the block it is in: the packed projections and the experts
run on the row-invariant EXL3 kernels, every other sum is `mm` (kernels.py: one program a row, every
element its own sum), and the rest is element-wise operations, gathers and a stable sort. So plain decode,
drafted decode and any block size write the same tokens, and the load-time self-check holds the device to
it row by row. Against the reference forward the pass is within rounding, measured, not bit-equal.

  rows(tokens, parents)   the staged block through `run` or its captured graph; only scratch rows written
  commit(path)            the path's scratch rows move into the state, the length moves
  prefill(tokens)         a prompt in chunks on batched library operations: the next token agrees with
                          the row path, the bits do not (a cache restore replays the same chunks, so it
                          gives a cold prefill's bits)
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from engine.models.flashnext.ref import topk_stable
from engine.models.hy4_flash.config import SPARSE, Hy4FlashConfig
from engine.models.hy4_flash import kernels as K_
from engine.models.hy4_flash import native as NV
from engine.models.hy4_flash.engram import fp8_round_trip
from engine.models.hy4_flash.table import gather_packed_many
from engine.models.hy4_flash.ref import Hy4FlashRef, rotate

#: rows a block may hold: the anchor and 31 drafted rows (the row kernels' limit)
BLOCK = 32
#: rows `dense_rows` takes in one call (the m16n8k16 tile's rows); a block past it goes in slices of this many
ROWS_TILE = 16
#: SPD-349: the joint row pass's padded row counts (one graph each and form) and its lanes at most
JOINT_ROWS = (4, 8, 12, 16, 24, 32, 40, 48, 64)
JOINT_LANES = 8
#: SPD-349: sparse indexer group-key buckets (multiples of BJ=64); a pass scores only groups 0..gcap-1
GCAP_BUCKETS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
#: rows a prompt chunk takes (the caches' anchor grid)
PREFILL_CHUNK = 4096
#: rows of a prompt chunk the attention and the experts take at a time
SUB = 256
#: rows of a prompt chunk a sparse layer takes at a time once its rows read lists
SUB_LIST = 32
#: rows of a prompt chunk the routed experts take in one launch (a row's bits do not depend on the batch: the
#: device check compares)
EXPERT_SUB = 256
#: SPD-321: rows from which a prompt chunk's routed experts take the shared prefill kernel
#: (tools/fn_prefill_kernels.moe_grouped, one call for the chunk) instead of the row kernels in EXPERT_SUB launches.
#: The device's crossover (tools/hy4_check.py --only experts-crossover, 41 MoE layers summed, GB10, 2026-11-21): 128 rows
#: on the 3.56 pack, 256 on the 3.08 pack. 128 for both: between 128 and 255 rows the 3.08 pack loses at most 41 ms a
#: chunk on the prefill kernel, where 256 would cost the 3.56 pack up to about 140 ms. The bits are the same either way
EXPERTS_MIN = 128
#: SPD-321: the prefill kernel's item GEMM sums its K range over 4 warps in order (prefill.cu `GG_WK`); the row kernels
#: give the same bits only where they split K over as many (`moe_launch()`'s warps), so only then may a chunk's row
#: count choose between the two
GROUP_WARPS = 4
#: SPD-321: rows an item of the prefill kernel's GEMM takes (32 and 64 gave the same bits, 64 the faster: gate 0)
GROUP_ROWS = 64
#: SPD-319: ids a prompt chunk's table read takes at a time on the reader thread (one table's lock held for one piece)
TABLE_PIECE = 4096


def moe_warps(device) -> int | None:
    """The warps the device's MoE row kernels split K over (its GPU adapter's `moe_launch()`), None where it has none."""
    from tools import gpu as _gpu
    try:
        return int(_gpu.profile(device).moe_launch()[1])
    except Exception:          # noqa: BLE001 -- a device without the MoE row kernels (NoInlineBuild, the CPU)
        return None


def pick_positions(z: torch.Tensor, qg: torch.Tensor, groups: int, rr: int) -> torch.Tensor:
    """SPD-323 (F3): a tile's group scores z fp32 [t, W] (column g: group g; each row's own group qg < W) -> the positions
    each row reads, int32 [t, groups rr], ascending: its `groups - 1` best groups below its own (ties to the lower
    group) and its own, last (the decode step's `_select`). The columns past a row's own group take -inf and its own
    +inf; then the shared block select (tools/fn_seq_kernels.qsa_select: the best TOPK by a stable sort, ascending, 4
    positions each) where the shapes are its own (groups of 4, TOPK of them), else the same selection in torch."""
    from tools import fn_seq_kernels as SK
    t, W = z.shape
    g = torch.arange(W, device=z.device)
    z = z.masked_fill(g[None, :] > qg[:, None], float("-inf")).masked_fill(g[None, :] == qg[:, None], float("inf"))
    if rr == 4 and groups == SK.TOPK:
        sel = torch.empty(t, 4 * groups + 3, dtype=torch.int32, device=z.device)
        SK.qsa_select(z.contiguous(), torch.zeros(t, 3, dtype=torch.int32, device=z.device), sel)
        return sel[:, :4 * groups]
    top = topk_stable(z, groups)[1].sort(dim=-1).values
    return (top.unsqueeze(-1) * rr + torch.arange(rr, device=z.device)).flatten(1).to(torch.int32)


#: what torch's rms_norm adds under the root when no eps is given (the reference's Engram norms), for every float type
ENGRAM_NORM_EPS = float(torch.finfo(torch.float32).eps)


class Hy4EngState:
    """One sequence: the by-slot buffers, the rings, the last tokens, and a block's scratch rows.

    The layers' buffers are slices of one tensor a kind (`lat_all` [sparse layers, cap, 576] and so on), so a
    commit is a handful of copies. `gk` mirrors the indexer keys of closed groups by group ([cap / group, D],
    then a few slots a block's own closed groups take while a row is scored); it is derived from `ik` and
    rebuilt when a length is set by hand (`gk_dirty`). `rb` holds a block's staged inputs and its outputs by
    row count: what a captured graph reads and writes.

    ENG-485: `paged` with the KV pool on (engine/kvpool.py), as Flash-Next's and LFM2.5's states: `lat_all` and
    `ik_all` are each one `PagedBuffer` (row axis 1) of the state's `pages`, reserved as addresses and committed
    by rows before every write and every read past the length (`Hy4FlashServing._ensure`, with the indexer's
    read floor). The rings, the last tokens, the group-key mirror and the scratch stay plain tensors. `graphs`:
    the slot's graph set (engine/models/graph_set.py), whose stream and pool its captures use."""

    def __init__(self, cfg: Hy4FlashConfig, cap: int, dtype, device, pick_layers=(), paged: bool = False):
        self.cfg, self.cap, self.length, self.delta = cfg, int(cap), 0, 0
        R, D, rr = cfg.kv_lora_rank + cfg.sparse_attn.rope, cfg.index_head_dim, cfg.index_kpool
        sp, wn = cfg.sparse_layers, cfg.window_layers

        def z(*shape):
            return torch.zeros(*shape, dtype=dtype, device=device)
        from engine import kvpool as _kvp
        self.pages = None
        self.graphs = None
        self.prompt_hidden = None           # the last prompt chunk's head inputs (`forward_chunk`)
        if paged and _kvp.POOL is not None and self.cap >= _kvp.PAGED_MIN_ROWS and str(device).startswith("cuda"):
            self.pages = _kvp.PagedSet(_kvp.OWNER)

        def by_row(name, *shape):
            if self.pages is None:
                return z(*shape)
            return self.pages.add(_kvp.PagedBuffer(shape, 1, dtype, device, owner=self.pages.owner,
                                                   name=f"hy4.{name}")).tensor
        self.lat_all, self.ik_all = by_row("lat", len(sp), self.cap, R), by_row("ik", len(sp), self.cap, D)
        self.win_all = z(len(wn), cfg.sliding_window, R)
        self.lat = {li: self.lat_all[i] for i, li in enumerate(sp)}
        self.ik = {li: self.ik_all[i] for i, li in enumerate(sp)}
        self.win = {li: self.win_all[i] for i, li in enumerate(wn)}
        self.hist = torch.full((max(cfg.engram_max_ngram_size - 1, 1),), cfg.engram_pad_id, dtype=torch.int64, device=device)
        # a block's rows, written by the row pass and read by `commit`
        self.blat_all = z(cfg.num_hidden_layers, BLOCK, R)
        self.bik_all, self.bgk_all = z(len(sp), BLOCK, D), z(len(sp), BLOCK, D)
        self.blat = {li: self.blat_all[li] for li in range(cfg.num_hidden_layers)}
        self.bik = {li: self.bik_all[i] for i, li in enumerate(sp)}
        self.bgk = {li: self.bgk_all[i] for i, li in enumerate(sp)}
        self.Gc, self.BG = self.cap // rr, BLOCK // rr + 1
        self.gk_all = z(len(sp), self.Gc + self.BG, D)
        self.gk = {li: self.gk_all[i] for i, li in enumerate(sp)}
        self.gk_dirty = False
        # the rows the last committed row read in the layers a draft head follows, good while `picks_at` is the
        # length (a restore or a length set by hand leaves them stale)
        self.pick_layers = sorted(pick_layers)
        self.pick_q = torch.zeros(len(self.pick_layers), cfg.index_topk, dtype=torch.int64, device=device)
        self.pick_ok = torch.zeros(len(self.pick_layers), cfg.index_topk, dtype=torch.bool, device=device)
        self.picks_at = -1
        self.rb: dict = {}
        self.btok: list[int] = []
        self.bparents: list[int] = []
        self._gcap: int = 0
        #: SPD-349: where the last block's scratch rows are: None (this state's own `blat_all`, ...), or a joint pass's
        #: buffers and the block's first row in them (`Hy4FlashEngine.verify_multi`)
        self.jsrc = None
        #: SPD-349: `commit`'s pinned host rows and their device copy (one copy a commit, no pageable transfer)
        self.cm_h = None
        self.cm_d = None
        self.cm_ev = None

    @property
    def picks(self) -> dict:
        return {li: (self.pick_q[i], self.pick_ok[i]) for i, li in enumerate(self.pick_layers)}

    def touched(self) -> None:
        """The rows or the length were set from outside (a restore): the derived parts are rebuilt or dropped."""
        self.gk_dirty, self.picks_at = True, -1
        self.btok, self.bparents = [], []
        self.jsrc = None

    def reset_inplace(self) -> None:
        """A fresh sequence in the same tensors (the by-slot rows stay: nothing reads past the length)."""
        self.length, self.delta = 0, 0
        self.win_all.zero_()
        self.hist.fill_(self.cfg.engram_pad_id)
        self.btok, self.bparents, self.picks_at = [], [], -1

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.lat_all, self.ik_all, self.win_all))


def _mirror(st: Hy4EngState, rr: int) -> None:
    """The group-key mirror rebuilt from the indexer keys after a length was set by hand: the committed groups
    only (ENG-485: a paged `ik_all` holds no pages past the committed rows; Flash-Next's `build_pool` the same).
    A group past the length is never read (`gmax`), so the bits a row reads do not move."""
    G = min(st.Gc, st.length // rr)
    if G:
        st.gk_all[:, :G] = st.ik_all[:, rr - 1::rr][:, :G]
    st.gk_dirty = False


def capture_args(st) -> dict:
    """ENG-485: a capture of this state's graphs on its slot's graph set (its own stream, one memory pool for the
    slot's graphs: they never run at once); without a set, torch's shared stream and a private pool as before."""
    gs = getattr(st, "graphs", None)
    return {} if gs is None else {"pool": gs.pool(), "stream": gs.stream()}


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
        # the value half packed, where the loader kept the projection and the device builds the kernel
        self.vpk = {}
        self.value_packed = True
        if self.kernels and dtype == torch.bfloat16 and NV.available(self.device):
            for li in range(cfg.num_hidden_layers):
                a, blk = cfg.attn(li), W.get(f"layers.{li}.self_attn.kv_b_proj.packed")
                if blk is not None and NV.fits(blk, a.heads, a.nope + a.v, a.nope, a.v):
                    self.vpk[li] = blk
        self.sinks = {li: W[f"layers.{li}.self_attn.learnable_sink_param"].float().to(self.device).reshape(-1, 1)
                      for li in range(cfg.num_hidden_layers)}
        #: the sparse layers whose last row's read list is kept for a draft head
        self.pick_layers = set(cfg.dfly.index_layers) if cfg.dfly is not None else set()
        self._sidx = torch.tensor(cfg.sparse_layers, dtype=torch.int64, device=self.device)
        self._widx = torch.tensor(cfg.window_layers, dtype=torch.int64, device=self.device)
        self._elayers = sorted(self.hashes)
        self._packed = bool(self._elayers) and all(hasattr(tables[li], "gather_packed") for li in self._elayers)
        self._ebias = {li: tables[li].bias.to(self.device) for li in self._elayers} if self._packed else {}
        self._ar: dict = {}
        self._ones: dict = {}
        #: a block's packed projections on `dense_rows` and its experts on the grouped kernels (the same bytes)
        self.block_rows = True
        self.block_grouped = True
        #: SPD-345 (HY4_MOE_AS): a block's grouped experts on the staged, streamed-rows kernels (the same bits)
        self.moe_as = True
        self._gpar_set = False
        #: SPD-345 (HY4_DFLY_FUSED): the DFly2 head's norms, rotary and softmax on the row kernels (the draft's bits
        #: are its own then: a draft never moves the text, the target's verify decides)
        self.dfly_fused = True
        #: SPD-345: the DFly2 head's lone rows on `dense_rows` too (`lin_any`)
        self.rows1 = True
        #: SPD-345: the attention's gate (out * sigmoid(gate)) in one launch, torch's bits
        self.mul_sig = True
        #: SPD-349: `commit` from one pinned row of host numbers (one copy, no device read)
        self.zero_sync_commit = True
        #: SPD-349 (HY4_ATTN_MULTI): a joint verify pass on the joint row pass (`run_multi`: every lane's rows through
        #: each layer at once, a row reading its own lane's committed rows by address), else the per-lane pass before
        self.attn_multi = True
        #: SPD-349 (HY4_JOINT_GRAPH): the joint row pass as CUDA graphs, one a padded row count and form, captured
        #: when the server lets it (`joint_capture_ok`)
        self.joint_graph = True
        #: SPD-349 (HY4_PACK_GRAPH): a packed DFly2 round from its CUDA graph (one a set of heads)
        self.pack_graph = True
        self.joint_capture_ok = True
        self._jp = None
        #: HY4_ROW_TORCH (debugging): the row pass's sums in plain torch, one item a call, instead of the kernel
        self.row_torch = False
        #: the prompt's routed experts on the grouped kernels (serving sets it from HY4_PREFILL_GROUPED)
        self.prefill_grouped = False
        #: SPD-321: a prompt chunk's routed experts on the shared prefill kernel ("group") or on the row kernels ("rows",
        #: the path before; serving sets it from HY4_PREFILL_EXPERTS). `experts_min`: the rows from which "group" takes the
        #: chunk; below them the row kernels, which give the same bits only at the prefill kernel's warp split, so on any
        #: other split every row count takes the prefill kernel
        self.prefill_experts = "group"
        self.experts_min = EXPERTS_MIN if moe_warps(self.device) == GROUP_WARPS else 0
        #: SPD-319: the prompt pass's one table-row reader thread, made at the first chunk
        self._reader = None
        #: SPD-323: the prompt pass on fixed-order kernels (serving sets it from HY4_PREFILL_V2): another record of the
        #: prompt pass's bits; off, the pass before
        self.prefill_v2 = False
        #: SPD-347 (HY4_PREFILL_FAST): the prompt pass on fixed-order kernels without host reads or copies: the
        #: rotary once a chunk, the read lists' bounds from the host's positions, the rotary and norms in place, the
        #: indexer's scores with the rows on `mm`'s tile axis, the Engram rows decoded on the device. Its bits
        self.prefill_fast = True
        #: SPD-347 (HY4_JOINT_PREFILL): a prefill burst's prompts in one pass (joint_prefill.py)
        self.joint_prefill = True
        self.prompt_hidden = None
        self.layer_yield = None
        self.stats = {"rows": 0, "blocks": 0, "prefill_rows": 0, "engram_rows": 0}

    def new_state(self, cap: int | None = None, paged: bool = False) -> Hy4EngState:
        return Hy4EngState(self.cfg, cap or self.max_len, self.dtype, self.device, self.pick_layers, paged=paged)

    def ar(self, n: int) -> torch.Tensor:
        """arange(n) on the device, made once (a captured graph must not build its constants)."""
        t = self._ar.get(n)
        if t is None:
            t = self._ar[n] = torch.arange(n, device=self.device)
        return t

    # ------------------------------------------------------------------------- primitives
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

    def head(self, n: torch.Tensor, w=None) -> torch.Tensor:
        w = self.w("lm_head.weight") if w is None else w
        if self.kernels and hasattr(w, "bf16_cached") and n.dtype == torch.bfloat16:
            return torch.cat([self.head_rows(n[i:i + BLOCK], w=w) for i in range(0, n.shape[0], BLOCK)])
        if w is self.w("lm_head.weight"):
            return super().head(n)
        if not torch.is_tensor(w):
            w = w.bf16_cached()
        return self.lin(n, w)[:, :self.cfg.vocab_size]

    def lin1(self, x: torch.Tensor, w) -> torch.Tensor:
        if self.kernels and hasattr(w, "bf16_cached") and x.dtype == torch.bfloat16:
            if x.shape[-1] < w.K:
                x = F.pad(x, (0, w.K - x.shape[-1]))
            y = self._dense_block(x.contiguous(), w)
            n = self.cuts.get(id(w))
            return y if n is None else y[:, :n]
        return self.lin(x, w)

    def lin_any(self, x: torch.Tensor, w) -> torch.Tensor:
        """SPD-345: an EXL3 projection of rows x [M, K] on the row kernels at every row count, a lone row too on
        `dense_rows` when `rows1` (the DFly2 head's lone and packed rounds then run one kernel: a row's bits are the
        same in any round), else `lin1`'s choice."""
        if not (self.rows1 and self.kernels and x.dtype == torch.bfloat16 and x.shape[0] == 1):
            return self.lin1(x, w)
        from tools import fn_kernels as FK
        if x.shape[-1] < w.K:
            x = F.pad(x, (0, w.K - x.shape[-1]))
        y = FK.dense_rows(x.contiguous(), w)
        n = self.cuts.get(id(w))
        return y if n is None else y[:, :n]

    def _dense_block(self, x: torch.Tensor, w) -> torch.Tensor:
        """x [M <= BLOCK, K] contiguous through an EXL3 projection on the row kernels. A block's rows on the kernel
        that decodes each tile once for up to 16 of them (tools/fn_kernels.dense_rows: every output byte is the
        one-row call's), in 16-row slices past 16 rows (SPD-345: up to 32, a joint verify's and a head's rows);
        a lone row, and every row with `block_rows` off, on `dense`. The same bytes a row either way."""
        from tools import fn_kernels as FK
        M = x.shape[0]
        if not self.block_rows or M <= 1:
            return FK.dense(x, w) if M <= BLOCK else torch.cat([FK.dense(x[i:i + BLOCK], w) for i in range(0, M, BLOCK)])
        if M <= ROWS_TILE:
            return FK.dense_rows(x, w)
        y = torch.empty(M, w.N, dtype=torch.float32 if w.out_fp32 else torch.bfloat16, device=x.device)
        for i in range(0, M, ROWS_TILE):
            j = min(M, i + ROWS_TILE)
            if j - i == 1:
                FK.dense(x[i:j], w, out=y[i:j])
            else:
                FK.dense_rows(x[i:j], w, out=y[i:j])
        return y

    def lin_prompt(self, x: torch.Tensor, w) -> torch.Tensor:
        """R83: a packed projection of the prompt pass in the prompt order (tools/exl3_rows.py: S1 w8 at every row count,
        the skinny GEMM to its CROSS rows and the slab's s1 above, a 3.5-bit projection the slab at every row count), so a
        prompt row's bits do not depend on the chunk it is read in, nor on where a cache resume cut the prompt. The row
        pass and the drafter keep `lin1` and `lin`. A plain weight, and off the kernels, as `lin`."""
        if not (self.kernels and hasattr(w, "bf16_cached") and x.dtype == torch.bfloat16):
            return self.lin(x, w)
        from tools import exl3_rows as R
        x2 = x.reshape(-1, x.shape[-1])
        if x2.shape[-1] < w.K:
            x2 = F.pad(x2, (0, w.K - x2.shape[-1]))
        y = R.linear(x2.contiguous(), w)
        n = self.cuts.get(id(w))
        return (y if n is None else y[:, :n]).view(*x.shape[:-1], -1)

    def bind_prompt_order(self) -> dict:
        """R83: every packed projection `lin_prompt` reads (the trunk's; not the experts' banks, not the head, not
        kv_b_proj's packed value half), checked once a shape in the prompt order (tools/exl3_rows.py `bind`): a shape
        whose kernels are not one order's bits raises `RowOrderError` here, at the load. Where `lin_prompt` does not run
        the prompt order (the CPU, a GPU without the inline extensions: `exl3_matmul` there) nothing runs."""
        from tools import exl3_rows as R
        from tools import gpu as _gpu
        if not (self.kernels and _gpu.profile(self.device).inline("exl3_linear")):
            return {"skipped": "the prompt order does not run on this device (exl3_matmul's path)"}
        n = 0
        for name, w in self.W.items():
            if (name.startswith("layers.") and hasattr(w, "bf16_cached") and ".experts." not in name
                    and not name.endswith(".packed")):
                R.bind(w)
                n += 1
        return {"weights": n, "shapes": len(R._BOUND), "cross": R.CROSS, "failed": len(R.FAILED)}

    def _dense_v2(self, x: torch.Tensor, p: str) -> torch.Tensor:
        """`dense` (the reference's SwiGLU MLP) on `lin_prompt`: the prompt pass's dense MLPs and shared experts."""
        g = F.silu(self.lin_prompt(x, self.w(p + "gate_proj.weight")))
        return self.lin_prompt(g * self.lin_prompt(x, self.w(p + "up_proj.weight")), self.w(p + "down_proj.weight"))

    def pos3(self, positions: torch.Tensor) -> torch.Tensor:
        return positions.reshape(1, -1).expand(3, -1)

    # ------------------------------------------------------------------------- the row pass's sums
    def mm(self, a: torch.Tensor, b: torch.Tensor, bj: int | None = None) -> torch.Tensor:
        """a [B, I, K] @ b [NB, K, J] -> fp32 [B, I, J] with every element its own sum (kernels.py): the one
        way the row pass adds things up, so a row's bits never depend on the block it is in."""
        if self.kernels and K_.HAVE_TRITON and not self.row_torch:
            return K_.mm(a, b, bj)
        return K_.mm_reference(a, b, bj)

    def ones(self, n: int) -> torch.Tensor:
        t = self._ones.get(n)
        if t is None:
            t = self._ones[n] = torch.ones(1, n, 1, dtype=torch.float32, device=self.device)
        return t

    def sumsq(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, K] -> the sum of its squares [B, 1] fp32."""
        return self.mm(x.unsqueeze(1), x.unsqueeze(2)).view(-1, 1)

    def rsum(self, x: torch.Tensor) -> torch.Tensor:
        """x fp32 [B, K] -> its sum [B, 1]."""
        return self.mm(x.unsqueeze(1), self.ones(x.shape[1])).view(-1, 1)

    def linr(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x [M, K] @ w^T for a plain weight [N, K] -> fp32 [M, N]."""
        return self.mm(x.unsqueeze(1), w.t().unsqueeze(0)).view(x.shape[0], w.shape[0])

    def lin_rows(self, x: torch.Tensor, w) -> torch.Tensor:
        """x [M, K] @ w^T in x's dtype: the row-invariant EXL3 kernel for a packed projection (a device), else
        `mm`."""
        if self.kernels and hasattr(w, "bf16_cached") and x.dtype == torch.bfloat16:
            return self.lin1(x, w)
        n = self.cuts.get(id(w))
        if not torch.is_tensor(w):
            w = w.bf16_cached()                 # a packed projection off the kernels (the CPU): its dequantised weight
        if x.shape[-1] < w.shape[1]:
            x = F.pad(x, (0, w.shape[1] - x.shape[-1]))
        y = self.linr(x, w).to(x.dtype)
        return y if n is None else y[:, :n]

    def rms_r(self, x: torch.Tensor, w: torch.Tensor | None, eps: float | None = None) -> torch.Tensor:
        """The RMS norm of rows x [M, K]: fp32, back in x's dtype, times the weight."""
        h = x.float() * torch.rsqrt(self.sumsq(x) / x.shape[-1] + (self.cfg.rms_norm_eps if eps is None else eps))
        h = h.to(x.dtype)
        return h if w is None else w.to(self.dtype) * h

    def hc_r(self, h: torch.Tensor, p: str) -> tuple[torch.Tensor, torch.Tensor]:
        """`hc` for a block's rows h [M, S, H]: the gates from `mm`, the streams added in order."""
        S = self.cfg.hc_mult
        flat = h.flatten(1)
        mixes = self.linr(flat, self.w(p + "hc_fn").float())
        mixes = mixes * torch.rsqrt(self.sumsq(flat) / flat.shape[1] + self.cfg.rms_norm_eps)
        base, scale = self.w(p + "hc_base").float(), self.w(p + "hc_scale").float()
        pre = torch.sigmoid(mixes[:, :S] * scale[0] + base[:S]) + self.cfg.hc_eps
        post = self.cfg.hc_magnitude * torch.sigmoid(mixes[:, S:] * scale[1] + base[S:]) + self.cfg.hc_eps
        x = pre[:, 0, None] * h[:, 0]
        for k in range(1, S):
            x = x + pre[:, k, None] * h[:, k]
        return x.to(h.dtype), post

    def final_r(self, h: torch.Tensor) -> torch.Tensor:
        S = self.cfg.hc_mult
        flat = h.flatten(1)
        mixes = self.linr(flat, self.w("hc_head.hc_head_fn").float())
        mixes = mixes * torch.rsqrt(self.sumsq(flat) / flat.shape[1] + self.cfg.rms_norm_eps)
        pre = torch.sigmoid(mixes * self.w("hc_head.hc_head_scale").float() + self.w("hc_head.hc_head_base").float()) + self.cfg.hc_eps
        x = pre[:, 0, None] * h[:, 0]
        for k in range(1, S):
            x = x + pre[:, k, None] * h[:, k]
        return self.rms_r(x.to(h.dtype), self.w("norm.weight"))

    def engram_r(self, h: torch.Tensor, rows: torch.Tensor, li: int, live: torch.Tensor, lin=None) -> torch.Tensor:
        """`engram_apply` for a block's rows: h [M, S, H], rows [M, heads, head_dim]. `lin`: the projections' sums
        (`linr`; the prompt pass on fixed-order kernels takes `_rows_mm`)."""
        cfg, p = self.cfg, f"layers.{li}.engram_layer."
        linr = lin or self.linr
        M, H, d, nh = h.shape[0], cfg.hidden_size, cfg.engram_head_dim, cfg.engram_heads
        # the reference's norms here are nn.RMSNorm without an eps: torch then adds the fp32 epsilon whatever the
        # input's type (measured on BF16 input; not the BF16 epsilon of 2^-7, which moves a norm of small input)
        eps = ENGRAM_NORM_EPS
        hw = torch.stack([self.w(f"{p}lookup_per_head_norms.{k}.weight") for k in range(nh)]).to(self.dtype)
        emb = (self.rms_r(rows.reshape(M * nh, d), None, eps).view(M, nh, d) * hw).flatten(1)
        gates = []
        for k in range(cfg.hc_mult):
            key = (linr(emb, self.w(f"{p}key_projs.{k}.weight")).to(self.dtype) + self.w(f"{p}key_projs.{k}.bias").to(self.dtype))
            key = self.rms_r(key, self.w(f"{p}norm_key.{k}.weight"), eps)
            query = self.rms_r(h[:, k].contiguous(), self.w(f"{p}norm_query.{k}.weight"), eps)
            g = self.mm(key.unsqueeze(1), query.unsqueeze(2)).view(M) * H ** -0.5
            g = g.abs().clamp_min(1e-6).sqrt() * g.sign()
            gates.append(g.sigmoid().to(self.dtype))
        gates = torch.stack(gates, dim=1).unsqueeze(-1)
        value = linr(emb, self.w(p + "value_proj.weight")).to(self.dtype) + self.w(p + "value_proj.bias").to(self.dtype)
        main = self.w(p + "unified_pre_gamma").to(self.dtype) * self.rms_r(value, None, eps)
        branch = torch.stack([self.w(f"{p}unified_per_hc_scales.{k}").to(self.dtype) * main for k in range(cfg.hc_mult)], dim=1)
        out = gates * main.unsqueeze(1) + branch
        return out.masked_fill(~live[:, None, None], 0.0)

    def route_weights(self, x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, k: int, scale: float,
                      norm_prob: bool, fused: bool = False, lin=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared sigmoid-topk + optional renormalization router (reused by target layers and MoE drafters)."""
        if fused and self.kernels and (k & (k - 1) == 0):
            return K_.router(x, gate_w, bias, k, scale, norm_prob)
        scores = (lin or self.linr)(x, gate_w.float()).sigmoid()
        idx = topk_stable(scores + bias.float(), k)[1]
        wts = scores.gather(1, idx)
        if norm_prob:
            wts = wts / (self.rsum(wts) + 1e-20)
        return idx, wts * scale

    def route_r(self, x: torch.Tensor, li: int, lin=None) -> tuple[torch.Tensor, torch.Tensor]:
        cfg, p = self.cfg, f"layers.{li}.mlp.gate."
        return self.route_weights(x, self.w(p + "weight"), self.w(p + "e_score_correction_bias"),
                                  cfg.num_experts_per_tok, cfg.routed_scaling_factor, cfg.norm_topk_prob, lin=lin)

    def group_keys_r(self, tks: torch.Tensor, li: int) -> torch.Tensor:
        """Token keys [M, r, D] of the group that ends at each row -> its pooled key [M, D] (before the rotary)."""
        cfg, p = self.cfg, f"layers.{li}.self_attn.compressed_indexer."
        if self.fused:
            M, rr, D = tks.shape
            flat = tks.reshape(M * rr, D)
            values = K_.gemv(flat, self.w(p + "key_value_projection.weight")).view(M, rr, D)
            gates = K_.gemv(flat, self.w(p + "key_gate_projection.weight")).view(M, rr, D) + self.w(p + "intra_group_ape").float()
            e = torch.exp(gates - gates.amax(dim=1, keepdim=True))
            z, pooled = e[:, 0], values[:, 0] * e[:, 0]
            for k in range(1, rr):
                z = z + e[:, k]
                pooled = pooled + values[:, k] * e[:, k]
            return K_.norm((pooled / z).contiguous(), self.w(p + "key_norm.weight").to(self.dtype), cfg.rms_norm_eps)
        g = tks.float()
        values = self.mm(g, self.w(p + "key_value_projection.weight").float().t().unsqueeze(0))
        gates = self.mm(g, self.w(p + "key_gate_projection.weight").float().t().unsqueeze(0)) + self.w(p + "intra_group_ape").float()
        e = torch.exp(gates - gates.amax(dim=1, keepdim=True))
        z, pooled = e[:, 0], values[:, 0] * e[:, 0]
        for k in range(1, cfg.index_kpool):
            z = z + e[:, k]
            pooled = pooled + values[:, k] * e[:, k]
        pooled = pooled / z
        pooled = pooled * torch.rsqrt(self.sumsq(pooled) / pooled.shape[1] + cfg.rms_norm_eps)
        return (self.w(p + "key_norm.weight").to(self.dtype) * pooled).to(tks.dtype)

    def attend_r(self, q: torch.Tensor, K: torch.Tensor, valid: torch.Tensor, li: int) -> torch.Tensor:
        """q [M, heads, qk] (turned), each row over the latents it reads K [M, n, R + rope] with valid [M, n]
        -> [M, heads, v]. The query takes the key half, the weighted latents go through the value half; the
        sink is one more logit."""
        cfg, a = self.cfg, self.cfg.attn(li)
        M, n, R = K.shape[0], K.shape[1], cfg.kv_lora_rank
        if self.fused and M > 1:
            # SPD-349: pack M rows onto mm's tile axis so wk[li] is read once across the block (0-bit diff)
            qa = self.mm(q[..., :a.nope].transpose(0, 1), self.wk[li]).to(self.dtype).transpose(0, 1)
        else:
            qa = self.mm(q[..., :a.nope].reshape(M * a.heads, 1, a.nope), self.wk[li]).view(M, a.heads, R).to(self.dtype)
        s = (self.mm(qa, K[..., :R].transpose(1, 2)) + self.mm(q[..., a.nope:], K[..., R:].transpose(1, 2))) * (a.qk ** -0.5)
        if self.fused:
            p = K_.softmax_sink(s, valid, self.sinks[li])
        else:
            s = torch.cat((s.masked_fill(~valid[:, None, :], float("-inf")), self.sinks[li].expand(M, -1, -1)), dim=-1)
            e = torch.exp(s - s.amax(dim=-1, keepdim=True))
            p = (e / self.rsum(e.view(M * a.heads, n + 1)).view(M, a.heads, 1))[..., :n].to(self.dtype)
        mix = self.mm(p, K[..., :R]).to(self.dtype)
        if self.fused and self.value_packed and li in self.vpk:
            return NV.vdense(mix.reshape(M * a.heads, R).contiguous(), self.vpk[li], a.heads, a.nope + a.v, a.nope, a.v).view(M, a.heads, a.v)
        if self.fused and M > 1:
            return self.mm(mix.transpose(0, 1), self.wv[li]).transpose(0, 1).to(self.dtype)
        return self.mm(mix.reshape(M * a.heads, 1, R), self.wv[li]).view(M, a.heads, a.v).to(self.dtype)

    def attend_lanes(self, q: torch.Tensor, Kl: torch.Tensor, valid: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor | None,
                     sinks: torch.Tensor, a, B: int, Q: int, latent: bool = False) -> torch.Tensor:
        """SPD-345: B lanes of Q rows, each lane's rows over its own keys Kl [B, n, R + rope] (valid [B Q, n]), on `mm`
        and the fused softmax: q [B Q, heads, qk] (turned) -> [B Q, heads, v] bf16 (`latent`: the weighted latents
        [B Q, heads, R] fp32, before the value half). A lane's rows sit on `mm`'s tile axis (Q x heads of them), the
        absorbed halves take the rows on it a head, so an element's bits never depend on B, Q or the other lanes:
        one lane alone and among eight give the same bits (the DFly2 head's lone and packed rounds)."""
        R, n = self.cfg.kv_lora_rank, Kl.shape[1]
        H = a.heads
        qa = self.mm(q[..., :a.nope].transpose(0, 1), wk).to(self.dtype).transpose(0, 1).reshape(B, Q * H, R)
        s = self.mm(qa, Kl[..., :R].transpose(1, 2)) + self.mm(q[..., a.nope:].reshape(B, Q * H, a.rope), Kl[..., R:].transpose(1, 2))
        p = K_.softmax_sink((s * (a.qk ** -0.5)).view(B * Q, H, n), valid, sinks)
        mix = self.mm(p.view(B, Q * H, n), Kl[..., :R]).view(B * Q, H, R)
        if latent:
            return mix
        return self.mm(mix.to(self.dtype).transpose(0, 1), wv).transpose(0, 1).to(self.dtype)

    def _gather(self, by_slot: torch.Tensor, scratch: torch.Tensor, q: torch.Tensor, Lt: torch.Tensor, anc: torch.Tensor) -> torch.Tensor:
        """Rows at positions q [M, n]: committed ones from the by-slot buffer, the block's from its scratch rows
        (`anc` [M, BLOCK]: each row's ancestors by depth, itself last and repeated)."""
        out = by_slot[q.clamp(0, by_slot.shape[0] - 1)]
        pr = anc.gather(1, (q - Lt).clamp(0, BLOCK - 1))
        return torch.where((q >= Lt).unsqueeze(-1), scratch[pr], out)

    def _gcap(self, select: bool, max_gmax: int, Gc: int) -> int:
        """SPD-349: bucketed group-key width for `_select` / `_select_m` (0 when no row selects; else the smallest
        multiple of BJ=64 in GCAP_BUCKETS covering every row's committed groups, clamped to Gc). Groups past gcap are
        all -inf-masked, so restricting the score/sort width to gcap is 100% bit-equal."""
        if not select:
            return 0
        need = max(self.cfg.group_topk, int(max_gmax))
        return min(Gc, next((b for b in GCAP_BUCKETS if b >= need), Gc))

    def _select(self, li: int, st: Hy4EngState, iq: torch.Tensor, hw: torch.Tensor, cos, sin, c: dict) -> torch.Tensor:
        """The positions rows past `index_topk` read in a sparse layer [M, index_topk], ascending: the 4 rows of
        each of a row's best `group_topk - 1` closed groups, then its own group. Committed groups are scored
        from `gk`, the groups a row's ancestors closed in this block from their scratch keys; both products
        take the same tile, so a group scores the same before and after its commit."""
        cfg = self.cfg
        rr, BG = cfg.index_kpool, st.BG
        gcap = c.get("gcap") or st.Gc
        nh, D, rd = cfg.index_n_heads, cfg.index_head_dim, cfg.sparse_attn.rope
        M = iq.shape[0]
        if self.fused:
            qi = iq.reshape(M, nh, D).contiguous()
            K_.rot_exact(qi, cos.contiguous(), sin.contiguous(), D - rd)
            sc_c = self.mm(qi.reshape(1, M * nh, D), st.gk[li][:gcap].t().unsqueeze(0), 64).view(M, nh, gcap)
        else:
            qi = iq.view(M, nh, D)
            qi = torch.cat((qi[..., :D - rd], rotate(qi[..., D - rd:], cos[:, None], sin[:, None])), dim=-1)
            sc_c = self.mm(qi, st.gk[li][:gcap].t().unsqueeze(0), 64)
        sc = torch.cat((sc_c, self.mm(qi, st.bgk[li][c["cidx"]].transpose(1, 2), 64)), dim=-1)    # [M, nh, gcap + BG]
        if self.fused and nh & (nh - 1) == 0:
            logit = K_.wsum_relu(sc, (hw.float() * (nh ** -0.5 * D ** -0.5)).contiguous())
        else:
            logit = self.mm((hw.float() * (nh ** -0.5 * D ** -0.5)).unsqueeze(1), F.relu(sc)).view(M, gcap + BG)
        gid = self.ar(gcap + BG)
        ok = (gid[None, :] < c["gmax"][:, None]) | ((gid[None, :] >= gcap) & c["bok"].gather(1, (gid - gcap).clamp(0, BG - 1).expand(M, -1)))
        top = topk_stable(logit.masked_fill(~ok, float("-inf")), cfg.group_topk - 1)[1]
        groups = torch.where(top < gcap, top, c["bgid"].gather(1, (top - gcap).clamp(0, BG - 1))).sort(dim=-1).values
        off = self.ar(rr)
        return torch.cat(((groups.unsqueeze(-1) * rr + off).flatten(1), c["qg"][:, None] * rr + off), dim=-1)

    # ------------------------------------------------------------------------- a block's rows
    #: the staged control columns of a row: its token, position, the state's length, its group, the committed
    #: groups it may read, its group key's first position, whether it owns an n-gram; then its ancestors by depth,
    #: and the groups its ancestors closed in the block (their rows, their ids, which slots are real)
    CTL = 7

    def _rb(self, st: Hy4EngState, M: int) -> dict:
        rb = st.rb.get(M)
        if rb is None:
            cfg, dev = self.cfg, self.device
            C = self.CTL + BLOCK + 3 * st.BG
            rb = {"ctl_h": torch.zeros(M, C, dtype=torch.int64), "ctl": torch.zeros(M, C, dtype=torch.int64, device=dev),
                  "logits": torch.zeros(M, cfg.vocab_size, dtype=torch.float32, device=dev),
                  "hidden": torch.zeros(M, cfg.hidden_size, dtype=self.dtype, device=dev),
                  "next": torch.zeros(M, dtype=torch.int64, device=dev),
                  "pick_q": torch.zeros(len(st.pick_layers), M, cfg.index_topk, dtype=torch.int64, device=dev),
                  "pick_ok": torch.zeros(len(st.pick_layers), M, cfg.index_topk, dtype=torch.bool, device=dev)}
            nE, heads = len(self._elayers), cfg.engram_heads
            if self._packed:
                words = self.tables[self._elayers[0]].words
                rb["epk_h"] = torch.zeros(nE, M * heads, words, dtype=torch.int16)
                rb["epk"] = torch.zeros(nE, M * heads, words, dtype=torch.int16, device=dev)
                rb["ehead"] = self.ar(heads).repeat(M)
            elif nE:
                rb["erows"] = torch.zeros(nE, M, heads, cfg.engram_head_dim, dtype=self.dtype, device=dev)
            st.rb[M] = rb
        return rb

    def stage(self, tokens: list[int], parents: list[int], st: Hy4EngState) -> tuple[int, bool]:
        """A block in DFS pre-order (row 0 at the state's length) into the state's row buffers: everything the
        host decides (positions, ancestors, the block's closed groups, the Engram rows read from the tables).
        Returns the row count and whether a row selects groups (the two forms of the row pass)."""
        cfg = self.cfg
        M = len(tokens)
        if not 1 <= M <= BLOCK or len(parents) != M:
            raise ValueError(f"a block holds 1 to {BLOCK} rows")
        rows, select, paths = self._stage_rows(tokens, parents, st)
        st.jsrc = None
        st._gcap = self._gcap(select, max(r[4] for r in rows), st.Gc)
        rb = self._rb(st, M)
        c = rb["ctl_h"]
        for r in range(M):
            c[r] = torch.tensor(rows[r], dtype=torch.int64)
        rb["ctl"].copy_(c)
        st.btok, st.bparents = [int(t) for t in tokens], [int(p) for p in parents]
        if self._elayers:
            grams = self._row_grams(st, st.btok, paths)
            if self._packed:
                # every table's missed rows are with the device before any is waited for
                got = gather_packed_many([self.tables[li] for li in self._elayers], [self.hashes[li].ids(grams) for li in self._elayers])
                for k, rows_ in enumerate(got):
                    rb["epk_h"][k] = rows_
                rb["epk"].copy_(rb["epk_h"])
            else:
                for k, li in enumerate(self._elayers):
                    ids = self.hashes[li].ids(grams)
                    rb["erows"][k].copy_(self.tables[li].gather(ids.reshape(-1), dtype=self.dtype).view(M, cfg.engram_heads, -1))
            self.stats["engram_rows"] += M * cfg.engram_heads * len(self._elayers)
        return M, bool(select)

    def _stage_rows(self, tokens: list[int], parents: list[int], st: Hy4EngState) -> tuple:
        """The host's part of `stage` for a block: each row's control columns (`CTL`), whether a row selects groups, and
        the rows' paths from the anchor."""
        cfg = self.cfg
        M = len(tokens)
        depth, paths = _paths([int(p) for p in parents])
        L, rr = st.length, cfg.index_kpool
        if L + max(depth) >= st.cap:
            raise ValueError(f"the state holds {L} of {st.cap} rows: no room for this block")
        if st.gk_dirty:
            _mirror(st, rr)
        BG = st.BG
        select = False
        out = []
        for r in range(M):
            pos, path, t = L + depth[r], paths[r], int(tokens[r])
            qg = pos // rr
            select = select or qg >= cfg.group_topk
            live = t != cfg.engram_pad_id and t not in (cfg.image_token_id, cfg.video_token_id)
            closed = [a for a in path[:-1] if (L + depth[a]) % rr == rr - 1]
            row = [t, pos, L, qg, min(qg, L // rr), pos - rr + 1, int(live)]
            row += path + [path[-1]] * (BLOCK - len(path))
            row += closed + [0] * (BG - len(closed))
            row += [(L + depth[a]) // rr for a in closed] + [0] * (BG - len(closed))
            row += [1] * len(closed) + [0] * (BG - len(closed))
            out.append(row)
        return out, bool(select), paths

    def _engram_staged(self, rb: dict, k: int, li: int, M: int) -> torch.Tensor:
        """The block's table rows of Engram layer li [M, heads, head_dim], from what `stage` read."""
        cfg = self.cfg
        if self._packed:
            from engine.models.hy4_flash.table import decode_rows
            t = self.tables[li]
            rows = decode_rows(rb["epk"][k], rb["ehead"], self._ebias[li], t.K, t.codebook(self.device), t.dim)
            return rows.to(self.dtype).view(M, cfg.engram_heads, -1)
        rows = rb["erows"][k]
        if cfg.engram_fp8_embedding and not getattr(self.tables[li], "fp8_done", False):
            rows = fp8_round_trip(rows)
        return rows

    def _build_c(self, st: Hy4EngState, rb: dict) -> dict:
        cfg, ctl, o, BG = self.cfg, rb["ctl"], self.CTL, st.BG
        Wn, rr = cfg.sliding_window, cfg.index_kpool
        Lt, p, g0 = ctl[0, 2], ctl[:, 1], ctl[:, 5]
        anc = ctl[:, o:o + BLOCK]
        wq = p[:, None] - Wn + 1 + self.ar(Wn)
        return {
            "p": p, "anc": anc, "Lt": Lt, "qg": ctl[:, 3], "gmax": ctl[:, 4],
            "wsrc": torch.where(wq < Lt, torch.remainder(wq, Wn), Wn + anc.gather(1, (wq - Lt).clamp(0, BLOCK - 1))),
            "wval": wq >= 0, "gq": g0[:, None] + self.ar(rr),
            "cidx": ctl[:, o + BLOCK:o + BLOCK + BG], "bgid": ctl[:, o + BLOCK + BG:o + BLOCK + 2 * BG],
            "bok": ctl[:, o + BLOCK + 2 * BG:] != 0,
            "rope": {kind: self.rope(self.pos3(p + st.delta), kind) for kind in set(cfg.layer_types)},
            "grope": self.rope(self.pos3(g0 + st.delta), SPARSE),
        }

    @torch.no_grad()
    def run(self, st: Hy4EngState, M: int, select: bool, gcap: int | None = None) -> None:
        """The staged block's rows on the device: logits, head inputs and greedy picks into the row buffers,
        the rows' latents and indexer keys into the scratch rows. No host decision and no host read: this is
        what a graph captures, once for a row count and a form. Every sum is `mm`'s, everything else is
        element-wise, a gather or a sort, so a row's bits are the same in any block."""
        cfg, rb = self.cfg, st.rb[M]
        c = self._build_c(st, rb)
        gc = (st._gcap if gcap is None else int(gcap)) if select else 0
        if select and gc <= 0:
            gc = st.Gc
        c["gcap"] = gc
        live = rb["ctl"][:, 6] != 0
        h = self.embed(rb["ctl"][:, 0]).unsqueeze(1).expand(-1, cfg.hc_mult, -1).contiguous()
        if self.fused:
            n = self._run_fused(st, M, select, c, rb, live, h)
            lg = self.head_rows(n)
            rb["hidden"].copy_(n)
            rb["logits"].copy_(lg)
            rb["next"].copy_(lg.argmax(dim=-1))
            return
        # the device takes the block's rows together: its element-wise kernels compute every element alone. A
        # CPU's vector math does not (a value's bits can depend on where it sits in the vector), so there, and
        # under HY4_ROW_TORCH, the same code takes one row at a time
        parts = [slice(0, M)] if (self.kernels and not self.row_torch) else [slice(r, r + 1) for r in range(M)]
        cs = [{k: ((v[0][sl], v[1][sl]) if k == "grope" else {kk: (vv[0][sl], vv[1][sl]) for kk, vv in v.items()} if k == "rope"
                   else v if k in ("Lt", "gcap") else v[sl]) for k, v in c.items()} for sl in parts]

        def rows(fn):
            out = [fn(sl, cc) for sl, cc in zip(parts, cs)]
            return out[0] if len(out) == 1 else torch.cat(out)
        for li in range(cfg.num_hidden_layers):
            pre = f"layers.{li}."
            if li in self.hashes:
                er = self._engram_staged(rb, self._elayers.index(li), li, M)
                h = h + rows(lambda sl, cc: self.engram_r(h[sl], er[sl], li, live[sl]))
            xp = [self.hc_r(h[sl], pre + "hc_attn_layer.hc_pre.") for sl in parts]
            y = rows(lambda sl, cc: self.attention_rows(self.rms_r(xp[parts.index(sl)][0], self.w(pre + "input_layernorm.weight")),
                                                        li, st, sl, select, cc))
            h = self.inject(h, y, torch.cat([q[1] for q in xp]))
            xp = [self.hc_r(h[sl], pre + "hc_mlp_layer.hc_pre.") for sl in parts]
            y = rows(lambda sl, cc: self.mlp_rows(self.rms_r(xp[parts.index(sl)][0], self.w(pre + "post_attention_layernorm.weight")), li))
            h = self.inject(h, y, torch.cat([q[1] for q in xp]))
        n = rows(lambda sl, cc: self.final_r(h[sl]))
        lg = rows(lambda sl, cc: self.head_rows(n[sl]))
        rb["hidden"].copy_(n)
        rb["logits"].copy_(lg)
        rb["next"].copy_(lg.argmax(dim=-1))

    @property
    def fused(self) -> bool:
        """The served row pass: the fused row kernels (a device, bf16, 4 streams and a power-of-two expert count a
        token, which the kernels' tiles need). Otherwise the same pass on `mm` (the fp32 tests, the debugging knob)."""
        c = self.cfg
        return (self.kernels and K_.HAVE_TRITON and not self.row_torch and self.dtype == torch.bfloat16
                and c.hc_mult & (c.hc_mult - 1) == 0 and c.num_experts_per_tok & (c.num_experts_per_tok - 1) == 0)

    @property
    def _ms(self) -> bool:
        return self.mul_sig and self.fused and K_.mul_sigmoid_ok()

    def _hcw(self, p: str) -> tuple:
        return self.w(p + "hc_fn"), self.w(p + "hc_base"), self.w(p + "hc_scale")

    def _run_fused(self, st: Hy4EngState, M: int, select: bool, c: dict, rb: dict, live: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """The layers on the fused kernels: a hyper-connection with the inject before it and the norm after it is
        three launches, a norm one, the router two. -> the head's inputs [M, H]."""
        cfg = self.cfg
        eps, he, mag = cfg.rms_norm_eps, cfg.hc_eps, cfg.hc_magnitude
        sl, pend = slice(0, M), None
        for li in range(cfg.num_hidden_layers):
            pre = f"layers.{li}."
            if li in self.hashes:
                if pend is not None:
                    h, pend = K_.inject(h, pend[0], pend[1]), None
                h = (h + self.engram_r(h, self._engram_staged(rb, self._elayers.index(li), li, M), li, live)).contiguous()
            x, post, h = K_.hc(h, pend, *self._hcw(pre + "hc_attn_layer.hc_pre."), self.w(pre + "input_layernorm.weight"), eps, he, mag)
            y = self.attention_fused(x, li, st, sl, select, c)
            x, post, h = K_.hc(h, (y, post), *self._hcw(pre + "hc_mlp_layer.hc_pre."), self.w(pre + "post_attention_layernorm.weight"), eps, he, mag)
            pend = (self.mlp_fused(x, li), post)
        return K_.hc(h, pend, self.w("hc_head.hc_head_fn"), self.w("hc_head.hc_head_base"), self.w("hc_head.hc_head_scale"),
                     self.w("norm.weight"), eps, he, mag, head=True)[0]

    def attention_fused(self, x: torch.Tensor, li: int, st: Hy4EngState, sl: slice, select: bool, c: dict) -> torch.Tensor:
        cfg = self.cfg
        a, kind, p_ = cfg.attn(li), cfg.layer_types[li], f"layers.{li}.self_attn."
        R, NS, Lt, M, eps = cfg.kv_lora_rank, cfg.index_topk, c["Lt"], x.shape[0], cfg.rms_norm_eps
        rbm = st.rb[len(st.bparents)]
        gate = self.lin1(x, self.w(p_ + "linear_gate.weight"))
        qres = K_.norm(self.lin1(x, self.w(p_ + "q_a_proj.weight")), self.w(p_ + "q_a_layernorm.weight"), eps)
        q = self.lin1(qres, self.w(p_ + "q_b_proj.weight")).contiguous().view(M, a.heads, a.qk)
        ckv = self.lin1(x, self.w(p_ + "kv_a_proj_with_mqa.weight")).contiguous()
        cos, sin = c["rope"][kind]
        K_.rot(q, cos, sin, a.nope)
        K_.norm(ckv[:, :R], self.w(p_ + "kv_a_layernorm.weight"), eps, out=ckv[:, :R])
        K_.rot(ckv.view(M, 1, R + a.rope), cos, sin, R)
        st.blat[li][sl] = ckv
        if kind == SPARSE:
            ip = p_ + "compressed_indexer."
            D, rd = cfg.index_head_dim, a.rope
            st.bik[li][sl] = K_.norm(K_.gemv(x, self.w(ip + "token_key_projection.weight"), bf16=True), self.w(ip + "token_key_norm.weight"), eps)
            gk = self.group_keys_r(self._gather(st.ik[li], st.bik[li], c["gq"], Lt, c["anc"]), li).contiguous()
            c0, s0 = c["grope"]
            K_.rot(gk.view(M, 1, D), c0, s0, D - rd)
            st.bgk[li][sl] = gk
            pq = self.ar(NS).expand(M, -1)
            if select:
                iq = self.lin_rows(qres, self.w(ip + "query_projection.weight"))
                hw = K_.gemv(x, self.w(ip + "head_weight_projection.weight"), bf16=True)
                pq = torch.where(c["qg"][:, None] < cfg.group_topk, pq, self._select(li, st, iq, hw, cos, sin, c))
            valid = pq <= c["p"][:, None]
            if li in st.pick_layers:
                pi = st.pick_layers.index(li)
                rbm["pick_q"][pi, sl].copy_(pq)
                rbm["pick_ok"][pi, sl].copy_(valid)
            Kr = self._gather(st.lat[li], st.blat[li], pq, Lt, c["anc"])
        else:
            Kr, valid = torch.cat((st.win[li], st.blat[li]))[c["wsrc"]], c["wval"]
        y = K_.mul_sigmoid(self.attend_r(q, Kr, valid, li).reshape(M, -1), gate) if self._ms else \
            (self.attend_r(q, Kr, valid, li).reshape(M, -1) * torch.sigmoid(gate))
        return K_.norm(self.lin1(y, self.w(p_ + "o_proj.weight")), self.w(p_ + "post_proj_norm.weight"), eps)

    def mlp_fused(self, x: torch.Tensor, li: int) -> torch.Tensor:
        cfg, p, eps = self.cfg, f"layers.{li}.mlp.", self.cfg.rms_norm_eps
        if cfg.is_dense_mlp(li):
            g = F.silu(self.lin1(x, self.w(p + "gate_proj.weight")))
            y = self.lin1(g * self.lin1(x, self.w(p + "up_proj.weight")), self.w(p + "down_proj.weight"))
            return K_.norm(y, self.w(p + "post_fc2_norm.weight"), eps)
        idx, wts = self.route_weights(x, self.w(p + "gate.weight"), self.w(p + "gate.e_score_correction_bias"),
                                      cfg.num_experts_per_tok, cfg.routed_scaling_factor, cfg.norm_topk_prob, fused=True)
        z = self.lin1(x, self.w(p + "routed_expert_down_proj.weight"))
        routed = self.lin1(self.combine(self.expert_pairs(z, idx, li, grouped=self.block_grouped), idx, wts), self.w(p + "routed_expert_up_proj.weight"))
        sp = p + "shared_experts."
        g = F.silu(self.lin1(x, self.w(sp + "gate_proj.weight")))
        shared = self.lin1(g * self.lin1(x, self.w(sp + "up_proj.weight")), self.w(sp + "down_proj.weight"))
        return K_.norm(routed + shared, self.w(p + "combine_output_norm.weight"), eps)

    def attention_rows(self, x: torch.Tensor, li: int, st: Hy4EngState, sl: slice, select: bool, c: dict) -> torch.Tensor:
        """The attention of the block's rows `sl` (all of them on a device); their scratch rows are written."""
        cfg = self.cfg
        a, kind, p_ = cfg.attn(li), cfg.layer_types[li], f"layers.{li}.self_attn."
        R, NS, Lt, M = cfg.kv_lora_rank, cfg.index_topk, c["Lt"], x.shape[0]
        rbm = st.rb[len(st.bparents)]
        gate = self.lin_rows(x, self.w(p_ + "linear_gate.weight"))
        qres = self.rms_r(self.lin_rows(x, self.w(p_ + "q_a_proj.weight")), self.w(p_ + "q_a_layernorm.weight"))
        q = self.lin_rows(qres, self.w(p_ + "q_b_proj.weight")).view(M, a.heads, a.qk)
        ckv = self.lin_rows(x, self.w(p_ + "kv_a_proj_with_mqa.weight"))
        cos, sin = c["rope"][kind]
        q = torch.cat((q[..., :a.nope], rotate(q[..., a.nope:], cos[:, None], sin[:, None])), dim=-1)
        st.blat[li][sl] = torch.cat((self.rms_r(ckv[:, :R].contiguous(), self.w(p_ + "kv_a_layernorm.weight")),
                                     rotate(ckv[:, R:], cos, sin)), dim=-1)
        if kind == SPARSE:
            ip = p_ + "compressed_indexer."
            D, rd = cfg.index_head_dim, a.rope
            st.bik[li][sl] = self.rms_r(self.lin_rows(x, self.w(ip + "token_key_projection.weight")), self.w(ip + "token_key_norm.weight"))
            # the pooled key of the group that ends at each row, turned at its first row (read only where the
            # row closes a group: by `commit` and by the rows after it in the block)
            gk = self.group_keys_r(self._gather(st.ik[li], st.bik[li], c["gq"], Lt, c["anc"]), li)
            c0, s0 = c["grope"]
            st.bgk[li][sl] = torch.cat((gk[:, :D - rd], rotate(gk[:, D - rd:], c0, s0)), dim=-1)
            pq = self.ar(NS).expand(M, -1)
            if select:
                iq = self.lin_rows(qres, self.w(ip + "query_projection.weight"))
                hw = self.lin_rows(x, self.w(ip + "head_weight_projection.weight"))
                pq = torch.where(c["qg"][:, None] < cfg.group_topk, pq, self._select(li, st, iq, hw, cos, sin, c))
            valid = pq <= c["p"][:, None]
            if li in st.pick_layers:
                pi = st.pick_layers.index(li)
                rbm["pick_q"][pi, sl].copy_(pq)
                rbm["pick_ok"][pi, sl].copy_(valid)
            K = self._gather(st.lat[li], st.blat[li], pq, Lt, c["anc"])
        else:
            K, valid = torch.cat((st.win[li], st.blat[li]))[c["wsrc"]], c["wval"]
        y = (self.attend_r(q, K, valid, li).reshape(M, -1) * torch.sigmoid(gate))
        return self.rms_r(self.lin_rows(y, self.w(p_ + "o_proj.weight")), self.w(p_ + "post_proj_norm.weight"))

    def expert_pairs_bank(self, z: torch.Tensor, idx: torch.Tensor, bank=None, experts: tuple | None = None,
                          lim: float | None = None, grouped: bool = False) -> torch.Tensor:
        """Every (row, slot) pair through an explicit `MoEBank` or `(G, U, D)` bank: [M * k, latent] fp32."""
        lim = self.cfg.swiglu_limit if lim is None else lim
        M, k = idx.shape
        if bank is not None:
            from tools import fn_kernels as FK
            mod = FK._module()
            flat = idx.reshape(-1).to(torch.int32).contiguous()
            order = FK.expert_order(idx.to(torch.int32)) if M > 1 else torch.empty(0, dtype=torch.int32, device=z.device)
            ygu = torch.empty(M * k, 2, bank.I, dtype=torch.bfloat16, device=z.device)
            yd = torch.empty(M * k, bank.H, dtype=torch.float32, device=z.device)
            if grouped and M > 1:
                gt = torch.empty(M * k, 2 + FK.MOE_GR, dtype=torch.int32, device=z.device)
                ng = torch.empty(1, dtype=torch.int32, device=z.device)
                if self.moe_as and not self._gpar_set:
                    mod.fn_set_moe_gpar(1)          # the groups table from one block of 1024 threads: the same groups
                    self._gpar_set = True
                mod.fn_moe_groups(flat, order, gt, ng)
                if self.moe_as:
                    # SPD-345 (HY4_MOE_AS): every pair's gate|up rows rotated once (moe_gu_stage) and streamed beside
                    # the tiles, the clamp on the gate|up buffer, the down rows rotated once (moe_down_stage), then
                    # the streamed down kernel: the grouped kernels' products in their order (the device check)
                    zc = z.contiguous()
                    xs = torch.empty(M * k, 2, bank.H, dtype=torch.float16, device=z.device)
                    mod.fn_moe_gu_stage(zc, flat, k, bank.tr, bank.kb, bank.su, bank.sv, xs)
                    mod.fn_moe_gu_group_s(gt, ng, bank.tr, bank.kb, bank.su, bank.sv, xs, ygu, bank.ks[-1])
                    if lim > 0:
                        ygu[:, 0].clamp_(max=lim)
                        ygu[:, 1].clamp_(min=-lim, max=lim)
                    xd = torch.empty(M * k, bank.I, dtype=torch.float16, device=z.device)
                    mod.fn_moe_down_stage(ygu, flat, bank.tr, bank.kb, bank.su, bank.sv, xd)
                    mod.fn_moe_down_group_s(gt, ng, bank.tr, bank.kb, bank.su, bank.sv, xd, yd, bank.ks[-1])
                    return yd
                xs = torch.empty(0, dtype=torch.float16, device=z.device)
                mod.fn_moe_gu_group(z.contiguous(), gt, ng, k, bank.tr, bank.kb, bank.su, bank.sv, ygu, bank.ks[-1], xs)
                if lim > 0:
                    ygu[:, 0].clamp_(max=lim)
                    ygu[:, 1].clamp_(min=-lim, max=lim)
                mod.fn_moe_down_group(ygu, gt, ng, bank.tr, bank.kb, bank.su, bank.sv, yd, bank.ks[-1])
                return yd
            mod.fn_moe_gu(z.contiguous(), flat, order, bank.tr, bank.kb, bank.su, bank.sv, ygu, bank.ks[-1])
            if lim > 0:
                ygu[:, 0].clamp_(max=lim)
                ygu[:, 1].clamp_(min=-lim, max=lim)
            mod.fn_moe_down(ygu, flat, order, bank.tr, bank.kb, bank.su, bank.sv, yd, bank.ks[-1])
            return yd
        G, U, D = experts
        out = []
        for r in range(M):
            for e in idx[r].tolist():
                g = F.linear(z[r:r + 1], G.weight(e, z.dtype))
                u = F.linear(z[r:r + 1], U.weight(e, z.dtype))
                if lim > 0:
                    g, u = g.clamp(max=lim), u.clamp(min=-lim, max=lim)
                out.append(F.linear(F.silu(g) * u, D.weight(e, z.dtype)).float())
        return torch.cat(out)

    def expert_pairs(self, z: torch.Tensor, idx: torch.Tensor, li: int, grouped: bool = False) -> torch.Tensor:
        """Every (row, slot) pair through its expert: [M * k, latent] fp32, unweighted. The clamp sits between
        the kernels' two launches, on the gate|up buffer. `grouped`: the kernels that decode an expert's tiles
        once for up to 8 of its pairs (a prompt chunk's pairs share experts); the same bits a pair."""
        bank = self.banks.get(li)
        experts = None
        if bank is None:
            p = f"layers.{li}.mlp.experts."
            experts = (self.w(p + "gate_proj"), self.w(p + "up_proj"), self.w(p + "down_proj"))
        return self.expert_pairs_bank(z, idx, bank=bank, experts=experts, lim=self.cfg.swiglu_limit, grouped=grouped)

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
            return self.rms_r(y, self.w(p + "post_fc2_norm.weight"))
        idx, wts = self.route_r(x, li)
        z = self.lin_rows(x, self.w(p + "routed_expert_down_proj.weight"))
        routed = self.lin_rows(self.combine(self.expert_pairs(z, idx, li), idx, wts), self.w(p + "routed_expert_up_proj.weight"))
        sp = p + "shared_experts."
        g = F.silu(self.lin_rows(x, self.w(sp + "gate_proj.weight")))
        shared = self.lin_rows(g * self.lin_rows(x, self.w(sp + "up_proj.weight")), self.w(sp + "down_proj.weight"))
        return self.rms_r(routed + shared, self.w(p + "combine_output_norm.weight"))

    def _row_grams(self, st: Hy4EngState, tokens: list[int], paths: list[list[int]]) -> torch.Tensor:
        """[M, n] each row's token and the n - 1 before it: its ancestors, then the state's last tokens."""
        n = self.cfg.engram_max_ngram_size
        hist = st.hist.tolist()
        out = []
        for path in paths:
            chain = hist + [tokens[a] for a in path]
            out.append([chain[-1 - j] for j in range(n)])
        return torch.tensor(out, dtype=torch.int64)

    def capture(self, st: Hy4EngState, M: int, select: bool, gcap: int | None = None):
        """The staged block's row pass as a CUDA graph (two runs on a side stream first: the kernels' first
        calls and the constants). The pass writes scratch rows and row buffers only, so nothing is restored."""
        from engine import capture_lock
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.run(st, M, select, gcap=gcap)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(g, **capture_args(st)):
            self.run(st, M, select, gcap=gcap)
        torch.cuda.synchronize()
        return g

    @torch.no_grad()
    def rows(self, tokens: list[int], parents: list[int], st: Hy4EngState, graphs: dict | None = None,
             clone: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """A block in DFS pre-order (row 0 at the state's length) -> fp32 logits [M, vocab] and the head's
        input [M, H]. Only the block's scratch rows are written. `graphs` (a device): the pass replays its
        captured graph for this state, row count and form, captured at its first use; the same operations,
        the same bits. `clone` False hands out the row buffers themselves (the next block overwrites them)."""
        M, select = self.stage(tokens, parents, st)
        gcap = st._gcap if select else 0
        if graphs is not None and self.device.type == "cuda":
            g = graphs.get((M, gcap))
            if g is None:
                g = graphs[(M, gcap)] = self.capture(st, M, select, gcap=gcap)
            g.replay()
        else:
            self.run(st, M, select, gcap=gcap)
        self.stats["rows"] += M
        self.stats["blocks"] += 1
        rb = st.rb[M]
        return (rb["logits"].clone(), rb["hidden"].clone()) if clone else (rb["logits"], rb["hidden"])

    def head_rows(self, n: torch.Tensor, w=None) -> torch.Tensor:
        w = self.w("lm_head.weight") if w is None else w
        if self.kernels and hasattr(w, "bf16_cached"):
            # SPD-345: the head's rows on `dense_rows` (the vocabulary's tiles decoded once for 16 rows, not once a
            # row): every output byte is the one-row call's
            y = self._dense_block(n.contiguous(), w)
            y = y if y.dtype == torch.float32 else y.float()
            cut = self.cuts.get(id(w))
            return y if cut is None else y[:, :cut]
        if not torch.is_tensor(w):
            w = w.bf16_cached()
        return self.linr(n, w)[:, :self.cfg.vocab_size]

    @torch.no_grad()
    def commit(self, st: Hy4EngState, path: list[int]) -> None:
        """Keep the block rows `path` (a chain from the anchor): their scratch rows into the state."""
        cfg = self.cfg
        path = [int(r) for r in path]
        for i, r in enumerate(path):
            if r >= len(st.bparents) or st.bparents[r] != (path[i - 1] if i else -1):
                raise ValueError(f"{path} is not a chain of the last block from its anchor")
        L, rr, Wn, n = st.length, cfg.index_kpool, cfg.sliding_window, len(path)
        if not path:
            return
        M = len(st.bparents)
        js = st.jsrc
        if js is not None:
            # SPD-349: the block ran in a joint pass: its scratch rows and picks are the pass's, from its first row
            jp, r0 = js
            blat_all, bik_all, bgk_all = jp["blat"], jp["bik"], jp["bgk"]
            pick_q, pick_ok = jp["pick_q"], jp["pick_ok"]
        else:
            r0 = 0
            blat_all, bik_all, bgk_all = st.blat_all, st.bik_all, st.bgk_all
            pick_q, pick_ok = (st.rb[M]["pick_q"], st.rb[M]["pick_ok"]) if st.pick_layers else (None, None)
        closes = [(L + i) % rr == rr - 1 for i in range(n)]
        nc = sum(closes)
        if self.zero_sync_commit and self.device.type == "cuda":
            # one pinned row of the host's numbers, one copy: the scratch rows, the slots, which rows close a group, the
            # closing rows and their groups, the tokens (no pageable transfer, no device read)
            if st.cm_h is None:
                st.cm_h = torch.zeros(6, BLOCK, dtype=torch.int64, pin_memory=True)
                st.cm_d = torch.zeros(6, BLOCK, dtype=torch.int64, device=self.device)
                st.cm_ev = torch.cuda.Event()
            else:
                st.cm_ev.synchronize()
            h = st.cm_h
            cl = [i for i in range(n) if closes[i]]
            h[0, :n] = torch.tensor([r0 + r for r in path])
            h[1, :n] = torch.arange(L, L + n)
            h[2, :n] = torch.tensor(closes, dtype=torch.int64)
            if nc:
                h[3, :nc] = torch.tensor([r0 + path[i] for i in cl])
                h[4, :nc] = torch.tensor([(L + i) // rr for i in cl])
            h[5, :n] = torch.tensor([st.btok[r] for r in path])
            st.cm_d.copy_(h, non_blocking=True)
            st.cm_ev.record()
            d = st.cm_d
            rows, slots, ct, toks = d[0, :n], d[1, :n], d[2, :n] != 0, d[5, :n]
            st.lat_all[:, slots] = blat_all[self._sidx][:, rows]
            st.ik_all[:, slots] = torch.where(ct[None, :, None], bgk_all[:, rows], bik_all[:, rows])
            if nc:
                st.gk_all[:, d[4, :nc]] = bgk_all[:, d[3, :nc]]
            st.win_all[:, slots[-Wn:] % Wn] = blat_all[self._widx][:, rows[-Wn:]]
        else:
            rows = torch.tensor([r0 + r for r in path], device=self.device)
            slots = torch.arange(L, L + n, device=self.device)
            ct = torch.tensor(closes, device=self.device)
            st.lat_all[:, slots] = blat_all[self._sidx][:, rows]
            st.ik_all[:, slots] = torch.where(ct[None, :, None], bgk_all[:, rows], bik_all[:, rows])
            if any(closes):
                st.gk_all[:, slots[ct] // rr] = bgk_all[:, rows[ct]]
            st.win_all[:, slots[-Wn:] % Wn] = blat_all[self._widx][:, rows[-Wn:]]
            toks = torch.tensor([st.btok[r] for r in path], dtype=torch.int64, device=self.device)
        st.hist.copy_(torch.cat((st.hist, toks))[-st.hist.shape[0]:])
        st.length = L + n
        if st.pick_layers:
            st.pick_q.copy_(pick_q[:, r0 + path[-1]])
            st.pick_ok.copy_(pick_ok[:, r0 + path[-1]])
            st.picks_at = st.length
        st.btok, st.bparents = [], []
        st.jsrc = None

    @torch.no_grad()
    def verify_multi(self, items: list) -> list:
        """SPD-343: [(tokens, parents, st), ...] verified in one joint pass reading all target linear,
        MoE expert, and head weights ONCE across active lanes; each state's row buffers receive the exact
        same bits as its lone `rows()` pass."""
        if not items:
            raise ValueError("verify_multi needs at least one lane")
        if len(items) == 1:
            tokens, parents, st = items[0]
            M, select = self.stage([int(t) for t in tokens], [int(p) for p in parents], st)
            self.run(st, M, select)
            self.stats["rows"] += M
            self.stats["blocks"] += 1
            return [st.rb[M]]
        if self._joint_ok(items):
            return self._verify_joint(items)

        cfg, fused = self.cfg, self.fused
        lin = self.lin1 if fused else self.lin_rows
        cx, off = [], 0
        for tokens, parents, st in items:
            M, select = self.stage([int(t) for t in tokens], [int(p) for p in parents], st)
            rb = st.rb[M]
            cx.append({"st": st, "M": M, "select": select, "rb": rb, "c": self._build_c(st, rb),
                       "live": rb["ctl"][:, 6] != 0, "sl": slice(off, off + M)})
            off += M
            self.stats["rows"] += M
        self.stats["blocks"] += 1

        eps, he, mag, R, NS = cfg.rms_norm_eps, cfg.hc_eps, cfg.hc_magnitude, cfg.kv_lora_rank, cfg.index_topk
        hs = [self.embed(x["rb"]["ctl"][:, 0]).unsqueeze(1).expand(-1, cfg.hc_mult, -1).contiguous() for x in cx]
        pends = [None] * len(cx)

        for li in range(cfg.num_hidden_layers):
            pre, a, kind = f"layers.{li}.", cfg.attn(li), cfg.layer_types[li]
            p_ = pre + "self_attn."
            if li in self.hashes:
                el = self._elayers.index(li)
                for j, x in enumerate(cx):
                    if pends[j] is not None:
                        hs[j] = K_.inject(hs[j], pends[j][0], pends[j][1]) if fused else self.inject(hs[j], pends[j][0], pends[j][1])
                        pends[j] = None
                    er = self._engram_staged(x["rb"], el, li, x["M"])
                    hs[j] = (hs[j] + self.engram_r(hs[j], er, li, x["live"])).contiguous()

            xs, attn_posts = [], []
            for j, x in enumerate(cx):
                xm, post, hs[j] = self._hc_v2(hs[j], pends[j], pre + "hc_attn_layer.hc_pre.", self.w(pre + "input_layernorm.weight"))
                xs.append(xm)
                attn_posts.append(post)
            X = torch.cat(xs)
            gate = lin(X, self.w(p_ + "linear_gate.weight"))
            q = lin(X, self.w(p_ + "q_a_proj.weight"))
            qres = K_.norm(q, self.w(p_ + "q_a_layernorm.weight"), eps) if fused else self.rms_r(q, self.w(p_ + "q_a_layernorm.weight"))
            q = lin(qres, self.w(p_ + "q_b_proj.weight")).contiguous().view(off, a.heads, a.qk)
            ckv = lin(X, self.w(p_ + "kv_a_proj_with_mqa.weight")).contiguous()
            cos = torch.cat([x["c"]["rope"][kind][0] for x in cx])
            sin = torch.cat([x["c"]["rope"][kind][1] for x in cx])
            if fused:
                K_.rot(q, cos, sin, a.nope)
                K_.norm(ckv[:, :R], self.w(p_ + "kv_a_layernorm.weight"), eps, out=ckv[:, :R])
                K_.rot(ckv.view(off, 1, R + a.rope), cos, sin, R)
            else:
                q = torch.cat((q[..., :a.nope], rotate(q[..., a.nope:], cos[:, None], sin[:, None])), dim=-1)
                ckv = torch.cat((self.rms_r(ckv[:, :R].contiguous(), self.w(p_ + "kv_a_layernorm.weight")),
                                 rotate(ckv[:, R:], cos, sin)), dim=-1)

            if kind == SPARSE:
                ip = p_ + "compressed_indexer."
                D, rd = cfg.index_head_dim, a.rope
                if fused:
                    bik_all = K_.norm(K_.gemv(X, self.w(ip + "token_key_projection.weight"), bf16=True),
                                      self.w(ip + "token_key_norm.weight"), eps)
                    iq_all = self.lin_rows(qres, self.w(ip + "query_projection.weight")) if any(x["select"] for x in cx) else None
                    hw_all = K_.gemv(X, self.w(ip + "head_weight_projection.weight"), bf16=True) if any(x["select"] for x in cx) else None
                else:
                    bik_all = self.rms_r(lin(X, self.w(ip + "token_key_projection.weight")), self.w(ip + "token_key_norm.weight"))
                    iq_all = lin(qres, self.w(ip + "query_projection.weight")) if any(x["select"] for x in cx) else None
                    hw_all = lin(X, self.w(ip + "head_weight_projection.weight")) if any(x["select"] for x in cx) else None

            ys = []
            for x in cx:
                st, M, c, sl = x["st"], x["M"], x["c"], x["sl"]
                st.blat[li][:M].copy_(ckv[sl])
                if kind == SPARSE:
                    Lt, rbm = c["Lt"], x["rb"]
                    st.bik[li][:M].copy_(bik_all[sl])
                    gk = self.group_keys_r(self._gather(st.ik[li], st.bik[li], c["gq"], Lt, c["anc"]), li).contiguous()
                    c0, s0 = c["grope"]
                    if fused:
                        K_.rot(gk.view(M, 1, D), c0, s0, D - rd)
                    else:
                        gk = torch.cat((gk[:, :D - rd], rotate(gk[:, D - rd:], c0, s0)), dim=-1)
                    st.bgk[li][:M].copy_(gk)
                    pq = self.ar(NS).expand(M, -1)
                    if x["select"]:
                        rc, rs = c["rope"][kind]
                        pq = torch.where(c["qg"][:, None] < cfg.group_topk, pq,
                                         self._select(li, st, iq_all[sl], hw_all[sl], rc, rs, c))
                    valid = pq <= c["p"][:, None]
                    if li in st.pick_layers:
                        pi = st.pick_layers.index(li)
                        rbm["pick_q"][pi, :M].copy_(pq)
                        rbm["pick_ok"][pi, :M].copy_(valid)
                    Kr = self._gather(st.lat[li], st.blat[li], pq, Lt, c["anc"])
                else:
                    Kr, valid = torch.cat((st.win[li], st.blat[li]))[c["wsrc"]], c["wval"]
                ys.append(self.attend_r(q[sl], Kr, valid, li).reshape(M, -1) * torch.sigmoid(gate[sl]))

            Y = lin(torch.cat(ys), self.w(p_ + "o_proj.weight"))
            attn_out = K_.norm(Y, self.w(p_ + "post_proj_norm.weight"), eps) if fused else self.rms_r(Y, self.w(p_ + "post_proj_norm.weight"))

            xs, mlp_posts = [], []
            for j, x in enumerate(cx):
                xm, post, hs[j] = self._hc_v2(hs[j], (attn_out[x["sl"]], attn_posts[j]), pre + "hc_mlp_layer.hc_pre.",
                                              self.w(pre + "post_attention_layernorm.weight"))
                xs.append(xm)
                mlp_posts.append(post)
            Ym = self.mlp_fused(torch.cat(xs), li) if fused else self.mlp_rows(torch.cat(xs), li)
            pends = [(Ym[x["sl"]], mlp_posts[j]) for j, x in enumerate(cx)]

        N = torch.cat([self._final_v2(hs[j], pends[j]) for j in range(len(cx))])
        LG = torch.cat([self.head_rows(N[i:i + BLOCK]) for i in range(0, off, BLOCK)])
        for x in cx:
            rb, sl = x["rb"], x["sl"]
            rb["hidden"].copy_(N[sl])
            rb["logits"].copy_(LG[sl])
            rb["next"].copy_(LG[sl].argmax(dim=-1))
        return [x["rb"] for x in cx]

    # ------------------------------------------------------------------------- the joint row pass (SPD-349)
    def _joint_ok(self, items: list) -> bool:
        """Whether `verify_multi` takes the joint row pass: on, the fused kernels with packed Engram tables, every lane's
        state of one shape and offset (the address tables give one stride a buffer), at most JOINT_ROWS rows."""
        if not (self.attn_multi and self.fused and self._packed and self.device.type == "cuda"):
            return False
        sts = [st for _, _, st in items]
        s0 = sts[0]
        if len({id(s) for s in sts}) != len(sts):
            return False
        if any(s.cap != s0.cap or s.delta != s0.delta or s.Gc != s0.Gc for s in sts):
            return False
        if any(not (s.lat_all.is_contiguous() and s.ik_all.is_contiguous() and s.win_all.is_contiguous()
                    and s.gk_all.is_contiguous()) for s in sts):
            return False
        return sum(len(t) for t, _, _ in items) <= JOINT_ROWS[-1] and len(items) <= JOINT_LANES

    def _joint_bufs(self, st0: Hy4EngState) -> dict:
        jp = self._jp
        if jp is not None:
            return jp
        cfg, dev = self.cfg, self.device
        R, B = JOINT_ROWS[-1], JOINT_LANES
        C = cfg.kv_lora_rank + cfg.sparse_attn.rope
        D, nS, nP = cfg.index_head_dim, len(cfg.sparse_layers), len(st0.pick_layers)
        CJ = self.CTL + BLOCK + 3 * st0.BG + 2
        nE, heads = len(self._elayers), cfg.engram_heads
        words = self.tables[self._elayers[0]].words if nE else 1
        ib = R * CJ + 4 * B
        jp = {"CJ": CJ, "ib": ib,
              "i_h": torch.zeros(ib, dtype=torch.int64, pin_memory=True), "i_d": torch.zeros(ib, dtype=torch.int64, device=dev),
              "e_h": torch.zeros(max(nE, 1), R * heads, words, dtype=torch.int16, pin_memory=True),
              "e_d": torch.zeros(max(nE, 1), R * heads, words, dtype=torch.int16, device=dev),
              "ehead": self.ar(heads).repeat(R),
              "blat": torch.zeros(cfg.num_hidden_layers, R, C, dtype=self.dtype, device=dev),
              "bik": torch.zeros(nS, R, D, dtype=self.dtype, device=dev),
              "bgk": torch.zeros(nS, R, D, dtype=self.dtype, device=dev),
              "logits": torch.zeros(R, cfg.vocab_size, dtype=torch.float32, device=dev),
              "hidden": torch.zeros(R, cfg.hidden_size, dtype=self.dtype, device=dev),
              "next": torch.zeros(R, dtype=torch.int64, device=dev),
              "pick_q": torch.zeros(max(nP, 1), R, cfg.index_topk, dtype=torch.int64, device=dev),
              "pick_ok": torch.zeros(max(nP, 1), R, cfg.index_topk, dtype=torch.bool, device=dev),
              "graphs": {}, "pool": None, "stream": None, "ev": torch.cuda.Event(),
              "stats": {"passes": 0, "replays": 0, "captures": 0, "capture_ms": 0.0, "eager": 0}}
        jp["ctl"] = jp["i_d"][:R * CJ].view(R, CJ)
        jp["ptr"] = jp["i_d"][R * CJ:].view(4, B)
        self._jp = jp
        return jp

    def _stage_joint(self, items: list) -> tuple:
        """Every lane's block into the joint pass's pinned rows (`_stage_rows` a lane, the lane and its first row as two
        more columns; pad rows repeat the first row), the lanes' buffer addresses, the Engram rows of all of them in one
        table read; one copy each to the device. -> (rows, padded rows, select, first rows)."""
        cfg = self.cfg
        st0 = items[0][2]
        jp = self._joint_bufs(st0)
        CJ = jp["CJ"]
        ctl_h = jp["i_h"][:JOINT_ROWS[-1] * CJ].view(JOINT_ROWS[-1], CJ)
        ptr_h = jp["i_h"][JOINT_ROWS[-1] * CJ:].view(4, JOINT_LANES)
        jp["ev"].synchronize()                  # the last pass's copy out of these pinned rows is done
        rows_all, grams, r0s, select = [], [], [], False
        for b, (tokens, parents, st) in enumerate(items):
            tokens, parents = [int(t) for t in tokens], [int(p) for p in parents]
            M = len(tokens)
            if not 1 <= M <= BLOCK or len(parents) != M:
                raise ValueError(f"a block holds 1 to {BLOCK} rows")
            rows, sel, paths = self._stage_rows(tokens, parents, st)
            select = select or sel
            r0 = len(rows_all)
            r0s.append(r0)
            rows_all += [row + [b, r0] for row in rows]
            st.btok, st.bparents = tokens, parents
            st.jsrc = (jp, r0)
            if self._elayers:
                grams.append(self._row_grams(st, tokens, paths))
            ptr_h[0, b] = st.lat_all.data_ptr()
            ptr_h[1, b] = st.ik_all.data_ptr()
            ptr_h[2, b] = st.win_all.data_ptr()
            ptr_h[3, b] = st.gk_all.data_ptr()
        R = len(rows_all)
        Rp = next(x for x in JOINT_ROWS if x >= R)
        ctl_h[:R] = torch.tensor(rows_all, dtype=torch.int64)
        if Rp > R:
            ctl_h[R:Rp] = ctl_h[0]
        if self._elayers:
            g = torch.cat(grams)
            if Rp > R:
                g = torch.cat((g, g[:1].expand(Rp - R, -1)))
            heads = cfg.engram_heads
            got = gather_packed_many([self.tables[li] for li in self._elayers], [self.hashes[li].ids(g) for li in self._elayers])
            for k, rows_ in enumerate(got):
                jp["e_h"][k, :Rp * heads] = rows_
            jp["e_d"][:, :Rp * heads].copy_(jp["e_h"][:, :Rp * heads], non_blocking=True)
            self.stats["engram_rows"] += R * heads * len(self._elayers)
        jp["ctl"][:Rp].copy_(ctl_h[:Rp], non_blocking=True)
        jp["ptr"].copy_(ptr_h, non_blocking=True)
        jp["ev"].record()
        gcap = self._gcap(select, max(r[4] for r in rows_all), st0.Gc)
        return R, Rp, select, gcap, r0s

    def _build_cm(self, Rp: int, st0: Hy4EngState) -> dict:
        """`_build_c` of the joint pass's rows: every row its own state's length, the scratch rows by their place in the
        joint buffers (the lane's first row added), the window's rows as (from scratch, index)."""
        cfg, jp, o, BG = self.cfg, self._jp, self.CTL, st0.BG
        Wn, rr, CJ = cfg.sliding_window, cfg.index_kpool, jp["CJ"]
        ctl = jp["ctl"][:Rp]
        Lt, p, g0 = ctl[:, 2], ctl[:, 1], ctl[:, 5]
        lane, r0 = ctl[:, CJ - 2], ctl[:, CJ - 1]
        anc = ctl[:, o:o + BLOCK] + r0[:, None]
        wq = p[:, None] - Wn + 1 + self.ar(Wn)
        wfl = wq >= Lt[:, None]
        return {
            "p": p, "anc": anc, "Lt": Lt[:, None], "qg": ctl[:, 3], "gmax": ctl[:, 4], "lane": lane.contiguous(),
            "wfl": wfl.contiguous(), "widx": torch.where(wfl, anc.gather(1, (wq - Lt[:, None]).clamp(0, BLOCK - 1)),
                                                        torch.remainder(wq, Wn)).contiguous(),
            "wval": wq >= 0, "gq": g0[:, None] + self.ar(rr),
            "cidx": ctl[:, o + BLOCK:o + BLOCK + BG] + r0[:, None], "bgid": ctl[:, o + BLOCK + BG:o + BLOCK + 2 * BG],
            "bok": ctl[:, o + BLOCK + 2 * BG:o + BLOCK + 3 * BG] != 0,
            "rope": {kind: self.rope(self.pos3(p + st0.delta), kind) for kind in set(cfg.layer_types)},
            "grope": self.rope(self.pos3(g0 + st0.delta), SPARSE),
        }

    def _gather_m(self, which: int, off: int, rows: int, width: int, scratch: torch.Tensor, q: torch.Tensor, c: dict) -> torch.Tensor:
        """`_gather` of the joint pass: positions q [R, n] from each row's own lane's buffer `which` (an address table
        row) at layer offset `off` (rows of `width`, `rows` of them), or the joint scratch rows past its state."""
        flag = (q >= c["Lt"]).contiguous()
        idx = torch.where(flag, c["anc"].gather(1, (q - c["Lt"]).clamp(0, BLOCK - 1)), q.clamp(0, rows - 1)).contiguous()
        return K_.gather_lanes(idx, flag, c["lane"], self._jp["ptr"][which], off, width, scratch, width)

    def _select_m(self, li: int, si: int, st0: Hy4EngState, iq: torch.Tensor, hw: torch.Tensor, cos, sin, c: dict, gcap: int) -> torch.Tensor:
        """`_select` of the joint pass: the committed groups' scores from each row's own lane's group-key mirror
        (`mm_lanes`: `mm`'s bits, width `gcap` with true buffer stride `Gc + BG`), the block's from the joint scratch;
        the rest as `_select`."""
        cfg = self.cfg
        rr, Gc, BG = cfg.index_kpool, st0.Gc, st0.BG
        nh, D, rd = cfg.index_n_heads, cfg.index_head_dim, cfg.sparse_attn.rope
        M = iq.shape[0]
        qi = iq.reshape(M, nh, D).contiguous()
        K_.rot_exact(qi, cos.contiguous(), sin.contiguous(), D - rd)
        sc = torch.cat((K_.mm_lanes(qi, self._jp["ptr"][3], c["lane"], si * (Gc + BG) * D, D, gcap, 1, D, 64),
                        self.mm(qi, self._jp["bgk"][si][c["cidx"]].transpose(1, 2), 64)), dim=-1)
        if nh & (nh - 1) == 0:
            logit = K_.wsum_relu(sc, (hw.float() * (nh ** -0.5 * D ** -0.5)).contiguous())
        else:
            logit = self.mm((hw.float() * (nh ** -0.5 * D ** -0.5)).unsqueeze(1), F.relu(sc)).view(M, gcap + BG)
        gid = self.ar(gcap + BG)
        ok = (gid[None, :] < c["gmax"][:, None]) | ((gid[None, :] >= gcap) & c["bok"].gather(1, (gid - gcap).clamp(0, BG - 1).expand(M, -1)))
        top = topk_stable(logit.masked_fill(~ok, float("-inf")), cfg.group_topk - 1)[1]
        groups = torch.where(top < gcap, top, c["bgid"].gather(1, (top - gcap).clamp(0, BG - 1))).sort(dim=-1).values
        off = self.ar(rr)
        return torch.cat(((groups.unsqueeze(-1) * rr + off).flatten(1), c["qg"][:, None] * rr + off), dim=-1)

    def _attention_m(self, x: torch.Tensor, li: int, st0: Hy4EngState, select: bool, gcap: int, c: dict) -> torch.Tensor:
        """`attention_fused` of the joint pass: the same statements, the scratch rows in the joint buffers and the
        committed rows read from each row's own lane (`_gather_m`, `gather_lanes`, `_select_m`)."""
        cfg, jp = self.cfg, self._jp
        a, kind, p_ = cfg.attn(li), cfg.layer_types[li], f"layers.{li}.self_attn."
        R, NS, M, eps = cfg.kv_lora_rank, cfg.index_topk, x.shape[0], cfg.rms_norm_eps
        C = R + a.rope
        gate = self.lin1(x, self.w(p_ + "linear_gate.weight"))
        qres = K_.norm(self.lin1(x, self.w(p_ + "q_a_proj.weight")), self.w(p_ + "q_a_layernorm.weight"), eps)
        q = self.lin1(qres, self.w(p_ + "q_b_proj.weight")).contiguous().view(M, a.heads, a.qk)
        ckv = self.lin1(x, self.w(p_ + "kv_a_proj_with_mqa.weight")).contiguous()
        cos, sin = c["rope"][kind]
        K_.rot(q, cos, sin, a.nope)
        K_.norm(ckv[:, :R], self.w(p_ + "kv_a_layernorm.weight"), eps, out=ckv[:, :R])
        K_.rot(ckv.view(M, 1, R + a.rope), cos, sin, R)
        jp["blat"][li][:M] = ckv
        if kind == SPARSE:
            si = self._sidx_of[li]
            ip = p_ + "compressed_indexer."
            D, rd = cfg.index_head_dim, a.rope
            jp["bik"][si][:M] = K_.norm(K_.gemv(x, self.w(ip + "token_key_projection.weight"), bf16=True), self.w(ip + "token_key_norm.weight"), eps)
            gk = self.group_keys_r(self._gather_m(1, si * st0.cap * D, st0.cap, D, jp["bik"][si], c["gq"], c), li).contiguous()
            c0, s0 = c["grope"]
            K_.rot(gk.view(M, 1, D), c0, s0, D - rd)
            jp["bgk"][si][:M] = gk
            pq = self.ar(NS).expand(M, -1)
            if select:
                iq = self.lin_rows(qres, self.w(ip + "query_projection.weight"))
                hw = K_.gemv(x, self.w(ip + "head_weight_projection.weight"), bf16=True)
                pq = torch.where(c["qg"][:, None] < cfg.group_topk, pq, self._select_m(li, si, st0, iq, hw, cos, sin, c, gcap))
            valid = pq <= c["p"][:, None]
            if li in st0.pick_layers:
                pi = st0.pick_layers.index(li)
                jp["pick_q"][pi, :M].copy_(pq)
                jp["pick_ok"][pi, :M].copy_(valid)
            Kr = self._gather_m(0, si * st0.cap * C, st0.cap, C, jp["blat"][li], pq, c)
        else:
            wi = self._widx_of[li]
            Kr = K_.gather_lanes(c["widx"], c["wfl"], c["lane"], jp["ptr"][2], wi * cfg.sliding_window * C, C, jp["blat"][li], C)
            valid = c["wval"]
        y = K_.mul_sigmoid(self.attend_r(q, Kr, valid, li).reshape(M, -1), gate) if self._ms else \
            (self.attend_r(q, Kr, valid, li).reshape(M, -1) * torch.sigmoid(gate))
        return K_.norm(self.lin1(y, self.w(p_ + "o_proj.weight")), self.w(p_ + "post_proj_norm.weight"), eps)

    @torch.no_grad()
    def run_multi(self, Rp: int, select: bool, st0: Hy4EngState, gcap: int | None = None) -> None:
        """The joint pass's staged rows on the device (`_run_fused`'s layers, `_attention_m`): logits, head inputs and
        picks into the joint buffers, the rows' latents and keys into the joint scratch. No host read and no lane's
        tensor named: what a joint graph captures, once a padded row count and form."""
        cfg, jp = self.cfg, self._jp
        gc = (st0.Gc if gcap is None else int(gcap)) if select else 0
        c = self._build_cm(Rp, st0)
        ctl = jp["ctl"][:Rp]
        live = ctl[:, 6] != 0
        h = self.embed(ctl[:, 0]).unsqueeze(1).expand(-1, cfg.hc_mult, -1).contiguous()
        eps, he, mag = cfg.rms_norm_eps, cfg.hc_eps, cfg.hc_magnitude
        heads = cfg.engram_heads
        pend = None
        for li in range(cfg.num_hidden_layers):
            pre = f"layers.{li}."
            if li in self.hashes:
                if pend is not None:
                    h, pend = K_.inject(h, pend[0], pend[1]), None
                from engine.models.hy4_flash.table import decode_rows
                t, k = self.tables[li], self._elayers.index(li)
                er = decode_rows(jp["e_d"][k, :Rp * heads], jp["ehead"][:Rp * heads], self._ebias[li], t.K,
                                 t.codebook(self.device), t.dim).to(self.dtype).view(Rp, heads, -1)
                h = (h + self.engram_r(h, er, li, live)).contiguous()
            x, post, h = K_.hc(h, pend, *self._hcw(pre + "hc_attn_layer.hc_pre."), self.w(pre + "input_layernorm.weight"), eps, he, mag)
            y = self._attention_m(x, li, st0, select, gc, c)
            x, post, h = K_.hc(h, (y, post), *self._hcw(pre + "hc_mlp_layer.hc_pre."), self.w(pre + "post_attention_layernorm.weight"), eps, he, mag)
            pend = (self.mlp_fused(x, li), post)
        n = K_.hc(h, pend, self.w("hc_head.hc_head_fn"), self.w("hc_head.hc_head_base"), self.w("hc_head.hc_head_scale"),
                  self.w("norm.weight"), eps, he, mag, head=True)[0]
        lg = self.head_rows(n)
        jp["hidden"][:Rp].copy_(n)
        jp["logits"][:Rp].copy_(lg)
        jp["next"][:Rp].copy_(lg.argmax(dim=-1))

    def _capture_multi(self, Rp: int, select: bool, st0: Hy4EngState, gcap: int):
        """The joint pass of Rp rows as a CUDA graph on its own stream and the joint graphs' one pool (two runs first:
        the kernels' first calls and the constants). It writes the joint buffers only, so nothing is restored."""
        from engine import capture_lock
        jp = self._jp
        if jp["stream"] is None:
            jp["stream"] = capture_lock.lease_stream()
        if jp["pool"] is None:
            jp["pool"] = torch.cuda.graph_pool_handle()
        s = jp["stream"]
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.run_multi(Rp, select, st0, gcap=gcap)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with capture_lock.LOCK, torch.cuda.graph(g, pool=jp["pool"], stream=s):
            self.run_multi(Rp, select, st0, gcap=gcap)
        torch.cuda.synchronize()
        return g

    def drop_joint_graphs(self) -> int:
        jp = self._jp
        if jp is None:
            return 0
        n = len(jp["graphs"])
        for g in jp["graphs"].values():
            g.reset()
        jp["graphs"].clear()
        jp["pool"] = None
        if n:
            from engine import capture_lock
            capture_lock.graphs_reset()
        return n

    def _verify_joint(self, items: list) -> list:
        """`verify_multi` on the joint row pass: staged into the joint rows, run from the key's graph (captured at its
        first use while `joint_capture_ok`) or eagerly, the same bits. Each lane gets views of its rows."""
        import time as _time
        jp_new = self._jp is None
        R, Rp, select, gcap, r0s = self._stage_joint(items)
        jp = self._jp
        if jp_new:
            self._sidx_of = {li: i for i, li in enumerate(self.cfg.sparse_layers)}
            self._widx_of = {li: i for i, li in enumerate(self.cfg.window_layers)}
        st0 = items[0][2]
        jp["stats"]["passes"] += 1
        key = (Rp, gcap)
        g = jp["graphs"].get(key) if self.joint_graph else None
        if g is None and self.joint_graph and self.joint_capture_ok:
            t0 = _time.perf_counter()
            g = jp["graphs"][key] = self._capture_multi(Rp, select, st0, gcap)
            jp["stats"]["captures"] += 1
            jp["stats"]["capture_ms"] += 1e3 * (_time.perf_counter() - t0)
        if g is not None:
            g.replay()
            jp["stats"]["replays"] += 1
        else:
            self.run_multi(Rp, select, st0, gcap=gcap)
            jp["stats"]["eager"] += 1
        self.stats["blocks"] += 1
        out = []
        for (tokens, _, st), r0 in zip(items, r0s):
            M = len(tokens)
            self.stats["rows"] += M
            out.append({"logits": jp["logits"][r0:r0 + M], "hidden": jp["hidden"][r0:r0 + M], "next": jp["next"][r0:r0 + M]})
        return out

    @torch.no_grad()
    def decode_multi(self, items: list, graphs=None) -> list:
        """One plain decode token per lane [(token, st, slot), ...] via shared weights (`verify_multi`)."""
        return self.verify_multi([([int(tok)], [-1], st) for tok, st, _ in items])

    @torch.no_grad()
    def step(self, token: int, st: Hy4EngState, graphs: dict | None = None) -> torch.Tensor:
        """One decode token: its row, committed. Logits [1, vocab] fp32."""
        lg, _ = self.rows([int(token)], [-1], st, graphs=graphs)
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
        return self.head(st.prompt_hidden[-1:]) if head == "last" else None

    def _chunk_begin(self, tokens: torch.Tensor, st: Hy4EngState, read: bool = True) -> tuple:
        """What a chunk's layers share: its slots, the rows that own an n-gram, the tokens with the state's last ones,
        and its table rows handed to the reader -> (T, L0, slots, live, full, reads). `read` False: the n-grams
        [T, n] in place of the reads (a joint prefill hands every prompt's to the reader layer by layer)."""
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
        if st.gk_dirty:
            _mirror(st, cfg.index_kpool)
        grams = torch.stack([full[n_ctx - j: n_ctx - j + T] for j in range(cfg.engram_max_ngram_size)], dim=-1) if self.hashes else None
        if not read:
            return T, L0, slots, live, full, grams
        # every Engram layer's table rows read and decoded on the reader thread, in layer order, while the layers run
        reads = self._engram_reads(grams) if self.hashes else {}
        return T, L0, slots, live, full, reads

    def _chunk_end(self, st: Hy4EngState, T: int, L0: int, full: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        st.hist.copy_(full[-st.hist.shape[0]:])
        st.length = L0 + T
        st.picks_at = st.length if self.pick_layers else -1
        st.btok, st.bparents = [], []
        st.prompt_hidden = self.prompt_hidden = hidden      # ENG-485: the state's; the engine's for the tools
        self.stats["prefill_rows"] += T
        return st.prompt_hidden

    def _yield_layer(self) -> None:
        """D12 (SPD-343): offer the GPU turn between prefill layers so concurrent decodes never stall 42 layers."""
        fn = self.layer_yield
        if fn is None:
            m = sys.modules.get("engine.model")
            fn = getattr(m, "WINDOW_YIELD", None) if m is not None else None
        if fn is not None:
            fn()

    def forward_chunk(self, tokens: torch.Tensor, st: Hy4EngState) -> torch.Tensor:
        """`tokens` [T] appended to the state on batched operations -> the head's inputs [T, H]. With `prefill_v2` the
        pass on fixed-order kernels (`_forward_chunk_v2`)."""
        if self.prefill_v2:
            return self._forward_chunk_v2(tokens, st)
        cfg = self.cfg
        T, L0, slots, live, full, reads = self._chunk_begin(tokens, st)
        h = self.embed(tokens).unsqueeze(1).expand(-1, cfg.hc_mult, -1).contiguous()
        for li in range(cfg.num_hidden_layers):
            p = f"layers.{li}."
            if li in self.hashes:
                rows = self._engram_take(li, reads[li])
                self.stats["engram_rows"] += rows.shape[0] * rows.shape[1]
                h = h + self.engram_apply(h, rows, li, live)
            x, post = self.hc(h, p + "hc_attn_layer.hc_pre.")
            y = self.attention_chunk(self.rms(x, self.w(p + "input_layernorm.weight")), li, st, slots)
            h = self.inject(h, y, post)
            x, post = self.hc(h, p + "hc_mlp_layer.hc_pre.")
            y = self.mlp_chunk(self.rms(x, self.w(p + "post_attention_layernorm.weight")), li)
            h = self.inject(h, y, post)
            if li + 1 < cfg.num_hidden_layers:
                self._yield_layer()
        return self._chunk_end(st, T, L0, full, self.final(h))

    def _forward_chunk_v2(self, tokens: torch.Tensor, st: Hy4EngState) -> torch.Tensor:
        """SPD-325 (F2): a chunk on the row pass's own layer body (`_run_fused`'s): the Engram rows, a hyper-connection
        with the pending inject before it and the norm after it, attention (`_attention_chunk_v2`), the hyper-connection,
        the MLP; the head form of the hyper-connection on every row at the end (the drafter reads every prompt row's
        head input). On a device the row kernels (`K_.hc`, `K_.norm`, `K_.inject`) and `mm` with the rows on its tile
        axis for the router and the Engram sums; the packed projections on `lin_prompt` (R83: one order at every row
        count) and the experts on `_experts_chunk`, as the pass before. Off the device the same body on the row pass's
        reference sums."""
        cfg = self.cfg
        c = self._v2_begin(tokens, st)
        for li in range(cfg.num_hidden_layers):
            x, post = self._v2_attn(c, li)
            c["pend"] = (self._mlp_chunk_v2(x, li), post)
            if li + 1 < cfg.num_hidden_layers:
                self._yield_layer()
        return self._v2_end(c)

    def _v2_begin(self, tokens: torch.Tensor, st: Hy4EngState, reads: dict | None = None) -> dict:
        """`_forward_chunk_v2`'s start for one chunk: its slots and Engram reads (`reads`: handed in by a joint prefill,
        else submitted here), its embedded streams, and (SPD-347) its rotary once a kind from the host's positions."""
        cfg = self.cfg
        if reads is None:
            T, L0, slots, live, full, reads = self._chunk_begin(tokens, st)
        else:
            T, L0, slots, live, full, _ = self._chunk_begin(tokens, st, read=False)
        rope = None
        if self.prefill_fast:
            pos = torch.arange(L0, L0 + T) + st.delta
            rope = {kind: tuple(t.contiguous() for t in self.rope(self.pos3(pos), kind)) for kind in set(cfg.layer_types)}
        return {"st": st, "T": T, "L0": L0, "slots": slots, "live": live.to(self.device), "full": full, "reads": reads,
                "h": self.embed(tokens).unsqueeze(1).expand(-1, cfg.hc_mult, -1).contiguous(), "pend": None, "rope": rope}

    def _v2_attn(self, c: dict, li: int) -> tuple:
        """Layer li of a chunk up to its MLP: the Engram rows, the attention's hyper-connection with the pending inject,
        the attention, the MLP's hyper-connection -> (the MLP's input, its post gates); the streams stay in `c`."""
        p = f"layers.{li}."
        st = c["st"]
        if li in self.hashes:
            rows = self._engram_take(li, c["reads"][li])
            self.stats["engram_rows"] += rows.shape[0] * rows.shape[1]
            if c["pend"] is not None:
                c["h"], c["pend"] = self._inject_v2(c["h"], c["pend"]), None
            c["h"] = (c["h"] + self.engram_r(c["h"], rows, li, c["live"], lin=self._rows_mm)).contiguous()
        x, post, c["h"] = self._hc_v2(c["h"], c["pend"], p + "hc_attn_layer.hc_pre.", self.w(p + "input_layernorm.weight"))
        y = self._attention_chunk_v2(x, li, st, c["slots"], rope=c["rope"])
        x, post, c["h"] = self._hc_v2(c["h"], (y, post), p + "hc_mlp_layer.hc_pre.", self.w(p + "post_attention_layernorm.weight"))
        return x, post

    def _v2_end(self, c: dict) -> torch.Tensor:
        return self._chunk_end(c["st"], c["T"], c["L0"], c["full"], self._final_v2(c["h"], c["pend"]))

    def _rows_mm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x [M, K] @ w^T for a plain weight [N, K] -> fp32 [M, N]: `linr`'s element sums, on a device with the rows on
        `mm`'s tile axis (no 16-fold padding of one-row items)."""
        if self.fused:
            return K_.mm(x.unsqueeze(0), w.t().unsqueeze(0)).view(x.shape[0], w.shape[0])
        return self.linr(x, w)

    def _inject_v2(self, h: torch.Tensor, pend: tuple) -> torch.Tensor:
        return K_.inject(h, pend[0], pend[1]) if self.fused else self.inject(h, pend[0], pend[1])

    def _hc_v2(self, h: torch.Tensor, pend: tuple | None, pre: str, w_norm: torch.Tensor) -> tuple:
        """A hyper-connection with the pending inject before it and the norm after it -> (normed mix, post gates, h)."""
        cfg = self.cfg
        if self.fused:
            return K_.hc(h, pend, *self._hcw(pre), w_norm, cfg.rms_norm_eps, cfg.hc_eps, cfg.hc_magnitude)
        if pend is not None:
            h = self.inject(h, pend[0], pend[1])
        x, post = self.hc_r(h, pre)
        return self.rms_r(x, w_norm), post, h

    def _final_v2(self, h: torch.Tensor, pend: tuple) -> torch.Tensor:
        """The last inject and the head's hyper-connection and norm, on every row -> the head's inputs [T, H]."""
        cfg = self.cfg
        if self.fused:
            return K_.hc(h, pend, self.w("hc_head.hc_head_fn"), self.w("hc_head.hc_head_base"), self.w("hc_head.hc_head_scale"),
                         self.w("norm.weight"), cfg.rms_norm_eps, cfg.hc_eps, cfg.hc_magnitude, head=True)[0]
        return self.final_r(self.inject(h, pend[0], pend[1]))

    def _mlp_chunk_v2(self, x: torch.Tensor, li: int) -> torch.Tensor:
        """`mlp_chunk` with the router's sums on `mm` (`route_r`) and the norms on the row kernel."""
        p = f"layers.{li}.mlp."
        if self.cfg.is_dense_mlp(li):
            return self._norm_v2(self._dense_v2(x, p), self.w(p + "post_fc2_norm.weight"))
        idx, wts = self.route_r(x, li, lin=self._rows_mm)
        z = self.lin_prompt(x, self.w(p + "routed_expert_down_proj.weight"))
        routed = self._experts_chunk(z, idx, wts, li) if li in self.banks else self._experts_grouped(z, idx, wts, li)
        routed = self.lin_prompt(routed, self.w(p + "routed_expert_up_proj.weight"))
        return self._norm_v2(routed + self._dense_v2(x, p + "shared_experts."), self.w(p + "combine_output_norm.weight"))

    def _experts_joinable(self, xs: list) -> bool:
        """SPD-347: whether a joint prefill's prompts may share one call of the prefill kernel for a layer's routed
        experts: each prompt alone takes that kernel (`_experts_chunk` "group"), or, under `experts_min` rows, the row
        kernels, which give the prefill kernel's bits a pair exactly where `experts_min` is not 0."""
        return self.prefill_experts == "group" and self._group_inline(self.device)

    def _moe_joint(self, xs: list, li: int) -> list:
        """SPD-347: `_mlp_chunk_v2` of a MoE layer for several prompts' rows xs: the router, the projections, the
        combine and the shared expert a prompt at a time, the routed experts of every prompt's pairs in one
        `moe_grouped` call (a pair's output is its own whatever call it is in) -> each prompt's MLP output."""
        from tools import fn_prefill_kernels as PK
        p = f"layers.{li}.mlp."
        parts = []
        for x in xs:
            idx, wts = self.route_r(x, li, lin=self._rows_mm)
            parts.append((idx, wts, self.lin_prompt(x, self.w(p + "routed_expert_down_proj.weight"))))
        ids = torch.cat([q[0].to(torch.int32) for q in parts]).contiguous()
        yd = PK.moe_grouped(torch.cat([q[2] for q in parts]).contiguous(), ids, self.banks[li], rows=GROUP_ROWS,
                            clamp=self.cfg.swiglu_limit)
        out, r0 = [], 0
        for x, (idx, wts, _) in zip(xs, parts):
            r1 = r0 + idx.numel()
            routed = self.lin_prompt(self.combine(yd[r0:r1], idx, wts), self.w(p + "routed_expert_up_proj.weight"))
            out.append(self._norm_v2(routed + self._dense_v2(x, p + "shared_experts."), self.w(p + "combine_output_norm.weight")))
            r0 = r1
        return out

    def _engram_reads(self, grams: torch.Tensor) -> dict:
        """SPD-319: a chunk's n-grams [T, n] -> {Engram layer: (its ids' shape, the future of its rows)}: every layer's ids
        hashed here, then one job a layer, in layer order, on the engine's one reader thread (`_engram_read`). The
        futures belong to the caller's frame: a chunk only ever takes its own rows."""
        if self._reader is None:
            from concurrent.futures import ThreadPoolExecutor
            self._reader = ThreadPoolExecutor(1, thread_name_prefix="hy4-table-rows")
        out = {}
        for li in self._elayers:
            ids = self.hashes[li].ids(grams)
            out[li] = (tuple(ids.shape), self._reader.submit(self._engram_read, li, ids))
        return out

    def _engram_read(self, li: int, ids: torch.Tensor) -> torch.Tensor:
        """The reader thread's job: the table rows of `ids` decoded on the CPU in the compute dtype [n, head_dim], the
        read and the decode of `engram_rows`, in pieces of TABLE_PIECE ids (a piece holds the table's lock, so a decode
        step's read waits for one piece at most)."""
        t, flat = self.tables[li], ids.reshape(-1).cpu()
        if self.prefill_fast and self.fused and hasattr(t, "gather_packed") and hasattr(t, "head_of"):
            # SPD-347: the packed words only (3.6x fewer bytes than the decoded rows, no trellis math on this thread),
            # pinned for the device's copy; `_engram_take` decodes them on the device with `decode_rows`' bits
            pk = torch.cat([t.gather_packed(flat[i:i + TABLE_PIECE]) for i in range(0, max(flat.shape[0], 1), TABLE_PIECE)])
            return ("packed", pk.pin_memory(), t.head_of(flat).to(torch.int32).pin_memory())
        return torch.cat([t.gather(flat[i:i + TABLE_PIECE], dtype=self.dtype) for i in range(0, max(flat.shape[0], 1), TABLE_PIECE)])

    def _engram_take(self, li: int, job: tuple) -> torch.Tensor:
        """Layer li's rows from its job [T, heads, head_dim] on the device, as `engram_rows` gives them."""
        shape, fut = job
        r = fut.result()
        if isinstance(r, tuple):
            _, pk, hd = r
            t = self.tables[li]
            dev = self.device
            cb = self.__dict__.setdefault("_ecb", {})
            if li not in cb:
                # the codebook and the bias made where `gather` makes them (the host), then moved: the same values
                cb[li] = (t.codebook("cpu").to(dev), t.bias.to(dev).contiguous())
            code, bias = cb[li]
            e = K_.engram_decode(pk.to(dev, non_blocking=True), hd.to(dev, non_blocking=True), bias, t.K, code, t.dim)
            if self.dtype != torch.bfloat16:
                raise RuntimeError("the device Engram decode writes bf16 rows")
            e = e.reshape(*shape, -1)
        else:
            e = r.to(self.device).reshape(*shape, -1)
        if self.cfg.engram_fp8_embedding and not getattr(self.tables[li], "fp8_done", False):
            e = fp8_round_trip(e)
        return e

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
                    pi = st.pick_layers.index(li)
                    st.pick_q[pi].copy_(self.ar(cfg.index_topk))
                    st.pick_ok[pi].copy_(self.ar(cfg.index_topk) <= L - 1)
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
                    pi = st.pick_layers.index(li)
                    st.pick_q[pi].copy_(pq[-1])
                    st.pick_ok[pi].copy_(valid[-1])
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

    # ------------------------------------------------------------------------- the prompt pass on fixed-order kernels
    def _norm_v2(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """The RMS norm of rows with its weight: the row kernel (`K_.norm`) on a device, the row pass's `rms_r` else."""
        if self.fused:
            return K_.norm(x.contiguous(), w, self.cfg.rms_norm_eps)
        return self.rms_r(x, w)

    def _heads_mm(self, a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """a [T, heads, K] @ w [heads, K, N] -> fp32 [T, heads, N]: on a device `mm` with the rows on the product's tile
        axis (an element is the sum the row pass's `attend_r` takes, without its 16-fold padding); else `mm` a row."""
        T, H, Kk = a.shape
        if self.fused:
            return K_.mm(a.transpose(0, 1), w).transpose(0, 1)
        return self.mm(a.reshape(T * H, 1, Kk), w).view(T, H, -1)

    def _mla(self, qa: torch.Tensor, qr: torch.Tensor, lat: torch.Tensor, ctl: torch.Tensor, sel: torch.Tensor,
             li: int) -> torch.Tensor:
        """Every row of a chunk over its own keys (kernels.py `mla_rows`; its torch reference off the device)."""
        a = self.cfg.attn(li)
        sink = self.sinks[li].reshape(-1)
        if self.fused:
            return K_.mla_rows(qa.contiguous(), qr.contiguous(), lat, ctl, sel, sink, a.qk ** -0.5)
        return K_.mla_rows_reference(qa, qr, lat, ctl, sel, sink, a.qk ** -0.5)

    def _sparse_reads(self, x: torch.Tensor, qres: torch.Tensor, li: int, st: Hy4EngState, slots: torch.Tensor,
                      cos: torch.Tensor, sin: torch.Tensor, L0: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """A sparse layer's read lists of a chunk's rows as `mla_rows` takes them: ctl int32 [T, 3] (base 0, the keys a
        row reads, whether it reads its list) and sel int32 [T, index_topk] (the positions, ascending; the keys past a
        row are the list's tail). The rule is each row's own: a row whose group is below `group_topk` reads every row
        up to itself. SPD-323 (F3): the indexer on `mm` and `wsum_relu` (`_select`'s products) in tiles of up to 256
        rows, over the groups closed before the tile's last row, then `pick_positions` (the shared block select).
        SPD-347 (`L0`, the chunk's first position on the host): the bounds from the host's positions (no device
        read), the rotary in place, and the scores with a tile's rows times heads on `mm`'s tile axis (the same
        element sums: `mm` computes every element alone, by K and the tile widths only)."""
        cfg, dev = self.cfg, self.device
        NS, rr, gt, T = cfg.index_topk, cfg.index_kpool, cfg.group_topk, slots.shape[0]
        fast = L0 is not None
        ctl = torch.zeros(T, 3, dtype=torch.int32, device=dev)
        ctl[:, 1] = (slots + 1).to(torch.int32)
        sel = torch.zeros(T, NS, dtype=torch.int32, device=dev)
        qg = slots // rr
        qlast = (L0 + T - 1) // rr if fast else int(qg[-1])
        if qlast < gt:
            return ctl, sel
        p_ = f"layers.{li}.self_attn.compressed_indexer."
        nh, D, rd = cfg.index_n_heads, cfg.index_head_dim, cfg.sparse_attn.rope
        iq = self.lin_prompt(qres, self.w(p_ + "query_projection.weight"))
        hw = self._rows_mm(x, self.w(p_ + "head_weight_projection.weight")).to(self.dtype)
        if fast:
            qi = iq.reshape(T, nh, D).contiguous()
            K_.rot_exact(qi, cos, sin, D - rd)
        else:
            qi = iq.view(T, nh, D)
            qi = torch.cat((qi[..., :D - rd], rotate(qi[..., D - rd:], cos[:, None], sin[:, None])), dim=-1)
        wsc = (hw.float() * (nh ** -0.5 * D ** -0.5)).contiguous()
        # a tile's scores are [rows, heads, groups] fp32: at most 512 MiB of them (an element's bits do not depend on the
        # tile: `mm` and `wsum_relu` compute every element alone)
        tile = max(1, min(256, (1 << 29) // (nh * (qlast + 1) * 4)))
        for i in range(0, T, tile):
            qt = qg[i:i + tile]
            t = qt.shape[0]
            G = (L0 + i + t - 1) // rr if fast else int(qt[-1])
            if G < gt:
                continue
            if fast:
                sc = self.mm(qi[i:i + tile].reshape(1, t * nh, D), st.gk[li][:G].t().unsqueeze(0), 64).view(t, nh, G)
            else:
                sc = self.mm(qi[i:i + tile], st.gk[li][:G].t().unsqueeze(0), 64)                    # [t, heads, G]
            if self.fused and nh & (nh - 1) == 0:
                logit = K_.wsum_relu(sc, wsc[i:i + tile])
            else:
                logit = self.mm(wsc[i:i + tile].unsqueeze(1), F.relu(sc)).view(-1, G)
            pos = pick_positions(torch.cat((logit, torch.zeros_like(logit[:, :1])), dim=1), qt, gt, rr)
            lst = qt >= gt
            sel[i:i + tile] = torch.where(lst[:, None], pos, sel[i:i + tile])
            ctl[i:i + tile, 1] = torch.where(lst, NS - (rr - 1) + slots[i:i + tile] % rr, slots[i:i + tile] + 1).to(torch.int32)
            ctl[i:i + tile, 2] = lst.to(torch.int32)
        return ctl, sel

    def _index_chunk_v2(self, x: torch.Tensor, qres: torch.Tensor, li: int, st: Hy4EngState, slots: torch.Tensor,
                        L0: int | None = None) -> None:
        """SPD-323 (F3): `_index_chunk` on the row kernels: the chunk's token keys by `mm` (rows on its tile axis) and the
        row kernel's norm, the pooled key of every group the chunk closes by the row pass's `group_keys_r`, turned at
        the group's first row. `L0`: the chunk's first position on the host (SPD-347: no device read)."""
        cfg, ip = self.cfg, f"layers.{li}.self_attn.compressed_indexer."
        rr, D, rd = cfg.index_kpool, cfg.index_head_dim, cfg.sparse_attn.rope
        tk = self._rows_mm(x, self.w(ip + "token_key_projection.weight")).to(self.dtype)
        st.ik[li][slots] = self._norm_v2(tk, self.w(ip + "token_key_norm.weight"))
        if L0 is not None:
            L = L0 + slots.shape[0]
        else:
            L0, L = int(slots[0]), int(slots[-1]) + 1
        g0, g1 = L0 // rr, L // rr
        if g1 <= g0:
            return
        gk = self.group_keys_r(st.ik[li][g0 * rr:g1 * rr].reshape(g1 - g0, rr, D), li)
        first = torch.arange(g0, g1) * rr
        cos, sin = self.rope(self.pos3(first + st.delta), SPARSE)
        keys = torch.cat((gk[:, :D - rd], rotate(gk[:, D - rd:], cos, sin)), dim=-1)
        st.ik[li][first.to(self.device) + rr - 1] = keys
        st.gk[li][g0:g1] = keys

    def _attention_chunk_v2(self, x: torch.Tensor, li: int, st: Hy4EngState, slots: torch.Tensor, rope: dict | None = None) -> torch.Tensor:
        """SPD-323 (F1): a chunk's attention, both kinds on `mla_rows`. The projections as the pass before (`lin`), the
        rotary by `rotate`, the norms on the row kernel; the query's absorbed key half by `mm` with the rows on the
        tile axis. A sparse layer's latents go into the state first and every row reads its own keys there in place; a
        window layer's rows read a buffer of the ring's last rows before the chunk and the chunk's own. The value
        half by `mm`, then the gate, `o_proj` and its norm."""
        cfg, dev = self.cfg, self.device
        a, kind, p_ = cfg.attn(li), cfg.layer_types[li], f"layers.{li}.self_attn."
        T, R, L0, Wn = x.shape[0], cfg.kv_lora_rank, st.length, cfg.sliding_window
        gate = self.lin_prompt(x, self.w(p_ + "linear_gate.weight"))
        qres = self._norm_v2(self.lin_prompt(x, self.w(p_ + "q_a_proj.weight")), self.w(p_ + "q_a_layernorm.weight"))
        q = self.lin_prompt(qres, self.w(p_ + "q_b_proj.weight")).view(T, a.heads, a.qk)
        ckv = self.lin_prompt(x, self.w(p_ + "kv_a_proj_with_mqa.weight"))
        fast = rope is not None and self.fused
        if fast:
            # SPD-347: the rotary and the latent's norm in place (`rot_exact`: `rotate`'s bits; `K_.norm` on the rows
            # where they sit: `_norm_v2`'s), no host read for the positions
            cos, sin = rope[kind]
            q = q.contiguous()
            K_.rot_exact(q, cos, sin, a.nope)
            qr = q[..., a.nope:]
            lat = ckv.contiguous()
            K_.norm(lat[:, :R], self.w(p_ + "kv_a_layernorm.weight"), cfg.rms_norm_eps, out=lat[:, :R])
            K_.rot_exact(lat.view(T, 1, lat.shape[1]), cos, sin, R)
        else:
            cos, sin = self.rope(self.pos3(slots.cpu() + st.delta), kind)
            qr = rotate(q[..., a.nope:], cos[:, None], sin[:, None])
            lat = torch.cat((self._norm_v2(ckv[:, :R], self.w(p_ + "kv_a_layernorm.weight")), rotate(ckv[:, R:], cos, sin)), dim=-1)
        qa = self._heads_mm(q[..., :a.nope], self.wk[li]).to(self.dtype)
        if kind == SPARSE:
            st.lat[li][slots] = lat
            # the host's first position only on the fast pass (the pass before keeps its own signature's calls)
            hk = {"L0": L0} if fast else {}
            self._index_chunk_v2(x, qres, li, st, slots, **hk)
            ctl, sel = self._sparse_reads(x, qres, li, st, slots, cos, sin, **hk)
            mix = self._mla(qa, qr, st.lat[li], ctl, sel, li)
            if li in self.pick_layers:
                # the last row's read list, for the draft head's layers
                pi, L, NS = st.pick_layers.index(li), L0 + T, cfg.index_topk
                lst = ((L - 1) // cfg.index_kpool >= cfg.group_topk) if fast else int(ctl[-1, 2])
                pq = sel[-1, :NS].long() if lst else self.ar(NS)
                st.pick_q[pi].copy_(pq)
                st.pick_ok[pi].copy_(pq <= L - 1)
        else:
            # the ring's Wn - 1 rows before the chunk (by position; those before row 0 are never read), then the chunk's
            before = torch.arange(L0 - Wn + 1, L0, device=dev)
            buf = torch.cat((st.win[li][before % Wn], lat))
            i = torch.arange(T, device=dev)
            n = (slots + 1).clamp(max=Wn)
            ctl = torch.stack((i + (Wn - 1 - slots).clamp(min=0), n, torch.zeros_like(n)), dim=1).to(torch.int32).contiguous()
            mix = self._mla(qa, qr, buf, ctl, torch.zeros(T, 1, dtype=torch.int32, device=dev), li)
            st.win[li][slots[-Wn:] % Wn] = lat[-Wn:]
        out = self._heads_mm(mix, self.wv[li]).to(self.dtype)
        if fast and self._ms:
            y = K_.mul_sigmoid(out.reshape(T, a.heads * a.v), gate)
        else:
            y = (out * torch.sigmoid(gate).view(T, a.heads, a.v)).reshape(T, a.heads * a.v)
        return self._norm_v2(self.lin_prompt(y, self.w(p_ + "o_proj.weight")), self.w(p_ + "post_proj_norm.weight"))

    def attend_chunk_latent(self, q: torch.Tensor, K: torch.Tensor, allowed: torch.Tensor, w: tuple,
                            return_fp32: bool = False) -> torch.Tensor:
        """q [t, heads, qk] over shared latents K [n, R + rope] -> latent mix [t, heads, R] before `wv`."""
        wk, _, sinks, a = w
        R = self.cfg.kv_lora_rank
        qa = torch.einsum("thd,hdr->thr", q[..., :a.nope], wk).float()
        Kf = K.float()
        s = (qa @ Kf[:, :R].T + q[..., a.nope:].float() @ Kf[:, R:].T) * (a.qk ** -0.5)        # [t, heads, n]
        s = s.masked_fill(~allowed[:, None, :], float("-inf"))
        s = torch.cat((s, sinks.expand(s.shape[0], -1, -1)), dim=-1)
        s = s - s.max(dim=-1, keepdim=True).values
        p = F.softmax(s, dim=-1)[..., :-1]
        mix = p @ Kf[:, :R]
        return mix if return_fp32 else mix.to(self.dtype)

    def attend_chunk(self, q: torch.Tensor, K: torch.Tensor, allowed: torch.Tensor, li: int, w: tuple | None = None) -> torch.Tensor:
        """q [t, heads, qk] over shared latents K [n, R + rope] with `allowed` [t, n] -> [t, heads, v]. `w`: another
        layer's absorbed halves, sinks and head shape (a draft layer reading this model's latents)."""
        w = w if w is not None else (self.wk[li], self.wv[li], self.sinks[li], self.cfg.attn(li))
        mix = self.attend_chunk_latent(q, K, allowed, w)
        return torch.einsum("thr,hrv->thv", mix, w[1])

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
        keys = torch.cat((gk[:, :D - rd], rotate(gk[:, D - rd:], cos, sin)), dim=-1)
        st.ik[li][first.to(self.device) + rr - 1] = keys
        st.gk[li][g0:g1] = keys

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
            routed = self._experts_chunk(z, idx, wts, li)
        else:
            routed = self._experts_grouped(z, idx, wts, li)
        routed = self.lin(routed, self.w(p + "routed_expert_up_proj.weight"))
        return self.rms(routed + self.dense(x, p + "shared_experts."), self.w(p + "combine_output_norm.weight"))

    def _group_inline(self, device) -> bool:
        """The prefill review (SPD-321): "group" only where the shared prefill kernel has its own inline kernels on this
        device. Without them `moe_grouped` walks its torch reference a pair at a time, far slower than the row kernels,
        so the chunk takes the row kernels there. Asked once an engine."""
        ok = getattr(self, "_group_ok", None)
        if ok is None:
            from tools import gpu as _gpu
            ok = self._group_ok = bool(torch.device(device).type == "cuda" and _gpu.profile(device).inline("fn_prefill_kernels"))
        return ok

    def _experts_chunk(self, z: torch.Tensor, idx: torch.Tensor, wts: torch.Tensor, li: int) -> torch.Tensor:
        """A prompt chunk's routed experts on the layer's kernel bank, combined -> [T, latent]. "group" at `experts_min`
        rows or more: one call of the shared prefill kernel for the whole chunk (an expert's tiles decoded once an item
        of GROUP_ROWS pairs) with the SwiGLU clamp between its two GEMMs, where the row kernels clamp between their
        launches. Otherwise the row kernels in EXPERT_SUB launches, grouped by HY4_PREFILL_GROUPED. By the code both
        apply the same products in the same order a pair (tools/hy4_check.py --only experts-equal compares every MoE
        layer on the device)."""
        if self.prefill_experts == "group" and z.shape[0] >= self.experts_min and self._group_inline(z.device):
            from tools import fn_prefill_kernels as PK
            ids = idx.to(torch.int32).contiguous()
            yd = PK.moe_grouped(z.contiguous(), ids, self.banks[li], rows=GROUP_ROWS, clamp=self.cfg.swiglu_limit)
            return self.combine(yd, idx, wts)
        B, gr = EXPERT_SUB, self.prefill_grouped
        return torch.cat([self.combine(self.expert_pairs(z[i:i + B], idx[i:i + B], li, grouped=gr), idx[i:i + B], wts[i:i + B])
                          for i in range(0, z.shape[0], B)])

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
    def check_rows(self) -> int:
        """Prompt rows a self-check needs to reach every read path: past the sparse layers' `index_topk` (their
        rows then read selected groups) and with the window layers' rings wrapped, ending inside a group."""
        c = self.cfg
        return c.index_topk + c.sliding_window + c.index_kpool + 1

    def copy_state(self, src: Hy4EngState, dst: Hy4EngState) -> None:
        """`dst` becomes `src` (the rows up to its length, the rings, the last tokens, the last row's picks)."""
        n = src.length
        dst.lat_all[:, :n] = src.lat_all[:, :n]
        dst.ik_all[:, :n] = src.ik_all[:, :n]
        dst.win_all.copy_(src.win_all)
        dst.hist.copy_(src.hist)
        dst.length, dst.delta = n, src.delta
        dst.pick_q.copy_(src.pick_q)
        dst.pick_ok.copy_(src.pick_ok)
        dst.touched()
        dst.picks_at = src.picks_at

    @torch.no_grad()
    def self_check(self, prompt: list[int], steps: int = 12, widths=(2, 4, 8), log=print, graphs: bool = False) -> dict:
        """A prompt, then `steps` greedy decode steps recorded; the same tokens replayed as chain blocks of
        every width and as trees with a wrong branch: every row must be the decode step's bit for bit.
        `covers` says which read paths the replay reached: a prompt shorter than `check_rows()` leaves the
        group selection or the wrapped rings unchecked."""
        c = self.cfg
        base_st = self.new_state(len(prompt) + steps + BLOCK + 8)
        lg = self.prefill(torch.tensor(prompt), base_st)
        base = base_st.length
        st = self.new_state(base_st.cap)
        self.copy_state(base_st, st)
        # `graphs`: the replay runs the captured graphs, the recorded decode stays on the plain pass, so a row
        # that differs names a graph that does not compute the plain pass's bits
        gs = {} if graphs and self.device.type == "cuda" else None
        toks, want = [], []
        nxt = int(lg[0].argmax())
        for _ in range(steps):
            toks.append(nxt)
            lg = self.step(nxt, st)
            want.append(lg[0].clone())
            nxt = int(lg[0].argmax())
        rows, bad, closed = 0, [], 0
        rr = c.index_kpool
        for wdt in widths:
            self.copy_state(base_st, st)
            for i in range(0, steps, wdt):
                blk = toks[i:i + wdt]
                # groups that close inside the block before its last row: scored from the block's own rows
                closed += sum(1 for j in range(len(blk) - 1) if (st.length + j) % rr == rr - 1)
                got, _ = self.rows(blk, list(range(-1, len(blk) - 1)), st, graphs=gs)
                for j in range(len(blk)):
                    rows += 1
                    if not torch.equal(got[j], want[i + j]):
                        bad.append(("chain", wdt, i + j))
                self.commit(st, list(range(len(blk))))
        # trees: after the anchor a wrong child with its own child, then the right branch
        self.copy_state(base_st, st)
        i = 0
        while i + 3 <= steps:
            wrong = (toks[i + 1] + 1) % c.vocab_size
            got, _ = self.rows([toks[i], wrong, toks[i + 2], toks[i + 1], toks[i + 2]], [-1, 0, 1, 0, 3], st, graphs=gs)
            for j, r in ((0, 0), (1, 3), (2, 4)):
                rows += 1
                if not torch.equal(got[r], want[i + j]):
                    bad.append(("tree", 5, i + j))
            self.commit(st, [0, 3, 4])
            i += 3
        ok = not bad
        covers = {"prompt_rows": base, "selection": base // rr >= c.group_topk, "rings_wrapped": base > c.sliding_window,
                  "groups_closed_in_blocks": closed}
        log(f"[hy4] self-check: {rows} rows against {steps} decode steps after {base} prompt rows "
            f"(group selection {'reached' if covers['selection'] else 'NOT reached'}, rings "
            f"{'wrapped' if covers['rings_wrapped'] else 'NOT wrapped'}, {closed} groups closed inside blocks): "
            + ("every row equal" if ok else f"{len(bad)} rows differ, first {bad[0]}"))
        return {"ok": ok, "rows": rows, "bad": bad[:8], "steps": steps, "covers": covers, "graphs": len(gs) if gs is not None else None}
