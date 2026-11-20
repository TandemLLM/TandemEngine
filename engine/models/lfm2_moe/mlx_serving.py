"""LFM2 MoE behind the server's generation loop on Apple silicon: `Lfm2MoeMLX` (mlx_board.py) in the
interface `server/app.py` drives every model through (the CUDA adapter's, serving.py).

  forward, one token      `Lfm2MoeMLX.decode`
  forward, more tokens    `Lfm2MoeMLX.prefill` (chain blocks of PREFILL_CHUNK rows)
  forward_tree            `Lfm2MoeMLX.rows`: logits [M, vocab], the greedy picks
  commit_tree             `Lfm2MoeMLX.commit`
  forward_block           a chain verify whose commit waits for `rollback_to` or the next call

Logits, picks and hidden rows cross to the server as torch tensors on the CPU (the device the server is
given), copied out of MLX's unified memory once a call.

Exactness: a verified row is the decode step's row bit for bit, as on the CUDA board. MLX's own matmuls
round by row count (tools/gpu/apple.py), so the board computes every row with its own Metal kernels
(mlx_kernels.py), in an order fixed by the weight's shape. The load-time check (`self_check`) compares
verified rows with the decode steps' (a greedy chain, then chains and a tree after a longer prompt);
speculation runs only when every row has the decode step's bits. `exact` in the weights info says so.

Off for now (`caps`): the prefix, session and disk caches, the KV pool, side slots, joint passes.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from engine.methods.manifest import Manifest
from engine.models.adapter import Capabilities, CostModel
from engine.models.lfm2_moe.serving import CONDUCTOR_FIT
from engine.models.state import SectionState

PREFILL_CHUNK = 4096                 # measured on the M5 Max: 4,096 rows a chunk prefill fastest (7.2k tok/s at 16k)
MODEL_ID = "lfm2.5-24b-a2b"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODULES = ("engine/models/lfm2_moe/config.py", "engine/models/lfm2_moe/ref.py",
           "engine/models/lfm2_moe/mlx_board.py", "engine/models/lfm2_moe/mlx_kernels.py",
           "engine/models/lfm2_moe/mlx_serving.py")


def manifest() -> Manifest:
    # MLX rows are not the CUDA board's: their own arch, so no cache entry crosses between the two
    return Manifest(arch="lfm2_moe-mlx", modules=MODULES)


def _t(a) -> torch.Tensor:
    import mlx.core as mx
    return torch.from_numpy(np.array(a.astype(mx.float32) if a.dtype != mx.float32 else a))


class _KV:
    """The `kv` the server reads: the length once pending work commits, the capacity."""

    pages = None

    def __init__(self, s: "Lfm2MoeMLXServing"):
        self._s = s

    @property
    def length(self) -> int:
        return self._s.st.length + (self._s._chain or 0)

    @length.setter
    def length(self, v: int) -> None:
        v = int(v)
        if v == 0:
            self._s.restart()
            return
        if v != self._s.st.length:
            # the conv states are not indexed by row: a length alone cannot move them (`resume_at`)
            raise ValueError(f"the LFM2 (MLX) state holds {self._s.st.length} rows and cannot be set to {v}")

    @property
    def max_len(self) -> int:
        return self._s.max_len


class Lfm2MoeMLXServing:
    caps = Capabilities(caches=False, side_slots=False, anchor_grid=None, shadow=False, joint_verify=False,
                        side_lane=False, joint_plain=False, conductor=CONDUCTOR_FIT)

    def __init__(self, board, max_len: int):
        self.fn, self.cfg, self.max_len = board, board.cfg, max_len
        self.st = board.new_state(max_len)
        self.kv = _KV(self)
        self.picks = None
        self.tap = None
        self.mm = None
        self.pos_delta = 0
        self._hidden = None
        self._chain = None
        self._mark = None                # (length, conv states) at the last `settle`
        self.commit_ms = 0.0
        self.captured = False
        self.plain_captured = False
        self.heads: dict = {}
        self.stats = {"prefill_rows": 0, "decode_steps": 0, "verify_blocks": 0, "verify_rows": 0}

    # --- the loop's interface -------------------------------------------------------------------
    @property
    def hidden_post_norm(self) -> torch.Tensor:
        return self._hidden

    def price_class(self, n: int) -> int:
        """The prices' context class: 2,048 keys, doubling (the CUDA board's classes)."""
        c, n = 2048, max(1, int(n))
        while c < n and c < self.max_len:
            c *= 2
        return c

    def last_streams(self) -> torch.Tensor:
        return self._hidden[0, -1:] if self._hidden is not None else torch.zeros(1, self.cfg.hidden_size)

    def cost(self) -> CostModel:
        c = self.cfg
        sec = c.conv_state_bytes(2)
        return CostModel(section_bytes=sec, view_bytes_per_token=c.kv_bytes_per_token(2),
                         parts={"recurrent": 0, "conv": sec})

    def manifest(self) -> Manifest:
        return manifest()

    def precapture(self, drafter) -> int | None:
        self.kv.length = 0
        return None

    def draft_heads(self) -> dict:
        return self.heads

    def settle(self) -> None:
        """Pending work in, and this point kept: `resume_at` can come back to it (the server's in-place
        ask, `_ask_in_place`: its `cache_state` snapshot is empty here, the conv states are MLX arrays)."""
        self._settle_chain()
        self._mark = (self.st.length, dict(self.st.conv))

    def cache_state(self):
        return SectionState({})

    def resume_at(self, n: int) -> None:
        """Back at `n` rows: where the state is, empty, or the point `settle` last kept (its conv states
        put back; the KV rows above `n` are written over by the next forward). Any other `n` is refused:
        the conv states of an arbitrary row are not kept."""
        self.discard_pending()
        n = int(n)
        if n == self.st.length and (self._mark is None or self._mark[0] != n):
            return
        if n == 0:
            self.restart()
        elif self._mark is not None and self._mark[0] == n:
            self.st.length, self.st.conv = n, dict(self._mark[1])
        else:
            raise ValueError(f"the LFM2 (MLX) state holds {self.st.length} rows and kept no point at {n}")

    def memory_parts(self) -> dict:
        w = sum(int(a.nbytes) for a in self.fn.W.values())
        st = self.st
        kv = sum(int(a.nbytes) for d in (st.k, st.v) for a in d.values())
        return {"weights": w, "kv_pool": kv, "drafter_kv": 0, "gdn_state": sum(int(a.nbytes) for a in st.conv.values())}

    def resident_files(self) -> list:
        return []

    def discard_pending(self) -> None:
        self._chain = None
        self.picks = None

    def trim_pending(self, n: int) -> bool:
        n, L = int(n), self.st.length
        if self._chain is None:
            return n == L
        if not L <= n <= L + self._chain:
            return False
        self._chain = None
        if n > L:
            self.commit_tree(list(range(n - L)))
        return True

    def window_ok(self, i: int) -> bool:
        return False

    def reset(self) -> None:
        self.restart()

    def restart(self) -> None:
        self._chain = None
        self._mark = None
        self.picks = None
        self.st.reset_inplace()

    def tree_batchable(self, parents) -> bool:
        return False

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, start: int, last_only: bool = True) -> torch.Tensor:
        """Rows `ids` at positions start..: logits [1, 1, vocab] of the last row (fp32)."""
        self._settle_chain()
        ids = ids.view(-1)
        self._at(start)
        T = ids.numel()
        if self.st.length + T > self.max_len:
            raise ValueError(f"{self.st.length + T} rows exceed the window {self.max_len}")
        if T == 1:
            lg = self.fn.decode(int(ids[0]), self.st)
            self.stats["decode_steps"] += 1
        else:
            lg = self.fn.prefill(ids.tolist(), self.st, chunk=PREFILL_CHUNK, last_only=True)
            self.stats["prefill_rows"] += T
            self._hidden = _t(self.fn.prompt_hidden).unsqueeze(0)          # [1, T, H]
            return _t(lg[-1:]).view(1, 1, -1)
        self._hidden = _t(self.fn.hidden[-1:]).view(1, 1, -1)
        return _t(lg[-1:]).view(1, 1, -1)

    @torch.no_grad()
    def forward_tree(self, block: torch.Tensor, parents: list[int], start: int) -> torch.Tensor:
        """A draft block in DFS pre-order (row 0 the anchor at `start`): every row's logits [M, vocab]
        fp32, its greedy pick in `picks`; nothing but the block's KV rows written until `commit_tree`."""
        self._settle_chain()
        self._at(start)
        toks = [int(t) for t in block.view(-1).tolist()]
        M = len(toks)
        if self.st.length + M > self.max_len:
            raise ValueError(f"{self.st.length + M} rows exceed the window {self.max_len}")
        from engine.models.lfm2_moe.mlx_kernels import ROWS_MAX
        if M > ROWS_MAX:
            # past it the board batches a chain on MLX's own matmul, whose rows are not the decode step's
            raise ValueError(f"a verify block of {M} rows exceeds the row kernels' {ROWS_MAX}")
        lg = self.fn.rows(toks, [int(p) for p in parents], self.st)
        out = _t(lg)
        self.picks = out.argmax(-1)
        self._hidden = _t(self.fn.hidden).unsqueeze(0)
        self.stats["verify_blocks"] += 1
        self.stats["verify_rows"] += M
        return out

    @torch.no_grad()
    def commit_tree(self, path: list[int]) -> None:
        t0 = time.perf_counter()
        self.fn.commit(self.st, [int(p) for p in path])
        self.commit_ms = (time.perf_counter() - t0) * 1e3

    @torch.no_grad()
    def forward_block(self, block: torch.Tensor, start: int) -> torch.Tensor:
        n = block.numel()
        lg = self.forward_tree(block, list(range(-1, n - 1)), start)
        self._chain = n
        return lg

    @torch.no_grad()
    def rollback_to(self, keep: int) -> None:
        self._chain = None
        self.commit_tree(list(range(int(keep))))

    # --- internals -------------------------------------------------------------------------------
    def _settle_chain(self) -> None:
        if self._chain is not None:
            n, self._chain = self._chain, None
            self.commit_tree(list(range(n)))

    def _at(self, start: int) -> None:
        if int(start) != self.st.length:
            raise ValueError(f"LFM2 forward at {start}, the state holds {self.st.length} rows")


#: the self-check's chain blocks after its long prompt. Each size is a kernel variant of its own on the served
#: shapes: 2 rows the scalar dense kernel and the 2-row attention; 3 and 4 the 4-row attention groups and
#: router; 5 and 9 one and two tiles of the matrix units; 16 and 32 the widths the cut verifies most and last
CHECK_CHAINS = (2, 3, 4, 5, 9, 16, 32)
#: and a tree: rows that share ancestors, so the conv's windows and the attention's paths are gathered
CHECK_TREE = ([7, 19, 23, 44, 31, 60], [-1, 0, 1, 1, 0, 4])
#: its long prompt: past the attention kernel's 256-token chunk, so a row's attention is reduced over chunks
CHECK_LONG = 300


def self_check(board, prompt: list[int], steps: int = 24, log=print) -> dict:
    """The load-time check of the verify path, bit for bit against the decode step.

    1. After `prompt`: `steps` greedy tokens decoded one at a time, then verified as one chain block.
    2. After a `CHECK_LONG`-token prompt (`prompt` repeated): chains of `CHECK_CHAINS` rows and the tree
       `CHECK_TREE`, each row against the decode of its path.

    `ok` when every row's logits are the decode step's, byte for byte: the row kernels hold that by
    construction on the chip they were measured on (mlx_kernels.py), and a chip, macOS or MLX that rounds
    one of them another way must not draft. On a failure `first_diff` is the first differing row (in the
    order above), `max_diff` the largest logit gap and `picks` whether the greedy picks at least agreed."""
    import mlx.core as mx

    def at(tokens):
        st = board.new_state(len(tokens) + max(steps, *CHECK_CHAINS) + 8)
        lg = board.prefill(tokens, st, last_only=True)
        return st, lg, st.length, dict(st.conv)

    def back(st, L, conv):
        st.length, st.conv = L, dict(conv)

    seen = {"rows": 0, "first": None, "max": 0.0, "picks": True}

    def hold(got, want):
        if got.tobytes() != want.tobytes():
            if seen["first"] is None:
                seen["first"] = seen["rows"]
            seen["max"] = max(seen["max"], float(np.abs(got.astype(np.float64) - want.astype(np.float64)).max()))
            seen["picks"] = seen["picks"] and int(got.argmax()) == int(want.argmax())
        seen["rows"] += 1

    def greedy(st, first: int, n: int):
        toks, rows = [first], []
        for _ in range(n):
            rows.append(np.array(board.decode(toks[-1], st)[0]))
            toks.append(int(rows[-1].argmax()))
        return toks[:n], rows

    # 1: the greedy chain after the chat prompt
    st, lg, L, conv = at(prompt)
    toks, want = greedy(st, int(mx.argmax(lg[-1]).item()), steps)
    back(st, L, conv)
    got = np.array(board.rows(toks, list(range(-1, steps - 1)), st))
    for i in range(steps):
        hold(got[i], want[i])
    # 2: the blocks the cut serves, after a prompt of more than one attention chunk
    long_prompt = (list(prompt) * (CHECK_LONG // max(1, len(prompt)) + 1))[:CHECK_LONG]
    st, lg, L, conv = at(long_prompt)
    toks, want = greedy(st, int(mx.argmax(lg[-1]).item()), max(CHECK_CHAINS))
    for m in CHECK_CHAINS:
        back(st, L, conv)
        got = np.array(board.rows(toks[:m], list(range(-1, m - 1)), st))
        for i in range(m):
            hold(got[i], want[i])
    t_toks, t_par = CHECK_TREE
    back(st, L, conv)
    got = np.array(board.rows(t_toks, t_par, st))
    for row in range(len(t_toks)):
        path, r = [], row
        while r >= 0:
            path.append(r)
            r = t_par[r]
        back(st, L, conv)
        for r in reversed(path):
            out = np.array(board.decode(t_toks[r], st)[0])
        hold(got[row], out)
    ok = seen["first"] is None
    res = {"ok": ok, "bitwise": ok, "rows": seen["rows"], "first_diff": seen["first"], "max_diff": seen["max"],
           "picks": seen["picks"]}
    if log:
        log(f"[lfm-mlx-selfcheck] {'PASS' if ok else 'FAIL: speculation stays off'}: {res}")
    return res


#: the one profile this serving has: the checkpoint as it is, in BF16
PROFILE = "lfm25-bf16"


def profile_of(path: str) -> str:
    """The weights profile the price file is named by: QSE_WEIGHTS when a launcher set it, else BF16's."""
    return os.environ.get("QSE_WEIGHTS") or PROFILE


def prices(path: str, log=print):
    """StairCut's prior on this chip (ops/prices/lfm2.5-24b-a2b/<profile>.<chip key>.json); none yet:
    the cut learns online."""
    from engine.methods.device import device_id
    from engine.methods.staircut import PriceModel
    return PriceModel.load(os.path.join(ROOT, "ops", "prices", MODEL_ID, f"{profile_of(path)}.{device_id()}.json"),
                           log=log)


def lookup_drafter(srv, path: str, corpus: str | None, log=print):
    """The lookup cut by StairCut, as on the CUDA board (serving.lookup_drafter)."""
    from engine.drafters.ngram import NgramDrafter
    from engine.methods.mtp_drafter import MTPTreeDrafter, NoHead
    from engine.methods.staircut import StairCut
    from engine.tokfp import fingerprint
    ng = NgramDrafter(corpus_path=corpus or "", tokenizer_sha=fingerprint(path), min_order=3, max_depth=16,
                      min_corpus_order=8)
    log(f"[server] LFM2 (MLX) lookup drafter: corpus {(corpus or 'none') if ng.corpus is not None else 'none'}")
    return MTPTreeDrafter(srv, NoHead(), StairCut(prices(path, log), siblings=False), ng)


def serve(a, ctx) -> dict:
    """LFM2 MoE from a BF16 checkpoint on the Mac's GPU (the registry's `serve` on Apple silicon,
    dispatch.py). `--drafter` other than none: the lookup cut by StairCut, gated by `self_check`."""
    from transformers import AutoTokenizer

    from engine.models.lfm2_moe import mlx_board
    from engine.spec import Relax
    ctx.keys(a.model, [a.model])
    if a.drafter not in ("none", "mtp"):
        ctx.log(f"[server] LFM2 MoE drafts with the lookup: --drafter {a.drafter} taken as the lookup")
    if getattr(a, "kv_pool", "off") == "on":
        ctx.log("[server] LFM2 (MLX): no KV pool on Apple silicon yet; the window is allocated whole")
    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    _, board = mlx_board.load(a.model, log=ctx.log)
    load_s = round(time.time() - t0, 2)
    ctx.mark("weights")
    srv = Lfm2MoeMLXServing(board, a.max_len)
    ctx.mark("engine")
    text = tok.apply_chat_template([{"role": "user", "content": "Count from one to twenty in words."}],
                                   tokenize=False, add_generation_prompt=True)
    res = self_check(board, tok(text, add_special_tokens=False)["input_ids"], log=ctx.log)
    drafter = None
    if a.drafter != "none":
        if res["ok"]:
            drafter = lookup_drafter(srv, a.model, a.corpus, ctx.log)
            a.k = max(int(a.k or 0), drafter.cut.max_rows - 1)
        else:
            ctx.log("[server] LFM2 (MLX) self-check FAILED: speculation off, plain decode only")
    ctx.mark("drafters")
    describe = (f"LFM2 MoE {a.model} (bf16, MLX on {_chip()}): load {load_s} s, drafter "
                f"{'lookup + StairCut' if drafter is not None else 'none'}, max_len {a.max_len:,},")
    return {"engine": srv, "tok": tok, "drafter": drafter, "path": a.model, "device": "cpu",
            "cfg": None, "w": None, "describe": describe,
            "serving": {"tree": drafter is not None, "sampled_tree": "det", "relax": Relax(1.0, 1),
                        "weights_info": {"format": "bf16", "weights": a.model, "backend": "mlx",
                                         "drafters": "lookup (no draft head)" if drafter is not None else None,
                                         "load_s": load_s, "self_check": res, "exact": res["bitwise"]}}}


def _chip() -> str:
    from tools.gpu.apple import chip_name
    return chip_name() or "Apple silicon"
