"""Hy4-flash in the server's engine interface: `Hy4FlashEngine` with one state, the loop's verbs and the
cold verbs the caches and the memory split read.

What a cache keeps of a sequence (`Hy4CacheState`, engine/models/state.py):

  sections   copied whole: the window layers' rings (`win.<layer>`, [sliding_window, 576] each: the last
             512 rows' latents by slot mod 512) and the last two tokens (`engram.hist`), which the
             Engram layers hash with the next one
  views      sliced by length, token axis 0: the sparse layers' latents (`lat.<layer>`, 576 values a
             row) and their indexer keys (`ik.<layer>`, 128 values a row: a row that closes a group
             holds the group's pooled key, any other its own token key)

A ring cannot be sliced by length, so a restore at n takes the sections of a snapshot made at n, as a
recurrent state's. A verify block writes neither: its rows sit in scratch buffers until `commit_tree`.
"""

from __future__ import annotations

import os
import time

import torch

from engine.methods.manifest import Manifest
from engine.models.adapter import Capabilities, CostModel
from engine.models.hy4_flash.engine import BLOCK, PREFILL_CHUNK, Hy4FlashEngine
from engine.models.state import SectionState

MODULES = ("engine/models/hy4_flash/__init__.py", "engine/models/hy4_flash/config.py", "engine/models/hy4_flash/engine.py",
           "engine/models/hy4_flash/engram.py", "engine/models/hy4_flash/ref.py", "engine/models/hy4_flash/knobs.py",
           "engine/models/hy4_flash/dfly.py",
           "engine/models/flashnext/ref.py", "engine/models/flashnext/ple.py", "engine/settings.py")
#: the model's name in ops/prices/ and models/
MODEL_ID = "hy4-flash"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def manifest() -> Manifest:
    return Manifest(arch="hy_v4_flash", modules=MODULES, data=())


class _KV:
    """The `kv` the server reads: the sequence's length (a pending chain commit counted) and its capacity.
    Zero starts a new sequence in place; any other length keeps the by-slot rows by length -- the rings
    and the last tokens are the caller's to restore first (`state.restore`)."""

    pages = None

    def __init__(self, serving: "Hy4FlashServing"):
        self._s = serving

    @property
    def length(self) -> int:
        s = self._s
        return s.st.length + (s._chain or 0)

    @length.setter
    def length(self, v: int) -> None:
        v = int(v)
        if v == 0:
            self._s.restart()
            return
        if not 0 < v <= self.max_len:
            raise ValueError(f"the Hy4-flash state cannot hold {v} rows (capacity {self.max_len})")
        self._s._chain = None
        self._s.st.length = v

    @property
    def max_len(self) -> int:
        return self._s.max_len


class Hy4CacheState(SectionState):
    def __init__(self, serving: "Hy4FlashServing"):
        self.s = serving

    def sections(self) -> dict:
        st = self.s.st
        out = {f"win.{li}": st.win[li] for li in sorted(st.win)}
        out["engram.hist"] = st.hist
        return out

    def views(self) -> list:
        st = self.s.st
        out = []
        for li in sorted(st.lat):
            out += [(f"lat.{li}", st.lat[li], 0), (f"ik.{li}", st.ik[li], 0)]
        return out


class Hy4FlashServing:
    """`Hy4FlashEngine` with one state, in the server's engine interface."""

    # the caches on the prefill's grid; one sequence at a time until the row path has its kernels and
    # graphs (no side slots, no joint passes, no side lane); the shadow reader was validated on Qwen text
    caps = Capabilities(caches=True, side_slots=False, anchor_grid=PREFILL_CHUNK, shadow=False, joint_verify=False,
                        side_lane=False, joint_plain=False)

    def __init__(self, eng: Hy4FlashEngine, max_len: int):
        self.fn, self.cfg, self.max_len = eng, eng.cfg, int(max_len)
        self.st = eng.new_state(self.max_len)
        self.state = Hy4CacheState(self)
        self.kv = _KV(self)
        self.picks = None
        self.tap = None
        self.mm = None                      # the vision tower is not served yet: the image path is off
        self.pos_delta = 0
        self._hidden = None
        self._chain = None
        self.commit_ms = 0.0
        self.captured = False
        self.plain_captured = False
        self.heads: dict = {}
        self.stats = {"prefill_rows": 0, "decode_steps": 0, "verify_blocks": 0, "verify_rows": 0}

    # --- cold verbs ------------------------------------------------------------------------------
    def hidden_post_norm(self) -> torch.Tensor:
        return self._hidden

    def last_streams(self) -> torch.Tensor:
        """The last row's head input [1, H] (what the draft layers fuse with the next token)."""
        return self._hidden[0, -1:] if self._hidden is not None else torch.zeros(1, self.cfg.hidden_size)

    def cost(self) -> CostModel:
        st, c = self.st, self.cfg
        sec = sum(t.numel() * t.element_size() for t in self.state.sections().values())
        row = sum(t.numel() // t.shape[ax] * t.element_size() for _, t, ax in self.state.views())
        return CostModel(section_bytes=sec, view_bytes_per_token=row, parts={"recurrent": sec, "conv": 0})

    def manifest(self) -> Manifest:
        return manifest()

    def precapture(self, drafter) -> int | None:
        self.kv.length = 0
        return None

    def draft_heads(self) -> dict:
        return self.heads

    def settle(self) -> None:
        self._settle_chain()

    def cache_state(self) -> Hy4CacheState:
        return self.state

    def resume_at(self, n: int) -> None:
        self.discard_pending()
        self.kv.length = n

    def memory_parts(self) -> dict:
        fn = self.fn
        seen, w = set(), 0
        objs = list(fn.W.values()) + list(fn.wk.values()) + list(fn.wv.values())
        for bank in fn.banks.values():
            objs += list(bank.keep)
        for o in objs:
            for t in _tensors(o):
                s = t.untyped_storage()
                if s.data_ptr() not in seen:
                    seen.add(s.data_ptr())
                    w += s.nbytes()
        st = self.st
        return {"weights": w, "kv_pool": sum(int(t.numel() * t.element_size()) for d in (st.lat, st.ik) for t in d.values()),
                "drafter_kv": 0, "gdn_state": sum(int(t.numel() * t.element_size()) for t in st.win.values())}

    def resident_files(self) -> list:
        """The Engram tables read through the page cache (a warm table's bytes count against the floor)."""
        return [f for t in self.fn.tables.values() for f in getattr(t, "resident_files", lambda: [])()]

    def discard_pending(self) -> None:
        self._chain = None
        self.picks = None

    def trim_pending(self, n: int) -> bool:
        """Back to length `n` inside work not committed yet (a chain block whose commit waits)."""
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
        """A new sequence in the same buffers: the rings and the last tokens cleared, the length 0."""
        self._chain = None
        self.picks = None
        self.st.reset_inplace()

    # --- the loop's verbs ------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, ids: torch.Tensor, start: int, last_only: bool = True) -> torch.Tensor:
        """Rows `ids` at positions start..: one token through the row path, more through the prefill.
        Logits [1, 1, vocab] of the last row (fp32)."""
        self._settle_chain()
        ids = ids.view(-1)
        self._at(start)
        T = ids.numel()
        if T == 1 and start > 0:
            lg, n = self.fn.rows([int(ids[0])], [-1], self.st)
            self.fn.commit(self.st, [0])
            self._hidden = n.view(1, 1, -1)
            self.stats["decode_steps"] += 1
            return lg.view(1, 1, -1)
        lg = self.fn.prefill(ids, self.st)
        self._hidden = self.fn.prompt_hidden.unsqueeze(0)
        self.stats["prefill_rows"] += T
        return lg.view(1, 1, -1)

    @torch.no_grad()
    def forward_tree(self, block: torch.Tensor, parents: list[int], start: int) -> torch.Tensor:
        """A draft block in DFS pre-order (row 0 the anchor at `start`): every row's logits [M, vocab]
        fp32, its greedy pick in `picks`. Nothing is written to the state until `commit_tree`."""
        self._settle_chain()
        self._at(start)
        toks = [int(t) for t in block.view(-1).tolist()]
        lg, n = self.fn.rows(toks, [int(p) for p in parents], self.st)
        self.picks = lg.argmax(dim=-1)
        self._hidden = n.unsqueeze(0)
        self.stats["verify_blocks"] += 1
        self.stats["verify_rows"] += len(toks)
        return lg

    def tree_batchable(self, parents) -> bool:
        return len(parents) <= BLOCK

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
            raise ValueError(f"Hy4-flash forward at {start}, the state holds {self.st.length} rows")


def _tensors(o) -> list:
    """The tensors a weight entry holds: a tensor, an EXL3 projection's, a dense expert bank's."""
    if torch.is_tensor(o):
        return [o]
    if hasattr(o, "suh"):
        return [o.t, o.suh, o.svh]
    if getattr(o, "dense", None) is not None:
        return [o.dense]
    return []
