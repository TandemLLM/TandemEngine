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

import dataclasses
import json
import os
import time

import torch

from engine.methods.manifest import Manifest
from engine.methods.side_slot import ensure_rows, lane_drafter, open_pool, side_slot_bytes
from engine.models.adapter import Capabilities, CostModel
from engine.models.graph_set import GraphSet
from engine.models.hy4_flash.engine import BLOCK, PREFILL_CHUNK, Hy4FlashEngine
from engine.models.state import SectionState
from engine.settings import SETTINGS as _S

MODULES = ("engine/models/hy4_flash/__init__.py", "engine/models/hy4_flash/config.py", "engine/models/hy4_flash/engine.py",
           "engine/models/hy4_flash/engram.py", "engine/models/hy4_flash/ref.py", "engine/models/hy4_flash/knobs.py",
           "engine/models/hy4_flash/kernels.py", "engine/models/hy4_flash/native.py",
           "engine/models/hy4_flash/dfly.py", "engine/models/hy4_flash/table.py", "engine/models/hy4_flash/pack.py",
           "engine/models/hy4_flash/load.py",
           "engine/models/flashnext/ref.py", "engine/models/flashnext/ple.py", "engine/settings.py")
#: the model's name in ops/prices/ and models/
MODEL_ID = "hy4-flash"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


#: the tables the arithmetic reads (a dense split is a summation order)
TABLES = ("ops/hy4-dense-splits.json",)
#: SRV-905: the rows a second a prompt is read at through the server (measured, docs/hy4-flash.md "Limits"): the
#: prior the router's guard, the disk tier and the spill price a read at until a request has measured one, in place
#: of the 27B's 740 (a 2,345-row question was priced at 3.2 s and took 18.8)
PREFILL_TOK_S = 140.0


def manifest() -> Manifest:
    return Manifest(arch="hy_v4_flash", modules=MODULES, data=TABLES)


class _KV:
    """The `kv` the server reads: the sequence's length (a pending chain commit counted) and its capacity.
    Zero starts a new sequence in place; any other length keeps the by-slot rows by length -- the rings
    and the last tokens are the caller's to restore first (`state.restore`)."""

    def __init__(self, serving: "Hy4FlashServing"):
        self._s = serving

    @property
    def pages(self):
        """ENG-485: the KV pool's paged set of the state's latents and indexer keys, or None (the pool off)."""
        return self._s.st.pages

    def ensure(self, rows: int) -> None:
        """ENG-485: rows [0, rows) committed before a pass writes or reads them, with the indexer's read floor (the
        round runner asks before every lane's pass: a pool that cannot give them stops that lane, not the round)."""
        self._s._ensure(rows)

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
        self._s.st.touched()

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
        """ENG-485: the sparse layers' latents and indexer keys as two whole tensors, row axis 1 ([layers, rows,
        dim]): the caches find a paged buffer by its tensor's base address (engine/kvpool.py), which a layer's
        slice does not have."""
        st = self.s.st
        return [("lat", st.lat_all, 1), ("ik", st.ik_all, 1)]

    def restore(self, snap) -> None:
        super().restore(snap)
        self.s.st.touched()


class Hy4FlashServing:
    """`Hy4FlashEngine` with one state, in the server's engine interface."""

    # the caches on the prefill's grid; side slots, each the full window through the KV pool, whose lanes take
    # turns (ENG-485: `joint_verify` False, each lane its own verify; no joint pass, no side lane); the shadow
    # reader was validated on Qwen text. The router's notes have no place in its prompt (`notes: False`, as
    # Flint's): it asks for none (SRV-899)
    caps = Capabilities(caches=True, side_slots=True, anchor_grid=PREFILL_CHUNK, shadow=False, joint_verify=False,
                        side_lane=False, joint_plain=False, conductor={"notes": False})

    # the conversation cache on one lane (SRV-907): a chat's next turn is never a guest of its own rows, and
    # the conductor's question is a guest at any size (engine/cache.py `ResidentPrefix`, `chat_first`); a
    # prompt without a plan is marked where its first user turn starts, so a repeated head is read once
    # (SRV-905, server/app.py `_head_mark`)
    caps = dataclasses.replace(caps, chat_first=True, head_mark=True)
    # R83 (T5b): a one-row prompt chunk runs the prompt pass (`forward(..., prompt=True)`), whose products are in the
    # prompt order at every row count, so a prompt-end resume with a one-row tail equals the cold read
    caps = dataclasses.replace(caps, prompt_rows=True)

    def __init__(self, eng: Hy4FlashEngine, max_len: int):
        self.fn, self.cfg, self.max_len = eng, eng.cfg, int(max_len)
        # ENG-485: paged through the KV pool when the server opened one (its owner named around `side_slot`)
        self.st = eng.new_state(self.max_len, paged=True)
        self.graph_set = GraphSet()         # the slot's captures: its stream, its one pool, dropped at once
        self.st.graphs = self.graph_set
        self.state = Hy4CacheState(self)
        self.kv = _KV(self)
        self.picks = None
        self.tap = None
        self.mm = None                      # the vision tower is not served yet: the image path is off
        self.pos_delta = 0
        self._hidden = None
        self._last = None                   # the last committed row's head input [1, H]
        self._chain = None
        self.commit_ms = 0.0
        self.captured = False
        self.plain_captured = False
        self.heads: dict = {}
        # HY4_GRAPHS: the row pass replays captured graphs (a device), one a row count and form
        self.graphs: dict | None = {} if (_S.get("HY4_GRAPHS") == "1" and eng.device.type == "cuda") else None
        self.stats = {"prefill_rows": 0, "decode_steps": 0, "verify_blocks": 0, "verify_rows": 0}

    def price_class(self, n: int) -> int:
        """The draft prices' context class of a round at context length `n` (engine/methods/staircut.py): 0 while
        the sparse layers read every row (below `index_topk` rows), else the next 4,096 x 2^k at or above n, up to
        262,144. Past `index_topk` rows a row reads a fixed number of latents and scores every closed group, so a
        block's price grows slowly with the context."""
        if n < self.cfg.index_topk:
            return 0
        c = 4096
        while c < n and c < 262144:
            c *= 2
        return c

    # --- cold verbs ------------------------------------------------------------------------------
    @property
    def hidden_post_norm(self) -> torch.Tensor:
        return self._hidden

    def last_streams(self) -> torch.Tensor:
        """The last committed row's head input [1, H] (what the draft layers fuse with the next token)."""
        return self._last if self._last is not None else torch.zeros(1, self.cfg.hidden_size)

    def cost(self) -> CostModel:
        st, c = self.st, self.cfg
        sec = sum(t.numel() * t.element_size() for t in self.state.sections().values())
        row = sum(t.numel() // t.shape[ax] * t.element_size() for _, t, ax in self.state.views())
        return CostModel(section_bytes=sec, view_bytes_per_token=row, parts={"recurrent": sec, "conv": 0},
                         prefill_tok_s=PREFILL_TOK_S)

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
        from engine import kvpool as _kvp
        # a paged tensor's size is its address range: its committed pages are the pool's figure (memsplit)
        kv = sum(int(t.numel() * t.element_size()) for t in (st.lat_all, st.ik_all) if not _kvp.paged(t))
        return {"weights": w, "kv_pool": kv, "drafter_kv": 0,
                "gdn_state": sum(int(t.numel() * t.element_size()) for t in st.win.values())}

    # --- side slots (ENG-485) ----------------------------------------------------------------------
    def side_slot(self, max_len: int, drafter0=None) -> tuple:
        """A side slot's serving and drafter over this engine (`fn`, the weights and the Engram tables once): its
        own state (paged as slot 0's when the pool is on, made here under the slot's owner), rings, row buffers,
        graph set and draft head; the drafter's cut from the same prices and freeze (`priced_cut`, as slot 0's),
        its lookup over slot 0's corpus. Lanes take turns, so only the turn's holder runs."""
        e2 = type(self)(self.fn, max_len)
        d2 = None
        if drafter0 is not None and hasattr(drafter0, "cut"):
            from engine.models.hy4_flash.dfly import DFlyHead
            head = DFlyHead(e2, graphs=e2.graphs is not None)
            e2.heads["dfly"] = head
            freeze = getattr(drafter0, "freeze", False)
            d2 = lane_drafter(e2, head, drafter0, make_cut=lambda prices, sib: priced_cut(prices, sib, freeze)[0])
        return e2, d2

    def side_slot_bytes(self, max_len: int, paged: bool) -> tuple:
        """(bytes a token, constant bytes) a side slot allocates at its build: its latents and indexer keys a token
        unless the window is paged; its rings and last tokens, and the group-key mirror (not paged)."""
        return side_slot_bytes(self.cost(), paged, _mirror_bytes(self.cfg, max_len))

    def _graph_objects(self) -> list:
        out = list(self.graphs.values()) if self.graphs is not None else []
        return out + [h.graph for h in self.heads.values() if getattr(h, "graph", None) is not None]

    def graph_count(self) -> int:
        """The graphs the slot keeps, for /health; no CUDA call."""
        return (len(self.graphs) if self.graphs is not None else 0) + sum(
            1 for h in list(self.heads.values()) if getattr(h, "graph", None) is not None)

    def graph_pools(self) -> set:
        """The memory pools of the slot's graphs (`CUDAGraph.pool()`): what a drop gives back."""
        out = set()
        for g in self._graph_objects():
            try:
                out.add(tuple(g.pool()))
            except (RuntimeError, AttributeError):
                pass
        return out

    def drop_graphs(self) -> int:
        """Every graph of this slot reset at once (`GraphSet.drop`: all captured on the slot's stream, into its one
        pool), as Flash-Next's; the passes capture them again at their next use. Only for a slot no round and no
        capture uses (server/app.py `_drop_slot_graphs` holds its lock). Returns the graphs reset."""
        self._settle_chain()
        n = self.graph_set.drop(self._graph_objects())
        if self.graphs is not None:
            self.graphs.clear()
        for h in self.heads.values():
            if getattr(h, "graph", None) is not None:
                h.graph = None
        return n

    def resident_files(self) -> list:
        """The Engram tables read through the page cache (a warm table's bytes count against the floor)."""
        return [t.report() for t in self.fn.tables.values() if hasattr(t, "report")]

    def discard_pending(self) -> None:
        """Forget what is pending after the state was overwritten (a restore)."""
        self._chain = None
        self.picks = None
        self.st.touched()

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
    def forward(self, ids: torch.Tensor, start: int, last_only: bool = True, prompt: bool = False) -> torch.Tensor:
        """Rows `ids` at positions start..: one token through the row path, more through the prefill.
        `prompt` (R83, `Capabilities.prompt_rows`): the rows are a prompt's, so one row is a one-row prompt chunk
        on the prefill too, not a decode step. Logits [1, 1, vocab] of the last row (fp32)."""
        self._settle_chain()
        ids = ids.view(-1)
        self._at(start)
        T = ids.numel()
        self._ensure(start + T + BLOCK)
        self.graph_set.used()
        if T == 1 and start > 0 and not prompt:
            n0 = len(self.graphs) if self.graphs is not None else 0
            lg, n = self.fn.rows([int(ids[0])], [-1], self.st, graphs=self.graphs, clone=False)
            self.plain_captured = self.graphs is not None and len(self.graphs) != n0
            self.fn.commit(self.st, [0])
            self._hidden = n.view(1, 1, -1)
            self._last = n.view(1, -1).clone()
            self.stats["decode_steps"] += 1
            return lg.view(1, 1, -1)
        lg = self.fn.prefill(ids, self.st)
        self._hidden = self.st.prompt_hidden.unsqueeze(0)
        self._last = self.st.prompt_hidden[-1:]
        self.stats["prefill_rows"] += T
        return lg.view(1, 1, -1)

    @torch.no_grad()
    def forward_tree(self, block: torch.Tensor, parents: list[int], start: int) -> torch.Tensor:
        """A draft block in DFS pre-order (row 0 the anchor at `start`): every row's logits [M, vocab]
        fp32, its greedy pick in `picks`. Nothing is written to the state until `commit_tree`."""
        self._settle_chain()
        self._at(start)
        self._ensure(start + BLOCK)
        self.graph_set.used()
        toks = [int(t) for t in block.view(-1).tolist()]
        n0 = len(self.graphs) if self.graphs is not None else 0
        lg, n = self.fn.rows(toks, [int(p) for p in parents], self.st, graphs=self.graphs, clone=False)
        self.captured = self.graphs is not None and len(self.graphs) != n0
        self.picks = self.st.rb[len(toks)]["next"]
        self._hidden = n.unsqueeze(0)
        self.stats["verify_blocks"] += 1
        self.stats["verify_rows"] += len(toks)
        return lg

    def tree_batchable(self, parents) -> bool:
        return len(parents) <= BLOCK

    @torch.no_grad()
    def commit_tree(self, path: list[int]) -> None:
        t0 = time.perf_counter()
        path = [int(p) for p in path]
        self.fn.commit(self.st, path)
        if path and self._hidden is not None and self._hidden.shape[1] > path[-1]:
            self._last = self._hidden[0, path[-1]:path[-1] + 1].clone()
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

    def _ensure(self, rows: int) -> None:
        """ENG-485: the KV pool's pages for rows [0, rows) committed before a write reaches them, and at least the
        indexer's read floor: a sparse layer gathers its first `index_topk` positions before it masks the ones past
        the length (and a draft head its picks), so a short state still reads those rows. A fresh page is zeros, as
        the buffer was, and a masked row weighs nothing: the bits do not move. Never inside a capture."""
        ensure_rows(self.st.pages, rows, self.max_len, floor=self.cfg.index_topk + BLOCK)

    def _at(self, start: int) -> None:
        if int(start) != self.st.length:
            raise ValueError(f"Hy4-flash forward at {start}, the state holds {self.st.length} rows")


def _mirror_bytes(cfg, max_len: int) -> int:
    """The group-key mirror of a state of `max_len` rows (`Hy4EngState.gk_all`, a plain tensor: every select pass
    scores all of it), bf16."""
    rr = cfg.index_kpool
    return len(cfg.sparse_layers) * (int(max_len) // rr + BLOCK // rr + 1) * cfg.index_head_dim * 2


def _tensors(o) -> list:
    """The tensors a weight entry holds: a tensor, an EXL3 projection's, a dense expert bank's."""
    if torch.is_tensor(o):
        return [o]
    if hasattr(o, "suh"):
        return [o.t, o.suh, o.svh]
    if getattr(o, "dense", None) is not None:
        return [o.dense]
    return []


def profile_of(path: str) -> str:
    """The weight profile a pack is served as (it names the pack's price file): the running one when it is one of
    this family's (QSE_WEIGHTS=hy4-...), else the pack's own bit rate (`quantization_config.bits`: hy4-3.08), else
    hy4-<dir>. A start that sets no profile, or runs under another model's, still finds the pack's prices."""
    env = os.environ.get("QSE_WEIGHTS") or ""
    if env.startswith("hy4-"):
        return env
    try:
        with open(os.path.join(path, "config.json")) as f:
            bits = (json.load(f).get("quantization_config") or {}).get("bits")
        if bits is not None:
            return f"hy4-{float(bits):g}"
    except (OSError, ValueError, TypeError):
        pass
    return "hy4-" + os.path.basename(os.path.normpath(path))


def build(path: str, max_len: int, tok=None, mark=lambda stage: None, log=print, device: str | None = None, check: bool = True,
          kv_pool=None):
    """Hy4-flash for the server: the pack's weights, the engine in the loop's interface, the load-time
    self-check (a block's rows against the decode step's). Returns (serving, info)."""
    from engine.models.hy4_flash.load import load
    from engine.settings import SETTINGS as _S
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pack, W, cuts, tables, banks, timing = load(path, device, log=log, table_policy=_S.get("HY4_TABLE"),
                                                table_cache_bytes=int(float(_S.get("HY4_TABLE_CACHE_GB")) * (1 << 30)))
    mark("weights")
    eng = Hy4FlashEngine(pack.cfg, W, tables, device=device, max_len=max_len, banks=banks, cuts=cuts)
    eng.prefill_grouped = _S.get("HY4_PREFILL_GROUPED") == "1"
    eng.row_torch = _S.get("HY4_ROW_TORCH") == "1"
    eng.prefill_experts = _S.get("HY4_PREFILL_EXPERTS")
    if eng.prefill_experts not in ("group", "rows"):
        raise ValueError(f"QWEN38_HY4_PREFILL_EXPERTS={eng.prefill_experts!r}: group or rows")
    eng.prefill_v2 = _S.get("HY4_PREFILL_V2") == "1"
    # R83: every packed projection of the prompt pass checked in its prompt order once a shape, here: a shape whose
    # kernels are not one order's bits fails the start (tools/exl3_rows.py `RowOrderError`), never a request
    order = eng.bind_prompt_order() if eng.prefill_v2 else {"skipped": "QWEN38_HY4_PREFILL_V2=0: the pass before"}
    log(f"[hy4] prompt order (R83): {order}")
    if kv_pool is not None:
        kv_pool()                           # ENG-485: the KV pool, before the state that pages through it
    srv = Hy4FlashServing(eng, max_len)
    mark("engine")
    res = None
    if check:
        # long enough to reach the group selection and wrapped rings (a short prompt leaves both unchecked),
        # as far as the window allows; blocks of 2, 4 and 6 rows close groups inside them
        n = min(eng.check_rows(), max_len - 64)
        if tok is not None:
            text = tok("Count from one to twenty in words, then name the days of the week. ", add_special_tokens=False)["input_ids"]
        else:
            text = [(7 * i + 3) % pack.cfg.vocab_size for i in range(24)]
        prompt = [int(text[i % len(text)]) for i in range(max(n, 8))]
        res = eng.self_check(prompt, widths=(2, 4, 6), log=log, graphs=srv.graphs is not None)
        srv.restart()
    info = {"pack": path, "load": timing, "device": device, "self_check": res, "prompt_order": order, "quant": pack.quant,
            "dfly": pack.has_dfly,
            "formats": timing.get("notes"), "tables": [t.report() for t in tables.values()]}
    return srv, info


#: every context class `Hy4FlashServing.price_class` names
PRICE_CLASSES = (0,) + tuple(4096 << k for k in range(7))


def priced_cut(prices, siblings: bool, freeze: bool):
    """The cut over `prices` and whether it is frozen (asked for, and there is a price file). A frozen cut learns
    nothing, so the draft's own cost comes from the file too, for every class a round can be in (a class the file
    does not name takes the nearest lower one's, as its other prices do)."""
    from engine.methods.staircut import StairCut
    cut = StairCut(prices, siblings=siblings)
    freeze = bool(freeze and prices.table)
    if freeze:
        for c in PRICE_CLASSES:
            d = prices.prior(1, "draft", c)
            if d is not None:
                cut.draft[c] = d
    return cut, freeze


def dfly_drafter(srv: Hy4FlashServing, path: str, corpus: str | None, log=print):
    """The DFly2 draft layers and the lookup (the request's own text, `corpus`, the server's suffix store) in one
    tree a round, cut by StairCut, through the shared tree drafter."""
    from engine.methods.device import device_id
    from engine.drafters.ngram import NgramDrafter
    from engine.methods.mtp_drafter import MTPTreeDrafter
    from engine.methods.staircut import PriceModel, StairCut
    from engine.models.hy4_flash.dfly import DFlyHead
    from engine.tokfp import fingerprint
    ng = NgramDrafter(corpus_path=corpus or "", tokenizer_sha=fingerprint(path), min_order=3, max_depth=16, min_corpus_order=8)
    prices = PriceModel.load(os.path.join(ROOT, "ops", "prices", MODEL_ID, f"{profile_of(path)}.{device_id()}.json"), log=log)
    head = DFlyHead(srv, graphs=srv.graphs is not None)
    srv.heads["dfly"] = head
    cut, freeze = priced_cut(prices, siblings=_S.get("HY4_DFLY_SIBLINGS") == "1", freeze=_S.get("HY4_DRAFT_FREEZE") == "1")
    if freeze:
        log("[server] Hy4-flash drafter state frozen (HY4_DRAFT_FREEZE): the cut keeps the price file's prices")
    return MTPTreeDrafter(srv, head, cut, ng, freeze=freeze)


#: the server's own default `--store-entry-gb`, and this family's (SRV-907). A snapshot's estimate is twice the
#: rings (17.7 MB) plus 1.25 times its rows at 16,896 bytes (engine/cache.py `_snapshot_estimate`): under 1 GiB one
#: long chat's snapshot took a third of the 2 GiB store at every turn and pushed the short entries out (694 MiB at
#: 32,797 rows, measured); at 0.25 GiB the store keeps prompts up to about 11,000 tokens, as the same cap means
#: about 7,400 on the 27B's state
SERVER_STORE_ENTRY_GB, STORE_ENTRY_GB = 1.0, 0.25


def store_entry(a) -> bool:
    """The state store's entry cap for this family: when the start names the server's default, this family's
    (`STORE_ENTRY_GB`); a start that names its own keeps it. Whether it changed. The server keeps it as the
    flag a request's refresh of the residency limits falls back to (server/app.py `BUILDER_LIMITS`)."""
    if getattr(a, "store_entry_gb", None) == SERVER_STORE_ENTRY_GB:
        a.store_entry_gb = STORE_ENTRY_GB
        return True
    return False


def serve(a, ctx) -> dict:
    """Tencent's Hy4-flash from an EXL3 pack for the server (`--model <dir>`, the registry's `serve`). Text only:
    the vision tower is not served yet. Plain decode on the row path; with `--drafter` other than none the DFly2
    draft layers and the lookup, when the pack holds the layers and the load-time self-check passed."""
    from transformers import AutoTokenizer

    from engine.settings import SETTINGS as _S
    from engine.spec import Relax
    ctx.keys(a.model, [a.model])
    from engine.models.hy4_flash.tokenizer import Hy4Tokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        tok = Hy4Tokenizer(tok)             # the model's suffixed tags under the names the server's parsers read
    except ValueError as e:
        ctx.log(f"[server] Hy4-flash: {e}; the tokenizer is served as it is (no tool calls, no reasoning split)")
    srv, info = build(a.model, a.max_len, tok, mark=ctx.mark, log=ctx.log, kv_pool=open_pool(a, ctx))
    # OPS-308: the memory ceiling and peak are the pack's profile's (ops/profiles/hy4-<bpw>.env), as every family's
    if store_entry(a):
        ctx.log(f"[server] Hy4-flash: state store entries up to {a.store_entry_gb:g} GiB (about 11,000 tokens)")
    drafter = None
    if a.drafter != "none" and _S.get("HY4_DFLY") == "1":
        if not info["dfly"]:
            ctx.log("[server] Hy4-flash: the pack has no DFly layers: plain decode only")
        elif not (info["self_check"] and info["self_check"]["ok"]):
            ctx.log("[server] Hy4-flash self-check FAILED: speculation off, plain decode only")
        else:
            drafter = dfly_drafter(srv, a.model, a.corpus, ctx.log)
            a.k = max(int(a.k or 0), srv.cfg.dfly.draft_tokens)
    ctx.mark("drafters")
    describe = (f"Hy4-flash {a.model} (exl3): load {info['load'].get('total_s')} s, drafter "
                f"{'DFly2 + lookup + StairCut' if drafter is not None else 'none'}, Engram tables {_S.get('HY4_TABLE')}, "
                f"max_len {a.max_len:,},")
    return {"engine": srv, "tok": tok, "drafter": drafter, "path": a.model, "device": info["device"], "cfg": None, "w": None,
            "describe": describe,
            "serving": {"tree": drafter is not None, "sampled_tree": "det", "relax": Relax(1.0, 1),
                        "weights_info": {"format": "exl3", "weights": a.model,
                                         "drafters": "DFly2 + lookup" if drafter is not None else None,
                                         "load_s": info["load"].get("total_s"), "self_check": info["self_check"],
                                         "kv_cache": "bf16 latents", "tables": info["tables"]}}}
