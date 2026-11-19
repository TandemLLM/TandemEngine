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
from engine.settings import SETTINGS as _S

MODULES = ("engine/models/hy4_flash/__init__.py", "engine/models/hy4_flash/config.py", "engine/models/hy4_flash/engine.py",
           "engine/models/hy4_flash/engram.py", "engine/models/hy4_flash/ref.py", "engine/models/hy4_flash/knobs.py",
           "engine/models/hy4_flash/kernels.py", "engine/models/hy4_flash/native.py",
           "engine/models/hy4_flash/dfly.py", "engine/models/hy4_flash/table.py", "engine/models/hy4_flash/pack.py",
           "engine/models/hy4_flash/load.py",
           "engine/models/flashnext/ref.py", "engine/settings.py")
#: the model's name in ops/prices/ and models/
MODEL_ID = "hy4-flash"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


#: the tables the arithmetic reads (a dense split is a summation order)
TABLES = ("ops/hy4-dense-splits.json",)


def manifest() -> Manifest:
    return Manifest(arch="hy_v4_flash", modules=MODULES, data=TABLES)


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
        st = self.s.st
        out = []
        for li in sorted(st.lat):
            out += [(f"lat.{li}", st.lat[li], 0), (f"ik.{li}", st.ik[li], 0)]
        return out

    def restore(self, snap) -> None:
        super().restore(snap)
        self.s.st.touched()


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
    def forward(self, ids: torch.Tensor, start: int, last_only: bool = True) -> torch.Tensor:
        """Rows `ids` at positions start..: one token through the row path, more through the prefill.
        Logits [1, 1, vocab] of the last row (fp32)."""
        self._settle_chain()
        ids = ids.view(-1)
        self._at(start)
        T = ids.numel()
        if T == 1 and start > 0:
            n0 = len(self.graphs) if self.graphs is not None else 0
            lg, n = self.fn.rows([int(ids[0])], [-1], self.st, graphs=self.graphs, clone=False)
            self.plain_captured = self.graphs is not None and len(self.graphs) != n0
            self.fn.commit(self.st, [0])
            self._hidden = n.view(1, 1, -1)
            self._last = n.view(1, -1).clone()
            self.stats["decode_steps"] += 1
            return lg.view(1, 1, -1)
        lg = self.fn.prefill(ids, self.st)
        self._hidden = self.fn.prompt_hidden.unsqueeze(0)
        self._last = self.fn.prompt_hidden[-1:]
        self.stats["prefill_rows"] += T
        return lg.view(1, 1, -1)

    @torch.no_grad()
    def forward_tree(self, block: torch.Tensor, parents: list[int], start: int) -> torch.Tensor:
        """A draft block in DFS pre-order (row 0 the anchor at `start`): every row's logits [M, vocab]
        fp32, its greedy pick in `picks`. Nothing is written to the state until `commit_tree`."""
        self._settle_chain()
        self._at(start)
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


def profile_of(path: str) -> str:
    """The weight profile a pack is served as: the running one (QSE_WEIGHTS), else hy4-<dir>."""
    return os.environ.get("QSE_WEIGHTS") or "hy4-" + os.path.basename(os.path.normpath(path))


def build(path: str, max_len: int, tok=None, mark=lambda stage: None, log=print, device: str | None = None, check: bool = True):
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
    info = {"pack": path, "load": timing, "device": device, "self_check": res, "quant": pack.quant, "dfly": pack.has_dfly,
            "formats": timing.get("notes"), "tables": [t.report() for t in tables.values()]}
    return srv, info


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
    cut = StairCut(prices, siblings=_S.get("HY4_DFLY_SIBLINGS") == "1")
    freeze = _S.get("HY4_DRAFT_FREEZE") == "1" and bool(prices.table)
    if freeze:
        for c in prices.table:                          # a frozen cut learns no draft cost: the file's
            d = prices.prior(1, "draft", c)
            if d is not None:
                cut.draft[c] = d
        log("[server] Hy4-flash drafter state frozen (HY4_DRAFT_FREEZE): the cut keeps the price file's prices")
    return MTPTreeDrafter(srv, head, cut, ng, freeze=freeze)


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
    srv, info = build(a.model, a.max_len, tok, mark=ctx.mark, log=ctx.log)
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
