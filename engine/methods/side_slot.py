"""ENG-485: what a family's side slot builds, written once.

A side slot (server/app.py `_build_side_slots`) is a second serving object over the family's one engine: its
own state, graphs and draft head over the same weights. Every family that serves side slots wrote the same four
small pieces into its serving class by copy; a family's `side_slot` and `side_slot_bytes`, its `_ensure` and its
`serve` call these instead. The families that have their own copies keep them until each switches in a commit of
its own, with tests/test_lane_contract.py's pins as the proof that nothing moved.

  lane_drafter     a side slot's tree drafter: a new price model over slot 0's table and step (its own
                   learning), a lookup over slot 0's corpus (read-only, slot 0's minimum order), slot 0's freeze
                   and lookup back-off, over the slot's own draft head, whose graphs are captured now
  ensure_rows      the KV pool's pages for rows [0, rows) committed before a write reaches them, from a family's
                   read floor up to its window; never inside a graph capture (its caller committed them first)
  side_slot_bytes  (bytes a token, constant bytes) a side slot takes before it serves anything: none a token
                   with the pool (its pages are committed as rows are written), else the view bytes a token
  open_pool        the opener `serve` hands `build`: the loader's, with `--kv-pool on`, else None

Method code: it chooses how a slot is built, never how a row is computed, so it is in no fingerprint.
"""

from __future__ import annotations

import torch


def lane_drafter(e2, head, drafter0, make_cut=None):
    """A side slot's tree drafter over slot 0's (`drafter0`) prices and corpus, on the slot's serving `e2` and its
    own draft `head`. `make_cut(prices, siblings)` builds a family's own cut over the new price model (a frozen
    draft prior, say); without it the plain StairCut with slot 0's siblings switch."""
    from engine.drafters.ngram import NgramDrafter
    from engine.methods.mtp_drafter import MTPTreeDrafter
    from engine.methods.staircut import PriceModel, StairCut
    ng = None
    if drafter0.ngram is not None:
        ng = NgramDrafter(corpus_path="", min_order=drafter0.ngram.min_order, max_depth=16, min_corpus_order=8)
        ng.corpus = drafter0.ngram.corpus
    c0 = drafter0.cut
    prices = PriceModel(c0.prices.table, c0.prices.step)
    cut = make_cut(prices, c0.siblings) if make_cut is not None else StairCut(prices, siblings=c0.siblings)
    d2 = MTPTreeDrafter(e2, head, cut, ng, freeze=getattr(drafter0, "freeze", False))
    d2.mtp_look_backoff = getattr(drafter0, "mtp_look_backoff", 0)
    d2.capture_all()
    return d2


def ensure_rows(pages, rows: int, cap: int | None = None, floor: int = 0) -> None:
    """Commit `pages` (a state's PagedSet; None: the pool is off) for rows [0, rows) before a write reaches them.
    `floor`: the rows a family's kernels read whatever the length (a gather before its mask), committed first;
    `cap`: the window, applied last, so a window under the floor commits the window. One int compare when the
    rows and the pool's margin are committed already."""
    if pages is None:
        return
    rows = max(int(rows), int(floor))
    if cap is not None:
        rows = min(rows, int(cap))
    if rows + pages.margin() <= pages.rows:
        return
    if pages.on_device() and torch.cuda.is_current_stream_capturing():
        return
    pages.ensure(rows)


def side_slot_bytes(cost, paged: bool, const: int) -> tuple:
    """(bytes a token, constant bytes) of a side slot from its family's `cost()`: the view bytes a token unless
    the window is paged, and the sections plus the family's own constant (its graphs and buffers)."""
    return (0 if paged else cost.view_bytes_per_token), cost.section_bytes + const


def open_pool(a, ctx):
    """The KV pool's opener for `build` (`ctx.kv_pool`, server/app.py's loader): with `--kv-pool on` and a loader
    that offers one, else None (the windows are memory at load)."""
    return ctx.kv_pool if getattr(a, "kv_pool", "off") == "on" and hasattr(ctx, "kv_pool") else None
