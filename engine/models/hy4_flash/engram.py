"""Engram: Hy4-flash's hashed n-gram memory, the part that turns tokens into table rows.

Four layers each own a table of about 165 M rows of 96 values. For every position the layer mixes the
current token with the one before it (8 hash heads) and with the two before it (8 more), reads those 16
rows and concatenates them into a 1,536-wide feature. The hash is the reference's
(`transformers` `HYV4FlashEngramLayer.hash_ngrams`) operation for operation, in int64 with the same
wrap-around, so the row indices are identical:

    c_j   = lookup_table[token_j]                 the checkpoint's frozen map to a normalised vocabulary
    m_j   = c_j * multiplier_j                    int64, wraps
    id_h  = (m_0 ^ ... ^ m_{n-1}) mod prime_h     n = 2 for heads 0..7, n = 3 for heads 8..15

The multipliers are not stored. They are drawn from NumPy's default generator seeded with
`engram_seed + 10007 * layer_id` (the one-indexed layer id), below a bound set by the size of the
normalised vocabulary (`lookup_table.max() + 1`), and made odd.

A position before the second (third) token of its document has no bigram (trigram): the missing
tokens read as `engram_pad_id`. The reference rounds every row it reads through e4m3 with one scale
a row (`fp8_round_trip`), as the memory was trained to be read.
"""

from __future__ import annotations

import numpy as np
import torch

_PRIME = 10007


def hash_multipliers(compressed_vocab_size: int, max_ngram_size: int, layer_id: int, seed: int) -> list[int]:
    """The odd multipliers of one Engram layer (`draw_hash_multipliers`)."""
    half_bound = max(1, np.iinfo(np.int64).max // int(compressed_vocab_size) // 2)
    g = np.random.default_rng(seed + _PRIME * layer_id)
    drawn = g.integers(0, half_bound, size=max_ngram_size, dtype=np.int64)
    return (drawn * 2 + 1).tolist()


class EngramHash:
    """Row indices of one Engram layer's table for n-grams of token ids."""

    def __init__(self, lookup_table: torch.Tensor, head_sizes: list[int], max_ngram_size: int,
                 heads_per_ngram: int, pad_id: int, layer_id: int, seed: int):
        if len(head_sizes) != (max_ngram_size - 1) * heads_per_ngram:
            raise ValueError("Engram: one table size a hash head")
        self.lookup = lookup_table.to(torch.int64).cpu()
        self.table_size = int(self.lookup.shape[0])
        if not 0 <= pad_id < self.table_size:
            raise ValueError("Engram: the pad id must be inside the lookup table")
        self.n, self.heads_per_ngram, self.pad_id = max_ngram_size, heads_per_ngram, pad_id
        self.multipliers = torch.tensor(
            hash_multipliers(int(self.lookup.max()) + 1, max_ngram_size, layer_id, seed), dtype=torch.int64)
        self.sizes = torch.tensor(head_sizes, dtype=torch.int64)
        self.offsets = torch.tensor([0] + head_sizes[:-1], dtype=torch.int64).cumsum(0)
        self.rows = int(sum(head_sizes))

    @classmethod
    def from_config(cls, cfg, layer_id: int, lookup_table: torch.Tensor) -> "EngramHash":
        if int(lookup_table.shape[0]) != cfg.engram_lookup_table_size:
            raise ValueError(f"Engram layer {layer_id}: lookup table of {int(lookup_table.shape[0])} ids, "
                             f"the config says {cfg.engram_lookup_table_size}")
        return cls(lookup_table, cfg.engram_head_vocab_sizes[layer_id], cfg.engram_max_ngram_size,
                   cfg.engram_n_head_per_ngram, cfg.engram_pad_id, layer_id, cfg.engram_seed)

    @property
    def heads(self) -> int:
        return int(self.sizes.shape[0])

    def head_ids(self, ngram_tokens: torch.Tensor) -> torch.Tensor:
        """[..., n] token ids (the position's token first, then the ones before it) -> [..., heads] ids,
        each inside its own head's slice."""
        t = ngram_tokens.to(torch.int64).cpu()
        c = self.lookup[t.clamp(0, self.table_size - 1)]
        c = torch.where((t < self.table_size) & (t >= 0), c, self.lookup[self.pad_id])
        mixed = c * self.multipliers
        out = []
        for size in range(2, self.n + 1):
            mix = mixed[..., 0]
            for j in range(1, size):
                mix = torch.bitwise_xor(mix, mixed[..., j])
            lo = (size - 2) * self.heads_per_ngram
            out.append(torch.remainder(mix.unsqueeze(-1), self.sizes[lo:lo + self.heads_per_ngram]))
        return torch.cat(out, dim=-1)

    def ids(self, ngram_tokens: torch.Tensor) -> torch.Tensor:
        """[..., n] token ids -> [..., heads] rows of the layer's one table."""
        return self.head_ids(ngram_tokens) + self.offsets


def ngram_tokens(tokens: torch.Tensor, history: torch.Tensor, doc_offsets: torch.Tensor, n: int, pad_id: int) -> torch.Tensor:
    """[T] tokens after `history` (the tokens before them, any length) -> [T, n]: each position's token
    with the n - 1 before it; where a position's offset in its document is below j, token j reads as pad
    (`build_ngram_tokens`)."""
    T = int(tokens.shape[0])
    tokens = tokens.to(torch.int64).cpu()
    hist = history.to(torch.int64).cpu()[-(n - 1):] if n > 1 else tokens[:0]
    lead = torch.full((n - 1 - hist.shape[0],), pad_id, dtype=torch.int64)
    full = torch.cat([lead, hist, tokens])
    off = doc_offsets.to(torch.int64).cpu()
    cols = [tokens]
    for j in range(1, n):
        prev = full[n - 1 - j: n - 1 - j + T]
        cols.append(torch.where(off < j, torch.full_like(prev, pad_id), prev))
    return torch.stack(cols, dim=-1)


def fp8_round_trip(x: torch.Tensor) -> torch.Tensor:
    """Each row to e4m3 with its own scale and back, in x's dtype (`_fp8_round_trip`, the forward value)."""
    info = torch.finfo(torch.float8_e4m3fn)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / info.max
    q = (x / scale).clamp(-info.max, info.max).to(torch.float8_e4m3fn).to(x.dtype) * scale
    return x + (q - x)
