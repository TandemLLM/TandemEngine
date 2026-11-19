"""The Engram tables of an EXL3 pack: one file a table, rows of 96 values coded as a ring.

A pack keeps each Engram layer's table in its own file, `engram.layers.<layer index>.safetensors`
(exllamav3's `exl3_ngram_trellis` format, version 1), under the prefix
`model.layers.<index>.engram_layer.multi_head_embedding.embedding.`:

    head_bias [heads, 96] fp16     head_offsets [heads] int64     head_vocab_sizes [heads] int64
    layer_multipliers [n] int64    trellis [rows, 1 + 6 K] int16  (the rows, last and contiguous)

A row is one fp16 scale and a tail-biting ring over the mul1 codebook at K bits a value: stream bits
[K i, K i + K) are the low bits of position i's 16-bit state, whose higher bits are the positions before
it (modulo 96). So

    row[i] = mul1(state_i) * scale + head_bias[head(row)][i]            (fp32, then the compute dtype)

The table encodes the rows as the reference reads them, after their e4m3 round trip (the file's
`target`), so a decoded row is used as it is (`fp8_done`). The table is never in device memory: `cold`
reads the 4 KiB blocks that hold a step's rows with O_DIRECT behind a hot-row cache, `warm` reads through
the page cache (engine/models/flashnext/ple.py has the readers; Flash-Next's rows are 160 wide, in one file).
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np
import torch

from engine.models.flashnext.ple import DirectRows, mul1_codebook, page_cache_bytes

ROW_DIM = 96
PREFIX = "model.layers.{}.engram_layer.multi_head_embedding.embedding."
_DT = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64, "I32": torch.int32}
_SRC: dict = {}


def table_file(pack_dir: str, layer: int) -> str:
    return os.path.join(pack_dir, f"engram.layers.{layer}.safetensors")


def _ring_src(K: int, dim: int, device) -> torch.Tensor:
    """[dim, 16] the stream bit of state bit m of position i: ((i - m // K) mod dim) * K + m % K."""
    key = (K, dim, str(device))
    if key not in _SRC:
        i = torch.arange(dim).view(-1, 1)
        m = torch.arange(16).view(1, -1)
        _SRC[key] = (((i - m // K) % dim) * K + m % K).to(device)
    return _SRC[key]


def decode_rows(packed: torch.Tensor, heads: torch.Tensor, bias: torch.Tensor, K: int, codebook: torch.Tensor,
                dim: int = ROW_DIM) -> torch.Tensor:
    """packed int16 [N, 1 + dim K / 16], heads [N] (each row's hash head), bias [n_heads, dim] -> fp32 [N, dim]."""
    dev = packed.device
    scales = packed[:, 0].contiguous().view(torch.float16).float()
    wd = packed[:, 1:].contiguous().view(torch.uint16).to(torch.int32)
    stream = ((wd.unsqueeze(-1) >> torch.arange(16, device=dev, dtype=torch.int32)) & 1).reshape(packed.shape[0], dim * K)
    m = torch.arange(16, device=dev, dtype=torch.int32)
    states = (stream[:, _ring_src(K, dim, dev)] << m).sum(-1)
    return codebook[states.long()].float() * scales[:, None] + bias[heads.long()].float()


class EngramTable:
    """One Engram layer's packed table, read in place from its file."""

    fp8_done = True          # the rows are the reference's rows after the e4m3 round trip

    def __init__(self, path: str, layer: int):
        self.path, self.layer = path, layer
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        self.meta = hdr.pop("__metadata__", None) or {}
        self._start, self._hdr, p = 8 + n, hdr, PREFIX.format(layer)
        self.prefix = p
        if self.meta.get("format") != "exl3_ngram_trellis" or str(self.meta.get("codebook", "mul1")) != "mul1":
            raise ValueError(f"{path}: not an exl3_ngram_trellis table on the mul1 codebook ({self.meta})")
        if self.meta.get("target") != "fp8_e4m3_row_round_trip":
            raise ValueError(f"{path}: rows target {self.meta.get('target')!r}; this reader expects rows that already "
                             "went through the e4m3 round trip")
        t = hdr.get(p + "trellis")
        if t is None or t["dtype"] != "I16" or len(t["shape"]) != 2:
            raise ValueError(f"{path}: no int16 row array {p}trellis")
        self.rows, self.words = int(t["shape"][0]), int(t["shape"][1])
        self.dim = int(self.meta.get("row_dim", ROW_DIM))
        self.K = (self.words - 1) * 16 // self.dim
        if 1 + self.dim * self.K // 16 != self.words or (self.dim * self.K) % 16:
            raise ValueError(f"{path}: {self.words} words a row is no K for rows of {self.dim}")
        self.row_bytes = 2 * self.words
        b0, b1 = t["data_offsets"]
        if b1 - b0 != self.rows * self.row_bytes or os.path.getsize(path) < self._start + b1:
            raise ValueError(f"{path}: the row array is truncated")
        self.offset = self._start + b0
        self.bias = self._aux("head_bias").to(torch.float16)
        self.offsets = self._aux("head_offsets").long()
        self.sizes = self._aux("head_vocab_sizes").long()
        self.multipliers = self._aux("layer_multipliers").long()
        if self.bias.shape != (self.offsets.numel(), self.dim) or int(self.sizes.sum()) != self.rows:
            raise ValueError(f"{path}: {self.offsets.numel()} heads of {self.sizes.tolist()} rows do not make {self.rows} rows")
        self.map = np.memmap(path, dtype=np.int16, mode="r", offset=self.offset, shape=(self.rows, self.words))
        self.direct: DirectRows | None = None
        self._codebook: dict = {}

    def _aux(self, name: str) -> torch.Tensor:
        t = self._hdr.get(self.prefix + name)
        if t is None:
            raise ValueError(f"{self.path}: missing {self.prefix}{name}")
        b0, b1 = t["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self._start + b0)
            raw = bytearray(f.read(b1 - b0))
        return torch.frombuffer(raw, dtype=_DT[t["dtype"]]).reshape(t["shape"]).clone()

    def check(self, hashing) -> None:
        """The file's hash constants against the ones the config and the lookup table give (`EngramHash`)."""
        if self.sizes.tolist() != hashing.sizes.tolist() or self.offsets.tolist() != hashing.offsets.tolist():
            raise ValueError(f"{self.path}: head sizes or offsets differ from the config's")
        if self.multipliers.tolist() != hashing.multipliers.tolist():
            raise ValueError(f"{self.path}: hash multipliers {self.multipliers.tolist()} differ from the drawn "
                             f"{hashing.multipliers.tolist()}")

    def set_policy(self, policy: str, cache_bytes: int = 512 << 20) -> None:
        if policy not in ("warm", "cold"):
            raise ValueError(f"Engram table policy {policy!r}: warm or cold")
        if self.direct is not None:
            self.direct.close()
            self.direct = None
        if policy == "cold":
            self.direct = DirectRows(self.path, self.offset, self.row_bytes, self.rows, cache_bytes)

    @property
    def policy(self) -> str:
        return "cold" if self.direct is not None else "warm"

    def head_of(self, ids: torch.Tensor) -> torch.Tensor:
        offs = self.offsets.to(ids.device)
        return (torch.searchsorted(offs, ids.long(), right=True) - 1).clamp(0, offs.numel() - 1)

    def gather_packed(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).cpu().numpy().astype(np.int64)
        if flat.size and (flat.min() < 0 or flat.max() >= self.rows):
            raise IndexError("Engram row id out of range")
        if self.direct is not None:
            return torch.from_numpy(self.direct.rows_of(flat))
        return torch.from_numpy(np.asarray(self.map[flat]))

    def codebook(self, device) -> torch.Tensor:
        key = str(device)
        if key not in self._codebook:
            self._codebook[key] = mul1_codebook(device)
        return self._codebook[key]

    def gather(self, ids: torch.Tensor, dtype=torch.bfloat16, device=None) -> torch.Tensor:
        """ids [n] -> rows [n, 96] in `dtype`, decoded on `device` (default the CPU)."""
        dev = device if device is not None else "cpu"
        packed = self.gather_packed(ids).to(dev)
        rows = decode_rows(packed, self.head_of(ids.reshape(-1).to(dev)), self.bias.to(dev), self.K, self.codebook(dev), self.dim)
        return rows.to(dtype)

    def report(self) -> dict:
        out = {"name": f"engram.{self.layer}", "path": self.path, "policy": self.policy, "file_bytes": self.rows * self.row_bytes,
               "rows": self.rows, "K": self.K,
               "resident_bytes": page_cache_bytes(self.path, self.offset, self.rows * self.row_bytes)}
        if self.direct is not None:
            out.update(cache_bytes=self.direct.cache_bytes, direct=self.direct.direct, **self.direct.stats)
        return out
