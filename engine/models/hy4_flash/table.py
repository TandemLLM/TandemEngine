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

import ctypes
import errno
import json
import mmap
import os
import platform
import struct
import sys

import numpy as np
import torch

from engine.models.flashnext.ple import BLOCK, DirectRows, page_cache_bytes

ROW_DIM = 96
MUL1 = 0x83DCD12D


def mul1_codebook(device=None) -> torch.Tensor:
    """All 65,536 mul1 codebook values (fp16), as exllamav3's `mul1_codebook` (MIT, Copyright (c) 2025 Turboderp)."""
    s = torch.arange(65536, dtype=torch.int64, device=device)
    prod = (s * MUL1) & 0xFFFFFFFF
    bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    k_inv = torch.tensor([0x1eee], dtype=torch.int16).view(torch.float16).float().item()
    k_bias = torch.tensor([0xc931 - 0x10000], dtype=torch.int16).view(torch.float16).float().item()
    return ((1024 + bsum).float() * k_inv + k_bias).to(torch.float16)



class _Iocb(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64), ("key", ctypes.c_uint32), ("rw_flags", ctypes.c_uint32),
                ("opcode", ctypes.c_uint16), ("reqprio", ctypes.c_int16), ("fildes", ctypes.c_uint32),
                ("buf", ctypes.c_uint64), ("nbytes", ctypes.c_uint64), ("offset", ctypes.c_int64),
                ("reserved2", ctypes.c_uint64), ("flags", ctypes.c_uint32), ("resfd", ctypes.c_uint32)]


class _IoEvent(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64), ("obj", ctypes.c_uint64), ("res", ctypes.c_int64), ("res2", ctypes.c_int64)]


#: io_setup, io_destroy, io_submit, io_getevents by machine (Linux)
_AIO_SYS = {"aarch64": (0, 1, 2, 4), "x86_64": (206, 207, 209, 208)}
_AIO_DEPTH = 256


class BatchRows(DirectRows):
    """`DirectRows` whose misses of one call go to the kernel in one submit (Linux AIO on the O_DIRECT file): a
    decode step's 10 to 16 missed rows a table wait for the device once, not for a thread each (measured on the
    released tables: 16 rows in 0.27 ms against 1.6 ms). The same blocks, the same rows; without kernel AIO, or
    past its queue depth, the parent's reads."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._ctx, self._libc, self._abuf, self._cbs = None, None, None, None
        nums = _AIO_SYS.get(platform.machine())
        if nums is None or not self.direct or not sys.platform.startswith("linux"):
            return
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            ctx = ctypes.c_ulong(0)
            if libc.syscall(nums[0], _AIO_DEPTH, ctypes.byref(ctx)) == 0:
                self._ctx, self._libc, self._nums = ctx, libc, nums
                self._cbs = (_Iocb * _AIO_DEPTH)()
                self._ptrs = (ctypes.POINTER(_Iocb) * _AIO_DEPTH)(*[ctypes.pointer(c) for c in self._cbs])
                self._evs = (_IoEvent * _AIO_DEPTH)()
        except (OSError, AttributeError):
            self._ctx = None

    def _submit(self, ids: np.ndarray):
        """The rows' block runs handed to the device (not waited for); None where this reader has no queue for them."""
        n = int(ids.size)
        if self._ctx is None or not 0 < n <= _AIO_DEPTH:
            return None
        start = self.offset + ids * self.row_bytes
        b0 = start // BLOCK * BLOCK
        span = 2 * BLOCK
        if self._abuf is None or len(self._abuf) < n * span:
            self._abuf = mmap.mmap(-1, max(n, 64) * span)
            self._addr = ctypes.addressof(ctypes.c_char.from_buffer(self._abuf))
        size = os.fstat(self.fd).st_size
        for i in range(n):
            c = self._cbs[i]
            c.data, c.opcode, c.fildes = i, 0, self.fd
            c.buf, c.offset = self._addr + i * span, int(b0[i])
            c.nbytes = min(span, -(-(size - int(b0[i])) // BLOCK) * BLOCK)
        sent = self._libc.syscall(self._nums[2], self._ctx, n, self._ptrs)
        return ids, start, b0, sent

    def _collect(self, job) -> np.ndarray:
        ids, start, b0, sent = job
        n, span = int(ids.size), 2 * BLOCK
        want = (start - b0) + self.row_bytes
        done = 0
        bad = sent != n
        while done < max(sent, 0):
            k = self._libc.syscall(self._nums[3], self._ctx, 1, sent - done, self._evs, None)
            if k <= 0:
                if ctypes.get_errno() == errno.EINTR:
                    continue
                raise OSError(ctypes.get_errno(), f"{self.path}: io_getevents")
            for j in range(k):
                e = self._evs[j]
                bad = bad or e.res < int(want[e.data])
            done += k
        if bad:
            return super()._read(ids)                    # a short or refused read: the parent's way, whole
        self.stats["blocks_read"] += n
        raw = np.frombuffer(self._abuf, dtype=np.uint8, count=n * span).reshape(n, span)
        rel = (start - b0)[:, None] + np.arange(self.row_bytes)[None, :]
        return np.take_along_axis(raw, rel, axis=1).copy().view(np.int16)

    def _read(self, ids: np.ndarray) -> np.ndarray:
        job = self._submit(ids)
        return super()._read(ids) if job is None else self._collect(job)

    def begin(self, ids: np.ndarray):
        """`rows_of` in two halves (the caller holds `_mu`): the cache looked up and the misses handed to the device."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        uniq, inv = np.unique(ids, return_inverse=True)
        sets = self._set_of(uniq)
        hit = self.keys[sets] == uniq[:, None]
        has = hit.any(1)
        way = hit.argmax(1)
        out = np.empty((uniq.size, self.words), dtype=np.int16)
        h = np.nonzero(has)[0]
        out[h] = self.vals[sets[h], way[h]]
        self.ref[sets[h], way[h]] = True
        m = np.nonzero(~has)[0]
        self.stats["hits"] += int(h.size)
        self.stats["misses"] += int(m.size)
        return uniq, inv, sets, out, m, (self._submit(uniq[m]) if m.size else None)

    def end(self, state) -> np.ndarray:
        uniq, inv, sets, out, m, job = state
        if m.size:
            got = self._collect(job) if job is not None else DirectRows._read(self, uniq[m])
            out[m] = got
            self._insert(sets[m], uniq[m], got)
        return out[inv]

    def close(self) -> None:
        if self._ctx is not None:
            self._libc.syscall(self._nums[1], self._ctx)
            self._ctx = None
        super().close()


def rows_of_many(readers: list, ids: list) -> list:
    """`rows_of` of several tables' readers at once: every table's misses are with the device before any is waited
    for (a decode step reads four tables). The same rows as one call a reader."""
    if not all(isinstance(r, BatchRows) for r in readers):
        return [r.rows_of(i) for r, i in zip(readers, ids)]
    held = []
    try:
        for r in readers:
            r._mu.acquire()
            held.append(r)
        states = [r.begin(i) for r, i in zip(readers, ids)]
        return [r.end(s) for r, s in zip(readers, states)]
    finally:
        for r in held:
            r._mu.release()


def gather_packed_many(tables: list, ids: list) -> list:
    """`gather_packed` of several tables in one pass over the device (`rows_of_many`) where all are cold."""
    if not all(t.direct is not None for t in tables):
        return [t.gather_packed(i) for t, i in zip(tables, ids)]
    flat = [i.reshape(-1).cpu().numpy().astype(np.int64) for i in ids]
    for t, f in zip(tables, flat):
        if f.size and (f.min() < 0 or f.max() >= t.rows):
            raise IndexError("Engram row id out of range")
    return [torch.from_numpy(r) for r in rows_of_many([t.direct for t in tables], flat)]


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
            self.direct = BatchRows(self.path, self.offset, self.row_bytes, self.rows, cache_bytes)

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
