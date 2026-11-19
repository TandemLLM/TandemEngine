"""Hy4-flash's knobs: the same shape as engine/settings.py's `KNOBS`, joined to it at import. This file is in
the family's manifest, so its defaults are part of its fingerprint and of no other model's."""

KNOBS: dict[str, tuple] = {
    # where the Engram tables' rows come from: cold (O_DIRECT block reads behind a hot-row cache; the four
    # tables never sit in memory) or warm (read through the page cache)
    'HY4_TABLE': ('cold', 'engine/models/hy4_flash/serving.py'),
    # the cold reader's hot-row cache, GiB over the four tables
    'HY4_TABLE_CACHE_GB': ('2', 'engine/models/hy4_flash/serving.py'),
    # the DFly2 draft layers (1) or plain decode only (0)
    'HY4_DFLY': ('1', 'engine/models/hy4_flash/serving.py'),
}
