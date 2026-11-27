"""Hy4-flash's knobs: the same shape as engine/settings.py's `KNOBS`, joined to it at import. This file is in
the family's manifest, so its defaults are part of its fingerprint and of no other model's."""

KNOBS: dict[str, tuple] = {
    # where the Engram tables' rows come from: cold (O_DIRECT block reads behind a hot-row cache; the four
    # tables never sit in memory) or warm (read through the page cache)
    'HY4_TABLE': ('cold', 'engine/models/hy4_flash/serving.py'),
    # the cold reader's hot-row cache, GiB over the four tables
    'HY4_TABLE_CACHE_GB': ('2', 'engine/models/hy4_flash/serving.py'),
    # the row pass and the draft as captured CUDA graphs (1: the plain pass's operations replayed, the same bits) or
    # launched one by one (0)
    'HY4_GRAPHS': ('1', 'engine/models/hy4_flash/serving.py'),
    # where a prompt's routed experts run on the row kernels (HY4_PREFILL_EXPERTS rows, or a chunk below its
    # EXPERTS_MIN): the kernels that decode an expert's tiles once for up to 8 of its pairs (1, the default: the same
    # bits a pair as the pair kernels, and on the 3.56 pack the first token of a 3,565-token chat came after 19.1 to
    # 24.2 s against 27.0 to 29.5 s), or the pair kernels (0)
    'HY4_PREFILL_GROUPED': ('1', 'engine/models/hy4_flash/serving.py'),
    # debugging: the row pass's sums in plain torch, one item a call, instead of the family's kernel (other bits;
    # the kernel is the one served path)
    'HY4_ROW_TORCH': ('0', 'engine/models/hy4_flash/serving.py'),
    # the DFly2 draft layers (1) or plain decode only (0)
    'HY4_DFLY': ('1', 'engine/models/hy4_flash/serving.py'),
    # standalone external DFly2 draft model path / HF repo (e.g. AngelSlim/Hy4-flash-DFly2-B6 or EXL3 quant),
    # overriding the checkpoint's embedded dfly_layers.* when non-empty
    'HY4_DFLY_MODEL': ('', 'engine/models/hy4_flash/serving.py'),
    # a drafted round as a tree: the cut may keep the draft layers' second token at a depth as a leaf beside the
    # chain (1), or the chain alone (0)
    'HY4_DFLY_SIBLINGS': ('0', 'engine/models/hy4_flash/serving.py'),
    # the drafter keeps the price file's prices and its calibration's start (1): the same request drafts the same
    # rounds on any server, and a first request's slow rounds do not send the class to the plain step; without a
    # price file for the pack on this device, or at 0, the cut learns its prices as served
    'HY4_DRAFT_FREEZE': ('1', 'engine/models/hy4_flash/serving.py'),
    # SPD-321: a prompt chunk's routed experts on the shared prefill kernel, one call a chunk (group), or on the row
    # kernels in launches of 256 rows (rows: the path before, the A/B arm and the way back); HY4_PREFILL_GROUPED
    # picks the row kernels' form wherever they run
    'HY4_PREFILL_EXPERTS': ('group', 'engine/models/hy4_flash/serving.py'),
    # SPD-323, SPD-325: the prompt pass on fixed-order kernels (1): attention on `mla_rows`, the hyper-connections, norms
    # and small sums on the row kernels, the indexer on `mm` and the shared block select: its own bits. By construction a
    # row's bits depend on its chunk only through the trunk's projections (tools/hy4_check.py --only v2-cuts checks it), and
    # a one-row forward past row 0 is still a decode step. On by default (SPD-327): on the 3.56 pack a 3,565-token chat's
    # first token after 4.9 to 6.6 s against 13.3 to 13.7 s, the cache tiers equal to a cold read, chat2048 KL 0.1015
    # against 0.1009. 0: the pass before, kept as the way back
    'HY4_PREFILL_V2': ('1', 'engine/models/hy4_flash/serving.py'),
    # SPD-345: a verify block's grouped experts on the staged kernels (1: every pair's rows rotated once and streamed
    # beside the tiles, the groups table from one 1024-thread block; the grouped kernels' bits), or the grouped kernels
    # rotating a group's rows in each CTA (0, the path before)
    'HY4_MOE_AS': ('1', 'engine/models/hy4_flash/engine.py'),
    # SPD-345: a hyper-connection in one launch from this many rows (the three launches' statements in one program,
    # their bits); 0: always the three launches
    'HY4_HC_FUSED_MIN': ('16', 'engine/models/hy4_flash/kernels.py'),
    # SPD-345: the DFly2 head on the row kernels (1: norms, rotary, softmax and attention on the row kernels, the
    # projections on dense_rows, the experts grouped; a packed round of any width drafts each lane's lone tokens), or
    # the torch arithmetic before (0). The draft's bits are its own either way; the target's verify decides the text
    'HY4_DFLY_FUSED': ('1', 'engine/models/hy4_flash/dfly.py'),
    # SPD-347: the fixed-order prompt pass without host reads or copies (1: the rotary once a chunk, the read lists'
    # bounds from the host's positions, the rotary and norms in place, the indexer's scores with the rows on the tile
    # axis, the Engram rows' packed words to the device and decoded there; the same bits), or the pass before (0)
    'HY4_PREFILL_FAST': ('1', 'engine/models/hy4_flash/engine.py'),
    # SPD-347: a prefill burst's prompts in one pass (engine/models/hy4_flash/joint_prefill.py: each MoE layer's
    # routed experts once over every prompt's rows, each prompt's bits its own prefill's), or each prompt alone (0)
    'HY4_JOINT_PREFILL': ('1', 'engine/models/hy4_flash/joint_prefill.py'),
}
