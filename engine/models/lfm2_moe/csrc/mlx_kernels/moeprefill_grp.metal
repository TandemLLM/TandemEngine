// Single-TG GPU expert grouping + flat BM-tile work-queue for batched prefill (zero skewed-expert stragglers!).
// Emits (tile_e[MAXT], tile_off[MAXT], tile_nr[MAXT]) splitting hot experts across parallel threadgroups, plus sel[P].
// Grid: (256, 1, 1), Threadgroup: (256, 1, 1).
const uint tid = thread_position_in_threadgroup.x;
const int P = pexp_shape[0];

threadgroup atomic_uint tg_cnt[E];
threadgroup atomic_uint tg_pos[E];

if ((int)tid < E) atomic_store_explicit(&tg_cnt[tid], 0u, memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int p = (int)tid; p < P; p += 256) {
    atomic_fetch_add_explicit(&tg_cnt[pexp[p]], 1u, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid == 0) {
    uint off = 0;
    int t = 0;
    for (int e = 0; e < E; e++) {
        const uint c = atomic_load_explicit(&tg_cnt[e], memory_order_relaxed);
        atomic_store_explicit(&tg_pos[e], off, memory_order_relaxed);
        for (int c0 = 0; c0 < (int)c; c0 += BM) {
            if (t < MAXT) {
                tile_e[t] = e;
                tile_off[t] = (int)off + c0;
                tile_nr[t] = min(BM, (int)c - c0);
                t++;
            }
        }
        off += c;
    }
    for (; t < MAXT; t++) {
        tile_e[t] = -1;
        tile_off[t] = 0;
        tile_nr[t] = 0;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int p = (int)tid; p < P; p += 256) {
    const uint pos = atomic_fetch_add_explicit(&tg_pos[pexp[p]], 1u, memory_order_relaxed);
    sel[pos] = p;
}
