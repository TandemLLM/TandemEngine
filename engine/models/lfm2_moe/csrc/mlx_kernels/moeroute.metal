// Fused MoE router top-k + ascending sort + ordered renorm + LPT compact expert grouping (1 threadgroup).
// - When M == 1 (GROUP == 0): all E threads compute sigmoid+bias in parallel (1 exp() per thread!),
//   and 2 simdgroups find the top-TOPK in parallel, skipping unused elist/eoff/ecnt/sel grouping.
// - When M > 1 (GROUP == 1): thread m processes row m's E logits independently with deterministic tie-breaking,
//   then outputs LPT-ordered (ns>=3 heavy MMA experts first, then ns==2, then ns==1) compact (elist, eoff, ecnt, sel).
// 100% bit-for-bit row-invariant across M=1..64.
const uint tid = thread_position_in_threadgroup.x;
const int M = logits_shape[0];
const int P = M * TOPK;

#if !GROUP
// M == 1 fast path: parallel sigmoid+bias across all E threads (no 64x serial exp() loop on thread 0!)
threadgroup float tg_s[E];
threadgroup float tg_c[E];
if ((int)tid < E) {
    const float lv = float(logits[tid]);
    const float sv = float(static_cast<T>(1.0f / (1.0f + exp(-lv))));
    tg_s[tid] = sv;
    tg_c[tid] = USE_BIAS ? (sv + bias[tid]) : sv;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    float top_c[TOPK], top_s[TOPK];
    int top_i[TOPK];
    for (int j = 0; j < TOPK; j++) { top_c[j] = -INFINITY; top_s[j] = 0.0f; top_i[j] = -1; }
    for (int e = 0; e < E; e++) {
        const float cv = tg_c[e];
        const float sv = tg_s[e];
        int min_j = 0;
        for (int j = 1; j < TOPK; j++) {
            if (top_c[j] < top_c[min_j] || (top_c[j] == top_c[min_j] && top_i[j] > top_i[min_j])) min_j = j;
        }
        if (cv > top_c[min_j]) { top_c[min_j] = cv; top_s[min_j] = sv; top_i[min_j] = e; }
    }
    for (int i = 1; i < TOPK; i++) {
        const int ki = top_i[i];
        const float ks = top_s[i];
        int j = i - 1;
        while (j >= 0 && top_i[j] > ki) { top_i[j + 1] = top_i[j]; top_s[j + 1] = top_s[j]; j--; }
        top_i[j + 1] = ki;
        top_s[j + 1] = ks;
    }
    float tot = top_s[0];
    for (int j = 1; j < TOPK; j++) tot = tot + top_s[j];
    const float denom = tot + 1e-6f;
    const float sc = scale[0];
    for (int j = 0; j < TOPK; j++) {
        pexp[j] = top_i[j];
        wts[j] = static_cast<T>((top_s[j] / denom) * sc);
    }
}
#else
threadgroup int tg_pexp[MAXP];
threadgroup int tg_cnt[E];
threadgroup int tg_uoff[E];

if ((int)tid < M) {
    const int m = (int)tid;
    const device T* row = logits + (size_t)m * E;
    float top_c[TOPK];
    float top_s[TOPK];
    int   top_i[TOPK];
    for (int j = 0; j < TOPK; j++) { top_c[j] = -INFINITY; top_s[j] = 0.0f; top_i[j] = -1; }
    for (int e = 0; e < E; e++) {
        const float lv = float(row[e]);
        const float sv = float(static_cast<T>(1.0f / (1.0f + exp(-lv))));
        const float cv = USE_BIAS ? (sv + bias[e]) : sv;
        int min_j = 0;
        for (int j = 1; j < TOPK; j++) {
            if (top_c[j] < top_c[min_j] || (top_c[j] == top_c[min_j] && top_i[j] > top_i[min_j])) min_j = j;
        }
        if (cv > top_c[min_j]) {
            top_c[min_j] = cv;
            top_s[min_j] = sv;
            top_i[min_j] = e;
        }
    }
    for (int i = 1; i < TOPK; i++) {
        const int ki = top_i[i];
        const float ks = top_s[i];
        int j = i - 1;
        while (j >= 0 && top_i[j] > ki) {
            top_i[j + 1] = top_i[j];
            top_s[j + 1] = top_s[j];
            j--;
        }
        top_i[j + 1] = ki;
        top_s[j + 1] = ks;
    }
    float tot = top_s[0];
    for (int j = 1; j < TOPK; j++) tot = tot + top_s[j];
    const float denom = tot + 1e-6f;
    const float sc = scale[0];
    for (int j = 0; j < TOPK; j++) {
        tg_pexp[m * TOPK + j] = top_i[j];
        pexp[m * TOPK + j] = top_i[j];
        wts[m * TOPK + j] = static_cast<T>((top_s[j] / denom) * sc);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if ((int)tid < E) {
    int c = 0;
    for (int p = 0; p < P; p++) {
        if (tg_pexp[p] == (int)tid) c++;
    }
    tg_cnt[tid] = c;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// LPT ordering: heavy MMA experts (ns >= 3) first, then ns == 2, then ns == 1 (minimizes GPU wave tail!)
if (tid == 0) {
    int u = 0;
    int off = 0;
    for (int e = 0; e < E; e++) tg_uoff[e] = -1;
    for (int e = 0; e < E; e++) {
        const int c = tg_cnt[e];
        if (c >= 3) { elist[u] = e; eoff[u] = off; ecnt[u] = c; tg_uoff[e] = off; off += c; u++; }
    }
    for (int e = 0; e < E; e++) {
        const int c = tg_cnt[e];
        if (c == 2) { elist[u] = e; eoff[u] = off; ecnt[u] = c; tg_uoff[e] = off; off += c; u++; }
    }
    for (int e = 0; e < E; e++) {
        const int c = tg_cnt[e];
        if (c == 1) { elist[u] = e; eoff[u] = off; ecnt[u] = c; tg_uoff[e] = off; off += c; u++; }
    }
    for (; u < EMAX; u++) {
        elist[u] = -1;
        eoff[u] = 0;
        ecnt[u] = 0;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if ((int)tid < E) {
    int pos = tg_uoff[tid];
    if (pos >= 0) {
        for (int p = 0; p < P; p++) {
            if (tg_pexp[p] == (int)tid) {
                sel[pos++] = p;
            }
        }
    }
}
#endif
