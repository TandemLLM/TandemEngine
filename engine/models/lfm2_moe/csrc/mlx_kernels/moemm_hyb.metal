// Row-invariant M>1 hybrid MoE projection (scalar chain for ns<=2, 8x8 bfloat simdgroup_matrix MMA for ns>=3).
// Reads LPT-ordered pre-grouped (elist, eoff, ecnt, sel) from moeroute.metal (O(1) startup, zero pair scan).
// Uses union-aliased threadgroup memory and FM_MAX template (1 for M<=8, 2 for M>8 -> half the registers and
// half the xs shared memory for M=3..8, while M=9..16 still reads every expert's weights once for up to 16 rows).
// Exact bit-for-bit twin of moemm_r1.metal (same (BK, KS) FMA chain). Optional fused SwiGLU for w13.
const int u = (int)threadgroup_position_in_grid.y;
const int e = elist[u];
if (e < 0) return;

const int ns = ecnt[u];
const int start = eoff[u];
const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
constexpr int CG = 4 / KS;
constexpr int N_TILE = CG * FN * 8;
const int nb = (int)threadgroup_position_in_grid.x * N_TILE;
if (nb >= NOUT) return;

const device T* we = w + (size_t)e * (SWIGLU ? (2 * NOUT) : NOUT) * K;

if (ns <= 2) {
    const int s = (int)tid % KS;
    const int col = (int)tid / KS;
    if (col >= N_TILE) return;
    const int n = nb + col;
    const int p0 = sel[start];
    const int p1 = (ns == 2) ? sel[start + 1] : p0;
    const int xr0 = PROW_IDENTITY ? p0 : (p0 / TOPK);
    const int xr1 = PROW_IDENTITY ? p1 : (p1 / TOPK);
    const device T* xp0 = x + (size_t)xr0 * K;
    const device T* xp1 = x + (size_t)xr1 * K;
    const device T* wp1 = we + (size_t)min(n, NOUT - 1) * K;
    const device T* wp3 = SWIGLU ? (we + (size_t)(NOUT + min(n, NOUT - 1)) * K) : wp1;

    float a0 = 0.0f, a1 = 0.0f, b0 = 0.0f, b1 = 0.0f;
    if (n < NOUT) {
        if (ns == 1) {
            for (int k0 = s * BK; k0 < K; k0 += KS * BK) {
                for (int kk = 0; kk < BK; kk += 8) {
                    const int k = k0 + kk;
                    const device vec<T, 4>* xv = (const device vec<T, 4>*)(xp0 + k);
                    const vec<T, 4> xa = xv[0], xb = xv[1];
                    const float x0 = float(xa[0]), x1 = float(xa[1]), x2 = float(xa[2]), x3 = float(xa[3]);
                    const float x4 = float(xb[0]), x5 = float(xb[1]), x6 = float(xb[2]), x7 = float(xb[3]);
                    {
                        const device vec<T, 4>* wv = (const device vec<T, 4>*)(wp1 + k);
                        const vec<T, 4> wa = wv[0], wb = wv[1];
                        a0 = fma(x0, float(wa[0]), a0); a0 = fma(x1, float(wa[1]), a0);
                        a0 = fma(x2, float(wa[2]), a0); a0 = fma(x3, float(wa[3]), a0);
                        a0 = fma(x4, float(wb[0]), a0); a0 = fma(x5, float(wb[1]), a0);
                        a0 = fma(x6, float(wb[2]), a0); a0 = fma(x7, float(wb[3]), a0);
                    }
                    if (SWIGLU) {
                        const device vec<T, 4>* wv = (const device vec<T, 4>*)(wp3 + k);
                        const vec<T, 4> wa = wv[0], wb = wv[1];
                        b0 = fma(x0, float(wa[0]), b0); b0 = fma(x1, float(wa[1]), b0);
                        b0 = fma(x2, float(wa[2]), b0); b0 = fma(x3, float(wa[3]), b0);
                        b0 = fma(x4, float(wb[0]), b0); b0 = fma(x5, float(wb[1]), b0);
                        b0 = fma(x6, float(wb[2]), b0); b0 = fma(x7, float(wb[3]), b0);
                    }
                }
            }
        } else {
            for (int k0 = s * BK; k0 < K; k0 += KS * BK) {
                for (int kk = 0; kk < BK; kk += 8) {
                    const int k = k0 + kk;
                    const device vec<T, 4>* xv0 = (const device vec<T, 4>*)(xp0 + k);
                    const device vec<T, 4>* xv1 = (const device vec<T, 4>*)(xp1 + k);
                    const vec<T, 4> x0a = xv0[0], x0b = xv0[1];
                    const vec<T, 4> x1a = xv1[0], x1b = xv1[1];
                    {
                        const device vec<T, 4>* wv = (const device vec<T, 4>*)(wp1 + k);
                        const vec<T, 4> wa = wv[0], wb = wv[1];
                        const float w0 = float(wa[0]), w1 = float(wa[1]), w2 = float(wa[2]), w3 = float(wa[3]);
                        const float w4 = float(wb[0]), w5 = float(wb[1]), w6 = float(wb[2]), w7 = float(wb[3]);
                        a0 = fma(float(x0a[0]), w0, a0); a0 = fma(float(x0a[1]), w1, a0);
                        a0 = fma(float(x0a[2]), w2, a0); a0 = fma(float(x0a[3]), w3, a0);
                        a0 = fma(float(x0b[0]), w4, a0); a0 = fma(float(x0b[1]), w5, a0);
                        a0 = fma(float(x0b[2]), w6, a0); a0 = fma(float(x0b[3]), w7, a0);
                        a1 = fma(float(x1a[0]), w0, a1); a1 = fma(float(x1a[1]), w1, a1);
                        a1 = fma(float(x1a[2]), w2, a1); a1 = fma(float(x1a[3]), w3, a1);
                        a1 = fma(float(x1b[0]), w4, a1); a1 = fma(float(x1b[1]), w5, a1);
                        a1 = fma(float(x1b[2]), w6, a1); a1 = fma(float(x1b[3]), w7, a1);
                    }
                    if (SWIGLU) {
                        const device vec<T, 4>* wv = (const device vec<T, 4>*)(wp3 + k);
                        const vec<T, 4> wa = wv[0], wb = wv[1];
                        const float w0 = float(wa[0]), w1 = float(wa[1]), w2 = float(wa[2]), w3 = float(wa[3]);
                        const float w4 = float(wb[0]), w5 = float(wb[1]), w6 = float(wb[2]), w7 = float(wb[3]);
                        b0 = fma(float(x0a[0]), w0, b0); b0 = fma(float(x0a[1]), w1, b0);
                        b0 = fma(float(x0a[2]), w2, b0); b0 = fma(float(x0a[3]), w3, b0);
                        b0 = fma(float(x0b[0]), w4, b0); b0 = fma(float(x0b[1]), w5, b0);
                        b0 = fma(float(x0b[2]), w6, b0); b0 = fma(float(x0b[3]), w7, b0);
                        b1 = fma(float(x1a[0]), w0, b1); b1 = fma(float(x1a[1]), w1, b1);
                        b1 = fma(float(x1a[2]), w2, b1); b1 = fma(float(x1a[3]), w3, b1);
                        b1 = fma(float(x1b[0]), w4, b1); b1 = fma(float(x1b[1]), w5, b1);
                        b1 = fma(float(x1b[2]), w6, b1); b1 = fma(float(x1b[3]), w7, b1);
                    }
                }
            }
        }
    }
    if (KS == 2) {
        a0 = a0 + simd_shuffle_down(a0, 1);
        if (ns == 2) a1 = a1 + simd_shuffle_down(a1, 1);
        if (SWIGLU) {
            b0 = b0 + simd_shuffle_down(b0, 1);
            if (ns == 2) b1 = b1 + simd_shuffle_down(b1, 1);
        }
    } else if (KS == 4) {
        a0 = ((a0 + simd_shuffle_down(a0, 1)) + simd_shuffle_down(a0, 2)) + simd_shuffle_down(a0, 3);
        if (ns == 2) a1 = ((a1 + simd_shuffle_down(a1, 1)) + simd_shuffle_down(a1, 2)) + simd_shuffle_down(a1, 3);
        if (SWIGLU) {
            b0 = ((b0 + simd_shuffle_down(b0, 1)) + simd_shuffle_down(b0, 2)) + simd_shuffle_down(b0, 3);
            if (ns == 2) b1 = ((b1 + simd_shuffle_down(b1, 1)) + simd_shuffle_down(b1, 2)) + simd_shuffle_down(b1, 3);
        }
    }
    if (s == 0 && n < NOUT) {
        if (SWIGLU) {
            const float ga0 = float(static_cast<T>(a0)), ub0 = float(static_cast<T>(b0));
            y[(size_t)p0 * NOUT + n] = static_cast<T>((1.0f / (1.0f + exp(-ga0))) * ga0 * ub0);
            if (ns == 2) {
                const float ga1 = float(static_cast<T>(a1)), ub1 = float(static_cast<T>(b1));
                y[(size_t)p1 * NOUT + n] = static_cast<T>((1.0f / (1.0f + exp(-ga1))) * ga1 * ub1);
            }
        } else {
            y[(size_t)p0 * NOUT + n] = static_cast<T>(a0);
            if (ns == 2) y[(size_t)p1 * NOUT + n] = static_cast<T>(a1);
        }
    }
    return;
}

constexpr int KSTEP = KS * BK;
constexpr int KVECS = KSTEP / 8;
constexpr int STRIDE = KSTEP + 8;
constexpr int RPASS = FM_MAX * 8;
struct StageMem {
    T xs[RPASS][STRIDE];
    T ws[N_TILE][STRIDE];
};
struct EpiMem {
    float os1[KS][8][N_TILE];
    float os3[SWIGLU ? KS : 1][SWIGLU ? 8 : 1][SWIGLU ? N_TILE : 1];
};
union ShMem {
    StageMem st;
    EpiMem ep;
};
threadgroup ShMem sm;

const int split = (int)sg % KS;
const int cg = (int)sg / KS;

for (int c0 = 0; c0 < ns; c0 += RPASS) {
    const int nr = min(RPASS, ns - c0);
    const int fm_max = (FM_MAX == 2 && nr > 8) ? 2 : 1;
    const int mr = fm_max * 8;

    simdgroup_matrix<float, 8, 8> acc1[FM_MAX][FN];
    simdgroup_matrix<float, 8, 8> acc3[SWIGLU ? FM_MAX : 1][SWIGLU ? FN : 1];
    for (int fm = 0; fm < FM_MAX; fm++) {
        for (int f = 0; f < FN; f++) {
            acc1[fm][f] = simdgroup_matrix<float, 8, 8>(0.0f);
            if (SWIGLU) acc3[fm][f] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (int k0 = 0; k0 < K; k0 += KSTEP) {
        for (int v = (int)tid; v < mr * KVECS; v += 128) {
            const int r = v / KVECS;
            const int kk = (v % KVECS) * 8;
            if (r < nr) {
                const int p = sel[start + c0 + r];
                const int xr = PROW_IDENTITY ? p : (p / TOPK);
                const device vec<T, 4>* xp = (const device vec<T, 4>*)(x + (size_t)xr * K + k0 + kk);
                *((threadgroup vec<T, 4>*)&sm.st.xs[r][kk])     = xp[0];
                *((threadgroup vec<T, 4>*)&sm.st.xs[r][kk + 4]) = xp[1];
            } else {
                *((threadgroup vec<T, 4>*)&sm.st.xs[r][kk])     = vec<T, 4>(0);
                *((threadgroup vec<T, 4>*)&sm.st.xs[r][kk + 4]) = vec<T, 4>(0);
            }
        }
        for (int v = (int)tid; v < N_TILE * KVECS; v += 128) {
            const int c = v / KVECS;
            const int kk = (v % KVECS) * 8;
            const int n = nb + c;
            if (n < NOUT) {
                const device vec<T, 4>* wp = (const device vec<T, 4>*)(we + (size_t)n * K + k0 + kk);
                *((threadgroup vec<T, 4>*)&sm.st.ws[c][kk])     = wp[0];
                *((threadgroup vec<T, 4>*)&sm.st.ws[c][kk + 4]) = wp[1];
            } else {
                *((threadgroup vec<T, 4>*)&sm.st.ws[c][kk])     = vec<T, 4>(0);
                *((threadgroup vec<T, 4>*)&sm.st.ws[c][kk + 4]) = vec<T, 4>(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = split * BK; kk < (split + 1) * BK; kk += 8) {
            simdgroup_matrix<T, 8, 8> a[FM_MAX];
            simdgroup_load(a[0], &sm.st.xs[0][kk], STRIDE);
            if (FM_MAX == 2 && fm_max == 2) simdgroup_load(a[1], &sm.st.xs[8][kk], STRIDE);
            for (int f = 0; f < FN; f++) {
                simdgroup_matrix<T, 8, 8> b;
                simdgroup_load(b, &sm.st.ws[(cg * FN + f) * 8][kk], STRIDE, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc1[0][f], a[0], b, acc1[0][f]);
                if (FM_MAX == 2 && fm_max == 2) simdgroup_multiply_accumulate(acc1[1][f], a[1], b, acc1[1][f]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (SWIGLU) {
            for (int v = (int)tid; v < N_TILE * KVECS; v += 128) {
                const int c = v / KVECS;
                const int kk = (v % KVECS) * 8;
                const int n = nb + c;
                if (n < NOUT) {
                    const device vec<T, 4>* wp = (const device vec<T, 4>*)(we + (size_t)(NOUT + n) * K + k0 + kk);
                    *((threadgroup vec<T, 4>*)&sm.st.ws[c][kk])     = wp[0];
                    *((threadgroup vec<T, 4>*)&sm.st.ws[c][kk + 4]) = wp[1];
                } else {
                    *((threadgroup vec<T, 4>*)&sm.st.ws[c][kk])     = vec<T, 4>(0);
                    *((threadgroup vec<T, 4>*)&sm.st.ws[c][kk + 4]) = vec<T, 4>(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int kk = split * BK; kk < (split + 1) * BK; kk += 8) {
                simdgroup_matrix<T, 8, 8> a[FM_MAX];
                simdgroup_load(a[0], &sm.st.xs[0][kk], STRIDE);
                if (FM_MAX == 2 && fm_max == 2) simdgroup_load(a[1], &sm.st.xs[8][kk], STRIDE);
                for (int f = 0; f < FN; f++) {
                    simdgroup_matrix<T, 8, 8> b;
                    simdgroup_load(b, &sm.st.ws[(cg * FN + f) * 8][kk], STRIDE, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(acc3[0][f], a[0], b, acc3[0][f]);
                    if (FM_MAX == 2 && fm_max == 2) simdgroup_multiply_accumulate(acc3[1][f], a[1], b, acc3[1][f]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (int fm = 0; fm < fm_max; fm++) {
        for (int f = 0; f < FN; f++) {
            simdgroup_store(acc1[fm][f], &sm.ep.os1[split][0][(cg * FN + f) * 8], N_TILE);
            if (SWIGLU) simdgroup_store(acc3[fm][f], &sm.ep.os3[split][0][(cg * FN + f) * 8], N_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int v = (int)tid; v < 8 * N_TILE; v += 128) {
            const int lr = v / N_TILE;
            const int c = v % N_TILE;
            const int r = fm * 8 + lr;
            const int n = nb + c;
            if (r < nr && n < NOUT) {
                const int p = sel[start + c0 + r];
                float s1 = sm.ep.os1[0][lr][c];
                for (int sp = 1; sp < KS; sp++) s1 += sm.ep.os1[sp][lr][c];
                if (SWIGLU) {
                    float s3 = sm.ep.os3[0][lr][c];
                    for (int sp = 1; sp < KS; sp++) s3 += sm.ep.os3[sp][lr][c];
                    const float a = float(static_cast<T>(s1));
                    const float b = float(static_cast<T>(s3));
                    y[(size_t)p * NOUT + n] = static_cast<T>((1.0f / (1.0f + exp(-a))) * a * b);
                } else {
                    y[(size_t)p * NOUT + n] = static_cast<T>(s1);
                }
            }
        }
        if (fm + 1 < fm_max || c0 + RPASS < ns) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
