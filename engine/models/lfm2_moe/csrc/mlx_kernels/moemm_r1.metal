// Dedicated M=1 row-invariant MoE projection (zero shared memory, zero barriers, max occupancy).
// Exact bit-for-bit twin of moemm_hyb.metal (same (BK, KS) FMA chain, ordered simd_shuffle_down).
// When SWIGLU == 1: w is [E, 2 * NOUT, K], computes gate (0..NOUT-1) and up (NOUT..2*NOUT-1) and fuses silu(a) * b.
const int u = (int)threadgroup_position_in_grid.y;
const int e = pexp[u];
const uint tid = thread_position_in_threadgroup.x;
const int s = (int)tid % KS;
const int cgi = (int)tid / KS;
const int nb = (int)threadgroup_position_in_grid.x * (TG / KS) * TN;
const int n0 = nb + cgi * TN;
if (n0 >= NOUT) return;

const int xr0 = PROW_IDENTITY ? u : 0;
const device T* xp = x + (size_t)xr0 * K;
const device T* we = w + (size_t)e * (SWIGLU ? (2 * NOUT) : NOUT) * K;
const device T* wb1 = we + (size_t)n0 * K;
const device T* wb3 = SWIGLU ? (we + (size_t)(NOUT + n0) * K) : wb1;

float acc1[TN];
float acc3[TN];
for (int c = 0; c < TN; c++) { acc1[c] = 0.0f; acc3[c] = 0.0f; }

for (int k0 = s * BK; k0 < K; k0 += KS * BK) {
    for (int kk = 0; kk < BK; kk += 8) {
        const int k = k0 + kk;
        const device vec<T, 4>* xv = (const device vec<T, 4>*)(xp + k);
        const vec<T, 4> xa = xv[0], xb = xv[1];
        const float x0 = float(xa[0]), x1 = float(xa[1]), x2 = float(xa[2]), x3 = float(xa[3]);
        const float x4 = float(xb[0]), x5 = float(xb[1]), x6 = float(xb[2]), x7 = float(xb[3]);
        for (int c = 0; c < TN; c++) {
            const int cl = (n0 + c < NOUT) ? c : 0;
            {
                const device vec<T, 4>* wv = (const device vec<T, 4>*)(wb1 + (size_t)cl * K + k);
                const vec<T, 4> wa = wv[0], wb = wv[1];
                float t = acc1[c];
                t = fma(x0, float(wa[0]), t); t = fma(x1, float(wa[1]), t);
                t = fma(x2, float(wa[2]), t); t = fma(x3, float(wa[3]), t);
                t = fma(x4, float(wb[0]), t); t = fma(x5, float(wb[1]), t);
                t = fma(x6, float(wb[2]), t); t = fma(x7, float(wb[3]), t);
                acc1[c] = t;
            }
            if (SWIGLU) {
                const device vec<T, 4>* wv = (const device vec<T, 4>*)(wb3 + (size_t)cl * K + k);
                const vec<T, 4> wa = wv[0], wb = wv[1];
                float t = acc3[c];
                t = fma(x0, float(wa[0]), t); t = fma(x1, float(wa[1]), t);
                t = fma(x2, float(wa[2]), t); t = fma(x3, float(wa[3]), t);
                t = fma(x4, float(wb[0]), t); t = fma(x5, float(wb[1]), t);
                t = fma(x6, float(wb[2]), t); t = fma(x7, float(wb[3]), t);
                acc3[c] = t;
            }
        }
    }
}

for (int c = 0; c < TN; c++) {
    float t1 = acc1[c];
    if (KS == 2) {
        t1 = t1 + simd_shuffle_down(t1, 1);
    } else if (KS == 4) {
        t1 = ((t1 + simd_shuffle_down(t1, 1)) + simd_shuffle_down(t1, 2)) + simd_shuffle_down(t1, 3);
    }
    if (SWIGLU) {
        float t3 = acc3[c];
        if (KS == 2) {
            t3 = t3 + simd_shuffle_down(t3, 1);
        } else if (KS == 4) {
            t3 = ((t3 + simd_shuffle_down(t3, 1)) + simd_shuffle_down(t3, 2)) + simd_shuffle_down(t3, 3);
        }
        if (s == 0 && n0 + c < NOUT) {
            const float a = float(static_cast<T>(t1));
            const float b = float(static_cast<T>(t3));
            y[(size_t)u * NOUT + n0 + c] = static_cast<T>((1.0f / (1.0f + exp(-a))) * a * b);
        }
    } else {
        if (s == 0 && n0 + c < NOUT) {
            y[(size_t)u * NOUT + n0 + c] = static_cast<T>(t1);
        }
    }
}
