// Row-invariant dense projection, the scalar twin of rowmma.metal (engine/models/lfm2_moe/mlx_kernels.py `dense`).
// y[m, n] = sum_k x[m, k] w[n, k] (+ res[m, n] when ADD_RES == 1).
// A thread is (TN output columns, one of the KS K-splits, R rows); a threadgroup is TG / KS column groups x KS splits.
// Split s takes [s BK, (s + 1) BK) of each KS BK step of K, as rowmma's does, and sums it in one sequential chain of
// fused multiply-adds from 0, k ascending; the KS adjacent lanes in the same simdgroup reduce in split order via
// simd_shuffle_down (zero threadgroup memory, zero barriers). Loads x once per 8-wide step and streams w per column
// to keep register pressure minimal. K is a multiple of KS BK and BK of 8 (`dense` checks).
const uint tid = thread_position_in_threadgroup.x;
const int M = x_shape[0];
const int s = (int)tid % KS;
const int cgi = (int)tid / KS;
const int nb = (int)threadgroup_position_in_grid.x * (TG / KS) * TN;
const int n0 = nb + cgi * TN;
const int m0 = (int)threadgroup_position_in_grid.y * R;
float acc[R][TN];
for (int r = 0; r < R; r++) for (int c = 0; c < TN; c++) acc[r][c] = 0.0f;
const device T* wb = w + (size_t)min(n0, N - 1) * K;
if (n0 < N) {
    for (int k0 = s * BK; k0 < K; k0 += KS * BK) {
        for (int kk = 0; kk < BK; kk += 8) {
            const int k = k0 + kk;
            if (R == 1) {
                const device vec<T, 4>* xv = (const device vec<T, 4>*)(x + (size_t)m0 * K + k);
                const vec<T, 4> xa = xv[0], xb = xv[1];
                const float x0 = float(xa[0]), x1 = float(xa[1]), x2 = float(xa[2]), x3 = float(xa[3]);
                const float x4 = float(xb[0]), x5 = float(xb[1]), x6 = float(xb[2]), x7 = float(xb[3]);
                for (int c = 0; c < TN; c++) {
                    const device vec<T, 4>* wv = (const device vec<T, 4>*)(wb + (size_t)(n0 + c < N ? c : 0) * K + k);
                    const vec<T, 4> wa = wv[0], wb4 = wv[1];
                    float t = acc[0][c];
                    t = fma(x0, float(wa[0]), t); t = fma(x1, float(wa[1]), t);
                    t = fma(x2, float(wa[2]), t); t = fma(x3, float(wa[3]), t);
                    t = fma(x4, float(wb4[0]), t); t = fma(x5, float(wb4[1]), t);
                    t = fma(x6, float(wb4[2]), t); t = fma(x7, float(wb4[3]), t);
                    acc[0][c] = t;
                }
            } else {
                float xv[R][8];
                for (int r = 0; r < R; r++) {
                    if (m0 + r < M) {
                        const device vec<T, 4>* xp = (const device vec<T, 4>*)(x + (size_t)(m0 + r) * K + k);
                        const vec<T, 4> xa = xp[0], xb = xp[1];
                        xv[r][0] = float(xa[0]); xv[r][1] = float(xa[1]); xv[r][2] = float(xa[2]); xv[r][3] = float(xa[3]);
                        xv[r][4] = float(xb[0]); xv[r][5] = float(xb[1]); xv[r][6] = float(xb[2]); xv[r][7] = float(xb[3]);
                    }
                }
                for (int c = 0; c < TN; c++) {
                    const device vec<T, 4>* wv = (const device vec<T, 4>*)(wb + (size_t)(n0 + c < N ? c : 0) * K + k);
                    const vec<T, 4> wa = wv[0], wb4 = wv[1];
                    const float w0 = float(wa[0]), w1 = float(wa[1]), w2 = float(wa[2]), w3 = float(wa[3]);
                    const float w4 = float(wb4[0]), w5 = float(wb4[1]), w6 = float(wb4[2]), w7 = float(wb4[3]);
                    for (int r = 0; r < R; r++) {
                        if (m0 + r < M) {
                            float t = acc[r][c];
                            t = fma(xv[r][0], w0, t); t = fma(xv[r][1], w1, t);
                            t = fma(xv[r][2], w2, t); t = fma(xv[r][3], w3, t);
                            t = fma(xv[r][4], w4, t); t = fma(xv[r][5], w5, t);
                            t = fma(xv[r][6], w6, t); t = fma(xv[r][7], w7, t);
                            acc[r][c] = t;
                        }
                    }
                }
            }
        }
    }
}
// Exact ordered reduction across KS adjacent lanes in the same simdgroup (KS in {1, 2, 4, 8})
for (int r = 0; r < R; r++) {
    for (int c = 0; c < TN; c++) {
        float t = acc[r][c];
        if (KS == 2) {
            t = t + simd_shuffle_down(t, 1);
        } else if (KS == 4) {
            const float s1 = simd_shuffle_down(t, 1);
            const float s2 = simd_shuffle_down(t, 2);
            const float s3 = simd_shuffle_down(t, 3);
            t = ((t + s1) + s2) + s3;
        } else if (KS == 8) {
            const float s1 = simd_shuffle_down(t, 1);
            const float s2 = simd_shuffle_down(t, 2);
            const float s3 = simd_shuffle_down(t, 3);
            const float s4 = simd_shuffle_down(t, 4);
            const float s5 = simd_shuffle_down(t, 5);
            const float s6 = simd_shuffle_down(t, 6);
            const float s7 = simd_shuffle_down(t, 7);
            t = ((((((t + s1) + s2) + s3) + s4) + s5) + s6) + s7;
        }
        if (s == 0 && m0 + r < M && n0 + c < N) {
            const size_t idx = (size_t)(m0 + r) * N + n0 + c;
#if ADD_RES
            y[idx] = static_cast<OT>(float(res[idx]) + float(static_cast<T>(t)));
#else
            y[idx] = static_cast<OT>(static_cast<T>(t));
#endif
        }
    }
}
