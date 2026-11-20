// Row-invariant y[p, n] = sum_k x[prow[p], k] * w[pexp[p], n, k]  (engine/models/lfm2_moe/mlx_kernels.py)
// General fallback for shapes not served by rowchain/rowmma or moemm (tiny test shapes, non-aligned K, etc.).
const int e = elist[threadgroup_position_in_grid.y];
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = thread_position_in_threadgroup.x;
const int split = (int)sg % KS;
const int cg = (int)sg / KS;
const int P = prow_shape[0];
threadgroup int sel[MAXP];
threadgroup int nsel;
threadgroup T xs[KS][R > 1 ? R : 1][R > 1 ? 256 : 4];
threadgroup float part[KS > 1 ? KS : 1][KS > 1 ? R : 1][KS > 1 ? (TN * (4 / KS)) : 1];
if (e < 0) return;
if (tid == 0) {
    int c = 0;
    for (int p = 0; p < P; p++) {
        if (pexp[p] == e) { sel[c] = p; c++; }
    }
    nsel = c;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const int ns = nsel;
const int n0 = ((int)threadgroup_position_in_grid.x * (4 / KS) + cg) * TN;
const device T* wb = w + ((size_t)e * N + (size_t)min(n0, N - 1)) * K;
const int nchunk = (K + 255) / 256;
for (int c0 = 0; c0 < ns; c0 += R) {
    const int nr = min(R, ns - c0);
    float acc[R][TN];
    for (int r = 0; r < R; r++) for (int c = 0; c < TN; c++) acc[r][c] = 0.0f;
    for (int ch0 = 0; ch0 < nchunk; ch0 += KS) {
      if (R > 1) {
        for (int v = (int)tid; v < KS * R * 64; v += 128) {
            const int s = v / (R * 64), r = (v / 64) % R, kk = (v % 64) * 4;
            const int kb = (ch0 + s) * 256 + kk;
            vec<T, 4> xv4 = vec<T, 4>(0);
            if (r < nr && kb < K) {
                const device T* xr = x + (size_t)prow[sel[c0 + r]] * K + kb;
                if (kb + 4 <= K) {
                    xv4 = *((const device vec<T, 4>*)xr);
                } else {
                    for (int i = 0; i < 4; i++) xv4[i] = (kb + i < K) ? xr[i] : T(0);
                }
            }
            *((threadgroup vec<T, 4>*)&xs[s][r][kk]) = xv4;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
        const int k = (ch0 + split) * 256 + (int)lane * 8;
        if (n0 < N && ch0 + split < nchunk) {
            float wv[TN][8];
            for (int c = 0; c < TN; c++) {
                const device T* wp = wb + (size_t)c * K + k;
                if (n0 + c < N && k + 8 <= K) {
                    const vec<T, 4> a = *((const device vec<T, 4>*)wp);
                    const vec<T, 4> b = *((const device vec<T, 4>*)(wp + 4));
                    for (int i = 0; i < 4; i++) { wv[c][i] = float(a[i]); wv[c][4 + i] = float(b[i]); }
                } else {
                    for (int i = 0; i < 8; i++) wv[c][i] = (n0 + c < N && k + i < K) ? float(wp[i]) : 0.0f;
                }
            }
            for (int r = 0; r < R; r++) {
                if (r < nr) {
                    float xv[8];
                    if (R > 1) {
                        for (int i = 0; i < 8; i++) xv[i] = float(xs[split][r][lane * 8 + i]);
                    } else {
                        const device T* xr = x + (size_t)prow[sel[c0 + r]] * K + k;
                        if (k + 8 <= K) {
                            const vec<T, 4> a = *((const device vec<T, 4>*)xr);
                            const vec<T, 4> b = *((const device vec<T, 4>*)(xr + 4));
                            for (int i = 0; i < 4; i++) { xv[i] = float(a[i]); xv[4 + i] = float(b[i]); }
                        } else {
                            for (int i = 0; i < 8; i++) xv[i] = (k + i < K) ? float(xr[i]) : 0.0f;
                        }
                    }
                    for (int c = 0; c < TN; c++) {
                        float s = acc[r][c];
                        for (int i = 0; i < 8; i++) s = fma(xv[i], wv[c][i], s);
                        acc[r][c] = s;
                    }
                }
            }
        }
        if (R > 1) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (KS == 1) {
        for (int r = 0; r < nr; r++) {
            for (int c = 0; c < TN; c++) {
                const float s = simd_sum(acc[r][c]);
                if (lane == 0 && n0 + c < N) {
                    y[(size_t)sel[c0 + r] * N + n0 + c] = static_cast<OT>(s);
                }
            }
        }
    } else {
        for (int r = 0; r < nr; r++) {
            for (int c = 0; c < TN; c++) {
                const float s = simd_sum(acc[r][c]);
                if (lane == 0) part[split][r][cg * TN + c] = s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int v = (int)tid; v < nr * TN * (4 / KS); v += 128) {
            const int r = v / (TN * (4 / KS)), col = v % (TN * (4 / KS));
            const int n = (int)threadgroup_position_in_grid.x * (4 / KS) * TN + col;
            if (n < N) {
                float s = part[0][r][col];
                for (int sp = 1; sp < KS; sp++) s += part[sp][r][col];
                y[(size_t)sel[c0 + r] * N + n] = static_cast<OT>(s);
            }
        }
        if (c0 + R < ns) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
