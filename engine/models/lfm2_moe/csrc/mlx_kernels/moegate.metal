// Row-invariant 2-simdgroup MoE router gate projection with optional fused RMSNorm (FUSE_RMS)
// and optional normalized hidden state output (WRITE_NORM: threadgroup e==0 writes n_out [M, K]).
// Grid: (E * 64, 1, 1), Threadgroup: (64, 1, 1): 2 simdgroups collaborate on each expert row (4 iters for K=2048).
// For R == 1 (M == 1), loads x, w, nw in a single overlapped pass into registers (zero second x load!).
// For R > 1, uses the exact same (sg, p, lane) summation order so M=1 and M=2..64 match bit for bit.
const uint e = threadgroup_position_in_grid.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int M = x_shape[0];
const device T* wr = w + (size_t)e * K;
constexpr int NP = (K + 511) / 512;

threadgroup float sh_acc[2][R];
#if FUSE_RMS
threadgroup float sh_ss[2][R];
#endif

for (int m0 = 0; m0 < M; m0 += R) {
    const int nr = min(R, M - m0);
#if FUSE_RMS
    if (R == 1) {
        ulong2 xr_reg[NP], wr_reg[NP], nr_reg[NP];
        float ss_loc = 0.0f;
        const device T* xr = x + (size_t)m0 * K;
        for (int p = 0; p < NP; p++) {
            const int k = (p * 2 + (int)sg) * 256 + (int)lane * 8;
            if (k < K) {
                const ulong2 xraw = *((const device ulong2*)(xr + k));
                wr_reg[p] = *((const device ulong2*)(wr + k));
                nr_reg[p] = *((const device ulong2*)(nw + k));
                xr_reg[p] = xraw;
                const vec<T, 4> xa = as_type<vec<T, 4>>(xraw.x), xb = as_type<vec<T, 4>>(xraw.y);
                ss_loc = fma(float(xa[0]), float(xa[0]), ss_loc); ss_loc = fma(float(xa[1]), float(xa[1]), ss_loc);
                ss_loc = fma(float(xa[2]), float(xa[2]), ss_loc); ss_loc = fma(float(xa[3]), float(xa[3]), ss_loc);
                ss_loc = fma(float(xb[0]), float(xb[0]), ss_loc); ss_loc = fma(float(xb[1]), float(xb[1]), ss_loc);
                ss_loc = fma(float(xb[2]), float(xb[2]), ss_loc); ss_loc = fma(float(xb[3]), float(xb[3]), ss_loc);
            }
        }
        const float ss_sg = simd_sum(ss_loc);
        if (lane == 0) sh_ss[sg][0] = ss_sg;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = rsqrt((sh_ss[0][0] + sh_ss[1][0]) / (float)K + eps[0]);
        float acc = 0.0f;
        for (int p = 0; p < NP; p++) {
            const int k = (p * 2 + (int)sg) * 256 + (int)lane * 8;
            if (k < K) {
                const vec<T, 4> wa = as_type<vec<T, 4>>(wr_reg[p].x), wb = as_type<vec<T, 4>>(wr_reg[p].y);
                const vec<T, 4> na = as_type<vec<T, 4>>(nr_reg[p].x), nb4 = as_type<vec<T, 4>>(nr_reg[p].y);
                const vec<T, 4> xa = as_type<vec<T, 4>>(xr_reg[p].x), xb = as_type<vec<T, 4>>(xr_reg[p].y);
                vec<T, 4> oa, ob;
                oa[0] = static_cast<T>(float(xa[0]) * inv * float(na[0]));
                oa[1] = static_cast<T>(float(xa[1]) * inv * float(na[1]));
                oa[2] = static_cast<T>(float(xa[2]) * inv * float(na[2]));
                oa[3] = static_cast<T>(float(xa[3]) * inv * float(na[3]));
                ob[0] = static_cast<T>(float(xb[0]) * inv * float(nb4[0]));
                ob[1] = static_cast<T>(float(xb[1]) * inv * float(nb4[1]));
                ob[2] = static_cast<T>(float(xb[2]) * inv * float(nb4[2]));
                ob[3] = static_cast<T>(float(xb[3]) * inv * float(nb4[3]));
#if WRITE_NORM
                if (e == 0) {
                    *((device ulong2*)(n_out + (size_t)m0 * K + k)) = ulong2(as_type<ulong>(oa), as_type<ulong>(ob));
                }
#endif
                acc = fma(float(oa[0]), float(wa[0]), acc); acc = fma(float(oa[1]), float(wa[1]), acc);
                acc = fma(float(oa[2]), float(wa[2]), acc); acc = fma(float(oa[3]), float(wa[3]), acc);
                acc = fma(float(ob[0]), float(wb[0]), acc); acc = fma(float(ob[1]), float(wb[1]), acc);
                acc = fma(float(ob[2]), float(wb[2]), acc); acc = fma(float(ob[3]), float(wb[3]), acc);
            }
        }
        const float a_sg = simd_sum(acc);
        if (lane == 0) sh_acc[sg][0] = a_sg;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && lane == 0) out[(size_t)m0 * E + e] = static_cast<T>(sh_acc[0][0] + sh_acc[1][0]);
        continue;
    }
    for (int r = 0; r < R; r++) {
        float ss_loc = 0.0f;
        if (r < nr) {
            const device T* xr = x + (size_t)(m0 + r) * K;
            for (int p = 0; p < NP; p++) {
                const int k = (p * 2 + (int)sg) * 256 + (int)lane * 8;
                if (k < K) {
                    const ulong2 xraw = *((const device ulong2*)(xr + k));
                    const vec<T, 4> xa = as_type<vec<T, 4>>(xraw.x), xb = as_type<vec<T, 4>>(xraw.y);
                    ss_loc = fma(float(xa[0]), float(xa[0]), ss_loc); ss_loc = fma(float(xa[1]), float(xa[1]), ss_loc);
                    ss_loc = fma(float(xa[2]), float(xa[2]), ss_loc); ss_loc = fma(float(xa[3]), float(xa[3]), ss_loc);
                    ss_loc = fma(float(xb[0]), float(xb[0]), ss_loc); ss_loc = fma(float(xb[1]), float(xb[1]), ss_loc);
                    ss_loc = fma(float(xb[2]), float(xb[2]), ss_loc); ss_loc = fma(float(xb[3]), float(xb[3]), ss_loc);
                }
            }
        }
        const float ss_sg = simd_sum(ss_loc);
        if (lane == 0) sh_ss[sg][r] = ss_sg;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_r[R];
    for (int r = 0; r < R; r++) inv_r[r] = rsqrt((sh_ss[0][r] + sh_ss[1][r]) / (float)K + eps[0]);
#endif
    float acc[R];
    for (int r = 0; r < R; r++) acc[r] = 0.0f;
    for (int p = 0; p < NP; p++) {
        const int k = (p * 2 + (int)sg) * 256 + (int)lane * 8;
        if (k < K) {
            const ulong2 wraw = *((const device ulong2*)(wr + k));
            const vec<T, 4> wa = as_type<vec<T, 4>>(wraw.x), wb = as_type<vec<T, 4>>(wraw.y);
            const float w0 = float(wa[0]), w1 = float(wa[1]), w2 = float(wa[2]), w3 = float(wa[3]);
            const float w4 = float(wb[0]), w5 = float(wb[1]), w6 = float(wb[2]), w7 = float(wb[3]);
#if FUSE_RMS
            const ulong2 nwraw = *((const device ulong2*)(nw + k));
            const vec<T, 4> na = as_type<vec<T, 4>>(nwraw.x), nb4 = as_type<vec<T, 4>>(nwraw.y);
            const float n0f = float(na[0]), n1f = float(na[1]), n2f = float(na[2]), n3f = float(na[3]);
            const float n4f = float(nb4[0]), n5f = float(nb4[1]), n6f = float(nb4[2]), n7f = float(nb4[3]);
#endif
            for (int r = 0; r < R; r++) {
                if (r < nr) {
                    const ulong2 xraw = *((const device ulong2*)(x + (size_t)(m0 + r) * K + k));
                    const vec<T, 4> xa = as_type<vec<T, 4>>(xraw.x), xb = as_type<vec<T, 4>>(xraw.y);
                    float t = acc[r];
#if FUSE_RMS
                    const float inv = inv_r[r];
                    vec<T, 4> oa, ob;
                    oa[0] = static_cast<T>(float(xa[0]) * inv * n0f);
                    oa[1] = static_cast<T>(float(xa[1]) * inv * n1f);
                    oa[2] = static_cast<T>(float(xa[2]) * inv * n2f);
                    oa[3] = static_cast<T>(float(xa[3]) * inv * n3f);
                    ob[0] = static_cast<T>(float(xb[0]) * inv * n4f);
                    ob[1] = static_cast<T>(float(xb[1]) * inv * n5f);
                    ob[2] = static_cast<T>(float(xb[2]) * inv * n6f);
                    ob[3] = static_cast<T>(float(xb[3]) * inv * n7f);
#if WRITE_NORM
                    if (e == 0) {
                        *((device ulong2*)(n_out + (size_t)(m0 + r) * K + k)) = ulong2(as_type<ulong>(oa), as_type<ulong>(ob));
                    }
#endif
                    t = fma(float(oa[0]), w0, t); t = fma(float(oa[1]), w1, t);
                    t = fma(float(oa[2]), w2, t); t = fma(float(oa[3]), w3, t);
                    t = fma(float(ob[0]), w4, t); t = fma(float(ob[1]), w5, t);
                    t = fma(float(ob[2]), w6, t); t = fma(float(ob[3]), w7, t);
#else
                    t = fma(float(xa[0]), w0, t); t = fma(float(xa[1]), w1, t);
                    t = fma(float(xa[2]), w2, t); t = fma(float(xa[3]), w3, t);
                    t = fma(float(xb[0]), w4, t); t = fma(float(xb[1]), w5, t);
                    t = fma(float(xb[2]), w6, t); t = fma(float(xb[3]), w7, t);
#endif
                    acc[r] = t;
                }
            }
        }
    }
    for (int r = 0; r < R; r++) {
        const float a_sg = simd_sum(acc[r]);
        if (lane == 0) sh_acc[sg][r] = a_sg;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        for (int r = 0; r < R; r++) {
            if (r < nr) out[(size_t)(m0 + r) * E + e] = static_cast<T>(sh_acc[0][r] + sh_acc[1][r]);
        }
    }
    if (m0 + R < M) threadgroup_barrier(mem_flags::mem_threadgroup);
}
