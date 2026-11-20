// Row-invariant FlashAttention-2 Split-K on 8x8 bfloat simdgroup_matrix units (D == 64, GQA == 4).
// Each logical key chunk [ch * BK, (ch + 1) * BK) is processed in fixed BT=32 tiles using 8x8 simdgroup MMA,
// in-register online softmax (simd_shuffle_xor(1) / simd_shuffle_xor(8)), and in-register O rescale.
// A threadgroup shares a single 4.6 KB bfloat kv[BT][D+8] buffer across all GQA=4 heads and BM rows
// (BM=2/NSG=1 for M<=2, BM=4/NSG=2 for M<=4, BM=8/NSG=4 for M<=8, BM=16/NSG=8 for M>8 -> QT_PER_SG=1 always).
// 100% bit-for-bit row-invariant across M=1..64, arbitrary trees, and any L/dlen split.
const uint hk = threadgroup_position_in_grid.x;
const uint ch = threadgroup_position_in_grid.y;
const uint rb = threadgroup_position_in_grid.z;
const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;

const int M = q_shape[0];
const int Hq = q_shape[1];
const int cap = kc_shape[1];
const int A = anc_shape[1];
const int L = lb[0];
const int nchunks = (int)threadgroups_per_grid.y;
const int r0 = (int)rb * BM;
const float scale = rsqrt((float)D) * 1.4426950408889634f; // log2(e) / sqrt(D)

const int j_start = (int)ch * BK;
const int j_end_cap = j_start + BK;

constexpr int NTHREADS = NSG * 32;
constexpr int NQT = (BM * GQA + 7) / 8;
constexpr int BT = 32;
constexpr int NT_BLOCKS = BT / 8;
constexpr int ND_BLOCKS = D / 8;
constexpr int KV_STRIDE = D + 8;

// Apple Silicon 8x8 simdgroup_matrix lane -> (row, col0) mapping
const int my_qr = (int)(((lane & 16) >> 2) | ((lane & 6) >> 1));
const int my_c0 = (int)(((lane & 8) >> 1) | ((lane & 1) << 1));

threadgroup T kv[BT][KV_STRIDE];

// Load Q into registers (1 Q-tile = 2 rows x GQA(4) heads per simdgroup) via kv scratch
simdgroup_matrix<T, 8, 8> Qm[ND_BLOCKS];
simdgroup_matrix<float, 8, 8> O[ND_BLOCKS];
float m_cur = -INFINITY;
float l_cur = 0.0f;
const int qt = (int)sg;

for (int sg_batch = 0; sg_batch < NSG; sg_batch += 4) {
    if ((int)sg >= sg_batch && (int)sg < sg_batch + 4) {
        const int sg_loc = (int)sg - sg_batch;
        for (int idx = (int)lane; idx < 8 * (D / 4); idx += 32) {
            const int lr = idx / (D / 4);
            const int d4 = (idx % (D / 4)) * 4;
            const int gr = r0 + qt * 2 + (lr / GQA);
            const int gh = (int)hk * GQA + (lr % GQA);
            if (qt < NQT && gr < M) {
                const device vec<T, 4>* qp = (const device vec<T, 4>*)(q + ((size_t)gr * Hq + gh) * D + d4);
                *((threadgroup vec<T, 4>*)&kv[sg_loc * 8 + lr][d4]) = *qp;
            } else {
                *((threadgroup vec<T, 4>*)&kv[sg_loc * 8 + lr][d4]) = vec<T, 4>(0);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int db = 0; db < ND_BLOCKS; db++) {
            simdgroup_load(Qm[db], &kv[sg_loc * 8][db * 8], KV_STRIDE);
            O[db] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

const device T* kbase = kc + (size_t)hk * cap * D;
const device T* vbase = vc + (size_t)hk * cap * D;

const int full_prefix_end = (j_end_cap <= L)
    ? j_end_cap
    : (j_start + ((max(0, min(j_end_cap, L) - j_start)) / BT) * BT);

// Shared prefix blocks: ALL rows and GQA heads share the exact same K and V loads!
for (int jb = j_start; jb < full_prefix_end; jb += BT) {
    for (int v = (int)tid; v < BT * (D / 8); v += NTHREADS) {
        const int t = v / (D / 8);
        const int d8 = (v % (D / 8)) * 8;
        const device vec<T, 4>* kp = (const device vec<T, 4>*)(kbase + (size_t)(jb + t) * D + d8);
        *((threadgroup vec<T, 4>*)&kv[t][d8])     = kp[0];
        *((threadgroup vec<T, 4>*)&kv[t][d8 + 4]) = kp[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<T, 8, 8> Pm[NT_BLOCKS];
    float cf = 1.0f;
    if (qt < NQT) {
        simdgroup_matrix<float, 8, 8> S[NT_BLOCKS];
        for (int tb = 0; tb < NT_BLOCKS; tb++) S[tb] = simdgroup_matrix<float, 8, 8>(0.0f);
        for (int db = 0; db < ND_BLOCKS; db++) {
            for (int tb = 0; tb < NT_BLOCKS; tb++) {
                simdgroup_matrix<T, 8, 8> km;
                simdgroup_load(km, &kv[tb * 8][db * 8], KV_STRIDE, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[tb], Qm[db], km, S[tb]);
            }
        }
        float mt = -INFINITY;
        for (int tb = 0; tb < NT_BLOCKS; tb++) {
            S[tb].thread_elements()[0] *= scale;
            S[tb].thread_elements()[1] *= scale;
            mt = max(mt, max(S[tb].thread_elements()[0], S[tb].thread_elements()[1]));
        }
        mt = max(mt, simd_shuffle_xor(mt, 1));
        mt = max(mt, simd_shuffle_xor(mt, 8));

        const float mo = m_cur;
        const float mn = max(mo, mt);
        cf = (mo == -INFINITY) ? 0.0f : exp2(mo - mn);
        float ls = 0.0f;
        for (int tb = 0; tb < NT_BLOCKS; tb++) {
            const float p0 = exp2(S[tb].thread_elements()[0] - mn);
            const float p1 = exp2(S[tb].thread_elements()[1] - mn);
            ls += p0 + p1;
            Pm[tb].thread_elements()[0] = static_cast<T>(p0);
            Pm[tb].thread_elements()[1] = static_cast<T>(p1);
        }
        ls = ls + simd_shuffle_xor(ls, 1);
        ls = ls + simd_shuffle_xor(ls, 8);
        l_cur = l_cur * cf + ls;
        m_cur = mn;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int v = (int)tid; v < BT * (D / 8); v += NTHREADS) {
        const int t = v / (D / 8);
        const int d8 = (v % (D / 8)) * 8;
        const device vec<T, 4>* vp = (const device vec<T, 4>*)(vbase + (size_t)(jb + t) * D + d8);
        *((threadgroup vec<T, 4>*)&kv[t][d8])     = vp[0];
        *((threadgroup vec<T, 4>*)&kv[t][d8 + 4]) = vp[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (qt < NQT) {
        for (int db = 0; db < ND_BLOCKS; db++) {
            if (jb > j_start) {
                O[db].thread_elements()[0] *= cf;
                O[db].thread_elements()[1] *= cf;
            }
            for (int tb = 0; tb < NT_BLOCKS; tb++) {
                simdgroup_matrix<T, 8, 8> vm;
                simdgroup_load(vm, &kv[tb * 8][db * 8], KV_STRIDE);
                simdgroup_multiply_accumulate(O[db], Pm[tb], vm, O[db]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Tail blocks crossing L: each row uses its own path anc[gr] with the exact same 8x8 MMA math
if (full_prefix_end < j_end_cap) {
    const int nr = min(BM, M - r0);
    for (int lr = 0; lr < nr; lr++) {
        const int gr = r0 + lr;
        const int n_r = L + dl[gr];
        const int target_sg = lr / 2;
        const int row_off = (lr % 2) * 4;
        for (int jb = full_prefix_end; jb < j_end_cap; jb += BT) {
            if (jb >= n_r) break;
            for (int v = (int)tid; v < BT * (D / 8); v += NTHREADS) {
                const int t = v / (D / 8);
                const int d8 = (v % (D / 8)) * 8;
                const int j = jb + t;
                if (j < n_r) {
                    const int slot = (j < L) ? j : (L + anc[(size_t)gr * A + (j - L)]);
                    const device vec<T, 4>* kp = (const device vec<T, 4>*)(kbase + (size_t)slot * D + d8);
                    *((threadgroup vec<T, 4>*)&kv[t][d8])     = kp[0];
                    *((threadgroup vec<T, 4>*)&kv[t][d8 + 4]) = kp[1];
                } else {
                    *((threadgroup vec<T, 4>*)&kv[t][d8])     = vec<T, 4>(0);
                    *((threadgroup vec<T, 4>*)&kv[t][d8 + 4]) = vec<T, 4>(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<T, 8, 8> Pm_tail[NT_BLOCKS];
            float cf_tail = 1.0f;
            if ((int)sg == target_sg) {
                simdgroup_matrix<float, 8, 8> S[NT_BLOCKS];
                for (int tb = 0; tb < NT_BLOCKS; tb++) S[tb] = simdgroup_matrix<float, 8, 8>(0.0f);
                for (int db = 0; db < ND_BLOCKS; db++) {
                    for (int tb = 0; tb < NT_BLOCKS; tb++) {
                        simdgroup_matrix<T, 8, 8> km;
                        simdgroup_load(km, &kv[tb * 8][db * 8], KV_STRIDE, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(S[tb], Qm[db], km, S[tb]);
                    }
                }
                const bool active_row = (my_qr >= row_off && my_qr < row_off + 4);
                if (active_row) {
                    float mt = -INFINITY;
                    for (int tb = 0; tb < NT_BLOCKS; tb++) {
                        const int t0 = jb + tb * 8 + my_c0;
                        const float s0 = (t0     < n_r) ? (S[tb].thread_elements()[0] * scale) : -INFINITY;
                        const float s1 = (t0 + 1 < n_r) ? (S[tb].thread_elements()[1] * scale) : -INFINITY;
                        S[tb].thread_elements()[0] = s0;
                        S[tb].thread_elements()[1] = s1;
                        mt = max(mt, max(s0, s1));
                    }
                    mt = max(mt, simd_shuffle_xor(mt, 1));
                    mt = max(mt, simd_shuffle_xor(mt, 8));
                    const float mo = m_cur;
                    const float mn = max(mo, mt);
                    cf_tail = (mo == -INFINITY) ? 0.0f : exp2(mo - mn);
                    float ls = 0.0f;
                    for (int tb = 0; tb < NT_BLOCKS; tb++) {
                        const int t0 = jb + tb * 8 + my_c0;
                        const float p0 = (t0     < n_r) ? exp2(S[tb].thread_elements()[0] - mn) : 0.0f;
                        const float p1 = (t0 + 1 < n_r) ? exp2(S[tb].thread_elements()[1] - mn) : 0.0f;
                        ls += p0 + p1;
                        Pm_tail[tb].thread_elements()[0] = static_cast<T>(p0);
                        Pm_tail[tb].thread_elements()[1] = static_cast<T>(p1);
                    }
                    ls = ls + simd_shuffle_xor(ls, 1);
                    ls = ls + simd_shuffle_xor(ls, 8);
                    l_cur = l_cur * cf_tail + ls;
                    m_cur = mn;
                } else {
                    for (int tb = 0; tb < NT_BLOCKS; tb++) {
                        Pm_tail[tb].thread_elements()[0] = T(0);
                        Pm_tail[tb].thread_elements()[1] = T(0);
                    }
                    cf_tail = 1.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int v = (int)tid; v < BT * (D / 8); v += NTHREADS) {
                const int t = v / (D / 8);
                const int d8 = (v % (D / 8)) * 8;
                const int j = jb + t;
                if (j < n_r) {
                    const int slot = (j < L) ? j : (L + anc[(size_t)gr * A + (j - L)]);
                    const device vec<T, 4>* vp = (const device vec<T, 4>*)(vbase + (size_t)slot * D + d8);
                    *((threadgroup vec<T, 4>*)&kv[t][d8])     = vp[0];
                    *((threadgroup vec<T, 4>*)&kv[t][d8 + 4]) = vp[1];
                } else {
                    *((threadgroup vec<T, 4>*)&kv[t][d8])     = vec<T, 4>(0);
                    *((threadgroup vec<T, 4>*)&kv[t][d8 + 4]) = vec<T, 4>(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if ((int)sg == target_sg) {
                for (int db = 0; db < ND_BLOCKS; db++) {
                    O[db].thread_elements()[0] *= cf_tail;
                    O[db].thread_elements()[1] *= cf_tail;
                    for (int tb = 0; tb < NT_BLOCKS; tb++) {
                        simdgroup_matrix<T, 8, 8> vm;
                        simdgroup_load(vm, &kv[tb * 8][db * 8], KV_STRIDE);
                        simdgroup_multiply_accumulate(O[db], Pm_tail[tb], vm, O[db]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

if (qt < NQT) {
    const int gr = r0 + qt * 2 + (my_qr / GQA);
    const int gh = (int)hk * GQA + (my_qr % GQA);
    if (gr < M) {
#if SINGLE_CHUNK
        const float invl = 1.0f / l_cur;
        device T* op = o + ((size_t)gr * Hq + gh) * D + my_c0;
        for (int db = 0; db < ND_BLOCKS; db++) {
            op[db * 8]     = static_cast<T>(O[db].thread_elements()[0] * invl);
            op[db * 8 + 1] = static_cast<T>(O[db].thread_elements()[1] * invl);
        }
#else
        const size_t base = ((size_t)gr * Hq + gh) * nchunks + ch;
        if (my_c0 == 0) {
            pm[base] = m_cur;
            pl[base] = l_cur;
        }
        device float* pap = pa + base * D + my_c0;
        for (int db = 0; db < ND_BLOCKS; db++) {
            pap[db * 8]     = O[db].thread_elements()[0];
            pap[db * 8 + 1] = O[db].thread_elements()[1];
        }
#endif
    }
}
