// Row-invariant 4-simdgroup cooperative FlashAttention-2 Split-K for M <= 2 (D == 64, GQA == 4).
// Exact bit-for-bit twin of pathattn_flash.metal (M >= 3): same ATTN_BK=256, BT=32, 8x8 bfloat MMA,
// and exact same softmax reduction order. Uses 128 threads/TG (4 simdgroups) to load K/V 4x faster,
// parallelize Q@K^T across tb=0..3 and P@V across db=0..7, with only 3 barriers per BT=32 tile.
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

constexpr int NTHREADS = 128;
constexpr int BT = 32;
constexpr int ND_BLOCKS = D / 8; // 8
constexpr int KV_STRIDE = D + 8;

const int my_qr = (int)(((lane & 16) >> 2) | ((lane & 6) >> 1));
const int my_c0 = (int)(((lane & 8) >> 1) | ((lane & 1) << 1));

threadgroup T kv_k[BT][KV_STRIDE];
threadgroup T kv_v[BT][KV_STRIDE];
threadgroup float sm_mt[4][8];
threadgroup float sm_ls[4][32];
threadgroup T sm_pm[8][32];

// Load Q [8, 64] into kv_k[0..7] collaboratively with 128 threads
for (int idx = (int)tid; idx < 8 * (D / 4); idx += NTHREADS) {
    const int lr = idx / (D / 4);
    const int d4 = (idx % (D / 4)) * 4;
    const int gr = r0 + (lr / GQA);
    const int gh = (int)hk * GQA + (lr % GQA);
    if (gr < M) {
        const device vec<T, 4>* qp = (const device vec<T, 4>*)(q + ((size_t)gr * Hq + gh) * D + d4);
        *((threadgroup vec<T, 4>*)&kv_k[lr][d4]) = *qp;
    } else {
        *((threadgroup vec<T, 4>*)&kv_k[lr][d4]) = vec<T, 4>(0);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

simdgroup_matrix<T, 8, 8> Qm[ND_BLOCKS];
for (int db = 0; db < ND_BLOCKS; db++) simdgroup_load(Qm[db], &kv_k[0][db * 8], KV_STRIDE);

// Each simdgroup owns 2 of the 8 D-blocks of O (d_base = sg * 16)
simdgroup_matrix<float, 8, 8> O0 = simdgroup_matrix<float, 8, 8>(0.0f);
simdgroup_matrix<float, 8, 8> O1 = simdgroup_matrix<float, 8, 8>(0.0f);
float m_cur = -INFINITY;
float l_cur = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);

const device T* kbase = kc + (size_t)hk * cap * D;
const device T* vbase = vc + (size_t)hk * cap * D;

const int full_prefix_end = (j_end_cap <= L)
    ? j_end_cap
    : (j_start + ((max(0, min(j_end_cap, L) - j_start)) / BT) * BT);

// Shared prefix blocks: 3 barriers per BT=32 tile (kv_k and kv_v ping-pong hazard-free)
for (int jb = j_start; jb < full_prefix_end; jb += BT) {
    for (int v = (int)tid; v < BT * (D / 8); v += NTHREADS) {
        const int t = v / (D / 8);
        const int d8 = (v % (D / 8)) * 8;
        *((threadgroup ulong2*)&kv_k[t][d8]) = *((const device ulong2*)(kbase + (size_t)(jb + t) * D + d8));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // B1: kv_k ready

    simdgroup_matrix<float, 8, 8> St = simdgroup_matrix<float, 8, 8>(0.0f);
    for (int db = 0; db < ND_BLOCKS; db++) {
        simdgroup_matrix<T, 8, 8> km;
        simdgroup_load(km, &kv_k[sg * 8][db * 8], KV_STRIDE, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(St, Qm[db], km, St);
    }
    const float s0 = St.thread_elements()[0] * scale;
    const float s1 = St.thread_elements()[1] * scale;
    float mt_loc = max(s0, s1);
    mt_loc = max(mt_loc, simd_shuffle_xor(mt_loc, 1));
    mt_loc = max(mt_loc, simd_shuffle_xor(mt_loc, 8));
    if (my_c0 == 0) sm_mt[sg][my_qr] = mt_loc;
    threadgroup_barrier(mem_flags::mem_threadgroup); // B2: sm_mt ready, kv_k & prev kv_v idle

    const float mt = max(max(sm_mt[0][my_qr], sm_mt[1][my_qr]), max(sm_mt[2][my_qr], sm_mt[3][my_qr]));
    const float mo = m_cur;
    const float mn = max(mo, mt);
    const float cf = (mo == -INFINITY) ? 0.0f : exp2(mo - mn);
    const float p0 = exp2(s0 - mn);
    const float p1 = exp2(s1 - mn);
    sm_ls[sg][lane] = p0 + p1;
    sm_pm[my_qr][sg * 8 + my_c0]     = static_cast<T>(p0);
    sm_pm[my_qr][sg * 8 + my_c0 + 1] = static_cast<T>(p1);

    for (int v = (int)tid; v < BT * (D / 8); v += NTHREADS) {
        const int t = v / (D / 8);
        const int d8 = (v % (D / 8)) * 8;
        *((threadgroup ulong2*)&kv_v[t][d8]) = *((const device ulong2*)(vbase + (size_t)(jb + t) * D + d8));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // B3: sm_ls, sm_pm, kv_v ready

    float ls = (((0.0f + sm_ls[0][lane]) + sm_ls[1][lane]) + sm_ls[2][lane]) + sm_ls[3][lane];
    ls = ls + simd_shuffle_xor(ls, 1);
    ls = ls + simd_shuffle_xor(ls, 8);
    l_cur = l_cur * cf + ls;
    m_cur = mn;

    if (jb > j_start) {
        O0.thread_elements()[0] *= cf; O0.thread_elements()[1] *= cf;
        O1.thread_elements()[0] *= cf; O1.thread_elements()[1] *= cf;
    }
    const int d_base = (int)sg * 16;
    for (int tb = 0; tb < 4; tb++) {
        simdgroup_matrix<T, 8, 8> pm, vm0, vm1;
        simdgroup_load(pm, &sm_pm[0][tb * 8], 32);
        simdgroup_load(vm0, &kv_v[tb * 8][d_base], KV_STRIDE);
        simdgroup_load(vm1, &kv_v[tb * 8][d_base + 8], KV_STRIDE);
        simdgroup_multiply_accumulate(O0, pm, vm0, O0);
        simdgroup_multiply_accumulate(O1, pm, vm1, O1);
    }
}

// Tail blocks crossing L: each row uses its own path anc[gr] with the exact same 8x8 MMA math
if (full_prefix_end < j_end_cap) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int nr = min(BM, M - r0);
    for (int lr = 0; lr < nr; lr++) {
        const int gr = r0 + lr;
        const int n_r = L + dl[gr];
        const int row_off = lr * 4;
        for (int jb = full_prefix_end; jb < j_end_cap; jb += BT) {
            if (jb >= n_r) break;
            for (int v = (int)tid; v < BT * (D / 8); v += NTHREADS) {
                const int t = v / (D / 8);
                const int d8 = (v % (D / 8)) * 8;
                const int j = jb + t;
                if (j < n_r) {
                    const int slot = (j < L) ? j : (L + anc[(size_t)gr * A + (j - L)]);
                    *((threadgroup ulong2*)&kv_k[t][d8]) = *((const device ulong2*)(kbase + (size_t)slot * D + d8));
                } else {
                    *((threadgroup ulong2*)&kv_k[t][d8]) = ulong2(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<float, 8, 8> St = simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < ND_BLOCKS; db++) {
                simdgroup_matrix<T, 8, 8> km;
                simdgroup_load(km, &kv_k[sg * 8][db * 8], KV_STRIDE, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(St, Qm[db], km, St);
            }
            const bool active_row = (my_qr >= row_off && my_qr < row_off + 4);
            const int t0 = jb + (int)sg * 8 + my_c0;
            const float s0 = (active_row && t0     < n_r) ? (St.thread_elements()[0] * scale) : -INFINITY;
            const float s1 = (active_row && t0 + 1 < n_r) ? (St.thread_elements()[1] * scale) : -INFINITY;
            float mt_loc = max(s0, s1);
            mt_loc = max(mt_loc, simd_shuffle_xor(mt_loc, 1));
            mt_loc = max(mt_loc, simd_shuffle_xor(mt_loc, 8));
            if (my_c0 == 0) sm_mt[sg][my_qr] = mt_loc;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float cf_tail = 1.0f;
            if (active_row) {
                const float mt = max(max(sm_mt[0][my_qr], sm_mt[1][my_qr]), max(sm_mt[2][my_qr], sm_mt[3][my_qr]));
                const float mo = m_cur;
                const float mn = max(mo, mt);
                cf_tail = (mo == -INFINITY) ? 0.0f : exp2(mo - mn);
                const float p0 = (t0     < n_r) ? exp2(s0 - mn) : 0.0f;
                const float p1 = (t0 + 1 < n_r) ? exp2(s1 - mn) : 0.0f;
                sm_ls[sg][lane] = p0 + p1;
                sm_pm[my_qr][sg * 8 + my_c0]     = static_cast<T>(p0);
                sm_pm[my_qr][sg * 8 + my_c0 + 1] = static_cast<T>(p1);
                m_cur = mn;
            } else {
                sm_ls[sg][lane] = 0.0f;
                sm_pm[my_qr][sg * 8 + my_c0]     = T(0);
                sm_pm[my_qr][sg * 8 + my_c0 + 1] = T(0);
            }
            for (int v = (int)tid; v < BT * (D / 8); v += NTHREADS) {
                const int t = v / (D / 8);
                const int d8 = (v % (D / 8)) * 8;
                const int j = jb + t;
                if (j < n_r) {
                    const int slot = (j < L) ? j : (L + anc[(size_t)gr * A + (j - L)]);
                    *((threadgroup ulong2*)&kv_v[t][d8]) = *((const device ulong2*)(vbase + (size_t)slot * D + d8));
                } else {
                    *((threadgroup ulong2*)&kv_v[t][d8]) = ulong2(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (active_row) {
                float ls = (((0.0f + sm_ls[0][lane]) + sm_ls[1][lane]) + sm_ls[2][lane]) + sm_ls[3][lane];
                ls = ls + simd_shuffle_xor(ls, 1);
                ls = ls + simd_shuffle_xor(ls, 8);
                l_cur = l_cur * cf_tail + ls;
            }
            O0.thread_elements()[0] *= cf_tail; O0.thread_elements()[1] *= cf_tail;
            O1.thread_elements()[0] *= cf_tail; O1.thread_elements()[1] *= cf_tail;
            const int d_base = (int)sg * 16;
            for (int tb = 0; tb < 4; tb++) {
                simdgroup_matrix<T, 8, 8> pm, vm0, vm1;
                simdgroup_load(pm, &sm_pm[0][tb * 8], 32);
                simdgroup_load(vm0, &kv_v[tb * 8][d_base], KV_STRIDE);
                simdgroup_load(vm1, &kv_v[tb * 8][d_base + 8], KV_STRIDE);
                simdgroup_multiply_accumulate(O0, pm, vm0, O0);
                simdgroup_multiply_accumulate(O1, pm, vm1, O1);
            }
        }
    }
}

const int gr = r0 + (my_qr / GQA);
const int gh = (int)hk * GQA + (my_qr % GQA);
if (gr < M) {
    const int d_base = (int)sg * 16;
#if SINGLE_CHUNK
    const float invl = 1.0f / l_cur;
    device T* op = o + ((size_t)gr * Hq + gh) * D + d_base + my_c0;
    op[0] = static_cast<T>(O0.thread_elements()[0] * invl);
    op[1] = static_cast<T>(O0.thread_elements()[1] * invl);
    op[8] = static_cast<T>(O1.thread_elements()[0] * invl);
    op[9] = static_cast<T>(O1.thread_elements()[1] * invl);
#else
    const size_t base = ((size_t)gr * Hq + gh) * nchunks + ch;
    if (sg == 0 && my_c0 == 0) {
        pm[base] = m_cur;
        pl[base] = l_cur;
    }
    device float* pap = pa + base * D + d_base + my_c0;
    pap[0] = O0.thread_elements()[0];
    pap[1] = O0.thread_elements()[1];
    pap[8] = O1.thread_elements()[0];
    pap[9] = O1.thread_elements()[1];
#endif
}
