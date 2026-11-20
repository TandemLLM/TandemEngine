// 512B-coalesced swizzled 8x8 bfloat simdgroup_matrix dense projection (M >= 3),
// bit-for-bit twin of rowchain_swiz.metal on the exact same w_swiz buffer (BK=16, KS=4, FN=2 -> N_TILE=16).
// Each thread loads ONE 16-byte ulong2 from w_swiz per KSTEP=64. Supports FUSE_RMS and ADD_RES.
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = thread_position_in_threadgroup.x;
const int split = (int)sg;
const int M = x_shape[0];
const int m0 = (int)threadgroup_position_in_grid.y * (FM * 8);
const int tile_n = (int)threadgroup_position_in_grid.x;
const int nb = tile_n * 16;

constexpr int KSTEP = 64;
constexpr int NC = 16;
constexpr int MR = FM * 8;
constexpr int STRIDE = KSTEP + 8;
constexpr int KSTEPS = K / KSTEP;

struct StageMem {
    T xs[MR][STRIDE];
    T ws[NC][STRIDE];
};
struct EpiMem {
    float os[4][MR][NC];
};
union ShMem {
    StageMem st;
    EpiMem ep;
};
threadgroup ShMem sm;

#if FUSE_RMS
// Compute exact same 32-lane simd_sum RMSNorm inv per row as rowchain_swiz.metal!
threadgroup float inv_tg[MR];
for (int r = (int)sg; r < MR; r += 4) {
    float ss_loc = 0.0f;
    if (m0 + r < M) {
        const device T* xr = x + (size_t)(m0 + r) * K;
        for (int k = (int)lane * 8; k < K; k += 256) {
            const ulong2 xraw = *((const device ulong2*)(xr + k));
            const vec<T, 4> ha = as_type<vec<T, 4>>(xraw.x), hb = as_type<vec<T, 4>>(xraw.y);
            ss_loc = fma(float(ha[0]), float(ha[0]), ss_loc); ss_loc = fma(float(ha[1]), float(ha[1]), ss_loc);
            ss_loc = fma(float(ha[2]), float(ha[2]), ss_loc); ss_loc = fma(float(ha[3]), float(ha[3]), ss_loc);
            ss_loc = fma(float(hb[0]), float(hb[0]), ss_loc); ss_loc = fma(float(hb[1]), float(hb[1]), ss_loc);
            ss_loc = fma(float(hb[2]), float(hb[2]), ss_loc); ss_loc = fma(float(hb[3]), float(hb[3]), ss_loc);
        }
    }
    const float inv = rsqrt(simd_sum(ss_loc) / (float)K + eps[0]);
    if (lane == 0) inv_tg[r] = inv;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
#endif

simdgroup_matrix<float, 8, 8> acc[FM][2];
for (int fm = 0; fm < FM; fm++) {
    acc[fm][0] = simdgroup_matrix<float, 8, 8>(0.0f);
    acc[fm][1] = simdgroup_matrix<float, 8, 8>(0.0f);
}

const int b_i   = (int)(tid >> 6);
const int c_i   = (int)((tid >> 5) & 1);
const int cg_sg = (int)((tid >> 2) & 7);
const int s_i   = (int)(tid & 3);
const int w_row = cg_sg * 2 + c_i;
const int w_kk  = s_i * 16 + b_i * 8;

const device ulong2* wtile = (const device ulong2*)w_swiz + (size_t)tile_n * KSTEPS * 128;

for (int ks_i = 0; ks_i < KSTEPS; ks_i++) {
    const int k0 = ks_i * KSTEP;
    for (int v = (int)tid; v < MR * 8; v += 128) {
        const int r = v / 8;
        const int kk = (v % 8) * 8;
        if (m0 + r < M) {
#if FUSE_RMS
            const ulong2 xr = *((const device ulong2*)(x + (size_t)(m0 + r) * K + k0 + kk));
            const ulong2 nr = *((const device ulong2*)(nw + k0 + kk));
            const vec<T, 4> xa = as_type<vec<T, 4>>(xr.x), xb = as_type<vec<T, 4>>(xr.y);
            const vec<T, 4> na = as_type<vec<T, 4>>(nr.x), nb4 = as_type<vec<T, 4>>(nr.y);
            const float inv = inv_tg[r];
            vec<T, 4> oa, ob;
            oa[0] = static_cast<T>(float(xa[0]) * inv * float(na[0]));
            oa[1] = static_cast<T>(float(xa[1]) * inv * float(na[1]));
            oa[2] = static_cast<T>(float(xa[2]) * inv * float(na[2]));
            oa[3] = static_cast<T>(float(xa[3]) * inv * float(na[3]));
            ob[0] = static_cast<T>(float(xb[0]) * inv * float(nb4[0]));
            ob[1] = static_cast<T>(float(xb[1]) * inv * float(nb4[1]));
            ob[2] = static_cast<T>(float(xb[2]) * inv * float(nb4[2]));
            ob[3] = static_cast<T>(float(xb[3]) * inv * float(nb4[3]));
            *((threadgroup vec<T, 4>*)&sm.st.xs[r][kk])     = oa;
            *((threadgroup vec<T, 4>*)&sm.st.xs[r][kk + 4]) = ob;
#else
            *((threadgroup ulong2*)&sm.st.xs[r][kk]) = *((const device ulong2*)(x + (size_t)(m0 + r) * K + k0 + kk));
#endif
        } else {
            *((threadgroup ulong2*)&sm.st.xs[r][kk]) = ulong2(0);
        }
    }
    *((threadgroup ulong2*)&sm.st.ws[w_row][w_kk]) = wtile[ks_i * 128 + tid];

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = split * 16; kk < (split + 1) * 16; kk += 8) {
        simdgroup_matrix<T, 8, 8> a[FM];
        for (int fm = 0; fm < FM; fm++) simdgroup_load(a[fm], &sm.st.xs[fm * 8][kk], STRIDE);
        simdgroup_matrix<T, 8, 8> b0, b1;
        simdgroup_load(b0, &sm.st.ws[0][kk], STRIDE, ulong2(0, 0), true);
        simdgroup_load(b1, &sm.st.ws[8][kk], STRIDE, ulong2(0, 0), true);
        for (int fm = 0; fm < FM; fm++) {
            simdgroup_multiply_accumulate(acc[fm][0], a[fm], b0, acc[fm][0]);
            simdgroup_multiply_accumulate(acc[fm][1], a[fm], b1, acc[fm][1]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (int fm = 0; fm < FM; fm++) {
    simdgroup_store(acc[fm][0], &sm.ep.os[split][fm * 8][0], NC);
    simdgroup_store(acc[fm][1], &sm.ep.os[split][fm * 8][8], NC);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int v = (int)tid; v < MR * NC; v += 128) {
    const int r = v / NC, c = v % NC;
    const int n = nb + c;
    if (m0 + r < M && n < N) {
        float s = ((sm.ep.os[0][r][c] + sm.ep.os[1][r][c]) + sm.ep.os[2][r][c]) + sm.ep.os[3][r][c];
        const size_t idx = (size_t)(m0 + r) * N + n;
#if ADD_RES
        y[idx] = static_cast<OT>(float(res[idx]) + float(static_cast<T>(s)));
#else
        y[idx] = static_cast<OT>(static_cast<T>(s));
#endif
    }
}
