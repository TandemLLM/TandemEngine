// Row-invariant M>1 512B-coalesced swizzled hybrid MoE projection (scalar for ns<=2, 8x8 bfloat MMA for ns>=3).
// Shares the exact same swizzled weights as moemm_r1_swiz.metal & moemm_w2add_swiz.metal (zero duplication).
// Stages w1 and w3 together in 6.9 KB shared mem (only 2 barriers/KSTEP, reuses A tiles in registers!).
const int u = (int)threadgroup_position_in_grid.y;
const int e = elist[u];
if (e < 0) return;

const int ns = ecnt[u];
const int start = eoff[u];
const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int tile_n = (int)threadgroup_position_in_grid.x;
const int nb = tile_n * 16;
if (nb >= NOUT) return;

constexpr int KSTEP = 64;
constexpr int KSTEPS = K / KSTEP;
constexpr size_t TILE_U2 = (size_t)KSTEPS * (SWIGLU ? 256 : 128);
constexpr size_t EXP_U2 = (size_t)(NOUT / 16) * TILE_U2;
const device ulong2* wtile = (const device ulong2*)w_swiz + (size_t)e * EXP_U2 + (size_t)tile_n * TILE_U2;

if (ns <= 2) {
    if (sg != 0) return;
    const int s = (int)(lane & 3);
    const int cg_sg = (int)(lane >> 2);
    const int n0 = nb + cg_sg * 2;
    const int p0 = sel[start];
    const int p1 = (ns == 2) ? sel[start + 1] : p0;
    const device T* xp0 = x + (size_t)(PROW_IDENTITY ? p0 : (p0 / TOPK)) * K;
    const device T* xp1 = x + (size_t)(PROW_IDENTITY ? p1 : (p1 / TOPK)) * K;

    float a0[2] = {0.0f, 0.0f}, a1[2] = {0.0f, 0.0f};
    float b0[2] = {0.0f, 0.0f}, b1[2] = {0.0f, 0.0f};

    for (int ks_i = 0; ks_i < KSTEPS; ks_i++) {
        const int k0 = ks_i * 64 + s * 16;
        const device ulong2* w1p = wtile + ks_i * (SWIGLU ? 256 : 128) + lane;
        const device ulong2* w3p = w1p + 128;
        for (int b_i = 0; b_i < 2; b_i++) {
            const int k = k0 + b_i * 8;
            const ulong2 x0r = *((const device ulong2*)(xp0 + k));
            const vec<T, 4> x0a = as_type<vec<T, 4>>(x0r.x), x0b = as_type<vec<T, 4>>(x0r.y);
            const float u0 = float(x0a[0]), u1 = float(x0a[1]), u2 = float(x0a[2]), u3 = float(x0a[3]);
            const float u4 = float(x0b[0]), u5 = float(x0b[1]), u6 = float(x0b[2]), u7 = float(x0b[3]);
            float v0=0,v1=0,v2=0,v3=0,v4=0,v5=0,v6=0,v7=0;
            if (ns == 2) {
                const ulong2 x1r = *((const device ulong2*)(xp1 + k));
                const vec<T, 4> x1a = as_type<vec<T, 4>>(x1r.x), x1b = as_type<vec<T, 4>>(x1r.y);
                v0 = float(x1a[0]); v1 = float(x1a[1]); v2 = float(x1a[2]); v3 = float(x1a[3]);
                v4 = float(x1b[0]); v5 = float(x1b[1]); v6 = float(x1b[2]); v7 = float(x1b[3]);
            }
            for (int c = 0; c < 2; c++) {
                const ulong2 r1 = w1p[(b_i * 2 + c) * 32];
                const vec<T, 4> wa = as_type<vec<T, 4>>(r1.x), wb = as_type<vec<T, 4>>(r1.y);
                const float w0 = float(wa[0]), w1 = float(wa[1]), w2 = float(wa[2]), w3 = float(wa[3]);
                const float w4 = float(wb[0]), w5 = float(wb[1]), w6 = float(wb[2]), w7 = float(wb[3]);
                float t0 = a0[c];
                t0 = fma(u0, w0, t0); t0 = fma(u1, w1, t0); t0 = fma(u2, w2, t0); t0 = fma(u3, w3, t0);
                t0 = fma(u4, w4, t0); t0 = fma(u5, w5, t0); t0 = fma(u6, w6, t0); t0 = fma(u7, w7, t0);
                a0[c] = t0;
                if (ns == 2) {
                    float t1 = a1[c];
                    t1 = fma(v0, w0, t1); t1 = fma(v1, w1, t1); t1 = fma(v2, w2, t1); t1 = fma(v3, w3, t1);
                    t1 = fma(v4, w4, t1); t1 = fma(v5, w5, t1); t1 = fma(v6, w6, t1); t1 = fma(v7, w7, t1);
                    a1[c] = t1;
                }
                if (SWIGLU) {
                    const ulong2 r3 = w3p[(b_i * 2 + c) * 32];
                    const vec<T, 4> qa = as_type<vec<T, 4>>(r3.x), qb = as_type<vec<T, 4>>(r3.y);
                    const float q0 = float(qa[0]), q1 = float(qa[1]), q2 = float(qa[2]), q3 = float(qa[3]);
                    const float q4 = float(qb[0]), q5 = float(qb[1]), q6 = float(qb[2]), q7 = float(qb[3]);
                    float p0t = b0[c];
                    p0t = fma(u0, q0, p0t); p0t = fma(u1, q1, p0t); p0t = fma(u2, q2, p0t); p0t = fma(u3, q3, p0t);
                    p0t = fma(u4, q4, p0t); p0t = fma(u5, q5, p0t); p0t = fma(u6, q6, p0t); p0t = fma(u7, q7, p0t);
                    b0[c] = p0t;
                    if (ns == 2) {
                        float p1t = b1[c];
                        p1t = fma(v0, q0, p1t); p1t = fma(v1, q1, p1t); p1t = fma(v2, q2, p1t); p1t = fma(v3, q3, p1t);
                        p1t = fma(v4, q4, p1t); p1t = fma(v5, q5, p1t); p1t = fma(v6, q6, p1t); p1t = fma(v7, q7, p1t);
                        b1[c] = p1t;
                    }
                }
            }
        }
    }
    for (int c = 0; c < 2; c++) {
        const float r0 = ((a0[c] + simd_shuffle_down(a0[c], 1)) + simd_shuffle_down(a0[c], 2)) + simd_shuffle_down(a0[c], 3);
        const float r1 = (ns == 2) ? (((a1[c] + simd_shuffle_down(a1[c], 1)) + simd_shuffle_down(a1[c], 2)) + simd_shuffle_down(a1[c], 3)) : 0.0f;
        if (SWIGLU) {
            const float g0 = ((b0[c] + simd_shuffle_down(b0[c], 1)) + simd_shuffle_down(b0[c], 2)) + simd_shuffle_down(b0[c], 3);
            const float g1 = (ns == 2) ? (((b1[c] + simd_shuffle_down(b1[c], 1)) + simd_shuffle_down(b1[c], 2)) + simd_shuffle_down(b1[c], 3)) : 0.0f;
            if (s == 0 && n0 + c < NOUT) {
                const float fa0 = float(static_cast<T>(r0)), fb0 = float(static_cast<T>(g0));
                y[(size_t)p0 * NOUT + n0 + c] = static_cast<T>((1.0f / (1.0f + exp(-fa0))) * fa0 * fb0);
                if (ns == 2) {
                    const float fa1 = float(static_cast<T>(r1)), fb1 = float(static_cast<T>(g1));
                    y[(size_t)p1 * NOUT + n0 + c] = static_cast<T>((1.0f / (1.0f + exp(-fa1))) * fa1 * fb1);
                }
            }
        } else if (s == 0 && n0 + c < NOUT) {
            y[(size_t)p0 * NOUT + n0 + c] = static_cast<T>(r0);
            if (ns == 2) y[(size_t)p1 * NOUT + n0 + c] = static_cast<T>(r1);
        }
    }
    return;
}

constexpr int STRIDE = KSTEP + 8;
constexpr int RPASS = FM_MAX * 8;
struct StageMem {
    T xs[RPASS][STRIDE];
    T ws1[16][STRIDE];
    T ws3[SWIGLU ? 16 : 1][SWIGLU ? STRIDE : 1];
};
struct EpiMem {
    float os1[4][8][16];
    float os3[SWIGLU ? 4 : 1][SWIGLU ? 8 : 1][SWIGLU ? 16 : 1];
};
union ShMem {
    StageMem st;
    EpiMem ep;
};
threadgroup ShMem sm;

const int split = (int)sg;
const int b_i   = (int)(tid >> 6);
const int c_i   = (int)((tid >> 5) & 1);
const int cg_sg = (int)((tid >> 2) & 7);
const int s_i   = (int)(tid & 3);
const int w_row = cg_sg * 2 + c_i;
const int w_kk  = s_i * 16 + b_i * 8;

for (int c0 = 0; c0 < ns; c0 += RPASS) {
    const int nr = min(RPASS, ns - c0);
    const int fm_max = (FM_MAX == 2 && nr > 8) ? 2 : 1;
    const int mr = fm_max * 8;

    simdgroup_matrix<float, 8, 8> acc1[FM_MAX][2];
    simdgroup_matrix<float, 8, 8> acc3[SWIGLU ? FM_MAX : 1][SWIGLU ? 2 : 1];
    for (int fm = 0; fm < FM_MAX; fm++) {
        for (int f = 0; f < 2; f++) {
            acc1[fm][f] = simdgroup_matrix<float, 8, 8>(0.0f);
            if (SWIGLU) acc3[fm][f] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (int ks_i = 0; ks_i < KSTEPS; ks_i++) {
        const int k0 = ks_i * KSTEP;
        for (int v = (int)tid; v < mr * 8; v += 128) {
            const int r = v / 8;
            const int kk = (v % 8) * 8;
            if (r < nr) {
                const int p = sel[start + c0 + r];
                const int xr_i = PROW_IDENTITY ? p : (p / TOPK);
                *((threadgroup ulong2*)&sm.st.xs[r][kk]) = *((const device ulong2*)(x + (size_t)xr_i * K + k0 + kk));
            } else {
                *((threadgroup ulong2*)&sm.st.xs[r][kk]) = ulong2(0);
            }
        }
        const device ulong2* wk = wtile + ks_i * (SWIGLU ? 256 : 128);
        *((threadgroup ulong2*)&sm.st.ws1[w_row][w_kk]) = wk[tid];
        if (SWIGLU) {
            *((threadgroup ulong2*)&sm.st.ws3[w_row][w_kk]) = wk[128 + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = split * 16; kk < (split + 1) * 16; kk += 8) {
            simdgroup_matrix<T, 8, 8> a0, a1;
            simdgroup_load(a0, &sm.st.xs[0][kk], STRIDE);
            if (FM_MAX == 2 && fm_max == 2) simdgroup_load(a1, &sm.st.xs[8][kk], STRIDE);
            simdgroup_matrix<T, 8, 8> b0, b1;
            simdgroup_load(b0, &sm.st.ws1[0][kk], STRIDE, ulong2(0, 0), true);
            simdgroup_load(b1, &sm.st.ws1[8][kk], STRIDE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc1[0][0], a0, b0, acc1[0][0]);
            simdgroup_multiply_accumulate(acc1[0][1], a0, b1, acc1[0][1]);
            if (FM_MAX == 2 && fm_max == 2) {
                simdgroup_multiply_accumulate(acc1[1][0], a1, b0, acc1[1][0]);
                simdgroup_multiply_accumulate(acc1[1][1], a1, b1, acc1[1][1]);
            }
            if (SWIGLU) {
                simdgroup_load(b0, &sm.st.ws3[0][kk], STRIDE, ulong2(0, 0), true);
                simdgroup_load(b1, &sm.st.ws3[8][kk], STRIDE, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc3[0][0], a0, b0, acc3[0][0]);
                simdgroup_multiply_accumulate(acc3[0][1], a0, b1, acc3[0][1]);
                if (FM_MAX == 2 && fm_max == 2) {
                    simdgroup_multiply_accumulate(acc3[1][0], a1, b0, acc3[1][0]);
                    simdgroup_multiply_accumulate(acc3[1][1], a1, b1, acc3[1][1]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int fm = 0; fm < fm_max; fm++) {
        simdgroup_store(acc1[fm][0], &sm.ep.os1[split][0][0], 16);
        simdgroup_store(acc1[fm][1], &sm.ep.os1[split][0][8], 16);
        if (SWIGLU) {
            simdgroup_store(acc3[fm][0], &sm.ep.os3[split][0][0], 16);
            simdgroup_store(acc3[fm][1], &sm.ep.os3[split][0][8], 16);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int v = (int)tid; v < 128; v += 128) {
            const int lr = v / 16, c = v % 16;
            const int r = fm * 8 + lr;
            const int n = nb + c;
            if (r < nr && n < NOUT) {
                const int p = sel[start + c0 + r];
                float s1 = ((sm.ep.os1[0][lr][c] + sm.ep.os1[1][lr][c]) + sm.ep.os1[2][lr][c]) + sm.ep.os1[3][lr][c];
                if (SWIGLU) {
                    float s3 = ((sm.ep.os3[0][lr][c] + sm.ep.os3[1][lr][c]) + sm.ep.os3[2][lr][c]) + sm.ep.os3[3][lr][c];
                    const float a = float(static_cast<T>(s1)), b = float(static_cast<T>(s3));
                    y[(size_t)p * NOUT + n] = static_cast<T>((1.0f / (1.0f + exp(-a))) * a * b);
                } else {
                    y[(size_t)p * NOUT + n] = static_cast<T>(s1);
                }
            }
        }
        if (fm + 1 < fm_max || c0 + RPASS < ns) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
