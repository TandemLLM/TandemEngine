// 512B-coalesced swizzled scalar dense projection (M <= 2), bit-for-bit twin of rowmma_swiz.metal.
// Weight layout w_swiz: [N // 16, K // 64, B8=2, TN=2, cg_sg=8, KS=4, 8] (BK=16, KS=4, TN=2 -> N_SG=16).
// Issues 4 upfront 16-byte ulong2 loads (2 KB contiguous per simdgroup) per KSTEP=64.
// Supports optional on-the-fly 32-lane simd_sum RMSNorm (FUSE_RMS) and fused residual add (ADD_RES).
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int M = x_shape[0];
const int m0 = (int)threadgroup_position_in_grid.y * R;
const int s = (int)(lane & 3);
const int cgi_sg = (int)(lane >> 2);
constexpr int N_SG = 16; // (32 / 4) * 2
const int sg_id = (int)threadgroup_position_in_grid.x * (TG / 32) + (int)sg;
const int n0 = sg_id * N_SG + cgi_sg * 2;
if (sg_id * N_SG >= N) return;

#if FUSE_RMS
float inv_r[R];
for (int r = 0; r < R; r++) {
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
    inv_r[r] = rsqrt(simd_sum(ss_loc) / (float)K + eps[0]);
}
#endif

float acc[R][2];
for (int r = 0; r < R; r++) { acc[r][0] = 0.0f; acc[r][1] = 0.0f; }

constexpr int KSTEPS = K / 64;
const device ulong2* wbase = (const device ulong2*)w_swiz + (size_t)sg_id * KSTEPS * 128;

for (int ks_i = 0; ks_i < KSTEPS; ks_i++) {
    const int k0 = ks_i * 64 + s * 16;
    const device ulong2* wp = wbase + ks_i * 128 + lane;
    const ulong2 w00 = wp[0];
    const ulong2 w01 = wp[32];
    const ulong2 w10 = wp[64];
    const ulong2 w11 = wp[96];
#if FUSE_RMS
    const device ulong2* nwp = (const device ulong2*)(nw + k0);
    const ulong2 nr0 = nwp[0], nr1 = nwp[1];
    const vec<T, 4> n0a = as_type<vec<T, 4>>(nr0.x), n0b = as_type<vec<T, 4>>(nr0.y);
    const vec<T, 4> n1a = as_type<vec<T, 4>>(nr1.x), n1b = as_type<vec<T, 4>>(nr1.y);
#endif
    for (int r = 0; r < R; r++) {
        if (m0 + r < M) {
            const device ulong2* xp = (const device ulong2*)(x + (size_t)(m0 + r) * K + k0);
            const ulong2 xr0 = xp[0], xr1 = xp[1];
            const vec<T, 4> x0a = as_type<vec<T, 4>>(xr0.x), x0b = as_type<vec<T, 4>>(xr0.y);
            const vec<T, 4> x1a = as_type<vec<T, 4>>(xr1.x), x1b = as_type<vec<T, 4>>(xr1.y);
#if FUSE_RMS
            const float inv = inv_r[r];
            const float a0 = float(static_cast<T>(float(x0a[0]) * inv * float(n0a[0])));
            const float a1 = float(static_cast<T>(float(x0a[1]) * inv * float(n0a[1])));
            const float a2 = float(static_cast<T>(float(x0a[2]) * inv * float(n0a[2])));
            const float a3 = float(static_cast<T>(float(x0a[3]) * inv * float(n0a[3])));
            const float a4 = float(static_cast<T>(float(x0b[0]) * inv * float(n0b[0])));
            const float a5 = float(static_cast<T>(float(x0b[1]) * inv * float(n0b[1])));
            const float a6 = float(static_cast<T>(float(x0b[2]) * inv * float(n0b[2])));
            const float a7 = float(static_cast<T>(float(x0b[3]) * inv * float(n0b[3])));
            const float b0 = float(static_cast<T>(float(x1a[0]) * inv * float(n1a[0])));
            const float b1 = float(static_cast<T>(float(x1a[1]) * inv * float(n1a[1])));
            const float b2 = float(static_cast<T>(float(x1a[2]) * inv * float(n1a[2])));
            const float b3 = float(static_cast<T>(float(x1a[3]) * inv * float(n1a[3])));
            const float b4 = float(static_cast<T>(float(x1b[0]) * inv * float(n1b[0])));
            const float b5 = float(static_cast<T>(float(x1b[1]) * inv * float(n1b[1])));
            const float b6 = float(static_cast<T>(float(x1b[2]) * inv * float(n1b[2])));
            const float b7 = float(static_cast<T>(float(x1b[3]) * inv * float(n1b[3])));
#else
            const float a0 = float(x0a[0]), a1 = float(x0a[1]), a2 = float(x0a[2]), a3 = float(x0a[3]);
            const float a4 = float(x0b[0]), a5 = float(x0b[1]), a6 = float(x0b[2]), a7 = float(x0b[3]);
            const float b0 = float(x1a[0]), b1 = float(x1a[1]), b2 = float(x1a[2]), b3 = float(x1a[3]);
            const float b4 = float(x1b[0]), b5 = float(x1b[1]), b6 = float(x1b[2]), b7 = float(x1b[3]);
#endif
            float t0 = acc[r][0], t1 = acc[r][1];
            {
                const vec<T, 4> wa = as_type<vec<T, 4>>(w00.x), wb = as_type<vec<T, 4>>(w00.y);
                t0 = fma(a0, float(wa[0]), t0); t0 = fma(a1, float(wa[1]), t0);
                t0 = fma(a2, float(wa[2]), t0); t0 = fma(a3, float(wa[3]), t0);
                t0 = fma(a4, float(wb[0]), t0); t0 = fma(a5, float(wb[1]), t0);
                t0 = fma(a6, float(wb[2]), t0); t0 = fma(a7, float(wb[3]), t0);
            }
            {
                const vec<T, 4> wa = as_type<vec<T, 4>>(w01.x), wb = as_type<vec<T, 4>>(w01.y);
                t1 = fma(a0, float(wa[0]), t1); t1 = fma(a1, float(wa[1]), t1);
                t1 = fma(a2, float(wa[2]), t1); t1 = fma(a3, float(wa[3]), t1);
                t1 = fma(a4, float(wb[0]), t1); t1 = fma(a5, float(wb[1]), t1);
                t1 = fma(a6, float(wb[2]), t1); t1 = fma(a7, float(wb[3]), t1);
            }
            {
                const vec<T, 4> wa = as_type<vec<T, 4>>(w10.x), wb = as_type<vec<T, 4>>(w10.y);
                t0 = fma(b0, float(wa[0]), t0); t0 = fma(b1, float(wa[1]), t0);
                t0 = fma(b2, float(wa[2]), t0); t0 = fma(b3, float(wa[3]), t0);
                t0 = fma(b4, float(wb[0]), t0); t0 = fma(b5, float(wb[1]), t0);
                t0 = fma(b6, float(wb[2]), t0); t0 = fma(b7, float(wb[3]), t0);
            }
            {
                const vec<T, 4> wa = as_type<vec<T, 4>>(w11.x), wb = as_type<vec<T, 4>>(w11.y);
                t1 = fma(b0, float(wa[0]), t1); t1 = fma(b1, float(wa[1]), t1);
                t1 = fma(b2, float(wa[2]), t1); t1 = fma(b3, float(wa[3]), t1);
                t1 = fma(b4, float(wb[0]), t1); t1 = fma(b5, float(wb[1]), t1);
                t1 = fma(b6, float(wb[2]), t1); t1 = fma(b7, float(wb[3]), t1);
            }
            acc[r][0] = t0; acc[r][1] = t1;
        }
    }
}

for (int r = 0; r < R; r++) {
    for (int c = 0; c < 2; c++) {
        float t = acc[r][c];
        t = ((t + simd_shuffle_down(t, 1)) + simd_shuffle_down(t, 2)) + simd_shuffle_down(t, 3);
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
