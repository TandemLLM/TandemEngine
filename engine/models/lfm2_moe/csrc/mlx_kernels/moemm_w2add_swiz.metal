// Dedicated M=1 512B-coalesced swizzled MoE w2 + ascending weighted sum + residual add.
// Uses 4-load upfront ulong2 prefetching (2 KB coalesced per simdgroup per KSTEP=64).
// When FUSE_ROUTE == 1, computes exact deterministic top-TOPK + wts in-register from (logits, bias, scale),
// eliminating all 38 standalone moeroute_m1 AND all 38 moewsum launches!
// Exact bit-for-bit twin of moeroute.metal + moemm_hyb_swiz.metal + moewsum.metal.
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int s = (int)(lane & 3);
const int cgi_sg = (int)(lane >> 2);
constexpr int N_SG = 16;
const int sg_id = (int)threadgroup_position_in_grid.x * (TG / 32) + (int)sg;
const int n0 = sg_id * N_SG + cgi_sg * 2;
if (sg_id * N_SG >= NOUT) return;

#if FUSE_ROUTE
constexpr int E_PER = (E + 31) / 32;
float loc_c[E_PER], loc_s[E_PER];
for (int i = 0; i < E_PER; i++) {
    const int ek = (int)lane + i * 32;
    if (ek < E) {
        const float lv = float(logits[ek]);
        const float sv = float(static_cast<T>(1.0f / (1.0f + exp(-lv))));
        loc_s[i] = sv;
        loc_c[i] = USE_BIAS ? (sv + bias[ek]) : sv;
    } else {
        loc_s[i] = 0.0f;
        loc_c[i] = -INFINITY;
    }
}
int top_i[TOPK];
float top_s[TOPK];
for (int j = 0; j < TOPK; j++) {
    float bc = -INFINITY, bs = 0.0f;
    int be = 1000000;
    for (int i = 0; i < E_PER; i++) {
        const int ek = (int)lane + i * 32;
        if (loc_c[i] > bc || (loc_c[i] == bc && ek < be)) { bc = loc_c[i]; bs = loc_s[i]; be = ek; }
    }
    const float gmax = simd_max(bc);
    const int win_e = simd_min((bc == gmax) ? be : 1000000);
    const float win_s = simd_broadcast(bs, (ushort)(win_e & 31));
    top_i[j] = win_e;
    top_s[j] = win_s;
    for (int i = 0; i < E_PER; i++) {
        if ((int)lane + i * 32 == win_e) loc_c[i] = -INFINITY;
    }
}
for (int i = 1; i < TOPK; i++) {
    const int ki = top_i[i];
    const float ks = top_s[i];
    int j = i - 1;
    while (j >= 0 && top_i[j] > ki) { top_i[j + 1] = top_i[j]; top_s[j + 1] = top_s[j]; j--; }
    top_i[j + 1] = ki;
    top_s[j + 1] = ks;
}
float tot = top_s[0];
for (int j = 1; j < TOPK; j++) tot = tot + top_s[j];
const float denom = tot + 1e-6f;
const float sc = scale[0];
float top_w[TOPK];
for (int j = 0; j < TOPK; j++) top_w[j] = float(static_cast<T>((top_s[j] / denom) * sc));
#endif

constexpr int KSTEPS = K / 64;
constexpr size_t TILE_U2 = (size_t)KSTEPS * 128;
constexpr size_t EXP_U2 = (size_t)(NOUT / 16) * TILE_U2;

float out_sum[2] = {0.0f, 0.0f};

for (int j = 0; j < TOPK; j++) {
#if FUSE_ROUTE
    const int e = top_i[j];
    const float wj = top_w[j];
#else
    const int e = pexp[j];
    const float wj = float(wts[j]);
#endif
    const device T* xp = x + (size_t)j * K;
    const device ulong2* wtile = (const device ulong2*)w_swiz + (size_t)e * EXP_U2 + (size_t)sg_id * TILE_U2;
    float t0 = 0.0f, t1 = 0.0f;
    for (int ks_i = 0; ks_i < KSTEPS; ks_i++) {
        const int k0 = ks_i * 64 + s * 16;
        const device ulong2* wp = wtile + ks_i * 128 + lane;
        const ulong2 w00 = wp[0];
        const ulong2 w01 = wp[32];
        const ulong2 w10 = wp[64];
        const ulong2 w11 = wp[96];
        const device ulong2* xpu = (const device ulong2*)(xp + k0);
        const ulong2 xr0 = xpu[0];
        const ulong2 xr1 = xpu[1];
        const vec<T, 4> x0a = as_type<vec<T, 4>>(xr0.x), x0b = as_type<vec<T, 4>>(xr0.y);
        const vec<T, 4> x1a = as_type<vec<T, 4>>(xr1.x), x1b = as_type<vec<T, 4>>(xr1.y);
        {
            const vec<T, 4> wa = as_type<vec<T, 4>>(w00.x), wb = as_type<vec<T, 4>>(w00.y);
            t0 = fma(float(x0a[0]), float(wa[0]), t0); t0 = fma(float(x0a[1]), float(wa[1]), t0);
            t0 = fma(float(x0a[2]), float(wa[2]), t0); t0 = fma(float(x0a[3]), float(wa[3]), t0);
            t0 = fma(float(x0b[0]), float(wb[0]), t0); t0 = fma(float(x0b[1]), float(wb[1]), t0);
            t0 = fma(float(x0b[2]), float(wb[2]), t0); t0 = fma(float(x0b[3]), float(wb[3]), t0);
        }
        {
            const vec<T, 4> wa = as_type<vec<T, 4>>(w01.x), wb = as_type<vec<T, 4>>(w01.y);
            t1 = fma(float(x0a[0]), float(wa[0]), t1); t1 = fma(float(x0a[1]), float(wa[1]), t1);
            t1 = fma(float(x0a[2]), float(wa[2]), t1); t1 = fma(float(x0a[3]), float(wa[3]), t1);
            t1 = fma(float(x0b[0]), float(wb[0]), t1); t1 = fma(float(x0b[1]), float(wb[1]), t1);
            t1 = fma(float(x0b[2]), float(wb[2]), t1); t1 = fma(float(x0b[3]), float(wb[3]), t1);
        }
        {
            const vec<T, 4> wa = as_type<vec<T, 4>>(w10.x), wb = as_type<vec<T, 4>>(w10.y);
            t0 = fma(float(x1a[0]), float(wa[0]), t0); t0 = fma(float(x1a[1]), float(wa[1]), t0);
            t0 = fma(float(x1a[2]), float(wa[2]), t0); t0 = fma(float(x1a[3]), float(wa[3]), t0);
            t0 = fma(float(x1b[0]), float(wb[0]), t0); t0 = fma(float(x1b[1]), float(wb[1]), t0);
            t0 = fma(float(x1b[2]), float(wb[2]), t0); t0 = fma(float(x1b[3]), float(wb[3]), t0);
        }
        {
            const vec<T, 4> wa = as_type<vec<T, 4>>(w11.x), wb = as_type<vec<T, 4>>(w11.y);
            t1 = fma(float(x1a[0]), float(wa[0]), t1); t1 = fma(float(x1a[1]), float(wa[1]), t1);
            t1 = fma(float(x1a[2]), float(wa[2]), t1); t1 = fma(float(x1a[3]), float(wa[3]), t1);
            t1 = fma(float(x1b[0]), float(wb[0]), t1); t1 = fma(float(x1b[1]), float(wb[1]), t1);
            t1 = fma(float(x1b[2]), float(wb[2]), t1); t1 = fma(float(x1b[3]), float(wb[3]), t1);
        }
    }
    const float r0 = ((t0 + simd_shuffle_down(t0, 1)) + simd_shuffle_down(t0, 2)) + simd_shuffle_down(t0, 3);
    const float r1 = ((t1 + simd_shuffle_down(t1, 1)) + simd_shuffle_down(t1, 2)) + simd_shuffle_down(t1, 3);
    if (s == 0 && n0 < NOUT) {
        const float term0 = float(static_cast<T>(float(static_cast<T>(r0)) * wj));
        out_sum[0] = (j == 0) ? term0 : float(static_cast<T>(out_sum[0] + term0));
    }
    if (s == 0 && n0 + 1 < NOUT) {
        const float term1 = float(static_cast<T>(float(static_cast<T>(r1)) * wj));
        out_sum[1] = (j == 0) ? term1 : float(static_cast<T>(out_sum[1] + term1));
    }
}
if (s == 0) {
    if (n0 < NOUT)     out[n0]     = static_cast<T>(float(h[n0]) + out_sum[0]);
    if (n0 + 1 < NOUT) out[n0 + 1] = static_cast<T>(float(h[n0 + 1]) + out_sum[1]);
}
