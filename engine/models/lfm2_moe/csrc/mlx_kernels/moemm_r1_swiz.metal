// Dedicated M=1 512B-coalesced swizzled MoE w13+SwiGLU kernel.
// Layout w_swiz: [E, NOUT // 16, K // 64, 2 (w1/w3), B8=2, TN=2, cg_sg=8, KS=4, 8] -> w1 & w3 are 2 KB apart.
// When FUSE_ROUTE == 1, computes exact deterministic top-TOPK in-register (~25ns, zero shared mem, zero barrier)
// directly from (logits, bias), eliminating all 38 standalone moeroute_m1 kernel launches!
// Exact bit-for-bit twin of moemm_hyb_swiz.metal + moeroute.metal.
const int u = (int)threadgroup_position_in_grid.y;
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
float loc_c[E_PER];
for (int i = 0; i < E_PER; i++) {
    const int ek = (int)lane + i * 32;
    if (ek < E) {
        const float lv = float(logits[ek]);
        const float sv = float(static_cast<T>(1.0f / (1.0f + exp(-lv))));
        loc_c[i] = USE_BIAS ? (sv + bias[ek]) : sv;
    } else {
        loc_c[i] = -INFINITY;
    }
}
int top_i[TOPK];
for (int j = 0; j < TOPK; j++) {
    float bc = -INFINITY;
    int be = 1000000;
    for (int i = 0; i < E_PER; i++) {
        const int ek = (int)lane + i * 32;
        if (loc_c[i] > bc || (loc_c[i] == bc && ek < be)) { bc = loc_c[i]; be = ek; }
    }
    const float gmax = simd_max(bc);
    const int win_e = simd_min((bc == gmax) ? be : 1000000);
    top_i[j] = win_e;
    for (int i = 0; i < E_PER; i++) {
        if ((int)lane + i * 32 == win_e) loc_c[i] = -INFINITY;
    }
}
for (int i = 1; i < TOPK; i++) {
    const int ki = top_i[i];
    int j = i - 1;
    while (j >= 0 && top_i[j] > ki) { top_i[j + 1] = top_i[j]; j--; }
    top_i[j + 1] = ki;
}
const int e = top_i[u];
#else
const int e = pexp[u];
#endif

float acc1[2] = {0.0f, 0.0f};
float acc3[2] = {0.0f, 0.0f};

constexpr int KSTEPS = K / 64;
constexpr size_t TILE_U2 = (size_t)KSTEPS * 256;
constexpr size_t EXP_U2 = (size_t)(NOUT / 16) * TILE_U2;
const device ulong2* wtile = (const device ulong2*)w_swiz + (size_t)e * EXP_U2 + (size_t)sg_id * TILE_U2;

for (int ks_i = 0; ks_i < KSTEPS; ks_i++) {
    const int k0 = ks_i * 64 + s * 16;
    const device ulong2* w1p = wtile + ks_i * 256 + lane;
    const device ulong2* w3p = w1p + 128;
    for (int b_i = 0; b_i < 2; b_i++) {
        const int k = k0 + b_i * 8;
        const ulong2 xraw = *((const device ulong2*)(x + k));
        const vec<T, 4> xa = as_type<vec<T, 4>>(xraw.x), xb = as_type<vec<T, 4>>(xraw.y);
        const float x0 = float(xa[0]), x1 = float(xa[1]), x2 = float(xa[2]), x3 = float(xa[3]);
        const float x4 = float(xb[0]), x5 = float(xb[1]), x6 = float(xb[2]), x7 = float(xb[3]);
        const ulong2 r1_0 = w1p[(b_i * 2 + 0) * 32];
        const ulong2 r1_1 = w1p[(b_i * 2 + 1) * 32];
        const ulong2 r3_0 = w3p[(b_i * 2 + 0) * 32];
        const ulong2 r3_1 = w3p[(b_i * 2 + 1) * 32];
        {
            const vec<T, 4> wa = as_type<vec<T, 4>>(r1_0.x), wb = as_type<vec<T, 4>>(r1_0.y);
            float t = acc1[0];
            t = fma(x0, float(wa[0]), t); t = fma(x1, float(wa[1]), t);
            t = fma(x2, float(wa[2]), t); t = fma(x3, float(wa[3]), t);
            t = fma(x4, float(wb[0]), t); t = fma(x5, float(wb[1]), t);
            t = fma(x6, float(wb[2]), t); t = fma(x7, float(wb[3]), t);
            acc1[0] = t;
        }
        {
            const vec<T, 4> wa = as_type<vec<T, 4>>(r1_1.x), wb = as_type<vec<T, 4>>(r1_1.y);
            float t = acc1[1];
            t = fma(x0, float(wa[0]), t); t = fma(x1, float(wa[1]), t);
            t = fma(x2, float(wa[2]), t); t = fma(x3, float(wa[3]), t);
            t = fma(x4, float(wb[0]), t); t = fma(x5, float(wb[1]), t);
            t = fma(x6, float(wb[2]), t); t = fma(x7, float(wb[3]), t);
            acc1[1] = t;
        }
        {
            const vec<T, 4> wa = as_type<vec<T, 4>>(r3_0.x), wb = as_type<vec<T, 4>>(r3_0.y);
            float t = acc3[0];
            t = fma(x0, float(wa[0]), t); t = fma(x1, float(wa[1]), t);
            t = fma(x2, float(wa[2]), t); t = fma(x3, float(wa[3]), t);
            t = fma(x4, float(wb[0]), t); t = fma(x5, float(wb[1]), t);
            t = fma(x6, float(wb[2]), t); t = fma(x7, float(wb[3]), t);
            acc3[0] = t;
        }
        {
            const vec<T, 4> wa = as_type<vec<T, 4>>(r3_1.x), wb = as_type<vec<T, 4>>(r3_1.y);
            float t = acc3[1];
            t = fma(x0, float(wa[0]), t); t = fma(x1, float(wa[1]), t);
            t = fma(x2, float(wa[2]), t); t = fma(x3, float(wa[3]), t);
            t = fma(x4, float(wb[0]), t); t = fma(x5, float(wb[1]), t);
            t = fma(x6, float(wb[2]), t); t = fma(x7, float(wb[3]), t);
            acc3[1] = t;
        }
    }
}

for (int c = 0; c < 2; c++) {
    const float t1 = ((acc1[c] + simd_shuffle_down(acc1[c], 1)) + simd_shuffle_down(acc1[c], 2)) + simd_shuffle_down(acc1[c], 3);
    const float t3 = ((acc3[c] + simd_shuffle_down(acc3[c], 1)) + simd_shuffle_down(acc3[c], 2)) + simd_shuffle_down(acc3[c], 3);
    if (s == 0 && n0 + c < NOUT) {
        const float a = float(static_cast<T>(t1));
        const float b = float(static_cast<T>(t3));
        y[(size_t)u * NOUT + n0 + c] = static_cast<T>((1.0f / (1.0f + exp(-a))) * a * b);
    }
}
