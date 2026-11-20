// Row-invariant attention of a block's rows (engine/models/lfm2_moe/mlx_kernels.py). A threadgroup is one
// (query head h, row r). The row's keys in logical order are the cache's slots 0..L-1, then its path in the
// block (slots L + anc[r][0..d-1], root first): what the same row sees when it is decoded alone, in the same
// order. Thread t takes logical keys t, t + NT, ... (an online softmax of its own), the lanes are combined by
// simd reductions and the simdgroups in order 0..NT/32-1: every step depends on the row's keys alone.
uint h = threadgroup_position_in_grid.x;
uint r = threadgroup_position_in_grid.y;
uint tid = thread_position_in_threadgroup.x;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
const int Hq = q_shape[1];
const int cap = kc_shape[1];
const int A = anc_shape[1];
const int L = lb[0];
const int n = L + dl[r];
const uint hk = h / GQA;
const float scale = rsqrt((float)D);
float qv[D];
for (int i = 0; i < D; i++) qv[i] = float(q[((size_t)r * Hq + h) * D + i]);
float m = -INFINITY, l = 0.0f;
float acc[D];
for (int i = 0; i < D; i++) acc[i] = 0.0f;
for (int j = (int)tid; j < n; j += NT) {
    const int slot = j < L ? j : L + anc[(size_t)r * A + (j - L)];
    const device T* kp = kc + ((size_t)hk * cap + slot) * D;
    const device T* vp = vc + ((size_t)hk * cap + slot) * D;
    float s = 0.0f;
    for (int i = 0; i < D; i++) s = fma(qv[i], float(kp[i]), s);
    s *= scale;
    const float mn = max(m, s);
    const float c = exp(m - mn);
    const float p = exp(s - mn);
    l = l * c + p;
    for (int i = 0; i < D; i++) acc[i] = acc[i] * c + p * float(vp[i]);
    m = mn;
}
threadgroup float tM[NT / 32];
threadgroup float tL[NT / 32];
threadgroup float tA[NT / 32][D];
const float Ms = simd_max(m);
const float f = (m == -INFINITY) ? 0.0f : exp(m - Ms);
const float ls = simd_sum(l * f);
for (int i = 0; i < D; i++) {
    const float a = simd_sum(acc[i] * f);
    if (lane == 0) tA[sg][i] = a;
}
if (lane == 0) { tM[sg] = Ms; tL[sg] = ls; }
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < (uint)D) {
    float Mx = -INFINITY;
    for (int g = 0; g < NT / 32; g++) Mx = max(Mx, tM[g]);
    float L2 = 0.0f, A2 = 0.0f;
    for (int g = 0; g < NT / 32; g++) {
        const float wg = (tM[g] == -INFINITY) ? 0.0f : exp(tM[g] - Mx);
        L2 += tL[g] * wg;
        A2 += tA[g][tid] * wg;
    }
    o[((size_t)r * Hq + h) * D + tid] = static_cast<T>(A2 / L2);
}
