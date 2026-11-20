// Stage 2 of Split-K pathattn_flash: combines chunks 0..nc-1 in exact logical order for each (row, head).
const uint h = threadgroup_position_in_grid.x;
const uint r = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
const int Hq = pm_shape[1];
const int nchunks = pm_shape[2];
const int n = lb[0] + dl[r];
const int nc = (n + BK - 1) / BK;
const int d0 = (int)lane * 2;
const size_t base = ((size_t)r * Hq + h) * nchunks;

float Mx = -INFINITY;
for (int c = 0; c < nc; c++) Mx = max(Mx, pm[base + c]);
float L2 = 0.0f, A0 = 0.0f, A1 = 0.0f;
for (int c = 0; c < nc; c++) {
    const float mc = pm[base + c];
    const float wg = (mc == -INFINITY) ? 0.0f : exp2(mc - Mx);
    L2 += pl[base + c] * wg;
    A0 += pa[(base + c) * D + d0] * wg;
    A1 += pa[(base + c) * D + d0 + 1] * wg;
}
const float invl = 1.0f / L2;
device T* op = o + ((size_t)r * Hq + h) * D + d0;
op[0] = static_cast<T>(A0 * invl);
op[1] = static_cast<T>(A1 * invl);
