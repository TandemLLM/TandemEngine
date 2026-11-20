// Fused MoE weighted expert sum + residual add in ascending expert slot order:
// out[m, n] = h[m, n] + sum_{j=0..TOPK-1} (y[m * TOPK + j, n] * wts[m, j])
const uint tid = thread_position_in_grid.x;
const int m = (int)threadgroup_position_in_grid.y;
const int n4 = (int)tid * 4;
if (n4 >= N) return;
const device vec<T, 4>* hp = (const device vec<T, 4>*)(h + (size_t)m * N + n4);
const vec<T, 4> hv = *hp;
float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
for (int j = 0; j < TOPK; j++) {
    const float wj = float(wts[(size_t)m * TOPK + j]);
    const device vec<T, 4>* yp = (const device vec<T, 4>*)(y + (size_t)(m * TOPK + j) * N + n4);
    const vec<T, 4> yv = *yp;
    if (j == 0) {
        s0 = float(static_cast<T>(float(yv[0]) * wj));
        s1 = float(static_cast<T>(float(yv[1]) * wj));
        s2 = float(static_cast<T>(float(yv[2]) * wj));
        s3 = float(static_cast<T>(float(yv[3]) * wj));
    } else {
        s0 = float(static_cast<T>(s0 + float(static_cast<T>(float(yv[0]) * wj))));
        s1 = float(static_cast<T>(s1 + float(static_cast<T>(float(yv[1]) * wj))));
        s2 = float(static_cast<T>(s2 + float(static_cast<T>(float(yv[2]) * wj))));
        s3 = float(static_cast<T>(s3 + float(static_cast<T>(float(yv[3]) * wj))));
    }
}
vec<T, 4> res;
res[0] = static_cast<T>(float(hv[0]) + s0);
res[1] = static_cast<T>(float(hv[1]) + s1);
res[2] = static_cast<T>(float(hv[2]) + s2);
res[3] = static_cast<T>(float(hv[3]) + s3);
*((device vec<T, 4>*)(out + (size_t)m * N + n4)) = res;
