// Fused Q/K per-head RMSNorm + RoPE + K/V transpose for attention layers.
// Inputs: qkv [M, (Hq + 2*Hk) * D], qn [D], kn [D], cos [M, D], sin [M, D], eps [1]
// Outputs: q_out [M, Hq, D], k_out [Hk, M, D], v_out [Hk, M, D]
// Grid: ((Hq + 2*Hk) * 32, M, 1), Threadgroup: (32, 1, 1). Works for any even D <= 64 (each lane handles d and d + D/2).
const int blk = (int)threadgroup_position_in_grid.x;
const int m = (int)threadgroup_position_in_grid.y;
const int lane = (int)thread_index_in_simdgroup;
const int M = qkv_shape[0];
const int halfD = D / 2;
const int qkv_stride = (Hq + 2 * Hk) * D;
const device T* row = qkv + (size_t)m * qkv_stride;

if (blk < Hq) {
    // Query head h = blk: RMSNorm + RoPE -> q_out[m, h, d]
    const device T* qp = row + (size_t)blk * D;
    float x0 = 0.0f, x1 = 0.0f;
    if (lane < halfD) {
        x0 = float(qp[lane]);
        x1 = float(qp[lane + halfD]);
    }
    const float ss = simd_sum(x0 * x0 + x1 * x1);
    const float inv = rsqrt(ss / (float)D + eps[0]);
    if (lane < halfD) {
        const float n0 = float(static_cast<T>(x0 * inv * float(qn[lane])));
        const float n1 = float(static_cast<T>(x1 * inv * float(qn[lane + halfD])));
        const float c0 = float(cos[(size_t)m * D + lane]);
        const float s0 = float(sin[(size_t)m * D + lane]);
        const float c1 = float(cos[(size_t)m * D + lane + halfD]);
        const float s1 = float(sin[(size_t)m * D + lane + halfD]);
        device T* op = q_out + ((size_t)m * Hq + blk) * D;
        op[lane]         = static_cast<T>(float(static_cast<T>(n0 * c0)) + float(static_cast<T>((-n1) * s0)));
        op[lane + halfD] = static_cast<T>(float(static_cast<T>(n1 * c1)) + float(static_cast<T>(n0 * s1)));
    }
} else if (blk < Hq + Hk) {
    // Key head hk = blk - Hq: RMSNorm + RoPE + transpose -> k_out[hk, m, d]
    const int hk = blk - Hq;
    const device T* kp = row + (size_t)(Hq + hk) * D;
    float x0 = 0.0f, x1 = 0.0f;
    if (lane < halfD) {
        x0 = float(kp[lane]);
        x1 = float(kp[lane + halfD]);
    }
    const float ss = simd_sum(x0 * x0 + x1 * x1);
    const float inv = rsqrt(ss / (float)D + eps[0]);
    if (lane < halfD) {
        const float n0 = float(static_cast<T>(x0 * inv * float(kn[lane])));
        const float n1 = float(static_cast<T>(x1 * inv * float(kn[lane + halfD])));
        const float c0 = float(cos[(size_t)m * D + lane]);
        const float s0 = float(sin[(size_t)m * D + lane]);
        const float c1 = float(cos[(size_t)m * D + lane + halfD]);
        const float s1 = float(sin[(size_t)m * D + lane + halfD]);
        device T* op = k_out + ((size_t)hk * M + m) * D;
        op[lane]         = static_cast<T>(float(static_cast<T>(n0 * c0)) + float(static_cast<T>((-n1) * s0)));
        op[lane + halfD] = static_cast<T>(float(static_cast<T>(n1 * c1)) + float(static_cast<T>(n0 * s1)));
    }
} else {
    // Value head hk = blk - Hq - Hk: transpose -> v_out[hk, m, d]
    const int hk = blk - Hq - Hk;
    if (lane < halfD) {
        const device T* vp = row + (size_t)(Hq + Hk + hk) * D;
        device T* op = v_out + ((size_t)hk * M + m) * D;
        op[lane]         = vp[lane];
        op[lane + halfD] = vp[lane + halfD];
    }
}
