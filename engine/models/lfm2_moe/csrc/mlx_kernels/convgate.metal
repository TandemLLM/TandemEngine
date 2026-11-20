// Fused row-invariant double-gated short convolution (no mx.concatenate copy!).
// Computes bx[m, h] = B[m, h] * x[m, h] and cy[m, h] = C[m, h] * sum_{t=0..KT-1} win[m, t, h] * cw[h, t]
// in exact ascending tap order 0..KT-1 (identical for M=1 decode, chain blocks, and arbitrary trees).
// When NEXT_STATE == 1 (M == 1 decode step), writes next_state [S, H] directly, eliminating 30 mx.concatenate calls!
// Grid: (H, M, 1), Threadgroup: (min(256, H), 1, 1).
const int h = (int)thread_position_in_grid.x;
const int m = (int)threadgroup_position_in_grid.y;
if (h >= H) return;
const int S = KT - 1;
const device T* row_m = bcx + (size_t)m * (3 * H);
const float Bm = float(row_m[h]);
const float Cm = float(row_m[H + h]);
const float xm = float(row_m[2 * H + h]);
const T bxm = static_cast<T>(Bm * xm);
#if NEXT_STATE
for (int t = 0; t < S - 1; t++) {
    next_state[(size_t)t * H + h] = state[(size_t)(t + 1) * H + h];
}
next_state[(size_t)(S - 1) * H + h] = bxm;
#else
bx[(size_t)m * H + h] = bxm;
#endif

float acc = 0.0f;
for (int t = 0; t < KT; t++) {
#if HAS_TAPS
    const int pos = taps[(size_t)m * KT + t];
#else
    const int pos = m + t;
#endif
    T val;
    if (pos < S) {
        val = state[(size_t)pos * H + h];
    } else {
        const int rr = pos - S;
        if (rr == m) {
            val = bxm;
        } else {
            const device T* row_r = bcx + (size_t)rr * (3 * H);
            val = static_cast<T>(float(row_r[h]) * float(row_r[2 * H + h]));
        }
    }
    const T prod = static_cast<T>(float(val) * float(cw[(size_t)h * KT + t]));
    acc = (t == 0) ? float(prod) : float(static_cast<T>(acc + float(prod)));
}
cy[(size_t)m * H + h] = static_cast<T>(Cm * acc);
