// Row-invariant dense projection on the 8x8 matrix units (engine/models/lfm2_moe/mlx_kernels.py `dense`).
// Its scalar twin for few rows is rowchain.metal: the same sums, bit for bit.
// y[m, n] = sum_k x[m, k] w[n, k] (+ res[m, n] when ADD_RES == 1).
// Uses simdgroup_matrix<T, 8, 8> (bfloat inputs -> float accumulator, measured bit-for-bit identical to
// rowchain on M5 Max), coalesced 16-byte bfloat staging into union-aliased threadgroup memory, and FM 8-row
// tiles per threadgroup (FM=1 for M<=8, FM=2 for M>8 so weights are read once for up to 16 rows).
const uint sg = simdgroup_index_in_threadgroup;
const uint tid = thread_position_in_threadgroup.x;
const int CG = 4 / KS;
const int split = (int)sg % KS;
const int cg = (int)sg / KS;
const int M = x_shape[0];
const int m0 = (int)threadgroup_position_in_grid.y * (FM * 8);
const int nb = (int)threadgroup_position_in_grid.x * (CG * FN * 8);

constexpr int KSTEP = KS * BK;
constexpr int NC = CG * FN * 8;
constexpr int MR = FM * 8;
constexpr int STRIDE = KSTEP + 8;
constexpr int KVECS = KSTEP / 8;

struct StageMem {
    T xs[MR][STRIDE];
    T ws[NC][STRIDE];
};
struct EpiMem {
    float os[KS][MR][NC];
};
union ShMem {
    StageMem st;
    EpiMem ep;
};
threadgroup ShMem sm;

simdgroup_matrix<float, 8, 8> acc[FM][FN];
for (int fm = 0; fm < FM; fm++)
    for (int f = 0; f < FN; f++)
        acc[fm][f] = simdgroup_matrix<float, 8, 8>(0.0f);

for (int k0 = 0; k0 < K; k0 += KSTEP) {
    for (int v = (int)tid; v < MR * KVECS; v += 128) {
        const int r = v / KVECS;
        const int kk = (v % KVECS) * 8;
        if (m0 + r < M) {
            const device vec<T, 4>* xp = (const device vec<T, 4>*)(x + (size_t)(m0 + r) * K + k0 + kk);
            *((threadgroup vec<T, 4>*)&sm.st.xs[r][kk])     = xp[0];
            *((threadgroup vec<T, 4>*)&sm.st.xs[r][kk + 4]) = xp[1];
        } else {
            *((threadgroup vec<T, 4>*)&sm.st.xs[r][kk])     = vec<T, 4>(0);
            *((threadgroup vec<T, 4>*)&sm.st.xs[r][kk + 4]) = vec<T, 4>(0);
        }
    }
    for (int v = (int)tid; v < NC * KVECS; v += 128) {
        const int c = v / KVECS;
        const int kk = (v % KVECS) * 8;
        const int n = nb + c;
        if (n < N) {
            const device vec<T, 4>* wp = (const device vec<T, 4>*)(w + (size_t)n * K + k0 + kk);
            *((threadgroup vec<T, 4>*)&sm.st.ws[c][kk])     = wp[0];
            *((threadgroup vec<T, 4>*)&sm.st.ws[c][kk + 4]) = wp[1];
        } else {
            *((threadgroup vec<T, 4>*)&sm.st.ws[c][kk])     = vec<T, 4>(0);
            *((threadgroup vec<T, 4>*)&sm.st.ws[c][kk + 4]) = vec<T, 4>(0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = split * BK; kk < (split + 1) * BK; kk += 8) {
        simdgroup_matrix<T, 8, 8> a[FM];
        for (int fm = 0; fm < FM; fm++) {
            simdgroup_load(a[fm], &sm.st.xs[fm * 8][kk], STRIDE);
        }
        for (int f = 0; f < FN; f++) {
            simdgroup_matrix<T, 8, 8> b;
            simdgroup_load(b, &sm.st.ws[(cg * FN + f) * 8][kk], STRIDE, ulong2(0, 0), true);
            for (int fm = 0; fm < FM; fm++) {
                simdgroup_multiply_accumulate(acc[fm][f], a[fm], b, acc[fm][f]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (int fm = 0; fm < FM; fm++) {
    for (int f = 0; f < FN; f++) {
        simdgroup_store(acc[fm][f], &sm.ep.os[split][fm * 8][(cg * FN + f) * 8], NC);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int v = (int)tid; v < MR * NC; v += 128) {
    const int r = v / NC;
    const int c = v % NC;
    const int n = nb + c;
    if (m0 + r < M && n < N) {
        float s = sm.ep.os[0][r][c];
        for (int sp = 1; sp < KS; sp++) s += sm.ep.os[sp][r][c];
        const size_t idx = (size_t)(m0 + r) * N + n;
#if ADD_RES
        y[idx] = static_cast<OT>(float(res[idx]) + float(static_cast<T>(s)));
#else
        y[idx] = static_cast<OT>(static_cast<T>(s));
#endif
    }
}
