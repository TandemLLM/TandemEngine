// High-throughput batched prefill MoE GEMM directly on 512B-coalesced w13_swiz / w2_swiz (zero unswizzle copy!).
// Uses flat (tile_e, tile_off, tile_nr) work-queue along grid.x so same-expert M-tiles run concurrently and
// share weights in L2 with zero hot-expert stragglers. Uses Metal 4 MPP matmul2d (M5 NAX) when available.
// Fuses w1 + w3 + SwiGLU when SWIGLU == 1, scattering directly to y[sel[r], n] with zero reorder copy.
const int t_idx = (int)threadgroup_position_in_grid.x;
const int e = tile_e[t_idx];
if (e < 0) return;

const int start = tile_off[t_idx];
const int nr = tile_nr[t_idx];
const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const int tile_n0 = (int)threadgroup_position_in_grid.y * (BN / 16);
const int n0 = tile_n0 * 16;

constexpr int KSTEPS = K / 64;
constexpr int NTILES_TOTAL = NOUT / 16;
constexpr size_t TILE_U2 = (size_t)KSTEPS * (SWIGLU ? 256 : 128);
constexpr size_t EXP_U2  = (size_t)NTILES_TOTAL * TILE_U2;

const device ulong2* wexp = (const device ulong2*)w_swiz + (size_t)e * EXP_U2;
const int b_i   = (int)(tid >> 6);
const int c_i   = (int)((tid >> 5) & 1);
const int cg_sg = (int)((tid >> 2) & 7);
const int s_i   = (int)(tid & 3);
const int w_row_base = cg_sg * 2 + c_i;
const int w_kk  = s_i * 16 + b_i * 8;

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
using extents_t = metal::dextents<int32_t, 2>;
using strides_t = metal::array<int32_t, 2>;
using tg_t = metal::tensor<threadgroup T, extents_t, metal::tensor_inline>;

constexpr int STRIDE = 64;
constexpr int TSM = BM / 2;
constexpr int TSN = BN / 2;
threadgroup T xs[BM][STRIDE];
threadgroup T ws1[BN][STRIDE];
threadgroup T ws3[SWIGLU ? BN : 1][SWIGLU ? STRIDE : 1];

const int sg_m = (int)(sg >> 1);
const int sg_n = (int)(sg & 1);

constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    TSM, TSN, 64, false, true, true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

tg_t tA(&xs[sg_m * TSM][0], extents_t(64, TSM), strides_t{1, STRIDE});
tg_t tB1(&ws1[sg_n * TSN][0], extents_t(64, TSN), strides_t{1, STRIDE});
tg_t tB3(&ws3[SWIGLU ? sg_n * TSN : 0][0], extents_t(64, TSN), strides_t{1, STRIDE});

auto Areg = op.template get_left_input_cooperative_tensor<T, T, float>();
auto Dacc1 = op.template get_destination_cooperative_tensor<tg_t, tg_t, float>();
auto Dacc3 = op.template get_destination_cooperative_tensor<tg_t, tg_t, float>();
for (uint16_t i = 0; i < Dacc1.get_capacity(); i++) {
    Dacc1[i] = 0.0f;
    if (SWIGLU) Dacc3[i] = 0.0f;
}

for (int ks_i = 0; ks_i < KSTEPS; ks_i++) {
    const int k0 = ks_i * 64;
    for (int v = (int)tid; v < BM * 8; v += 128) {
        const int r = v >> 3;
        const int kk = (v & 7) << 3;
        if (r < nr) {
            const int p = sel[start + r];
            const int xr_i = PROW_IDENTITY ? p : (p / TOPK);
            *((threadgroup ulong2*)&xs[r][kk]) = *((const device ulong2*)(x + (size_t)xr_i * K + k0 + kk));
        } else {
            *((threadgroup ulong2*)&xs[r][kk]) = ulong2(0);
        }
    }
    for (int st = 0; st < BN / 16; st++) {
        const int tn = tile_n0 + st;
        const int w_row = st * 16 + w_row_base;
        if (tn < NTILES_TOTAL) {
            const device ulong2* wk = wexp + (size_t)tn * TILE_U2 + (size_t)ks_i * (SWIGLU ? 256 : 128);
            *((threadgroup ulong2*)&ws1[w_row][w_kk]) = wk[tid];
            if (SWIGLU) *((threadgroup ulong2*)&ws3[w_row][w_kk]) = wk[128 + tid];
        } else {
            *((threadgroup ulong2*)&ws1[w_row][w_kk]) = ulong2(0);
            if (SWIGLU) *((threadgroup ulong2*)&ws3[w_row][w_kk]) = ulong2(0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (SWIGLU) {
        Areg.load(tA);
        op.run(Areg, tB1, Dacc1);
        op.run(Areg, tB3, Dacc3);
    } else {
        op.run(tA, tB1, Dacc1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint16_t i = 0; i < Dacc1.get_capacity(); i++) {
    auto idx = Dacc1.get_multidimensional_index(i);
    const int r = sg_m * TSM + idx[1];
    const int c = n0 + sg_n * TSN + idx[0];
    if (r < nr && c < NOUT) {
        const int p = sel[start + r];
        if (SWIGLU) {
            const float a = float(static_cast<T>(Dacc1[i]));
            const float b = float(static_cast<T>(Dacc3[i]));
            y[(size_t)p * NOUT + c] = static_cast<T>((1.0f / (1.0f + exp(-a))) * a * b);
        } else {
            y[(size_t)p * NOUT + c] = static_cast<T>(Dacc1[i]);
        }
    }
}
#else
const uint lane = thread_index_in_simdgroup;
constexpr int STRIDE = 72;
constexpr int FM = BM / 16;
constexpr int FN = BN / 16;
threadgroup T xs[BM][STRIDE];
threadgroup T ws1[BN][STRIDE];
threadgroup T ws3[SWIGLU ? BN : 1][SWIGLU ? STRIDE : 1];

const uint sg_m = sg >> 1;
const uint sg_n = sg & 1;

simdgroup_matrix<float, 8, 8> acc1[FM][FN];
simdgroup_matrix<float, 8, 8> acc3[SWIGLU ? FM : 1][SWIGLU ? FN : 1];
for (int fm = 0; fm < FM; fm++) {
    for (int fn = 0; fn < FN; fn++) {
        acc1[fm][fn] = simdgroup_matrix<float, 8, 8>(0.0f);
        if (SWIGLU) acc3[fm][fn] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
}

for (int ks_i = 0; ks_i < KSTEPS; ks_i++) {
    const int k0 = ks_i * 64;
    for (int v = (int)tid; v < BM * 8; v += 128) {
        const int r = v >> 3;
        const int kk = (v & 7) << 3;
        if (r < nr) {
            const int p = sel[start + r];
            const int xr_i = PROW_IDENTITY ? p : (p / TOPK);
            *((threadgroup ulong2*)&xs[r][kk]) = *((const device ulong2*)(x + (size_t)xr_i * K + k0 + kk));
        } else {
            *((threadgroup ulong2*)&xs[r][kk]) = ulong2(0);
        }
    }
    for (int st = 0; st < BN / 16; st++) {
        const int tn = tile_n0 + st;
        const int w_row = st * 16 + w_row_base;
        if (tn < NTILES_TOTAL) {
            const device ulong2* wk = wexp + (size_t)tn * TILE_U2 + (size_t)ks_i * (SWIGLU ? 256 : 128);
            *((threadgroup ulong2*)&ws1[w_row][w_kk]) = wk[tid];
            if (SWIGLU) *((threadgroup ulong2*)&ws3[w_row][w_kk]) = wk[128 + tid];
        } else {
            *((threadgroup ulong2*)&ws1[w_row][w_kk]) = ulong2(0);
            if (SWIGLU) *((threadgroup ulong2*)&ws3[w_row][w_kk]) = ulong2(0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < 64; kk += 8) {
        simdgroup_matrix<T, 8, 8> a[FM], b[FN];
        for (int fm = 0; fm < FM; fm++) simdgroup_load(a[fm], &xs[sg_m * (BM / 2) + fm * 8][kk], STRIDE);
        for (int fn = 0; fn < FN; fn++) simdgroup_load(b[fn], &ws1[sg_n * (BN / 2) + fn * 8][kk], STRIDE, ulong2(0, 0), true);
        for (int fm = 0; fm < FM; fm++)
            for (int fn = 0; fn < FN; fn++)
                simdgroup_multiply_accumulate(acc1[fm][fn], a[fm], b[fn], acc1[fm][fn]);
        if (SWIGLU) {
            for (int fn = 0; fn < FN; fn++) simdgroup_load(b[fn], &ws3[sg_n * (BN / 2) + fn * 8][kk], STRIDE, ulong2(0, 0), true);
            for (int fm = 0; fm < FM; fm++)
                for (int fn = 0; fn < FN; fn++)
                    simdgroup_multiply_accumulate(acc3[fm][fn], a[fm], b[fn], acc3[fm][fn]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

const int my_r  = (int)(((lane & 16) >> 2) | ((lane & 6) >> 1));
const int my_c0 = (int)(((lane & 8) >> 1) | ((lane & 1) << 1));
for (int fm = 0; fm < FM; fm++) {
    const int r = (int)sg_m * (BM / 2) + fm * 8 + my_r;
    if (r < nr) {
        const int p = sel[start + r];
        for (int fn = 0; fn < FN; fn++) {
            const int col = n0 + (int)sg_n * (BN / 2) + fn * 8 + my_c0;
            if (col + 1 < NOUT) {
                device T* yp = y + (size_t)p * NOUT + col;
                if (SWIGLU) {
                    const float a0 = float(static_cast<T>(acc1[fm][fn].thread_elements()[0]));
                    const float b0 = float(static_cast<T>(acc3[fm][fn].thread_elements()[0]));
                    const float a1 = float(static_cast<T>(acc1[fm][fn].thread_elements()[1]));
                    const float b1 = float(static_cast<T>(acc3[fm][fn].thread_elements()[1]));
                    yp[0] = static_cast<T>((1.0f / (1.0f + exp(-a0))) * a0 * b0);
                    yp[1] = static_cast<T>((1.0f / (1.0f + exp(-a1))) * a1 * b1);
                } else {
                    yp[0] = static_cast<T>(acc1[fm][fn].thread_elements()[0]);
                    yp[1] = static_cast<T>(acc1[fm][fn].thread_elements()[1]);
                }
            }
        }
    }
}
#endif
