// High-throughput batched prefill GEMM directly on 512B-coalesced w_swiz (zero unswizzle copy!).
// Layout w_swiz: [N // 16, K // 64, B8=2, TN=2, cg_sg=8, KS=4, 8].
// Uses Metal 4 MPP matmul2d (M5 Neural Accelerators) when __METAL_VERSION__ >= 400, else simdgroup_matrix.
// Tile: BM=64, BN=64, BK=64, TG=128 (4 simdgroups: 2 M-tiles of 32 x 2 N-tiles of 32).
const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const int M = x_shape[0];
// Threadgroups go along M first so concurrent TGs share w_swiz in L2
const int m0 = (int)threadgroup_position_in_grid.x * 64;
const int tile_n0 = (int)threadgroup_position_in_grid.y * 4;
const int n0 = tile_n0 * 16;

constexpr int KSTEPS = K / 64;
constexpr int NTILES_TOTAL = N / 16;

const device ulong2* wbase = (const device ulong2*)w_swiz;
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
threadgroup T xs[64][STRIDE];
threadgroup T ws[64][STRIDE];

const int sg_m = (int)(sg >> 1); // 0..1 (32 rows each)
const int sg_n = (int)(sg & 1);  // 0..1 (32 cols each)

constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    32, 32, 64, false, true, true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

tg_t tA(&xs[sg_m * 32][0], extents_t(64, 32), strides_t{1, STRIDE});
tg_t tB(&ws[sg_n * 32][0], extents_t(64, 32), strides_t{1, STRIDE});
auto Dacc = op.template get_destination_cooperative_tensor<tg_t, tg_t, float>();
for (uint16_t i = 0; i < Dacc.get_capacity(); i++) Dacc[i] = 0.0f;

for (int ks_i = 0; ks_i < KSTEPS; ks_i++) {
    const int k0 = ks_i * 64;
    for (int p = 0; p < 4; p++) {
        const int v = p * 128 + (int)tid;
        const int r = v >> 3;
        const int kk = (v & 7) << 3;
        if (m0 + r < M) {
            *((threadgroup ulong2*)&xs[r][kk]) = *((const device ulong2*)(x + (size_t)(m0 + r) * K + k0 + kk));
        } else {
            *((threadgroup ulong2*)&xs[r][kk]) = ulong2(0);
        }
    }
    for (int st = 0; st < 4; st++) {
        const int tn = tile_n0 + st;
        const int w_row = st * 16 + w_row_base;
        if (tn < NTILES_TOTAL) {
            *((threadgroup ulong2*)&ws[w_row][w_kk]) = wbase[((size_t)tn * KSTEPS + ks_i) * 128 + tid];
        } else {
            *((threadgroup ulong2*)&ws[w_row][w_kk]) = ulong2(0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(tA, tB, Dacc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint16_t i = 0; i < Dacc.get_capacity(); i++) {
    auto idx = Dacc.get_multidimensional_index(i);
    const int r = m0 + sg_m * 32 + idx[1];
    const int c = n0 + sg_n * 32 + idx[0];
    if (r < M && c < N) {
        y[(size_t)r * N + c] = static_cast<OT>(static_cast<T>(Dacc[i]));
    }
}
#else
const uint lane = thread_index_in_simdgroup;
constexpr int STRIDE = 72;
threadgroup T xs[64][STRIDE];
threadgroup T ws[64][STRIDE];

const uint sg_m = sg >> 1;
const uint sg_n = sg & 1;

simdgroup_matrix<float, 8, 8> acc[4][4];
for (int fm = 0; fm < 4; fm++)
    for (int fn = 0; fn < 4; fn++)
        acc[fm][fn] = simdgroup_matrix<float, 8, 8>(0.0f);

for (int ks_i = 0; ks_i < KSTEPS; ks_i++) {
    const int k0 = ks_i * 64;
    for (int p = 0; p < 4; p++) {
        const int v = p * 128 + (int)tid;
        const int r = v >> 3;
        const int kk = (v & 7) << 3;
        if (m0 + r < M) {
            *((threadgroup ulong2*)&xs[r][kk]) = *((const device ulong2*)(x + (size_t)(m0 + r) * K + k0 + kk));
        } else {
            *((threadgroup ulong2*)&xs[r][kk]) = ulong2(0);
        }
    }
    for (int st = 0; st < 4; st++) {
        const int tn = tile_n0 + st;
        const int w_row = st * 16 + w_row_base;
        if (tn < NTILES_TOTAL) {
            *((threadgroup ulong2*)&ws[w_row][w_kk]) = wbase[((size_t)tn * KSTEPS + ks_i) * 128 + tid];
        } else {
            *((threadgroup ulong2*)&ws[w_row][w_kk]) = ulong2(0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < 64; kk += 8) {
        simdgroup_matrix<T, 8, 8> a[4], b[4];
        for (int fm = 0; fm < 4; fm++) simdgroup_load(a[fm], &xs[sg_m * 32 + fm * 8][kk], STRIDE);
        for (int fn = 0; fn < 4; fn++) simdgroup_load(b[fn], &ws[sg_n * 32 + fn * 8][kk], STRIDE, ulong2(0, 0), true);
        for (int fm = 0; fm < 4; fm++)
            for (int fn = 0; fn < 4; fn++)
                simdgroup_multiply_accumulate(acc[fm][fn], a[fm], b[fn], acc[fm][fn]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

const int my_r  = (int)(((lane & 16) >> 2) | ((lane & 6) >> 1));
const int my_c0 = (int)(((lane & 8) >> 1) | ((lane & 1) << 1));
for (int fm = 0; fm < 4; fm++) {
    const int row = m0 + (int)sg_m * 32 + fm * 8 + my_r;
    if (row < M) {
        for (int fn = 0; fn < 4; fn++) {
            const int col = n0 + (int)sg_n * 32 + fn * 8 + my_c0;
            if (col + 1 < N) {
                y[(size_t)row * N + col]     = static_cast<OT>(static_cast<T>(acc[fm][fn].thread_elements()[0]));
                y[(size_t)row * N + col + 1] = static_cast<OT>(static_cast<T>(acc[fm][fn].thread_elements()[1]));
            }
        }
    }
}
#endif
