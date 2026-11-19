
// ------------------------------------------------------------------- Hy4-flash: the value half, packed
// The attention's value half is a slice of kv_b_proj's output columns a head: head h owns columns
// [h per + off, h per + off + V) and reads its own input row (the head's weighted latent sum). One CTA a
// (128-column block of the head's slice, row and head): dense_body's arithmetic on that block with the head's
// input (the rotation with suh, RowGemm over all of K, the warp partials added in warp order, the output
// rotation and svh), and only the head's columns are written. A block two heads share is computed once for
// each, with each one's input. grid (NB, M heads): block j of the head's slice; a head with fewer blocks
// leaves the last CTA idle. A row's bytes do not depend on the rows beside it.
template <int KB, int ST, int WARPS>
__global__ void __launch_bounds__(32 * WARPS) hy4_vdense_kernel(const __nv_bfloat16* __restrict__ X, int ldx,
        const uint32_t* __restrict__ T, int ldt, const __half* __restrict__ SUH, const __half* __restrict__ SVH,
        __nv_bfloat16* __restrict__ Y, int ldy, int heads, int Kin, int per, int off, int V) {
    extern __shared__ uint32_t smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int mh = blockIdx.y, h = mh % heads;
    const int c0 = h * per + off;
    const int cb = c0 / 128 + (int)blockIdx.x;
    if (cb * 128 >= c0 + V) { pdl_wait(); pdl_trigger(); return; }
    __half* xh = reinterpret_cast<__half*>(smem);                          // [Kin]
    float* red = reinterpret_cast<float*>(smem + Kin / 2);                 // [WARPS][128]
    RowGemm<KB, ST, WARPS> rg(T, ldt, cb, 0, Kin / 16, smem + Kin / 2 + WARPS * 128);
    rg.issue_first();                                  // the weights do not depend on the predecessor
    pdl_wait();
    pdl_trigger();
    for (int blk = warp; blk < Kin / 128; blk += WARPS) {
        const int k = blk * 128 + 4 * lane;
        const uint2 xr = *reinterpret_cast<const uint2*>(X + (size_t)mh * ldx + k);
        const uint2 sr = *reinterpret_cast<const uint2*>(SUH + k);
        float v0 = __uint_as_float(xr.x << 16), v1 = __uint_as_float(xr.x & 0xffff0000u);
        float v2 = __uint_as_float(xr.y << 16), v3 = __uint_as_float(xr.y & 0xffff0000u);
        const __half2 s01 = *reinterpret_cast<const __half2*>(&sr.x), s23 = *reinterpret_cast<const __half2*>(&sr.y);
        v0 *= __low2float(s01); v1 *= __high2float(s01); v2 *= __low2float(s23); v3 *= __high2float(s23);
        wht128(v0, v1, v2, v3, lane);
        __half2 o01 = __floats2half2_rn(v0 * RSQRT128, v1 * RSQRT128);
        __half2 o23 = __floats2half2_rn(v2 * RSQRT128, v3 * RSQRT128);
        uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(xh + k) = o;
    }
    __syncthreads();
    rg.run(xh, red);
    if (warp == 0) {
        float v0, v1, v2, v3;
        red_rot<WARPS>(red, lane, v0, v1, v2, v3);
        const int n = cb * 128 + 4 * lane, col = n - c0;
        if (col >= 0 && col < V) {
            const uint2 sv = *reinterpret_cast<const uint2*>(SVH + n);
            const __half2 s01 = *reinterpret_cast<const __half2*>(&sv.x), s23 = *reinterpret_cast<const __half2*>(&sv.y);
            v0 = v0 * RSQRT128 * __low2float(s01);  v1 = v1 * RSQRT128 * __high2float(s01);
            v2 = v2 * RSQRT128 * __low2float(s23);  v3 = v3 * RSQRT128 * __high2float(s23);
            __nv_bfloat162 o01 = __floats2bfloat162_rn(v0, v1), o23 = __floats2bfloat162_rn(v2, v3);
            uint2 o; o.x = *reinterpret_cast<uint32_t*>(&o01); o.y = *reinterpret_cast<uint32_t*>(&o23);
            *reinterpret_cast<uint2*>(Y + (size_t)mh * ldy + col) = o;
        }
    }
}

template <int KB>
static void hy4_vdense_launch(const torch::Tensor& x, const torch::Tensor& t, const torch::Tensor& suh,
                              const torch::Tensor& svh, torch::Tensor& y, int heads, int per, int off, cudaStream_t st) {
    constexpr int ST = 3, WARPS = 8, TW = tile_u32(KB);
    const int MH = x.size(0), Kin = x.size(1), V = y.size(1);
    const size_t smem = (size_t)(Kin / 2 + WARPS * 128) * 4 + (size_t)WARPS * ST * 8 * TW * 4;
    auto kern = hy4_vdense_kernel<KB, ST, WARPS>;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        attr = true;
    }
    launch(kern, dim3((V + 127) / 128 + 1, MH), dim3(32 * WARPS), smem, st,
        (const __nv_bfloat16*)x.data_ptr(), (int)x.stride(0), (const uint32_t*)t.data_ptr(),
        (int)(t.stride(0) / t.stride(1)), (const __half*)suh.data_ptr(), (const __half*)svh.data_ptr(),
        (__nv_bfloat16*)y.data_ptr(), (int)y.stride(0), heads, Kin, per, off, V);
}

// x [M heads, Kin] bf16 (row m, head h at m heads + h) through head h's V columns of the packed projection
// -> y [M heads, V] bf16
void hy4_vdense(torch::Tensor x, torch::Tensor t, torch::Tensor suh, torch::Tensor svh, torch::Tensor y,
                int64_t heads, int64_t per, int64_t off) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 4 == 0, "x bf16 [M heads, Kin]");
    TORCH_CHECK(t.dtype() == torch::kInt16 && t.dim() == 3 && t.stride(2) == 1 && t.stride(1) == t.size(2) && t.size(2) % 16 == 0,
                "trellis [Kin/16, N/16, 16 K] with dense tiles at a whole rate");
    const int kb = (int)(t.size(2) / 16);
    const int Kin = x.size(1), N = t.size(1) * 16, V = y.size(1);
    TORCH_CHECK(t.size(0) * 16 == Kin && N % 128 == 0 && Kin % 128 == 0, "shapes");
    TORCH_CHECK(suh.dtype() == torch::kHalf && suh.is_contiguous() && suh.numel() == Kin, "suh fp16 [Kin]");
    TORCH_CHECK(svh.dtype() == torch::kHalf && svh.numel() == N, "svh");
    TORCH_CHECK(heads >= 1 && x.size(0) % heads == 0 && heads * per == N && off >= 0 && off + V <= per, "heads of `per` columns, V from `off`");
    TORCH_CHECK(per % 4 == 0 && off % 4 == 0 && V % 4 == 0, "a head's slice starts and ends on four columns");
    TORCH_CHECK(y.dtype() == torch::kBFloat16 && y.dim() == 2 && y.size(0) == x.size(0) && y.stride(1) == 1 && y.stride(0) % 4 == 0, "y bf16 [M heads, V]");
    switch (kb) {
        case 2: hy4_vdense_launch<2>(x, t, suh, svh, y, heads, per, off, st); break;
        case 3: hy4_vdense_launch<3>(x, t, suh, svh, y, heads, per, off, st); break;
        case 4: hy4_vdense_launch<4>(x, t, suh, svh, y, heads, per, off, st); break;
        case 5: hy4_vdense_launch<5>(x, t, suh, svh, y, heads, per, off, st); break;
        case 6: hy4_vdense_launch<6>(x, t, suh, svh, y, heads, per, off, st); break;
        case 7: hy4_vdense_launch<7>(x, t, suh, svh, y, heads, per, off, st); break;
        case 8: hy4_vdense_launch<8>(x, t, suh, svh, y, heads, per, off, st); break;
        default: TORCH_CHECK(false, "hy4_vdense: K 2..8");
    }
}
