// Gather rows from 512B-coalesced swizzled [V, K] embedding table (zero extra bytes when lm_head is tied).
// Layout w_swiz: [V // 16, K // 64, B8=2, TN=2, cg_sg=8, KS=4, 8].
// Grid: (K / 8, M, 1), Threadgroup: (min(256, K / 8), 1, 1).
const uint k8 = thread_position_in_grid.x;
const uint m  = threadgroup_position_in_grid.y;
if (k8 >= (uint)(K / 8)) return;
const int n = tokens[m];
const int tile_n = n >> 4;
const int cg_sg  = (n >> 1) & 7;
const int c_i    = n & 1;
const int ks_i   = (int)(k8 >> 3);
const int s_i    = (int)((k8 >> 1) & 3);
const int b_i    = (int)(k8 & 1);
const size_t src = ((size_t)tile_n * (K / 64) + ks_i) * 128 + b_i * 64 + c_i * 32 + cg_sg * 4 + s_i;
((device ulong2*)out)[(size_t)m * (K / 8) + k8] = ((const device ulong2*)w_swiz)[src];
