#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <vector>

using at::Tensor;

namespace {

__device__ inline float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ inline int clampi(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}

__device__ inline int ndc_to_pix_min(float x_ndc, int W) {
    // Map [-1,1] -> [0,W], then floor, clamp to [0, W-1]
    float xf = (x_ndc * 0.5f + 0.5f) * (float)W;
    int xi = (int)floorf(xf);
    return clampi(xi, 0, W - 1);
}

__device__ inline int ndc_to_pix_max(float x_ndc, int W) {
    float xf = (x_ndc * 0.5f + 0.5f) * (float)W;
    int xi = (int)ceilf(xf);
    return clampi(xi, 0, W - 1);
}

__global__ void bin_faces_kernel(
    const float* __restrict__ verts,      // (B,V,3)
    const int64_t* __restrict__ faces,    // (F,3)
    int B, int V, int F,
    int H, int W,
    int tile_size,
    int tiles_x,
    int tiles_y,
    int max_per_tile,
    int32_t* __restrict__ tile_counts,    // (B*T)
    int32_t* __restrict__ tile_faces      // (B*T*max)
) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (b >= B || f >= F) return;

    int64_t i0 = faces[3 * f + 0];
    int64_t i1 = faces[3 * f + 1];
    int64_t i2 = faces[3 * f + 2];

    if (i0 < 0 || i0 >= V || i1 < 0 || i1 >= V || i2 < 0 || i2 >= V) return;

    const float* v0 = verts + (b * V + (int)i0) * 3;
    const float* v1 = verts + (b * V + (int)i1) * 3;
    const float* v2 = verts + (b * V + (int)i2) * 3;

    float x0 = v0[0], y0 = v0[1];
    float x1 = v1[0], y1 = v1[1];
    float x2 = v2[0], y2 = v2[1];

    // NDC bbox
    float minx = fminf(x0, fminf(x1, x2));
    float maxx = fmaxf(x0, fmaxf(x1, x2));
    float miny = fminf(y0, fminf(y1, y2));
    float maxy = fmaxf(y0, fmaxf(y1, y2));

    // quick reject if completely outside NDC
    if (maxx < -1.0f || minx >  1.0f || maxy < -1.0f || miny >  1.0f) return;

    // Convert to pixel bbox
    int x_min = ndc_to_pix_min(minx, W);
    int x_max = ndc_to_pix_max(maxx, W);
    int y_min = ndc_to_pix_min(miny, H);
    int y_max = ndc_to_pix_max(maxy, H);

    // Tile bbox
    int tx0 = x_min / tile_size;
    int tx1 = x_max / tile_size;
    int ty0 = y_min / tile_size;
    int ty1 = y_max / tile_size;

    tx0 = clampi(tx0, 0, tiles_x - 1);
    tx1 = clampi(tx1, 0, tiles_x - 1);
    ty0 = clampi(ty0, 0, tiles_y - 1);
    ty1 = clampi(ty1, 0, tiles_y - 1);

    int T = tiles_x * tiles_y;

    // Insert face into all overlapped tiles
    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            int tile = ty * tiles_x + tx;
            int idx_counts = b * T + tile;

            int write_idx = atomicAdd(&tile_counts[idx_counts], 1);
            if (write_idx < max_per_tile) {
                int out_base = (b * T + tile) * max_per_tile + write_idx;
                tile_faces[out_base] = (int32_t)f;
            }
            // else overflow: we keep counting but drop writes past max_per_tile
        }
    }
}

} // namespace

std::vector<Tensor> build_bins_cuda(
    Tensor verts,   // (B,V,3) float
    Tensor faces,   // (F,3) int64
    int image_size,
    int tile_size,
    int max_per_tile
) {
    TORCH_CHECK(verts.is_cuda(), "verts must be CUDA");
    TORCH_CHECK(faces.is_cuda(), "faces must be CUDA");
    TORCH_CHECK(verts.dim() == 3 && verts.size(2) == 3, "verts must be (B,V,3)");
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces must be (F,3)");
    TORCH_CHECK(verts.scalar_type() == at::kFloat, "verts must be float32");
    TORCH_CHECK(faces.scalar_type() == at::kLong, "faces must be int64");

    verts = verts.contiguous();
    faces = faces.contiguous();

    const int B = (int)verts.size(0);
    const int V = (int)verts.size(1);
    const int F = (int)faces.size(0);

    const int H = image_size;
    const int W = image_size;

    TORCH_CHECK(tile_size > 0, "tile_size must be > 0");
    TORCH_CHECK(max_per_tile > 0, "max_per_tile must be > 0");

    const int tiles_x = (W + tile_size - 1) / tile_size;
    const int tiles_y = (H + tile_size - 1) / tile_size;
    const int T = tiles_x * tiles_y;

    auto opts_i32 = faces.options().dtype(at::kInt);

    Tensor tile_counts = at::zeros({B, T}, opts_i32);                 // int32
    Tensor tile_faces  = at::zeros({B, T, max_per_tile}, opts_i32);   // int32

    dim3 block(256, 1, 1);
    dim3 grid((F + block.x - 1) / block.x, B, 1);

    bin_faces_kernel<<<grid, block>>>(
        verts.data_ptr<float>(),
        faces.data_ptr<int64_t>(),
        B, V, F,
        H, W,
        tile_size,
        tiles_x, tiles_y,
        max_per_tile,
        tile_counts.data_ptr<int32_t>(),
        tile_faces.data_ptr<int32_t>()
    );

    return {tile_counts, tile_faces};
}