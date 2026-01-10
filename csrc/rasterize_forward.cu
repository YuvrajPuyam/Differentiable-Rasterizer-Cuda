// csrc/rasterize_forward.cu (EDITED)

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <vector>

using at::Tensor;

namespace {

__device__ inline int pixel_index(int b, int y, int x, int image_size) {
    return b * image_size * image_size + y * image_size + x;
}

__device__ inline bool soft_barycentric_metric(
    float px, float py,
    float x0, float y0,
    float x1, float y1,
    float x2, float y2,
    float &D_out
) {
    float minx = fminf(x0, fminf(x1, x2)) - 1e-4f;
    float maxx = fmaxf(x0, fmaxf(x1, x2)) + 1e-4f;
    float miny = fminf(y0, fminf(y1, y2)) - 1e-4f;
    float maxy = fmaxf(y0, fmaxf(y1, y2)) + 1e-4f;

    if (px < minx || px > maxx || py < miny || py > maxy) return false;

    float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    if (fabsf(denom) < 1e-8f) return false;
    float inv_denom = 1.0f / denom;

    float a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) * inv_denom;
    float b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) * inv_denom;
    float c = 1.0f - a - b;

    D_out = fminf(a, fminf(b, c));
    return true;
}

__global__ void rasterize_forward_kernel_tiled(
    const float* __restrict__ verts,       // (B,V,3)
    const int64_t* __restrict__ faces,     // (F,3)
    const float* __restrict__ colors,      // (B,V,3)
    const int32_t* __restrict__ tile_counts, // (B*T)
    const int32_t* __restrict__ tile_faces,  // (B*T*max)
    int B, int V, int F,
    int image_size,
    int tile_size,
    int tiles_x,
    int tiles_y,
    int max_per_tile,
    float* __restrict__ out_sil,           // (B*H*W)
    float* __restrict__ out_rgb            // (B*3*H*W)
) {
    int b = blockIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || y >= image_size || x >= image_size) return;

    const int pix = pixel_index(b, y, x, image_size);

    const float px = ((x + 0.5f) / image_size) * 2.0f - 1.0f;
    const float py = ((y + 0.5f) / image_size) * 2.0f - 1.0f;

    float bg_prod = 1.0f;
    float3 accum_rgb = make_float3(0.0f, 0.0f, 0.0f);
    float  w_rgb = 0.0f;

    const float sigma = 1e-2f;

    // Tile id
    int tx = x / tile_size;
    int ty = y / tile_size;
    tx = min(tx, tiles_x - 1);
    ty = min(ty, tiles_y - 1);
    int tile = ty * tiles_x + tx;
    int T = tiles_x * tiles_y;

    int idx_counts = b * T + tile;
    int count = tile_counts[idx_counts];
    if (count > max_per_tile) count = max_per_tile;

    int base_faces = (b * T + tile) * max_per_tile;

    for (int ii = 0; ii < count; ++ii) {
        int f = (int)tile_faces[base_faces + ii];
        if (f < 0 || f >= F) continue;

        int64_t i0 = faces[3 * f + 0];
        int64_t i1 = faces[3 * f + 1];
        int64_t i2 = faces[3 * f + 2];

        const float* v0 = verts + (b * V + (int)i0) * 3;
        const float* v1 = verts + (b * V + (int)i1) * 3;
        const float* v2 = verts + (b * V + (int)i2) * 3;

        float x0 = v0[0], y0 = v0[1], z0 = v0[2];
        float x1 = v1[0], y1 = v1[1], z1 = v1[2];
        float x2 = v2[0], y2 = v2[1], z2 = v2[2];

        float z_min = fminf(z0, fminf(z1, z2));
        float z_max = fmaxf(z0, fmaxf(z1, z2));
        if (z_max < -1.0f || z_min > 1.0f) continue;

        float D;
        if (!soft_barycentric_metric(px, py, x0, y0, x1, y1, x2, y2, D)) continue;

        float alpha = 1.0f / (1.0f + expf(-D / sigma));
        if (alpha <= 1e-6f) continue;

        bg_prod *= (1.0f - alpha);

        const float* c0 = colors + (b * V + (int)i0) * 3;
        const float* c1 = colors + (b * V + (int)i1) * 3;
        const float* c2 = colors + (b * V + (int)i2) * 3;

        float3 cf;
        cf.x = (c0[0] + c1[0] + c2[0]) / 3.0f;
        cf.y = (c0[1] + c1[1] + c2[1]) / 3.0f;
        cf.z = (c0[2] + c1[2] + c2[2]) / 3.0f;

        accum_rgb.x += alpha * cf.x;
        accum_rgb.y += alpha * cf.y;
        accum_rgb.z += alpha * cf.z;
        w_rgb       += alpha;
    }

    float S = 1.0f - bg_prod;
    out_sil[pix] = S;

    int plane_size = image_size * image_size;
    int base = b * 3 * plane_size + y * image_size + x;

    if (S > 1e-6f && w_rgb > 1e-6f) {
        float inv_w = 1.0f / (w_rgb + 1e-6f);
        float s_clamped = fminf(fmaxf(S, 0.0f), 1.0f);

        float3 c;
        c.x = accum_rgb.x * inv_w * s_clamped;
        c.y = accum_rgb.y * inv_w * s_clamped;
        c.z = accum_rgb.z * inv_w * s_clamped;

        out_rgb[base + 0 * plane_size] = c.x;
        out_rgb[base + 1 * plane_size] = c.y;
        out_rgb[base + 2 * plane_size] = c.z;
    } else {
        out_rgb[base + 0 * plane_size] = 0.0f;
        out_rgb[base + 1 * plane_size] = 0.0f;
        out_rgb[base + 2 * plane_size] = 0.0f;
    }
}

} // namespace

std::vector<Tensor> rasterize_forward_cuda(
    Tensor verts,
    Tensor faces,
    Tensor colors,
    Tensor tile_counts,
    Tensor tile_faces,
    int image_size,
    int tile_size,
    int max_per_tile
) {
    TORCH_CHECK(verts.is_cuda(), "verts must be CUDA");
    TORCH_CHECK(faces.is_cuda(), "faces must be CUDA");
    TORCH_CHECK(colors.is_cuda(), "colors must be CUDA");
    TORCH_CHECK(tile_counts.is_cuda(), "tile_counts must be CUDA");
    TORCH_CHECK(tile_faces.is_cuda(), "tile_faces must be CUDA");

    TORCH_CHECK(verts.dim() == 3 && verts.size(2) == 3, "verts must be (B,V,3)");
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces must be (F,3)");
    TORCH_CHECK(colors.sizes() == verts.sizes(), "colors must match verts");
    TORCH_CHECK(tile_counts.scalar_type() == at::kInt, "tile_counts must be int32");
    TORCH_CHECK(tile_faces.scalar_type() == at::kInt, "tile_faces must be int32");

    verts = verts.contiguous();
    faces = faces.contiguous();
    colors = colors.contiguous();
    tile_counts = tile_counts.contiguous();
    tile_faces = tile_faces.contiguous();

    const int B = (int)verts.size(0);
    const int V = (int)verts.size(1);
    const int F = (int)faces.size(0);

    const int H = image_size;
    const int W = image_size;

    const int tiles_x = (W + tile_size - 1) / tile_size;
    const int tiles_y = (H + tile_size - 1) / tile_size;

    auto opts = verts.options().dtype(at::kFloat);

    Tensor sil = at::zeros({B, 1, image_size, image_size}, opts);
    Tensor rgb = at::zeros({B, 3, image_size, image_size}, opts);

    dim3 block(16, 16, 1);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y,
              B);

    rasterize_forward_kernel_tiled<<<grid, block>>>(
        verts.data_ptr<float>(),
        faces.data_ptr<int64_t>(),
        colors.data_ptr<float>(),
        tile_counts.data_ptr<int32_t>(),
        tile_faces.data_ptr<int32_t>(),
        B, V, F,
        image_size,
        tile_size,
        tiles_x, tiles_y,
        max_per_tile,
        sil.data_ptr<float>(),
        rgb.data_ptr<float>());

    return {sil, rgb};
}
