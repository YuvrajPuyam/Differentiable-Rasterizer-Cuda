// csrc/rasterize_backward.cu
//
// Tiled backward for minimal Soft Rasterizer.
//
// Inputs:
//   grad_sil:   (B,1,H,W) float
//   grad_rgb:   (B,3,H,W) float
//   verts:      (B,V,3) float  (NDC)
//   faces:      (F,3)   int64
//   colors:     (B,V,3) float
//   tile_counts:(B,T)   int32
//   tile_faces: (B,T,max) int32
//
// Output:
//   grad_verts: (B,V,3) float  (only x,y get non-zero; z grads = 0)
//
// Notes:
// - Uses your barycentric derivative helper.
// - Performs two passes per pixel over tile faces:
//     pass1: compute w_sum and col_sum to get C
//     pass2: distribute gradients to verts (atomicAdd)
// - use the simplified silhouette gradient dL/dalpha += gS (
//   

#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

using at::Tensor;

// ---------------------------------------------
// Utility: pixel index in flattened (B, H, W)
// ---------------------------------------------
__device__ inline int pixel_index(int b, int y, int x, int image_size) {
    return b * image_size * image_size + y * image_size + x;
}

// ---------------------------------------------
// Utility: barycentric-based "distance" metric
// D = min(a,b,c), computed from standard barycentrics.
// ---------------------------------------------
__device__ inline bool soft_barycentric_metric(
    float px, float py,
    float x0, float y0,
    float x1, float y1,
    float x2, float y2,
    float &D_out)
{
    // Quick bbox reject
    float minx = fminf(x0, fminf(x1, x2)) - 1e-4f;
    float maxx = fmaxf(x0, fmaxf(x1, x2)) + 1e-4f;
    float miny = fminf(y0, fminf(y1, y2)) - 1e-4f;
    float maxy = fmaxf(y0, fmaxf(y1, y2)) + 1e-4f;

    if (px < minx || px > maxx || py < miny || py > maxy) {
        return false;
    }

    float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    if (fabsf(denom) < 1e-8f) {
        return false; // degenerate tri
    }
    float inv_denom = 1.0f / denom;

    float a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) * inv_denom;
    float b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) * inv_denom;
    float c = 1.0f - a - b;

    D_out = fminf(a, fminf(b, c));
    return true;
}

// ---------------------------------------------
// Barycentric derivative helper 
// Computes d(bary)/d(x_i,y_i) using quotient rule.
// which_bary: 0->a, 1->b   (c handled as -da-db outside)
// which_vert: 0->v0, 1->v1, 2->v2
// ---------------------------------------------
__device__ void get_barycentric_deriv(
    int which_bary,
    int which_vert,
    float px, float py,
    float x0, float y0,
    float x1, float y1,
    float x2, float y2,
    float denom,
    float bary_val,
    float &dg_dx, float &dg_dy
) {
    float inv_denom = 1.0f / (denom + 1e-8f);

    float d_denom_dx = 0.0f;
    float d_denom_dy = 0.0f;
    float d_numer_dx = 0.0f;
    float d_numer_dy = 0.0f;

    // denom = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
    // Derivative of denom w.r.t each vertex coord:
    if (which_vert == 0) {
        d_denom_dx = (y1 - y2);
        d_denom_dy = (x2 - x1);
    } else if (which_vert == 1) {
        d_denom_dx = (y2 - y0);
        d_denom_dy = (x0 - x2);
    } else { // which_vert == 2
        d_denom_dx = (y0 - y1);
        d_denom_dy = (x1 - x0);
    }

    // Numerators:
    // a_num = (y1-y2)*(px-x2) + (x2-x1)*(py-y2)
    // b_num = (y2-y0)*(px-x2) + (x0-x2)*(py-y2)
    if (which_bary == 0) { // a
        if (which_vert == 1) {
            // da_num/dx1, dy1
            d_numer_dx = -(py - y2);
            d_numer_dy =  (px - x2);
        } else if (which_vert == 2) {
            // da_num/dx2, dy2
            d_numer_dx =  (py - y1);
            d_numer_dy = -(px - x1);
        }
        // which_vert==0 -> 0 for numerator
    } else if (which_bary == 1) { // b
        if (which_vert == 0) {
            // db_num/dx0, dy0
            d_numer_dx =  (py - y2);
            d_numer_dy = -(px - x2);
        } else if (which_vert == 2) {
            // db_num/dx2, dy2
            d_numer_dx = -(py - y0);
            d_numer_dy =  (px - x0);
        }
        // which_vert==1 -> 0 for numerator
    }

    dg_dx = (d_numer_dx - bary_val * d_denom_dx) * inv_denom;
    dg_dy = (d_numer_dy - bary_val * d_denom_dy) * inv_denom;
}

// ---------------------------------------------
// Tiled backward kernel
// ---------------------------------------------
__global__ void rasterize_backward_kernel_tiled(
    const float* __restrict__ verts,        // (B,V,3)
    const int64_t* __restrict__ faces,      // (F,3)
    const float* __restrict__ colors,       // (B,V,3)
    const int32_t* __restrict__ tile_counts,// (B*T)
    const int32_t* __restrict__ tile_faces, // (B*T*max)
    const float* __restrict__ grad_sil,     // (B*H*W) flattened view of (B,1,H,W)
    const float* __restrict__ grad_rgb,     // (B*3*H*W)
    int B, int V, int F,
    int image_size,
    int tile_size,
    int tiles_x,
    int tiles_y,
    int max_per_tile,
    float* __restrict__ grad_verts          // (B,V,3)
) {
    int b = blockIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || y >= image_size || x >= image_size) return;

    const int H = image_size;
    const int W = image_size;

    const int pix = pixel_index(b, y, x, image_size);

    // Pixel center in NDC
    const float px = ((x + 0.5f) / image_size) * 2.0f - 1.0f;
    const float py = ((y + 0.5f) / image_size) * 2.0f - 1.0f;

    float gS = grad_sil[pix];

    // grad_rgb layout is (B,3,H,W)
    const int rgb_base = b * 3 * H * W + y * W + x;
    float3 gC;
    gC.x = grad_rgb[rgb_base + 0 * H * W];
    gC.y = grad_rgb[rgb_base + 1 * H * W];
    gC.z = grad_rgb[rgb_base + 2 * H * W];

    const float gC_mag = fabsf(gC.x) + fabsf(gC.y) + fabsf(gC.z);
    if (fabsf(gS) < 1e-8f && gC_mag < 1e-8f) {
        return;
    }

    const float sigma = 1e-2f;

    // Tile id for (x,y)
    int tx = x / tile_size;
    int ty = y / tile_size;
    tx = min(tx, tiles_x - 1);
    ty = min(ty, tiles_y - 1);
    int tile = ty * tiles_x + tx;

    const int T = tiles_x * tiles_y;
    int idx_counts = b * T + tile;
    int count = tile_counts[idx_counts];
    if (count > max_per_tile) count = max_per_tile;

    int base_faces = (b * T + tile) * max_per_tile;

    // ---------------------------------------------
    // Pass 1: compute w_sum, col_sum to get C
    // ---------------------------------------------
    float w_sum = 0.0f;
    float3 col_sum = make_float3(0.0f, 0.0f, 0.0f);

    for (int ii = 0; ii < count; ++ii) {
        int f = (int)tile_faces[base_faces + ii];
        if (f < 0 || f >= F) continue;

        int64_t i0 = faces[3 * f + 0];
        int64_t i1 = faces[3 * f + 1];
        int64_t i2 = faces[3 * f + 2];

        const float* v0 = verts + (b * V + (int)i0) * 3;
        const float* v1 = verts + (b * V + (int)i1) * 3;
        const float* v2 = verts + (b * V + (int)i2) * 3;

        float x0 = v0[0], y0 = v0[1];
        float x1 = v1[0], y1 = v1[1];
        float x2 = v2[0], y2 = v2[1];

        float D;
        if (!soft_barycentric_metric(px, py, x0, y0, x1, y1, x2, y2, D)) continue;

        float alpha = 1.0f / (1.0f + expf(-D / sigma));
        if (alpha <= 1e-6f) continue;

        const float* c0 = colors + (b * V + (int)i0) * 3;
        const float* c1 = colors + (b * V + (int)i1) * 3;
        const float* c2 = colors + (b * V + (int)i2) * 3;

        float3 cf;
        cf.x = (c0[0] + c1[0] + c2[0]) / 3.0f;
        cf.y = (c0[1] + c1[1] + c2[1]) / 3.0f;
        cf.z = (c0[2] + c1[2] + c2[2]) / 3.0f;

        w_sum += alpha;
        col_sum.x += alpha * cf.x;
        col_sum.y += alpha * cf.y;
        col_sum.z += alpha * cf.z;
    }

    if (w_sum <= 1e-8f) return;

    float inv_w_sum = 1.0f / (w_sum + 1e-6f);
    float3 C;
    C.x = col_sum.x * inv_w_sum;
    C.y = col_sum.y * inv_w_sum;
    C.z = col_sum.z * inv_w_sum;

    // ---------------------------------------------
    // Pass 2: distribute gradients
    // ---------------------------------------------
    for (int ii = 0; ii < count; ++ii) {
        int f = (int)tile_faces[base_faces + ii];
        if (f < 0 || f >= F) continue;

        int64_t i0 = faces[3 * f + 0];
        int64_t i1 = faces[3 * f + 1];
        int64_t i2 = faces[3 * f + 2];

        const float* v0 = verts + (b * V + (int)i0) * 3;
        const float* v1 = verts + (b * V + (int)i1) * 3;
        const float* v2 = verts + (b * V + (int)i2) * 3;

        float x0 = v0[0], y0 = v0[1];
        float x1 = v1[0], y1 = v1[1];
        float x2 = v2[0], y2 = v2[1];

        // recompute barycentrics for min routing + derivatives
        float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
        if (fabsf(denom) < 1e-8f) continue;
        float inv_denom_local = 1.0f / denom;

        float a_val = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) * inv_denom_local;
        float b_val = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) * inv_denom_local;
        float c_val = 1.0f - a_val - b_val;

        float D = fminf(a_val, fminf(b_val, c_val));

        // bbox reject 
        float minx = fminf(x0, fminf(x1, x2)) - 1e-4f;
        float maxx = fmaxf(x0, fmaxf(x1, x2)) + 1e-4f;
        float miny = fminf(y0, fminf(y1, y2)) - 1e-4f;
        float maxy = fmaxf(y0, fmaxf(y1, y2)) + 1e-4f;
        if (px < minx || px > maxx || py < miny || py > maxy) continue;

        float alpha = 1.0f / (1.0f + expf(-D / sigma));
        if (alpha <= 1e-6f) continue;

        const float* c0 = colors + (b * V + (int)i0) * 3;
        const float* c1 = colors + (b * V + (int)i1) * 3;
        const float* c2 = colors + (b * V + (int)i2) * 3;

        float3 cf;
        cf.x = (c0[0] + c1[0] + c2[0]) / 3.0f;
        cf.y = (c0[1] + c1[1] + c2[1]) / 3.0f;
        cf.z = (c0[2] + c1[2] + c2[2]) / 3.0f;

        // 1) dL/d(alpha) from silhouette 
        float dL_dalpha = gS;

        // 2) RGB contribution:
        // C = sum(alpha*cf)/sum(alpha)
        // dC/dalpha_f = (cf - C) / w_sum
        float3 diff;
        diff.x = cf.x - C.x;
        diff.y = cf.y - C.y;
        diff.z = cf.z - C.z;

        float dot_gC_diff = gC.x * diff.x + gC.y * diff.y + gC.z * diff.z;
        dL_dalpha += dot_gC_diff * inv_w_sum;

        // 3) d(alpha)/dD
        float dalpha_dD = alpha * (1.0f - alpha) / sigma;
        float dL_dD = dL_dalpha * dalpha_dD;

        // 4) D = min(a,b,c) routing
        int min_idx = 0; // 0->a, 1->b, 2->c
        if (b_val < a_val && b_val < c_val) min_idx = 1;
        if (c_val < a_val && c_val < b_val) min_idx = 2;

        // distribute to v0,v1,v2
        for (int v_idx = 0; v_idx < 3; ++v_idx) {
            float gx = 0.0f;
            float gy = 0.0f;

            if (min_idx == 2) {
                // c = 1 - a - b  => dc = -da - db
                float da_dx, da_dy, db_dx, db_dy;
                get_barycentric_deriv(0, v_idx, px, py, x0, y0, x1, y1, x2, y2, denom, a_val, da_dx, da_dy);
                get_barycentric_deriv(1, v_idx, px, py, x0, y0, x1, y1, x2, y2, denom, b_val, db_dx, db_dy);
                gx = -da_dx - db_dx;
                gy = -da_dy - db_dy;
            } else {
                float val_to_use = (min_idx == 0) ? a_val : b_val;
                get_barycentric_deriv(min_idx, v_idx, px, py, x0, y0, x1, y1, x2, y2, denom, val_to_use, gx, gy);
            }

            float grad_x = dL_dD * gx;
            float grad_y = dL_dD * gy;

            int64_t target_idx = (v_idx == 0) ? i0 : ((v_idx == 1) ? i1 : i2);
            int base = (b * V + (int)target_idx) * 3;

            atomicAdd(&grad_verts[base + 0], grad_x);
            atomicAdd(&grad_verts[base + 1], grad_y);
            // z-grad left as 0
        }
    }
}

Tensor rasterize_backward_cuda(
    Tensor grad_sil,    // (B,1,H,W)
    Tensor grad_rgb,    // (B,3,H,W)
    Tensor verts,       // (B,V,3)
    Tensor faces,       // (F,3)
    Tensor colors,      // (B,V,3)
    Tensor tile_counts, // (B,T) int32
    Tensor tile_faces,  // (B,T,max) int32
    int image_size,
    int tile_size,
    int max_per_tile
) {
    TORCH_CHECK(grad_sil.is_cuda(), "grad_sil must be CUDA");
    TORCH_CHECK(grad_rgb.is_cuda(), "grad_rgb must be CUDA");
    TORCH_CHECK(verts.is_cuda(), "verts must be CUDA");
    TORCH_CHECK(faces.is_cuda(), "faces must be CUDA");
    TORCH_CHECK(colors.is_cuda(), "colors must be CUDA");
    TORCH_CHECK(tile_counts.is_cuda(), "tile_counts must be CUDA");
    TORCH_CHECK(tile_faces.is_cuda(), "tile_faces must be CUDA");

    TORCH_CHECK(grad_sil.dim() == 4 && grad_sil.size(1) == 1, "grad_sil must be (B,1,H,W)");
    TORCH_CHECK(grad_rgb.dim() == 4 && grad_rgb.size(1) == 3, "grad_rgb must be (B,3,H,W)");
    TORCH_CHECK(verts.dim() == 3 && verts.size(2) == 3, "verts must be (B,V,3)");
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces must be (F,3)");
    TORCH_CHECK(colors.sizes() == verts.sizes(), "colors must be (B,V,3)");
    TORCH_CHECK(tile_counts.scalar_type() == at::kInt, "tile_counts must be int32");
    TORCH_CHECK(tile_faces.scalar_type() == at::kInt, "tile_faces must be int32");

    grad_sil = grad_sil.contiguous();
    grad_rgb = grad_rgb.contiguous();
    verts = verts.contiguous();
    faces = faces.contiguous();
    colors = colors.contiguous();
    tile_counts = tile_counts.contiguous();
    tile_faces = tile_faces.contiguous();

    int B = (int)verts.size(0);
    int V = (int)verts.size(1);
    int F = (int)faces.size(0);

    const int H = image_size;
    const int W = image_size;

    TORCH_CHECK(tile_size > 0, "tile_size must be > 0");
    TORCH_CHECK(max_per_tile > 0, "max_per_tile must be > 0");

    const int tiles_x = (W + tile_size - 1) / tile_size;
    const int tiles_y = (H + tile_size - 1) / tile_size;

    // Zero-init gradients
    auto grad_verts = at::zeros_like(verts);

    dim3 block(16, 16, 1);
    dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y,
        B
    );

    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(verts));

    rasterize_backward_kernel_tiled<<<grid, block>>>(
        verts.data_ptr<float>(),
        faces.data_ptr<int64_t>(),
        colors.data_ptr<float>(),
        tile_counts.data_ptr<int32_t>(),
        tile_faces.data_ptr<int32_t>(),
        grad_sil.data_ptr<float>(),
        grad_rgb.data_ptr<float>(),
        B, V, F,
        image_size,
        tile_size,
        tiles_x, tiles_y,
        max_per_tile,
        grad_verts.data_ptr<float>());

    return grad_verts;
}
