// csrc/rasterize_backward.cu
// Modular tiled backward with correct product-form silhouette gradient.
//
// Forward per pixel (over tile face list):
//   bg_prod = Π_f (1 - alpha_f)
//   S = 1 - bg_prod
//   C = (Σ alpha_f * cf) / (Σ alpha_f)
//   rgb = clamp(S,0,1) * C
//
// Backward uses:
//   dS/dalpha_f = bg_prod / (1 - alpha_f)
// and includes RGB via scaling by S and via C.

#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

using at::Tensor;

// ------------------------------
// Constants / eps
// ------------------------------
__device__ constexpr float kBBoxMargin = 1e-4f;
__device__ constexpr float kDenomEps   = 1e-8f;
__device__ constexpr float kAlphaEps   = 1e-6f;
__device__ constexpr float kWSumEps    = 1e-6f;

// ------------------------------
// Index helpers
// ------------------------------
__device__ inline int pixel_index(int b, int y, int x, int image_size) {
    return b * image_size * image_size + y * image_size + x;
}

__device__ inline float2 pixel_center_ndc(int x, int y, int image_size) {
    float px = ((x + 0.5f) / image_size) * 2.0f - 1.0f;
    float py = ((y + 0.5f) / image_size) * 2.0f - 1.0f;
    return make_float2(px, py);
}

__device__ inline float clamp01(float v) {
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

__device__ inline float3 make_face_color_avg(
    const float* __restrict__ colors,
    int b, int V, int64_t i0, int64_t i1, int64_t i2
) {
    const float* c0 = colors + (b * V + (int)i0) * 3;
    const float* c1 = colors + (b * V + (int)i1) * 3;
    const float* c2 = colors + (b * V + (int)i2) * 3;

    float3 cf;
    cf.x = (c0[0] + c1[0] + c2[0]) / 3.0f;
    cf.y = (c0[1] + c1[1] + c2[1]) / 3.0f;
    cf.z = (c0[2] + c1[2] + c2[2]) / 3.0f;
    return cf;
}

__device__ inline bool bbox_reject(
    float px, float py,
    float x0, float y0,
    float x1, float y1,
    float x2, float y2
) {
    float minx = fminf(x0, fminf(x1, x2)) - kBBoxMargin;
    float maxx = fmaxf(x0, fmaxf(x1, x2)) + kBBoxMargin;
    float miny = fminf(y0, fminf(y1, y2)) - kBBoxMargin;
    float maxy = fmaxf(y0, fmaxf(y1, y2)) + kBBoxMargin;
    return (px < minx || px > maxx || py < miny || py > maxy);
}

// ------------------------------
// Barycentrics and alpha
// ------------------------------
struct BaryInfo {
    float a, b, c;
    float denom;
    float D;       // min(a,b,c)
    int min_idx;   // 0:a, 1:b, 2:c
};

__device__ inline bool compute_bary_info(
    float px, float py,
    float x0, float y0,
    float x1, float y1,
    float x2, float y2,
    BaryInfo& out
) {
    if (bbox_reject(px, py, x0, y0, x1, y1, x2, y2)) return false;

    float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    if (fabsf(denom) < kDenomEps) return false;

    float inv_denom = 1.0f / denom;

    float a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) * inv_denom;
    float b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) * inv_denom;
    float c = 1.0f - a - b;

    float D = fminf(a, fminf(b, c));
    int min_idx = 0;
    if (b < a && b < c) min_idx = 1;
    if (c < a && c < b) min_idx = 2;

    out.a = a; out.b = b; out.c = c;
    out.D = D;
    out.denom = denom;
    out.min_idx = min_idx;
    return true;
}

__device__ inline float sigmoid_alpha(float D, float sigma) {
    // You can swap expf -> __expf once you enable fast math
    return 1.0f / (1.0f + expf(-D / sigma));
}

__device__ inline float dalpha_dD(float alpha, float sigma) {
    return alpha * (1.0f - alpha) / sigma;
}

// ------------------------------
// Barycentric derivative helper (same math as your original)
// ------------------------------
__device__ void get_barycentric_deriv(
    int which_bary, // 0->a, 1->b
    int which_vert, // 0->v0, 1->v1, 2->v2
    float px, float py,
    float x0, float y0,
    float x1, float y1,
    float x2, float y2,
    float denom,
    float bary_val,
    float &dg_dx, float &dg_dy
) {
    float inv_denom = 1.0f / (denom + kDenomEps);

    float d_denom_dx = 0.0f;
    float d_denom_dy = 0.0f;
    float d_numer_dx = 0.0f;
    float d_numer_dy = 0.0f;

    // denom deriv
    if (which_vert == 0) {
        d_denom_dx = (y1 - y2);
        d_denom_dy = (x2 - x1);
    } else if (which_vert == 1) {
        d_denom_dx = (y2 - y0);
        d_denom_dy = (x0 - x2);
    } else {
        d_denom_dx = (y0 - y1);
        d_denom_dy = (x1 - x0);
    }

    // numerator deriv
    if (which_bary == 0) { // a
        if (which_vert == 1) {
            d_numer_dx = -(py - y2);
            d_numer_dy =  (px - x2);
        } else if (which_vert == 2) {
            d_numer_dx =  (py - y1);
            d_numer_dy = -(px - x1);
        }
    } else { // b
        if (which_vert == 0) {
            d_numer_dx =  (py - y2);
            d_numer_dy = -(px - x2);
        } else if (which_vert == 2) {
            d_numer_dx = -(py - y0);
            d_numer_dy =  (px - x0);
        }
    }

    dg_dx = (d_numer_dx - bary_val * d_denom_dx) * inv_denom;
    dg_dy = (d_numer_dy - bary_val * d_denom_dy) * inv_denom;
}

// Route dD/d(x_i,y_i) given BaryInfo and min_idx
__device__ inline void dD_dvertex_xy(
    int min_idx,
    int which_vert,
    float px, float py,
    float x0, float y0,
    float x1, float y1,
    float x2, float y2,
    float denom,
    float a_val,
    float b_val,
    float &dD_dx, float &dD_dy
) {
    if (min_idx == 2) {
        // D = c = 1 - a - b
        float da_dx, da_dy, db_dx, db_dy;
        get_barycentric_deriv(0, which_vert, px, py, x0, y0, x1, y1, x2, y2, denom, a_val, da_dx, da_dy);
        get_barycentric_deriv(1, which_vert, px, py, x0, y0, x1, y1, x2, y2, denom, b_val, db_dx, db_dy);
        dD_dx = -da_dx - db_dx;
        dD_dy = -da_dy - db_dy;
    } else {
        // D = a or b
        float val = (min_idx == 0) ? a_val : b_val;
        get_barycentric_deriv(min_idx, which_vert, px, py, x0, y0, x1, y1, x2, y2, denom, val, dD_dx, dD_dy);
    }
}

// ------------------------------
// Tile iteration
// ------------------------------
struct TileRange {
    int base_faces;
    int count;
    int T;
};

__device__ inline TileRange get_tile_range(
    int b, int x, int y,
    int tile_size, int tiles_x, int tiles_y,
    int max_per_tile,
    const int32_t* __restrict__ tile_counts,
    const int32_t* __restrict__ tile_faces
) {
    TileRange tr;
    int tx = x / tile_size;
    int ty = y / tile_size;
    tx = min(tx, tiles_x - 1);
    ty = min(ty, tiles_y - 1);
    int tile = ty * tiles_x + tx;

    tr.T = tiles_x * tiles_y;
    int idx = b * tr.T + tile;
    int c = tile_counts[idx];
    if (c > max_per_tile) c = max_per_tile;
    tr.count = c;
    tr.base_faces = (b * tr.T + tile) * max_per_tile;
    (void)tile_faces; // just to keep signature symmetric if you want to extend later
    return tr;
}

// ------------------------------
// Pass 1 accumulation
// ------------------------------
struct PixelAgg {
    float w_sum;
    float3 col_sum;
    float bg_prod;
};

__device__ inline PixelAgg init_pixel_agg() {
    PixelAgg a;
    a.w_sum = 0.0f;
    a.col_sum = make_float3(0.0f, 0.0f, 0.0f);
    a.bg_prod = 1.0f;
    return a;
}

__device__ inline void accumulate_face(PixelAgg& agg, float alpha, const float3& cf) {
    if (alpha <= 1e-6f) return;
    agg.bg_prod *= (1.0f - alpha);
    agg.w_sum   += alpha;
    agg.col_sum.x += alpha * cf.x;
    agg.col_sum.y += alpha * cf.y;
    agg.col_sum.z += alpha * cf.z;
}

// ------------------------------
// dL/dalpha (product-form + rgb)
// ------------------------------
__device__ inline float dot3(const float3& a, const float3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ inline float compute_dL_dalpha(
    float gS,               // dL/dS
    const float3& gRGB,     // dL/dRGB
    float alpha,
    float bg_prod,
    float S,
    float s_clamped,
    float ds_dS,
    const float3& C,
    float inv_w_sum,
    const float3& cf
) {
    float one_minus_alpha = fmaxf(1.0f - alpha, kAlphaEps);
    float dS_dalpha = bg_prod / one_minus_alpha;

    // (1) silhouette contribution
    float dL_dalpha = gS * dS_dalpha;

    // (2) rgb contribution via scaling by s_clamped (rgb = s*C)
    float dot_gRGB_C = dot3(gRGB, C);
    dL_dalpha += (dot_gRGB_C * ds_dS) * dS_dalpha;

    // (3) rgb contribution via C
    float3 diff = make_float3(cf.x - C.x, cf.y - C.y, cf.z - C.z);
    float dot_gRGB_diff = dot3(gRGB, diff);
    dL_dalpha += s_clamped * (dot_gRGB_diff * inv_w_sum);

    return dL_dalpha;
}

// ------------------------------
// Kernel
// ------------------------------
__global__ void rasterize_backward_kernel_tiled_modular(
    const float* __restrict__ verts,          // (B,V,3)
    const int64_t* __restrict__ faces,        // (F,3)
    const float* __restrict__ colors,         // (B,V,3)
    const int32_t* __restrict__ tile_counts,  // (B*T)
    const int32_t* __restrict__ tile_faces,   // (B*T*max)
    const float* __restrict__ grad_sil,       // (B*H*W)
    const float* __restrict__ grad_rgb,       // (B*3*H*W)
    int B, int V, int F,
    int image_size,
    int tile_size,
    int tiles_x,
    int tiles_y,
    int max_per_tile,
    float sigma,
    float* __restrict__ grad_verts            // (B,V,3)
) {
    int b = blockIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || y >= image_size || x >= image_size) return;

    const int H = image_size;
    const int W = image_size;
    const int pix = pixel_index(b, y, x, image_size);

    float2 p = pixel_center_ndc(x, y, image_size);
    float px = p.x, py = p.y;

    float gS = grad_sil[pix];

    // (B,3,H,W) flattened
    int rgb_base = b * 3 * H * W + y * W + x;
    float3 gRGB;
    gRGB.x = grad_rgb[rgb_base + 0 * H * W];
    gRGB.y = grad_rgb[rgb_base + 1 * H * W];
    gRGB.z = grad_rgb[rgb_base + 2 * H * W];

    float gRGB_mag = fabsf(gRGB.x) + fabsf(gRGB.y) + fabsf(gRGB.z);
    if (fabsf(gS) < 1e-8f && gRGB_mag < 1e-8f) return;

    TileRange tr = get_tile_range(
        b, x, y, tile_size, tiles_x, tiles_y, max_per_tile,
        tile_counts, tile_faces
    );
    if (tr.count <= 0) return;

    // -------- Pass 1: aggregate --------
    PixelAgg agg = init_pixel_agg();

    for (int ii = 0; ii < tr.count; ++ii) {
        int f = (int)tile_faces[tr.base_faces + ii];
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

        BaryInfo bi;
        if (!compute_bary_info(px, py, x0, y0, x1, y1, x2, y2, bi)) continue;

        float alpha = sigmoid_alpha(bi.D, sigma);
        if (alpha <= 1e-6f) continue;

        float3 cf = make_face_color_avg(colors, b, V, i0, i1, i2);
        accumulate_face(agg, alpha, cf);
    }

    if (agg.w_sum <= 1e-8f) return;

    float inv_w_sum = 1.0f / (agg.w_sum + kWSumEps);
    float3 C = make_float3(
        agg.col_sum.x * inv_w_sum,
        agg.col_sum.y * inv_w_sum,
        agg.col_sum.z * inv_w_sum
    );

    float S = 1.0f - agg.bg_prod;
    float s_clamped = clamp01(S);
    float ds_dS = (S > 0.0f && S < 1.0f) ? 1.0f : 0.0f;

    // -------- Pass 2: distribute grads --------
    for (int ii = 0; ii < tr.count; ++ii) {
        int f = (int)tile_faces[tr.base_faces + ii];
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

        BaryInfo bi;
        if (!compute_bary_info(px, py, x0, y0, x1, y1, x2, y2, bi)) continue;

        float alpha = sigmoid_alpha(bi.D, sigma);
        if (alpha <= 1e-6f) continue;

        float3 cf = make_face_color_avg(colors, b, V, i0, i1, i2);

        float dL_dalpha = compute_dL_dalpha(
            gS, gRGB, alpha,
            agg.bg_prod, S,
            s_clamped, ds_dS,
            C, inv_w_sum,
            cf
        );

        float dL_dD = dL_dalpha * dalpha_dD(alpha, sigma);

        // dD/d(x_i,y_i) via min routing
        for (int v_idx = 0; v_idx < 3; ++v_idx) {
            float dD_dx = 0.0f, dD_dy = 0.0f;
            dD_dvertex_xy(
                bi.min_idx, v_idx,
                px, py,
                x0, y0, x1, y1, x2, y2,
                bi.denom,
                bi.a, bi.b,
                dD_dx, dD_dy
            );

            float grad_x = dL_dD * dD_dx;
            float grad_y = dL_dD * dD_dy;

            int64_t vid = (v_idx == 0) ? i0 : ((v_idx == 1) ? i1 : i2);
            int base = (b * V + (int)vid) * 3;

            atomicAdd(&grad_verts[base + 0], grad_x);
            atomicAdd(&grad_verts[base + 1], grad_y);
        }
    }
}

// ------------------------------
// Host wrapper
// ------------------------------
Tensor rasterize_backward_cuda(
    Tensor grad_sil,     // (B,1,H,W)
    Tensor grad_rgb,     // (B,3,H,W)
    Tensor verts,        // (B,V,3)
    Tensor faces,        // (F,3)
    Tensor colors,       // (B,V,3)
    Tensor tile_counts,  // (B,T) int32
    Tensor tile_faces,   // (B,T,max) int32
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

    auto grad_verts = at::zeros_like(verts);

    dim3 block(16, 16, 1);
    dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y,
        B
    );

    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(verts));

    // Keep sigma consistent with forward
    const float sigma = 1e-2f;

    rasterize_backward_kernel_tiled_modular<<<grid, block>>>(
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
        sigma,
        grad_verts.data_ptr<float>());

    return grad_verts;
}
