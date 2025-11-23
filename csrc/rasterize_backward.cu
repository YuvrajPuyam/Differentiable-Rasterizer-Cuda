// csrc/rasterize_backward.cu
//
// Backward CUDA kernel for multi-triangle soft silhouettes.
// We recompute the same coverage as in forward, then propagate
// gradients dL/dS -> dL/dverts.
//
// NOTE: This uses a small finite-difference for the derivative
// of the edge distance w.r.t. the segment endpoints, but all
// chain rule around it is analytic.
//

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// ---- device helpers (same as in forward) ----

__device__ inline float edge_function(
    float ax, float ay,
    float bx, float by,
    float px, float py) {
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

__device__ inline float point_segment_distance(
    float px, float py,
    float ax, float ay,
    float bx, float by) {

    float vx = bx - ax;
    float vy = by - ay;
    float wx = px - ax;
    float wy = py - ay;

    float vv = vx * vx + vy * vy + 1e-8f;
    float t = (vx * wx + vy * wy) / vv;
    t = fmaxf(0.0f, fminf(1.0f, t));

    float cx = ax + t * vx;
    float cy = ay + t * vy;

    float dx = px - cx;
    float dy = py - cy;
    return sqrtf(dx * dx + dy * dy + 1e-12f);
}

// ---- backward kernel ----

__global__ void rasterize_backward_kernel(
    const float* __restrict__ grad_image,  // (B, 1, H, W)
    const float* __restrict__ verts,       // (B, V, 3)
    const int64_t* __restrict__ faces,     // (F, 3)
    int B,
    int V,
    int F,
    int image_size,
    float* __restrict__ grad_verts) {      // (B, V, 3)

    int b = blockIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || y >= image_size || x >= image_size) {
        return;
    }

    // index into grad_image: (B,1,H,W) -> B*H*W + y*W + x
    int pixel_index = b * image_size * image_size + y * image_size + x;
    float dL_dS = grad_image[pixel_index];  // scalar for this pixel

    if (dL_dS == 0.0f) {
        // No contribution from this pixel
        return;
    }

    // Pixel center in NDC
    float px = 2.0f * ((x + 0.5f) / float(image_size)) - 1.0f;
    float py = 2.0f * ((y + 0.5f) / float(image_size)) - 1.0f;

    const float PIXEL_SIZE   = 2.0f / float(image_size);
    const float GAMMA_PIXELS = 0.5f;
    const float EPS          = 1e-6f;
    const float MIN_BG       = 1e-6f;

    // -------- Pass 1: compute background probability bg_prob --------
    float bg_prob = 1.0f;

    for (int f = 0; f < F; ++f) {
        int64_t i0 = faces[f * 3 + 0];
        int64_t i1 = faces[f * 3 + 1];
        int64_t i2 = faces[f * 3 + 2];

        if (i0 < 0 || i0 >= V || i1 < 0 || i1 >= V || i2 < 0 || i2 >= V) {
            continue;
        }

        int base0 = (b * V + int(i0)) * 3;
        int base1 = (b * V + int(i1)) * 3;
        int base2 = (b * V + int(i2)) * 3;

        float v0x = verts[base0 + 0];
        float v0y = verts[base0 + 1];
        float v1x = verts[base1 + 0];
        float v1y = verts[base1 + 1];
        float v2x = verts[base2 + 0];
        float v2y = verts[base2 + 1];

        float w0 = edge_function(v1x, v1y, v2x, v2y, px, py);
        float w1 = edge_function(v2x, v2y, v0x, v0y, px, py);
        float w2 = edge_function(v0x, v0y, v1x, v1y, px, py);

        float area = edge_function(v0x, v0y, v1x, v1y, v2x, v2y);
        if (area == 0.0f) {
            continue;
        }

        bool same_sign = (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) ||
                         (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
        bool inside = same_sign;

        float d0 = point_segment_distance(px, py, v0x, v0y, v1x, v1y);
        float d1 = point_segment_distance(px, py, v1x, v1y, v2x, v2y);
        float d2 = point_segment_distance(px, py, v2x, v2y, v0x, v0y);
        float min_dist = fminf(d0, fminf(d1, d2));

        float signed_d = inside ? -min_dist : min_dist;
        float d_pixels = signed_d / PIXEL_SIZE;

        float alpha = 1.0f / (1.0f + expf(d_pixels / GAMMA_PIXELS));
        if (alpha < EPS)         alpha = EPS;
        if (alpha > 1.0f - EPS)  alpha = 1.0f - EPS;

        float one_minus_alpha = 1.0f - alpha;
        bg_prob *= one_minus_alpha;

        if (bg_prob < MIN_BG) {
            bg_prob = 0.0f;
            break;
        }
    }

    if (bg_prob <= 0.0f) {
        // Pixel almost surely foreground => derivative wrt alpha will be small
        // but still finite; we continue with bg_prob=0.
        bg_prob = 0.0f;
    }

    // -------- Pass 2: per-face gradient --------
    // We now loop again over faces, recompute alpha, and propagate:
    //
    // S = 1 - Π_k (1 - α_k)
    // ∂S/∂α_f = bg_prob / (1 - α_f)
    // α = σ(-d/γ), etc.
    //
    const float FD_EPS_V = 1e-4f;  // small step for vertex FD inside kernel

    for (int f = 0; f < F; ++f) {
        int64_t i0 = faces[f * 3 + 0];
        int64_t i1 = faces[f * 3 + 1];
        int64_t i2 = faces[f * 3 + 2];

        if (i0 < 0 || i0 >= V || i1 < 0 || i1 >= V || i2 < 0 || i2 >= V) {
            continue;
        }

        int base0 = (b * V + int(i0)) * 3;
        int base1 = (b * V + int(i1)) * 3;
        int base2 = (b * V + int(i2)) * 3;

        float v0x = verts[base0 + 0];
        float v0y = verts[base0 + 1];
        float v1x = verts[base1 + 0];
        float v1y = verts[base1 + 1];
        float v2x = verts[base2 + 0];
        float v2y = verts[base2 + 1];

        float w0 = edge_function(v1x, v1y, v2x, v2y, px, py);
        float w1 = edge_function(v2x, v2y, v0x, v0y, px, py);
        float w2 = edge_function(v0x, v0y, v1x, v1y, px, py);

        float area = edge_function(v0x, v0y, v1x, v1y, v2x, v2y);
        if (area == 0.0f) {
            continue;
        }

        bool same_sign = (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) ||
                         (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
        bool inside = same_sign;

        float d0 = point_segment_distance(px, py, v0x, v0y, v1x, v1y);
        float d1 = point_segment_distance(px, py, v1x, v1y, v2x, v2y);
        float d2 = point_segment_distance(px, py, v2x, v2y, v0x, v0y);

        // Determine which edge controls the signed distance
        float min_dist = d0;
        int min_edge = 0;
        if (d1 < min_dist) { min_dist = d1; min_edge = 1; }
        if (d2 < min_dist) { min_dist = d2; min_edge = 2; }

        float sign = inside ? -1.0f : 1.0f;
        float signed_d = sign * min_dist;
        float d_pixels = signed_d / PIXEL_SIZE;

        float alpha = 1.0f / (1.0f + expf(d_pixels / GAMMA_PIXELS));
        if (alpha < EPS)         alpha = EPS;
        if (alpha > 1.0f - EPS)  alpha = 1.0f - EPS;
        float one_minus_alpha = 1.0f - alpha;

        // dL/dalpha_f
        float dS_dalpha = (one_minus_alpha > EPS) ? (bg_prob / one_minus_alpha) : 0.0f;
        float dL_dalpha = dL_dS * dS_dalpha;

        // alpha = σ(-d_pixels / gamma) = 1 / (1 + exp(d_pixels / gamma))
        float d_alpha_dd_pixels = -(alpha * (1.0f - alpha)) / GAMMA_PIXELS;

        // d_pixels = signed_d / PIXEL_SIZE
        float dd_pixels_dsigned = 1.0f / PIXEL_SIZE;

        // signed_d = sign * min_dist
        float dsigned_ddmin = sign;

        // Chain rule to dL/dd_edge
        float dL_dd_edge = dL_dalpha * d_alpha_dd_pixels * dd_pixels_dsigned * dsigned_ddmin;

        if (fabsf(dL_dd_edge) < 1e-12f) {
            continue;
        }

        // ---- Finite-difference for edge distance w.r.t. segment endpoints ----
        float dd_dax = 0.0f, dd_day = 0.0f, dd_dbx = 0.0f, dd_dby = 0.0f;

        if (min_edge == 0) {
            // edge v0 -> v1
            float d_ax_pos = point_segment_distance(px, py, v0x + FD_EPS_V, v0y, v1x, v1y);
            float d_ax_neg = point_segment_distance(px, py, v0x - FD_EPS_V, v0y, v1x, v1y);
            dd_dax = (d_ax_pos - d_ax_neg) / (2.0f * FD_EPS_V);

            float d_ay_pos = point_segment_distance(px, py, v0x, v0y + FD_EPS_V, v1x, v1y);
            float d_ay_neg = point_segment_distance(px, py, v0x, v0y - FD_EPS_V, v1x, v1y);
            dd_day = (d_ay_pos - d_ay_neg) / (2.0f * FD_EPS_V);

            float d_bx_pos = point_segment_distance(px, py, v0x, v0y, v1x + FD_EPS_V, v1y);
            float d_bx_neg = point_segment_distance(px, py, v0x, v0y, v1x - FD_EPS_V, v1y);
            dd_dbx = (d_bx_pos - d_bx_neg) / (2.0f * FD_EPS_V);

            float d_by_pos = point_segment_distance(px, py, v0x, v0y, v1x, v1y + FD_EPS_V);
            float d_by_neg = point_segment_distance(px, py, v0x, v0y, v1x, v1y - FD_EPS_V);
            dd_dby = (d_by_pos - d_by_neg) / (2.0f * FD_EPS_V);

            // Accumulate into grad_verts for v0, v1
            atomicAdd(&grad_verts[base0 + 0], dL_dd_edge * dd_dax);
            atomicAdd(&grad_verts[base0 + 1], dL_dd_edge * dd_day);
            atomicAdd(&grad_verts[base1 + 0], dL_dd_edge * dd_dbx);
            atomicAdd(&grad_verts[base1 + 1], dL_dd_edge * dd_dby);
        }
        else if (min_edge == 1) {
            // edge v1 -> v2
            float d_ax_pos = point_segment_distance(px, py, v1x + FD_EPS_V, v1y, v2x, v2y);
            float d_ax_neg = point_segment_distance(px, py, v1x - FD_EPS_V, v1y, v2x, v2y);
            dd_dax = (d_ax_pos - d_ax_neg) / (2.0f * FD_EPS_V);

            float d_ay_pos = point_segment_distance(px, py, v1x, v1y + FD_EPS_V, v2x, v2y);
            float d_ay_neg = point_segment_distance(px, py, v1x, v1y - FD_EPS_V, v2x, v2y);
            dd_day = (d_ay_pos - d_ay_neg) / (2.0f * FD_EPS_V);

            float d_bx_pos = point_segment_distance(px, py, v1x, v1y, v2x + FD_EPS_V, v2y);
            float d_bx_neg = point_segment_distance(px, py, v1x, v1y, v2x - FD_EPS_V, v2y);
            dd_dbx = (d_bx_pos - d_bx_neg) / (2.0f * FD_EPS_V);

            float d_by_pos = point_segment_distance(px, py, v1x, v1y, v2x, v2y + FD_EPS_V);
            float d_by_neg = point_segment_distance(px, py, v1x, v1y, v2x, v2y - FD_EPS_V);
            dd_dby = (d_by_pos - d_by_neg) / (2.0f * FD_EPS_V);

            atomicAdd(&grad_verts[base1 + 0], dL_dd_edge * dd_dax);
            atomicAdd(&grad_verts[base1 + 1], dL_dd_edge * dd_day);
            atomicAdd(&grad_verts[base2 + 0], dL_dd_edge * dd_dbx);
            atomicAdd(&grad_verts[base2 + 1], dL_dd_edge * dd_dby);
        }
        else { // min_edge == 2
            // edge v2 -> v0
            float d_ax_pos = point_segment_distance(px, py, v2x + FD_EPS_V, v2y, v0x, v0y);
            float d_ax_neg = point_segment_distance(px, py, v2x - FD_EPS_V, v2y, v0x, v0y);
            dd_dax = (d_ax_pos - d_ax_neg) / (2.0f * FD_EPS_V);

            float d_ay_pos = point_segment_distance(px, py, v2x, v2y + FD_EPS_V, v0x, v0y);
            float d_ay_neg = point_segment_distance(px, py, v2x, v2y - FD_EPS_V, v0x, v0y);
            dd_day = (d_ay_pos - d_ay_neg) / (2.0f * FD_EPS_V);

            float d_bx_pos = point_segment_distance(px, py, v2x, v2y, v0x + FD_EPS_V, v0y);
            float d_bx_neg = point_segment_distance(px, py, v2x, v2y, v0x - FD_EPS_V, v0y);
            dd_dbx = (d_bx_pos - d_bx_neg) / (2.0f * FD_EPS_V);

            float d_by_pos = point_segment_distance(px, py, v2x, v2y, v0x, v0y + FD_EPS_V);
            float d_by_neg = point_segment_distance(px, py, v2x, v2y, v0x, v0y - FD_EPS_V);
            dd_dby = (d_by_pos - d_by_neg) / (2.0f * FD_EPS_V);

            atomicAdd(&grad_verts[base2 + 0], dL_dd_edge * dd_dax);
            atomicAdd(&grad_verts[base2 + 1], dL_dd_edge * dd_day);
            atomicAdd(&grad_verts[base0 + 0], dL_dd_edge * dd_dbx);
            atomicAdd(&grad_verts[base0 + 1], dL_dd_edge * dd_dby);
        }
    }
}

// ---- C++ launcher ----

torch::Tensor rasterize_backward_cuda(
    torch::Tensor grad_image,  // (B,1,H,W)
    torch::Tensor verts,       // (B,V,3)
    torch::Tensor faces,
    int image_size) {

    const int B = verts.size(0);
    const int V = verts.size(1);
    const int F = faces.size(0);

    auto grad_verts = torch::zeros_like(verts);

    const int H = grad_image.size(2);
    const int W = grad_image.size(3);
    TORCH_CHECK(H == image_size && W == image_size,
                "grad_image spatial size must match image_size.");

    dim3 block(16, 16, 1);
    dim3 grid(
        (image_size + block.x - 1) / block.x,
        (image_size + block.y - 1) / block.y,
        B
    );

    rasterize_backward_kernel<<<grid, block>>>(
        grad_image.data_ptr<float>(),
        verts.data_ptr<float>(),
        faces.data_ptr<int64_t>(),
        B,
        V,
        F,
        image_size,
        grad_verts.data_ptr<float>());

    return grad_verts;
}
