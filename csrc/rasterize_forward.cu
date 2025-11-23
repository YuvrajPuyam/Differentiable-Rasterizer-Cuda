// csrc/rasterize_forward.cu
//
// Forward CUDA kernel for multi-triangle soft silhouettes (SoftRas-style)
//
// Notes:
// - verts: (B, V, 3) in NDC space [-1, 1]
// - faces: (F, 3) with int64 indices
// - output: (B, 1, H, W), values in (0, 1)
// - Silhouette per pixel: S = 1 - prod_f (1 - alpha_pf)
//   where alpha_pf = sigmoid(-d_pixels / gamma)
//
// We ignore z here (pure orthographic silhouette in screen space).

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ inline float edge_function(
    float ax, float ay,
    float bx, float by,
    float px, float py) {
    // 2D cross product (B - A) x (P - A)
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

    float vv = vx * vx + vy * vy + 1e-8f;  // avoid divide-by-zero
    float t = (vx * wx + vy * wy) / vv;
    t = fmaxf(0.0f, fminf(1.0f, t));       // clamp to segment

    float cx = ax + t * vx;
    float cy = ay + t * vy;

    float dx = px - cx;
    float dy = py - cy;
    return sqrtf(dx * dx + dy * dy);
}

__global__ void rasterize_forward_kernel(
    const float* __restrict__ verts,   // (B, V, 3)
    const int64_t* __restrict__ faces, // (F, 3)
    int B,
    int V,
    int F,
    int image_size,
    float* __restrict__ out) {         // (B, 1, H, W) flattened to (B * H * W)

    int b = blockIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || y >= image_size || x >= image_size) {
        return;
    }

    int pixel_index = b * image_size * image_size + y * image_size + x;

    // Map pixel center to NDC [-1, 1]
    // (x + 0.5) / image_size in [0,1] → scale & shift to [-1,1]
    float px = 2.0f * ((x + 0.5f) / float(image_size)) - 1.0f;
    float py = 2.0f * ((y + 0.5f) / float(image_size)) - 1.0f;

    // 1 pixel in NDC space
    const float PIXEL_SIZE   = 2.0f / float(image_size);

    // SoftRas-like gamma, in *pixel units*:
    // smaller  -> sharper edges (more binary)
    // larger   -> blurrier edges
    const float GAMMA_PIXELS = 0.5f;

    const float EPS       = 1e-6f;
    const float MIN_BG    = 1e-6f;   // early-out threshold for background prob

    // Background probability: prod_f (1 - alpha_pf)
    float bg_prob = 1.0f;

    // Loop over faces for this pixel
    for (int f = 0; f < F; ++f) {
        // Face vertex indices
        int64_t i0 = faces[f * 3 + 0];
        int64_t i1 = faces[f * 3 + 1];
        int64_t i2 = faces[f * 3 + 2];

        // Safety check for indices
        if (i0 < 0 || i0 >= V || i1 < 0 || i1 >= V || i2 < 0 || i2 >= V) {
            continue;
        }

        // Fetch vertices for batch b
        // verts layout: [B, V, 3]
        int base0 = (b * V + int(i0)) * 3;
        int base1 = (b * V + int(i1)) * 3;
        int base2 = (b * V + int(i2)) * 3;

        float v0x = verts[base0 + 0];
        float v0y = verts[base0 + 1];
        float v1x = verts[base1 + 0];
        float v1y = verts[base1 + 1];
        float v2x = verts[base2 + 0];
        float v2y = verts[base2 + 1];

        // Edge functions for inside test
        float w0 = edge_function(v1x, v1y, v2x, v2y, px, py);
        float w1 = edge_function(v2x, v2y, v0x, v0y, px, py);
        float w2 = edge_function(v0x, v0y, v1x, v1y, px, py);

        // Triangle signed area
        float area = edge_function(v0x, v0y, v1x, v1y, v2x, v2y);
        if (area == 0.0f) {
            // Degenerate triangle, skip
            continue;
        }

        bool same_sign = (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) ||
                         (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
        bool inside = same_sign;

        // Distance to triangle edges (unsigned)
        float d0 = point_segment_distance(px, py, v0x, v0y, v1x, v1y);
        float d1 = point_segment_distance(px, py, v1x, v1y, v2x, v2y);
        float d2 = point_segment_distance(px, py, v2x, v2y, v0x, v0y);
        float min_dist = fminf(d0, fminf(d1, d2));

        // Signed distance in NDC units
        float signed_d = inside ? -min_dist : min_dist;

        // Convert to pixel units
        float d_pixels = signed_d / PIXEL_SIZE;

        // SoftRas-like silhouette coverage:
        // alpha = sigmoid(-d / gamma) = 1 / (1 + exp(d / gamma))
        float alpha = 1.0f / (1.0f + expf(d_pixels / GAMMA_PIXELS));

        // Clamp alpha to avoid exact 0 or 1
        if (alpha < EPS)         alpha = EPS;
        if (alpha > 1.0f - EPS)  alpha = 1.0f - EPS;

        float one_minus_alpha = 1.0f - alpha;
        bg_prob *= one_minus_alpha;

        // If background prob ~0, pixel is effectively foreground already
        if (bg_prob < MIN_BG) {
            bg_prob = 0.0f;
            break;
        }
    }

    // Final soft silhouette: S = 1 - background probability
    float S = 1.0f - bg_prob;
    out[pixel_index] = S;
}

} // anonymous namespace

torch::Tensor rasterize_forward_cuda(
    torch::Tensor verts,
    torch::Tensor faces,
    int image_size) {

    const int B = verts.size(0);
    const int V = verts.size(1);
    const int F = faces.size(0);

    auto options = verts.options().dtype(torch::kFloat32);
    auto image = torch::zeros({B, 1, image_size, image_size}, options);

    dim3 block(16, 16, 1);
    dim3 grid(
        (image_size + block.x - 1) / block.x,
        (image_size + block.y - 1) / block.y,
        B
    );

    rasterize_forward_kernel<<<grid, block>>>(
        verts.data_ptr<float>(),
        faces.data_ptr<int64_t>(),
        B,
        V,
        F,
        image_size,
        image.data_ptr<float>());

    return image;
}
