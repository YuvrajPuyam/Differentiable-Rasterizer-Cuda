#include <torch/types.h> 
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void rasterize_forward_kernel(
    const float* verts,    // (B, V, 3)
    const int64_t* faces,  // (F, 3) not used yet
    int B,
    int V,
    int image_size,
    float* out) {          // (B, 1, H, W)
    
    int b = blockIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || y >= image_size || x >= image_size) return;

    // NCHW with C=1: [B, 1, H, W]
    int idx = b * image_size * image_size + y * image_size + x;

    // Simple pattern: normalized coords
    float fx = float(x) / float(image_size - 1);
    float fy = float(y) / float(image_size - 1);

    // *** NEW: make image depend on first vertex x-coordinate: verts[b, 0, 0] ***
    // verts is (B, V, 3), flattened as [B * V * 3]
    float v0x = verts[b * V * 3 + 0];  // (0 * 3 + 0) = 0

    // Out now depends on v0x. 0.1 is just a small scaling factor.
    out[idx] = 0.5f * fx + 0.5f * fy + 0.1f * v0x;
}

} // anonymous namespace

torch::Tensor rasterize_forward_cuda(
    torch::Tensor verts,
    torch::Tensor faces,
    int image_size) {

    const int B = verts.size(0);
    const int V = verts.size(1);

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
        image_size,
        image.data_ptr<float>()
    );

    return image;
}
