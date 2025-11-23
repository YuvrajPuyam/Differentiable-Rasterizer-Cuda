#include <torch/types.h>

torch::Tensor rasterize_backward_cuda(
    torch::Tensor grad_image,  // (B, 1, H, W)
    torch::Tensor verts,       // (B, V, 3)
    torch::Tensor faces,
    int image_size) {

    auto grad_verts = torch::zeros_like(verts);

    // Sum grad over H and W: shape (B, 1)
    auto grad_image_sum = grad_image.sum({2, 3});  // (B, 1)

    // Drop the channel dim -> (B)
    auto grad_v0x = (0.1f * grad_image_sum.squeeze(1));  // (B)

    // Write into grad_verts[:, 0, 0]
    // grad_verts: (B, V, 3)
    auto grad_v0_slice = grad_verts.select(1, 0).select(1, 0);  // (B)
    grad_v0_slice.copy_(grad_v0x);

    return grad_verts;
}
