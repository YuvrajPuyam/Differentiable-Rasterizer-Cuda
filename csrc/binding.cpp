#include <torch/extension.h>


#pragma warning(disable:4005)
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

// Declarations of our CUDA launchers
torch::Tensor rasterize_forward_cuda(
    torch::Tensor verts,
    torch::Tensor faces,
    int image_size);

torch::Tensor rasterize_backward_cuda(
    torch::Tensor grad_image,
    torch::Tensor verts,
    torch::Tensor faces,
    int image_size);

// C++ wrappers that check types and call CUDA
torch::Tensor rasterize_forward(
    torch::Tensor verts,
    torch::Tensor faces,
    int image_size) {
    
    // Basic sanity checks
    TORCH_CHECK(verts.is_cuda(), "verts must be a CUDA tensor");
    TORCH_CHECK(faces.is_cuda(), "faces must be a CUDA tensor");
    TORCH_CHECK(verts.dim() == 3 && verts.size(2) == 3, "verts should be (B, V, 3)");

    return rasterize_forward_cuda(verts, faces, image_size);
}

torch::Tensor rasterize_backward(
    torch::Tensor grad_image,
    torch::Tensor verts,
    torch::Tensor faces,
    int image_size) {
    
    TORCH_CHECK(grad_image.is_cuda(), "grad_image must be CUDA");
    TORCH_CHECK(verts.is_cuda(), "verts must be CUDA");
    TORCH_CHECK(faces.is_cuda(), "faces must be CUDA");

    return rasterize_backward_cuda(grad_image, verts, faces, image_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_forward", &rasterize_forward, "Dummy rasterizer forward (CUDA)");
    m.def("rasterize_backward", &rasterize_backward, "Dummy rasterizer backward (CUDA)");
}