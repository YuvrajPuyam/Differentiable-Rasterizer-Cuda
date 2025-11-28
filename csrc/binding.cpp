// csrc/binding.cpp

#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> rasterize_forward_cuda(
    at::Tensor verts,
    at::Tensor faces,
    at::Tensor colors,
    int image_size);

at::Tensor rasterize_backward_cuda(
    at::Tensor grad_sil,
    at::Tensor grad_rgb,
    at::Tensor verts,
    at::Tensor faces,
    at::Tensor colors,
    int image_size);


std::vector<at::Tensor> rasterize_forward(
    at::Tensor verts,
    at::Tensor faces,
    at::Tensor colors,
    int image_size)
{
    return rasterize_forward_cuda(verts, faces, colors, image_size);
}

at::Tensor rasterize_backward(
    at::Tensor grad_sil,
    at::Tensor grad_rgb,
    at::Tensor verts,
    at::Tensor faces,
    at::Tensor colors,
    int image_size)
{
    return rasterize_backward_cuda(grad_sil, grad_rgb, verts, faces, colors, image_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_forward", &rasterize_forward,
          "Soft rasterizer forward (silhouette + RGB)");

    m.def("rasterize_backward", &rasterize_backward,
          "Soft rasterizer backward (silhouette + RGB -> verts)");
}
