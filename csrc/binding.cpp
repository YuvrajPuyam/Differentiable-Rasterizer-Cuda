// csrc/binding.cpp

#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> build_bins_cuda(
    at::Tensor verts,
    at::Tensor faces,
    int image_size,
    int tile_size,
    int max_per_tile);

std::vector<at::Tensor> rasterize_forward_cuda(
    at::Tensor verts,
    at::Tensor faces,
    at::Tensor colors,
    at::Tensor tile_counts,
    at::Tensor tile_faces,
    int image_size,
    int tile_size,
    int max_per_tile);

at::Tensor rasterize_backward_cuda(
    at::Tensor grad_sil,
    at::Tensor grad_rgb,
    at::Tensor verts,
    at::Tensor faces,
    at::Tensor colors,
    at::Tensor tile_counts,
    at::Tensor tile_faces,
    int image_size,
    int tile_size,
    int max_per_tile);

std::vector<at::Tensor> build_bins(
    at::Tensor verts,
    at::Tensor faces,
    int image_size,
    int tile_size,
    int max_per_tile
) {
    return build_bins_cuda(verts, faces, image_size, tile_size, max_per_tile);
}

std::vector<at::Tensor> rasterize_forward(
    at::Tensor verts,
    at::Tensor faces,
    at::Tensor colors,
    at::Tensor tile_counts,
    at::Tensor tile_faces,
    int image_size,
    int tile_size,
    int max_per_tile
) {
    return rasterize_forward_cuda(verts, faces, colors, tile_counts, tile_faces,
                                  image_size, tile_size, max_per_tile);
}

at::Tensor rasterize_backward(
    at::Tensor grad_sil,
    at::Tensor grad_rgb,
    at::Tensor verts,
    at::Tensor faces,
    at::Tensor colors,
    at::Tensor tile_counts,
    at::Tensor tile_faces,
    int image_size,
    int tile_size,
    int max_per_tile
) {
    return rasterize_backward_cuda(grad_sil, grad_rgb, verts, faces, colors,
                                   tile_counts, tile_faces,
                                   image_size, tile_size, max_per_tile);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_bins", &build_bins,
          "Build tile bins (tile_counts, tile_faces)");

    m.def("rasterize_forward", &rasterize_forward,
          "Soft rasterizer forward (tiled) (silhouette + RGB)");

    m.def("rasterize_backward", &rasterize_backward,
          "Soft rasterizer backward (tiled) (silhouette + RGB -> verts)");
}
