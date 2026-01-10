import torch
from torch.autograd import Function
import diff_rast._C as _C

_TILE_SIZE = 16
_MAX_PER_TILE = 512

class SoftRasterizerFunction(Function):
    @staticmethod
    def forward(ctx, verts_ndc, faces, colors, image_size, use_fd=False):
        image_size = int(image_size)

        tile_counts, tile_faces = _C.build_bins(
            verts_ndc, faces, image_size, _TILE_SIZE, _MAX_PER_TILE
        )

        sil, rgb = _C.rasterize_forward(
            verts_ndc, faces, colors,
            tile_counts, tile_faces,
            image_size, _TILE_SIZE, _MAX_PER_TILE
        )

        ctx.save_for_backward(verts_ndc, faces, colors, tile_counts, tile_faces)
        ctx.image_size = image_size
        ctx.use_fd = use_fd
        return sil, rgb

    @staticmethod
    def backward(ctx, grad_sil, grad_rgb):
        verts_ndc, faces, colors, tile_counts, tile_faces = ctx.saved_tensors
        image_size = ctx.image_size

        B = verts_ndc.size(0)
        H = W = image_size

        if grad_sil is None:
            grad_sil = verts_ndc.new_zeros(B, 1, H, W)
        if grad_rgb is None:
            grad_rgb = verts_ndc.new_zeros(B, 3, H, W)

        grad_sil = grad_sil.contiguous()
        grad_rgb = grad_rgb.contiguous()

        grad_verts = _C.rasterize_backward(
            grad_sil, grad_rgb,
            verts_ndc, faces, colors,
            tile_counts, tile_faces,
            image_size, _TILE_SIZE, _MAX_PER_TILE
        )

        return grad_verts, None, None, None, None
