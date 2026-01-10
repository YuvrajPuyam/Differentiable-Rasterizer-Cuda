import torch
from torch.autograd import Function
from . import _C  # compiled CUDA extension


class SoftRasterizerFunction(Function):
    @staticmethod
    def forward(ctx, verts_ndc, faces, colors, image_size, use_fd=False):
        """
        verts_ndc: (B, V, 3) in NDC
        faces:     (F, 3) long
        colors:    (B, V, 3) RGB in [0,1]
        image_size: int
        use_fd: kept for API compatibility (not used in CUDA)
        """
        sil, rgb = _C.rasterize_forward(verts_ndc, faces, colors, int(image_size))

        # Save for backward
        ctx.save_for_backward(verts_ndc, faces, colors)
        ctx.image_size = int(image_size)
        ctx.use_fd = use_fd  # not used in CUDA, but kept for interface

        # Two outputs: silhouette + rgb
        return sil, rgb

    @staticmethod
    def backward(ctx, grad_sil, grad_rgb):
        """
        grad_sil: (B, 1, H, W)
        grad_rgb: (B, 3, H, W) or None if upstream ignored RGB
        """
        verts_ndc, faces, colors = ctx.saved_tensors
        image_size = ctx.image_size

        B = verts_ndc.size(0)
        H = W = image_size

        # If no gradients were provided for one of the outputs, use zeros
        if grad_sil is None:
            grad_sil = verts_ndc.new_zeros(B, 1, H, W)
        if grad_rgb is None:
            grad_rgb = verts_ndc.new_zeros(B, 3, H, W)

        grad_sil = grad_sil.contiguous()
        grad_rgb = grad_rgb.contiguous()

        # CUDA backward: (grad_sil, grad_rgb) -> grad_verts
        grad_verts = _C.rasterize_backward(
            grad_sil, grad_rgb, verts_ndc, faces, colors, int(image_size)
        )

        # No gradients for faces/colors/image_size/use_fd (for now)
        grad_faces = None
        grad_colors = None
        grad_image_size = None
        grad_use_fd = None

        return grad_verts, grad_faces, grad_colors, grad_image_size, grad_use_fd
