import torch
from torch.autograd import Function
import diff_rast._C as _C # compiled CUDA extension


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
        grad_rgb: (B, 3, H, W)
        """
        verts_ndc, faces, colors = ctx.saved_tensors
        image_size = ctx.image_size

        B = verts_ndc.size(0)
        H = W = image_size

        # If no gradients come from upstream for an output,
        # autograd passes None → replace with zeros
        if grad_sil is None:
            grad_sil = verts_ndc.new_zeros(B, 1, H, W)
        if grad_rgb is None:
            grad_rgb = verts_ndc.new_zeros(B, 3, H, W)

        grad_sil = grad_sil.contiguous()
        grad_rgb = grad_rgb.contiguous()

        # Our new CUDA backward
        grad_verts = _C.rasterize_backward(
            grad_sil,          # (B,1,H,W)
            grad_rgb,          # (B,3,H,W)
            verts_ndc,         # (B,V,3)
            faces,             # (F,3)
            colors,            # (B,V,3)
            int(image_size)
        )

        # No grad for faces/colors/image_size/use_fd
        return grad_verts, None, None, None, None
