import torch
import torch.nn as nn
from torch.autograd import Function

import diff_rast._C as _C  # built from our extension


class SoftRasterizerFunction(Function):
    @staticmethod
    def forward(ctx, verts, faces, image_size: int):
        """
        verts: (B, V, 3) float32
        faces: (F, 3) int32 or int64
        """
        # For checkpoint 1, we just call a dummy CUDA op that returns zeros
        image = _C.rasterize_forward(verts, faces, image_size)
        # Save things for backward if needed later
        ctx.save_for_backward(verts, faces)
        ctx.image_size = image_size
        return image

    @staticmethod
    def backward(ctx, grad_image):
        verts, faces = ctx.saved_tensors
        image_size = ctx.image_size

        # Dummy backward: return zeros for vertex gradients
        grad_verts = _C.rasterize_backward(
            grad_image.contiguous(), verts, faces, image_size
        )
        # faces and image_size are not differentiable
        return grad_verts, None, None


class SoftRasterizer(nn.Module):
    def __init__(self, image_size: int = 128):
        super().__init__()
        self.image_size = image_size

    def forward(self, verts, faces):
        return SoftRasterizerFunction.apply(verts, faces, self.image_size)
