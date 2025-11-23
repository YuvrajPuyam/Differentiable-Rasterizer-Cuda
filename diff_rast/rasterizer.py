# rasterizer.py
import torch
import torch.nn as nn
from torch.autograd import Function
import diff_rast._C as _C  # extension


class SoftRasterizerFunction(Function):
    @staticmethod
    def forward(ctx, verts, faces, image_size: int, use_fd: bool = False):
        image = _C.rasterize_forward(verts, faces, image_size)
        ctx.save_for_backward(verts, faces)
        ctx.image_size = image_size
        ctx.use_fd = use_fd
        return image

    @staticmethod
    def backward(ctx, grad_image):
        verts, faces = ctx.saved_tensors
        image_size = ctx.image_size

        if ctx.use_fd:
            # ------------ finite-difference backward (slow, but "truthy") ------------
            eps = 1e-3
            B, V, C = verts.shape
            grad_verts = torch.zeros_like(verts)

            for b in range(B):
                for v in range(V):
                    for c in range(2):  # x,y only
                        verts_pos = verts.clone()
                        verts_pos[b, v, c] += eps
                        img_pos = _C.rasterize_forward(verts_pos, faces, image_size)

                        verts_neg = verts.clone()
                        verts_neg[b, v, c] -= eps
                        img_neg = _C.rasterize_forward(verts_neg, faces, image_size)

                        dS_dv = (img_pos - img_neg) / (2.0 * eps)
                        g = (grad_image * dS_dv).sum()
                        grad_verts[b, v, c] = g
        else:
            # ------------ fast CUDA backward (currently dummy, we will improve) ------------
            grad_verts = _C.rasterize_backward(
                grad_image.contiguous(), verts, faces, image_size
            )

        return grad_verts, None, None, None


class SoftRasterizer(nn.Module):
    def __init__(self, image_size: int = 128, use_fd: bool = True):
        super().__init__()
        self.image_size = image_size
        self.use_fd = use_fd

    def forward(self, verts, faces):
        return SoftRasterizerFunction.apply(verts, faces, self.image_size, self.use_fd)
