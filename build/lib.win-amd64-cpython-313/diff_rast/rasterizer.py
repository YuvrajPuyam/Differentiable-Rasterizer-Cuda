# rasterizer.py
import torch
import torch.nn as nn
from torch.autograd import Function
import diff_rast._C as _C  # extension
import math



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

# =========================
# 3D camera + projection utils
# =========================


def look_at(eye: torch.Tensor,
            center: torch.Tensor,
            up: torch.Tensor) -> torch.Tensor:
    """
    Build a simple look-at view matrix.
    eye, center, up: (3,)
    Returns: (4,4) view matrix.
    """
    # Ensure 1D tensors
    eye = eye.view(-1)
    center = center.view(-1)
    up = up.view(-1)

    f = center - eye
    f = f / (f.norm() + 1e-8)

    u = up / (up.norm() + 1e-8)
    s = torch.cross(f, u)
    s = s / (s.norm() + 1e-8)
    u = torch.cross(s, f)

    M = torch.eye(4, device=eye.device, dtype=eye.dtype)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[:3, 3] = -torch.matmul(M[:3, :3], eye)
    return M


def perspective(fov_deg: float,
                aspect: float,
                near: float,
                far: float,
                device: torch.device,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Simple perspective projection matrix (OpenGL-style).
    """
    fov = fov_deg * math.pi / 180.0
    f = 1.0 / math.tan(fov / 2.0)

    M = torch.zeros((4, 4), device=device, dtype=dtype)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2.0 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


def project_verts_world_to_ndc(verts_world: torch.Tensor,
                               view: torch.Tensor,
                               proj: torch.Tensor) -> torch.Tensor:
    """
    verts_world: (B, V, 3)
    view, proj: (4,4)
    Returns verts_ndc: (B, V, 3) in NDC space [-1,1].
    """
    B, V, _ = verts_world.shape
    device = verts_world.device
    dtype = verts_world.dtype

    ones = torch.ones((B, V, 1), device=device, dtype=dtype)
    verts_h = torch.cat([verts_world, ones], dim=-1)  # (B,V,4)

    vp = proj @ view  # (4,4)
    # (B,V,4) @ (4,4)^T -> (B,V,4)
    verts_clip = torch.matmul(verts_h, vp.t())

    w = verts_clip[..., 3:4]              # (B,V,1)
    verts_ndc = verts_clip[..., :3] / (w + 1e-8)
    return verts_ndc


# =========================
# High-level 3D rasterizer
# =========================

class SoftRasterizer3D(torch.nn.Module):
    """
    World-space 3D wrapper:
      - takes verts_world (B,V,3)
      - applies view + projection in PyTorch
      - calls CUDA NDC rasterizer under the hood
    """

    def __init__(self,
                 image_size: int = 128,
                 use_fd: bool = False,
                 fov_deg: float = 45.0,
                 aspect: float = 1.0,
                 near: float = 0.1,
                 far: float = 10.0):
        super().__init__()
        self.image_size = image_size
        self.use_fd = use_fd
        self.fov_deg = fov_deg
        self.aspect = aspect
        self.near = near
        self.far = far

    def forward(self,
                verts_world: torch.Tensor,
                faces: torch.Tensor,
                eye,
                center=None,
                up=None) -> torch.Tensor:
        """
        verts_world: (B,V,3)
        faces: (F,3)
        eye, center, up: (3,) tensors or 3-element lists
        """
        device = verts_world.device
        dtype = verts_world.dtype

        eye = torch.as_tensor(eye, device=device, dtype=dtype)
        if center is None:
            center = torch.zeros(3, device=device, dtype=dtype)
        else:
            center = torch.as_tensor(center, device=device, dtype=dtype)

        if up is None:
            up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        else:
            up = torch.as_tensor(up, device=device, dtype=dtype)

        view = look_at(eye, center, up)
        proj = perspective(
            self.fov_deg,
            self.aspect,
            self.near,
            self.far,
            device=device,
            dtype=dtype,
        )

        verts_ndc = project_verts_world_to_ndc(verts_world, view, proj)

        # Reuse your existing NDC-level autograd function
        return SoftRasterizerFunction.apply(
            verts_ndc, faces, self.image_size, self.use_fd
        )
