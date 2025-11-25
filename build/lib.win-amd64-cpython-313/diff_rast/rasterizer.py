# diff_rast/rasterizer.py

import torch
from torch import nn
from torch.autograd import Function

from . import _C  # C++/CUDA extension built from setup.py


def look_at(eye, center, up):
    """
    Simple look-at view matrix (4x4) in torch.
    eye, center, up: (..., 3)
    """
    eye = eye.float()
    center = center.float()
    up = up.float()

    z = torch.nn.functional.normalize(eye - center, dim=-1)        # forward
    x = torch.nn.functional.normalize(torch.cross(up, z, dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)

    # 3x3 rotation
    R = torch.stack([x, y, z], dim=-2)  # (..., 3, 3)

    # translation
    t = -torch.matmul(R, eye.unsqueeze(-1)).squeeze(-1)  # (..., 3)

    # assemble 4x4
    batch_shape = eye.shape[:-1]
    M = torch.eye(4, device=eye.device).expand(*batch_shape, 4, 4).clone()
    M[..., :3, :3] = R
    M[..., :3, 3] = t
    return M


def perspective(fov_y, aspect, near, far, device):
    """
    Simple perspective projection matrix (4x4).
    fov_y in radians.
    """
    f = 1.0 / torch.tan(torch.tensor(fov_y, device=device) / 2.0)
    M = torch.zeros(4, 4, device=device)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2.0 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


def project_world_to_ndc(verts_world, eye, center, up,
                         fov_y=60.0, near=0.1, far=10.0, aspect=1.0):
    """
    verts_world: (B, V, 3)
    eye, center, up: (B, 3)
    Returns verts_ndc: (B, V, 3) in [-1, 1]
    """
    device = verts_world.device
    B, V, _ = verts_world.shape

    view = look_at(eye, center, up)            # (B, 4, 4)
    proj = perspective(fov_y * torch.pi / 180.0, aspect, near, far, device)
    proj = proj.expand(B, 4, 4)

    M = torch.matmul(proj, view)               # (B, 4, 4)

    verts_h = torch.cat([verts_world, torch.ones(B, V, 1, device=device)], dim=-1)
    verts_clip = torch.matmul(M, verts_h.transpose(1, 2))  # (B, 4, V)
    verts_clip = verts_clip.transpose(1, 2)                # (B, V, 4)

    xyz = verts_clip[..., :3]
    w = verts_clip[..., 3:4]
    verts_ndc = xyz / (w + 1e-8)
    return verts_ndc


class SoftRasterizerFunction(Function):
    @staticmethod
    def forward(ctx, verts_ndc, faces, colors, image_size, use_fd=False):
        """
        verts_ndc: (B, V, 3) on CUDA
        faces:     (F, 3) int64 on CUDA
        colors:    (B, V, 3) on CUDA, [0, 1]
        """
        assert verts_ndc.is_cuda and faces.is_cuda and colors.is_cuda

        # Call C++/CUDA extension: returns [sil, rgb]
        sil, rgb = _C.rasterize_forward(verts_ndc, faces, colors, image_size)

        # Save for backward (only verts/faces are used; colors ignored)
        ctx.image_size = image_size
        ctx.use_fd = use_fd
        ctx.save_for_backward(verts_ndc, faces, colors)

        # We return both silhouette and rgb
        return sil, rgb

    @staticmethod
    def backward(ctx, grad_sil, grad_rgb):
        """
        grad_sil: (B, 1, H, W)
        grad_rgb: (B, 3, H, W)  (ignored in this cheap Stage 2)
        """
        verts_ndc, faces, colors = ctx.saved_tensors
        image_size = ctx.image_size

        # Only propagate geometry gradients from silhouette loss.
        grad_verts = _C.rasterize_backward(
            grad_sil.contiguous(), verts_ndc, faces, image_size
        )

        # No gradients for faces or colors at this stage
        grad_faces = None
        grad_colors = None

        # No gradients for image_size and use_fd
        return grad_verts, grad_faces, grad_colors, None, None


class SoftRasterizer3D(nn.Module):
    def __init__(self, image_size=128, use_fd=False):
        super().__init__()
        self.image_size = image_size
        self.use_fd = use_fd

    def forward(self, verts_world, faces, colors, eye,
                center=None, up=None):
        """
        verts_world: (B, V, 3)
        faces:      (F, 3) (CPU or CUDA int64)
        colors:     (B, V, 3) in [0,1]
        eye:        (B, 3)
        center:     (B, 3) or None
        up:         (B, 3) or None
        """
        device = verts_world.device
        B, V, _ = verts_world.shape

        if center is None:
            center = torch.zeros(B, 3, device=device)
        if up is None:
            up = torch.tensor([[0.0, 1.0, 0.0]],
                              device=device).expand(B, 3)

        faces_cuda = faces.to(device=device, dtype=torch.int64)
        colors = colors.to(device=device)

        verts_ndc = project_world_to_ndc(
            verts_world, eye, center, up,
            fov_y=60.0, near=0.1, far=10.0, aspect=1.0
        )

        sil, rgb = SoftRasterizerFunction.apply(
            verts_ndc, faces_cuda, colors, self.image_size, self.use_fd
        )
        return sil, rgb
