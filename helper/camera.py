import torch
import math


def make_orbit_cameras(n_views, radius, device, dtype, elevation_deg=10.0):
    cams = []
    elev = elevation_deg * math.pi / 180.0

    for i in range(n_views):
        yaw = 2.0 * math.pi * i / n_views

        x = radius * math.sin(yaw) * math.cos(elev)
        z = radius * math.cos(yaw) * math.cos(elev)
        y = radius * math.sin(elev)

        eye = torch.tensor([x, y, z], device=device, dtype=dtype)
        center = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

        cams.append((eye, center, up))

    return cams

def look_at(verts_world, eye, center, up):
    z = torch.nn.functional.normalize(center - eye, dim=-1)
    x = torch.nn.functional.normalize(torch.cross(z, up, dim=-1), dim=-1)
    y = torch.cross(x, z, dim=-1)
    R = torch.stack([x, y, -z], dim=-2)
    t = -torch.bmm(R, eye.unsqueeze(-1))
    verts = verts_world.transpose(1, 2)
    verts_cam = torch.bmm(R, verts) + t
    verts_cam = verts_cam.transpose(1, 2)
    return verts_cam

def perspective(verts_cam, fov_deg, aspect, near, far):
    fov = math.radians(fov_deg)
    f = 1.0 / math.tan(fov / 2.0)
    P = torch.zeros((4, 4), device=verts_cam.device, dtype=verts_cam.dtype)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (far+near)/(near-far)
    P[2, 3] = (2*far*near)/(near-far)
    P[3, 2] = -1
    
    B, V, _ = verts_cam.shape
    verts_h = torch.cat([verts_cam, torch.ones(B, V, 1, device=verts_cam.device, dtype=verts_cam.dtype)], dim=-1)
    clip = torch.matmul(verts_h, P.t().unsqueeze(0))
    ndc = clip[..., :3] / (clip[..., 3:4] + 1e-8)
    return ndc

def project_world_to_ndc(verts_world, eye, center, up, fov=45.0, aspect=1.0, near=0.1, far=10.0):
    if eye.dim() == 1:
        eye = eye.unsqueeze(0)
        center = center.unsqueeze(0)
        up = up.unsqueeze(0)
    verts_cam = look_at(verts_world, eye, center, up)
    return perspective(verts_cam, fov, aspect, near, far)