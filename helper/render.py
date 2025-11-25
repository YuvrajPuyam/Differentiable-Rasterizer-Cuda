import torch
from diff_rast.rasterizer import SoftRasterizerFunction
from helper.camera import project_world_to_ndc

def make_vertex_colors_from_world(verts_world):
    # Procedural coloring based on position (XYZ -> RGB)
    v = verts_world.clone()
    v = v - v.mean(dim=0, keepdim=True)
    v = v / (v.abs().max() + 1e-8)
    c = (v + 1.0) * 0.5
    r = c[:, 0]
    g = 0.5 * c[:, 1] + 0.5 * (1.0 - c[:, 2])
    b = 0.5 * c[:, 2] + 0.5 * c[:, 0]
    return torch.stack([r, g, b], dim=-1).clamp(0,1)

def render_sil_rgb(verts_world_b, faces, colors_b, eye, center, up, image_size, fov=45.0):
    verts_ndc = project_world_to_ndc(verts_world_b, eye, center, up, fov=fov)
    # sigma=1e-4 is standard for SoftRas "sharp" rendering
    return SoftRasterizerFunction.apply(verts_ndc, faces, colors_b, image_size, False)