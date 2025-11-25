# diff_rast/losses.py
import torch
import torch.nn.functional as F

def build_edge_index(faces: torch.Tensor) -> torch.Tensor:
    """
    Build unique undirected edge list from triangle faces.

    faces: (F,3) long tensor
    returns: (E,2) long tensor of vertex index pairs
    """
    f = faces.detach().cpu().numpy()
    edges = set()
    for tri in f:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            if a > b:
                a, b = b, a
            edges.add((a, b))
    edges = torch.tensor(list(edges), dtype=torch.long)
    return edges


def precompute_edge_data(verts_ref: torch.Tensor,
                         faces: torch.Tensor):
    """
    Precompute edges and their reference lengths, used for edge-length regularization.

    verts_ref: (V,3) in world space (after normalization)
    faces    : (F,3)

    returns:
      edges: (E,2) long tensor
      ref_lengths: (E,) float tensor
    """
    edges = build_edge_index(faces)        # (E,2) on CPU (for now)
    v0 = verts_ref[edges[:, 0]]            # (E,3)
    v1 = verts_ref[edges[:, 1]]            # (E,3)
    ref_lengths = (v0 - v1).norm(dim=1)    # (E,)
    return edges, ref_lengths


def l2_vertex_reg(verts: torch.Tensor,
                  ref_verts: torch.Tensor) -> torch.Tensor:
    """
    Simple L2 vertex regularization:
      ||verts - ref_verts||^2 mean.

    verts, ref_verts: (B,V,3) or (V,3); shapes must match.
    """
    return (verts - ref_verts).pow(2).mean()


def edge_smoothness_loss(verts: torch.Tensor,
                         edges: torch.Tensor) -> torch.Tensor:
    """
    Edge-based Laplacian-like smoothness:
      average squared difference between vertices of each edge.

    verts: (B,V,3) or (V,3)
    edges: (E,2) long tensor of vertex indices
    """
    if verts.dim() == 3:
        verts = verts[0]  # (V,3)

    v_i = verts[edges[:, 0]]  # (E,3)
    v_j = verts[edges[:, 1]]  # (E,3)

    return (v_i - v_j).pow(2).sum(dim=1).mean()


def edge_length_loss(verts: torch.Tensor,
                     edges: torch.Tensor,
                     ref_lengths: torch.Tensor) -> torch.Tensor:
    """
    Edge-length preservation loss:
      average squared deviation from reference edge lengths.

    verts      : (B,V,3) or (V,3)
    edges      : (E,2)
    ref_lengths: (E,) reference edge lengths (same ordering as edges)
    """
    if verts.dim() == 3:
        verts = verts[0]  # (V,3)

    v0 = verts[edges[:, 0]]     # (E,3)
    v1 = verts[edges[:, 1]]     # (E,3)
    lengths = (v0 - v1).norm(dim=1)  # (E,)

    return ((lengths - ref_lengths) ** 2).mean()


def multiview_silhouette_mse(renderer3d,
                             verts_world: torch.Tensor,
                             faces: torch.Tensor,
                             cameras,
                             targets) -> torch.Tensor:
    """
    Sum MSE silhouette loss over multiple views.

    renderer3d : SoftRasterizer3D-like module
    verts_world: (B,V,3)
    faces      : (F,3)
    cameras    : list of (eye, center, up)
    targets    : list of target silhouette tensors, same shape as renderer output
    """
    total = 0.0
    for (eye, center, up), target_k in zip(cameras, targets):
        pred_k = renderer3d(verts_world, faces, eye, center, up)
        mse_k = ((pred_k - target_k) ** 2).mean()
        total = total + mse_k
    return total

def multiview_rgb_mse(renderer, verts_world, faces, colors, cameras, targets_rgb):
    """
    renderer: SoftRasterizer3D instance
    verts_world: (B, V, 3)
    faces:       (F, 3)
    colors:      (B, V, 3)
    cameras: list of (eye, center, up) tuples
    targets_rgb: list of target RGB images, each (B, 3, H, W)
    """
    total = 0.0
    for (eye, center, up), tgt_rgb in zip(cameras, targets_rgb):
        _, pred_rgb = renderer(verts_world, faces, colors, eye, center, up)
        total = total + F.mse_loss(pred_rgb, tgt_rgb)
    return total