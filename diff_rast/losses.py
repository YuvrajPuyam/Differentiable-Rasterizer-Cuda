import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========================================================================
#  1. Official SoftRas Losses (Optimized & Robust)
# =========================================================================

class LaplacianLoss(nn.Module):
    """
    Computes the Laplacian smoothing loss.
    This prevents vertices from flying apart and regularizes the mesh geometry.
    """
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        faces_np = faces.detach().cpu().numpy()
        laplacian[faces_np[:, 0], faces_np[:, 1]] = -1
        laplacian[faces_np[:, 1], faces_np[:, 0]] = -1
        laplacian[faces_np[:, 1], faces_np[:, 2]] = -1
        laplacian[faces_np[:, 2], faces_np[:, 1]] = -1
        laplacian[faces_np[:, 2], faces_np[:, 0]] = -1
        laplacian[faces_np[:, 0], faces_np[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            if laplacian[i, i] > 0:
                laplacian[i, :] /= laplacian[i, i]

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, x):
        # x: (B, V, 3)
        batch_size = x.size(0)
        x = torch.matmul(self.laplacian, x)
        dims = tuple(range(x.ndimension())[1:])
        x = x.pow(2).sum(dims)
        if self.average:
            return x.sum() / batch_size
        else:
            return x


class FlattenLoss(nn.Module):
    """
    Computes the "Flatness" or Normal Consistency loss.
    Penalizes the angle between neighboring faces to prevent spikes/crumpling.
    """
    def __init__(self, faces, average=False):
        super(FlattenLoss, self).__init__()
        self.nf = faces.size(0)
        self.average = average

        faces = faces.detach().cpu().numpy()
        
        # Find all unique edges
        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3], faces[:, [0, 2]]), axis=0))]))

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []

        # Find the two opposite vertices for every shared edge
        for v0, v1 in zip(v0s, v1s):
            count = 0
            for face in faces:
                if v0 in face and v1 in face:
                    v = np.copy(face)
                    v = v[v != v0]
                    v = v[v != v1]
                    if count == 0:
                        v2s.append(int(v[0]))
                        count += 1
                    else:
                        v3s.append(int(v[0]))
                        count += 1
            # Handle boundary edges (only 1 face) -> pad with v2 to avoid index errors
            if count < 2:
                v3s.append(v2s[-1]) 

        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')

        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

    def forward(self, vertices, eps=1e-6):
        # vertices: (B, V, 3)
        batch_size = vertices.size(0)

        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        # Calculate normals for the two faces sharing the edge
        a1 = v1s - v0s
        b1 = v2s - v0s
        a1l2 = a1.pow(2).sum(-1)
        b1l2 = b1.pow(2).sum(-1)
        a1l1 = (a1l2 + eps).sqrt()
        b1l1 = (b1l2 + eps).sqrt()
        ab1 = (a1 * b1).sum(-1)
        cos1 = ab1 / (a1l1 * b1l1 + eps)
        sin1 = (1 - cos1.pow(2) + eps).sqrt()
        c1 = a1 * (ab1 / (a1l2 + eps))[:, :, None]
        cb1 = b1 - c1
        cb1l1 = b1l1 * sin1

        a2 = v1s - v0s
        b2 = v3s - v0s
        a2l2 = a2.pow(2).sum(-1)
        b2l2 = b2.pow(2).sum(-1)
        a2l1 = (a2l2 + eps).sqrt()
        b2l1 = (b2l2 + eps).sqrt()
        ab2 = (a2 * b2).sum(-1)
        cos2 = ab2 / (a2l1 * b2l1 + eps)
        sin2 = (1 - cos2.pow(2) + eps).sqrt()
        c2 = a2 * (ab2 / (a2l2 + eps))[:, :, None]
        cb2 = b2 - c2
        cb2l1 = b2l1 * sin2

        # Cosine of angle between normals
        cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)

        dims = tuple(range(cos.ndimension())[1:])
        loss = (cos + 1).pow(2).sum(dims)
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss


# =========================================================================
#  2. Naive / Functional Losses (Kept for utilities)
# =========================================================================

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


def precompute_edge_data(verts_ref: torch.Tensor, faces: torch.Tensor):
    """
    Precompute edges and their reference lengths.
    """
    edges = build_edge_index(faces)        # (E,2) on CPU
    v0 = verts_ref[edges[:, 0]]            # (E,3)
    v1 = verts_ref[edges[:, 1]]            # (E,3)
    ref_lengths = (v0 - v1).norm(dim=1)    # (E,)
    return edges, ref_lengths


def edge_smoothness_loss(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """
    Naive functional edge smoothness. 
    (Note: LaplacianLoss class above is preferred for better results).
    """
    if verts.dim() == 3:
        verts = verts[0] 

    v_i = verts[edges[:, 0]] 
    v_j = verts[edges[:, 1]] 

    return (v_i - v_j).pow(2).sum(dim=1).mean()


def edge_length_loss(verts: torch.Tensor, edges: torch.Tensor, ref_lengths: torch.Tensor) -> torch.Tensor:
    """
    Edge-length preservation loss.
    """
    if verts.dim() == 3:
        verts = verts[0] 

    v0 = verts[edges[:, 0]] 
    v1 = verts[edges[:, 1]] 
    lengths = (v0 - v1).norm(dim=1) 

    return ((lengths - ref_lengths) ** 2).mean()


def multiview_silhouette_mse(renderer3d, verts_world, faces, cameras, targets):
    total = 0.0
    for (eye, center, up), target_k in zip(cameras, targets):
        pred_k, _ = renderer3d(verts_world, faces, None, eye, center, up) # Assuming renderer returns sil, rgb
        mse_k = ((pred_k - target_k) ** 2).mean()
        total = total + mse_k
    return total

def multiview_rgb_mse(renderer, verts_world, faces, colors, cameras, targets_rgb):
    total = 0.0
    for (eye, center, up), tgt_rgb in zip(cameras, targets_rgb):
        _, pred_rgb = renderer(verts_world, faces, colors, eye, center, up)
        total = total + F.mse_loss(pred_rgb, tgt_rgb)
    return total