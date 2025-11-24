import torch
import trimesh


def load_mesh_as_tensors(path: str, device="cuda"):
    loaded = trimesh.load(path, process=False)

    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values()]
        if len(meshes) == 0:
            raise ValueError("Scene has no geometry!")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, list):
        if len(loaded) == 0:
            raise ValueError("Loaded list is empty!")
        mesh = trimesh.util.concatenate(loaded)
    else:
        raise TypeError(f"Cannot handle loaded type: {type(loaded)}")

    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
    return verts, faces


def normalize_to_unit_world(verts: torch.Tensor) -> torch.Tensor:
    center = verts.mean(dim=0, keepdim=True)
    verts_c = verts - center
    max_coord = verts_c.abs().max()
    scale = 0.9 / (max_coord + 1e-8)
    return verts_c * scale


def save_mesh_as_obj(verts_tensor: torch.Tensor,
                     faces_tensor: torch.Tensor,
                     filename: str):

    if verts_tensor.dim() == 3:
        verts_tensor = verts_tensor[0]

    verts = verts_tensor.detach().cpu().numpy()
    faces = faces_tensor.detach().cpu().numpy()

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(filename)
    print(f"Saved mesh to: {filename}")
