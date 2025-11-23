# optimize_gub_silhouette.py
import os
import torch
import trimesh
import matplotlib.pyplot as plt

from diff_rast.rasterizer import SoftRasterizer  # your differentiable rasterizer


# ---------- Config ----------

MESH_PATH = r"D:\Computer Graphics\diff_rast\examples\gub.glb"
OUT_DIR = "opt_vis_gub"
IMAGE_SIZE = 128
ITERS = 200
SNAPSHOT_EVERY = 25

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- Helpers ----------

def load_mesh_as_tensors(path: str, device="cuda"):
    """
    Robust loader: handles .glb/.gltf/.obj that may load as:
    - a Trimesh object
    - a Scene with multiple geometries
    - a list of Trimesh objects
    """
    loaded = trimesh.load(path, process=False)

    # ---- Case 1: Already a Trimesh ----
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded

    # ---- Case 2: Scene → merge all geometries ----
    elif isinstance(loaded, trimesh.Scene):
        # Convert scene geometry dict to list of meshes
        meshes = [g for g in loaded.geometry.values()]
        if len(meshes) == 0:
            raise ValueError("Scene has no geometry!")
        mesh = trimesh.util.concatenate(meshes)

    # ---- Case 3: list of Trimesh objects ----
    elif isinstance(loaded, list):
        if len(loaded) == 0:
            raise ValueError("Loaded list is empty!")
        # Merge all meshes into one
        mesh = trimesh.util.concatenate(loaded)

    else:
        raise TypeError(f"Cannot handle loaded type: {type(loaded)}")

    # Convert to tensors
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
    return verts, faces


def normalize_to_ndc(verts: torch.Tensor) -> torch.Tensor:
    """
    Center and scale vertices into approx [-1,1] range in XY.
    """
    center = verts.mean(dim=0, keepdim=True)   # (1,3)
    verts_c = verts - center

    max_xy = verts_c[:, :2].abs().max()
    scale = 0.9 / (max_xy + 1e-8)
    verts_ndc = verts_c * scale

    return verts_ndc


def save_silhouette_image(tensor_image: torch.Tensor, filename: str):
    """
    Save a (1,1,H,W) or (H,W) tensor as a grayscale PNG.
    """
    if tensor_image.dim() == 4:
        img = tensor_image[0, 0].detach().cpu().numpy()
    elif tensor_image.dim() == 2:
        img = tensor_image.detach().cpu().numpy()
    else:
        raise ValueError(f"Unexpected image shape: {tensor_image.shape}")

    plt.figure()
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# ---------- Main optimization ----------

def optimize_blob_to_silhouette():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load mesh
    verts_ref, faces = load_mesh_as_tensors(MESH_PATH, device=device)
    print("Loaded mesh:", verts_ref.shape, faces.shape)  # (V,3), (F,3)

    verts_ref = normalize_to_ndc(verts_ref)        # (V,3)
    verts_ref_b = verts_ref.unsqueeze(0)           # (1,V,3)

    renderer = SoftRasterizer(image_size=IMAGE_SIZE, use_fd=False).to(device)

    # 2) Target silhouette
    with torch.no_grad():
        target = renderer(verts_ref_b, faces)      # (1,1,H,W)
    save_silhouette_image(target, os.path.join(OUT_DIR, "target.png"))
    print("Saved target silhouette ->", os.path.join(OUT_DIR, "target.png"))

    # 3) Random blob with same topology
    V = verts_ref.shape[0]
    blob = torch.randn(V, 3, device=device) * 0.5
    blob[:, 2] = 0.0  # keep z ~ 0 for now
    blob = normalize_to_ndc(blob).unsqueeze(0).requires_grad_(True)  # (1,V,3)

  
    optimizer = torch.optim.Adam([blob], lr=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,    # halve LR
        patience=10,   # 10 iterations with no improvement
    )

    best_loss = float("inf")
    best_blob = None

    # 4) Optimize
    for it in range(ITERS):
        optimizer.zero_grad()
        pred = renderer(blob, faces)               # (1,1,H,W)

        loss = ((pred - target) ** 2).mean()
        loss.backward()

        grad_norm = blob.grad.norm().item()
        optimizer.step()
        scheduler.step(loss)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_blob = blob.detach().clone()

        print(f"iter {it:03d} | loss={loss.item():.6f} | grad_norm={grad_norm:.6f}")

        if it % SNAPSHOT_EVERY == 0 or it == ITERS - 1:
            fname = os.path.join(OUT_DIR, f"step_{it:03d}.png")
            save_silhouette_image(pred, fname)
            print("  saved", fname)

    # 5) Final silhouette
    with torch.no_grad():
        final_img = renderer(blob, faces)
    save_silhouette_image(final_img, os.path.join(OUT_DIR, "final.png"))
    print("Saved final silhouette ->", os.path.join(OUT_DIR, "final.png"))

    save_mesh_as_obj(blob, faces, os.path.join(OUT_DIR, "final_recovered.obj"))
    return blob.detach(), verts_ref, faces

def save_mesh_as_obj(verts_tensor, faces_tensor, filename):
    """
    verts_tensor: (1, V, 3) or (V, 3)
    faces_tensor: (F, 3)
    filename: "output.obj"
    """
    # Remove batch dimension if present
    if verts_tensor.dim() == 3:
        verts_tensor = verts_tensor[0]

    verts = verts_tensor.detach().cpu().numpy()
    faces = faces_tensor.detach().cpu().numpy()

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(filename)
    print(f"Saved mesh to: {filename}")

if __name__ == "__main__":
    print("Using device:", device)
    optimize_blob_to_silhouette()
