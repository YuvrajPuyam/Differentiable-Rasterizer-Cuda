import os
import sys
import math
import torch
import trimesh
import matplotlib.pyplot as plt

# Add parent directory to path to import helper module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diff_rast.rasterizer import SoftRasterizer3D
from diff_rast.losses import (
    precompute_edge_data,
    edge_smoothness_loss,
    edge_length_loss,
    multiview_silhouette_mse,
)

from helper.mesh_io import (
    load_mesh_as_tensors,
    normalize_to_unit_world,
    save_mesh_as_obj,
)
from helper.camera import make_orbit_cameras
from helper.visualizer import save_silhouette_image

# ---------- Config ----------

MESH_PATH = r"D:\Computer Graphics\diff_rast\data\models\Tree_2.obj"  #
OUT_DIR = "opt_vis_car_stage1_multiview_pure_sil"

IMAGE_SIZE = 128
ITERS = 400
SNAPSHOT_EVERY = 50

# More views, better constraints
N_VIEWS = 5
CAM_RADIUS = 3.0

# SoftRas-style priors (only priors, no mesh-to-mesh supervision)
LAMBDA_SMOOTH = 5e-2   # Laplacian smoothness
LAMBDA_EDGE   = 1e-1   # edge-length preservation

device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------- Stage 1: pure silhouette + priors ----------

def run_stage1_multiview():
    os.makedirs(OUT_DIR, exist_ok=True)

    verts_ref, faces = load_mesh_as_tensors(MESH_PATH, device=device)
    print("Loaded mesh:", verts_ref.shape, faces.shape)

    # Normalize reference to world space cube
    verts_ref_world = normalize_to_unit_world(verts_ref)      # (V,3)
    verts_ref_world_b = verts_ref_world.unsqueeze(0)          # (1,V,3)

    # Precompute edges and reference edge lengths for priors
    edges, ref_edge_lengths = precompute_edge_data(verts_ref_world, faces)
    edges = edges.to(device)
    ref_edge_lengths = ref_edge_lengths.to(device)
    print("Num edges:", edges.shape[0])

    renderer3d = SoftRasterizer3D(
        image_size=IMAGE_SIZE,
        use_fd=False,
        fov_deg=45.0,
        aspect=1.0,
        near=0.1,
        far=10.0,
    ).to(device)

    dtype = verts_ref_world_b.dtype

    # Cameras around the object
    cameras = make_orbit_cameras(N_VIEWS, CAM_RADIUS, device, dtype)

    # Precompute target silhouettes (our only supervision)
    targets = []
    with torch.no_grad():
        for idx, (eye, center, up) in enumerate(cameras):
            target_k = renderer3d(verts_ref_world_b, faces, eye, center, up)
            targets.append(target_k)
            save_silhouette_image(
                target_k,
                os.path.join(OUT_DIR, f"target_view{idx:02d}.png"),
            )
            print(f"Saved target silhouette view {idx}.")

    # ---------- Sphere initialization (no mesh-to-mesh supervision) ----------

    V = verts_ref_world.shape[0]

    # Unit direction from origin for each vertex
    dirs = verts_ref_world_b / (verts_ref_world_b.norm(dim=-1, keepdim=True) + 1e-8)

    # Sphere radius
    R = 0.6
    blob_world = R * dirs

    # Small noise for asymmetry
    noise_scale = 0.005
    blob_world = blob_world + noise_scale * torch.randn_like(blob_world)

    blob_world.requires_grad_(True)

    optimizer = torch.optim.Adam([blob_world], lr=0.01)

    best_loss = float("inf")
    best_blob_world = None

    for it in range(ITERS):
        optimizer.zero_grad()

        # 1) Multi-view silhouette loss (only data term)
        sil_loss = multiview_silhouette_mse(
            renderer3d, blob_world, faces, cameras, targets
        )

        # 2) Priors (no ground-truth mesh supervision)
        smooth = edge_smoothness_loss(blob_world, edges)
        edge_len = edge_length_loss(blob_world, edges, ref_edge_lengths)

        # total loss: silhouettes + SoftRas-style geometric priors
        total_loss = (
            sil_loss
            + LAMBDA_SMOOTH * smooth
            + LAMBDA_EDGE * edge_len
        )

        total_loss.backward()
        grad_norm = blob_world.grad.norm().item()
        optimizer.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_blob_world = blob_world.detach().clone()

        avg_mse = (sil_loss / len(cameras)).item()

        print(
            f"iter {it:03d} | total_loss={total_loss.item():.6f} "
            f"| avg_mse={avg_mse:.6f} "
            f"| smooth={smooth.item():.6f} "
            f"| edge_len={edge_len.item():.6f} "
            f"| grad_norm={grad_norm:.6f}"
        )

        if it % SNAPSHOT_EVERY == 0 or it == ITERS - 1:
            # Save view 0 prediction as representative snapshot
            eye0, center0, up0 = cameras[0]
            with torch.no_grad():
                pred0 = renderer3d(blob_world, faces, eye0, center0, up0)
            fname = os.path.join(OUT_DIR, f"step_view0_{it:03d}.png")
            save_silhouette_image(pred0, fname)
            print("  saved", fname)

    # Use best verts
    final_verts_world = best_blob_world if best_blob_world is not None else blob_world.detach()

    # Save final silhouettes for all views
    with torch.no_grad():
        for idx, (eye, center, up) in enumerate(cameras):
            pred_k = renderer3d(final_verts_world, faces, eye, center, up)
            save_silhouette_image(
                pred_k,
                os.path.join(OUT_DIR, f"final_view{idx:02d}.png"),
            )
            print(f"Saved final silhouette view {idx}.")

    print("Best total loss:", best_loss)

    # Save final 3D mesh
    save_mesh_as_obj(
        final_verts_world,
        faces,
        os.path.join(OUT_DIR, "final_3d_world.obj"),
    )


if __name__ == "__main__":
    print("Using device:", device)
    run_stage1_multiview()
