# examples/render_reference_rgb_silhouettes_like_script.py
import os
import sys
import torch
import matplotlib.pyplot as plt

# Add repo path (same as your script)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper.mesh_io import load_mesh_as_tensors, normalize_to_unit_world
from helper.camera import make_orbit_cameras
from helper.render import render_sil_rgb, make_vertex_colors_from_world

# ===================== Config ===================== #
MESH_PATH = r"D:\Computer Graphics\diff_rast\data\models\kettle.glb"
OUT_DIR = "ref_rgb_silhouettes_like_script"
IMAGE_SIZE = 256
N_VIEWS, CAM_RADIUS = 12, 3.0

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading reference: {MESH_PATH}")

    # 1) Load mesh exactly like your original script
    verts_ref, faces_ref = load_mesh_as_tensors(MESH_PATH, device=device)
    faces_ref = faces_ref.to(device, dtype=torch.long)
    verts_ref = normalize_to_unit_world(verts_ref)

    # 2) Same vertex coloring method
    colors_ref = make_vertex_colors_from_world(verts_ref).unsqueeze(0)

    # 3) Same orbit cameras
    cameras = make_orbit_cameras(N_VIEWS, CAM_RADIUS, device, verts_ref.dtype)

    # 4) Render & save
    with torch.no_grad():
        for i, (eye, center, up) in enumerate(cameras):
            sil, rgb = render_sil_rgb(
                verts_ref.unsqueeze(0),   # (1,V,3)
                faces_ref,                # (F,3)
                colors_ref,               # (1,V,3)
                eye, center, up,
                IMAGE_SIZE
            )

            # sil: (1,1,H,W) in [0,1]
            # rgb: (1,3,H,W) in [0,1] (as used in your script)
            rgb_sil = rgb * sil  # colored object only where silhouette is 1

            # Save RGB silhouette (colored cutout)
            out_rgb_sil = os.path.join(OUT_DIR, f"rgb_sil_{i:02d}.png")
            plt.imsave(
                out_rgb_sil,
                rgb_sil[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            )

            # (Optional) Save full RGB render for sanity
            out_rgb = os.path.join(OUT_DIR, f"rgb_{i:02d}.png")
            plt.imsave(
                out_rgb,
                rgb[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            )

            # (Optional) Save grayscale silhouette mask
            out_sil = os.path.join(OUT_DIR, f"sil_{i:02d}.png")
            plt.imsave(
                out_sil,
                sil[0, 0].clamp(0, 1).cpu().numpy(),
                cmap="gray"
            )

            print("Saved:", out_rgb_sil)

    print("Done.")


if __name__ == "__main__":
    main()
