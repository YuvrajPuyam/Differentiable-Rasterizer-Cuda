import os
import sys
import torch
import matplotlib.pyplot as plt

# Add repo path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diff_rast.losses import LaplacianLoss, FlattenLoss
from helper.mesh_io import load_mesh_as_tensors, normalize_to_unit_world, save_mesh_as_obj
from helper.camera import make_orbit_cameras
from helper.geometry import generate_icosphere
from helper.render import render_sil_rgb, make_vertex_colors_from_world

# ===================== Config ===================== #
MESH_PATH = r"D:\Computer Graphics\diff_rast\data\models\Pistol_02.obj"
OUT_DIR = "render_complex_fix"
IMAGE_SIZE, ITERS, SNAPSHOT_EVERY = 256, 2000, 100
N_VIEWS, CAM_RADIUS = 12, 3.0

# Phase 1 ends earlier to allow detail to grow
PHASE_1_ITERS = 500 
LOG_EVERY = 20

LAMBDA_SIL, LAMBDA_RGB = 1.0, 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_inverse_render_rgb():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Load Reference
    print(f"Loading reference: {MESH_PATH}")
    verts_ref, faces_ref = load_mesh_as_tensors(MESH_PATH, device=device)
    faces_ref = faces_ref.to(device, dtype=torch.long)
    verts_ref = normalize_to_unit_world(verts_ref)
    colors_ref = make_vertex_colors_from_world(verts_ref).unsqueeze(0)
    cameras = make_orbit_cameras(N_VIEWS, CAM_RADIUS, device, verts_ref.dtype)

    targets_sil, targets_rgb = [], []
    with torch.no_grad():
        for (eye, center, up) in cameras:
            s, c = render_sil_rgb(verts_ref.unsqueeze(0), faces_ref, colors_ref, eye, center, up, IMAGE_SIZE)
            targets_sil.append(s.detach()); targets_rgb.append(c.detach())

    # 2. Setup Optimization Mesh (Level 4 for detail)
    print("Generating Icosphere ")
    verts_opt, faces_opt = generate_icosphere(level=3, device=device) 
    
    # --- STRATEGY CHANGE: SHRINK WRAP ---
    # Start LARGER (1.2x) than the object. 
    # It is easier for the mesh to collapse onto branches than to grow into them.
    blob = (verts_opt.clone() * 1.2).unsqueeze(0) 
    blob.requires_grad_(True)
    
    # 3. Setup Losses
    lap_loss_fn = LaplacianLoss(verts_opt, faces_opt, average=True).to(device)
    flat_loss_fn = FlattenLoss(faces_opt, average=True).to(device)
    
    # Use a moderate learning rate. Too high = explosion, Too low = stuck.
    optimizer = torch.optim.Adam([blob], lr=0.005)

    print("Starting optimization...")
    for it in range(ITERS):
        optimizer.zero_grad()
        
        # --- THE "NO-BLOB" SCHEDULE ---
        if it < PHASE_1_ITERS:
            # Phase 1: SHRINKING
            # Laplacian is 0.1 (Previous scripts used 1.0 or 2.0 -> THIS WAS THE ERROR)
            w_rgb, w_sil = 0.0, 1.0
            w_lap, w_flat = 0.1, 0.01 
        else:
            # Phase 2: CARVING
            # Decay Regularization to ZERO quickly.
            # We need the mesh to be free to fold into thin branches.
            p = (it - PHASE_1_ITERS) / (ITERS - PHASE_1_ITERS)
            
            w_rgb = LAMBDA_RGB
            w_sil = LAMBDA_SIL
            
            # Decay to 0.0. 
            # If we keep ANY Laplacian weight, the branches will shrink away.
            w_lap = 0.1 * (1.0 - p) 
            w_flat = 0.01 * (1.0 - p)

        colors_blob = make_vertex_colors_from_world(blob.squeeze(0)).unsqueeze(0)
        
        # Data Term
        loss_sil, loss_rgb = 0.0, 0.0
        for (eye, center, up), t_sil, t_rgb in zip(cameras, targets_sil, targets_rgb):
            p_sil, p_rgb = render_sil_rgb(blob, faces_opt, colors_blob, eye, center, up, IMAGE_SIZE)
            loss_sil += torch.nn.functional.mse_loss(p_sil, t_sil)
            if w_rgb > 0: loss_rgb += torch.nn.functional.mse_loss(p_rgb, t_rgb)
        
        loss_sil /= N_VIEWS
        loss_rgb /= N_VIEWS if w_rgb > 0 else 1.0
        
        # Priors
        lap = lap_loss_fn(blob)
        flat = flat_loss_fn(blob)
        
        total_loss = w_sil*loss_sil + w_rgb*loss_rgb + w_lap*lap + w_flat*flat
        total_loss.backward()
        
        # Clip grads
        torch.nn.utils.clip_grad_norm_([blob], max_norm=1.0)
        optimizer.step()

        if it % LOG_EVERY == 0:
            print(f"[{it:04d}] Tot={total_loss.item():.5f} Sil={loss_sil.item():.5f} "
                  f"Lap_W={w_lap:.4f} Flat_W={w_flat:.4f}")

        if SNAPSHOT_EVERY and (it % SNAPSHOT_EVERY == 0 or it == ITERS-1):
            with torch.no_grad():
                eye, center, up = cameras[0]
                _, res_rgb = render_sil_rgb(blob, faces_opt, colors_blob, eye, center, up, IMAGE_SIZE)
                plt.imsave(os.path.join(OUT_DIR, f"step_{it:04d}.png"), res_rgb[0].permute(1,2,0).clamp(0,1).cpu().numpy())
                save_mesh_as_obj(blob.detach().squeeze(0), faces_opt, os.path.join(OUT_DIR, f"step_{it:04d}.obj"))

    save_mesh_as_obj(blob.detach().squeeze(0), faces_opt, os.path.join(OUT_DIR, "final.obj"))
    print("Done.")

if __name__ == "__main__":
    run_inverse_render_rgb()