import os
import sys
import torch
import math
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np

# Add repo path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diff_rast.losses import LaplacianLoss, FlattenLoss
from helper.mesh_io import load_mesh_as_tensors, normalize_to_unit_world, save_mesh_as_obj
from helper.camera import make_orbit_cameras
from helper.geometry import generate_icosphere
from helper.render import render_sil_rgb, make_vertex_colors_from_world

# ===================== Config ===================== #
MESH_PATH = r"D:\Computer Graphics\diff_rast\data\models\Avocado.glb"
OUT_DIR = "render_complex_fix"

# 1.5x Iterations (2000 * 1.5 = 3000)
# SNAPSHOT_EVERY = 40 ensures we get roughly 75 frames for the evolution (3000/40)
IMAGE_SIZE, ITERS, SNAPSHOT_EVERY = 256, 3000, 40 
N_VIEWS, CAM_RADIUS = 12, 3.0

# Scaled Phase 1 proportional to total iterations (500 * 1.5 = 750)
PHASE_1_ITERS = 500 
LOG_EVERY = 20
LAMBDA_SIL, LAMBDA_RGB = 1.0, 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

# Standard camera vectors
cam_center = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)
cam_up = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)

def run_inverse_render_rgb():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Load Reference
    print(f"Loading reference: {MESH_PATH}")
    verts_ref, faces_ref = load_mesh_as_tensors(MESH_PATH, device=device)
    faces_ref = faces_ref.to(device, dtype=torch.long)
    verts_ref = normalize_to_unit_world(verts_ref)
    colors_ref = make_vertex_colors_from_world(verts_ref).unsqueeze(0)
    
    cameras_train = make_orbit_cameras(N_VIEWS, CAM_RADIUS, device, verts_ref.dtype)

    targets_sil, targets_rgb = [], []
    with torch.no_grad():
        for (eye, center, up) in cameras_train:
            s, c = render_sil_rgb(verts_ref.unsqueeze(0), faces_ref, colors_ref, eye, center, up, IMAGE_SIZE)
            targets_sil.append(s.detach()); targets_rgb.append(c.detach())

    # 2. Setup Optimization Mesh
    print("Generating Icosphere")
    verts_opt, faces_opt = generate_icosphere(level=3, device=device) 
    
    blob = (verts_opt.clone() * 1.2).unsqueeze(0) 
    blob.requires_grad_(True)
    
    # 3. Setup Losses
    lap_loss_fn = LaplacianLoss(verts_opt, faces_opt, average=True).to(device)
    flat_loss_fn = FlattenLoss(faces_opt, average=True).to(device)
    
    optimizer = torch.optim.Adam([blob], lr=0.005)

    # Accumulator for the GIF
    all_frames = [] 

    print("Starting optimization...")
    # ================= PHASE A: EVOLUTION (SPINNING) =================
    for it in range(ITERS + 1):
        optimizer.zero_grad()
        
        # Schedule
        if it < PHASE_1_ITERS:
            w_rgb, w_sil = 0.0, 1.0
            w_lap, w_flat = 0.1, 0.01 
        else:
            p = (it - PHASE_1_ITERS) / (ITERS - PHASE_1_ITERS)
            w_rgb = LAMBDA_RGB
            w_sil = LAMBDA_SIL
            w_lap = 0.1 * (1.0 - p) 
            w_flat = 0.01 * (1.0 - p)

        colors_blob = make_vertex_colors_from_world(blob.squeeze(0)).unsqueeze(0)
        
        # Data Term
        loss_sil, loss_rgb = 0.0, 0.0
        for (eye, center, up), t_sil, t_rgb in zip(cameras_train, targets_sil, targets_rgb):
            p_sil, p_rgb = render_sil_rgb(blob, faces_opt, colors_blob, eye, center, up, IMAGE_SIZE)
            loss_sil += torch.nn.functional.mse_loss(p_sil, t_sil)
            if w_rgb > 0: loss_rgb += torch.nn.functional.mse_loss(p_rgb, t_rgb)
        
        loss_sil /= N_VIEWS
        loss_rgb /= N_VIEWS if w_rgb > 0 else 1.0
        
        # Priors
        lap = lap_loss_fn(blob)
        flat = flat_loss_fn(blob)
        
        total_loss = w_sil*loss_sil + w_rgb*loss_rgb + w_lap*lap + w_flat*flat
        
        if it < ITERS:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([blob], max_norm=1.0)
            optimizer.step()

        if it % LOG_EVERY == 0 and it < ITERS:
            print(f"[{it:04d}] Tot={total_loss.item():.5f} Sil={loss_sil.item():.5f}")

        # Capture Spinning Evolution Frames
        if SNAPSHOT_EVERY and (it % SNAPSHOT_EVERY == 0 or it == ITERS):
            with torch.no_grad():
                # Map progress 0.0 -> 1.0 to Angle 0 -> 2pi
                progress = it / ITERS
                angle = progress * 2 * math.pi

                cam_x = CAM_RADIUS * math.cos(angle)
                cam_z = CAM_RADIUS * math.sin(angle)
                cam_y = 1.0
                eye_spin = torch.tensor([cam_x, cam_y, cam_z], device=device, dtype=torch.float32)

                _, res_rgb = render_sil_rgb(blob, faces_opt, colors_blob, eye_spin, cam_center, cam_up, IMAGE_SIZE)
                
                img_np = res_rgb[0].permute(1,2,0).clamp(0,1).cpu().numpy()
                img_uint8 = (img_np * 255).astype(np.uint8)
                all_frames.append(img_uint8)

    save_mesh_as_obj(blob.detach().squeeze(0), faces_opt, os.path.join(OUT_DIR, "final.obj"))
    
    # ================= PHASE B: FINAL 360 TURNTABLE =================
    print("Rendering final 360 turntable...")
    N_TURN_FRAMES = 60 # 60 Frames for the final spin
    
    # Precompute final colors
    with torch.no_grad():
        final_colors = make_vertex_colors_from_world(blob.squeeze(0)).unsqueeze(0)

        for i in range(N_TURN_FRAMES):
            # We continue the angle from where evolution left off (2pi == 0)
            angle = 2 * math.pi * (i / N_TURN_FRAMES)
            
            cam_x = CAM_RADIUS * math.cos(angle)
            cam_z = CAM_RADIUS * math.sin(angle)
            cam_y = 1.0 
            eye_spin = torch.tensor([cam_x, cam_y, cam_z], device=device, dtype=torch.float32)

            _, res_rgb = render_sil_rgb(blob, faces_opt, final_colors, eye_spin, cam_center, cam_up, IMAGE_SIZE)
            
            img_np = res_rgb[0].permute(1,2,0).clamp(0,1).cpu().numpy()
            img_uint8 = (img_np * 255).astype(np.uint8)
            all_frames.append(img_uint8)

    print(f"Saving combined GIF to {OUT_DIR}/full_evolution.gif...")
    # Saving at 50ms per frame (20fps) for fluid motion
    iio.imwrite(os.path.join(OUT_DIR, 'full_evolution.gif'), all_frames, duration=50, loop=0)
    print("Done.")

if __name__ == "__main__":
    run_inverse_render_rgb()