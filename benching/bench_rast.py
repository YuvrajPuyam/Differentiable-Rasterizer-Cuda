import os
import sys
import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

import diff_rast._C as _C


# -------------------------------
# Config 
# -------------------------------
TILE_SIZE = 16
MAX_PER_TILE = 512


@torch.no_grad()
def make_dummy(B=1, V=3000, F=20000, image_size=256, device="cuda"):
    verts  = torch.randn(B, V, 3, device=device, dtype=torch.float32)
    colors = torch.rand(B, V, 3, device=device, dtype=torch.float32)
    faces  = torch.randint(0, V, (F, 3), device=device, dtype=torch.int64)
    return verts, faces, colors, image_size


def bench(verts, faces, colors, image_size, iters=200, warmup=50):
    torch.cuda.synchronize()

    # -------------------------------
    # Build bins once (important!)
    # -------------------------------
    tile_counts, tile_faces = _C.build_bins(
        verts, faces, image_size, TILE_SIZE, MAX_PER_TILE
    )
    torch.cuda.synchronize()

    # -------------------------------
    # Warmup forward
    # -------------------------------
    for _ in range(warmup):
        sil, rgb = _C.rasterize_forward(
            verts, faces, colors,
            tile_counts, tile_faces,
            image_size, TILE_SIZE, MAX_PER_TILE
        )
    torch.cuda.synchronize()

    # -------------------------------
    # Forward timing
    # -------------------------------
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        sil, rgb = _C.rasterize_forward(
            verts, faces, colors,
            tile_counts, tile_faces,
            image_size, TILE_SIZE, MAX_PER_TILE
        )
    e.record()
    torch.cuda.synchronize()
    fwd_ms = s.elapsed_time(e) / iters

    # -------------------------------
    # Prepare gradients
    # -------------------------------
    grad_sil = torch.randn_like(sil)
    grad_rgb = torch.randn_like(rgb)

    # -------------------------------
    # Warmup backward
    # -------------------------------
    for _ in range(warmup):
        gv = _C.rasterize_backward(
            grad_sil, grad_rgb,
            verts, faces, colors,
            tile_counts, tile_faces,
            image_size, TILE_SIZE, MAX_PER_TILE
        )
    torch.cuda.synchronize()

    # -------------------------------
    # Backward timing
    # -------------------------------
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        gv = _C.rasterize_backward(
            grad_sil, grad_rgb,
            verts, faces, colors,
            tile_counts, tile_faces,
            image_size, TILE_SIZE, MAX_PER_TILE
        )
    e.record()
    torch.cuda.synchronize()
    bwd_ms = s.elapsed_time(e) / iters

    # -------------------------------
    # Report
    # -------------------------------
    print(f"H=W={image_size}, B={verts.shape[0]}, V={verts.shape[1]}, F={faces.shape[0]}")
    print(f"TILE_SIZE={TILE_SIZE}, MAX_PER_TILE={MAX_PER_TILE}")
    print(f"Forward:  {fwd_ms:.3f} ms/iter")
    print(f"Backward: {bwd_ms:.3f} ms/iter")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"

    verts, faces, colors, image_size = make_dummy(
        B=1, V=3000, F=20000, image_size=256
    )

    bench(verts, faces, colors, image_size)
