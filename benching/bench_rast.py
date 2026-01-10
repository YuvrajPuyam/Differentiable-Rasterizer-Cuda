import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # if script is at repo root
# OR if script is inside some subfolder, insert the parent:
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import diff_rast._C as _C 


@torch.no_grad()
def make_dummy(B=1, V=2000, F=4000, image_size=256, device="cuda"):
    verts  = torch.randn(B, V, 3, device=device, dtype=torch.float32)
    colors = torch.rand(B, V, 3, device=device, dtype=torch.float32)
    faces  = torch.randint(0, V, (F, 3), device=device, dtype=torch.int64)
    return verts, faces, colors, image_size

def bench(verts, faces, colors, image_size, iters=200, warmup=50):
    torch.cuda.synchronize()

    # warmup forward
    for _ in range(warmup):
        sil, rgb = _C.rasterize_forward(verts, faces, colors, int(image_size))
    torch.cuda.synchronize()

    # forward timing
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        sil, rgb = _C.rasterize_forward(verts, faces, colors, int(image_size))
    e.record()
    torch.cuda.synchronize()
    fwd_ms = s.elapsed_time(e) / iters

    grad_sil = torch.randn_like(sil)
    grad_rgb = torch.randn_like(rgb)

    # warmup backward
    for _ in range(warmup):
        gv = _C.rasterize_backward(grad_sil, grad_rgb, verts, faces, colors, int(image_size))
    torch.cuda.synchronize()

    # backward timing
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        gv = _C.rasterize_backward(grad_sil, grad_rgb, verts, faces, colors, int(image_size))
    e.record()
    torch.cuda.synchronize()
    bwd_ms = s.elapsed_time(e) / iters

    print(f"H=W={image_size}, B={verts.shape[0]}, V={verts.shape[1]}, F={faces.shape[0]}")
    print(f"Forward:  {fwd_ms:.3f} ms/iter")
    print(f"Backward: {bwd_ms:.3f} ms/iter")

if __name__ == "__main__":
    assert torch.cuda.is_available()
    verts, faces, colors, image_size = make_dummy(B=1, V=3000, F=6000, image_size=256)
    bench(verts, faces, colors, image_size)
