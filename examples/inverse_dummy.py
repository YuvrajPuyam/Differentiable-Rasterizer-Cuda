import torch
from diff_rast.rasterizer import SoftRasterizer  # your module


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    image_size = 64

    # Two versions: FD (=slow but trusted) and CUDA-backward (=fast)
    renderer_fd = SoftRasterizer(image_size=image_size, use_fd=True).to(device)
    renderer_fast = SoftRasterizer(image_size=image_size, use_fd=False).to(device)

    # Simple single-triangle test
    B, V = 1, 3

    verts_init = torch.tensor(
        [[[-0.5, -0.5, 0.0],
          [ 0.5, -0.5, 0.0],
          [ 0.0,  0.5, 0.0]]],
        dtype=torch.float32,
        device=device,
    )

    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)

    # Make a target silhouette by slightly shifting verts
    with torch.no_grad():
        target = renderer_fd(verts_init + 0.1, faces)  # (1,1,H,W)

    # ----- Forward shape + basic grad flow check (FD mode) -----
    print("\n[1] Forward + grad check (FD mode)")
    verts_fd = verts_init.clone().detach().requires_grad_(True)
    img_fd = renderer_fd(verts_fd, faces)
    print("FD image shape:", img_fd.shape)

    loss_fd = ((img_fd - target) ** 2).mean()
    loss_fd.backward()
    print("FD loss:", float(loss_fd))
    print("FD grad norm:", float(verts_fd.grad.norm()))

    # Quick optimization for 5 steps just to see loss go down
    verts_opt = verts_init.clone().detach().requires_grad_(True)
    optim = torch.optim.SGD([verts_opt], lr=0.1)

    print("\n[2] Mini optimization (FD mode)")
    for it in range(5):
        optim.zero_grad()
        img = renderer_fd(verts_opt, faces)
        loss = ((img - target) ** 2).mean()
        loss.backward()
        optim.step()
        print(f"iter {it} | loss={float(loss):.6f} | grad_norm={float(verts_opt.grad.norm()):.6f}")

    # ----- Compare FD-grad and CUDA-grad for the same verts -----
    print("\n[3] Compare FD gradient vs CUDA-backward gradient")

    verts_fd = verts_init.clone().detach().requires_grad_(True)
    verts_fast = verts_init.clone().detach().requires_grad_(True)

    # FD gradient
    img_fd = renderer_fd(verts_fd, faces)
    loss_fd = ((img_fd - target) ** 2).mean()
    loss_fd.backward()
    grad_fd = verts_fd.grad.detach()

    # CUDA gradient (using whatever is currently implemented in rasterize_backward)
    img_fast = renderer_fast(verts_fast, faces)
    loss_fast = ((img_fast - target) ** 2).mean()
    loss_fast.backward()
    grad_fast = verts_fast.grad.detach()

    print("grad_fd:", grad_fd)
    print("grad_fast:", grad_fast)

    # Some scalar comparisons
    diff = (grad_fd - grad_fast).abs()
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    print("max |grad_fd - grad_fast|:", max_diff)
    print("mean |grad_fd - grad_fast|:", mean_diff)

    # Cosine similarity (flattened)
    fd_flat = grad_fd.view(-1)
    fast_flat = grad_fast.view(-1)

    if fd_flat.norm() > 0 and fast_flat.norm() > 0:
        cos_sim = torch.nn.functional.cosine_similarity(
            fd_flat.unsqueeze(0), fast_flat.unsqueeze(0)
        ).item()
        print("cosine similarity(grad_fd, grad_fast):", cos_sim)
    else:
        print("One of the gradients is zero; cosine similarity undefined.")

    print("\nSanity check complete.")


if __name__ == "__main__":
    main()
