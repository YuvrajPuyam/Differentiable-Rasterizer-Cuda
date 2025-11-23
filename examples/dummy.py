import torch
from diff_rast.rasterizer import SoftRasterizer  

B, V = 2, 10
image_size = 64

verts = torch.randn(B, V, 3, device="cuda", requires_grad=True)
faces = torch.zeros(1, 3, dtype=torch.long, device="cuda")  # dummy

# 1. Initialize rasterizer
rast = SoftRasterizer(image_size=image_size)

# 2. Forward
image = rast(verts, faces)
print("image:", image.shape)

# 3. Backward
loss = image.sum()
loss.backward()

print("grad shape:", verts.grad.shape)
print("grad norm:", verts.grad.norm())
print("grad[0,0,0]:", verts.grad[0,0,0])
