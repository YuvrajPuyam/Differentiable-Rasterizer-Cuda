import torch
from diff_rast.rasterizer import SoftRasterizer

device = "cuda"

# Image size
H = W = 64

# 1 batch, 3 vertices
B, V = 1, 3

# Define a simple triangle in NDC space [-1, 1]
# Let's make a big upright triangle in the center:
# v0: top (0, 0.5)
# v1: bottom-left (-0.5, -0.5)
# v2: bottom-right (0.5, -0.5)
verts = torch.tensor(
    [[
        [ 0.0,  0.2, 0.0],   # v0
        [-0.5, -0.5, 0.0],   # v1
        [ 0.5, -0.5, 0.0],   # v2
    ]],
    dtype=torch.float32,
    device=device,
)

# Single face using these three vertices
faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

# Create rasterizer
rast = SoftRasterizer(image_size=H).to(device)

# Forward: render silhouette
with torch.no_grad():
    image = rast(verts, faces)   # (1,1,H,W)

print("Rendered image shape:", image.shape)
print("Image min/max:", image.min().item(), image.max().item())

# Move to CPU and squeeze to (H, W)
img = image[0, 0].detach().cpu()

# Optional: downsample for easier printing (e.g., 32x32)
# Uncomment if you want:
# import torch.nn.functional as F
# img_small = F.interpolate(
#     img.unsqueeze(0).unsqueeze(0), size=(32, 32), mode="nearest"
# )[0, 0]
# H_small, W_small = img_small.shape

# For now, print a coarse ASCII version from the 64x64 image by sampling every few pixels
step = 2  # sample every 4 pixels to keep console output small
for y in range(0, H, step):
    row = ""
    for x in range(0, W, step):
        v = img[y, x].item()
        # If inside triangle (1.0) print '#', else '.'
        row += "#" if v > 0.5 else "."
    print(row)
