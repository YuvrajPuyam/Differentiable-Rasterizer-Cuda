import torch
import math


def make_orbit_cameras(n_views, radius, device, dtype, elevation_deg=10.0):
    cams = []
    elev = elevation_deg * math.pi / 180.0

    for i in range(n_views):
        yaw = 2.0 * math.pi * i / n_views

        x = radius * math.sin(yaw) * math.cos(elev)
        z = radius * math.cos(yaw) * math.cos(elev)
        y = radius * math.sin(elev)

        eye = torch.tensor([x, y, z], device=device, dtype=dtype)
        center = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

        cams.append((eye, center, up))

    return cams
