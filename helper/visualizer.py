import torch
import matplotlib.pyplot as plt


def save_silhouette_image(tensor_image: torch.Tensor, filename: str):
    if tensor_image.dim() == 4:
        img = tensor_image[0, 0].detach().cpu().numpy()
    elif tensor_image.dim() == 3:
        img = tensor_image[0].detach().cpu().numpy()
    else:
        img = tensor_image.detach().cpu().numpy()

    plt.figure()
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
