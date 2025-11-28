# What is a Differentiable Rasterizer

A differentiable rasterizer is a rasterization-based rendering system where the steps that convert 3D meshes into 2D images are modified or relaxed so that they remain differentiable with respect to the underlying scene parameters. Classic rasterization uses hard decisions (inside vs outside a triangle, visibility tests, z-buffer comparisons) that block gradient flow. A differentiable rasterizer replaces these discontinuous operations with smooth, gradient-friendly formulations such as soft visibility, probabilistic coverage, and differentiable edge functions.

This enables gradients to flow from pixel-level losses back to mesh vertices, camera parameters, and other scene attributes. As a result, the renderer can be used not only for forward rendering but also for inverse graphics tasks such as silhouette-based reconstruction, geometry refinement, pose estimation, or joint appearance–shape optimization.


## Reconstruction Gallery

|                | Avocado | Kettle | Tree |
|----------------|---------|--------|-------|
| **Original**   | ![avocado_original](/media/avocado.png) | ![kettle_original](/media/kettle_original.png) | ![tree_original](/media/tree_original.png) |
| **Reconstruction** | ![avocado_recon](/media/avocado_recon.gif) | ![kettle_recon](/media/kettle_recon.gif) | ![tree_recon](/media/tree_recon.gif) |


# Implementation

This project implements a custom differentiable rasterizer in CUDA and C++ with a PyTorch autograd interface. It supports both forward rendering and inverse rendering using silhouette-based loss functions. All core components including rasterization, camera projection, loss terms, and the multi-view optimization loop are implemented manually for clarity, inspection, and research experimentation.

The system is capable of reconstructing 3D geometry using only a small number of silhouette images by performing gradient-based optimization directly on mesh vertex positions.  

# Repository Capabilities

### Forward Rendering
- World-space to NDC-space projection
- Custom CUDA silhouette rasterizer
- Soft coverage computation
- Fully differentiable forward pass

### Backward Rendering
- Custom CUDA backward pass
- Gradients w.r.t. vertex positions

### Inverse Rendering
- Reconstruction from silhouettes only
- No mesh-to-mesh or dataset supervision



# Installation and Setup

### 1. Update CUDA Architecture Flags

Open `setup.py` and update:

```python
"--generate-code=arch=compute_86,code=sm_86"
```
Replace 86 with your GPU compute capability.

## 2. Build the CUDA Extension

Run the following command from the project root:

```bash
python setup.py install 
```

For editable development:

```bash
pip install setup.py .
```

## 3. Add Models

Place `.obj` or `.glb` files inside:

```bash
data/models/
```

For example:

```bash
data/models/kettle.glb
```

## 4. Configure and Run Inverse Rendering Script

Edit:

```bash
examples/inverse_render.py
```

Update the model path:

```python
MESH_PATH = "data/models/your_model.glb"

```
Run the optimization script:

```bash
python examples/inverse_render.py
```





## Next Steps

### Planned Features

- Stage 2 shading and RGB reconstruction  
- Vertex normal computation  
- Lambertian shading  
- Combined silhouette + photometric optimization  
- Improved differentiability and visibility modeling  

### Medium Article

A future long-form article will cover:

- The mathematics of differentiable rasterization  
- CUDA kernel design  
- PyTorch autograd integration  
- Multi-view inverse graphics  
- Limitations and future directions  


