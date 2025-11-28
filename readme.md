# What is a Differentiable Renderer

A differentiable renderer is a rendering system where the entire image formation process is differentiable end-to-end. This enables the use of gradient-based optimization to solve inverse graphics problems. Instead of only generating images from 3D scenes (forward rendering), a differentiable renderer allows adjusting 3D geometry, camera parameters, materials, or lighting by comparing rendered images with target images and propagating gradients back to scene parameters.

Differentiable rendering is widely used in 3D reconstruction, neural rendering, inverse graphics, generative modeling, shape optimization, and machine learning pipelines that require rendering as a differentiable operation.

Each example includes the original model and the reconstructed geometry obtained through inverse rendering.

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


