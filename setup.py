from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
cc_major, cc_minor = torch.cuda.get_device_capability()
cc = f"{cc_major}{cc_minor}"  # e.g. "86" for RTX 30xx

print("THIS IS CC", cc)
CUDA_FLAGS = {
    "nvcc": [
        "-O2",
        f"-gencode=arch=compute_{cc},code=sm_{cc}",
    ]
}

ext_modules = [
    CUDAExtension(
        name="diff_rast._C",
        sources=[
            "csrc/binding.cpp",
            "csrc/rasterize_forward.cu",
            "csrc/rasterize_backward.cu",
            "csrc/bin_faces.cu"
        ],
        extra_compile_args=CUDA_FLAGS,
    )
]

setup(
    name="diff_rast",
    version="0.0.1",
    description="Minimal differentiable rasterizer experiments",
    packages=["diff_rast"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)