from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = {
    "nvcc": [
        "-O2",
        "--generate-code=arch=compute_120,code=sm_120"  # Change this line!
    ]
}

ext_modules = [
    CUDAExtension(
        name="diff_rast._C",
        sources=[
            "csrc/binding.cpp",
            "csrc/rasterize_forward.cu",
            "csrc/rasterize_backward.cu",
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