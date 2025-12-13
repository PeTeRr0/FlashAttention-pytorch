from pathlib import Path
import os
from setuptools import find_packages, setup


def build_cuda_extension():
    flag = os.environ.get("FLASH_ATTENTION_BUILD_CUDA", "0").lower()
    if flag not in {"1", "true", "yes", "on"}:
        return None, None

    try:
        import torch  # type: ignore
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # type: ignore
    except Exception as exc:  # pragma: no cover - build-time dependency
        raise RuntimeError(
            "Building the CUDA extension requires a CUDA-enabled PyTorch install. "
            "Install torch with CUDA support first, then re-run with FLASH_ATTENTION_BUILD_CUDA=1."
        ) from exc

    root = Path(__file__).parent
    sources = [
        "csrc/common/torch.extension.cpp",
        "csrc/common/bindings.cpp",
        "csrc/fa1/fa1_fwd.cu",
        "csrc/fa1/fa1_bwd.cu",
        "csrc/fa2/fa2_fwd.cu",
        "csrc/fa2/fa2_bwd.cu",
        "csrc/fa3/fa3_fwd.cu",
        "csrc/fa3/fa3_bwd.cu",
    ]
    ext = CUDAExtension(
        name="flashattention_lab_cuda",
        sources=[str(root / src) for src in sources],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
    )
    build_ext_cls = BuildExtension.with_options(use_ninja=False)
    return ext, build_ext_cls


ext_modules = []
cmdclass = {}
maybe_ext, maybe_build_ext = build_cuda_extension()
if maybe_ext is not None and maybe_build_ext is not None:
    ext_modules.append(maybe_ext)
    cmdclass["build_ext"] = maybe_build_ext

setup(
    name="flashattention-lab",
    version="0.0.0",
    description="Minimal FlashAttention reference kernels (PyTorch/Triton/CUDA) for benchmarking and testing.",
    author="",
    python_requires=">=3.9",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[],
    extras_require={
        "test": ["pytest>=7.0"],
        "bench": ["matplotlib>=3.5"],
        "dev": ["pytest>=7.0", "matplotlib>=3.5"],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)
