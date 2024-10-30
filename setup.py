from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_saver',
    ext_modules=[
        CUDAExtension(
            name='cuda_saver',
            sources=['tensor_offloader.cpp'],
            extra_compile_args={
                'nvcc': [],  # No extra flags for now
                'cxx': []  # No extra flags for now
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
