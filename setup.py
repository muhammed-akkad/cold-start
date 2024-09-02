from setuptools import setup, Extension
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_saver',
    ext_modules=[
        CUDAExtension('cuda_saver', [
            'cuda_saver.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
