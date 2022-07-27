# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 19:09:31 2022

@author: AustinHsu
"""

from setuptools import setup, find_packages
import re

try:
    import torch

    has_dev_pytorch = "dev" in torch.__version__
except ImportError:
    has_dev_pytorch = False
    
try:
    import qtorch
except ImportError:
    print("Please install QPyTorch from https://github.com/B05901022/QPyTorch")
    exit()

# Base requirements
install_requires = [
    "torch>=1.10.0",
    #"torchaudio>=0.10.0"
]
if has_dev_pytorch:  # Remove the PyTorch requirement
    install_requires = [
        install_require for install_require in install_requires if "torch" != re.split(r"(=|<|>)", install_require)[0]
    ]

setup(
    name = "torch_hfaf",
    version = "0.1.0",
    description = "PyTorch implementation of R22SDF FFT with fixed-point support",
    long_description=open("README.md").read(),
    author = "Jui-Yang Hsu",
    author_email = "r09943025@ntu.edu.tw",
    packages = find_packages(),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=install_requires,
    )