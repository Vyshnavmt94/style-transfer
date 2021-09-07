import os
import json
from setuptools import setup, find_packages

version = "0.0.1"

setup(
    name="style_transfer",
    version=version,
    description="",
    install_requires=["tensorflow"],
    author="Vyshnav M T",
    author_email="vyshnav94.mec@gmail.com",
    license="MIT License",
    packages=find_packages(),
    zip_safe=False
)
