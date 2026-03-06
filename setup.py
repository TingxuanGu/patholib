"""
Minimal setup.py for installing patholib as a local package.
Allows `from patholib.xxx import yyy` to work from any working directory.
"""

from setuptools import setup, find_packages

setup(
    name='patholib',
    version='0.1.0',
    description='Pathology Image Analysis Library for IHC and H&E',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.24',
        'scipy>=1.10',
        'scikit-image>=0.21',
        'opencv-python-headless>=4.8',
        'Pillow>=10.0',
        'matplotlib>=3.7',
    ],
    extras_require={
        'gpu': ['torch>=2.0', 'cellpose>=3.0'],
        'wsi': ['openslide-python>=1.3.1'],
        'all': ['torch>=2.0', 'cellpose>=3.0', 'openslide-python>=1.3.1'],
    },
)
