from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='edgemaze',
    ext_modules=cythonize('edgemaze/speedup.pyx', language_level=3),
    include_dirs=[numpy.get_include()],
    setup_requires=[
        'Cython',
        'NumPy',
    ],
    install_requires=[
        'NumPy',
    ],
)
