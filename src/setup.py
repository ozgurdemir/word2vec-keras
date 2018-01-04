from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([Extension("src.skip_gram", ["skip_gram.pyx"],
                                     define_macros=[('CYTHON_TRACE', '1')])])
)
