from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("C:/Users/Jon/git/CS6140/HW4/PB6_boosting_spam.pyx")
)
