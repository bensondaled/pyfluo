"""
The accompanied C file called tifffile.c is an addition to the tifffile.py provided by the developer for speed. In order to use it, you must run the setup command below. It will produce a .so file, which needs to be placed IN THE PYTHONPATH shell variable in order to be used. If you do so, importing anything from the tifffile.py module will automatically include the fast C versions.

"""
#Run this command to build the .so file:
#python setup.py build_ext --inplace

from distutils.core import setup, Extension
import numpy
setup(name='_tifffile',
      ext_modules=[Extension('_tifffile', ['tifffile.c'],
                             include_dirs=[numpy.get_include()])])