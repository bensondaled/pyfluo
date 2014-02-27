#from distutils.core import setup, Extension #it's either this line (distutils) or the next (setuptools). both work, using setuptools because it supports develop mode
from setuptools import find_packages, setup, Extension
from distutils.sysconfig import get_python_inc
readme = open('README.txt').read()
import os

incdir = os.path.join(get_python_inc(plat_specific=1), 'Numerical')

setup(	name = 'PyFluo',
		version = '1.0',
		author = 'Ben Deverett',
		author_email = 'bendeverett@gmail.com',
		url = 'https://github.com/bensondaled/pyfluo',
		description = 'A python library for Ca imaging data analysis.',
		long_description = readme,
		keywords = 'pyfluo fluorescence calcium ca imaging',
		packages = ['pyfluo'],
		data_files = [	('', ['LICENSE.txt']),
						('', ['README.txt']),
												],
		install_requires = ['numpy', 'scipy', 'matplotlib'], 
		ext_modules=[ Extension('_tifffile', ['pyfluo/tiff/_installation/tifffile.c'],include_dirs=[incdir]) ]		)
