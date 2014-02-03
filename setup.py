from distutils.core import setup, Extension
#from setuptools import find_packages, setup, Extension
import numpy
readme = open('README.txt').read()

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
		ext_modules=[ Extension('_tifffile', ['pyfluo/tiff/_installation/tifffile.c'],include_dirs=[numpy.get_include()]) ]		)