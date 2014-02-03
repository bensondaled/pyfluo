from setuptools import setup, find_packages
readme = open('README.txt').read()

setup(	name = 'PyFluo',
		version = '1.0',
		author = 'Ben Deverett',
		author_email = 'bendeverett@gmail.com',
		description = 'A python library for Ca imaging data analysis.',
		long_description = readme,
		packages = find_packages()
		)