#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages
import os
import io

setup_path = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(setup_path, 'README.rst'), 'r') as f:
    README = f.read()

setup(name='DESlib',
      version='0.1',
      url='https://github.com/Menelau/DESlib',
      maintainer='Rafael M. O. Cruz, L. G. Hafemann',
      maintainer_email='rafaelmenelau@gmail.com',
      description='Implementation of Dynamic Ensemble Selection methods',
      long_description=README,
      author='Rafael M. O. Cruz',
      author_email='rafaelmenelau@gmail.com',
      license='BSD 3-clause "New" or "Revised License"',

      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      install_requires=[
          'scikit-learn>=0.19.0',
          'numpy>=1.10.4',
          'scipy>=0.13.3',
      ],
      python_requires='>=3',      

      packages=find_packages())

