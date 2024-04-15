#!/usr/bin/env python

import codecs
import os
from distutils.core import setup

from setuptools import find_packages

setup_path = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(setup_path, 'README.rst'), encoding='utf-8-sig') as f:
    README = f.read()

setup(name='DESlib',
      version='0.3.7',
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
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      install_requires=[
          'scikit-learn>=1.0.2',
          'numpy>=1.17.0',
          'scipy>=1.4.0',
      ],
      python_requires='>=3',      

      packages=find_packages())

