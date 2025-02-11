#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
import subprocess
try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup


setup(
    name="uberMS_binary",
    url="https://github.com/Lafayette-Cool-Stars/uberMS_binary",
    version="0.0",
    author="Phillip Cargile",
    author_email="pcargile@cfa.harvard.edu",
    packages=["uberMS_binary",
	      "uberMS_binary.binary",
              "uberMS_binary.spots",
              "uberMS_binary.dva",
              "uberMS_binary.smes",
              "uberMS_binary.utils"],
    license="LICENSE",
    description="Optimized MINESweeper",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    install_requires=["Payne", "misty", "astropy", "numpyro", "jax", "optax"],
)

# write top level __init__.py file with the correct absolute path to package repo
toplevelstr = ("""try:
    from ._version import __version__
except(ImportError):
    pass

from . import spots
from . import dva
from . import utils"""
)

with open('uberMS_binary/__init__.py','w') as ff:
  ff.write(toplevelstr)
  ff.write('\n')
  ff.write("""__abspath__ = '{0}/'\n""".format(os.getcwd()))
