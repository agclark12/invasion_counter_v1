#!/opt/local/bin/python

__author__ = "Andrew G. Clark"
__date__ = "2015"
__copyright__ = "Copyright 2015, Andrew Clark"
__maintainer__ = "Andrew G. Clark"
__email__ = "andrew.clark@curie.fr"
__status__ = "Production"

"""
builds an executable for windows

note: you must add msvcp90.dll to /c/Anaconda/DLLs/ (available at http://www.microsoft.com/en-us/download/details.aspx?id=29)
the extra imports are to ensure that everything is packaged correctly (otherwise some packages are not included)

"""

from py2exe.build_exe import py2exe
from distutils.core import setup

import os
import zmq
import numpy
import matplotlib
import scipy

# libzmq.dll is in same directory as zmq's __init__.py

os.environ["PATH"] = \
    os.environ["PATH"] + \
    os.path.pathsep + os.path.split(zmq.__file__)[0]

setup( console=[{"script": "run_invasion_counter_gui.py"}],
       options={
           "py2exe": {
               "includes":
               ["zmq.utils", "zmq.utils.jsonapi",
                "zmq.utils.strtypes",
                "scipy.special._ufuncs_cxx",
                "matplotlib.backends.backend_qt4agg",
                "scipy.sparse.csgraph._validation"],
           }
       },
       data_files=matplotlib.get_py2exe_datafiles(),
    )
