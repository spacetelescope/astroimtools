# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys

__minimum_python_version__ = '3.5'
__minimum_numpy_version__ = '1.11'


class UnsupportedPythonError(Exception):
    pass


if (sys.version_info <
        tuple((int(val) for val in __minimum_python_version__.split('.')))):
    raise UnsupportedPythonError('Photutils does not support Python < {}'
                                 .format(__minimum_python_version__))


if not _ASTROPY_SETUP_:
    # For egg_info test builds to pass, put package imports here.
    from .arithmetic import *
    from .cutout_tools import *
    from .filtering import *
    from .nddata_adapters import *
    from .scripts import imarith, imstats
    from .stats import *
    from .utils import *
