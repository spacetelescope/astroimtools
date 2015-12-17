# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import itertools
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest, assert_quantity_allclose
from astropy.modeling import models
from astropy.table import Table
import astropy.units as u
from astropy.utils.misc import isiterable
import astropy.wcs as WCS
from ..arithmetic import nddata_arith

class TestNDDataArith(object):
	def setup_class(self):
		self.data = np.arange(10)

