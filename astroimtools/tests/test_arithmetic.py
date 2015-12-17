# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.nddata import NDData
from ..arithmetic import nddata_arith


class TestNDDataArith(object):
    def setup_class(self):
        self.nd1 = NDData([0, 1, 2, 3, 4])
        self.nd2 = NDData([1, 7, 5, 4, 2])

    def test_nddata_arith(self):
        nd = nddata_arith(self.nd1, self.nd2, '+')
        ref = np.array([1, 8, 7, 7, 6])
        assert_allclose(nd.data, ref)

    def test_min(self):
        nd = nddata_arith(self.nd1, self.nd2, 'min')
        ref = np.array([0, 1, 2, 3, 2])
        assert_allclose(nd.data, ref)

    def test_max(self):
        nd = nddata_arith(self.nd1, self.nd2, 'max')
        ref = np.array([1, 7, 5, 4, 4])
        assert_allclose(nd.data, ref)

    def test_scalar(self):
        nd = nddata_arith(self.nd1, 5, '+')
        ref = np.array([5, 6, 7, 8, 9])
        assert_allclose(nd.data, ref)

        nd = nddata_arith(5, self.nd1, '*')
        ref = np.array([0, 5, 10, 15, 20])
        assert_allclose(nd.data, ref)

        nd = nddata_arith(self.nd1, 2, '/')
        ref = np.array([0., 0.5, 1., 1.5, 2.])
        assert_allclose(nd.data, ref)

        nd = nddata_arith(self.nd1, 2, '//')
        ref = np.array([0, 0, 1, 1, 2])
        assert_allclose(nd.data, ref)

    def test_metadata(self):
        self.nd1.meta['exptime'] = 500
        self.nd2.meta['exptime'] = 1000
        nd = nddata_arith(self.nd1, self.nd2, '+', keywords='exptime')
        assert_allclose(nd.meta['exptime'], 1500)

    def test_metadata_min_max(self):
        val1 = 500
        val2 = 1000
        self.nd1.meta['exptime'] = val1
        self.nd2.meta['exptime'] = val2
        nd = nddata_arith(self.nd1, self.nd2, 'min', keywords='exptime')
        assert nd.meta['exptime'] == val1
        nd = nddata_arith(self.nd1, self.nd2, 'max', keywords='exptime')
        assert nd.meta['exptime'] == val2

    def test_mask(self):
        self.nd1.mask = (self.nd1.data > 3)
        self.nd2.mask = (self.nd2.data < 2)
        nd = nddata_arith(self.nd1, self.nd2, '+')
        ref_data = np.array([0, 8, 7, 7, 0])
        ref_mask = np.array([True, False, False, False, True], dtype=bool)
        assert_allclose(nd.data, ref_data)
        assert_allclose(nd.mask, ref_mask)

    def test_invalid_scalar_inputs(self):
        with pytest.raises(ValueError):
            nddata_arith(4, 4, '+')

    def test_invalid_array_inputs(self):
        with pytest.raises(ValueError):
            nddata_arith(np.arange(3), self.nd1, '+')
        with pytest.raises(ValueError):
            nddata_arith(self.nd2, np.arange(3), '+')

    def test_invalid_nddata_shapes(self):
        nd2 = NDData(np.arange(10))
        with pytest.raises(ValueError):
            nddata_arith(self.nd1, nd2, '+')

    def test_invalid_operator(self):
        with pytest.raises(ValueError):
            nddata_arith(self.nd1, self.nd2, 'z')
