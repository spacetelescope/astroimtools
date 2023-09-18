# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.nddata import NDData
from numpy.testing import assert_allclose

from ..utils import (listpixels, mask_databounds, nddata_cutout2d,
                     radial_distance)


class TestRadialDistance:
    def test_radial_distance(self):
        result = radial_distance((1, 1), (3, 3))
        x = np.sqrt(2)
        ref = np.array([[x, 1, x], [1, 0, 1], [x, 1, x]])
        assert_allclose(result, ref)

    def test_bad_position(self):
        with pytest.raises(ValueError):
            radial_distance((2, 2, 2), (1, 1))

    def test_bad_shape(self):
        with pytest.raises(ValueError):
            radial_distance((2, 2), (1, 1, 1))


class TestListPixels:
    def setup_class(self):
        self.data = np.arange(100).reshape(10, 10)

    def test_listpixels(self):
        tbl = listpixels(self.data, (5, 5), (2, 2))
        assert len(tbl) == 4
        assert len(tbl.colnames) == 3
        assert_allclose(tbl['x'].data, [4, 5, 4, 5])
        assert_allclose(tbl['y'].data, [4, 4, 5, 5])
        assert_allclose(tbl['value'].data, [44, 45, 54, 55])

    def test_subarray(self):
        tbl = listpixels(self.data, (5, 5), (2, 2), subarray_indices=True)
        assert len(tbl) == 4
        assert len(tbl.colnames) == 3
        assert_allclose(tbl['x'].data, [0, 1, 0, 1])
        assert_allclose(tbl['y'].data, [0, 0, 1, 1])
        assert_allclose(tbl['value'].data, [44, 45, 54, 55])


class TestMaskDataBounds:
    def test_mask_databounds(self):
        data = np.arange(7)
        ref = np.array([True, True, False, True, False, False, True])
        result = mask_databounds(data, lower_bound=2, upper_bound=5, value=3)
        assert_allclose(result, ref)

    def test_invalid_mask_shape(self):
        with pytest.raises(ValueError):
            mask_databounds(np.arange(5), mask=np.arange(3).astype(bool))

    def test_all_masked(self):
        with pytest.raises(ValueError):
            mask_databounds(np.arange(5), lower_bound=10)


class TestNDDataCutout2D:
    def test_nddata_cutout2d(self):
        data = np.random.random((100, 100))
        unit = u.electron / u.s
        mask = (data > 0.7)
        meta = {'exptime': 1234 * u.s}
        nddata = NDData(data, mask=mask, unit=unit, meta=meta)
        shape = (10, 10)
        cutout = nddata_cutout2d(nddata, (50, 50), shape)
        assert cutout.data.shape == shape
        assert cutout.mask.shape == shape
        assert cutout.unit == unit

    def test_not_nddata(self):
        with pytest.raises(TypeError):
            nddata_cutout2d(np.ones((10, 10)), (5, 5), (2, 2))

    def test_skycoord_no_wcs(self):
        pos = SkyCoord('13h11m29.96s -01d19m18.7s', frame='icrs')
        nddata = NDData(np.ones((10, 10)))
        with pytest.raises(ValueError):
            nddata_cutout2d(nddata, pos, (2, 2))
