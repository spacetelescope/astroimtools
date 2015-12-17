# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose
from astropy.nddata import NDData
from ..stats import (minmax, NDDataStats, nddata_stats)


class TestMinMax(object):
    def setup_class(self):
        np.random.seed(12345)
        self.data = np.random.random((3, 3))

    def test_minmax(self):
        ref = (0.18391881167709445, 0.96451451973562163)
        assert_allclose(minmax(self.data), ref)

    def test_mask(self):
        mask = (self.data < 0.3)
        ref = (0.3163755545817859, 0.96451451973562163)
        assert_allclose(minmax(self.data, mask=mask), ref)

    def test_axis(self):
        ref = (np.array([0.18391881, 0.20456028, 0.6531771]),
               np.array([0.92961609, 0.5955447, 0.96451452]))
        assert_allclose(minmax(self.data, axis=1), ref)


def test_nddata_stats_class():
    nddata = NDData(np.arange(10))
    stats = NDDataStats(nddata)
    assert_allclose(stats.mean, 4.5)
    assert_allclose(stats.median, 4.5)
    assert_allclose(stats.std, 2.8722813232690143)
    assert_allclose(stats.mad_std, 3.7065055462640051)


def test_nddata_stats_func():
    nddata = NDData(np.arange(10))
    columns = ['mean', 'median', 'mode', 'std', 'mad_std', 'min', 'max']

    columns = ['biweight_location', 'biweight_midvariance', 'kurtosis',
               'mad_std', 'max', 'mean', 'median', 'min', 'mode', 'npixels',
               'nrejected', 'skew', 'std']
    tbl = nddata_stats(nddata, columns=columns)
    assert len(tbl) == 1
    assert tbl.colnames == columns
    row = tbl[0]
    assert_allclose(row['mean'], 4.5)
    assert_allclose(row['median'], 4.5)
    assert_allclose(row['std'], 2.8722813232690143)
    assert_allclose(row['mad_std'], 3.7065055462640051)
    assert_allclose(row['min'], 0.)
    assert_allclose(row['max'], 9.)


def test_nddata_stats_func_2rows():
    nddata = NDData(np.arange(10))
    columns = ['mean', 'median', 'mode', 'std', 'mad_std', 'min', 'max']
    tbl = nddata_stats([nddata, nddata], columns=columns)
    assert len(tbl) == 2
    assert tbl.colnames == columns
    row1 = tbl[0]
    row2 = tbl[1]
    assert_allclose(row1['mean'], row2['mean'])
    assert_allclose(row1['median'], row2['median'])
    assert_allclose(row1['std'], row2['std'])
    assert_allclose(row1['mad_std'], row2['mad_std'])
    assert_allclose(row1['min'], row2['min'])
    assert_allclose(row1['max'], row2['max'])
