# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pytest
from astropy.nddata import NDData
from astropy.utils import minversion
from numpy.testing import assert_allclose

from ..stats import NDDataStats, minmax, nddata_stats

try:
    import scipy  # noqa: F401
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TestMinMax:
    def setup_class(self):
        rng = np.random.default_rng(12345)
        self.data = rng.uniform(0, 1, (3, 3))

    def test_minmax(self):
        ref = np.min(self.data), np.max(self.data)
        assert_allclose(minmax(self.data), ref)

    def test_mask(self):
        mask = (self.data < 0.3)
        data2 = self.data[self.data >= 0.3]
        ref = (np.min(data2), np.max(data2))
        assert_allclose(minmax(self.data, mask=mask), ref)

    def test_axis(self):
        ref = (np.min(self.data, axis=1), np.max(self.data, axis=1))
        assert_allclose(minmax(self.data, axis=1), ref)


def test_nddata_stats_class():
    nddata = NDData(np.arange(10))
    stats = NDDataStats(nddata)
    assert_allclose(stats.mean, 4.5)
    assert_allclose(stats.median, 4.5)
    assert_allclose(stats.std, 2.8722813232690143)
    assert_allclose(stats.mad_std, 3.7065055462640051)


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.skipif(minversion(np, '1.25.0'), reason='numpy 1.25 deprecation')
def test_nddata_stats_func():
    nddata = NDData(np.arange(10))
    columns = ['mean', 'median', 'mode', 'std', 'mad_std', 'min', 'max']

    columns = ['biweight_location', 'biweight_midvariance', 'kurtosis',
               'mad_std', 'max', 'mean', 'median', 'min', 'mode', 'npixels',
               'nrejected', 'skew', 'std']

    # Numpy 1.25 deprecation warning coming from
    # scipy/stats/_stats_py.py:1069
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
