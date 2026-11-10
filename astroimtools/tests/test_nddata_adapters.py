# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import NDData, StdDevUncertainty
from numpy.testing import assert_equal

from ..nddata_adapters import basic_fits_to_nddata, basic_nddata_to_fits


@pytest.fixture
def nddata():
    rng = np.random.default_rng(0)
    data = rng.normal(10.0, 1.0, (5, 6))
    uncertainty = StdDevUncertainty(np.full(data.shape, 0.5))
    mask = data > 11.0
    meta = {'OBJECT': 'test', 'EXPTIME': 100.0}
    return NDData(data, uncertainty=uncertainty, mask=mask, meta=meta)


def test_round_trip(nddata, tmp_path):
    filename = tmp_path / 'test.fits'
    basic_nddata_to_fits(nddata, filename)

    with fits.open(filename) as hdulist:
        assert [hdu.name for hdu in hdulist] == ['PRIMARY', 'SCI', 'ERROR',
                                                 'MASK']
        assert hdulist[0].header['OBJECT'] == 'test'
        assert hdulist[0].header['EXPTIME'] == 100.0
        assert_equal(hdulist['ERROR'].data, nddata.uncertainty.array)
        assert_equal(hdulist['MASK'].data, nddata.mask.astype(int))

    nddata2 = basic_fits_to_nddata(filename, exten=1)
    assert_equal(nddata2.data, nddata.data)

    # The primary header is always included in the output meta
    assert nddata2.meta['OBJECT'] == 'test'
    assert nddata2.meta['EXPTIME'] == 100.0
    assert nddata2.meta['EXTNAME'] == 'SCI'

    # Write the NDData object read from the FITS file
    filename2 = tmp_path / 'test2.fits'
    basic_nddata_to_fits(nddata2, filename2)
    nddata3 = basic_fits_to_nddata(filename2, exten=1)
    assert_equal(nddata3.data, nddata.data)
    assert nddata3.meta['OBJECT'] == 'test'


def test_data_only(tmp_path):
    filename = tmp_path / 'test.fits'
    basic_nddata_to_fits(NDData(np.arange(12.0).reshape(3, 4)), filename)
    with fits.open(filename) as hdulist:
        assert [hdu.name for hdu in hdulist] == ['PRIMARY', 'SCI']


def test_primary_extension(tmp_path):
    filename = tmp_path / 'test.fits'
    data = np.arange(12.0).reshape(3, 4)
    fits.PrimaryHDU(data).writeto(filename)
    nddata = basic_fits_to_nddata(filename)
    assert_equal(nddata.data, data)


def test_overwrite(nddata, tmp_path):
    filename = tmp_path / 'test.fits'
    basic_nddata_to_fits(nddata, filename)
    with pytest.raises(OSError, match='already exists'):
        basic_nddata_to_fits(nddata, filename)
    basic_nddata_to_fits(nddata, filename, overwrite=True)
