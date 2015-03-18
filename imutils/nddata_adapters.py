# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
NDData tools for interfacing with FITS files.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import astropy.io.fits as fits
from astropy.nddata import NDData


__all__ = ['basic_fits_to_nddata', 'basic_nddata_to_fits']


def basic_fits_to_nddata(filename, exten=0):
    """
    Read a single FITS extension into a `~astropy.nddata.NDData` object.

    This is an *extremely* simple reader that reads data from only a
    single FITS extension.

    Parameters
    ----------
    filename : str
        The path to a FITS file.

    exten : int, optional
        The FITS extension number for the ``data`` array.  Default is 0.

    Returns
    -------
    nddata : `~astropy.nddata.NDData`
        An `~astropy.nddata.NDData` object with a ``data`` attribute
        containing the FITS data array and a ``meta`` attribute,
        containing the FITS header as a python `dict`.
    """

    hdulist = fits.open(filename)
    header = hdulist[exten].header
    data = hdulist[exten].data
    return NDData(data, meta=header)


def basic_nddata_to_fits(nddata, filename, clobber=False):
    """
    Write a `~astropy.nddata.NDData` object to a FITS file.
    """

    if nddata.meta is not None:
        hdu = fits.PrimaryHDU(header=fits.Header(nddata.meta))
    else:
        hdu = fits.PrimaryHDU()
    hdus = [hdu]

    hdus.append(fits.ImageHDU(data=nddata.data))
    hdus[-1].header['EXTNAME'] = 'SCI'

    if nddata.uncertainty is not None:
        hdus.append(fits.ImageHDU(data=nddata.uncertainty.value))
        hdus[-1].header['EXTNAME'] = 'ERR'

    if nddata.mask is not None:
        hdus.append(fits.ImageHDU(data=nddata.mask.astype(np.int)))
        hdus[-1].header['EXTNAME'] = 'DQ'

    hdulist = fits.HDUList(hdus)
    hdulist.writeto(filename, clobber=clobber)

    return
