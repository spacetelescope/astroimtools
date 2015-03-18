# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools reading FITS files into NDData objects.
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

    hdu = fits.ImageHDU(data=nddata.data, header=nddata.meta)
    hdu.header['EXTNAME'] = 'SCI'
    hdus = [hdu]

    if nddata.uncertainty is not None:
        hdu = fits.ImageHDU(data=nddata.uncertainty.value)
        hdu.header['EXTNAME'] = 'ERR'
        hdus.append(hdu)

    if nddata.dq is not None:
        if nddata.mask is not None:
            nddata.dq |= nddata.mask.astype(np.int)
        hdu = fits.ImageHDU(data=nddata.dq)
        hdu.header['EXTNAME'] = 'DQ'
        hdus.append(hdu)
    else:
        if nddata.mask is not None:
            hdu = fits.ImageHDU(data=nddata.mask.astype(np.int))
            hdu.header['EXTNAME'] = 'DQ'
            hdus.append(hdu)

    hdulist = fits.HDUList(hdus)
    hdulist.writeto(filename, clobber=clobber)

    return
