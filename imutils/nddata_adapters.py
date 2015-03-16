# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools reading FITS files into NDData objects.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import astropy.io.fits as fits
from astropy.nddata import NDData


__all__ = ['basic_fits_to_nddata']


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
