# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools reading FITS files into NDData objects.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import astropy.io.fits as fits
from astropy.nddata import NDData


__all__ = ['basic_fits_to_nddata']


def basic_fits_to_nddata(fits_filename, exten=0):
    """
    Extremely simple reader.

    Q:  default mask from ERR and DQ arrays (bad pixels and coverage mask)
    """

    hdulist = fits.open(fits_filename)
    header = hdulist[exten].header
    data = hdulist[exten].data
    return NDData(data, meta=header)
