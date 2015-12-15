# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
NDData tools for interfacing with FITS files.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import astropy.io.fits as fits
from astropy.nddata import NDData
from astropy import log


__all__ = ['basic_fits_to_nddata', 'basic_nddata_to_fits']


def basic_fits_to_nddata(filename, exten=0):
    """
    Read a single FITS extension into a `~astropy.nddata.NDData` object.

    This is an *extremely* simple reader that reads data from only a
    single FITS extension.

    Note the the primary FITS header will always be included in the
    `~astropy.nddata.NDData` meta `dict`, regardless of the value of
    ``exten``.

    Parameters
    ----------
    filename : str
        The path to a FITS file.

    exten : int, optional
        The FITS extension number for array to place in the NDData
        object.  The default is 0.

    Returns
    -------
    nddata : `~astropy.nddata.NDData`
        An `~astropy.nddata.NDData` object with a ``data`` attribute
        containing the FITS data array and a ``meta`` attribute,
        containing the FITS header as a python `dict`.
    """

    with fits.open(filename) as hdulist:
        header = hdulist[0].header
        header += hdulist[exten].header
        data = hdulist[exten].data
    return NDData(data, meta=header)


def basic_nddata_to_fits(nddata, filename, clobber=False):
    """
    Write a `~astropy.nddata.NDData` object to a FITS file.

    The `~astropy.nddata.NDData` data will be saved in a FITS extension
    called 'SCI'.  This simple writer will also attempt to save the
    `~astropy.nddata.NDData` uncertainty and mask to an 'ERROR' and
    'MASK' FITS extension, respectively.

    If present, the `~astropy.nddata.NDData` meta dictionary will be
    stored as the FITS header.

    Parameters
    ----------
    nddata : `~astropy.nddata.NDData`
        An `~astropy.nddata.NDData` object to write to a FITS file.

    filename : str
        The path of the output FITS file.

    clobber : bool, optional
        Set to `True` to overwrite ``filename`` if it already exists.
        The default is `False`.
    """

    hdu = fits.PrimaryHDU()
    if nddata.meta is not None:
        for k, v in nddata.meta.iteritems():
            hdu.header[k] = v
    hdus = [hdu]

    hdus.append(fits.ImageHDU(data=nddata.data))
    hdus[-1].header['EXTNAME'] = 'SCI'

    if nddata.uncertainty is not None:
        hdus.append(fits.ImageHDU(data=nddata.uncertainty.value))
        hdus[-1].header['EXTNAME'] = 'ERROR'

    if nddata.mask is not None:
        hdus.append(fits.ImageHDU(data=nddata.mask.astype(np.int)))
        hdus[-1].header['EXTNAME'] = 'MASK'

    hdulist = fits.HDUList(hdus)
    hdulist.writeto(filename, clobber=clobber)
    log.info('Wrote {0}'.format(filename))

    return
