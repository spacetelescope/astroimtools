# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
NDData tools for interfacing with FITS files.
"""

import re

from astropy import log
from astropy.io import fits
from astropy.nddata import NDData
from astropy.utils.decorators import deprecated_renamed_argument

__all__ = ['basic_fits_to_nddata', 'basic_nddata_to_fits']

# FITS keywords that describe the structure or the checksums of an HDU.
# They are set by astropy or by basic_nddata_to_fits when the HDUs are
# written.
_STRUCTURAL_KEYWORDS = {'SIMPLE', 'BITPIX', 'EXTEND', 'XTENSION', 'PCOUNT',
                        'GCOUNT', 'EXTNAME', 'EXTVER', 'EXTLEVEL',
                        'CHECKSUM', 'DATASUM'}


def _is_structural_keyword(keyword):
    keyword = keyword.upper()
    return (keyword in _STRUCTURAL_KEYWORDS
            or re.fullmatch(r'NAXIS\d*', keyword) is not None)


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
        # Copy the primary header so that the extension keywords (e.g.,
        # EXTNAME) are not added to the primary HDU, which would change
        # the lookup of the extension by name
        header = hdulist[0].header.copy()
        header += hdulist[exten].header
        data = hdulist[exten].data
    return NDData(data, meta=header)


@deprecated_renamed_argument('clobber', 'overwrite', '0.4')
def basic_nddata_to_fits(nddata, filename, overwrite=False):
    """
    Write a `~astropy.nddata.NDData` object to a FITS file.

    The `~astropy.nddata.NDData` data will be saved in a FITS extension
    called 'SCI'.  This simple writer will also attempt to save the
    `~astropy.nddata.NDData` uncertainty and mask to an 'ERROR' and
    'MASK' FITS extension, respectively.

    If present, the `~astropy.nddata.NDData` meta dictionary will be
    stored in the primary FITS header. Meta keywords that describe the
    structure of a FITS extension (e.g., EXTNAME, XTENSION, or NAXIS)
    are not copied because they are set by this writer.

    Parameters
    ----------
    nddata : `~astropy.nddata.NDData`
        An `~astropy.nddata.NDData` object to write to a FITS file.

    filename : str
        The path of the output FITS file.

    overwrite : bool, optional
        Set to `True` to overwrite ``filename`` if it already exists.
        The default is `False`.
    """
    hdu = fits.PrimaryHDU()
    if nddata.meta is not None:
        for k, v in nddata.meta.items():
            if not _is_structural_keyword(k):
                hdu.header[k] = v
    hdus = [hdu]

    hdus.append(fits.ImageHDU(data=nddata.data))
    hdus[-1].header['EXTNAME'] = 'SCI'

    if nddata.uncertainty is not None:
        hdus.append(fits.ImageHDU(data=nddata.uncertainty.array))
        hdus[-1].header['EXTNAME'] = 'ERROR'

    if nddata.mask is not None:
        hdus.append(fits.ImageHDU(data=nddata.mask.astype(int)))
        hdus[-1].header['EXTNAME'] = 'MASK'

    hdulist = fits.HDUList(hdus)
    hdulist.writeto(filename, overwrite=overwrite)
    log.info(f'Wrote {filename}')
