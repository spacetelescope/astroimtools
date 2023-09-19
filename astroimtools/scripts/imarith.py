# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
``imarith`` is a command-line script based on
``astroimtools.nddata_arith`` for performing basic arithmetic on arrays
extracted from two FITS files.  The arithmetic is performed for the data
in only a single FITS extension in both FITS file.

.. highlight:: none

Example usage:

1.  Add the array in extension 1 of the first FITS file with extension 3
    in the second FITS file, placing the result in "result.fits"::

    $ imarith filename1.fits filename2.fits '+' -e1 1 -e2 3 -o 'result.fits'

2.  Perform the same operation above, but also add the exposure time
    values in the headers of the FITS files, placing the result in
    "result.fits"::

    $ imarith filename1.fits filename2.fits '+' -e1 1 -e2 3 -k 'exptime' -o 'result.fits'

.. highlight:: python3
"""

import argparse

from astropy.nddata import NDData

from ..arithmetic import nddata_arith
from ..nddata_adapters import basic_fits_to_nddata, basic_nddata_to_fits


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Perform basic arithmetic on two FITS files.')
    parser.add_argument('fits_filename',
                        nargs=2, help='FITS filename (or scalar value)')
    parser.add_argument('-e1', '--exten1', metavar='exten1', type=int,
                        default=0, help='')
    parser.add_argument('-e2', '--exten2', metavar='exten2', type=int,
                        default=0, help='')
    parser.add_argument('operator', metavar='operator',
                        help="Arithmetic operator.  Must be one "
                        "of '+', '-', '*', '/', '//', 'min', or 'max'")
    parser.add_argument('-f', '--fill_value', metavar='fill_value',
                        type=float, default=0., help='')
    parser.add_argument('-k', '--keywords', metavar='keywords', type=str,
                        default=None, help='')
    parser.add_argument('-o', '--outfilename', metavar='outfilename',
                        type=str, default='imarith.fits', help='')
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='Overwrite existing file')

    args = parser.parse_args(args)

    # TODO: we need better FITS to NDData and NDData to FITS adapters
    try:
        nddata1 = float(args.fits_filename[0])
    except ValueError:
        nddata1 = basic_fits_to_nddata(args.fits_filename[0],
                                       exten=args.exten1)
    try:
        nddata2 = float(args.fits_filename[1])
    except ValueError:
        nddata2 = basic_fits_to_nddata(args.fits_filename[1],
                                       exten=args.exten2)

    if not isinstance(nddata1, NDData) and not isinstance(nddata2, NDData):
        raise ValueError('Both "fits_filenames" cannot be scalars.')

    keywords = None
    if args.keywords is not None:
        keywords = args.keywords.replace(' ', '').split(',')

    nddata = nddata_arith(nddata1, nddata2, args.operator,
                          fill_value=args.fill_value, keywords=keywords)

    basic_nddata_to_fits(nddata, args.outfilename, overwrite=args.overwrite)
