# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.utils.compat import argparse
from ..nddata_adapters import basic_fits_to_nddata, basic_nddata_to_fits
from ..core import imarith


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Perform basic arithmetic on two FITS files.')
    parser.add_argument('fits_filename',
                        nargs=2, help='FITS filenames')
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
    parser.add_argument('-c', '--clobber', default=False,
                        action='store_true', help='')

    args = parser.parse_args(args)

    # TODO: better FITS to NDData and NDData to FITS adapters
    nddata1 = basic_fits_to_nddata(args.fits_filename[0], exten=args.exten1)
    nddata2 = basic_fits_to_nddata(args.fits_filename[1], exten=args.exten2)

    keywords = None
    if args.keywords is not None:
        keywords = args.keywords.replace(' ', '').split(',')

    nddata = imarith(nddata1, nddata2, args.operator,
                     fill_value=args.fill_value, keywords=keywords)

    basic_nddata_to_fits(nddata, args.outfilename, clobber=args.clobber)
