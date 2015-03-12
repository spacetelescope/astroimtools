# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import astropy.io.fits as fits
from astropy.table import Column
from astropy.nddata import NDData
from astropy.utils.compat import argparse
from ..stats import imstats


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Calculate image statistics.')
    parser.add_argument('fits_filename', metavar='fits_filename',
                        nargs='*', help='FITS filename(s)')
    parser.add_argument('-s', '--sigma', metavar='sigma', type=float,
                        default=3., help=('The number of standard '
                        'deviations to use as the clipping limit'))
    parser.add_argument('-i', '--iters', metavar='iters', type=int,
                        default=1, help='')
    parser.add_argument('-c', '--columns', metavar='columns', type=str,
                        default='npixels, mean, std, min, max', help='')
    parser.add_argument('--lower', metavar='lower', type=float,
                        default=None, help='')
    parser.add_argument('--upper', metavar='upper', type=float,
                        default=None, help='')
    args = parser.parse_args(args)

    # TODO: need FITS file to NDData object adapter
    nddata = []
    for fits_fn in args.fits_filename:
        hdulist = fits.open(fits_fn)
        header = hdulist[0].header
        data = hdulist[0].data
        nddata.append(NDData(data, meta=header))

    columns = args.columns.replace(' ', '').split(',')
    tbl = imstats(nddata, sigma=args.sigma, iters=args.iters,
                  cenfunc=np.ma.median, varfunc=np.var, columns=columns,
                  lower_bound=args.lower, upper_bound=args.upper)

    if 'filename' in columns:
        filenames = Column(args.fits_filename, name='filename')
        tbl.add_column(filenames, 0)

    tbl.pprint(max_lines=-1, max_width=-1)
