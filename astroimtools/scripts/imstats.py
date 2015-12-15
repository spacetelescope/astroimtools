# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
``imstats`` is a command-line script based on
``astroimtools.nddata_stats`` to calculate various statistics of an
array from a single extension of a FITS file.  Sigma-clipped statistics
can be calculated by specifying the ``sigma`` option.

The currently available statistics are:

  * ``'mean'``
  * ``'median'``
  * ``'mode'``
  * ``'std'``
  * ``'mad_std'``
  * ``'npixels'`` (for sigma-clipped statistics)
  * ``'nrejected'`` (for sigma-clipped statistics)
  * ``'min'``
  * ``'max'``
  * ``'biweight_location'``
  * ``'biweight_midvariance'``
  * ``'kurtosis'``
  * ``'skew'``

Example usage:

1.  Calculate simple image statistics on the data in the first FITS
    extension::

    $ imstats filename.fits -e 1

    ::

       filename    npixels   mean    std     min      max
     ------------- ------- -------- ------ -------- -------
     filename.fits 1027865 0.438514 0.6974 -59.8228 65.8734

2.  Same as above, but specify different statistics::

    $ imstats filename.fits -e 1 --columns 'mean, mode, std, mad_std'

    ::

       filename      mean    mode   std   mad_std
     ------------- -------- ------ ------ -------
     filename.fits 0.438514 0.4271 0.6974 0.68134


3.  Calculate sigma-clipped (at 3 standard deviations) image statistics
    on the data in the first FITS extension::

    $ imstats filename.fits -e 1 -sigma 3

    ::

       filename    npixels   mean    std     min      max
     ------------- ------- -------- ------ -------- -------
     filename.fits 1020413 0.425871 0.5742 -39.2302 55.2304
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.table import Column
from astropy.utils.compat import argparse
from ..stats import nddata_stats
from ..nddata_adapters import basic_fits_to_nddata


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Calculate image statistics.')
    parser.add_argument('fits_filename', metavar='fits_filename',
                        nargs='*', help='FITS filename(s)')
    parser.add_argument('-e', '--exten', metavar='exten', type=int,
                        default=0, help='')
    parser.add_argument('-s', '--sigma', metavar='sigma', type=float,
                        default=3., help='The number of standard '
                        'deviations to use as the clipping limit')
    parser.add_argument('-i', '--iters', metavar='iters', type=int,
                        default=1, help='')
    parser.add_argument('-c', '--columns', metavar='columns', type=str,
                        default='npixels, mean, std, min, max',
                        help='')
    parser.add_argument('-l', '--lower', metavar='lower', type=float,
                        default=None, help='')
    parser.add_argument('-u', '--upper', metavar='upper', type=float,
                        default=None, help='')
    parser.add_argument('-m', '--mask_value', metavar='mask_value',
                        type=float, default=None, help='')
    args = parser.parse_args(args)

    # TODO: better FITS to NDData object adapters!
    nddata = []
    for fits_fn in args.fits_filename:
        nddata.append(basic_fits_to_nddata(fits_fn, exten=args.exten))

    columns = args.columns.replace(' ', '').split(',')
    tbl = nddata_stats(nddata, sigma=args.sigma, iters=args.iters,
                       cenfunc=np.ma.median, varfunc=np.var, columns=columns,
                       lower_bound=args.lower, upper_bound=args.upper,
                       mask_value=args.mask_value)

    filenames = Column(args.fits_filename, name='filename')
    tbl.add_column(filenames, 0)

    tbl.pprint(max_lines=-1, max_width=-1)
