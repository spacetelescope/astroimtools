# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Image utilities.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import copy
from astropy.table import Table
from astropy.nddata import NDData, support_nddata
from astropy.nddata.utils import overlap_slices
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.utils import lazyproperty
from astropy import log
import warnings
from astropy.utils.exceptions import AstropyUserWarning


__all__ = ['StdUncertainty', 'imarith']


warnings.filterwarnings('always', category=AstropyUserWarning)


class StdUncertainty(object):
    """
    `~astropy.nddata.NDData` uncertainty class to hold 1-sigma standard
    deviations.
    """

    def __init__(self, value):
        self.value = value

    @property
    def uncertainty_type(self):
        return 'std'


def imarith(nddata1, nddata2, operator, fill_value=0.0, keywords=None):
    """
    Perform basic arithmetic on two `~astropy.nddata.NDData` objects and
    return a new `~astropy.nddata.NDData` object.

    Parameters
    ----------
    nddata1 : `~astropy.nddata.NDData` or scalar
        ``nddata1`` and ``nddata2`` cannot both be scalars.

    nddata2 : `~astropy.nddata.NDData` or scalar
        ``nddata1`` and ``nddata2`` cannot both be scalars.
    """

    allowed_operators = ['+', '-', '*', '/', '//', 'min', 'max']
    operator = operator.strip()
    if operator not in allowed_operators:
        raise ValueError('operator "{0}" is not allowed'.format(operator))

    if not isinstance(nddata1, NDData) and not isinstance(nddata2, NDData):
        raise ValueError('nddata1 or nddata2 input must be an '
                         'astropy.nddata.NDData object.')

    # if nddata1 is a scalar, then make it a NDData object
    if not isinstance(nddata1, NDData):
        nddata1 = np.asanyarray(nddata1)
        if nddata1.size != 1:
            raise ValueError('nddata1 input must be an astropy.nddata.NDData '
                             'object or a scalar.')
        nddata1 = NDData(np.full(nddata2.data.shape, nddata1))

    # if nddata2 is a scalar, then make it a NDData object
    if not isinstance(nddata2, NDData):
        nddata2 = np.asanyarray(nddata2)
        if nddata2.size != 1:
            raise ValueError('nddata2 input must be an astropy.nddata.NDData '
                             'object or a scalar.')
        nddata2 = NDData(np.full(nddata1.data.shape, nddata2))

    if nddata1.data.shape != nddata2.data.shape:
        raise ValueError('nddata1 and nddata2 arrays must have the same '
                         'shape')

    if keywords is not None:
        keywords = np.atleast_1d(keywords)

    mdata1 = np.ma.masked_array(nddata1.data, mask=nddata1.mask)
    mdata2 = np.ma.masked_array(nddata2.data, mask=nddata2.mask)

    if operator in allowed_operators[:5]:
        data_expr = 'mdata1 {0} mdata2'.format(operator)
        mdata = eval(data_expr)
    elif operator == 'min':
        mdata = np.minimum(mdata1, mdata2)
    elif operator == 'max':
        mdata = np.maximum(mdata1, mdata2)

    # keyword arithmetic
    meta_out = copy.deepcopy(nddata1.meta)
    if keywords is not None:
        for key in keywords:
            value1 = nddata1.meta.get(key, None)
            value2 = nddata2.meta.get(key, None)
            if value1 is not None and value2 is not None:
                if operator in allowed_operators[:5]:
                    hdr_expr = 'value1 {0} value2'.format(operator)
                    value = eval(hdr_expr)
                elif operator == 'min':
                    value = min(value1, value2)
                elif operator == 'max':
                    value = max(value1, value2)
                meta_out[key] = value

    # propagate errors
    if nddata1.uncertainty is not None and nddata2.uncertainty is not None:
        if operator in ['+', '-']:
            error_out = np.sqrt(nddata1.uncertainty.value**2 +
                                nddata2.uncertainty.value**2)
        elif operator in ['*', '/']:
            error_out = mdata * np.sqrt((nddata1.uncertainty.value /
                                         mdata1)**2 +
                                        (nddata2.uncertainty.value /
                                         mdata2)**2)
        else:
            log.info("Error propagation is not performed for the '//', "
                     "'min', and 'max' operators.")
            error_out = None

        if error_out is not None:
            uncertainty_out = copy.deepcopy(nddata1.uncertainty)
            uncertainty_out.value = error_out
        else:
            uncertainty_out = None
    else:
        uncertainty_out = None

    return NDData(np.ma.filled(mdata, fill_value=fill_value),
                  uncertainty=uncertainty_out, mask=mdata.mask, meta=meta_out)
