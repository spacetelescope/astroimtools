# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Image utilities.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.nddata import NDData
from astropy.wcs import WCS


__all__ = ['imarith']


def imarith(nddata1, nddata2, operator):
    """
    Perform basic arithmetic on two `~astropy.nddata.NDData` objects and
    return a new `~astropy.nddata.NDData` object.
    """

    allowed_operators = ['+', '-', '*', '/', '//', 'min', 'max']
    operator = operator.strip()
    if operator not in allowed_operators:
        raise ValueError('operator "{0}" is not allowed'.format(operator))

    if nddata1.data.shape != nddata2.data.shape:
        raise ValueError('nddata1 and nddata2 arrays must have the same '
                         'shape')

    mdata1 = np.ma.masked_array(nddata1.data, mask=nddata1.mask)
    mdata2 = np.ma.masked_array(nddata2.data, mask=nddata2.mask)

    if operator in allowed_operators[:5]:
        data_expr = 'mdata1 {0} mdata2'.format(operator)
        mdata = eval(data_expr)
    elif operator == 'min':
        mdata = np.minimum(data1, data2)
    elif operator == 'max':
        mdata = np.maximum(data1, data2)

    nddata_out = NDData(mdata.data, mask=mdata.mask)

    return nddata_out
