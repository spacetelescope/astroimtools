# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Image utilities.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.nddata import NDData
from astropy.wcs import WCS
import copy


__all__ = ['imarith']


def imarith(nddata1, nddata2, operator, fill_value=0.0, header_keywords=None):
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

    if header_keywords is not None:
        header_keywords = np.atleast_1d(header_keywords)

    mdata1 = np.ma.masked_array(nddata1.data, mask=nddata1.mask)
    mdata2 = np.ma.masked_array(nddata2.data, mask=nddata2.mask)

    meta_out = copy.deepcopy(nddata1.meta)
    if operator in allowed_operators[:5]:
        data_expr = 'mdata1 {0} mdata2'.format(operator)
        mdata = eval(data_expr)
        if header_keywords is not None:
            for key in header_keywords:
                value1 = nddata1.meta.get(key, None)
                value2 = nddata2.meta.get(key, None)
                if value1 is not None and value2 is not None:
                    hdr_expr = 'value1 {0} value2'.format(operator)
                    value = eval(hdr_expr)
                else:
                    value = None
                meta_out[key] = value
    elif operator == 'min':
        mdata = np.minimum(data1, data2)
    elif operator == 'max':
        mdata = np.maximum(data1, data2)

    nddata_out = NDData(np.ma.filled(mdata, fill_value=fill_value),
                        mask=mdata.mask, meta=meta_out)

    return nddata_out
