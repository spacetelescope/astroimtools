# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Image utilities.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.nddata import NDData
from astropy import log
import copy
from astropy.wcs import WCS


__all__ = ['StdUncertainty', 'imarith', 'block_reduce']


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
    """

    allowed_operators = ['+', '-', '*', '/', '//', 'min', 'max']
    operator = operator.strip()
    if operator not in allowed_operators:
        raise ValueError('operator "{0}" is not allowed'.format(operator))

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


def block_reduce(data, block_size, func=np.sum, wcs=None, wcs_origin=0):
    """
    Downsample data by applying a function to local blocks.
    """

    from skimage.measure import block_reduce
    data = np.asanyarray(data)
    if len(block_size) != data.ndim:
        raise ValueError('`block_size` must have the same length as '
                         '`data.shape`')

    block_size = np.array([int(i) for i in block_size])
    size_new = np.array(data.shape) // block_size
    size_init = size_new * block_size
    if size_init[0] != data.shape[0] or size_init[1] != data.shape[1]:
        data = data[:size_init[0], :size_init[1]]   # trim data if necessary

    data_reduced = block_reduce(data, tuple(block_size), func=func)

    if wcs is not None:
        if not isinstance(wcs, WCS):
            raise ValueError('wcs must be an astropy.wcs.WCS object')
        wcs_out = _scale_image_wcs(wcs, 1.0 / block_size, origin=wcs_origin)
    else:
        wcs_out = None

    return data_reduced, wcs_out


def _scale_image_wcs(wcs, scale, origin=0):
    """
    Scale the WCS for a 2D image.
    """

    # interally use scale in (x, y) order to match WCS order convention
    scale = scale[::-1]

    wcs_out = wcs.deepcopy()
    wcs_out._naxis1 = int(wcs._naxis1 * scale[0])
    wcs_out._naxis2 = int(wcs._naxis2 * scale[1])
    origin = int(origin)
    if origin == 0:
        crpix_new = ((np.array(wcs.wcs.crpix) + 0.5) * scale) - 0.5
    elif origin == 1:
        crpix_new = ((np.array(wcs.wcs.crpix) - 0.5) * scale) + 0.5
    else:
        raise ValueError('origin must be 0 or 1')
    wcs_out.wcs.crpix = tuple(crpix_new)

    if not wcs.wcs.has_cd():
        wcs_out.wcs.cdelt = tuple(np.array(wcs.wcs.cdelt) / scale)
    else:
        wcs_out.wcs.cd /= scale
        # TODO: if aspect ratio changes, need to update PC matrix and remove
        # CROTA1 and CROTA2?

    # TODO: update SIP coefficients
    # wcs_out = _scale_sip(wcs_out, block_ratio)

    return wcs_out


def _scale_sip(wcs, scale, origin=0):
    """
    Update SIP coefficients for an image scale change.
    """

    wcs_out = wcs.deepcopy()
    # TODO: need to update SIP A and B matrices
    # a = wcs.sip.a
    # b = wcs.sip.b
    # coef *= np.power(block_ratio[i], np.arange(len(coef)))

    return wcs_out
