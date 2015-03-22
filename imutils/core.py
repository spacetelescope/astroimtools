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


__all__ = ['StdUncertainty', 'imarith', 'block_reduce', 'block_replicate',
           'radial_distance', 'listpixels', 'Cutout', 'NDDataCutout',
           'mask_databounds']


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


@support_nddata
def block_reduce(data, block_size, error=None, func=np.sum, wcs=None,
                 wcs_origin=0):
    """
    Downsample data by applying a function to local blocks.

    If ``data`` is not perfectly divisible by ``block_size`` along a
    given axis then the data will be trimmed (from the end) along that
    axis.

    WCS propagation currently works only if ``data`` is a 2D image.

    Parameters
    ----------
    data : array_like
        The data to be resampled.

    block_size : array_like (int)
        An array containing the integer downsampling factor along each
        axis.  ``block_size`` must have the same length as
        ``data.shape``.

    error : array-like
        The 1-sigma pixelwise errors of ``data``.  Errors will be
        propagated only if ``func`` is ``np.sum`` or ``np.mean``.

    function : callable
        The method to use to downsample the data.  Must be a callable
        that takes in a `~numpy.ndarray` along with an ``axis`` keyword,
        which defines the axis along which the function is applied.  The
        default is `~numpy.sum`, which provides block summation (and
        conserves the data sum).

    wcs : `~astropy.wcs.WCS`, optional
        The WCS corresponding to ``data``.  If ``wcs`` is input, then
        the transformed WCS will be output (``wcs_output``).  WCS
        propagation currently works only if ``data`` is a 2D image.

    wcs_origin : {0, 1}, optional
        Whether the WCS contains 0 or 1-based pixel coordinates.  The
        default is 0.  For FITS WCS, set ``wcs_origin=1``.

    Returns
    -------
    output : array-like
        The resampled data.

    error : array-like
        The propagated error array.  Returned only if ``func`` is
        ``np.sum`` or ``np.mean``.

    wcs_output : `~astropy.wcs.WCS`, optional
        The transformed WCS.  Returned only if ``wcs`` is input and
        ``data`` is a 2D image.
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

    error_out = None
    if error is not None:
        if func == np.sum or func == np.mean:
            error_out = np.sqrt(block_reduce(error**2, tuple(block_size),
                                             func=func))
        if func == np.mean:
            error_out /= len(data)

    return data_reduced, error_out, wcs_out


@support_nddata
def block_replicate(data, block_size, conserve_sum=True):
    """
    Upsample a 1D, 2D, or 3D data array by block replication.

    Parameters
    ----------
    data : array_like (1D, 2D, or 3D)
        The data to be block replicated.

    block_size : int or array_like (int)
        The integer block size (upsampling factor) along each axis.  If
        ``block_size`` is a scalar and ``data`` is a 2D array, then the
        data will be upsampled by ``block_size`` along both dimensions.

    conserve_sum : bool
        If `True` (the default) then the sum of the output
        block-replicated data will equal the sum of the input ``data``.

    Returns
    -------
    output : array_like
        The block-replicated data.

    Examples
    --------
    >>> import numpy as np
    >>> from imutils import block_replicate
    >>> data = np.array([[0., 1.], [2., 3.]])
    >>> block_replicate(data, 2)
    array([[ 0.  ,  0.  ,  0.25,  0.25],
           [ 0.  ,  0.  ,  0.25,  0.25],
           [ 0.5 ,  0.5 ,  0.75,  0.75],
           [ 0.5 ,  0.5 ,  0.75,  0.75]])

    >>> block_replicate(data, 2, conserve_sum=False)
    array([[ 0.,  0.,  1.,  1.],
           [ 0.,  0.,  1.,  1.],
           [ 2.,  2.,  3.,  3.],
           [ 2.,  2.,  3.,  3.]])
    """

    data = np.asanyarray(data)
    block_size = np.atleast_1d(block_size)

    if data.ndim > 1 and len(block_size) == 1:
        block_size = np.repeat(block_size, data.ndim)

    if len(block_size) != data.ndim:
        raise ValueError('`block_size` must have the same length as '
                         '`data.shape`')

    block_size = np.array([int(i) for i in block_size])

    if data.ndim == 1:
        output = np.broadcast_arrays(
            data.reshape(data.shape[0], 1),
            np.ones((1, block_size[0])))[0].reshape(
                data.shape[0]*block_size[0])

    elif data.ndim == 2:
        output = np.broadcast_arrays(
            data.reshape(data.shape[0], 1,
                         data.shape[1], 1),
            np.ones((1, block_size[0],
                     1, block_size[1])))[0].reshape(
                         data.shape[0] * block_size[0],
                         data.shape[1] * block_size[1])

    elif data.ndim == 3:
        output = np.broadcast_arrays(
            data.reshape(data.shape[0], 1,
                         data.shape[1], 1,
                         data.shape[2], 1),
            np.ones((1, block_size[0],
                     1, block_size[1],
                     1, block_size[2])))[0].reshape(
                         data.shape[0] * block_size[0],
                         data.shape[1] * block_size[1],
                         data.shape[2] * block_size[2])

    else:
        raise ValueError('data.ndim must be <= 3')

    if conserve_sum:
        output = output / float(np.prod(block_size))

    return output


def radial_distance(shape, position):
    """
    Return an array where each value is the Euclidean distance from a
    given position.
    """

    x = np.arange(shape[1]) - position[1]
    y = np.arange(shape[0]) - position[0]
    xx, yy = np.meshgrid(x, y)
    return np.sqrt(xx**2 + yy**2)


class Cutout(object):
    def __init__(self, data, position, shape, wcs=None):
        if isinstance(position, SkyCoord):
            if wcs is None:
                raise ValueError('wcs must be input if position is a '
                                 'SkyCoord')

            x, y = skycoord_to_pixel(position, wcs, mode='all')
            position = (y, x)

        data = np.asanyarray(data)
        slices_large, slices_small = overlap_slices(data.shape, shape,
                                                    position)
        self.slices_large = slices_large
        self.slices_small = slices_small
        self.data = data[slices_large]
        self.requested_position = position
        self.requested_shape = shape

    @staticmethod
    def _calc_bbox(slices):
        """
        Calculate minimimal bounding box.
        Output:  (bottom, left, top, right)   (y0, x0, y1, x1)
        """
        return (slices[0].start, slices[1].start,
                slices[0].stop, slices[1].stop)

    @lazyproperty
    def origin(self):
        """
        The origin pixel of the cutout in the large array.
        """
        slices = self.slices_large
        return np.array([slices[0].start, slices[1].start])

    @lazyproperty
    def position(self):
        """
        The actual "central" position in the large array.
        """
        slices = self.slices_large
        return (slices[0].start + np.int(np.ceil(
            (slices[0].stop - slices[0].start - 1) / 2.)),
            (slices[1].start + np.int(np.ceil(
                (slices[1].stop - slices[1].start - 1) / 2.))))

    @lazyproperty
    def bbox_large(self):
        return self._calc_bbox(self.slices_large)

    @lazyproperty
    def bbox_small(self):
        return self._calc_bbox(self.slices_small)


class NDDataCutout(object):
    def __init__(self, nddata, position, shape):
        if isinstance(position, SkyCoord):
            if nddata.wcs is None:
                raise ValueError('nddata must contain WCS if the input '
                                 'position is a SkyCoord')

            x, y = skycoord_to_pixel(position, nddata.wcs, mode='all')
            position = (y, x)

        data = np.asanyarray(nddata.data)
        print(data.shape, shape, position)
        slices_large, slices_small = overlap_slices(data.shape, shape,
                                                    position)
        self.slices_large = slices_large
        self.slices_small = slices_small

        data = nddata.data[slices_large]
        mask = None
        uncertainty = None
        if nddata.mask is not None:
            mask = nddata.mask[slices_large]
        if nddata.uncertainty is not None:
            uncertainty = nddata.uncertainty[slices_large]

        self.nddata = NDData(data, mask=mask, uncertainty=uncertainty)

    @staticmethod
    def _calc_bbox(slices):
        """
        Calculate minimimal bounding box.
        Output:  (bottom, left, top, right)   (y0, x0, y1, x1)
        """
        return (slices[0].start, slices[1].start,
                slices[0].stop, slices[1].stop)

    @lazyproperty
    def bbox_large(self):
        return self._calc_bbox(self.slices_large)

    @lazyproperty
    def bbox_small(self):
        return self._calc_bbox(self.slices_small)


@support_nddata
def listpixels(data, position, shape, subarray_indices=False, wcs=None):
    """
    Return a `~astropy.table.Table` listing the ``(row, col)``
    (``(y, x)``) positions and ``data`` values for a subarray.

    Given a position of the center of the subarray, with respect to the
    large array, the array indices and values are returned.  This
    function takes care of the correct behavior at the boundaries, where
    the small array is appropriately trimmed.

    Parameters
    ----------
    data : array-like
        The input data.

    position : tuple (int) or `~astropy.coordinates.SkyCoord`
        The position of the subarray center with respect to the data
        array.  The position can be specified either as an integer
        ``(row, col)`` (``(y, x)``) tuple or a
        `~astropy.coordinates.SkyCoord`, in which case ``wcs`` is a
        required input.

    shape : tuple (int)
        The integer shape (``(ny, nx)``) of the subarray.

    subarray_indices : bool, optional
        If `True` then the returned positions are relative to the small
        subarray.  If `False` (default) then the returned positions are
        relative to the ``data`` array.

    wcs : `~astropy.wcs.WCS`, optional
        The WCS transformation to use if ``position`` is a
        `~astropy.coordinates.SkyCoord`.

    Returns
    -------
    table : `~astropy.table.Table`
        A table containing the ``x`` and ``y`` positions and data
        values.

    Notes
    -----
    This function is decorated with `~astropy.nddata.support_nddata` and
    thus supports `~astropy.nddata.NDData` objects as input.

    See Also
    --------
    :func:`astropy.nddata.utils.overlap_slices`

    Examples
    --------
    >>> import numpy as np
    >>> from imutils import listpixels
    >>> data = np.arange(625).reshape(25, 25)
    >>> tbl = listpixels(data, (10, 12), (3, 3))
    >>> print(len(tbl))
    3

    >>> tbl.pprint(max_lines=-1)
     x   y  value
    --- --- -----
     11   9   236
     12   9   237
     13   9   238
     11  10   261
     12  10   262
     13  10   263
     11  11   286
     12  11   287
     13  11   288
    """

    if isinstance(position, SkyCoord):
        if wcs is None:
            raise ValueError('wcs must be input if positions is a SkyCoord')

        x, y = skycoord_to_pixel(position, wcs, mode='all')
        position = (y, x)

    data = np.asanyarray(data)
    slices_large, slices_small = overlap_slices(data.shape, shape, position)
    slices = slices_large
    yy, xx = np.mgrid[slices]
    values = data[yy, xx]

    if subarray_indices:
        slices = slices_small
        yy, xx = np.mgrid[slices]

    tbl = Table()
    tbl['x'] = xx.ravel()
    tbl['y'] = yy.ravel()
    tbl['value'] = values.ravel()

    return tbl


def mask_databounds(nddata, mask=None, lower_bound=None, upper_bound=None,
                    mask_value=None):
    """
    Update a `~astropy.nddata.NDData` mask by masking data values that
    are below a lower bound, above an upper bound, equal to particular
    value, or are invalid (e.g. np.nan or np.inf).
    """

    if not isinstance(nddata, NDData):
        raise ValueError('nddata input must be an astropy.nddata.NDData '
                         'object')

    if nddata.mask is not None:
        if nddata.mask.shape != nddata.data.shape:
            raise ValueError('mask and data must have the same shape')
        data = np.ma.MaskedArray(nddata.data, nddata.mask)
    else:
        data = np.ma.MaskedArray(nddata.data, None)

    if lower_bound is not None:
        data = np.ma.masked_less(data, lower_bound)
    if upper_bound is not None:
        data = np.ma.masked_greater(data, upper_bound)
    if mask_value is not None:
        data = np.ma.masked_values(data, mask_value)

    nmasked = data.count()
    data = np.ma.masked_invalid(data)    # mask np.nan, np.inf
    if data.count() != nmasked:
        warnings.warn(('The data array contains at least one unmasked '
                       'invalid value (NaN or inf). These values will be '
                       'automatically masked.'), AstropyUserWarning)

    if np.all(data.mask):
        raise ValueError('All data values are masked')

    if np.any(data.mask):
        nddata_out = copy.deepcopy(nddata)
        nddata_out.mask = data.mask
        return nddata_out
    else:
        return nddata


def _scale_image_wcs(wcs, scale, origin=0):
    """
    Scale the WCS for a 2D image.

    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        The WCS corresponding to ``data``.  If ``wcs`` is input, then
        the transformed WCS will be output (``wcs_output``).  WCS
        propagation currently works only if ``data`` is a 2D image.

    scale : 2-tuple
        Scale ratio along each data axis.

    origin : {0, 1}, optional
        Whether the WCS contains 0 or 1-based pixel coordinates.  The
        default is 0.  For FITS WCS, set ``wcs_origin=1``.

    Returns
    -------
    wcs_output : `~astropy.wcs.WCS`, optional
        The transformed WCS.
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
