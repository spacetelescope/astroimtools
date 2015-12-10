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


__all__ = ['radial_distance', 'listpixels', 'NDDataCutout',
           'mask_databounds']


warnings.filterwarnings('always', category=AstropyUserWarning)


def radial_distance(shape, position):
    """
    Return an array where each value is the Euclidean distance from a
    given position.
    """

    x = np.arange(shape[1]) - position[1]
    y = np.arange(shape[0]) - position[0]
    xx, yy = np.meshgrid(x, y)
    return np.sqrt(xx**2 + yy**2)


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
        ``(row, col)`` (``(y, x)``) tuple of pixel coordinates or a
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
