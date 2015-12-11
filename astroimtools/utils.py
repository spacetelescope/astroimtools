# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Image utilities.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.table import Table
from astropy.nddata import NDData, support_nddata
from astropy.nddata.utils import overlap_slices
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy.utils import lazyproperty
import warnings
from astropy.utils.exceptions import AstropyUserWarning


__all__ = ['radial_distance', 'listpixels', 'NDDataCutout',
           'mask_databounds']


warnings.filterwarnings('always', category=AstropyUserWarning)


def radial_distance(position, shape):
    """
    Return an array where each value is the Euclidean distance from a
    given position.

    Parameters
    ----------
    position : tuple
        An ``(x, y)`` tuple of pixel coordinates for the central
        position (i.e., zero radial distance).

    shape : tuple
        The size of the output array along each axis.

    Returns
    -------
    result : `~numpy.ndarray`
        An array containing the Euclidian radial distances from the
        input ``position``.

    Examples
    --------
    >>> from astroimtools import radial_distance
    >>> radial_distance((1, 1), (3, 3))
    array([[ 1.41421356,  1.        ,  1.41421356],
           [ 1.        ,  0.        ,  1.        ],
           [ 1.41421356,  1.        ,  1.41421356]])
    """

    if len(position) != 2:
        raise ValueError('position must have only 2 elements')
    if len(shape) != 2:
        raise ValueError('shape must have only 2 elements')

    x = np.arange(shape[1]) - position[1]
    y = np.arange(shape[0]) - position[0]
    xx, yy = np.meshgrid(x, y)
    return np.sqrt(xx**2 + yy**2)


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
        array.  The position can be specified either as an integer ``(y,
        x)`` tuple of pixel coordinates or a
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

    Examples
    --------
    >>> import numpy as np
    >>> from astroimtools import listpixels
    >>> np.random.seed(12345)
    >>> data = np.random.random((25, 25))
    >>> tbl = listpixels(data, (8, 11), (3, 3))
    >>> tbl.pprint(max_lines=-1)
     x   y       value
    --- --- ---------------
     10   7  0.758572036918
     11   7 0.0695296661543
     12   7  0.705473438596
     10   8  0.840662495709
     11   8  0.469314693584
     12   8  0.562643429012
     10   9 0.0341315835241
     11   9  0.230496547915
     12   9  0.228353706465
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


@support_nddata
def mask_databounds(data, mask=None, lower_bound=None, upper_bound=None,
                    value=None, mask_invalid=True):
    """
    Create or update a mask by masking data values that are below a
    lower bound, above an upper bound, equal to particular value, or are
    invalid (e.g. np.nan or np.inf).

    Parameters
    ----------
    data : `~numpy.ndarray`
        The data array.

    mask : bool `~numpy.ndarray`, optional
        A mask...

    lower_bound : float, optional
    upper_bound : float, optional
    value : float, optional
    mask_invalid : bool, optional

    Returns
    -------
    mask : bool `~numpy.ndarray`

    Examples
    --------
    >>> from astroimtools import mask_databounds
    >>> data = np.arange(7)
    >>> print(data)
    [0 1 2 3 4 5 6]
    >>> mask_databounds(data, lower_bound=2, upper_bound=5, value=3)
    array([ True,  True, False,  True, False, False,  True], dtype=bool)
    """

    if mask is None:
        data = np.ma.MaskedArray(data, mask=None)
    else:
        if mask.shape != data.shape:
            raise ValueError('mask and data must have the same shape')
        data = np.ma.MaskedArray(data, mask=mask)

    if lower_bound is not None:
        data = np.ma.masked_less(data, lower_bound)
    if upper_bound is not None:
        data = np.ma.masked_greater(data, upper_bound)
    if value is not None:
        data = np.ma.masked_values(data, value)

    if mask_invalid:
        nmasked = data.count()
        data = np.ma.masked_invalid(data)    # mask np.nan, np.inf
        if data.count() != nmasked:
            warnings.warn('The data array contains unmasked invalid '
                          'values (NaN or inf), which are now masked.',
                           AstropyUserWarning)

    if np.all(data.mask):
        raise ValueError('All data values are masked')

    return data.mask


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
