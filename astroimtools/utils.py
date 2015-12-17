# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Misc utility functions.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.table import Table
from astropy.nddata import NDData, support_nddata
from astropy.nddata.utils import overlap_slices
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
import warnings
from astropy.utils.exceptions import AstropyUserWarning


__all__ = ['radial_distance', 'listpixels', 'mask_databounds',
           'nddata_cutout2d']

# requires Astropy >= 1.1
__doctest_skip__ = ['nddata_cutout2d']

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
    Return a `~astropy.table.Table` listing the ``(y, x)`` positions and
    ``data`` values for a subarray.

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
        A boolean mask array with the same shape as ``data``.

    lower_bound : float, optional
        The value of the lower bound.  Data values lower than
        ``lower_bound`` will be masked.

    upper_bound : float, optional
        The value of the upper bound.  Data values greater than
        ``upper_bound`` will be masked.

    value : float, optional
        A data value (e.g., ``0.0``) to mask.

    mask_invalid : bool, optional
        If `True` (the default), then any unmasked invalid values (e.g.
        NaN, inf) will be masked.

    Returns
    -------
    mask : bool `~numpy.ndarray`
        The resulting boolean mask array with the same shape as
        ``data``.

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
        mask = np.asanyarray(mask)
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


def nddata_cutout2d(nddata, position, size, mode='trim', fill_value=np.nan):
    """
    Create a 2D cutout of a `~astropy.nddata.NDData` object.

    Specifically, cutouts will made for the ``nddata.data`` and
    ``nddata.mask`` (if present) arrays.  If ``nddata.wcs`` exists, then
    it will also be updated.

    Note that cutouts will not be made for ``nddata.uncertainty`` (if
    present) because they are general objects and not arrays.

    Parameters
    ----------
    nddata : `~astropy.nddata.NDData`
        The 2D `~astropy.nddata.NDData` from which the cutout is taken.

    position : tuple or `~astropy.coordinates.SkyCoord`
        The position of the cutout array's center with respect to the
        ``nddata.data`` array.  The position can be specified either as
        a ``(x, y)`` tuple of pixel coordinates or a
        `~astropy.coordinates.SkyCoord`, in which case ``nddata.wcs``
        must exist.

    size : int, array-like, `~astropy.units.Quantity`
        The size of the cutout array along each axis.  If ``size`` is a
        scalar number or a scalar `~astropy.units.Quantity`, then a
        square cutout of ``size`` will be created.  If ``size`` has two
        elements, they should be in ``(ny, nx)`` order.  Scalar numbers
        in ``size`` are assumed to be in units of pixels.  ``size`` can
        also be a `~astropy.units.Quantity` object or contain
        `~astropy.units.Quantity` objects.  Such
        `~astropy.units.Quantity` objects must be in pixel or angular
        units.  For all cases, ``size`` will be converted to an integer
        number of pixels, rounding the the nearest integer.  See the
        ``mode`` keyword for additional details on the final cutout
        size.

    mode : {'trim', 'partial', 'strict'}, optional
        The mode used for creating the cutout data array.  For the
        ``'partial'`` and ``'trim'`` modes, a partial overlap of the
        cutout array and the input ``nddata.data`` array is sufficient.
        For the ``'strict'`` mode, the cutout array has to be fully
        contained within the ``nddata.data`` array, otherwise an
        `~astropy.nddata.utils.PartialOverlapError` is raised.   In all
        modes, non-overlapping arrays will raise a
        `~astropy.nddata.utils.NoOverlapError`.  In ``'partial'`` mode,
        positions in the cutout array that do not overlap with the
        ``nddata.data`` array will be filled with ``fill_value``.  In
        ``'trim'`` mode only the overlapping elements are returned, thus
        the resulting cutout array may be smaller than the requested
        ``size``.

    fill_value : number, optional
        If ``mode='partial'``, the value to fill pixels in the cutout
        array that do not overlap with the input ``nddata.data``.
        ``fill_value`` must have the same ``dtype`` as the input
        ``nddata.data`` array.

    Returns
    -------
    result : `~astropy.nddata.NDData`
        A `~astropy.nddata.NDData` object with cutouts for the data and
        mask, if input.

    Examples
    --------
    >>> from astropy.nddata import NDData
    >>> import astropy.units as u
    >>> from astroimtools import nddata_cutout2d
    >>> data = np.random.random((500, 500))
    >>> unit = u.electron / u.s
    >>> mask = (data > 0.7)
    >>> meta = {'exptime': 1234 * u.s}
    >>> nddata = NDData(data, mask=mask, unit=unit, meta=meta)
    >>> cutout = nddata_cutout2d(nddata, (100, 100), (10, 10))
    >>> cutout.data.shape
    (10, 10)
    >>> cutout.mask.shape
    (10, 10)
    >>> cutout.unit
    Unit("electron / s")
    """

    from astropy.nddata.utils import Cutout2D

    if not isinstance(nddata, NDData):
        raise ValueError('nddata input must be an NDData object')

    if isinstance(position, SkyCoord):
        if nddata.wcs is None:
            raise ValueError('nddata must contain WCS if the input '
                             'position is a SkyCoord')
        position = skycoord_to_pixel(position, nddata.wcs, mode='all')

    data_cutout = Cutout2D(np.asanyarray(nddata.data), position, size,
                           wcs=nddata.wcs, mode=mode, fill_value=fill_value)
    # need to create a new NDData instead of copying/replacing
    nddata_out = NDData(data_cutout.data, unit=nddata.unit,
                        uncertainty=nddata.uncertainty, meta=nddata.meta)

    if nddata.wcs is not None:
        nddata_out.wcs = data_cutout.wcs

    if nddata.mask is not None:
        mask_cutout = Cutout2D(np.asanyarray(nddata.mask), position, size,
                               mode=mode, fill_value=fill_value)
        nddata_out.mask = mask_cutout.data

    return nddata_out
