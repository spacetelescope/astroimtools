# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Statistics tools.
"""

import numpy as np
from astropy.nddata import NDData, support_nddata
from astropy.stats import (biweight_location, biweight_midvariance, mad_std,
                           SigmaClip)
from astropy.table import Table
from astropy.utils import lazyproperty
from astropy.utils.decorators import deprecated_renamed_argument

from .utils import mask_databounds


__all__ = ['minmax', 'NDDataStats', 'nddata_stats']


@support_nddata
def minmax(data, mask=None, axis=None):
    """
    Return the minimum and maximum values of an array (or along an array
    axis).

    Parameters
    ----------
    data : array-like
        The input data.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    axis : int, optional
        The axis along which to operate.  By default, flattened input is
        used.

    Returns
    -------
    min : scalar or `~numpy.ndarray`
        The minimum value of ``data``.  If ``axis`` is `None`, the
        result is a scalar value.  If ``axis`` is input, the result is
        an array of dimension ``data.ndim - 1``.

    max : scalar or `~numpy.ndarray`
        The maximum value of ``data``.  If ``axis`` is `None`, the
        result is a scalar value.  If ``axis`` is input, the result is
        an array of dimension ``data.ndim - 1``.

    Notes
    -----
    This function is decorated with `~astropy.nddata.support_nddata` and
    thus supports `~astropy.nddata.NDData` objects as input.

    Examples
    --------
    >>> import numpy as np
    >>> from astroimtools import minmax
    >>> np.random.seed(12345)
    >>> data = np.random.random((3, 3))
    >>> minmax(data)  # doctest: +FLOAT_CMP
    (0.18391881167709445, 0.9645145197356216)

    >>> mask = (data < 0.3)
    >>> minmax(data, mask=mask)  # doctest: +FLOAT_CMP
    (0.3163755545817859, 0.9645145197356216)

    >>> minmax(data, axis=1)  # doctest: +FLOAT_CMP
    (array([0.18391881, 0.20456028, 0.6531771 ]),
     array([0.92961609, 0.5955447 , 0.96451452]))
    """

    if mask is not None:
        funcs = [np.ma.min, np.ma.max]
        data = np.ma.masked_array(data, mask=mask)
    else:
        funcs = [np.min, np.max]

    return funcs[0](data, axis=axis), funcs[1](data, axis=axis)


class NDDataStats:
    """
    Class to calculate (sigma-clipped) image statistics on NDData
    objects.

    Set the ``sigma_clip`` keyword to perform sigma clipping.

    Parameters
    ----------
    nddata : `~astropy.nddata.NDData`
        NDData object containing the data array (and an optional mask)
        on which to calculate statistics.  Masked pixels are excluded
        when computing the image statistics.

    sigma_clip : `astropy.stats.SigmaClip` instance, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters.  If `None` then no sigma clipping will be
        performed (default).

    lower_bound : float, optional
        The minimum data value to include in the statistics.  All pixel
        values less than ``lower_bound`` will be ignored.  `None` means
        that no lower bound is applied (default).

    upper_bound : float, optional
        The maximum data value to include in the statistics.  All pixel
        values greater than ``upper_bound`` will be ignored.  `None`
        means that no upper bound is applied (default).

    mask_value : float, optional
        A data value (e.g., ``0.0``) to be masked.  ``mask_value`` will
        be masked in addition to any input ``mask``.

    mask_invalid : bool, optional
        If `True` (the default), then any unmasked invalid values (e.g.
        NaN, inf) will be masked.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.nddata import NDData
    >>> from astroimtools import NDDataStats
    >>> data = np.arange(10)
    >>> data[0] = 100.
    >>> nddata = NDData(data)
    >>> stats = NDDataStats(nddata)
    >>> stats.mean
    14.5
    >>> stats.std  # doctest: +FLOAT_CMP
    28.605069480775605
    >>> stats.mad_std  # doctest: +FLOAT_CMP
    3.706505546264005
    >>> from astropy.stats import SigmaClip
    >>> sigclip = SigmaClip(sigma=2.5)
    >>> stats = NDDataStats(nddata, sigma_clip=sigclip)
    >>> stats.mean
    5.0
    >>> stats.std  # doctest: +FLOAT_CMP
    2.581988897471611
    >>> stats.mad_std  # doctest: +FLOAT_CMP
    2.965204437011204
    """

    @deprecated_renamed_argument('sigma', 'sigma_clip', '0.2')
    def __init__(self, nddata, sigma_clip=None, lower_bound=None,
                 upper_bound=None, mask_value=None, mask_invalid=True):

        if not isinstance(nddata, NDData):
            raise TypeError('nddata input must be an astropy.nddata.NDData '
                            'instance')

        # update the mask
        mask = mask_databounds(nddata.data, mask=nddata.mask,
                               lower_bound=lower_bound,
                               upper_bound=upper_bound, value=mask_value,
                               mask_invalid=mask_invalid)

        if np.all(mask):
            raise ValueError('All data values are masked')

        # remove masked values
        data = nddata.data[~mask]

        if sigma_clip is not None:
            if not isinstance(sigma_clip, SigmaClip):
                raise TypeError('sigma_clip must be an '
                                'astropy.stats.SigmaClip instance')

            data = sigma_clip(data, masked=False, axis=None)  # 1D ndarray

        self.goodvals = data.ravel()  # 1D array
        self._original_npixels = nddata.data.size

    def __getitem__(self, key):
        return getattr(self, key, None)

    @lazyproperty
    def npixels(self):
        """
        The number of good (unmasked/unclipped) pixels.
        """

        return len(self.goodvals)

    @lazyproperty
    def nrejected(self):
        """
        The number of rejected (masked/clipped) pixels.
        """

        return self._original_npixels - self.npixels

    @lazyproperty
    def mean(self):
        """
        The mean of pixel values.
        """

        return np.mean(self.goodvals)

    @lazyproperty
    def median(self):
        """
        The median of the pixel values.
        """

        return np.median(self.goodvals)

    @lazyproperty
    def mode(self):
        """
        The mode of the pixel values.

        The mode is estimated simply as ``(3 * median) - (2 * mean)``.
        """

        return 3. * np.median(self.goodvals) - 2. * np.mean(self.goodvals)

    @lazyproperty
    def std(self):
        """
        The standard deviation of the pixel values.
        """

        return np.std(self.goodvals)

    @lazyproperty
    def min(self):
        """
        The minimum pixel value.
        """

        return np.min(self.goodvals)

    @lazyproperty
    def max(self):
        """
        The maximum pixel value.
        """

        return np.max(self.goodvals)

    @lazyproperty
    def mad_std(self):
        r"""
        A robust standard deviation using the `median absolute deviation
        (MAD)
        <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_.
        The MAD is defined as ``median(abs(a - median(a)))``.

        The standard deviation estimator is given by:

        .. math::

            \sigma \approx \frac{\textrm{MAD}}{\Phi^{-1}(3/4)}
            \approx 1.4826 \ \textrm{MAD}

        where :math:`\Phi^{-1}(P)` is the normal inverse cumulative
        distribution function evaluated at probability :math:`P = 3/4`.
        """

        return mad_std(self.goodvals)

    @lazyproperty
    def biweight_location(self):
        """
        The biweight location of the pixel values.
        """

        return biweight_location(self.goodvals)

    @lazyproperty
    def biweight_midvariance(self):
        """
        The biweight midvariance of the pixel values.
        """

        return biweight_midvariance(self.goodvals)

    @lazyproperty
    def skew(self):
        """
        The skew of the pixel values.
        """

        from scipy.stats import skew
        return skew(self.goodvals)

    @lazyproperty
    def kurtosis(self):
        """
        The kurtosis of the pixel values.
        """

        from scipy.stats import kurtosis
        return kurtosis(self.goodvals)


@deprecated_renamed_argument('sigma', 'sigma_clip', '0.2')
def nddata_stats(nddata, sigma_clip=None, columns=None, lower_bound=None,
                 upper_bound=None, mask_value=None, mask_invalid=True):
    """
    Calculate various statistics on the input data.

    Set the ``sigma_clip`` keyword to perform sigma clipping.

    Parameters
    ----------
    nddata : `~astropy.nddata.NDData` or list of `~astropy.nddata.NDData`
        `~astropy.nddata.NDData` object containing the data array and
        optional mask on which to calculate statistics.  Masked pixels
        are excluded when computing the image statistics.

    sigma_clip : `astropy.stats.SigmaClip` instance, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters.  If `None` then no sigma clipping will be
        performed (default).

    columns : str or list of str, optional
        The names of columns, in order, to include in the output
        `~astropy.table.Table`.  The column names can include any of the
        following statistic names: 'biweight_location',
        'biweight_midvariance', 'kurtosis', 'mad_std', 'max', 'mean',
        'median', 'min', 'mode', 'npixels', 'nrejected', 'skew', or
        'std'.  The column names can also include a name of a key in the
        `astropy.nddata.NDData.meta` dictionary.  The default is
        ``['npixels', 'mean', 'std', 'min', 'max']``.

    lower_bound : float, optional
        The minimum data value to include in the statistics.  All pixel
        values less than ``lower_bound`` will be ignored.  `None` means
        that no lower bound is applied (default).

    upper_bound : float, optional
        The maximum data value to include in the statistics.  All pixel
        values greater than ``upper_bound`` will be ignored.  `None` means
        that no upper bound is applied (default).

    mask_value : float, optional
        A data value (e.g., ``0.0``) to be masked.  ``mask_value`` will
        be masked in addition to any input ``mask``.

    mask_invalid : bool, optional
        If `True` (the default), then any unmasked invalid values (e.g.
        NaN, inf) will be masked.

    Returns
    -------
    table : `~astropy.table.Table`
        A table containing the calculated image statistics (or
        `~astropy.nddata.NDData` metadata).  Each table row corresponds
        to a single data array.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.nddata import NDData
    >>> from astroimtools import nddata_stats
    >>> data = np.arange(10)
    >>> data[0] = 100.
    >>> nddata = NDData(data)
    >>> columns = ['mean', 'median', 'mode', 'std', 'mad_std', 'min', 'max']
    >>> tbl = nddata_stats(nddata, columns=columns)
    >>> for col in tbl.colnames:
    ...     tbl[col].info.format = '%.8g'  # for consistent table output
    >>> print(tbl)
    mean median  mode    std     mad_std  min max
    ---- ------ ----- --------- --------- --- ---
    14.5    5.5 -12.5 28.605069 3.7065055   1 100
    >>> from astropy.stats import SigmaClip
    >>> sigclip = SigmaClip(sigma=2.5)
    >>> tbl = nddata_stats(nddata, sigma_clip=sigclip, columns=columns)
    >>> for col in tbl.colnames:
    ...     tbl[col].info.format = '%.8g'  # for consistent table output
    >>> print(tbl)
    mean median mode    std     mad_std  min max
    ---- ------ ---- --------- --------- --- ---
       5      5    5 2.5819889 2.9652044   1   9
    """

    stats = []
    if not isinstance(nddata, list):
        nddata = np.atleast_1d(nddata)

    for nddata_obj in nddata:
        stats.append(NDDataStats(
            nddata_obj, sigma_clip=sigma_clip, lower_bound=lower_bound,
            upper_bound=upper_bound, mask_value=mask_value,
            mask_invalid=mask_invalid))

    output_columns = None
    default_columns = ['npixels', 'mean', 'std', 'min', 'max']
    allowed_columns = ['biweight_location', 'biweight_midvariance',
                       'kurtosis', 'mad_std', 'max', 'mean', 'median', 'min',
                       'mode', 'npixels', 'nrejected', 'skew', 'std']

    if columns is None:
        output_columns = default_columns
    else:
        output_columns = np.atleast_1d(columns)

    output_table = Table()
    for column in output_columns:
        if column not in allowed_columns:
            values = [nddata_obj.meta.get(column, None) for nddata_obj
                      in nddata]
        else:
            values = [getattr(stat, column) for stat in stats]
        output_table[column] = values

    return output_table
