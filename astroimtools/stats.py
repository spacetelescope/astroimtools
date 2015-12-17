# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Statistics tools.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.stats import (sigma_clip, biweight_location,
                           biweight_midvariance, mad_std)
from astropy.utils import lazyproperty
from astropy.table import Table
from astropy.nddata import NDData, support_nddata
from .utils import mask_databounds
import astropy

majv, minv = astropy.__version__.split('.')[:2]
minv = minv.split('rc')[0]
ASTROPY_LT_1P1 = ([int(majv), int(minv)] < [1, 1])


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
    >>> minmax(data)
    (0.18391881167709445, 0.96451451973562163)

    >>> mask = (data < 0.3)
    >>> minmax(data, mask=mask)
    (0.3163755545817859, 0.96451451973562163)

    >>> minmax(data, axis=1)
    (array([ 0.18391881,  0.20456028,  0.6531771 ]),
     array([ 0.92961609,  0.5955447 ,  0.96451452]))
    """

    if mask is not None:
        funcs = [np.ma.min, np.ma.max]
        data = np.ma.masked_array(data, mask=mask)
    else:
        funcs = [np.min, np.max]

    return funcs[0](data, axis=axis), funcs[1](data, axis=axis)


class NDDataStats(object):
    """
    Class to calculate (sigma-clipped) image statistics on NDData
    objects.

    Set the ``sigma`` keyword to perform sigma clipping.
    """

    def __init__(self, nddata, sigma=None, sigma_lower=None, sigma_upper=None,
                 iters=1, cenfunc=np.ma.median, stdfunc=np.std,
                 lower_bound=None, upper_bound=None, mask_value=None,
                 mask_invalid=True):
        """
        Parameters
        ----------
        nddata : `~astropy.nddata.NDData`
            NDData object containing the data array (and an optional
            mask) on which to calculate statistics.  Masked pixels are
            excluded when computing the image statistics.

        sigma : `None` or float, optional
            The number of standard deviations to use for both the lower
            and upper clipping limit. These limits are overridden by
            ``sigma_lower`` and ``sigma_upper``, if input. Defaults to
            `None`, which means that sigma clipping will not be
            performed.

        sigma_lower : float or `None`, optional
            The number of standard deviations to use as the lower bound
            for the clipping limit. If `None` then the value of
            ``sigma`` is used. Defaults to `None`.  Requires Astropy >=
            1.1.

        sigma_upper : float or `None`, optional
            The number of standard deviations to use as the upper bound
            for the clipping limit. If `None` then the value of
            ``sigma`` is used. Defaults to `None`.  Requires Astropy >=
            1.1.

        iters : int or `None`, optional
            The number of sigma clipping iterations to perform, or
            `None` to clip until convergence is achieved (i.e. continue
            until the last iteration clips nothing).

        cenfunc : callable, optional
            The function used to compute the center for the clipping.
            Must be a callable that takes in a masked array and outputs
            the central value. Defaults to the median
            (`numpy.ma.median`).

        stdfunc : callable, optional
            The function used to compute the standard deviation about
            the center. Must be a callable that takes in a masked array
            and outputs a width estimator. Masked (rejected) pixels are
            those where::

                deviation < (-sigma_lower * stdfunc(deviation))
                deviation > (sigma_upper * stdfunc(deviation))

            where::

                deviation = data - cenfunc(data [,axis=int])

            Defaults to the standard deviation (`numpy.std`).

        lower_bound : float, optional
            The minimum data value to include in the statistics.  All
            pixel values less than ``lower_bound`` will be ignored.
            `None` means that no lower bound is applied (default).

        upper_bound : float, optional
            The maximum data value to include in the statistics.  All
            pixel values greater than ``upper_bound`` will be ignored.
            `None` means that no upper bound is applied (default).

        mask_value : float, optional
            A data value (e.g., ``0.0``) to be masked.  ``mask_value``
            will be masked in addition to any input ``mask``.

        mask_invalid : bool, optional
            If `True` (the default), then any unmasked invalid values
            (e.g.  NaN, inf) will be masked.

        Examples
        --------
        >>> import numpy as np
        >>> from astropy.nddata import NDData
        >>> from astroimtools import NDDataStats
        >>> nddata = NDData(np.arange(10))
        >>> stats = NDDataStats(nddata)
        >>> stats.mean
        4.5
        >>> stats.std
        2.8722813232690143
        >>> stats.mad_std
        3.7065055462640051
        """

        if not isinstance(nddata, NDData):
            raise ValueError('nddata input must be an astropy.nddata.NDData '
                             'object')

        # update the mask
        mask = mask_databounds(nddata.data, mask=nddata.mask,
                               lower_bound=lower_bound,
                               upper_bound=upper_bound, value=mask_value,
                               mask_invalid=mask_invalid)

        if np.all(mask):
            raise ValueError('All data values are masked')
        if mask is not None:
            data = np.ma.MaskedArray(nddata.data, mask)
        else:
            data = nddata.data

        if sigma is not None:
            if ASTROPY_LT_1P1:
                data = sigma_clip(data, sig=sigma, cenfunc=cenfunc,
                                  varfunc=np.ma.var, iters=iters)
            else:
                data = sigma_clip(data, sigma=sigma, sigma_lower=sigma_lower,
                                  sigma_upper=sigma_upper, cenfunc=cenfunc,
                                  stdfunc=stdfunc, iters=iters)

        if np.ma.is_masked(data):
            self.goodvals = data.data[~data.mask]
            self._npixels = np.ma.count(data)
            self._nrejected = np.ma.count_masked(data)
        else:
            self.goodvals = data
            self._npixels = data.size
            self._nrejected = 0

    def __getitem__(self, key):
        return getattr(self, key, None)

    @lazyproperty
    def npixels(self):
        """
        The number of unclipped pixels.
        """

        return self._npixels

    @lazyproperty
    def nrejected(self):
        """
        The number of rejected (clipped) pixels.
        """

        return self._nrejected

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
        """
        A robust standard deviation using the `median absolute deviation
        (MAD)
        <http://en.wikipedia.org/wiki/Median_absolute_deviation>`_.
        The MAD is defined as ``median(abs(a - median(a)))``.

        The standard deviation estimator is given by:

        .. math::

            \\sigma \\approx \\frac{\\textrm{MAD}}{\Phi^{-1}(3/4)}
            \\approx 1.4826 \ \\textrm{MAD}

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


def nddata_stats(nddata, sigma=None, sigma_lower=None, sigma_upper=None,
                 iters=5, cenfunc=np.ma.median, stdfunc=np.std, columns=None,
                 lower_bound=None, upper_bound=None, mask_value=None,
                 mask_invalid=True):
    """
    Calculate various statistics on the input data.

    Set the ``sigma`` keyword to perform sigma clipping.

    Parameters
    ----------
    nddata : `~astropy.nddata.NDData` or list of `~astropy.nddata.NDData`
        `~astropy.nddata.NDData` object containing the data array and
        optional mask on which to calculate statistics.  Masked pixels
        are excluded when computing the image statistics.

    sigma : `None` or float, optional
        The number of standard deviations to use for both the lower and
        upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. Defaults to
        `None`, which means that sigma clipping will not be performed.

    sigma_lower : float or `None`, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.

    sigma_upper : float or `None`, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.

    iters : int or `None`, optional
        The number of sigma clipping iterations to perform, or `None` to
        clip until convergence is achieved (i.e. continue until the last
        iteration clips nothing).

    cenfunc : callable, optional
        The function used to compute the center for the clipping. Must
        be a callable that takes in a masked array and outputs the
        central value. Defaults to the median (`numpy.ma.median`).

    stdfunc : callable, optional
        The function used to compute the standard deviation about the
        center. Must be a callable that takes in a masked array and
        outputs a width estimator. Masked (rejected) pixels are those
        where::

             deviation < (-sigma_lower * stdfunc(deviation))
             deviation > (sigma_upper * stdfunc(deviation))

        where::

            deviation = data - cenfunc(data [,axis=int])

        Defaults to the standard deviation (`numpy.std`).

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
    >>> nddata = NDData(np.arange(10))
    >>> columns = ['mean', 'median', 'mode', 'std', 'mad_std', 'min', 'max']
    >>> tbl = nddata_stats(nddata, columns=columns)
    >>> print(tbl)
    mean median mode      std         mad_std    min max
    ---- ------ ---- ------------- ------------- --- ---
     4.5    4.5  4.5 2.87228132327 3.70650554626   0   9
    """

    stats = []
    if not isinstance(nddata, list):
        nddata = np.atleast_1d(nddata)

    for nddata_obj in nddata:
        stats.append(NDDataStats(
            nddata_obj, sigma=sigma, sigma_lower=sigma_lower,
            sigma_upper=sigma_upper, iters=iters, cenfunc=cenfunc,
            stdfunc=stdfunc, lower_bound=lower_bound, upper_bound=upper_bound,
            mask_value=mask_value, mask_invalid=mask_invalid))

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
