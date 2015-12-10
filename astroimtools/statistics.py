# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for image statistics.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.stats import (sigma_clip, biweight_location,
                           biweight_midvariance, mad_std)
from astropy.utils import lazyproperty
from astropy.table import Table
from astropy.nddata import NDData, support_nddata
from .core import mask_databounds


__all__ = ['ImageStatistics', 'imstats', 'minmax']


class ImageStatistics(object):
    """Class to calculate (sigma-clipped) image statistics."""

    def __init__(self, nddata, sigma=None, iters=1, cenfunc=np.ma.median,
                 varfunc=np.var, lower_bound=None, upper_bound=None,
                 mask_value=None):
        """
        Set the ``sigma`` keyword to perform sigma clipping.

        Parameters
        ----------
        nddata : `~astropy.nddata.NDData`
            NDData object containing the data array (and an optional
            mask) on which to calculate statistics.  Masked pixels are
            excluded when computing the image statistics.

        sigma : `None` or float, optional
            The number of standard deviations to use as the sigma
            clipping limit.  If `None` (default), then sigma clipping is
            not performed.

        iters : int or `None`, optional
            The number of sigma clipping iterations to perform, or `None`
            to clip until convergence is achieved (i.e. continue until
            the last iteration clips nothing).

        cenfunc : callable, optional
            The technique to compute the center for the sigma clipping.
            Must be a callable that takes in a masked array and outputs
            the central value.  Defaults to the median
            (`numpy.ma.median`).

        varfunc : callable, optional
            The technique to compute the standard deviation about the
            center for the sigma clipping. Must be a callable that takes
            in a masked array and outputs a width estimator.  Masked
            (rejected) pixels are those where::

                deviation**2 > sigma**2 * varfunc(deviation)

            Defaults to the variance (`numpy.var`).

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
        """

        if not isinstance(nddata, NDData):
            raise ValueError('nddata input must be an astropy.nddata.NDData '
                             'object')

        nddata = mask_databounds(nddata)
        if np.all(nddata.mask):
            raise ValueError('All data values are masked')

        if nddata.mask is not None:
            data = np.ma.MaskedArray(nddata.data, nddata.mask)
        else:
            data = nddata.data

        if sigma is not None:
            data = sigma_clip(data, sig=sigma, iters=iters)

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

            \\sigma \\approx \\frac{\\textrm{MAD}}{\Phi^{-1}(3/4)} \\approx 1.4826 \ \\textrm{MAD}

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


def imstats(nddata, sigma=None, iters=1, cenfunc=np.ma.median,
            varfunc=np.var, columns=None, lower_bound=None, upper_bound=None,
            mask_value=None):
    """
    Compute image statistics.  Set the ``sigma`` keyword to perform
    sigma clipping.

    Parameters
    ----------
    nddata : `~astropy.nddata.NDData` or list of `~astropy.nddata.NDData`
        NDData object containing the data array and optional mask on
        which to calculate statistics.  Masked pixels are excluded when
        computing the image statistics.

    sigma : `None` or float, optional
        The number of standard deviations to use as the sigma clipping
        limit.  If `None` (default), then sigma clipping is not
        performed.

    iters : int or `None`, optional
        The number of sigma clipping iterations to perform, or `None` to
        clip until convergence is achieved (i.e. continue until the last
        iteration clips nothing).

    cenfunc : callable, optional
        The technique to compute the center for the sigma clipping.
        Must be a callable that takes in a masked array and outputs the
        central value.  Defaults to the median (`numpy.ma.median`).

    varfunc : callable, optional
        The technique to compute the standard deviation about the center
        for the sigma clipping. Must be a callable that takes in a
        masked array and outputs a width estimator.  Masked (rejected)
        pixels are those where::

            deviation**2 > sigma**2 * varfunc(deviation)

        Defaults to the variance (`numpy.var`).

    columns : str or list of str, optional
        The names of columns, in order, to include in the output
        `~astropy.table.Table`.  The column names can include any of the
        statistic names: 'biweight_location', 'biweight_midvariance',
        'kurtosis', 'mad_std', 'max', 'mean', 'median', 'min', 'mode',
        'npixels', 'nrejected', 'skew', 'std' or a name of a key in the
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

    Returns
    -------
    table : `~astropy.table.Table`
        A table containing the calculated image statistics.  Each table
        row corresponds to a single data array.

    Examples
    --------
    >>> import numpy as np
    >>> from imutils import imstats
    >>> data = np.arange(10)
    >>> columns = ['mean', 'median', 'mode', 'std', 'mad_std', 'min', 'max']
    >>> tbl = imstats(data, columns=columns)
    >>> tbl
    <Table masked=False length=1>
      mean   median   mode       std         mad_std     min   max
    float64 float64 float64    float64       float64    int64 int64
    ------- ------- ------- ------------- ------------- ----- -----
        4.5     4.5     4.5 2.87228132327 3.70650554626     0     9
    """

    imstats = []
    if not isinstance(nddata, list):
        nddata = np.atleast_1d(nddata)

    if len(nddata) == 0:
        raise ValueError('nddata is an empty list')

    for nddata_obj in nddata:
        imstats.append(ImageStatistics(nddata_obj, sigma=sigma,
                                       iters=iters, cenfunc=cenfunc,
                                       varfunc=varfunc,
                                       lower_bound=lower_bound,
                                       upper_bound=upper_bound,
                                       mask_value=mask_value))

    output_columns = None
    default_columns = ['npixels', 'mean', 'std', 'min', 'max']
    property_columns = ['biweight_location', 'biweight_midvariance',
                        'kurtosis', 'mad_std', 'max', 'mean', 'median',
                        'min', 'mode', 'npixels', 'nrejected', 'skew',
                        'std']

    if columns is None:
        output_columns = default_columns
    else:
        output_columns = np.atleast_1d(columns)

    output_table = Table()
    for column in output_columns:
        if column not in property_columns:
            values = [nddata_obj.meta.get(column, None) for nddata_obj
                      in nddata]
        else:
            values = [getattr(imgstats, column) for imgstats in imstats]
        output_table[column] = values

    return output_table


@support_nddata
def minmax(data, mask=None, axis=None):
    """
    Return the minimum and maximum values of an array or the minimum and
    maximum along an axis.

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
    """

    if mask is not None:
        funcs = [np.ma.min, np.ma.max]
        data = np.ma.masked_array(data, mask=mask)
    else:
        funcs = [np.min, np.max]

    return funcs[0](data, axis=axis), funcs[1](data, axis=axis)
