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
from astropy.extern.six import string_types
from astropy.nddata import NDData
from itertools import izip_longest
import warnings
from astropy.utils.exceptions import AstropyUserWarning


__all__ = ['ImageStatistics', 'imstats', 'minmax']


class ImageStatistics(object):
    """Class to calculate sigma-clipped image statistics."""

    def __init__(self, data, mask=None, name=None, sigma=3., iters=1,
                 cenfunc=np.ma.median, varfunc=np.var, lower_bound=None,
                 upper_bound=None):
        """
        Parameters
        ----------
        data : `~numpy.ndarray`
            Data array on which to calculate statistics.

        mask : bool `numpy.ndarray`, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked.  Masked pixels are excluded when computing the
            image statistics.

        name : str, optional
            The name to attach to the input data array.

        sigma : float, optional
            The number of standard deviations (*not* variances) to use
            as the clipping limit.

        iters : int or `None`, optional
            The number of iterations to perform clipping for, or `None`
            to clip until convergence is achieved (i.e. continue until
            the last iteration clips nothing).

        cenfunc : callable, optional
            The technique to compute the center for the clipping. Must
            be a callable that takes in a masked array and outputs the
            central value.  Defaults to the median (`numpy.ma.median`).

        varfunc : callable, optional
            The technique to compute the standard deviation about the
            center. Must be a callable that takes in a masked array and
            outputs a width estimator::

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
        """

        if np.any(np.isnan(data)):
            warnings.warn(('data array contains at least one np.nan. '
                           'Please create a mask for these values (e.g. '
                           'with np.isnan())'), AstropyUserWarning)

        mask = self._create_mask(data, mask, lower_bound, upper_bound)
        if mask is not None:
            if mask.shape != data.shape:
                raise ValueError('mask and data must have the same shape')
            if np.all(mask):
                raise ValueError('All data values are masked')
            data = np.ma.MaskedArray(data, mask)

        data_clip = sigma_clip(data, sig=sigma, iters=iters)
        self.goodvals = data_clip.data[~data_clip.mask]
        if name is not None and not isinstance(name, string_types):
            raise ValueError('name must be a string')
        self.name = name

    def __getitem__(self, key):
        return getattr(self, key, None)

    @staticmethod
    def _create_mask(data, mask, lower_bound, upper_bound):
        if lower_bound is not None and upper_bound is not None:
            bound_mask = np.logical_or(data < lower_bound, data > upper_bound)
        elif lower_bound is not None and upper_bound is None:
            bound_mask = (data < lower_bound)
        elif lower_bound is None and upper_bound is not None:
            bound_mask = (data > upper_bound)
        else:
            bound_mask = None

        if bound_mask is not None:
            if mask is not None:
                mask = np.logical_or(mask, bound_mask)
            else:
                mask = bound_mask
        return mask

    @lazyproperty
    def npixels(self):
        """
        The number of unclipped pixels.
        """
        return len(self.goodvals)

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


def imstats(nddata, sigma=3., iters=1, cenfunc=np.ma.median,
            varfunc=np.var, columns=None, lower_bound=None, upper_bound=None):
    """
    Compute image statistics.

    Parameters
    ----------
    nddata : `~astropy.nddata.NDData` or list of `~astropy.nddata.NDData`
        NDData object containing the data array and optional mask on
        which to calculate statistics.  Masked pixels are excluded when
        computing the image statistics.

    sigma : float, optional
        The number of standard deviations (*not* variances) to use
        as the clipping limit.

    iters : int or `None`, optional
        The number of iterations to perform clipping for, or `None`
        to clip until convergence is achieved (i.e. continue until
        the last iteration clips nothing).

    cenfunc : callable, optional
        The technique to compute the center for the clipping. Must
        be a callable that takes in a masked array and outputs the
        central value.  Defaults to the median (`numpy.ma.median`).

    varfunc : callable, optional
        The technique to compute the standard deviation about the
        center. Must be a callable that takes in a masked array and
        outputs a width estimator::

            deviation**2 > sigma**2 * varfunc(deviation)

        Defaults to the variance (`numpy.var`).

    columns : str or list of str, optional
        The names of columns, in order, to include in the output
        `~astropy.table.Table`.  The allowed column names are
        'biweight_location', 'biweight_midvariance', 'kurtosis',
        'mad_std', 'max', 'mean', 'median', 'min', 'mode', 'npixels',
        'skew', and 'std'.  The default is ``['name', 'npixels', 'mean',
        'std', 'min', 'max']``.

    lower_bound : float, optional
        The minimum data value to include in the statistics.  All pixel
        values less than ``lower_bound`` will be ignored.  `None` means
        that no lower bound is applied (default).

    upper_bound : float, optional
        The maximum data value to include in the statistics.  All pixel
        values greater than ``upper_bound`` will be ignored.  `None` means
        that no upper bound is applied (default).

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
    if isinstance(nddata, list):
        if len(nddata) == 0:
            raise ValueError('nddata is an empty list')

        for nddata_obj in nddata:
            if not isinstance(nddata_obj, NDData):
                raise ValueError('nddata must be an astropy.nddata.NDData '
                                 'object')
            imstats.append(ImageStatistics(nddata_obj.data,
                mask=nddata_obj.mask, name=nddata_obj.meta.get('name', None),
                sigma=sigma, iters=iters, cenfunc=cenfunc, varfunc=varfunc,
                lower_bound=lower_bound, upper_bound=upper_bound))
    else:
        if not isinstance(nddata, NDData):
            raise ValueError('nddata must be an astropy.nddata.NDData '
                             'object')
        imstats.append(ImageStatistics(nddata.data, mask=nddata.mask,
            name=nddata.meta.get('name', None), sigma=sigma, iters=iters,
            cenfunc=cenfunc, varfunc=varfunc, lower_bound=lower_bound,
            upper_bound=upper_bound))

    output_columns = None
    default_columns = ['name', 'npixels', 'mean', 'std', 'min', 'max']
    if columns is not None:
        output_columns = np.atleast_1d(columns)
    if output_columns is None:
        output_columns = default_columns

    output_table = Table()
    for column in output_columns:
        values = [getattr(imgstats, column) for imgstats in imstats]
        output_table[column] = values

    return output_table


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
    """

    if mask is not None:
        funcs = [np.ma.min, np.ma.max]
        data = np.ma.masked_array(data, mask=mask)
    else:
        funcs = [np.min, np.max]

    return funcs[0](data, axis=axis), funcs[1](data, axis=axis)
