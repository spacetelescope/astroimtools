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
from itertools import izip_longest


__all__ = ['ImageStatistics', 'imstats']


class ImageStatistics(object):
    """Class to calculate sigma-clipped image statistics."""

    def __init__(self, data, mask=None, name=None, sigma=3., iters=1,
                 cenfunc=np.ma.median, varfunc=np.var):
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
        """

        if mask is not None:
            if mask.shape != data.shape:
                raise ValueError('mask and data must have the same shape')
            data = np.ma.MaskedArray(data, mask)

        data_clip = sigma_clip(data, sig=sigma, iters=iters)
        self.goodvals = data_clip.data[~data_clip.mask]
        if name is not None and not isinstance(name, string_types):
            raise ValueError('name must be a string')
        self.name = name

    def __getitem__(self, key):
        return getattr(self, key, None)

    @lazyproperty
    def npix(self):
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


def imstats(data, mask=None, name=None, sigma=3., iters=1,
            cenfunc=np.ma.median, varfunc=np.var, columns=None):
    """
    Compute image statistics.

    Parameters
    ----------
    data : `~numpy.ndarray` or list of `~numpy.ndarray`
        Data array(s) on which to calculate statistics.

    mask : bool `numpy.ndarray` or list of bool `~numpy.ndarray`, optional
        A boolean mask (or list of masks) with the same shape as
        ``data``, where a `True` value indicates the corresponding
        element of ``data`` is masked.  Masked pixels are excluded when
        computing the image statistics.

    name : str or list of str
        The name (or list of names) to attach to the input data array(s).

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
        'mad_std', 'max', 'mean', 'median', 'min', 'mode', 'npix',
        'skew', and 'std'.  The default is ``['name', 'npix', 'mean',
        'std', 'min', 'max']``.

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
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError('data is an empty list')

        if mask is not None:
            if not isinstance(mask, list):
                raise ValueError('mask must be a list if data is a list')
            if len(mask) != len(data):
                raise ValueError('length of mask list must match length of '
                                 'data list')
        else:
            mask = [None]

        if name is not None:
            if not isinstance(name, list):
                raise ValueError('name must be a list if data is a list')
            if len(name) != len(data):
                raise ValueError('length of name list must match length of '
                                 'data list')
        else:
            name = [None]

        for (data_arr, mask_arr, name_val) in izip_longest(data, mask, name):
            imstats.append(ImageStatistics(data_arr, mask=mask_arr,
                                           name=name_val, sigma=sigma,
                                           iters=iters, cenfunc=cenfunc,
                                           varfunc=varfunc))
    else:
        imstats.append(ImageStatistics(data, mask=mask, name=name, sigma=3.,
                                       iters=1, cenfunc=np.median,
                                       varfunc=np.var))

    output_columns = None
    default_columns = ['name', 'npix', 'mean', 'std', 'min', 'max']
    if columns is not None:
        output_columns = np.atleast_1d(columns)
    if output_columns is None:
        output_columns = default_columns

    output_table = Table()
    for column in output_columns:
        values = [getattr(imgstats, column) for imgstats in imstats]
        output_table[column] = values

    return output_table
