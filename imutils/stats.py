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
    def __init__(self, data, mask=None, name=None, sigma=3., iters=1,
                 cenfunc=np.median, varfunc=np.var):
        """
        Parameters
        ----------
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
        # TODO: replace with histogram-based mode
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
        A robust standard deviation using the median absolute deviation
        (MAD).

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


def imstats(data, mask=None, name=None, sigma=3., iters=1, cenfunc=np.median,
            varfunc=np.var, columns=None):
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
