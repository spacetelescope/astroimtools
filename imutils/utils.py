# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Image utilities.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np


__all__ = ['circular_footprint']


def circular_footprint(radius, dtype=np.int):
    """
    Create a 2D circular footprint.

    A pixel is considered to be entirely in or out of the footprint
    depending on whether its center is in or out of the circle.  The
    size of the output array is the minimal bounding box for the
    circular footprint.

    Parameters
    ----------
    radius : int
        The radius of the circular footprint.

    Other Parameters
    ----------------
    dtype : data-type
        The data type of the output `~numpy.ndarray`.

    Returns
    -------
    footprint : `~numpy.ndarray`
        A circular footprint where array elements are 1 within the
        footprint and 0 otherwise.
    """

    x = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, x)
    return np.array((xx**2 + yy**2) <= radius**2, dtype=dtype)
