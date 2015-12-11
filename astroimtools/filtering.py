# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Image filtering utilities.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.modeling.models import Ellipse2D


__all__ = ['circular_footprint', 'circular_annulus_footprint',
           'elliptical_footprint', 'elliptical_annulus_footprint']


def circular_footprint(radius, dtype=np.int):
    """
    Create a circular footprint.

    A pixel is considered to be entirely in or out of the footprint
    depending on whether its center is in or out of the footprint.  The
    size of the output array is the minimal bounding box for the
    footprint.

    Parameters
    ----------
    radius : int
        The radius of the circular footprint.

    dtype : data-type, optional
        The data type of the output `~numpy.ndarray`.

    Returns
    -------
    footprint : `~numpy.ndarray`
        A footprint where array elements are 1 within the footprint and
        0 otherwise.
    """

    x = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, x)
    return np.array((xx**2 + yy**2) <= radius**2, dtype=dtype)


def circular_annulus_footprint(radius_inner, radius_outer, dtype=np.int):
    """
    Create a circular annulus footprint.

    A pixel is considered to be entirely in or out of the footprint
    depending on whether its center is in or out of the footprint.  The
    size of the output array is the minimal bounding box for the
    footprint.

    Parameters
    ----------
    radius_inner : int
        The inner radius of the circular annulus.

    radius_outer : int
        The outer radius of the circular annulus.

    dtype : data-type, optional
        The data type of the output `~numpy.ndarray`.

    Returns
    -------
    footprint : `~numpy.ndarray`
        A footprint where array elements are 1 within the footprint and
        0 otherwise.
    """

    size = (radius_outer * 2) + 1
    y, x = np.mgrid[0:size, 0:size]
    circle_outer = Ellipse2D(1, radius_outer, radius_outer, radius_outer,
                             radius_outer, theta=0)(x, y)
    circle_inner = Ellipse2D(1., radius_outer, radius_outer, radius_inner,
                             radius_inner, theta=0)(x, y)
    return np.asarray(circle_outer - circle_inner, dtype=dtype)


def elliptical_footprint(a, b, theta=0, dtype=np.int):
    """
    Create an elliptical footprint.

    A pixel is considered to be entirely in or out of the footprint
    depending on whether its center is in or out of the footprint.  The
    size of the output array is the minimal bounding box for the
    footprint.

    Parameters
    ----------
    a : int
        The semimajor axis.

    b : int
        The semiminor axis.

    theta : float, optional
        The angle in radians of the semimajor axis.  The angle is
        measured counterclockwise from the positive x axis.

    dtype : data-type, optional
        The data type of the output `~numpy.ndarray`.

    Returns
    -------
    footprint : `~numpy.ndarray`
        A footprint where array elements are 1 within the footprint and
        0 otherwise.
    """

    size = (a * 2) + 1
    y, x = np.mgrid[0:size, 0:size]
    ellipse = Ellipse2D(1, a, a, a, b, theta=theta)(x, y)

    # crop to minimal bounding box
    yi, xi = ellipse.nonzero()
    idx = (slice(min(yi), max(yi) + 1), slice(min(xi), max(xi) + 1))
    return np.asarray(ellipse[idx], dtype=dtype)


def elliptical_annulus_footprint(a_inner, a_outer, b_outer, theta=0,
                                 dtype=np.int):
    """
    Create an elliptical annulus footprint.

    A pixel is considered to be entirely in or out of the footprint
    depending on whether its center is in or out of the footprint.  The
    size of the output array is the minimal bounding box for the
    footprint.

    Parameters
    ----------
    a_inner : int
        The inner semimajor axis.

    a_outer : int
        The outer semimajor axis.

    b_outer : int
        The outer semiminor axis.  The inner semiminor axis is
        calculated by using the same axis ratio as the semimajor axis:

        .. math::
            b_{inner} = b_{outer} \\frac{a_{inner}}{a_{outer}}

    theta : float, optional
        The angle in radians of the semimajor axis.  The angle is
        measured counterclockwise from the positive x axis.

    dtype : data-type, optional
        The data type of the output `~numpy.ndarray`.

    Returns
    -------
    footprint : `~numpy.ndarray`
        A footprint where array elements are 1 within the footprint and
        0 otherwise.
    """

    size = (a_outer * 2) + 1
    y, x = np.mgrid[0:size, 0:size]
    ellipse_outer = Ellipse2D(1, a_outer, a_outer, a_outer, b_outer,
                              theta=theta)(x, y)
    b_inner = b_outer * (a_inner / a_outer)
    ellipse_inner = Ellipse2D(1, a_outer, a_outer, a_inner, b_inner,
                              theta=theta)(x, y)
    annulus = ellipse_outer - ellipse_inner

    # crop to minimal bounding box
    yi, xi = annulus.nonzero()
    idx = (slice(min(yi), max(yi) + 1), slice(min(xi), max(xi) + 1))
    return np.asarray(annulus[idx], dtype=dtype)
