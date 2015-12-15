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

    Examples
    --------
    >>> from astroimtools import circular_footprint
    >>> circular_footprint(2)
    array([[0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0]])
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

    Examples
    --------
    >>> from astroimtools import circular_footprint
    >>> circular_annulus_footprint(1, 2)
    array([[0, 0, 1, 0, 0],
           [0, 1, 0, 1, 0],
           [1, 0, 0, 0, 1],
           [0, 1, 0, 1, 0],
           [0, 0, 1, 0, 0]])
    """

    if radius_inner > radius_outer:
        raise ValueError('radius_outer must be >= radius_inner')

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
        The rotation angle in radians of the semimajor axis.  The angle
        is measured counterclockwise from the positive x axis.

    dtype : data-type, optional
        The data type of the output `~numpy.ndarray`.

    Returns
    -------
    footprint : `~numpy.ndarray`
        A footprint where array elements are 1 within the footprint and
        0 otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from astroimtools import elliptical_footprint
    >>> elliptical_footprint(3, 1, theta=np.pi/4.)
    array([[1, 1, 0, 0, 0],
           [1, 1, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1]])
    """

    if b > a:
        raise ValueError('a must be >= b')

    size = (a * 2) + 1
    y, x = np.mgrid[0:size, 0:size]
    ellipse = Ellipse2D(1, a, a, a, b, theta=theta)(x, y)

    # crop to minimal bounding box
    yi, xi = ellipse.nonzero()
    idx = (slice(min(yi), max(yi) + 1), slice(min(xi), max(xi) + 1))
    return np.asarray(ellipse[idx], dtype=dtype)


def elliptical_annulus_footprint(a_inner, a_outer, b_inner, theta=0,
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

    b_inner : int
        The inner semiminor axis.  The outer semiminor axis is
        calculated using the same axis ratio as the semimajor axis:

        .. math::
            b_{outer} = b_{inner} \\left( \\frac{a_{outer}}{a_{inner}}
            \\right)

    theta : float, optional
        The rotation angle in radians of the semimajor axis.  The angle
        is measured counterclockwise from the positive x axis.

    dtype : data-type, optional
        The data type of the output `~numpy.ndarray`.

    Returns
    -------
    footprint : `~numpy.ndarray`
        A footprint where array elements are 1 within the footprint and
        0 otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from astroimtools import elliptical_annulus_footprint
    >>> elliptical_annulus_footprint(2, 4, 1, theta=np.pi/4.)
    array([[0, 1, 1, 0, 0, 0, 0],
           [1, 1, 1, 1, 0, 0, 0],
           [1, 1, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 1, 0],
           [0, 0, 1, 0, 0, 1, 1],
           [0, 0, 0, 1, 1, 1, 1],
           [0, 0, 0, 0, 1, 1, 0]])
    """

    if a_inner > a_outer:
        raise ValueError('a_outer must be >= a_inner')

    if b_inner > a_inner:
        raise ValueError('a_inner must be >= b_inner')

    size = (a_outer * 2) + 1
    y, x = np.mgrid[0:size, 0:size]
    b_outer = b_inner * (a_outer / a_inner)
    ellipse_outer = Ellipse2D(1, a_outer, a_outer, a_outer, b_outer,
                              theta=theta)(x, y)
    ellipse_inner = Ellipse2D(1, a_outer, a_outer, a_inner, b_inner,
                              theta=theta)(x, y)
    annulus = ellipse_outer - ellipse_inner

    # crop to minimal bounding box
    yi, xi = annulus.nonzero()
    idx = (slice(min(yi), max(yi) + 1), slice(min(xi), max(xi) + 1))
    return np.asarray(annulus[idx], dtype=dtype)
