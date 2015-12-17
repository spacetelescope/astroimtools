# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose
from ..filtering import (circular_footprint, circular_annulus_footprint,
                         elliptical_footprint, elliptical_annulus_footprint)


def test_circular_footprint():
    result = circular_footprint(2)
    ref = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0]])
    assert_allclose(result, ref)


def test_circular_annulus_footprint():
    result = circular_annulus_footprint(1, 2)
    ref = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [1, 0, 0, 0, 1],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0]])
    assert_allclose(result, ref)


def test_elliptical_footprint():
    result = elliptical_footprint(3, 1, theta=np.pi/4.)
    ref = np.array([[1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1]])
    assert_allclose(result, ref)


def test_elliptical_annulus_footprint():
    result = elliptical_annulus_footprint(2, 4, 1, theta=np.pi/4.)
    ref = np.array([[0, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 0]])
    assert_allclose(result, ref)
