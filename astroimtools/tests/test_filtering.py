# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose

from ..filtering import (circular_annulus_footprint, circular_footprint,
                         elliptical_annulus_footprint, elliptical_footprint)


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
    result = elliptical_footprint(4, 2, theta=np.pi / 2.0)
    ref = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0]])
    assert_allclose(result, ref)


def test_elliptical_annulus_footprint():
    result = elliptical_annulus_footprint(2, 4, 1, theta=np.pi / 2.0)
    ref = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [1, 0, 0, 0, 1],
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0]])
    assert_allclose(result, ref)
