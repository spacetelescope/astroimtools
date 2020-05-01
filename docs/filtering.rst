Filtering Tools
===============

Astroimtools provides functions for generating circular and elliptical
(or annulus versions of each) footprint arrays, which can be used to
perform specialized image filtering.  The functions are:

  * `~astroimtools.filtering.circular_footprint`
  * `~astroimtools.filtering.circular_annulus_footprint`
  * `~astroimtools.filtering.elliptical_footprint`
  * `~astroimtools.filtering.elliptical_annulus_footprint`

A pixel is considered to be entirely in or out of a particular
footprint depending on whether its center is in or out of the
footprint. The size of the output array is the minimal bounding box
for the footprint.

An example of use case of these footprints is median filtering an
image using a circular (or elliptical) ring filter (see below).


Getting Started
---------------

Let's generate footprint arrays using each function.

Here's a circular footprint with radius of 2 pixels::

    >>> from astroimtools import circular_footprint
    >>> circular_footprint(2)
    array([[0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0]])

Here's a circular annulus footprint with an inner radius of 1 pixel
and an outer radius of 2 pixels::

    >>> from astroimtools import circular_annulus_footprint
    >>> circular_annulus_footprint(1, 2)
    array([[0, 0, 1, 0, 0],
           [0, 1, 0, 1, 0],
           [1, 0, 0, 0, 1],
           [0, 1, 0, 1, 0],
           [0, 0, 1, 0, 0]])

Here's an elliptical footprint with a semimajor axis of 3 and a
semiminor axis of 1 where the semimajor axis is rotated 45 degrees
counterclockwise from the positive x axis (note that the data values
are printed with y=0 at the top such that the array appears vertically
flipped)::

    >>> import numpy as np
    >>> from astroimtools import elliptical_footprint
    >>> elliptical_footprint(3, 1, theta=np.pi/4.)
    array([[1, 1, 0, 0, 0],
           [1, 1, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1]])

Here's an elliptical annulus footprint with an inner semimajor axis of
2, an outer semimajor axis of 4, and an inner semiminor axis of 1
where the semimajor axis is rotated 45 degrees counterclockwise from
the positive x axis (note that the data values are printed with y=0 at
the top such that the array appears vertically flipped)::

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


Ring Median Filter
------------------

The :func:`~astroimtools.filtering.circular_annulus_footprint`
function can be used to generate a ring-shaped footprint to implement
a ring median filter.  The effect of the ring filter is to remove
objects from an image which have a size less than the inner radius and
replace them with an estimate of the local background value
(determined by the median image values within the footprint).

First, let's create a ring filter with an inner radius of 10 pixels
and and outer radius of 12 pixels.  This will filter objects with a
size of 10 pixels or less::

    >>> from astroimtools import circular_annulus_footprint
    >>> fp = circular_annulus_footprint(10, 12)

We can now use this footprint with the
:func:`~scipy.ndimage.median_filter` function from `Scipy
<https://www.scipy.org/>`_:

.. doctest-skip::

    >>> from scipy.ndimage import median_filter
    >>> result = median_filter(data, footprint=fp)

Several additional filters are available in `scipy.ndimage` that
accept ``footprint`` inputs.


Reference/API
-------------

.. automodapi:: astroimtools.filtering
    :no-heading:
