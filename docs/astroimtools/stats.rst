Statistics
==========

Getting Started
---------------

minmax
^^^^^^

The :func:`~astroimtools.minmax` function returns the minimum and
maximum values of an array (or along an array axis)::

    >>> import numpy as np
    >>> from astroimtools import minmax
    >>> np.random.seed(12345)
    >>> data = np.random.random((3, 3))
    >>> data
    array([[ 0.92961609,  0.31637555,  0.18391881],
           [ 0.20456028,  0.56772503,  0.5955447 ],
           [ 0.96451452,  0.6531771 ,  0.74890664]])
    >>> minmax(data)
    (0.18391881167709445, 0.96451451973562163)

:func:`~astroimtools.minmax` also accepts a mask array to ignore certain data values::

    >>> mask = (data < 0.3)
    >>> mask
    array([[False, False,  True],
           [ True, False, False],
           [False, False, False]], dtype=bool)

    >>> minmax(data, mask=mask)
    (0.3163755545817859, 0.96451451973562163)

The minimum and maximum can also be determined along a particular axis::

    >>> minmax(data, axis=1)
    (array([ 0.18391881,  0.20456028,  0.6531771 ]),
     array([ 0.92961609,  0.5955447 ,  0.96451452]))


nddata_stats
^^^^^^^^^^^^

The :func:`~astroimtools.nddata_stats` function calculates various
statistics on `~astropy.nddata.NDData` objects.  Sigma-clipped
statistics can be calculated by specifying the ``sigma``,
``sigma_lower``, and/or ``sigma_upper`` keywords.  The currently
available statistics are:

  * ``'mean'``
  * ``'median'``
  * ``'mode'``
  * ``'std'``
  * ``'mad_std'``
  * ``'npixels'``
  * ``'nrejected'`` (number of pixels rejected by sigma clipping)
  * ``'min'``
  * ``'max'``
  * ``'biweight_location'``
  * ``'biweight_midvariance'``
  * ``'kurtosis'``
  * ``'skew'``

Here is a simple example::

    >>> import numpy as np
    >>> from astropy.nddata import NDData
    >>> from astroimtools import nddata_stats
    >>> nd1 = NDData(np.arange(10))
    >>> columns = ['mean', 'median', 'mode', 'std', 'mad_std', 'min', 'max']
    >>> tbl = nddata_stats(nd1, columns=columns)
    >>> print(tbl)
    mean median mode      std         mad_std    min max
    ---- ------ ---- ------------- ------------- --- ---
     4.5    4.5  4.5 2.87228132327 3.70650554626   0   9

Multiple `~astropy.nddata.NDData` objects can be input as a list,
resulting in a multi-row output table::

    >>> nd2 = NDData(np.arange(20))
    >>> tbl = nddata_stats([nd1, nd2], columns=columns)
    >>> print(tbl)
    mean median mode      std         mad_std    min max
    ---- ------ ---- ------------- ------------- --- ---
     4.5    4.5  4.5 2.87228132327 3.70650554626   0   9
     9.5    9.5  9.5 5.76628129734 7.41301109253   0  19

Sigma-clipped statistics can be calculated by specifying the
``sigma``, ``sigma_lower``, and/or ``sigma_upper`` keywords.  For this
example, let's sigma clip at 3 standard deviations::

    >>> np.random.seed(12345)
    >>> arr1 = np.random.random((100, 100))
    >>> arr1[40:50, 40:50] = 400
    >>> arr2 = np.random.random((100, 100))
    >>> arr2[40:50, 40:50] = 500
    >>> nd1 = NDData(arr1)
    >>> nd2 = NDData(arr2)
    >>> columns = ['npixels', 'nrejected', 'mean', 'median', 'std']
    >>> tbl = nddata_stats([nd1, nd2], sigma=3, columns=columns)
    >>> print(tbl)
    npixels nrejected      mean          median          std
    ------- --------- -------------- -------------- --------------
       9900       100 0.502487325973 0.502480088488 0.289861511955
       9900       100 0.500908259706 0.502192339391 0.289830629258


Reference/API
-------------

.. automodapi:: astroimtools.stats
    :no-heading:
