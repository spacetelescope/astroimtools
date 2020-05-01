Statistics
==========

Getting Started
---------------

minmax
^^^^^^

The :func:`~astroimtools.stats.minmax` function returns the minimum
and maximum values of an array (or along an array axis)::

    >>> import numpy as np
    >>> from astroimtools import minmax
    >>> np.random.seed(12345)
    >>> data = np.random.random((3, 3))
    >>> data  # doctest: +FLOAT_CMP
    array([[0.92961609, 0.31637555, 0.18391881],
           [0.20456028, 0.56772503, 0.5955447 ],
           [0.96451452, 0.6531771 , 0.74890664]])
    >>> minmax(data)  # doctest: +FLOAT_CMP
    (0.18391881167709445, 0.9645145197356216)

:func:`~astroimtools.stats.minmax` also accepts a mask array to ignore
certain data values::

    >>> mask = (data < 0.3)
    >>> mask
    array([[False, False,  True],
           [ True, False, False],
           [False, False, False]]...)

    >>> minmax(data, mask=mask)  # doctest: +FLOAT_CMP
    (0.3163755545817859, 0.9645145197356216)

The minimum and maximum can also be determined along a particular axis::

    >>> minmax(data, axis=1)  # doctest: +FLOAT_CMP
    (array([0.18391881, 0.20456028, 0.6531771 ]),
     array([0.92961609, 0.5955447 , 0.96451452]))


nddata_stats
^^^^^^^^^^^^

The :func:`~astroimtools.stats.nddata_stats` function calculates
various statistics on `~astropy.nddata.NDData` objects.  Sigma-clipped
statistics can be calculated by inputting a `~astropy.stats.SigmaClip`
instance to the ``sigma_clip`` keyword.  The currently available
statistics are:

  * ``'mean'``
  * ``'median'``
  * ``'mode'``
  * ``'std'``
  * ``'mad_std'``
  * ``'npixels'``
  * ``'nrejected'`` (number of pixels rejected by masking or sigma clipping)
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
    >>> data = np.arange(10)
    >>> data[0] = 100.
    >>> nddata = NDData(data)
    >>> columns = ['mean', 'median', 'mode', 'std', 'mad_std', 'min', 'max']
    >>> tbl = nddata_stats(nddata, columns=columns)
    >>> for col in tbl.colnames:
    ...     tbl[col].info.format = '%.8g'  # for consistent table output
    >>> print(tbl)
    mean median  mode    std     mad_std  min max
    ---- ------ ----- --------- --------- --- ---
    14.5    5.5 -12.5 28.605069 3.7065055   1 100

Multiple `~astropy.nddata.NDData` objects can be input as a list,
resulting in a multi-row output table::

    >>> nd1 = NDData(np.arange(10))
    >>> nd2 = NDData(np.arange(20))
    >>> tbl = nddata_stats([nd1, nd2], columns=columns)
    >>> for col in tbl.colnames:
    ...     tbl[col].info.format = '%.8g'  # for consistent table output
    >>> print(tbl)
    mean median mode    std     mad_std  min max
    ---- ------ ---- --------- --------- --- ---
     4.5    4.5  4.5 2.8722813 3.7065055   0   9
     9.5    9.5  9.5 5.7662813 7.4130111   0  19

Sigma-clipped statistics can be calculated by specifying the
``sigma_clip`` keyword.  For this example, let's sigma clip at 2.5
standard deviations::

    >>> np.random.seed(12345)
    >>> arr1 = np.random.random((100, 100))
    >>> arr1[40:50, 40:50] = 400
    >>> arr2 = np.random.random((100, 100))
    >>> arr2[40:50, 40:50] = 500
    >>> nd1 = NDData(arr1)
    >>> nd2 = NDData(arr2)
    >>> from astropy.stats import SigmaClip
    >>> sigclip = SigmaClip(sigma=3.)
    >>> columns = ['npixels', 'nrejected', 'mean', 'median', 'std']
    >>> tbl = nddata_stats([nd1, nd2], sigma_clip=sigclip, columns=columns)
    >>> for col in tbl.colnames:
    ...     tbl[col].info.format = '%.8g'  # for consistent table output
    >>> print(tbl)
    npixels nrejected    mean      median      std
    ------- --------- ---------- ---------- ----------
       9900       100 0.50248733 0.50248009 0.28986151
       9900       100 0.50090826 0.50219234 0.28983063


Reference/API
-------------

.. automodapi:: astroimtools.stats
    :no-heading:
