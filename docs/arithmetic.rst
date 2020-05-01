Image Arithmetic
================

Getting Started
---------------

nddata_arith
^^^^^^^^^^^^

The :func:`~astroimtools.arithmetic.nddata_arith` function can be used
to perform basic arithmetic on two `~astropy.nddata.NDData` objects,
returning a new `~astropy.nddata.NDData` object.  The operations that
can be performed include:

  * ``'+'``:  addition
  * ``'-'``:  subtraction
  * ``'*'``:  multiplication
  * ``'/'``:  floating-point division
  * ``'//'``:  integer-truncated division
  * ``'min'``:  the element-wise minimum value
  * ``'max'``:  the element-wise maximum value

First, let's define two `~astropy.nddata.NDData` objects::

    >>> from astroimtools import nddata_arith
    >>> from astropy.nddata import NDData
    >>> nd1 = NDData([0, 1, 2, 3, 4])
    >>> nd2 = NDData([1, 7, 5, 4, 2])

Now let's add the objects::

    >>> nd = nddata_arith(nd1, nd2, '+')
    >>> nd.data
    array([1, 8, 7, 7, 6])

or take the element-wise minimum of the two::

    >>> nd = nddata_arith(nd1, nd2, 'min')
    >>> nd.data
    array([0, 1, 2, 3, 2])

The operations can also be performed with a single
`~astropy.nddata.NDData` object and a scalar value::

    >>> nd = nddata_arith(nd1, 2, '/')
    >>> nd.data  # doctest: +FLOAT_CMP
    array([0. , 0.5, 1. , 1.5, 2. ])

The ``'//'`` operator performs integer-truncated division::

    >>> nd = nddata_arith(nd1, 2, '//')
    >>> nd.data  # doctest: +FLOAT_CMP
    array([0, 0, 1, 1, 2])

The operand can also be applied to one or more NDData meta keywords.
Let's add an exposure time value to the ``meta`` dictionary of both
NDData objects:

    >>> nd1.meta['exptime'] = 500
    >>> nd2.meta['exptime'] = 1000

Now, we include ``'exptime'`` in the keywords input (which could also
be a list of keywords) to perform the operation on the exposure time::

    >>> nd = nddata_arith(nd1, nd2, '+', keywords='exptime')
    >>> nd.data
    array([1, 8, 7, 7, 6])
    >>> nd.meta['exptime']
    1500

If present, the `~astropy.nddata.NDData` masks are used in the
operation such that if either value is masked then the output value
will be masked.  First, let's add a mask for each::

    >>> nd1.mask = (nd1.data > 3)
    >>> nd2.mask = (nd2.data < 2)

and then add them::

    >>> nd = nddata_arith (nd1, nd2, '+')
    >>> nd.data
    array([0, 8, 7, 7, 0])
    >>> nd.mask
    array([ True, False, False, False,  True]...)

Note that the resulting `~astropy.nddata.NDData` object's mask is
propagated from the input masks.  The data fill value for masked
values can be set with the ``fill_value`` keyword.


Reference/API
-------------

.. automodapi:: astroimtools.arithmetic
    :no-heading:
