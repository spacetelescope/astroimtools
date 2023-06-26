************
Installation
************

Requirements
============

Astroimtools has the following strict requirements:

* `Python <https://www.python.org/>`_ 3.8 or later

* `Numpy <https://numpy.org/>`_ 1.18 or later

* `Astropy`_ 5.0 or later

* `Scipy <https://scipy.org/>`_ 1.6.0 or later

`pytest-astropy <https://github.com/astropy/pytest-astropy>`_ is
required to run the test suite.

Some functionality is available only if the following optional
dependencies are installed:

* `Photutils <https://photutils.readthedocs.io/en/latest/>`_ 1.0.0 or
  later:  Used in cutout tools.

* `Matplotlib <https://matplotlib.org/>`_ 3.1 or later:  Used in
  cutout tools.


Installing the latest released version
======================================

The latest released (stable) version of astroimtools can be installed
with `pip`_ by running::

    pip install astroimtools

If you want to make sure that none of your existing dependencies get
upgraded, instead you can do::

    pip install astroimtools --no-deps


Testing an Installed Astroimtools
=================================

The easiest way to test your installed version of astroimtools is
running correctly is to use the ``test()`` function:

.. doctest-skip::

    >>> import astroimtools
    >>> astroimtools.test()

Note that this may not work if you start Python from within the
astroimtools source distribution directory.

The tests should run and print out any failures, which you can report
to the `astroimtools issue tracker
<https://github.com/spacetelescope/astroimtools/issues>`_.

.. _pip: https://pip.pypa.io/en/latest/
