************
Installation
************

Requirements
============

Astroimtools has the following strict requirements:

* `Python <http://www.python.org/>`_ |minimum_python_version| or later

* `Numpy <http://www.numpy.org/>`_ |minimum_numpy_version| or later

* `Astropy`_ 2.0 or later

Astroimtools also depends on `pytest-astropy
<https://github.com/astropy/pytest-astropy>`_ (0.4 or later) to run
the test suite.

Additionally, some functionality is available only if the following
optional dependencies are installed:

* `Scipy`_ 0.16 or later


Installing the latest released version
======================================

The latest released (stable) version of Astroimtools can be installed
either with `conda`_ or `pip`_.


Using conda
-----------

Astroimtools can be installed with `conda`_ using the `Astroconda Anaconda
channel <https://astroconda.readthedocs.io/en/latest/>`_::

    conda install astroimtools -c http://ssb.stsci.edu/astroconda


Using pip
---------

To install the latest released version of Astroimtools with `pip`_,
simply run::

    pip install --no-deps astroimtools

.. note::

    The ``--no-deps`` flag is optional, but highly recommended if you
    already have Numpy and Astropy installed, since otherwise pip will
    sometimes try to "help" you by upgrading your Numpy and Astropy
    installations, which may not always be desired.

.. note::

    If you get a ``PermissionError`` this means that you do not have
    the required administrative access to install new packages to your
    Python installation.  In this case you may consider using the
    ``--user`` option to install the package into your home directory.
    You can read more about how to do this in the `pip documentation
    <http://www.pip-installer.org/en/1.2.1/other-tools.html#using-pip-with-the-user-scheme>`_.

    Do **not** install Astroimtools or other third-party packages using
    ``sudo`` unless you are fully aware of the risks.


Testing an Installed Astroimtools
=================================

The easiest way to test your installed version of Astroimtools is
running correctly is to use the :func:`astroimtools.test()` function:

.. doctest-skip::

    >>> import astroimtools
    >>> astroimtools.test()

The tests should run and print out any failures, which you can report
to the `Astroimtools issue tracker
<http://github.com/spacetelescope/astroimtools/issues>`_.

.. note::

    This way of running the tests may not work if you do it in the
    Astroimtools source distribution directory.


.. _Scipy: http://www.scipy.org/
.. _pip: https://pip.pypa.io/en/latest/
.. _conda: http://conda.pydata.org/docs/
