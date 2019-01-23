************
Installation
************

Requirements
============

Astroimtools has the following strict requirements:

* `Python <http://www.python.org/>`_ 2.7, 3.3, 3.4 or 3.5

* `Numpy <http://www.numpy.org/>`_ 1.6 or later

* `Astropy`_ 3.1 or later

Some functionality is available only if the following optional
dependencies are installed:

* `Scipy`_ 0.15 or later

.. _Scipy: http://www.scipy.org/
.. _pip: https://pip.pypa.io/en/latest/
.. _conda: http://conda.pydata.org/docs/


Installing Astroimtools Using pip
=================================

To install the latest Astroimtools **stable** version with `pip`_,
simply run::

    pip install --no-deps astroimtools

To install the current Astroimtools **development** version using
`pip`_::

    pip install --no-deps git+https://github.com/spacetelescope/astroimtools.git

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

    Do **not** install Astroimtools or other third-party packages
    using ``sudo`` unless you are fully aware of the risks.


Obtaining the Source Package
============================

Stable Version
--------------

The latest stable source package for Astroimtools can be `downloaded
here <https://pypi.python.org/pypi/astroimtools>`_.


Development Version
-------------------

The latest development version of Astroimtools can be cloned from
github using this command::

   git clone https://github.com/spacetelescope/astroimtools.git


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
