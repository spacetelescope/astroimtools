************
Installation
************

Requirements
============

Astroimtools has the following strict requirements:

* `Python <https://www.python.org/>`_ 3.9 or later

* `NumPy <https://numpy.org/>`_ 1.22 or later

* `Astropy`_ 5.0 or later

* `SciPy <https://scipy.org/>`_ 1.7.2 or later

Some functionality is available only if the following optional
dependencies are installed:

* `Photutils <https://photutils.readthedocs.io/en/latest/>`_ 1.5 or
  later:  Used in cutout tools.

* `Matplotlib <https://matplotlib.org/>`_ 3.5 or later:  Used in
  cutout tools.

* `reproject <https://reproject.readthedocs.io/en/stable/>`_ 0.9 or
  later: Used in cutout tools.


Installing the latest released version
======================================

The latest released (stable) version of astroimtools can be installed
with `pip`_ by running::

    pip install astroimtools

If you want to install astroimtools along with all of its optional
dependencies, you can instead do::

    pip install "astroimtools[all]"


Testing an Installed Astroimtools
=================================

To test your installed version of astroimtools, you can
run the test suite using the `pytest`_ command. Running
the test suite requires installing the `pytest-astropy
<https://github.com/astropy/pytest-astropy>`_ (0.11 or later) package.

To run the test suite, use the following command::

    pytest --pyargs astroimtools

Any test failures can be reported to the `astroimtools issue tracker
<https://github.com/spacetelescope/astroimtools/issues>`_.

.. _pip: https://pip.pypa.io/en/latest/
.. _pytest: https://docs.pytest.org/en/latest/
