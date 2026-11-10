0.5 (unreleased)
----------------

General
^^^^^^^

- The minimum required Python is now 3.12. [#137]

- The minimum required NumPy is now 2.0. [#137]

- The minimum required Astropy is now 7.0. [#137]

- The minimum required SciPy is now 1.15. [#137]

- The minimum required Matplotlib is now 3.10. [#137]

- The minimum required photutils is now 2.0. [#137]

- The minimum required reproject is now 0.15. [#137]

- The source distribution no longer includes the example data files
  used by the cutouts notebook in the documentation. [#139]

New Features
^^^^^^^^^^^^

Bug Fixes
^^^^^^^^^

- Fixed ``show_cutout_with_slit`` to import the aperture classes from
  ``photutils.aperture``, which is required for photutils 2.0 and later.
  [#137]

- Fixed ``basic_nddata_to_fits``, which always raised an error, and the
  ``imarith`` script, which uses it. The ``NDData`` uncertainty is now
  also correctly written to the ``ERROR`` extension. [#140]

- Fixed ``make_cutouts`` with ``apply_rotation=True`` for images with a
  WCS defined by CD or PC keywords. Previously, it raised an error or
  wrote cutouts containing only NaN values. [#140]

- Fixed ``show_cutout_with_slit`` to save the plot when both ``ax`` and
  ``plotname`` are input. [#140]

- Fixed ``basic_nddata_to_fits`` to not copy meta keywords that describe
  the structure of a FITS extension (e.g., ``EXTNAME``) to the primary
  header, and fixed ``basic_fits_to_nddata`` to not add the extension
  keywords to the primary HDU. Previously, a FITS extension could not
  be read back by its name. [#140]

API Changes
^^^^^^^^^^^

- The ``astroimtools.test`` function has been removed. Instead use the
  ``pytest --pyargs astroimtools`` command. [#106]


0.4 (2023-09-18)
----------------

General
^^^^^^^

- The minimum required Python is now 3.9. [#92]

- The minimum required NumPy is now 1.22. [#92]

- The minimum required Astropy is now 5.0. [#82]

- The minimum required SciPy is now 1.7.2. [#92]

- The minimum required Matplotlib is now 3.5. [#92]

- The minimum required photutils is now 1.5. [#92]

Bug Fixes
^^^^^^^^^

- Fixed ``make_cutouts`` to use the ``overwrite`` keyword instead of the
  removed ``clobber`` keyword when writing FITS files. [#91]

API Changes
^^^^^^^^^^^

- The ``clobber`` keyword is deprecated in favor of ``overwrite`` in
  ``make_cutouts`` and ``basic_nddata_to_fits``. [#99]


0.3 (2020-08-03)
----------------

General
^^^^^^^

- The minimum required Python is now 3.6. [#70]

- The minimum required NumPy is now 1.16. [#70]

- The minimum required Astropy is now 3.2. [#70]

API Changes
^^^^^^^^^^^

- The ``mask_databounds`` function no longer raises a warning if the
  ``mask_invalid`` keyword is set to ``True``. [#72]


0.2 (2019-03-01)
----------------

General
^^^^^^^

- Astroimtools requires Python version 3.5 or later.

- Astroimtools requires Numpy version 1.11 or later.

- Astroimtools requires Astropy version 3.1 or later. [#67]

New Features
^^^^^^^^^^^^

- Added ``cutout_tools`` module, including relevant documentation and
  example Jupyter notebooks. [#31]

API Changes
^^^^^^^^^^^

- ``nddata_stats`` and ``NDDataStats`` sigma-clipping parameters are
  now specified by passing a ``astropy.stats.SigmaClip`` instance to the
  ``sigma_clip`` keyword. [#66]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Use ``argparse`` directly instead of deprecated
  ``astropy.utils.compat.argparse``.

- Updated ``astropy-helpers`` to v1.1.1.


0.1 (2015-12-17)
----------------

- Astroimtools requires Astropy version 1.1 or later.
