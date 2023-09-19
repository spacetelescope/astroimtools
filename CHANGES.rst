0.4 (2023-09-18)
----------------

General
^^^^^^^

- The minimum required Python version is 3.9. [#92]

- The minimum required Numpy version is 1.22. [#92]

- The minimum required Astropy version is 5.0. [#82]

- The minimum required Scipy version is 1.7.2. [#92]

- The minimum required matplotlib version is 3.5. [#92]

- The minimum required photutils version is 1.5. [#92]

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

- The minimum required Python version is 3.6. [#70]

- The minimum required Numpy version is 1.16. [#70]

- The minimum required Astropy version is 3.2. [#70]

API changes
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

API changes
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
