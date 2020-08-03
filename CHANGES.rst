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
