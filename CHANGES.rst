0.2 (unreleased)
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

Astroimtools requires Astropy version 1.1 or later.
