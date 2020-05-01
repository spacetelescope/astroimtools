NDData Adapters
===============

Astroimtools provides some very basic tools for interfacing
`~astropy.nddata.NDData` objects with FITS files.

Getting Started
---------------

basic_nddata_to_fits
^^^^^^^^^^^^^^^^^^^^

The :func:`~astroimtools.nddata_adapters.basic_nddata_to_fits`
function writes a `~astropy.nddata.NDData` object to a FITS file.

The `~astropy.nddata.NDData` data values will be saved in an FITS
extension called 'SCI'.  This simple writer will also attempt to save
the `~astropy.nddata.NDData` uncertainty and mask to a FITS 'ERROR'
and 'MASK' extension, respectively.

If present, the `~astropy.nddata.NDData` meta dictionary will be
stored as the FITS header.

Here's a simple example:

.. doctest-skip::

    >>> from astroimtools import basic_nddata_to_fits
    >>> from astropy.nddata import NDData
    >>> nd = NDData(np.random.random((500, 500)
    >>> basic_nddata_to_fits(nd, 'example_data.fits')

Set the ``clobber`` keyword to `True` to overwrite any existing files:

.. doctest-skip::

    >>> basic_nddata_to_fits(nd, 'example_data.fits', clobber=True)


basic_fits_to_nddata
^^^^^^^^^^^^^^^^^^^^

The :func:`~astroimtools.nddata_adapters.basic_fits_to_nddata`
function reads a single FITS extension into a `~astropy.nddata.NDData`
object.  The extension to read is set via the ``exten`` keyword, which
defaults to 0.

Here's a simple example:

.. doctest-skip::

    >>> from astroimtools import basic_fits_to_nddata
    >>> nddata = basic_fits_to_nddata('example_data.fits', exten=0)

The data from the FITS extension are in ``nddata.data`` and the header
values are in ``nddata.meta``.  Note that the primary FITS header is
always included in resulting `~astropy.nddata.NDData` meta `dict`,
regardless of the value of ``exten``.


Reference/API
-------------

.. automodapi:: astroimtools.nddata_adapters
    :no-heading:
