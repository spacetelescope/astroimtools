# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions for cutout images."""

# STDLIB
import os
from functools import partial
import math

# THIRD-PARTY
import numpy as np

# ASTROPY
from astropy import log
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata.utils import (Cutout2D, NoOverlapError)
from astropy.table import QTable
import astropy.units as u
from astropy.wcs import WCS, NoConvergence


__all__ = ['make_cutouts', 'show_cutout_with_slit']


def make_cutouts(catalogname, imagename, image_label, apply_rotation=False,
                 table_format='ascii.ecsv', image_ext=0, clobber=False,
                 verbose=True):
    """Make cutouts from a 2D image and write them to FITS files.

    Catalog must have the following columns with unit info, where applicable:

        * ``'id'`` - ID string; no unit necessary.
        * ``'ra'`` - RA (e.g., in degrees).
        * ``'dec'`` - DEC (e.g., in degrees).
        * ``'cutout_x_size'`` - Cutout width (e.g., in arcsec).
        * ``'cutout_y_size'`` - Cutout height (e.g., in arcsec).
        * ``'cutout_pa'`` - Cutout angle (e.g., in degrees). This is only
          use if user chooses to rotate the cutouts. Positive value
          will result in a clockwise rotation.
        * ``'spatial_pixel_scale'`` - Pixel scale (e.g., in arcsec/pix).

    The following are no longer used, so they are now optional:

        * ``'slit_pa'`` - Slit angle (e.g., in degrees).
        * ``'slit_width'`` - Slit width (e.g., in arcsec).
        * ``'slit_length'`` - Slit length (e.g., in arcsec).

    Cutouts are organized as follows::

        working_dir/
            <image_label>_cutouts/
                <id>_<image_label>_cutout.fits

    Each cutout image is a simple single-extension FITS with updated WCS.
    Its header has the following special keywords:

        * ``OBJ_RA`` - RA of the cutout object in degrees.
        * ``OBJ_DEC`` - DEC of the cutout object in degrees.
        * ``OBJ_ROT`` - Rotation of cutout object in degrees.

    Parameters
    ----------
    catalogname : str
        Catalog table defining the sources to cut out.

    imagename : str
        Image to cut.

    image_label : str
        Label to name the cutout sub-directory and filenames.

    apply_rotation : bool
        Cutout will be rotated to a given angle. Default is `False`.

    table_format : str, optional
        Format as accepted by `~astropy.table.QTable`. Default is ECSV.

    image_ext : int, optional
        Image extension to extract header and data. Default is 0.

    clobber : bool, optional
        Overwrite existing files. Default is `False`.

    verbose : bool, optional
        Print extra info. Default is `True`.

    """
    # Optional dependencies...
    from reproject import reproject_interp

    table = QTable.read(catalogname, format=table_format)

    with fits.open(imagename) as pf:
        data = pf[image_ext].data
        wcs = WCS(pf[image_ext].header)

    # It is more efficient to operate on an entire column at once.
    c = SkyCoord(table['ra'], table['dec'])
    x = (table['cutout_x_size'] / table['spatial_pixel_scale']).value  # pix
    y = (table['cutout_y_size'] / table['spatial_pixel_scale']).value  # pix
    pscl = table['spatial_pixel_scale'].to(u.deg / u.pix)

    # Do not rotate if column is missing.
    if 'cutout_pa' not in table.colnames:
        apply_rotation = False

    # Sub-directory, relative to working directory.
    path = '{0}_cutouts'.format(image_label)
    if not os.path.exists(path):
        os.mkdir(path)

    cutcls = partial(Cutout2D, data, wcs=wcs, mode='partial')

    for position, x_pix, y_pix, pix_scl, row in zip(c, x, y, pscl, table):

        if apply_rotation:
            pix_rot = row['cutout_pa'].to(u.degree).value

            cutout_wcs = WCS(naxis=2)
            cutout_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            cutout_wcs.wcs.crval = [position.ra.deg, position.dec.deg]
            cutout_wcs.wcs.crpix = [(x_pix - 1) * 0.5, (y_pix - 1) * 0.5]

            try:
                cutout_wcs.wcs.cd = wcs.wcs.cd
                cutout_wcs.rotateCD(-pix_rot)
            except AttributeError:
                cutout_wcs.wcs.cdelt = wcs.wcs.cdelt
                cutout_wcs.wcs.crota = [0, -pix_rot]

            cutout_hdr = cutout_wcs.to_header()

            try:
                cutout_arr = reproject_interp(
                    (data, wcs), cutout_hdr,
                    shape_out=(math.floor(y_pix + math.copysign(0.5, y_pix)),
                               math.floor(x_pix + math.copysign(0.5, x_pix))),
                    order=2)
            except Exception:
                if verbose:
                    log.info('reproject failed: '
                             'Skipping {0}'.format(row['id']))
                continue

            cutout_arr = cutout_arr[0]  # Ignore footprint
            cutout_hdr['OBJ_ROT'] = (pix_rot, 'Cutout rotation in degrees')

        else:
            try:
                cutout = cutcls(position, size=(y_pix, x_pix))
            except NoConvergence:
                if verbose:
                    log.info('WCS solution did not converge: '
                             'Skipping {0}'.format(row['id']))
                continue
            except NoOverlapError:
                if verbose:
                    log.info('Cutout is not on image: '
                             'Skipping {0}'.format(row['id']))
                continue
            else:
                cutout_hdr = cutout.wcs.to_header()
                cutout_arr = cutout.data

        if np.array_equiv(cutout_arr, 0):
            if verbose:
                log.info('No data in cutout: Skipping {0}'.format(row['id']))
            continue

        fname = os.path.join(
            path, '{0}_{1}_cutout.fits'.format(row['id'], image_label))

        # Construct FITS HDU.
        hdu = fits.PrimaryHDU(cutout_arr)
        hdu.header.update(cutout_hdr)
        hdu.header['OBJ_RA'] = (position.ra.deg, 'Cutout object RA in deg')
        hdu.header['OBJ_DEC'] = (position.dec.deg, 'Cutout object DEC in deg')

        hdu.writeto(fname, clobber=clobber)

        if verbose:
            log.info('Wrote {0}'.format(fname))


def show_cutout_with_slit(hdr, data=None, slit_ra=None, slit_dec=None,
                          slit_shape='rectangular', slit_width=0.2,
                          slit_length=3.3, slit_angle=90, slit_radius=0.2,
                          slit_rout=0.5, cmap='Greys_r', plotname='',
                          **kwargs):
    """Show a cutout image with the slit(s) superimposed.

    Parameters
    ----------
    hdr : dict
        Cutout image header.

    data : ndarray or `None`, optional
        Cutout image data. If not given, data is not shown.

    slit_ra, slit_dec : float or array or `None`, optional
        Slit RA and DEC in degrees. Default is to use object position
        from image header. If an array is given, each pair of RA and
        DEC becomes a slit.

    slit_shape : {'annulus', 'circular', 'rectangular'}, optional
        Shape of the slit (circular or rectangular).
        Default is rectangular.

    slit_width, slit_length : float, optional
        Rectangular slit width and length in arcseconds.
        Defaults are some fudge values.

    slit_angle : float, optional
        Rectangular slit angle in degrees for the display.
        Default is vertical.

    slit_radius : float, optional
        Radius of a circular or annulus slit in arcseconds.
        For annulus, this is the inner radius.
        Default is some fudge value.

    slit_rout : float, optional
        Outer radius of an annulus slit in arcseconds.
        Default is some fudge value.

    cmap : str or obj, optional
        Matplotlib color map for image display. Default is grayscale.

    plotname : str, optional
        Filename to save plot as. If not given, it is not saved.

    kwargs : dict, optional
        Keyword argument(s) for the aperture overlay.
        If ``ax`` is given, it will also be used for image display.

    See Also
    --------
    make_cutouts

    """
    # Optional dependencies...
    import matplotlib.pyplot as plt
    from photutils import (SkyCircularAnnulus, SkyCircularAperture,
                           SkyRectangularAperture)

    if slit_ra is None:
        slit_ra = hdr['OBJ_RA']
    if slit_dec is None:
        slit_dec = hdr['OBJ_DEC']

    position = SkyCoord(slit_ra, slit_dec, unit='deg')

    if slit_shape == 'circular':
        slit_radius = u.Quantity(slit_radius, u.arcsec)
        aper = SkyCircularAperture(position, slit_radius)

    elif slit_shape == 'annulus':
        slit_rin = u.Quantity(slit_radius, u.arcsec)
        slit_rout = u.Quantity(slit_rout, u.arcsec)
        aper = SkyCircularAnnulus(position, slit_rin, slit_rout)

    else:  # rectangular
        slit_width = u.Quantity(slit_width, u.arcsec)
        slit_length = u.Quantity(slit_length, u.arcsec)
        slit_angle = u.Quantity(slit_angle, u.degree)
        aper = SkyRectangularAperture(position, slit_width, slit_length,
                                      theta=slit_angle)

    wcs = WCS(hdr)
    aper_pix = aper.to_pixel(wcs)
    ax = kwargs.get('ax', plt)

    if data is not None:
        ax.imshow(data, cmap=cmap, origin='lower')

    aper_pix.plot(**kwargs)

    if plotname:
        ax.savefig(plotname)
