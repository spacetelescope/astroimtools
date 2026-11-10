# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os

import astropy.units as u
import numpy as np
import pytest
from astropy import log
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import QTable
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from ..cutout_tools import make_cutouts, show_cutout_with_slit

PIXEL_SCALE = 0.1  # arcsec / pixel
IMAGE_SHAPE = (201, 201)

# (x, y) pixel positions of the catalog sources. src3 is off the image
# and src4 lies in a region of the image where all values are zero.
SOURCES = {'src1': (60.0, 80.0),
           'src2': (130.0, 110.0),
           'src3': (400.0, 100.0),
           'src4': (175.0, 175.0)}
CUTOUT_X_SIZE = 2.0  # arcsec
CUTOUT_Y_SIZE = 3.0  # arcsec
CUTOUT_SHAPE = (round(CUTOUT_Y_SIZE / PIXEL_SCALE),
                round(CUTOUT_X_SIZE / PIXEL_SCALE))


def make_wcs_header(wcs_keywords='cdelt'):
    """
    Make a north-up image WCS header.

    Parameters
    ----------
    wcs_keywords : {'cdelt', 'cd', 'pc'}, optional
        The keywords that define the pixel scale. 'cdelt' uses only
        CDELT, 'cd' uses CD, and 'pc' uses PC with unit CDELT, which is
        how astropy writes a WCS defined with a CD matrix.

    Returns
    -------
    header : `~astropy.io.fits.Header`
        The WCS header.
    """
    scale = PIXEL_SCALE / 3600.0
    header = fits.Header()
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CRVAL1'] = 150.0
    header['CRVAL2'] = 2.0
    header['CRPIX1'] = 101.0
    header['CRPIX2'] = 101.0
    if wcs_keywords == 'cdelt':
        header['CDELT1'] = -scale
        header['CDELT2'] = scale
    else:
        prefix = 'CD' if wcs_keywords == 'cd' else 'PC'
        header[f'{prefix}1_1'] = -scale
        header[f'{prefix}1_2'] = 0.0
        header[f'{prefix}2_1'] = 0.0
        header[f'{prefix}2_2'] = scale
    return header


def make_wcs():
    return WCS(make_wcs_header())


def make_image(header):
    rng = np.random.default_rng(0)
    data = rng.normal(10.0, 0.1, IMAGE_SHAPE)
    yy, xx = np.mgrid[0:IMAGE_SHAPE[0], 0:IMAGE_SHAPE[1]]
    for name in ('src1', 'src2'):
        x0, y0 = SOURCES[name]
        data += 100.0 * np.exp(-((xx - x0)**2 + (yy - y0)**2) / 8.0)
    data[150:, 150:] = 0.0
    return fits.PrimaryHDU(data, header=header)


def make_catalog(wcs):
    xpos, ypos = np.transpose(list(SOURCES.values()))
    coords = wcs.pixel_to_world(xpos, ypos)
    n_sources = len(SOURCES)
    table = QTable()
    table['id'] = list(SOURCES.keys())
    table['ra'] = coords.ra.to(u.deg)
    table['dec'] = coords.dec.to(u.deg)
    table['cutout_x_size'] = np.full(n_sources, CUTOUT_X_SIZE) * u.arcsec
    table['cutout_y_size'] = np.full(n_sources, CUTOUT_Y_SIZE) * u.arcsec
    table['cutout_pa'] = np.full(n_sources, 30.0) * u.deg
    table['spatial_pixel_scale'] = (np.full(n_sources, PIXEL_SCALE)
                                    * u.arcsec / u.pixel)
    return table


@pytest.fixture(params=['cdelt', 'cd', 'pc'])
def cutout_inputs(request, tmp_path, monkeypatch):
    # Write a synthetic image and catalog and change to their directory
    header = make_wcs_header(request.param)
    imagename = tmp_path / 'image.fits'
    catalogname = tmp_path / 'catalog.ecsv'
    make_image(header).writeto(imagename)
    make_catalog(WCS(header)).write(catalogname, format='ascii.ecsv')
    monkeypatch.chdir(tmp_path)
    return str(catalogname), str(imagename)


def cutout_filename(label, source_id):
    return os.path.join(f'{label}_cutouts',
                        f'{source_id}_{label}_cutout.fits')


def assert_centered_on_source(filename, source_id):
    with fits.open(filename) as hdulist:
        header = hdulist[0].header
        data = hdulist[0].data

    assert data.shape == CUTOUT_SHAPE

    wcs = make_wcs()
    expected = wcs.pixel_to_world(*SOURCES[source_id])
    assert_allclose(header['OBJ_RA'], expected.ra.deg)
    assert_allclose(header['OBJ_DEC'], expected.dec.deg)

    # The source peak must be at the sky position given in the catalog
    ypeak, xpeak = np.unravel_index(np.nanargmax(data), data.shape)
    peak = WCS(header).pixel_to_world(xpeak, ypeak)
    assert peak.separation(expected) < 1.5 * PIXEL_SCALE * u.arcsec


class TestMakeCutouts:
    @pytest.fixture(autouse=True)
    def requires_reproject(self):
        pytest.importorskip('reproject')

    def test_cutouts(self, cutout_inputs):
        catalogname, imagename = cutout_inputs
        with log.log_to_list() as log_list:
            make_cutouts(catalogname, imagename, 'plain')

        for source_id in ('src1', 'src2'):
            filename = cutout_filename('plain', source_id)
            assert_centered_on_source(filename, source_id)
            assert 'OBJ_ROT' not in fits.getheader(filename)

        assert not os.path.exists(cutout_filename('plain', 'src3'))
        assert not os.path.exists(cutout_filename('plain', 'src4'))

        messages = [record.message for record in log_list]
        assert 'Cutout is not on image: Skipping src3' in messages
        assert 'No data in cutout: Skipping src4' in messages

    def test_cutouts_rotated(self, cutout_inputs):
        catalogname, imagename = cutout_inputs
        make_cutouts(catalogname, imagename, 'rotated', apply_rotation=True,
                     verbose=False)

        for source_id in ('src1', 'src2'):
            filename = cutout_filename('rotated', source_id)
            assert_centered_on_source(filename, source_id)
            header = fits.getheader(filename)
            assert header['OBJ_ROT'] == 30.0

            # The cutout north axis must be rotated by the cutout angle
            # with respect to the image north axis
            wcs = WCS(header)
            center = wcs.pixel_to_world(10, 15)
            north = wcs.pixel_to_world(10, 16)
            assert_allclose(center.position_angle(north).to_value(u.deg),
                            30.0, atol=1.0e-3)

            # The pixel scale must be preserved
            scale = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))
            assert_allclose(scale * 3600.0, PIXEL_SCALE)

    def test_no_cutout_pa_column(self, cutout_inputs):
        catalogname, imagename = cutout_inputs
        table = QTable.read(catalogname)
        table.remove_column('cutout_pa')
        table.write(catalogname, format='ascii.ecsv', overwrite=True)

        # The rotation is ignored if the cutout angle is not available
        make_cutouts(catalogname, imagename, 'nopa', apply_rotation=True,
                     verbose=False)
        filename = cutout_filename('nopa', 'src1')
        assert_centered_on_source(filename, 'src1')
        assert 'OBJ_ROT' not in fits.getheader(filename)

    def test_overwrite(self, cutout_inputs):
        catalogname, imagename = cutout_inputs
        make_cutouts(catalogname, imagename, 'plain', verbose=False)
        with pytest.raises(OSError, match='already exists'):
            make_cutouts(catalogname, imagename, 'plain', verbose=False)
        make_cutouts(catalogname, imagename, 'plain', overwrite=True,
                     verbose=False)


class TestShowCutoutWithSlit:
    @pytest.fixture(autouse=True)
    def plt(self):
        pytest.importorskip('photutils')
        matplotlib = pytest.importorskip('matplotlib')
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        yield plt
        plt.close('all')

    @pytest.fixture
    def cutout(self):
        wcs = make_wcs()
        position = wcs.pixel_to_world(*SOURCES['src1'])
        header = wcs.to_header()
        header['OBJ_RA'] = position.ra.deg
        header['OBJ_DEC'] = position.dec.deg
        rng = np.random.default_rng(0)
        data = rng.normal(10.0, 0.1, (30, 20))
        return header, data

    @pytest.mark.parametrize(('slit_shape', 'patch_type'),
                             [('rectangular', 'Rectangle'),
                              ('circular', 'Circle'),
                              ('annulus', 'PathPatch')])
    def test_slit_shapes(self, plt, cutout, slit_shape, patch_type):
        header, data = cutout
        show_cutout_with_slit(header, data=data, slit_shape=slit_shape,
                              color='red')
        ax = plt.gca()
        assert len(ax.images) == 1
        assert [type(patch).__name__ for patch in ax.patches] == [patch_type]

    def test_slit_position(self, plt, cutout):
        header, _ = cutout
        slit = SkyCoord(header['OBJ_RA'], header['OBJ_DEC'], unit='deg')
        show_cutout_with_slit(header, slit_ra=slit.ra.deg + 1.0e-4,
                              slit_dec=slit.dec.deg, slit_angle=45)
        ax = plt.gca()

        # No image is shown if the data are not input
        assert len(ax.images) == 0
        assert len(ax.patches) == 1

        # The slit angle is measured from north, which is the image y axis
        assert_allclose(ax.patches[0].get_angle(), 45.0 + 90.0, atol=1.0e-3)

    def test_ax(self, plt, cutout):
        header, data = cutout
        _, (ax1, ax2) = plt.subplots(1, 2)
        show_cutout_with_slit(header, data=data, ax=ax2)
        assert len(ax1.images) == 0
        assert len(ax2.images) == 1
        assert len(ax2.patches) == 1

    def test_plotname(self, cutout, tmp_path):
        header, data = cutout
        plotname = tmp_path / 'slit.png'
        show_cutout_with_slit(header, data=data, plotname=plotname)
        assert plotname.stat().st_size > 0
