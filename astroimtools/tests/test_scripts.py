# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pytest
from astropy.io import fits
from numpy.testing import assert_allclose

from ..scripts import imarith, imstats


@pytest.fixture
def fits_files(tmp_path):
    filenames = []
    for i, exptime in enumerate((100.0, 50.0)):
        filename = tmp_path / f'image{i}.fits'
        data = np.arange(20.0).reshape(4, 5) * (i + 1)
        hdulist = fits.HDUList([fits.PrimaryHDU(),
                                fits.ImageHDU(data, name='SCI')])
        hdulist[0].header['EXPTIME'] = exptime
        hdulist.writeto(filename)
        filenames.append(str(filename))
    return filenames


def test_imstats(fits_files, capsys):
    imstats.main([*fits_files, '-e', '1'])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ['filename', 'npixels', 'mean', 'std', 'min',
                                'max']
    assert len(lines) == 4
    for line, filename, scale in zip(lines[2:], fits_files, (1, 2)):
        values = line.split()
        assert values[0] == filename
        assert int(values[1]) == 20
        assert_allclose(float(values[2]), 9.5 * scale)
        assert_allclose(float(values[4]), 0.0)
        assert_allclose(float(values[5]), 19.0 * scale)


def test_imstats_columns(fits_files, capsys):
    imstats.main([fits_files[0], '-e', '1', '-c', 'median, max', '-u', '10'])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ['filename', 'median', 'max']
    assert_allclose([float(value) for value in lines[2].split()[1:]],
                    [5.0, 10.0])


def test_imarith(fits_files, tmp_path):
    outfilename = tmp_path / 'result.fits'
    imarith.main([*fits_files, '+', '-e1', '1', '-e2', '1', '-k', 'exptime',
                  '-o', str(outfilename)])
    with fits.open(outfilename) as hdulist:
        assert_allclose(hdulist[1].data,
                        np.arange(20.0).reshape(4, 5) * 3)
        assert hdulist[0].header['EXPTIME'] == 150.0

    # The output file is not overwritten by default
    with pytest.raises(OSError, match='already exists'):
        imarith.main([*fits_files, '+', '-e1', '1', '-e2', '1', '-o',
                      str(outfilename)])
    imarith.main([*fits_files, '-', '-e1', '1', '-e2', '1', '-o',
                  str(outfilename), '--overwrite'])
    assert_allclose(fits.getdata(outfilename, 1),
                    -np.arange(20.0).reshape(4, 5))


def test_imarith_scalar(fits_files, tmp_path):
    outfilename = tmp_path / 'result.fits'
    imarith.main(['2', fits_files[0], '*', '-e2', '1', '-o',
                  str(outfilename)])
    assert_allclose(fits.getdata(outfilename, 1),
                    np.arange(20.0).reshape(4, 5) * 2)


def test_imarith_two_scalars():
    with pytest.raises(ValueError, match='cannot be scalars'):
        imarith.main(['1', '2', '+'])
