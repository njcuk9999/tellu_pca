from astropy.time import Time
from scipy.interpolate import InterpolatedUnivariateSpline as IUVSpline
import numpy as np
import sys


# Define list of absorbers in the tapas fits table
TELLU_ABSORBERS = ['combined', 'h2o', 'o3', 'n2o', 'o2', 'co2', 'ch4']


def WLOG(p, level, message):
    """
    Logging function (simple version of SPIRou DRS version)
    :param p:
    :param level:
    :param message:
    :return:
    """
    if type(message) is str:
        message = [message]

    timenow = Time.now()
    for mess in message:
        print('{0}: {1}: {2}'.format(timenow, level, mess))

    if level.lower() == 'error':
        sys.exit()


def fwhm():
    """
    The true value of 2 sqrt(ln(2) - at least to float point accuracy
    from SPIRou DRS version
    :return:
    """
    return 2 * np.sqrt(2 * np.log(2))



def kernal_with_instrument(lsf):
    # get the number of pixels
    npix = int(np.ceil(3 * lsf * 3.0 / 2) * 2 + 1)
    # set up the kernal x values
    xpix = np.arange(npix) - npix // 2
    # get the gaussian kernel
    ker = np.exp(-0.5 * (xpix / (lsf / fwhm())) ** 2)
    # we only want an approximation of the absorption to find the continuum
    #    and estimate chemical abundances.
    #    there's no need for a varying kernel shape
    ker /= np.sum(ker)
    # return kernel
    return ker



def resample_tapas(p, loc, mwave, npix, nord, tapas, kernel):
    """
    Resample TAPAS (from SPIRou DRS version)
    :param p:
    :param loc:
    :param mwave:
    :param npix:
    :param nord:
    :param tapas:
    :param kernel:
    :return:
    """

    tapas_all_species = np.zeros([len(TELLU_ABSORBERS), npix * nord])

    for n_species, molecule in enumerate(TELLU_ABSORBERS):
        # log process
        wmsg = 'Processing molecule: {0}'
        WLOG(p, '', wmsg.format(molecule))
        # get wavelengths
        lam = tapas['wavelength']
        # get molecule transmission
        trans = tapas['trans_{0}'.format(molecule)]
        # interpolate with Univariate Spline
        tapas_spline = IUVSpline(lam, trans)
        # log the mean transmission level
        wmsg = '\tMean Trans level: {0:.3f}'.format(np.mean(trans))
        WLOG(p, '', wmsg)
        # convolve all tapas absorption to the SPIRou approximate resolution
        for order_num in range(nord):
            # get the order position
            start = order_num * npix
            end = (order_num * npix) + npix
            # interpolate the values at these points
            svalues = tapas_spline(mwave[order_num])
            # convolve with a gaussian function
            cvalues = np.convolve(svalues, kernel, mode='same')
            # add to storage
            tapas_all_species[n_species, start: end] = cvalues

    # deal with non-real values (must be between 0 and 1
    tapas_all_species[tapas_all_species > 1] = 1
    tapas_all_species[tapas_all_species < 0] = 0

    # extract the water and other line-of-sight optical depths
    loc['TAPAS_ALL'] = tapas_all_species
    loc['TAPAS_WATER'] = tapas_all_species[1, :]
    loc['TAPAS_OTHERS'] = np.prod(tapas_all_species[2:, :], axis=0)

    return loc