from astropy.io import fits
from astropy.table import Table
import numpy as np


# -----------------------------------------------------------------------------
# path values
# -----------------------------------------------------------------------------
p = dict()

# Directory of files
p['WORKSPACE'] = '/tell_hack/project/data_car/'
# Transmission file (TAPAS with molecular species in
p['TRANS_MODEL'] = p['WORKSPACE'] + 'tapas_all_sp.fits.gz'
# Master wavelength grid file to shift outputs to
p['MASTER_WAVE_FILE'] = p['WORKSPACE'] + 'car-20170109T06h16m29s-sci-gtoc-nir_A.fits'
# input spectrum
p['INPUT_DEFAULT'] = p['WORKSPACE'] + 'car-20170204T06h05m53s-sci-gtoc-nir_A.fits'

# -----------------------------------------------------------------------------
# Constant values
# -----------------------------------------------------------------------------

# value below which the blaze in considered too low to be useful
#     for all blaze profiles, we normalize to the 95th percentile.
#     That's pretty much the peak value, but it is resistent to
#     eventual outliers
p['MKTELLU_CUT_BLAZE_NORM'] = 0.1
p['MKTELLU_BLAZE_PERCENTILE'] = 95

# Define mean line width expressed in pix
p['FWHM_PIXEL_LSF'] = 2.1

# threshold in absorbance where we will stop iterating the absorption
#     model fit
p['MKTELLU_DPARAM_THRES'] = 0.001

# max number of iterations, normally converges in about 12 iterations
p['MKTELLU_MAX_ITER'] = 50

# minimum transmission required for use of a given pixel in the TAPAS
#    and SED fitting
p['MKTELLU_THRES_TRANSFIT'] = 0.3

# Defines the bad pixels if the spectrum is larger than this value.
#    These values are likely an OH line or a cosmic ray
p['MKTELLU_TRANS_FIT_UPPER_BAD'] = 1.1

# Defines the minimum allowed value for the recovered water vapor optical
#    depth (should not be able 1)
p['MKTELLU_TRANS_MIN_WATERCOL'] = 0.2

# Defines the maximum allowed value for the recovered water vapor optical
#    depth
p['MKTELLU_TRANS_MAX_WATERCOL'] = 99

# Defines the minimum number of good points required to normalise the
#    spectrum, if less than this we don't normalise the spectrum by its
#    median
p['MKTELLU_TRANS_MIN_NUM_GOOD'] = 100

# Defines the percentile used to gauge which transmission points should
#    be used to median (above this percentile is used to median)
p['MKTELLU_TRANS_TAU_PERCENTILE'] = 95

# sigma-clipping of the residuals of the difference between the
# spectrum divided by the fitted TAPAS absorption and the
# best guess of the SED
p['MKTELLU_TRANS_SIGMA_CLIP'] = 20.0

# median-filter the trans data measured in pixels
p['MKTELLU_TRANS_TEMPLATE_MEDFILT'] = 31

# Define the threshold for "small" values that do not add to the weighting
p['MKTELLU_SMALL_WEIGHTING_ERROR'] = 0.01

# Define the median sampling expressed in km/s / pix
p['MKTELLU_MED_SAMPLING'] = 2.2

# Define the orders to plot (not too many) - but can put 'all' to show all
#    'all' are shown one-by-one and then closed (in non-interactive mode)
p['MKTELLU_PLOT_ORDER_NUMS'] = [14, 18, 22]

# Set an upper limit for the allowed line-of-sight optical depth of water
p['MKTELLU_TAU_WATER_ULIMIT'] = 99

# set a lower and upper limit for the allowed line-of-sight optical depth
#    for other absorbers (upper limit equivalent to airmass limit)
# line-of-sight optical depth for other absorbers cannot be less than one
#      (that's zenith) keep the limit at 0.2 just so that the value gets
#      propagated to header and leaves open the possibility that during
#      the convergence of the algorithm, values go slightly below 1.0
p['MKTELLU_TAU_OTHER_LLIMIT'] = 0.2
# line-of-sight optical depth for other absorbers cannot be greater than 5
# ... that would be an airmass of 5 and SPIRou cannot observe there
p['MKTELLU_TAU_OTHER_ULIMIT'] = 5.0

# bad values and small values are set to this value (as a lower limit to
#   avoid dividing by small numbers or zero
p['MKTELLU_SMALL_LIMIT'] = 1.0e-9

# define the default convolution width [in pixels]
p['MKTELLU_DEFAULT_CONV_WIDTH'] = 900
# define the finer convolution width [in pixels]
p['MKTELLU_FINER_CONV_WIDTH'] = 100

# Selected plot orderbetween 0 and 27 or 'all'
p['TELLU_FIT_RECON_PLT_ORDER'] = 'all'

# number of principle components to use
p['TELLU_NUMBER_OF_PRINCIPLE_COMP'] = 5

# Define the median sampling expressed in km/s / pix
p['TELLU_MED_SAMPLING'] = 2.2

# Wavelength bounds of instrument
p['TELLU_LAMBDA_MIN'] = 1000.0
p['TELLU_LAMBDA_MAX'] = 2100.0

# Minimum tranmission to fit
p['TELLU_FIT_MIN_TRANSMISSION'] = 0.2

# Gaussian kernal size km/s
p['TELLU_FIT_VSINI'] = 15.0
p['TELLU_FIT_VSINI2'] = 30.0

# Number of iterations to run (SPIROU converges in less than 3 to 4)
p['TELLU_FIT_NITER'] = 4

# Define min transmission in tapas models to consider an
#     element part of continuum
p['TRANSMISSION_CUT'] = 0.98

# Smallest log reconstructed absorption to keep?
p['TELLU_FIT_LOG_LIMIT'] = -0.5

# Define the allowed difference between recovered and input airmass
p['QC_MKTELLU_AIRMASS_DIFF'] = 0.1



# -----------------------------------------------------------------------------

# Instrument specific functions

# -----------------------------------------------------------------------------
def begin():
    return p


def get_mk_tellu_data(p, loc, input_filename=None):
    if input_filename is None:
        input_filename = p['INPUT_DEFAULT']
    # ----------------------------------------------------------------------
    # get master wave grid
    hdu = fits.open(p['MASTER_WAVE_FILE'])
    mwave = hdu[4].data
    # ----------------------------------------------------------------------
    # get the telluric spectrum
    hdu = fits.open(input_filename)
    sflux = hdu[1].data
    swave = hdu[4].data
    # ----------------------------------------------------------------------
    # get data from the header
    airmass = hdu[0].header['AIRMASS']

    # ----------------------------------------------------------------------
    # load the transmission model
    tapas = Table.read(p['TRANS_MODEL'])

    # ----------------------------------------------------------------------
    p['INPUT_SPEC'] = input_filename
    # wave solution in nm
    loc['MWAVE'] = mwave / 10.0
    loc['SFLUX'] = sflux
    loc['BLAZE'] = np.ones_like(sflux)
    # wave solution in nm
    loc['SWAVE'] = swave / 10.0

    loc['TAPAS'] = tapas
    loc['AIRMASS'] = airmass

    return p, loc


def get_fit_tellu_data(p, loc, input_filename=None):
    if input_filename is None:
        input_filename = p['INPUT_DEFAULT']
    # ----------------------------------------------------------------------
    # get master wave grid
    hdu = fits.open(p['MASTER_WAVE_FILE'])
    mwave = hdu[4].data
    # ----------------------------------------------------------------------
    # get the telluric spectrum
    hdu = fits.open(input_filename)
    sflux = hdu[1].data
    swave = hdu[4].data
    # ----------------------------------------------------------------------
    # get data from the header
    airmass = hdu[0].header['AIRMASS']

    # ----------------------------------------------------------------------
    # load the transmission model
    tapas = Table.read(p['TRANS_MODEL'])

    # ----------------------------------------------------------------------
    p['INPUT_SPEC'] = input_filename
    # wave solution in nm
    loc['MWAVE'] = mwave / 10.0
    loc['SFLUX'] = sflux
    loc['BLAZE'] = np.ones_like(sflux)
    # wave solution in nm
    loc['SWAVE'] = swave / 10.0

    loc['TAPAS'] = tapas
    loc['AIRMASS'] = airmass

    return p, loc