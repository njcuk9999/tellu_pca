from astropy.io import fits
import numpy as np


# -----------------------------------------------------------------------------
# path values
# -----------------------------------------------------------------------------
# Directory of files
WORKSPACE = '/tell_hack/project/data_car/'
# Transmission file (TAPAS with molecular species in
TRANS_MODEL = WORKSPACE + 'tapas_all_sp.fits.gz'
# Master wavelength grid file to shift outputs to
MASTER_WAVE_FILE = WORKSPACE + 'car-20170109T06h16m29s-sci-gtoc-nir_A.fits'
# input spectrum
INPUT_TSS = WORKSPACE + 'car-20170109T06h16m29s-sci-gtoc-nir_A.fits'

# -----------------------------------------------------------------------------
# Constant values
# -----------------------------------------------------------------------------
# value below which the blaze in considered too low to be useful
#     for all blaze profiles, we normalize to the 95th percentile.
#     That's pretty much the peak value, but it is resistent to
#     eventual outliers
MKTELLU_CUT_BLAZE_NORM = 0.1
MKTELLU_BLAZE_PERCENTILE = 95


# Define mean line width expressed in pix
# TODO: SPIROU default = 2.1
FWHM_PIXEL_LSF = 2.1

# threshold in absorbance where we will stop iterating the absorption
#     model fit
MKTELLU_DPARAM_THRES = 0.001

# max number of iterations, normally converges in about 12 iterations
MKTELLU_MAX_ITER = 50

# minimum transmission required for use of a given pixel in the TAPAS
#    and SED fitting
MKTELLU_THRES_TRANSFIT = 0.3

# Defines the bad pixels if the spectrum is larger than this value.
#    These values are likely an OH line or a cosmic ray
MKTELLU_TRANS_FIT_UPPER_BAD = 1.1

# Defines the minimum allowed value for the recovered water vapor optical
#    depth (should not be able 1)
MKTELLU_TRANS_MIN_WATERCOL = 0.2

# Defines the maximum allowed value for the recovered water vapor optical
#    depth
MKTELLU_TRANS_MAX_WATERCOL = 99

# Defines the minimum number of good points required to normalise the
#    spectrum, if less than this we don't normalise the spectrum by its
#    median
MKTELLU_TRANS_MIN_NUM_GOOD = 100

# Defines the percentile used to gauge which transmission points should
#    be used to median (above this percentile is used to median)
MKTELLU_TRANS_TAU_PERCENTILE = 95

# sigma-clipping of the residuals of the difference between the
# spectrum divided by the fitted TAPAS absorption and the
# best guess of the SED
MKTELLU_TRANS_SIGMA_CLIP = 20.0

# median-filter the trans data measured in pixels
MKTELLU_TRANS_TEMPLATE_MEDFILT = 31

# Define the threshold for "small" values that do not add to the weighting
MKTELLU_SMALL_WEIGHTING_ERROR = 0.01

# Define the median sampling expressed in km/s / pix
# TODO: MED_SAMPLING = 2.2
MKTELLU_MED_SAMPLING = 2.2

# Define the orders to plot (not too many) - but can put 'all' to show all
#    'all' are shown one-by-one and then closed (in non-interactive mode)
MKTELLU_PLOT_ORDER_NUMS = [12, 14, 16]

# Set an upper limit for the allowed line-of-sight optical depth of water
MKTELLU_TAU_WATER_ULIMIT = 99

# set a lower and upper limit for the allowed line-of-sight optical depth
#    for other absorbers (upper limit equivalent to airmass limit)
# line-of-sight optical depth for other absorbers cannot be less than one
#      (that's zenith) keep the limit at 0.2 just so that the value gets
#      propagated to header and leaves open the possibility that during
#      the convergence of the algorithm, values go slightly below 1.0
MKTELLU_TAU_OTHER_LLIMIT = 0.2
# line-of-sight optical depth for other absorbers cannot be greater than 5
# ... that would be an airmass of 5 and SPIRou cannot observe there
MKTELLU_TAU_OTHER_ULIMIT = 5.0

# bad values and small values are set to this value (as a lower limit to
#   avoid dividing by small numbers or zero
MKTELLU_SMALL_LIMIT = 1.0e-9

# define the default convolution width [in pixels]
# TODO: SPIROU DEFAULT = 900
MKTELLU_DEFAULT_CONV_WIDTH = 900
# define the finer convolution width [in pixels]
MKTELLU_FINER_CONV_WIDTH = 100

# Define list of absorbers in the tapas fits table
TELLU_ABSORBERS = ['combined', 'h2o', 'o3', 'n2o', 'o2', 'co2', 'ch4']

# Selected plot order
TELLU_FIT_RECON_PLT_ORDER = 33

# number of principle components to use
TELLU_NUMBER_OF_PRINCIPLE_COMP = 5

# Define the median sampling expressed in km/s / pix
TELLU_MED_SAMPLING = 2.2

# Wavelength bounds of instrument
TELLU_LAMBDA_MIN = 1000.0
TELLU_LAMBDA_MAX = 2100.0

# Minimum tranmission to fit
TELLU_FIT_MIN_TRANSMISSION = 0.2

# Gaussian kernal size km/s
TELLU_FIT_VSINI = 15.0
TELLU_FIT_VSINI2 = 30.0

# Number of iterations to run (SPIROU converges in less than 3 to 4)
TELLU_FIT_NITER = 4

# Define min transmission in tapas models to consider an
#     element part of continuum
TRANSMISSION_CUT = 0.98

# Smallest log reconstructed absorption to keep?
TELLU_FIT_LOG_LIMIT = -0.5


# -----------------------------------------------------------------------------

# Instrument specific functions
# -----------------------------------------------------------------------------
def get_mk_tellu_data(p, loc):
    # ----------------------------------------------------------------------
    # get master wave grid
    hdu = fits.open(MASTER_WAVE_FILE)
    mwave = hdu[4].data
    # ----------------------------------------------------------------------
    # get the telluric spectrum
    hdu = fits.open(INPUT_TSS)
    sflux = hdu[1].data
    swave = hdu[4].data
    # ----------------------------------------------------------------------
    # get data from the header
    airmass = hdu[0].header['AIRMASS']


    # wave solution in nm
    loc['MWAVE'] = mwave / 10.0
    loc['SFLUX'] = sflux
    loc['BLAZE'] = np.ones_like(sflux)
    # wave solution in nm
    loc['SWAVE'] = swave / 10.0
    loc['AIRMASS'] = airmass

    return loc


def get_fit_tellu_data(p, loc):
    # ----------------------------------------------------------------------
    # get master wave grid
    hdu = fits.open(MASTER_WAVE_FILE)
    mwave = hdu[4].data
    # ----------------------------------------------------------------------
    # get the telluric spectrum
    hdu = fits.open(INPUT_TSS)
    sflux = hdu[1].data
    swave = hdu[4].data
    # ----------------------------------------------------------------------
    # get data from the header
    airmass = hdu[0].header['AIRMASS']

    # wave solution in nm
    loc['MWAVE'] = mwave / 10.0
    loc['SFLUX'] = sflux
    loc['BLAZE'] = np.ones_like(sflux)
    # wave solution in nm
    loc['SWAVE'] = swave / 10.0
    loc['AIRMASS'] = airmass

    return loc