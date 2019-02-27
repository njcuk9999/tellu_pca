#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# CODE DESCRIPTION HERE

Created on 2019-02-26 13:29
@author: ncook
Version 0.0.1
"""
from shared_functions import *

from astropy.io import fits
import numpy as np
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline as IUVSpline
from scipy.optimize import curve_fit
from scipy.ndimage import filters
import warnings
import matplotlib.pyplot as plt




# =============================================================================
# Define variables
# =============================================================================
# plotting options
DEBUG_PLOT = True
PLOT = True

# -----------------------------------------------------------------------------
# define instrument name
INSTRUMENT = 'CARMENES'

# get instrument setup
if INSTRUMENT == 'CARMENES':
    from setup_car import *
if INSTRUMENT == 'SPIROU':
    from setup_spirou import *
# -----------------------------------------------------------------------------
# output directory
OUTPUT_DIR = WORKSPACE + '/trans/'
OUT_SUFFIX = '_trans.fits'


# =============================================================================
# Define functions
# =============================================================================
def mk_tellu_wave_flux_plot(p, order_num, wave, tau1, sp, sp3, sed,
                            sed_update, keep):
    plot_name = 'mk_tellu_wave_flux_plot_order_{0}'.format(order_num)
    # get order values
    good = keep[order_num]
    x = wave[order_num]
    y1 = tau1[order_num]
    y2 = sp[order_num]
    y3 = sp[order_num] / sed[order_num]
    y4 = sp3
    y5 = sed_update

    # deal with no good values
    if np.sum(good) == 0:
        y4 = np.repeat(np.nan, len(x))
        good = np.ones(len(x), dtype=bool)

    # set up fig
    fig, frame = plt.subplots(ncols=1, nrows=1)
    # plot data
    frame.plot(x, y1, color='c', label='tapas fit')
    frame.plot(x, y2, color='k', label='input spectrum')
    frame.plot(x, y3, color='b', label='measured transmission')

    frame.plot(x[good], y4[good], color='r', marker='.', linestyle='None',
               label='SED calculation value')
    frame.plot(x, y5, color='g', linestyle='--', label='SED best guess')

    # get max / min y
    values = list(y1) + list(y2) + list(y3) + list(y4[good]) + list(y5)
    mins = 0.95 * np.nanmin([0, np.nanmin(values)])
    maxs = 1.05 * np.nanmax(values)

    # plot legend and set up labels / limits / title
    frame.legend(loc=0)
    frame.set(xlim=(np.min(x[good]), np.max(x[good])),
              ylim=(mins, maxs),
              xlabel='Wavelength [nm]', ylabel='Normalised flux',
              title='Order: {0}'.format(order_num))
    # end plotting function properly
    plt.show()
    plt.close()


def calculate_telluric_absorption(p, loc):
    func_name = 'calculate_telluric_absorption()'
    # get parameters from p
    dparam_threshold = MKTELLU_DPARAM_THRES
    maximum_iteration = MKTELLU_MAX_ITER
    threshold_transmission_fit = MKTELLU_THRES_TRANSFIT
    transfit_upper_bad = MKTELLU_TRANS_FIT_UPPER_BAD
    min_watercol = MKTELLU_TRANS_MIN_WATERCOL
    max_watercol = MKTELLU_TRANS_MAX_WATERCOL
    min_number_good_points = MKTELLU_TRANS_MIN_NUM_GOOD
    btrans_percentile = MKTELLU_TRANS_TAU_PERCENTILE
    nsigclip = MKTELLU_TRANS_SIGMA_CLIP
    med_filt = MKTELLU_TRANS_TEMPLATE_MEDFILT
    small_weight = MKTELLU_SMALL_WEIGHTING_ERROR
    tellu_med_sampling = MKTELLU_MED_SAMPLING
    plot_order_nums = MKTELLU_PLOT_ORDER_NUMS

    # get data from loc
    airmass = loc['AIRMASS']
    wave = loc['WAVE']
    sp = np.array(loc['SP'])
    wconv = loc['WCONV']
    # get dimensions of data
    norders, npix = loc['SP'].shape

    # define function for curve_fit
    def tapas_fit(keep, tau_water, tau_others):
        return calc_tapas_abso(p, loc, keep, tau_water, tau_others)

    # starting point for the optical depth of water and other gasses
    guess = [airmass, airmass]

    # first estimate of the absorption spectrum
    tau1 = tapas_fit(np.isfinite(wave), guess[0], guess[1])
    tau1 = tau1.reshape(sp.shape)

    # first guess at the SED estimate for the hot start (we guess with a
    #   spectrum full of ones
    sed = np.ones_like(wave)

    # flag to see if we converged (starts off very large)
    # this is the quadratic sum of the change in airmass and water column
    # when the change in the sum of these two values is very small between
    # two steps, we assume that the code has converges
    dparam = np.inf

    # count the number of iterations
    iteration = 0

    # if the code goes out-of-bound, then we'll get out of the loop with this
    #    keyword
    fail = False
    skip = False

    # conditions to carry on looping
    cond1 = dparam > dparam_threshold
    cond2 = iteration < maximum_iteration
    cond3 = not fail

    # set up empty arrays
    sp3_arr = np.zeros((norders, npix), dtype=float)
    sed_update_arr = np.zeros(npix, dtype=float)
    keep = np.zeros(npix, dtype=bool)

    # loop around until one condition not met
    while cond1 and cond2 and cond3:
        # ---------------------------------------------------------------------
        # previous guess
        prev_guess = np.array(guess)
        # ---------------------------------------------------------------------
        # we have an estimate of the absorption spectrum
        fit_sp = sp / sed
        # ---------------------------------------------------------------------
        # some masking of NaN regions
        nanmask = ~np.isfinite(fit_sp)
        fit_sp[nanmask] = 0
        # ---------------------------------------------------------------------
        # vector used to mask invalid regions
        keep = fit_sp != 0
        # only fit where the transmission is greater than a certain value
        keep &= tau1 > threshold_transmission_fit
        # considered bad if the spectrum is larger than '. This is
        #     likely an OH line or a cosmic ray
        keep &= fit_sp < transfit_upper_bad
        # ---------------------------------------------------------------------
        # fit telluric absorption of the spectrum
        with warnings.catch_warnings(record=True) as _:
            popt, pcov = curve_fit(tapas_fit, keep, fit_sp.ravel(), p0=guess)
        # update our guess
        guess = np.array(popt)
        # ---------------------------------------------------------------------
        # if our tau_water guess is bad fail
        if (guess[0] < min_watercol) or (guess[0] > max_watercol):
            wmsg = ('Recovered water vapor optical depth not between {0:.2f} '
                    'and {1:.2f}')
            WLOG(p, 'warning', wmsg.format(min_watercol, max_watercol))
            fail = True
            break
        # ---------------------------------------------------------------------
        # we will use a stricter condition later, but for all steps
        #    we expect an agreement within an airmass difference of 1
        if np.abs(guess[1] - airmass) > 1:
            wmsg = ('Recovered optical depth of others too diffferent from '
                    'airmass (airmass={0:.3f} recovered depth={1:.3f})')
            WLOG(p, 'warning', wmsg.format(airmass, guess[1]))
            fail = True
            break
        # ---------------------------------------------------------------------
        # calculate how much the optical depth params change
        dparam = np.sqrt(np.sum((guess - prev_guess) ** 2))
        # ---------------------------------------------------------------------
        # print progress
        wmsg = ('Iteration {0}/{1} H20 depth: {2:.4f} Other gases depth: '
                '{3:.4f} Airmass: {4:.4f}')
        wmsg2 = '\tConvergence parameter: {0:.4f}'.format(dparam)
        wargs = [iteration, maximum_iteration, guess[0], guess[1], airmass]
        WLOG(p, '', [wmsg.format(*wargs), wmsg2])
        # ---------------------------------------------------------------------
        # get current best-fit spectrum
        tau1 = tapas_fit(np.isfinite(wave), guess[0], guess[1])
        tau1 = tau1.reshape(sp.shape)
        # ---------------------------------------------------------------------
        # for each order, we fit the SED after correcting for absorption
        for order_num in range(norders):
            # -----------------------------------------------------------------
            # get the per-order spectrum divided by best guess
            sp2 = sp[order_num] / tau1[order_num]
            # -----------------------------------------------------------------
            # find this orders good pixels
            good = keep[order_num]
            # -----------------------------------------------------------------
            # if we have enough valid points, we normalize the domain by its
            #    median
            if np.sum(good) > min_number_good_points:
                limit = np.percentile(tau1[order_num][good], btrans_percentile)
                best_trans = tau1[order_num] > limit
                norm = np.nanmedian(sp2[best_trans])
            else:
                norm = np.ones_like(sp2)
            # -----------------------------------------------------------------
            # normalise this orders spectrum
            sp[order_num] = sp[order_num] / norm
            # normalise sp2 and the sed
            sp2 = sp2 / norm
            sed[order_num] = sed[order_num] / norm
            # -----------------------------------------------------------------
            # find far outliers to the SED for sigma-clipping
            with warnings.catch_warnings(record=True) as _:
                res = sp2 - sed[order_num]
                res -= np.nanmedian(res)
                res /= np.nanmedian(np.abs(res))
            # set all NaN pixels to large value
            nanmask = ~np.isfinite(res)
            res[nanmask] = 99
            # -----------------------------------------------------------------
            # apply sigma clip
            good &= np.abs(res) < nsigclip
            # apply median to sed
            with warnings.catch_warnings(record=True) as _:
                good &= sed[order_num] > 0.5 * np.nanmedian(sed[order_num])
            # only fit where the transmission is greater than a certain value
            good &= tau1[order_num] > threshold_transmission_fit
            # -----------------------------------------------------------------
            # set non-good (bad) pixels to NaN
            sp2[~good] = np.nan
            # sp3 is a median-filtered version of sp2 where pixels that have
            #     a transmission that is too low are clipped.
            sp3 = filters.median_filter(sp2 - sed[order_num], med_filt)
            sp3 = sp3 + sed[order_num]
            # find all the NaNs and set to zero
            nanmask = ~np.isfinite(sp3)
            sp3[nanmask] = 0.0
            # also set zero pixels in sp3 to be non-good pixels in "good"
            good[sp3 == 0.0] = False
            # -----------------------------------------------------------------
            # we smooth sp3 with a kernel. This kernel has to be small
            #    enough to preserve stellar features and large enough to
            #    subtract noise this is why we have an order-dependent width.
            #    the stellar lines are expected to be larger than 200km/s,
            #    so a kernel much smaller than this value does not make sense
            ew = wconv[order_num] / tellu_med_sampling / fwhm()
            # get the kernal x values
            wconv_ord = int(wconv[order_num])
            kernal_x = np.arange(-2 * wconv_ord, 2 * wconv_ord)
            # get the gaussian kernal
            kernal_y = np.exp(-0.5*(kernal_x / ew) **2 )
            # normalise kernal so it is max at unity
            kernal_y = kernal_y / np.sum(kernal_y)
            # -----------------------------------------------------------------
            # construct a weighting matrix for the sed
            ww1 = np.convolve(good, kernal_y, mode='same')
            # need to weight the spectrum accordingly
            spconv = np.convolve(sp3 * good, kernal_y, mode='same')
            # update the sed
            with warnings.catch_warnings(record=True) as _:
                sed_update = spconv / ww1
            # set all small values to 1% to avoid small weighting errors
            sed_update[ww1 < small_weight] = np.nan

            if wconv[order_num] == MKTELLU_FINER_CONV_WIDTH:
                rms_limit = 0.1
            else:
                rms_limit = 0.3

            # iterate around and sigma clip
            for iteration_sig_clip_good in range(1, 6):

                with warnings.catch_warnings(record=True) as _:
                    residual_SED = sp3 - sed_update
                    residual_SED[~good] = np.nan

                rms = np.abs(residual_SED)

                with warnings.catch_warnings(record=True) as _:
                    good[rms > (rms_limit / iteration_sig_clip_good)] = 0

                # ---------------------------------------------------------
                # construct a weighting matrix for the sed
                ww1 = np.convolve(good, kernal_y, mode='same')
                # need to weight the spectrum accordingly
                spconv = np.convolve(sp3 * good, kernal_y, mode='same')
                # update the sed
                with warnings.catch_warnings(record=True) as _:
                    sed_update = spconv / ww1
                # set all small values to 1% to avoid small weighting errors
                sed_update[ww1 < 0.01] = np.nan

            # -----------------------------------------------------------------
            # if we have lots of very strong absorption, we subtract the
            #    median value of pixels where the transmission is expected to
            #    be smaller than 1%. This improves things in the stronger
            #    absorptions
            pedestal = tau1[order_num] < 0.01
            # check if we have enough strong absorption
            if np.sum(pedestal) > 100:
                zero_point = np.nanmedian(sp[order_num, pedestal])
                # if zero_point is finite subtract it off the spectrum
                if np.isfinite(zero_point):
                    sp[order_num] -= zero_point
            # -----------------------------------------------------------------
            # update the sed
            sed[order_num] = sed_update
            # append sp3
            sp3_arr[order_num] = np.array(sp3)
            # -----------------------------------------------------------------
            # debug plot
            if PLOT and DEBUG_PLOT and not skip:
                # plot only every 10 iterations
                if iteration == 10:
                    # plot the transmission map plot
                    pargs = [order_num, wave, tau1, sp, sp3,
                             sed, sed_update, keep]
                    mk_tellu_wave_flux_plot(p, *pargs)
                    # get user input to continue or skip
                    imsg = 'Press [Enter] for next or [s] for skip:\t'
                    uinput = input(imsg)
                    if 's' in uinput.lower():
                        skip = True
                    # close plot
                    plt.close()
        # ---------------------------------------------------------------------
        # update the iteration number
        iteration += 1
        # ---------------------------------------------------------------------
        # update while parameters
        cond1 = dparam > dparam_threshold
        cond2 = iteration < maximum_iteration
        cond3 = not fail
    # ---------------------------------------------------------------------
    if PLOT and not DEBUG_PLOT:
        # if plot orders is 'all' plot all
        if plot_order_nums == 'all':
            plot_order_nums = np.arange(norders).astype(int)
            # start non-interactive plot
            plt.ioff()
            off = True
        else:
            plt.ion()
            off = False
        # loop around the orders to show
        for order_num in plot_order_nums:
            pargs = [order_num, wave, tau1, sp, sp3_arr[order_num], sed,
                     sed[order_num], keep]
            mk_tellu_wave_flux_plot(p, *pargs)
        if off:
            plt.ion()

    # return values via loc
    loc['PASSED'] = not fail
    loc['RECOV_AIRMASS'] = guess[1]
    loc['RECOV_WATER'] = guess[0]
    loc['SP_OUT'] = sp
    loc['SED_OUT'] = sed
    # return loc
    return loc


def calc_tapas_abso(p, loc, keep, tau_water, tau_others):
    """
    generates a Tapas spectrum from the saved temporary .npy
    structure and scales with the given optical depth for
    water and all other absorbers

    as an input, we give a "keep" vector, values set to keep=0
    will be set to zero and not taken into account in the fitting
    algorithm

    """
    # get constants from p
    tau_water_upper = MKTELLU_TAU_WATER_ULIMIT
    tau_others_lower = MKTELLU_TAU_OTHER_LLIMIT
    tau_others_upper = MKTELLU_TAU_OTHER_ULIMIT
    tapas_small_number = MKTELLU_SMALL_LIMIT

    # get data from loc
    sp_water = np.array(loc['TAPAS_WATER'])
    sp_others = np.array(loc['TAPAS_OTHERS'])

    # line-of-sight optical depth for water cannot be negative
    if tau_water < 0:
        tau_water = 0
    # line-of-sight optical depth for water cannot be too big -
    #    set uppder threshold
    if tau_water > tau_water_upper:
        tau_water = tau_water_upper
    # line-of-sight optical depth for other absorbers cannot be less than
    #     one (that's zenith) keep the limit at 0.2 just so that the value
    #     gets propagated to header and leaves open the possibility that
    #     during the convergence of the algorithm, values go slightly
    #     below 1.0
    if tau_others < tau_others_lower:
        tau_others = tau_others_lower
    # line-of-sight optical depth for other absorbers cannot be greater than 5
    #     that would be an airmass of 5 and SPIRou cannot observe there
    if tau_others > tau_others_upper:
        tau_others = tau_others_upper

    # we will set to a fractional exponent, we cannot have values below zero
    #    for water
    water_zero = sp_water < 0
    sp_water[water_zero] = 0
    # for others
    others_zero = sp_others < 0
    sp_others[others_zero] = 0

    # calculate the tapas spectrum from absorbers
    sp = (sp_water ** tau_water) * (sp_others ** tau_others)

    # values not to be kept are set to a very low value
    sp[~keep.ravel()] = tapas_small_number

    # to avoid divisons by 0, we set values that are very low to 1e-9
    sp[sp < tapas_small_number] = tapas_small_number

    # return the tapas spectrum
    return sp


def construct_convolution_kernal():
    lsf = FWHM_PIXEL_LSF
    # get the number of pixels
    npix = int(np.ceil(3 * lsf * 3.0 / 2) * 2 + 1)
    # set up the kernal x values
    xpix = np.arange(npix) - npix // 2
    # get the gaussian kernel
    ker = np.exp(-0.5 * ( xpix / lsf / fwhm()) ** 2 )
    # return kernel
    return ker


def resample_tapas(p, loc, mwave, npix, nord, tapas):

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
            svalues = tapas_spline(mwave[order_num] / 10.0)
            # convolve with a gaussian function
            cvalues = np.convolve(svalues, kernel, mode='same')
            # add to storage
            tapas_all_species[n_species, start: end] = cvalues

    # extract the water and other line-of-sight optical depths
    loc['TAPAS_WATER'] = tapas_all_species[1, :]
    loc['TAPAS_OTHERS'] = np.prod(tapas_all_species[2:, :], axis=0)

    return loc


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":

    # define the dictionaries to store stuff
    p, loc = dict(), dict()
    # get the data based on the instrument
    loc = get_mk_tellu_data(p, loc)
    # ----------------------------------------------------------------------
    # extract out data from loc
    mwave = loc['MWAVE']
    sflux = loc['SFLUX']
    swave = loc['SWAVE']
    airmass = loc['AIRMASS']
    # ----------------------------------------------------------------------
    # get the number of orders and number of pixels
    nord, npix = mwave.shape
    # define the convolution size (per order)
    wconv = np.repeat(MKTELLU_DEFAULT_CONV_WIDTH, nord).astype(float)
    # ----------------------------------------------------------------------
    # load the transmission model
    tapas = Table.read(TRANS_MODEL)
    # construct kernal (gaussian for instrument FWHM PIXEL LSF
    kernel = construct_convolution_kernal()
    # tapas spectra resampled onto our data wavelength vector
    loc = resample_tapas(p, loc, mwave, npix, nord, tapas)
    # ----------------------------------------------------------------------
    # get airmass from header
    loc['AIRMASS'] = airmass
    # set master wave
    loc['WAVE'] = mwave
    # set the telluric spectrum
    loc['SP'] = sflux
    # set up the convolution size
    loc['WCONV'] = wconv

    # calculate the telluric absorption
    loc = calculate_telluric_absorption(p, loc)
    # calculate tranmission map from sp and sed
    transmission_map = loc['SP_OUT'] / loc['SED_OUT']
    # ----------------------------------------------------------------------
    # construct output filename
    outname = os.path.join(OUTPUT_DIR, INPUT_TSS.replace('.fits', OUT_SUFFIX))
    # write to file
    fits.writeto(outname, transmission_map)


# =============================================================================
# End of code
# =============================================================================