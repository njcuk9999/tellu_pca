#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# CODE DESCRIPTION HERE

Created on 2019-02-26 19:07
@author: ncook
Version 0.0.1
"""
from shared_functions import *

from astropy.io import fits
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUVSpline
import warnings
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA


# =============================================================================
# Define variables
# =============================================================================
# plotting options
DEBUG_PLOT = True
PLOT = True

# -----------------------------------------------------------------------------
# define instrument name
INSTRUMENT = 'CARMENES'
INSTRUMENT = 'SPIROU'


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
def tellu_fit_recon_abso_plot(p, loc):
    # get constants from p
    selected_order = TELLU_FIT_RECON_PLT_ORDER
    # get data dimensions
    ydim, xdim = loc['DATA'].shape
    # get selected order wave lengths
    swave = loc['WAVE_IT'][selected_order, :]
    # get the data from loc for selected order
    start, end = selected_order * xdim, selected_order * xdim + xdim
    ssp = np.array(loc['SP'][selected_order, :])
    ssp2 = np.array(loc['SP2'][start:end])
    stemp2 = np.array(loc['TEMPLATE2'][start:end])
    srecon_abso = np.array(loc['RECON_ABSO'][start:end])
    # set up fig
    fig, frame = plt.subplots(ncols=1, nrows=1)
    # plot spectra for selected order
    frame.plot(swave, ssp / np.nanmedian(ssp), color='k', label='input SP')
    frame.plot(swave, ssp2 / np.nanmedian(ssp2) / srecon_abso, color='g',
               label='Cleaned SP')
    frame.plot(swave, stemp2 / np.nanmedian(stemp2), color='c',
               label='Template')
    frame.plot(swave, srecon_abso, color='r', label='recon abso')
    # add legend
    frame.legend(loc=0)
    # add labels
    title = 'Reconstructed Absorption (Order = {0})'
    frame.set(title=title.format(selected_order),
              xlabel='Wavelength [nm]', ylabel='Normalised flux')

    # end plotting function properly
    plt.show()
    plt.close()


def calculate_absorption_pca(p, loc, x, mask):
    # get constants from p
    npc = TELLU_NUMBER_OF_PRINCIPLE_COMP

    pca = PCA(n_components=npc)
    pcs = pca.fit_transform(x[:, mask])

    # save pc image to loc
    loc['PC'] = pcs
    loc['NPC'] = npc
    loc['FIT_PC'] = pcs
    # return loc
    return loc


def wave2wave(p, spectrum, wave1, wave2, reshape=False):
    """
    Shifts a "spectrum" at a given wavelength solution (map), "wave1", to
    another wavelength solution (map) "wave2"

    :param p: ParamDict, the parameter dictionary
    :param spectrum: numpy array (2D),  flux in the reference frame of the
                     file wave1
    :param wave1: numpy array (2D), initial wavelength grid
    :param wave2: numpy array (2D), destination wavelength grid
    :param reshape: bool, if True try to reshape spectrum to the shape of
                    the output wave solution

    :return output_spectrum: numpy array (2D), spectrum resampled to "wave2"
    """
    func_name = 'wave2wave()'
    # deal with reshape
    if reshape or (spectrum.shape != wave2.shape):
        try:
            spectrum = spectrum.reshape(wave2.shape)
        except ValueError:
            emsg1 = ('Spectrum (shape = {0}) cannot be reshaped to'
                     ' shape = {1}')
            emsg2 = '\tfunction = {0}'.format(func_name)
            eargs = [spectrum.shape, wave2.shape]
            WLOG(p, 'error', [emsg1.format(*eargs), emsg2])

    # if they are the same
    if np.sum(wave1 != wave2) == 0:
        return spectrum

    # size of array, assumes wave1, wave2 and spectrum have same shape
    sz = np.shape(spectrum)
    # create storage for the output spectrum
    output_spectrum = np.zeros(sz) + np.nan

    # looping through the orders to shift them from one grid to the other
    for iord in range(sz[0]):
        # only interpolate valid pixels
        g = np.isfinite(spectrum[iord, :])

        # if no valid pixel, thn skip order
        if np.sum(g) != 0:
            # spline the spectrum
            spline = IUVSpline(wave1[iord, g], spectrum[iord, g], k=5, ext=1)

            # keep track of pixels affected by NaNs
            splinemask = IUVSpline(wave1[iord, :], g, k=5, ext=1)

            # spline the input onto the output
            output_spectrum[iord, :] = spline(wave2[iord, :])

            # find which pixels are not NaNs
            mask = splinemask(wave2[iord, :])

            # set to NaN pixels outside of domain
            bad = (output_spectrum[iord, :] == 0)
            output_spectrum[iord, bad] = np.nan

            # affected by a NaN value
            # normally we would use only pixels ==1, but we get values
            #    that are not exactly one due to the interpolation scheme.
            #    We just set that >50% of the
            # flux comes from valid pixels
            bad = (mask <= 0.5)
            # mask pixels affected by nan
            output_spectrum[iord, bad] = np.nan

    # return the filled output spectrum
    return output_spectrum


def construct_convolution_kernal2(p, loc, vsini):
    func_name = 'construct_convolution_kernal2()'

    # gaussian ew for vinsi km/s
    ew = vsini / TELLU_MED_SAMPLING / fwhm()
    # set up the kernel exponent
    xx = np.arange(ew * 6) - ew * 3
    # kernal is the a gaussian
    ker2 = np.exp(-.5 * (xx / ew) ** 2)

    ker2 /= np.sum(ker2)
    # add to loc
    loc['KER2'] = ker2
    loc.set_source('KER2', func_name)
    # return loc
    return loc


def calc_recon_abso(p, loc):
    func_name = 'calc_recon_abso()'
    # get data from loc
    sp = loc['sp']
    tapas_all_species = loc['TAPAS_ALL_SPECIES']
    amps_abso_total = loc['AMPS_ABSOL_TOTAL']
    # get data dimensions
    ydim, xdim = loc['DATA'].shape
    # redefine storage for recon absorption
    recon_abso = np.ones(np.product(loc['DATA'].shape))
    # flatten spectrum and wavelengths
    sp2 = sp.ravel()
    wave2 = loc['WAVE_IT'].ravel()
    # define the good pixels as those above minimum transmission
    with warnings.catch_warnings(record=True) as _:
        keep = tapas_all_species[0, :] > TELLU_FIT_MIN_TRANSMISSION
    # also require wavelength constraints
    keep &= (wave2 > TELLU_LAMBDA_MIN)
    keep &= (wave2 < TELLU_LAMBDA_MAX)
    # construct convolution kernel
    loc = construct_convolution_kernal2(p, loc, TELLU_FIT_VSINI)
    # ------------------------------------------------------------------
    # loop around a number of times
    template2 = None
    for ite in range(TELLU_FIT_NITER):
        # log progress
        wmsg = 'Iteration {0} of {1}'.format(ite + 1, TELLU_FIT_NITER)
        WLOG(p, '', wmsg)

        # --------------------------------------------------------------
        # if we don't have a template construct one
        if not loc['FLAG_TEMPLATE']:
            # define template2 to fill
            template2 = np.zeros(np.product(loc['DATA'].shape))
            # loop around orders
            for order_num in range(ydim):
                # get start and end points
                start = order_num * xdim
                end = order_num * xdim + xdim
                # produce a mask of good transmission
                order_tapas = tapas_all_species[0, start:end]
                with warnings.catch_warnings(record=True) as _:
                    mask = order_tapas > TRANSMISSION_CUT
                # get good transmission spectrum
                spgood = sp[order_num, :] * np.array(mask, dtype=float)
                recongood = recon_abso[start:end]
                # convolve spectrum
                ckwargs = dict(v=loc['KER2'], mode='same')
                sp2b = np.convolve(spgood / recongood, **ckwargs)
                # convolve mask for weights
                ww = np.convolve(np.array(mask, dtype=float), **ckwargs)
                # wave weighted convolved spectrum into template2
                with warnings.catch_warnings(record=True) as _:
                    template2[start:end] = sp2b / ww
        # else we have template so load it
        else:
            template2 = loc['TEMPLATE2']
            # -----------------------------------------------------------------
            # Shift the template to correct frame
            # -----------------------------------------------------------------
            # log process
            wmsg1 = '\tShifting template on to master wavelength grid'
            wargs = [os.path.basename(loc['MASTERWAVEFILE'])]
            wmsg2 = '\t\tFile = {0}'.format(*wargs)
            WLOG(p, '', [wmsg1, wmsg2])
            # shift template
            wargs = [p, template2, loc['MASTERWAVE'], loc['WAVE_IT']]
            template2 = wave2wave(*wargs, reshape=True).reshape(template2.shape)
        # --------------------------------------------------------------
        # get residual spectrum
        with warnings.catch_warnings(record=True) as _:
            resspec = (sp2 / template2) / recon_abso
        # --------------------------------------------------------------
        if loc['FLAG_TEMPLATE']:
            # construct convolution kernel
            vsini = TELLU_FIT_VSINI2
            loc = construct_convolution_kernal2(p, loc, vsini)
            # loop around orders
            for order_num in range(ydim):
                # get start and end points
                start = order_num * xdim
                end = order_num * xdim + xdim
                # catch NaN warnings and ignore
                with warnings.catch_warnings(record=True) as _:
                    # produce a mask of good transmission
                    order_tapas = tapas_all_species[0, start:end]
                    mask = order_tapas > TRANSMISSION_CUT
                    fmask = np.array(mask, dtype=float)
                    # get good transmission spectrum
                    resspecgood = resspec[start:end] * fmask
                    recongood = recon_abso[start:end]
                # convolve spectrum
                ckwargs = dict(v=loc['KER2'], mode='same')
                with warnings.catch_warnings(record=True) as _:
                    sp2b = np.convolve(resspecgood / recongood, **ckwargs)
                # convolve mask for weights
                ww = np.convolve(np.array(mask, dtype=float), **ckwargs)
                # wave weighted convolved spectrum into dd
                with warnings.catch_warnings(record=True) as _:
                    resspec[start:end] = resspec[start:end] / (sp2b / ww)
        # --------------------------------------------------------------
        # Log dd and subtract median
        # log dd
        with warnings.catch_warnings(record=True) as _:
            log_resspec = np.log(resspec)
        # --------------------------------------------------------------
        # subtract off the median from each order
        for order_num in range(ydim):
            # get start and end points
            start = order_num * xdim
            end = order_num * xdim + xdim
            # skip if whole order is NaNs
            if np.sum(np.isfinite(log_resspec[start:end])) == 0:
                continue
            # get median
            log_resspec_med = np.nanmedian(log_resspec[start:end])
            # subtract of median
            log_resspec[start:end] = log_resspec[start:end] - log_resspec_med
        # --------------------------------------------------------------
        # set up fit
        fit_dd = np.array(log_resspec)
        # --------------------------------------------------------------
        # identify good pixels to keep
        keep &= np.isfinite(fit_dd)
        keep &= np.sum(np.isfinite(loc['FIT_PC']), axis=1) == loc['NPC']
        # log number of kept pixels
        wmsg = '\tNumber to keep total = {0}'.format(np.sum(keep))
        WLOG(p, '', wmsg)
        # --------------------------------------------------------------
        # calculate amplitudes and reconstructed spectrum
        largs = [fit_dd[keep], loc['FIT_PC'][keep, :]]
        amps, recon = linear_minimization(*largs)
        # --------------------------------------------------------------
        # set up storage for absorption array 2
        abso2 = np.zeros(len(resspec))
        with warnings.catch_warnings(record=True) as _:
            for ipc in range(len(amps)):
                amps_abso_total[ipc] += amps[ipc]
                abso2 += loc['PC'][:, ipc] * amps[ipc]
            recon_abso *= np.exp(abso2)

    # save outputs to loc
    loc['SP2'] = sp2
    loc['TEMPLATE2'] = template2
    loc['RECON_ABSO'] = recon_abso
    loc['AMPS_ABSOL_TOTAL'] = amps_abso_total
    # set the source
    skeys = ['SP2', 'TEMPLATE2', 'RECON_ABSO', 'AMPS_ABSOL_TOTAL']
    loc.set_sources(skeys, func_name)
    # return loc
    return loc


def calc_molecular_absorption(p, loc):

    # get constants from p
    limit = TELLU_FIT_LOG_LIMIT
    # get data from loc
    recon_abso = loc['RECON_ABSO']
    tapas_all_species = loc['TAPAS_ALL_SPECIES']

    # log data
    log_recon_abso = np.log(recon_abso)
    with warnings.catch_warnings(record=True) as _:
        log_tapas_abso = np.log(tapas_all_species[1:, :])

    # get good pixels
    with warnings.catch_warnings(record=True) as _:
        keep = np.min(log_tapas_abso, axis=0) > limit
        keep &= log_recon_abso > limit
    keep &= np.isfinite(recon_abso)

    # get keep arrays
    klog_recon_abso = log_recon_abso[keep]
    klog_tapas_abso = log_tapas_abso[:, keep]

    # work out amplitudes and recon
    amps, recon = linear_minimization(klog_recon_abso, klog_tapas_abso)

    # load amplitudes into loc
    for it, molecule in enumerate(TELLU_ABSORBERS[1:]):
        # get molecule keyword store and key
        molkey = molecule.upper()
        # load into loc
        loc[molkey] = amps[it]
    # return loc
    return loc


def linear_minimization(vector, sample):
    func_name = 'linear_minimization()'

    vector = np.array(vector)
    sample = np.array(sample)
    sz_sample = sample.shape
    sz_vector = vector.shape

    if sz_vector[0] == sz_sample[0]:
        case = 2
    elif sz_vector[0] == sz_sample[1]:
        case = 1
    else:
        emsg = ('Neither vector[0]==sample[0] nor vector[0]==sample[1] '
                '(function = {0})')
        raise ValueError(emsg.format(func_name))

    # vector of N elements
    # sample: matrix N * M each M column is adjusted in amplitude to minimize
    # the chi2 according to the input vector
    # output: vector of length M gives the amplitude of each column

    if case == 1:
        # set up storage
        mm = np.zeros([sz_sample[0], sz_sample[0]])
        v = np.zeros(sz_sample[0])
        for i in range(sz_sample[0]):
            for j in range(i, sz_sample[0]):
                mm[i, j] = np.nansum(sample[i, :] * sample[j, :])
                mm[j, i] = mm[i, j]
            v[i] = np.nansum(vector * sample[i, :])

        if np.linalg.det(mm) == 0:
            amps = np.zeros(sz_sample[0]) + np.nan
            recon = np.zeros_like(v)
            return amps, recon

        amps = np.matmul(np.linalg.inv(mm), v)
        recon = np.zeros(sz_sample[1])
        for i in range(sz_sample[0]):
            recon += amps[i] * sample[i, :]
        return amps, recon

    if case == 2:
        # set up storage
        mm = np.zeros([sz_sample[1], sz_sample[1]])
        v = np.zeros(sz_sample[1])
        for i in range(sz_sample[1]):
            for j in range(i, sz_sample[1]):
                mm[i, j] = np.nansum(sample[:, i] * sample[:, j])
                mm[j, i] = mm[i, j]
            v[i] = np.nansum(vector * sample[:, i])

        if np.linalg.det(mm) == 0:
            amps = np.zeros(sz_sample[1]) + np.nan
            recon = np.zeros_like(v)
            return amps, recon

        amps = np.matmul(np.linalg.inv(mm), v)
        recon = np.zeros(sz_sample[0])
        for i in range(sz_sample[1]):
            recon += amps[i] * sample[:, i]
        return amps, recon


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":


    # define the dictionaries to store stuff
    p, loc = dict(), dict()

    # ----------------------------------------------------------------------
    # get data
    loc = get_fit_tellu_data(p, loc)

    # ----------------------------------------------------------------------
    # extract out data from loc
    mwave = loc['MWAVE']
    sflux = loc['SFLUX']
    swave = loc['SWAVE']

    # ------------------------------------------------------------------
    # Shift the pca components to correct frame
    # ------------------------------------------------------------------
    # log process
    wmsg1 = 'Shifting PCA components from master wavelength grid'
    WLOG(p, '', wmsg1)
    # shift pca components (one by one)
    for comp in range(loc['NPC']):
        wargs = [p, loc['PC'][:, comp], mwave, swave]
        shift_pc = wave2wave(*wargs, reshape=True)
        loc['PC'][:, comp] = shift_pc.reshape(wargs[1].shape)

        wargs = [p, loc['FIT_PC'][:, comp], mwave, swave]
        shift_fpc = wave2wave(*wargs, reshape=True)
        loc['FIT_PC'][:, comp] = shift_fpc.reshape(wargs[1].shape)

    # ------------------------------------------------------------------
    # Shift the pca components to correct frame
    # ------------------------------------------------------------------
    # log process
    wmsg1 = 'Shifting TAPAS spectrum from master wavelength grid'
    wmsg2 = '\tFile = {0}'.format(os.path.basename(loc['MASTERWAVEFILE']))
    WLOG(p, '', [wmsg1, wmsg2])
    # shift tapas
    for comp in range(len(loc['TAPAS_ALL_SPECIES'])):
        wargs = [p, loc['TAPAS_ALL_SPECIES'][comp], mwave, swave]
        stapas = wave2wave(*wargs, reshape=True)
        loc['TAPAS_ALL_SPECIES'][comp] = stapas.reshape(wargs[1].shape)


    # ------------------------------------------------------------------
    # Calculate reconstructed absorption
    # ------------------------------------------------------------------
    # Requires p:
    #           TELLU_FIT_MIN_TRANSMISSION
    #           TELLU_FIT_NITER
    #           TELLU_LAMBDA_MIN
    #           TELLU_LAMBDA_MAX
    #           TELLU_FIT_VSINI
    #           TRANSMISSION_CUT
    #           FIT_DERIV_PC
    #           LOG_OPT
    # Requires loc:
    #           FLAG_TEMPLATE
    #           TAPAS_ALL_SPECIES
    #           AMPS_ABSOL_TOTAL
    #           WAVE_IT
    #           TEMPLATE2
    #           FIT_PC
    #           NPC
    #           PC
    # Returns loc:
    #           SP2
    #           TEMPLATE2
    #           RECON_ABSO
    #           AMPS_ABSOL_TOTAL
    # TODO: Need to make sure we have everything for loc
    loc = calc_recon_abso(p, loc)
    # debug plot
    if PLOT > 0:
        # plot the recon abso plot
        # TODO: Need to make sure we have everything for loc
        tellu_fit_recon_abso_plot(p, loc)

    # ------------------------------------------------------------------
    # Get molecular absorption
    # ------------------------------------------------------------------
    # Requires p:
    #           TELLU_FIT_LOG_LIMIT
    # Requeres loc:
    #           RECON_ABSO
    #           TAPAS_ALL_SPECIES
    # Returns loc:
    #           TAPAS_{molecule}
    # TODO: Need to make sure we have everything for loc
    loc = calc_molecular_absorption(p, loc)

    # ------------------------------------------------------------------
    # Write corrected spectrum to E2DS
    # ------------------------------------------------------------------
    # reform the E2DS
    sp_out = loc['SP2'] / loc['RECON_ABSO']
    sp_out = sp_out.reshape(loc['DATA'].shape)

    # ----------------------------------------------------------------------
    # construct output filename
    outname = os.path.join(OUTPUT_DIR, INPUT_TSS.replace('.fits', OUT_SUFFIX))

    # write to file
    fits.writeto(outname, sp_out, overwrite=True)

# =============================================================================
# End of code
# =============================================================================