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


# =============================================================================
# Define variables
# =============================================================================
# plotting options
DEBUG_PLOT = True
PLOT = True

# -----------------------------------------------------------------------------
# output directory
IN_SUFFIX = '_trans.fits'
# intput directory
OUT_SUFFIX = '_corrected.fits'


# =============================================================================
# Define functions
# =============================================================================
def tellu_fit_recon_abso_plot(p, loc, sflux, swaveall):
    # get constants from p
    selected_order = p['TELLU_FIT_RECON_PLT_ORDER']

    if selected_order == 'all':
        selected_orders = np.arange(sflux.shape[0])
    else:
        selected_orders = [selected_order]

    for selected_order in selected_orders:
        # get data dimensions
        ydim, xdim = sflux.shape
        # get selected order wave lengths
        swave = swaveall[selected_order, :]
        # get the data from loc for selected order
        start, end = selected_order * xdim, selected_order * xdim + xdim
        ssp = np.array(sflux[selected_order, :])
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


def tellu_pca_comp_plot(p, loc, mwaveall):
    plot_name = 'tellu_pca_comp_plot'
    # get constants from p
    npc = loc['NPC']
    # get data from loc
    wave = mwaveall.ravel()
    pc = loc['PC']
    # set up fig
    fig, frame = plt.subplots(ncols=1, nrows=1)
    # plot principle components
    for it in range(npc):
        # define the label for the component
        label = 'pc {0}'.format(it + 1)
        # plot the component with correct label
        frame.plot(wave, pc[:, it], label=label)
    # add legend
    frame.legend(loc=0)
    # add labels
    title = 'Principle component plot'
    frame.set(title=title, xlabel='Wavelength [nm]',
              ylabel='Principle component power')
    # end plotting function properly
    plt.show()
    plt.close()


def calculate_absorption_pca(p, loc, x, mask, sflux):
    # get constants from p
    npc = p['TELLU_NUMBER_OF_PRINCIPLE_COMP']

    # get eigen values
    eig_u, eig_s, eig_vt = np.linalg.svd(x[:, mask], full_matrices=False)

    # create pc image
    pc = np.zeros([np.product(sflux.shape), npc])

    # fill pc image
    with warnings.catch_warnings(record=True) as _:
        for it in range(npc):
            for jt in range(x.shape[0]):
                pc[:, it] += eig_u[jt, it] * x[jt, :]

    fit_pc = np.array(pc)

    # save pc image to loc
    loc['PC'] = pc
    loc['NPC'] = npc
    loc['FIT_PC'] = fit_pc
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
    ew = vsini / (p['TELLU_MED_SAMPLING'] / fwhm())
    # set up the kernel exponent
    xx = np.arange(ew * 6) - ew * 3
    # kernal is the a gaussian
    ker2 = np.exp(-.5 * (xx / ew) ** 2)

    ker2 /= np.sum(ker2)
    # add to loc
    loc['KER2'] = ker2
    # return loc
    return loc


def calc_recon_abso(p, loc, sflux, swave):
    func_name = 'calc_recon_abso()'
    # get data from loc
    sp = np.array(sflux)
    tapas_all_species = loc['TAPAS_ALL']
    amps_abso_total = loc['AMPS_ABSOL_TOTAL']
    # get data dimensions
    ydim, xdim = sflux.shape
    # redefine storage for recon absorption
    recon_abso = np.ones(np.product(sflux.shape))
    # flatten spectrum and wavelengths
    sp2 = sp.ravel()
    wave2 = swave.ravel()
    # define the good pixels as those above minimum transmission
    with warnings.catch_warnings(record=True) as _:
        keep = tapas_all_species[0, :] > p['TELLU_FIT_MIN_TRANSMISSION']
    # also require wavelength constraints
    keep &= (wave2 > p['TELLU_LAMBDA_MIN'])
    keep &= (wave2 < p['TELLU_LAMBDA_MAX'])
    # construct convolution kernel
    loc = construct_convolution_kernal2(p, loc, p['TELLU_FIT_VSINI'])
    # ------------------------------------------------------------------
    # loop around a number of times
    template2 = None
    for ite in range(p['TELLU_FIT_NITER']):
        # log progress
        wmsg = 'Iteration {0} of {1}'.format(ite + 1, p['TELLU_FIT_NITER'])
        WLOG(p, '', wmsg)
        # setup storage for template2
        template2 = np.zeros(np.product(sflux.shape))
        # --------------------------------------------------------------
        # loop around orders
        for order_num in range(ydim):
            # get start and end points
            start = order_num * xdim
            end = order_num * xdim + xdim
            # produce a mask of good transmission
            order_tapas = tapas_all_species[0, start:end]
            with warnings.catch_warnings(record=True) as _:
                mask = order_tapas > p['TRANSMISSION_CUT']
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
        # --------------------------------------------------------------
        # get residual spectrum
        with warnings.catch_warnings(record=True) as _:
            resspec = (sp2 / template2) / recon_abso
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


def get_abso(p, sflux):

    # load absorption map files
    rawfiles = os.listdir(os.path.join(p['WORKSPACE'], 'trans'))

    # sort through files
    nfiles = 0
    trans_files = []
    for it, filename in enumerate(rawfiles):
        if filename.endswith(IN_SUFFIX):
            # construct absolute file path
            absfilename = os.path.join(p['WORKSPACE'], 'trans', filename)
            # append and iterate
            nfiles += 1
            trans_files.append(absfilename)

    # set up storage for the absorption
    abso = np.zeros([nfiles, np.product(sflux.shape)])

    for it, filename in enumerate(trans_files):

        wmsg = 'Loading trans file={0}'.format(filename)
        WLOG(p, '', wmsg)
        # load data
        data_it = fits.getdata(filename)
        # push data and flatten
        abso[it, :] = data_it.reshape(np.product(sflux.shape))

    # log the absorption cube
    with warnings.catch_warnings(record=True) as w:
        log_abso = np.log(abso)
    # return the log_absorption
    return abso, log_abso


# =============================================================================
# Start of code
# =============================================================================
# Main code here
def main(instrument, filename=None):
    # define the dictionaries to store stuff
    p, loc = dict(), dict()

    if instrument.upper() == 'CARMENES':
        import setup_car as setup
        p = setup.begin()
    elif instrument.upper() == 'SPIROU':
        import setup_spirou as setup
        p = setup.begin()
    else:
        raise ValueError('Instrument = "{0}" not supported'.format(instrument))

    # get the data based on the instrument
    p, loc = setup.get_fit_tellu_data(p, loc, filename)

    # ----------------------------------------------------------------------
    # extract out data from loc
    mwave = loc['MWAVE']
    sflux = loc['SFLUX']
    blaze = loc['BLAZE']
    swave = loc['SWAVE']
    airmass = loc['AIRMASS']
    tapas = loc['TAPAS']


    sflux, blaze = normalise_by_blaze(p, sflux, blaze)

    # get the number of orders and number of pixels
    nord, npix = mwave.shape
    # ----------------------------------------------------------------------
    # construct kernal (gaussian for instrument FWHM PIXEL LSF
    kernel = kernal_with_instrument(p['FWHM_PIXEL_LSF'])
    # tapas spectra resampled onto our data wavelength vector
    loc = resample_tapas(p, loc, mwave, npix, nord, tapas, kernel)
    # ----------------------------------------------------------------------
    # get abso files and log them
    abso, log_abso = get_abso(p, sflux)

    # ----------------------------------------------------------------------
    # Check we have enough files
    # ----------------------------------------------------------------------
    nfiles = abso.shape[0]
    npc = p['TELLU_NUMBER_OF_PRINCIPLE_COMP']
    # check that we have enough files (greater than number of principle
    #    components)
    if nfiles <= npc:
        emsg1 = 'Not enough "TELL_MAP" files in telluDB to run PCA analysis'
        emsg2 = '\tNumber of files = {0}, number of PCA components = {1}'
        emsg3 = '\tNumber of files > number of PCA components'
        emsg4 = '\tAdd more files or reduce number of PCA components'
        WLOG(p, 'error', [emsg1, emsg2.format(nfiles, npc),
                                     emsg3, emsg4])

    # ----------------------------------------------------------------------
    # Locate valid pixels for PCA
    # ----------------------------------------------------------------------
    # determining the pixels relevant for PCA construction
    keep = np.isfinite(np.sum(abso, axis=0))
    # log fraction of valid (non NaN) pixels
    fraction = np.sum(keep) / len(keep)
    wmsg = 'Fraction of valid pixels (not NaNs) for PCA construction = {0:.3f}'
    WLOG(p, '', wmsg.format(fraction))
    # log fraction of valid pixels > 1 - (1/e)
    with warnings.catch_warnings(record=True) as w:
        keep &= np.min(log_abso, axis=0) > -1
    fraction = np.sum(keep) / len(keep)
    wmsg = 'Fraction of valid pixels with transmission > 1 - (1/e) = {0:.3f}'
    WLOG(p, '', wmsg.format(fraction))


    # ----------------------------------------------------------------------
    # Perform PCA analysis on the log of the telluric absorption map
    # ----------------------------------------------------------------------
    # Requires p:
    #           TELLU_NUMBER_OF_PRINCIPLE_COMP
    #           ADD_DERIV_PC
    #           FIT_DERIV_PC
    # Requires loc:
    #           DATA
    # Returns loc:
    #           PC
    #           NPC
    #           FIT_PC
    loc = calculate_absorption_pca(p, loc, log_abso, keep, sflux)

    # Plot PCA components
    # debug plot
    if PLOT:
        # plot the transmission map plot
        tellu_pca_comp_plot(p, loc, mwave)

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
    WLOG(p, '', [wmsg1])
    # shift tapas
    for comp in range(len(loc['TAPAS_ALL'])):
        wargs = [p, loc['TAPAS_ALL'][comp], mwave, swave]
        stapas = wave2wave(*wargs, reshape=True)
        loc['TAPAS_ALL'][comp] = stapas.reshape(wargs[1].shape)

    # ------------------------------------------------------------------
    # set up storage in loc
    loc['RECON_ABSO'] = np.ones(np.product(sflux.shape))
    loc['AMPS_ABSOL_TOTAL'] = np.zeros(loc['NPC'])
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
    loc = calc_recon_abso(p, loc, sflux, swave)
    # debug plot
    if PLOT > 0:
        # plot the recon abso plot
        tellu_fit_recon_abso_plot(p, loc, sflux, swave)

    # ------------------------------------------------------------------
    # Write corrected spectrum to E2DS
    # ------------------------------------------------------------------
    # reform the E2DS
    sp_out = loc['SP2'] / loc['RECON_ABSO']
    sp_out = sp_out.reshape(sflux.shape)

    # ----------------------------------------------------------------------
    # construct output filename
    outfile = os.path.basename(p['INPUT_SPEC'].replace('.fits', OUT_SUFFIX))
    outname = os.path.join(p['WORKSPACE'], 'corrected', outfile)

    # write to file
    fits.writeto(outname, sp_out, overwrite=True)

# =============================================================================
# End of code
# =============================================================================