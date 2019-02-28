#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# CODE DESCRIPTION HERE

Created on 2019-02-27 13:25
@author: ncook
Version 0.0.1
"""
import mk_tellu
import os
from shared_functions import *


# =============================================================================
# Define variables
# =============================================================================

INSTRUMENT = 'Carmenes'
INSTRUMENT = 'SPIROU'

if INSTRUMENT.upper() == 'CARMENES':
    WORKSPACE = '/media/sf_D_DRIVE/tell_hack/project/data_car'
    TELLU_SUFFIX = '_A.fits'
elif INSTRUMENT.upper() == 'SPIROU':
    WORKSPACE = '/media/sf_D_DRIVE/tell_hack/project/data_spirou/tellu'
    TELLU_SUFFIX = '_e2dsff_AB.fits'
# -----------------------------------------------------------------------------


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # get files
    files = os.listdir(WORKSPACE)
    # loop around files
    for it, filename in enumerate(files):
        # print progress
        wmsg1 = 'Processing file {0} of {1}'.format(it + 1, len(files))
        wmsg2 = '\t File = {0}'.format(filename)
        WLOG(None, '', ['=' * 50, '', wmsg1, wmsg2, '', '=' * 50])
        # only process correct files
        if filename.endswith(TELLU_SUFFIX):
            # construct absolute file name
            absfilename = os.path.join(WORKSPACE, filename)
            # run mk tellu code
            mk_tellu.main(INSTRUMENT, absfilename)


# =============================================================================
# End of code
# =============================================================================