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
WORKSPACE = '/media/sf_D_DRIVE/tell_hack/project/data_car'
INSTRUMENT = 'Carmenes'
TELLU_SUFFIX = '_A.fits'
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
        wmsg = 'Processing file {0} of {1}'.format(it + 1, len(files))
        WLOG(None, '', ['=' * 50, '', wmsg, '', '=' * 50])
        # only process correct files
        if filename.endswith(TELLU_SUFFIX):
            # construct absolute file name
            absfilename = os.path.join(WORKSPACE, filename)
            # run mk tellu code
            mk_tellu.main(INSTRUMENT, absfilename)


# =============================================================================
# End of code
# =============================================================================