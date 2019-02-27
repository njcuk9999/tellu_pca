from astropy.time import Time

import numpy as np



def WLOG(p, level, message):
    if type(message) is str:
        message = [message]

    timenow = Time.now()
    for mess in message:
        print('{0}: {1}: {2}'.format(timenow, level, mess))


def fwhm():
    return 2 * np.sqrt(2 * np.log(2))


