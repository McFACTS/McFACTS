import numpy as np
from numpy import random
from astropy.table import Table
from astropy import units as u
from astropy import constants as const
from scipy.stats import gamma


def analytical_kick_velocity(length):
    # took this from chat:
    # Define shape and scale parameters for the gamma distribution
    shape = 5  # Adjust to control peak sharpness
    scale = 40  # Controls spread; tweak to shift peak to ~200

    # Generate random numbers from gamma distribution
    r = gamma.rvs(a=shape, scale=scale, size=length)
    return r

# bh_binary_id_num_merger