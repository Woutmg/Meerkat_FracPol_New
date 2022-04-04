#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:23:46 2022

@author: goesaert
"""

import numpy as np


import scipy.constants as c
from scipy import stats


from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from astropy import wcs
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy import units as u

import os
import sys
from astropy.table import Table, join, vstack
from astropy.nddata.utils import extract_array
import pyregion
import argparse

import integratedflux as flux
import warnings
from astropy.utils.exceptions import AstropyWarning

import time

directory_univ_I = r'/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/I_slices/'
directory_local = r'/data2/goesaert/'
directory_lofar1 = r'/net/lofar1/data1/GoesaertW/Abell_85/'

region_name = "Abell85_catalog_BDSF_rsl.reg"

rsl = fits.open(get_pkg_data_filename('Abell85_catalog_BDSF_rsl.fits'))[1].data


warnings.filterwarnings('ignore', category=AstropyWarning, append=True)
mask = flux.mask_regions(directory_lofar1+'I_slices/Abell_85_I_plane_freq1.fits', region_name, \
                         directory_lofar1+'testmask.fits', maskoutside=False)

start = time.time()

hdul_image = fits.open(directory_univ_I+'Abell_85_I_plane_freq1.fits')

for n in range(100):
    measurement = flux.integratedflux(directory_lofar1+'I_slices/Abell_85_I_plane_freq1.fits'\
                                      , mask, rsl, n, hdul=hdul_image)
    measurement = flux.integratedflux(directory_lofar1+'I_slices/Abell_85_I_plane_freq1.fits'\
                                      , np.invert(mask), rsl, n, hdul=hdul_image)

#uncertainty = flux.uncertainty_flux(I_dir, measurement, beamnumb, rms=0.01*measurement, delta_cal=0.1)
#print(measurement, beamnumb, uncertainty)
end = time.time()
print("The time of execution of above program is :", end-start)

