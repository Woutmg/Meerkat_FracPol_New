import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import scipy.constants as c
from scipy import stats

from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from astropy import wcs
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy import units as u

import bdsf

'''WITH FREQ AVERAGE:'''
#directory_univ_mkt = r'/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/'
#img = bdsf.process_image('Abell_85_Linpol_Freqmean.fits', rms_box=(30, 10), trim_box=(0, 1000,0,1000), use_scipy_fft=True, output_all = True)
#img.write_catalog(outfile=r'Abell85_catalog_BDSF_output_freqmean_rmsbox.reg', catalog_type='gaul', format='ds9', clobber='True')

#img.export_image(outfile=r'Abell85_ch0_BDSF_freqmean.fits', img_type='ch0', clobber='True')
#img.export_image(outfile=r'Abell85_rms_BDSF_freqmean.fits', img_type='rms', clobber='True')
#img.export_image(outfile=r'Abell85_mean_BDSF_freqmean.fits', img_type='mean', clobber='True')
#img.export_image(outfile=r'Abell85_gaus_model_BDSF_freqmean.fits', img_type='gaus_model', clobber='True')
#img.export_image(outfile=r'Abell85_gaus_resid_BDSF_freqmean.fits', img_type='gaus_resid', clobber='True')

'''WITH MULTIPLE FREQS:'''
'''
directory_univ_mkt = r'/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/'
img_2 = bdsf.process_image('Abell_85_aFix_pol_I_Farcsec_fcube_cor.fits', rms_box=(30, 10), use_scipy_fft=True)
img_2.write_catalog(outfile=r'Abell85_catalog_BDSF_gaul.fits', catalog_type='gaul', format='fits', clobber='True')
img_2.write_catalog(outfile=r'Abell85_catalog_BDSF_gaul.reg', catalog_type='gaul', format='ds9', clobber='True')
img_2.write_catalog(outfile=r'Abell85_catalog_BDSF_rsl.fits', catalog_type='srl', format='fits', clobber='True')
img_2.write_catalog(outfile=r'Abell85_catalog_BDSF_rsl.reg', catalog_type='srl', format='ds9', clobber='True')

img_2.export_image(outfile=r'Abell85_ch0_BDSF.fits', img_type='ch0', clobber='True')
img_2.export_image(outfile=r'Abell85_rms_BDSF.fits', img_type='rms', clobber='True')
img_2.export_image(outfile=r'Abell85_mean_BDSF.fits', img_type='mean', clobber='True')
img_2.export_image(outfile=r'Abell85_gaus_model_BDSF.fits', img_type='gaus_model', clobber='True')
img_2.export_image(outfile=r'Abell85_gaus_resid_BDSF.fits', img_type='gaus_resid', clobber='True')
'''

'''WITH LINPOL:'''
'''
directory_univ_mkt = r'/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/rmsynth/'
img_2 = bdsf.process_image(directory_univ_mkt+'rmsynth_polint.fits', rms_box=(30, 10), frequency=1230917968, thresh_pix=8, use_scipy_fft=True)
img_2.write_catalog(outfile=r'Abell85_Linpol_catalog_BDSF_gaul.fits', catalog_type='gaul', format='fits', clobber='True')
img_2.write_catalog(outfile=r'Abell85_Linpol_catalog_BDSF_gaul.reg', catalog_type='gaul', format='ds9', clobber='True')
img_2.write_catalog(outfile=r'Abell85_Linpol_catalog_BDSF_rsl.fits', catalog_type='srl', format='fits', clobber='True')
img_2.write_catalog(outfile=r'Abell85_Linpol_catalog_BDSF_rsl.reg', catalog_type='srl', format='ds9', clobber='True')

img_2.export_image(outfile=r'Abell85_Linpol_ch0_BDSF.fits', img_type='ch0', clobber='True')
img_2.export_image(outfile=r'Abell85_Linpol_rms_BDSF.fits', img_type='rms', clobber='True')
img_2.export_image(outfile=r'Abell85_Linpol_mean_BDSF.fits', img_type='mean', clobber='True')
img_2.export_image(outfile=r'Abell85_Linpol_gaus_model_BDSF.fits', img_type='gaus_model', clobber='True')
img_2.export_image(outfile=r'Abell85_Linpol_gaus_resid_BDSF.fits', img_type='gaus_resid', clobber='True')
'''

'''WITH Convolve trick:'''
directory_univ_mkt = r'/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/'
img_2 = bdsf.process_image(directory_univ_mkt+'Abell_85_Fits_Data_Mean_Std.fits', frequency=1230917968, use_scipy_fft=True)
img_2.write_catalog(outfile=r'Abell85_Linpol_catalog_BDSF_gaul.fits', catalog_type='gaul', format='fits', clobber='True')
img_2.write_catalog(outfile=r'Abell85_Linpol_catalog_BDSF_gaul.reg', catalog_type='gaul', format='ds9', clobber='True')
img_2.write_catalog(outfile=r'Abell85_Linpol_catalog_BDSF_rsl.fits', catalog_type='srl', format='fits', clobber='True')
img_2.write_catalog(outfile=r'Abell85_Linpol_catalog_BDSF_rsl.reg', catalog_type='srl', format='ds9', clobber='True')

img_2.export_image(outfile=r'Abell85_Conv_ch0_BDSF.fits', img_type='ch0', clobber='True')
img_2.export_image(outfile=r'Abell85_Conv_rms_BDSF.fits', img_type='rms', clobber='True')
img_2.export_image(outfile=r'Abell85_Conv_mean_BDSF.fits', img_type='mean', clobber='True')
img_2.export_image(outfile=r'Abell85_Conv_gaus_model_BDSF.fits', img_type='gaus_model', clobber='True')
img_2.export_image(outfile=r'Abell85_Conv_gaus_resid_BDSF.fits', img_type='gaus_resid', clobber='True')