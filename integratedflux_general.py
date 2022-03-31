import numpy as np
import os
import sys
from astropy.io import fits
from astropy.table import Table, join, vstack
from astropy import wcs
from astropy.nddata.utils import extract_array
import astropy.units as u
import pyregion
import tqdm 
import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

"""
@Author: Erik Osinga

Calculate integrated radio flux of a source. 
Using PYBDSF region file or manually defined region file. 

To make a manually defined region file: 
1. Go to casaviewer
2. Draw a region.
3. Go to regions -> file   (have to have enabled view->regions)
4. Set file format: ds9, coordinate system: J2000 !!!
5. Choose filename and click 'save now'

"""

def flatten(f):
    """
    Flatten a fits file so that it becomes a 2D image.
    Return new header and data.

    Taken from Jort Boxelaar
    """

    naxis=f[0].header['NAXIS']
    if naxis<2:
        raise RadioError('Can\'t make map from this')
    if naxis == 2:
        return fits.PrimaryHDU(header=f[0].header,data=f[0].data)

    w  = wcs.WCS(f[0].header)
    wn = wcs.WCS(naxis=2)

    wn.wcs.crpix[0]=w.wcs.crpix[0]
    wn.wcs.crpix[1]=w.wcs.crpix[1]
    wn.wcs.cdelt=w.wcs.cdelt[0:2]
    wn.wcs.crval=w.wcs.crval[0:2]
    wn.wcs.ctype[0]=w.wcs.ctype[0]
    wn.wcs.ctype[1]=w.wcs.ctype[1]

    header = wn.to_header()
    header["NAXIS"]=2
    copy=('EQUINOX','EPOCH','BMAJ', 'BMIN', 'BPA', 'RESTFRQ', 'TELESCOP', 'OBSERVER')
    for k in copy:
        r=f[0].header.get(k)
        if r is not None:
            header[k]=r

    slice=[]
    for i in range(naxis,0,-1):
        if i<=2:
            slice.append(np.s_[:],)
        else:
            slice.append(0)

    hdu = fits.PrimaryHDU(header=header,data=f[0].data[tuple(slice)])
    return hdu

def mask_source(fitsimage, ds9region, i, maskoutside=True):
    """Adapted from Jort Boxelaar

    Given a fits image 'fitsimage' with a corresponding ds9 region file
    'ds9region', return a mask that masks ONLY source with index 'i'
    
    If maskoutside=True, then mask (set to True) everything outside the source
    If maskoutside=False, then mask (set to True) everything that is a source

    RETURNS
    DATA -- np.array -- data array but with masking of source i
    """
    with fits.open(fitsimage) as hdu:
        print('a')
        hduflat = flatten(hdu)
        print('b')
        r = pyregion.open(ds9region)
        print('c')
        myfilter = r.get_filter(header=hdu[0].header)
        print('d')
        imshape = hdu[0].data.shape
        if len(imshape) == 2: # data only has two axes, X and Y
            print("Two axes detected")
            manualmask = myfilter[i].mask(hdu[0].data)
        elif len(imshape) == 4: # data probably also has FREQ and STOKES axis
            print ("Image shape: %s. ASSUMING FREQ and STOKES axis are the first two axes. PLEASE CONFIRM"%(str(imshape)))
            manualmask = myfilter[i].mask(hdu[0].data[0,0].shape)
        else:
            raise ValueError("data shape %s not implemented"%(str(hdu[0].data.shape)))

        if maskoutside: 
            outside = 1.0 
            inside = 0.0    

        if len(imshape) == 2: # data only has two axes, X and Y
            hdu[0].data[manualmask == False] = outside
            hdu[0].data[manualmask == True] = inside

        elif len(imshape) == 4: # data probably also has FREQ and STOKES axis
            # Mask everything outside the region 1.0 or 0.0 depending on maskoutside=True or False
            hdu[0].data[0][0][manualmask == False] = outside
            hdu[0].data[0][0][manualmask == True] = inside

        data = hdu[0].data

    return np.array(data,dtype=np.bool)

def convert_units(data, fitsimage):
    """
    Convert the units of 'data' array which is assumed to be Jy/beam 
    to Jy/pix using the beam information given in the header of 'fitsimage'

    data      -- np.array -- data from fitsimage with units Jy/beam to be converted
    fitsimage -- str      -- location of fitsimage with beam information in header
                 or HDUL  -- In that case it's assumed already opened HDUL

    Returns
    data -- np.array -- data from fitsimage with new units Jy/pix
    """

    if type(fitsimage) == str:
        with fits.open(fitsimage) as hdul:
            header = hdul[0].header 
            if header['BUNIT']=='JY/BEAM' or header['BUNIT']=='Jy/beam':
                # BEAM AND PIXEL INFORMATION
                bmaj      = header['BMIN']*u.deg
                bmin      = header['BMAJ']*u.deg
                bpa       = header['BPA']*u.deg
                pix_size  = abs(header['CDELT2'])*u.deg # assume square pix size

                beammaj = bmaj/(2.*(2.*np.log(2.))**0.5) # Convert to sigma
                beammin = bmin/(2.*(2.*np.log(2.))**0.5) # Convert to sigma
                pix_area  = abs(header['CDELT1']*header['CDELT2'])*u.deg*u.deg
                beam_area = 2.*np.pi*1.0*beammaj*beammin # beam area in 
                beam2pix  = beam_area/pix_area # beam area in pixels
            else:
                raise ValueError("UNITS ARE NOT Jy/beam PLEASE CHECK HEADER.")
    else:
        hdul = fitsimage
        header = hdul[0].header 
        if header['BUNIT']=='JY/BEAM' or header['BUNIT']=='Jy/beam':
            # BEAM AND PIXEL INFORMATION
            bmaj      = header['BMIN']*u.deg
            bmin      = header['BMAJ']*u.deg
            bpa       = header['BPA']*u.deg
            pix_size  = abs(header['CDELT2'])*u.deg # assume square pix size

            beammaj = bmaj/(2.*(2.*np.log(2.))**0.5) # Convert to sigma
            beammin = bmin/(2.*(2.*np.log(2.))**0.5) # Convert to sigma
            pix_area  = abs(header['CDELT1']*header['CDELT2'])*u.deg*u.deg
            beam_area = 2.*np.pi*1.0*beammaj*beammin # beam area in 
            beam2pix  = beam_area/pix_area # beam area in pixels
        else:
            raise ValueError("UNITS ARE NOT Jy/beam PLEASE CHECK HEADER.")

    data = data/beam2pix # convert to Jy/pix
    return data
     
def integratedflux(fitsimage, maskarray, hdul=None, test=False):
    """
    Given a 2D image in Jy/beam, with a .ds9 regionfile indicating the sources
    and a .fits table with the PyBDSF source parameters (RA,DEC,MAJ,MIN)
    calculate the integrated flux of source i

    INPUTS
    fitsimage  -- str    -- Location of the fits image containing the sources
    maskarray  -- array  -- False where there is a source, True where not. See mask_region()
    hdul       -- str    -- Optional. If we want to call this function many times
                            with the same fitscube, then it's better to open the
                            hdul in advance so we don't have to keep loading into memory

    RETURNS
    totalflux  -- float  -- the integrated flux of the source in Jy
    Nbeams     -- float  -- the number of beams the source covers
    """

    if hdul is None:
        closehdul = True
        hdul = fits.open(fitsimage)
    else:
        closehdul = False

    head = hdul[0].header

    # First mask (set to zero) all pixels that are not a source
    data = hdul[0].data 
    masked_data = np.copy(data)
    masked_data[maskarray] = 0

    if test:
        import matplotlib.pyplot as plt
        if len(masked_data.shape) == 4:
            md = masked_data[0][0] # assume first two axies are FREQ and STOKES
            d = data[0][0]
        else:
            md = masked_data
            d = data
        fig, axes = plt.subplots(1,2,sharex=True,sharey=True)
        axes[0].imshow(md,vmin=-1*np.std(md),vmax=10*np.std(md),origin='lower')
        axes[0].set_title("Showing region that is integrated over")
        axes[1].imshow(d,vmin=-1*np.std(md),vmax=10*np.std(md),origin='lower')
        axes[1].set_title("Showing full image")
        plt.show()

    # Now convert the units from Jy/beam to Jy/pix
    masked_data = convert_units(masked_data, hdul)

    # The total flux of the source is then the sum of the pixels
    totalflux = masked_data.sum()


    # Also find how much beams the source covers 

    # BEAM AND PIXEL INFORMATION
    bmaj      = head['BMIN']*u.deg
    bmin      = head['BMAJ']*u.deg
    bpa       = head['BPA']*u.deg
    pix_size  = abs(head['CDELT2'])*u.deg # assume square pix size

    beammaj = bmaj/(2.*(2.*np.log(2.))**0.5) # Convert to sigma
    beammin = bmin/(2.*(2.*np.log(2.))**0.5) # Convert to sigma
    pix_area  = abs(head['CDELT1']*head['CDELT2'])*u.deg*u.deg
    beam_area = 2.*np.pi*1.0*beammaj*beammin # beam area in 
    beam2pix  = beam_area/pix_area # beam area in pixels

    # Find how many pixels this source covers 
    Npix = np.sum(np.invert(maskarray))
    #Nbeams =  Npix / (beam size in pixels)
    Nbeams = (Npix/beam2pix).value
    
    if closehdul: hdul.close()

    return totalflux.value, Nbeams # flux given in Jy

def uncertainty_flux(fitsimage, flux, Nbeams, maskarray_empty=None, rms=None, delta_cal=0.1, hdul=None
    ,verbose=True):
    """
    Calculate the uncertainty on the integrated flux.

    The uncertainty is defined as (Cassano+2013) Equation 1

        sigma_f = sqrt[ (rms*sqrt(Nbeams))**2 + (delta_cal*flux)**2 ] 

    where Nbeams is the number of beams that cover the source.


    INPUTS
    fitsimage        -- str    -- Location of the fits image containing the sources
    flux             -- float  -- flux of the source calculated by integratedflux()
    Nbeams           -- float  -- returned by integratedflux(), amount of beams the source covers
    maskarray_empty  -- array  -- Made by mask_region(). Should contain no sources. Used to calculate rms
    rms              -- float  -- or just give the rms value manually in Jy/beam
    delta_cal        -- float  -- uncertainty on flux scale. 10% or 20% makes sense for LOFAR.
    hdul             -- str    -- Optional. If we want to call this function many times
                                  with the same fitscube, then it's better to open the
                                  hdul in advance so we don't have to keep loading into memory

    RETURNS
    uncertainty   -- uncertainty on the flux
    """
    if maskarray_empty is None and rms is None:
        raise ValueError("Please give either a region to calculate the rms or a value for the rms in Jy/beam")

    if hdul is None:
        closehdul = True
        hdul = fits.open(fitsimage)
    else:
        closehdul = False

    header = hdul[0].header

    if rms is None:
        # First mask (set to zero) all pixels outside the region
        data = hdul[0].data 
        masked_data = np.copy(data)
        masked_data[maskarray_empty] = 0

        # Find how many pixels this emprty region covers 
        Npix = np.nansum(np.invert(maskarray_empty))
        # Calculate the rms noise in this region
        rmsnoise = np.sqrt((1./Npix)*(masked_data**2).nansum())
        if verbose: print ("rms noise in given region: %.2f \muJy/beam"%(rmsnoise*1e6))

    else:
        rmsnoise = rms
        if verbose: print ("Using manual rms value  %.2f \muJy/beam"%(rmsnoise*1e6))

    # Now we have all information. Calculate the equation for the uncertainty
    fluxscale = delta_cal*flux
    rmsterm = rmsnoise*np.sqrt(Nbeams)
    uncertainty = np.sqrt(fluxscale**2 + rmsterm**2)

    return uncertainty

if __name__ == '__main__':
    
    # Use PYBDSF region file or manually defined region file. 
    #### in casaviewer: Save region file as 'DS9 region file' in J2000 coordinates. 

    parser = argparse.ArgumentParser(
        description="""Calculate integrated radio flux of a source. """)


    parser.add_argument('-tr', '--region', help='target .reg file. Can contain multiple regions'
        , type=str, required=True)
    parser.add_argument('-i', '--regionindex', help='Which index in the target .reg file to use'
        , type=int, required=True)
    parser.add_argument('-nr', '--rmsregion', help='.reg file to calculate RMS. Should contain only 1 region'
        , type=str, required=False, default=None)
    parser.add_argument('-f', '--fitsim', help='ds9 .fits radio image'
        , type=str, required=True)

    args = vars(parser.parse_args())

    regionfile = args['region']
    regionfile_rms = args['rmsregion']
    fitsimage = args['fitsim']
    i = args['regionindex']

    # Array masking Source i=0 in the region file
    maskarray = mask_source(fitsimage, regionfile, i, maskoutside=True)

    ## Calculate the integrated flux (in Janksy). Use the array masking the source
    flux, Nbeams = integratedflux(fitsimage, maskarray, hdul=None)

    if regionfile_rms is not None:
        # Array masking empty region i=0 in the region file
        maskarray_rms = mask_source(fitsimage, regionfile_rms, 0, maskoutside=True)
        ## Calculate the uncertainty. Use the array masking the empty region
        uncertainty = uncertainty_flux(fitsimage, flux, Nbeams, maskarray_rms, None, hdul=None)

        print ("Source %i has flux %.2f +/- %.2f mJy"%(i,flux*1e3,uncertainty*1e3))
    else:
        print ("Source %i has flux %.2f mJy. No RMS region given"%(i,flux*1e3))
