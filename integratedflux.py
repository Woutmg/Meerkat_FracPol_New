import numpy as np
import os
import sys
from astropy.io import fits
from astropy import wcs
from astropy.table import Table, join, vstack
from astropy.nddata.utils import extract_array
import astropy.units as u
import pyregion
import tqdm 
sys.path.append('/net/lofar4/data1/osinga/phd/year1/PlanckESZ_RM/')
from primary_beam_corrections import scale_by_pb_allchannels

"""
Needs to be run in python
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
        hduflat = flatten(hdu)
        r = pyregion.open(ds9region)
        myfilter = r.get_filter(header=hdu[0].header)
        manualmask = myfilter[i].mask(hdu[0].data[0,0].shape)

        if maskoutside: 
            # Mask everything outside the region. i.e., mask = 1 outside the region
            hdu[0].data[manualmask == False] = 1.0
            hdu[0].data[manualmask == True] = 0.0
        else:
            # Mask everything IN the region. i.e., mask = 1 in the region
            hdu[0].data[manualmask == False] = 0.0
            hdu[0].data[manualmask == True] = 1.0

        data = hdu[0].data

    return np.array(data,dtype=np.bool)

def mask_regions(fitsimage,ds9region,outfilename,maskoutside=True):
    """Adapted from Jort Boxelaar
    Given a fits image 'fitsimage' with a corresponding ds9 region file
    'ds9region', write a mask file of the data to 'outfile' 

    If maskoutside=True, then mask (set to True) everything outside the source
    If maskoutside=False, then mask (set to True) everything that is a source
    
    RETURNS
    DATA -- np.array -- data array but with masking of all sources
    
    """
    with fits.open(fitsimage) as hdu:
        hduflat = flatten(hdu)
        r = pyregion.open(ds9region)

        manualmask = r.get_mask(hdu=hduflat)
        if maskoutside: 
            # Mask everything outside the region. i.e., mask = 1 outside the region
            hdu[0].data[manualmask == False] = 1.0
            hdu[0].data[manualmask == True] = 0.0
        else:
            # Mask everything IN the region. i.e., mask = 1 in the region
            hdu[0].data[manualmask == False] = 0.0
            hdu[0].data[manualmask == True] = 1.0

        print ("Writing mask file to %s"%outfilename)
        hdu.writeto(outfilename,overwrite=True)
        data = hdu[0].data

    return np.array(data,dtype=np.bool)

def extr_array(data, ra, dec, head, hdulist, s = (3/60.)):
    """
    Produces a smaller image from the entire fitsim data array,
    with dimension s x s (degrees) around coordinates ra,dec. 


    Arguments:
    data       -- 2D Data array from (.fits) image that the source is located in. 
    ra,dec     -- Right ascension and declination of the source. Will be center of image
    s          -- Dimension of the cutout image in degrees. Default 3 arcminutes.
    head       -- The header of the (.fits) image that the data array is taken from
    hdulist    -- The hdulist of the (.fits) image that the data array is taken from

    Returns:
    data_array -- Numpy array containing the extracted cutout image.

    """

    datashape = data.shape
    # Parse the WCS keywords in the primary HDU
    wcs1 = wcs.WCS(hdulist[0].header)
    # Some pixel coordinates of interest.
    skycrd = np.array([[ra,dec,0,0]], np.float_)
    # Convert pixel coordinates to world coordinates
    # The second argument is "origin" -- in this case we're declaring we
    # have 1-based (Fortran-like) coordinates.
    pixel = wcs1.all_world2pix(skycrd, 1)
    # Some pixel coordinates of interest.
    x = pixel
    
    
    y = pixel[0][1]
    pixsize = abs(wcs1.wcs.cdelt[0])
    N = (s/pixsize)
    # print 'x=%.5f, y=%.5f, N=%i' %(x,y,N)
    ximgsize = head.get('NAXIS1')
    yimgsize = head.get('NAXIS2')
    if x ==0:
        x = ximgsize/2
    if y ==0:
        y = yimgsize/2
    offcentre = False
    # subimage limits: check if runs over edges
    xlim1 =  x - (N/2)
    if(xlim1<1):
        xlim1=1
        offcentre=True
    xlim2 =  x + (N/2)
    if(xlim2>ximgsize):
        xlim2=ximgsize
        offcentre=True
    ylim1 =  y - (N/2)
    if(ylim1<1):
        ylim1=1
        offcentre=True
    ylim2 =  y + (N/2)
    if(ylim2>yimgsize):
        offcentre=True
        ylim2=yimgsize

    xl = int(xlim1)
    yl = int(ylim1)
    xu = int(xlim2)
    yu = int(ylim2)


    # extract the data array instead of making a postage stamp
    if len(datashape) == 2:
        data_array = extract_array(data,(yu-yl,xu-xl),(y,x))
    
    elif len(datashape) == 3:
        if int(datashape[0]/2) != datashape[0]/2:
            raise ValueError("Uneven amount of pixels in 3rd axis not yet implemented")
        # Don't touch the first axis
        data_array = extract_array(data,(datashape[0],yu-yl,xu-xl),(int(datashape[0]/2),y,x))

    elif len(datashape) == 4:
        if int(datashape[1]/2) != datashape[1]/2:
            raise ValueError("Uneven amount of pixels in 3rd axis not yet implemented")
        # Don't touch the first two axes
        data_array = extract_array(data,(datashape[0],datashape[1],yu-yl,xu-xl),(0,int(datashape[0]/2),y,x))

    return data_array

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
     
def integratedflux(fitsimage, maskarray, tdata, i, hdul=None):
    """
    Given a 2D image in Jy/beam, with a .ds9 regionfile indicating the sources
    and a .fits table with the PyBDSF source parameters (RA,DEC,MAJ,MIN)
    calculate the integrated flux of source i

    INPUTS
    fitsimage  -- str    -- Location of the fits image containing the sources
    maskarray  -- array  -- False where there is a source, True where not. See mask_region()
    tdata      -- table  -- PYBDSF table with the source properties
    i          -- int    -- Which source in the PYBDSF table to extract
    hdul       -- str    -- Optional. If we want to call this function many times
                            with the same fitscube, then it's better to open the
                            hdul in advance so we don't have to keep loading into memory

    RETURNS
    totalflux  -- float  -- the (approximately) integrated flux of the source in Jy
    """

    RA, DEC, Maj = tdata[i]['RA'], tdata[i]['DEC'], tdata['Maj'][i]
    s = 4*Maj
    # TODO: check if any other source is in vicinity

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

    # Now convert the units from Jy/beam to Jy/pix
    masked_data = convert_units(masked_data, hdul)

    # Easier to give 2D data array to this function
    subdata = extr_array(masked_data[0,0], RA, DEC, head, hdul, s)

    # The total flux of the source is then the sum of the pixels
    totalflux = subdata.sum()

    if closehdul: hdul.close()

    return totalflux.value # in Jy

def integratedflux_perchannel(fitscube, maskarray, tdata, i, hdul=None):
    """
    Similar to the previous function, but for a fits cube with 3rd axis FREQ

    INPUTS
    fitscube   -- str    -- Location of the fits cube containing the sources
    maskarray  -- array  -- False where there is a source, True where not. See mask_region()
    tdata      -- table  -- PYBDSF table with the source properties
    i          -- int    -- Which source in the PYBDSF table to extract
    hdul       -- str    -- Optional. If we want to call this function many times
                            with the same fitscube, then it's better to open the
                            hdul in advance so we don't have to keep loading into memory
    
    RETURNS
    totalflux  -- float  -- the (approximately) integrated flux of the source in Jy
    """

    RA, DEC, Maj = tdata[i]['RA'], tdata[i]['DEC'], tdata['Maj'][i]
    s = 4*Maj
    
    # TODO: check if any other source is in vicinity

    if hdul is None:
        closehdul = True
        hdul = fits.open(fitscube)
    else:
        closehdul = False

    head = hdul[0].header

    # First mask (set to zero) all pixels that are not a source
    data = hdul[0].data 
    masked_data = np.copy(data) # (1,90,2736,2736)
    # Because masked_data is now 3D, maskarray should be 3D as well
    mask3d = np.zeros(masked_data.shape,dtype=np.bool)
    mask3d[:,:,:,:] = maskarray # (1,90,2736,2736)
    
    masked_data[mask3d] = 0

    # Now convert the units from Jy/beam to Jy/pix
    masked_data = convert_units(masked_data, hdul)

    # Easier to give 3D data array to this function
    subdata = extr_array(masked_data[0], RA, DEC, head, hdul, s)
    # print (subdata.shape) ## subdata is now (90,30,30) or so

    # The total flux of the source is then the sum of the pixels over the last two axes
    totalflux = subdata.sum(axis=(1,2)) # shape (90,)

    if closehdul: hdul.close()

    return totalflux.value # 90 channels with the flux in Jy 

def uncertainty_perchannel(directory, sourcename, stokes, maskarray, tdata, i, wcs, hdul):
    """
    For given Stokes parameter 'I','Q' or 'U' calculate the uncertainty on the 
    integrated flux.

    The uncertainty is defined as sigma_f = rms*sqrt(Nbeams) where Nbeams is 
    the number of beams that cover the source.

    First we open the RMS noise in the center of the field, then calculate 
    the PB corrected RMS noise at the location of the source and multiply this by
    Nbeams.

    stokes     -- string      -- either 'I', 'Q' or 'U'
    maskarray  -- array       -- False where there is a source, True where not. See mask_region()
    tdata      -- table       -- contains sources
    i          -- integer     -- which source to do
    wcs        -- astropy.wcs -- wcs of the stokes image containing the source
    hdul       -- HDUList     -- HDUList of the Stokes image

    RETURNS
    noise   -- uncertainty on the flux, defined as sigma_f = rms*sqrt(Nbeams)
    rms_pix -- rms value at this particular pixel
    """
    if stokes not in ['I','Q','U']:
        raise ValueError("Not implemented Stokes %s"%stokes)

    # Noise in center of field, computed earlier
    noise = './%s/%s/all_noise%s.npy'%(directory,sourcename,stokes)
    noise = np.load(noise)

    # Remove failed channels
    failedchannels = np.array(np.load('./%s/%s/failedchannels.npy'%(directory,sourcename)),dtype='int')
    noise = np.delete(noise,failedchannels)

    # Find value of PB at location of the source, for all channels
    RA, DEC, Maj = tdata[i]['RA'], tdata[i]['DEC'], tdata['Maj'][i]
    size = hdul[0].data.shape[-1]
    all_scales = scale_by_pb_allchannels(size,wcs,RA,DEC)
    # Remove failed channels
    all_scales = np.delete(all_scales,failedchannels)

    # Scale the noise in center of field by the pb
    noise /= all_scales

    # Find how many beams this source covers
    header = hdul[0].header 
    s = 4*Maj
    subdata = extr_array(maskarray, RA, DEC, header, hdul, s)
    Npix = np.sum(np.invert(subdata))

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

    Nbeams = (Npix/beam2pix).value
    rms_pix = noise
    noise *= np.sqrt(Nbeams)

    return noise, rms_pix

def all_integrated_fluxes(sourcename, directory, fitsimage, regionfile, tdata, sourcefluxdir):
    """
    Calculates the Stokes IQU fluxes for all sources in the Table 'tdata'
    Writes the result to 'sourcefluxdir'

    fitsimage     -- str   -- Location of the fits image containing the sources
    regionfile    -- str   -- Location with the regionfile containing the sources
    tdata         -- table -- with matches sources from Stokes I and polint
    sourcefluxdir -- str   -- where to write the results to

    Basically a for loop around integratedflux_perchannel()
    """

    print ("Calculating the integrated IQU fluxes for all %i sources"%len(tdata))
    print ("Saving results in %s"%sourcefluxdir)

    # Stokes I flux per channel. For every source
    fitscube = './%s/%s/Idatacube_%s.fits'%(directory,sourcename,sourcename)
    with fits.open(fitscube) as hdul:
        wcs1 = wcs.WCS(hdul[0].header)
        for i in tqdm.tqdm(range(len(tdata)),desc="Calculating Stokes I flux"):
            # Array masking Source i
            maskarray = mask_source(fitsimage, regionfile, i, maskoutside=True)
            # Use that array to find integrated flux
            Iflux = integratedflux_perchannel(fitscube, maskarray, tdata, i, hdul)
            Iunc, Irms = uncertainty_perchannel(directory, sourcename, 'I',maskarray,tdata, i, wcs1, hdul)
            np.save(sourcefluxdir+'StokesI_source_%i.npy'%i,Iflux)
            np.save(sourcefluxdir+'StokesI_unc_source_%i.npy'%i,Iunc)
            np.save(sourcefluxdir+'StokesI_pixrms_source_%i.npy'%i,Irms)

    # Stokes Q flux per channel. For every source
    fitscube = './%s/%s/Qdatacube_%s.fits'%(directory,sourcename,sourcename)
    with fits.open(fitscube) as hdul:
        wcs1 = wcs.WCS(hdul[0].header)
        for i in tqdm.tqdm(range(len(tdata)),desc="Calculating Stokes Q flux"):
            # Array masking Source i
            maskarray = mask_source(fitsimage, regionfile, i, maskoutside=True)
            # Use that array to find integrated flux            
            Qflux = integratedflux_perchannel(fitscube, maskarray, tdata, i, hdul)
            Qunc, Qrms = uncertainty_perchannel(directory, sourcename, 'Q',maskarray,tdata, i, wcs1, hdul)
            np.save(sourcefluxdir+'StokesQ_source_%i.npy'%i,Qflux)
            np.save(sourcefluxdir+'StokesQ_unc_source_%i.npy'%i,Qunc)
            np.save(sourcefluxdir+'StokesQ_pixrms_source_%i.npy'%i,Qrms)

    # Stokes U flux per channel. For every source
    fitscube = './%s/%s/Udatacube_%s.fits'%(directory,sourcename,sourcename)
    with fits.open(fitscube) as hdul:
        wcs1 = wcs.WCS(hdul[0].header)
        for i in tqdm.tqdm(range(len(tdata)),desc="Calculating Stokes U flux"):
            # Array masking Source i
            maskarray = mask_source(fitsimage, regionfile, i, maskoutside=True)
            # Use that array to find integrated flux            
            Uflux = integratedflux_perchannel(fitscube, maskarray, tdata, i, hdul)
            Uunc, Urms = uncertainty_perchannel(directory, sourcename, 'U',maskarray,tdata, i, wcs1, hdul)
            np.save(sourcefluxdir+'StokesU_source_%i.npy'%i,Uflux)
            np.save(sourcefluxdir+'StokesU_unc_source_%i.npy'%i,Uunc)
            np.save(sourcefluxdir+'StokesU_pixrms_source_%i.npy'%i,Urms)

    return

if __name__ == '__main__':
    
    # Call this script with .e.g., python integratedflux.py 81019 G134.73+48.89
    try:
        directory, sourcename, which = int(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])
    except:
        print("ERROR: Please call this script with arguments 'directory', 'sourcename' and 'pol'/'unpol' ")
        raise


    if which == 'pol':
        sourcefluxdir = './%s/%s/sourcefluxes/'%(directory,sourcename)
        if not os.path.exists(sourcefluxdir):
            print ("Creating directory %s"%sourcefluxdir + "to store the Stokes IQU flux per polarised source")
            os.mkdir(sourcefluxdir)
        
        # Matched polint + stokes I sources region file and catalogue
        regionfile = './%s/%s/rmsynth/rmsynth_polint.regrid.pb-decor.pybdsm.srl.2.matched.reg'%(directory,sourcename)    
        tdata = './%s/%s/rmsynth/rmsynth_polint.regrid.pb-decor.pybdsm.srl.2.matched.fits'%(directory,sourcename)
        tdata = Table.read(tdata)
        # Need this image for header information
        fitsimage = './%s/%s/rmsynth/rmsynth_polint.regrid.fits'%(directory,sourcename)
        
        # Where to save the mask file
        maskfile = './%s/%s/rmsynth/rmsynth_polint.srl.2.matched.mask.fits'%(directory,sourcename)
        # 'mask' the source such that data[mask] selects all pixels inside the sources
        # This maskarray is not actually being used anymore by this code. But is used for QU fitting map
        # For integrated flux it is better to mask per source.
        maskarray = mask_regions(fitsimage,regionfile,maskfile,maskoutside=False) 

        # Calculate the integrated flux and uncertainty per source and save it to sourcefluxdir
        all_integrated_fluxes(sourcename, directory, fitsimage, regionfile, tdata, sourcefluxdir)

    elif which == 'unpol':
        sourcefluxdir = './%s/%s/sourcefluxes_unpol/'%(directory,sourcename)
        if not os.path.exists(sourcefluxdir):
            print ("Creating directory %s"%sourcefluxdir + "to store the Stokes IQU flux per unpolarised source")
            os.mkdir(sourcefluxdir)

        # stokes I sources region file and catalogue
        regionfile = './%s/%s/rmsynth/sourcelistStokesI.srl.reg'%(directory,sourcename)    
        tdata = './%s/%s/rmsynth/sourcelistStokesI.srl.fits'%(directory,sourcename)
        tdata = Table.read(tdata)
        # Need this image for header information
        fitsimage = './%s/%s/rmsynth/rmsynth_polint.regrid.fits'%(directory,sourcename)
        # Where to save the mask file
        maskfile = './%s/%s/rmsynth/sourcelistStokesI.mask.fits'%(directory,sourcename)
        # 'mask' the source such that data[mask] selects all pixels inside the sources
        # This maskarray is not actually being used anymore by this code. But is used for QU fitting map
        # For integrated flux it is better to mask per source.
        maskarray = mask_regions(fitsimage,regionfile,maskfile,maskoutside=False)         
        # Calculate the integrated flux and uncertainty per source and save it to sourcefluxdir
        all_integrated_fluxes(sourcename, directory, fitsimage, regionfile, tdata, sourcefluxdir)
        

    else:
        print("ERROR: Please call this script with third argument either 'pol' or 'unpol'")




    """
    ### Calculate the integrated flux of one source
    ### Test with polint img
    # totalflux = integratedflux(fitsimage, maskarray, tdata, 0)
    # print ("Total flux of source %i is %.2f mJy"%(i,totalflux*1000))
    """



