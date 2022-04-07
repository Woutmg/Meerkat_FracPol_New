import numpy as np
import os
import glob
import subprocess
import sys
import time
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from matplotlib.patches import Ellipse
import tqdm

"""
Analyze polarization data. See also rmsynth_G280.19+47.81.py

meant to be run WITH CASA in
/net/lofar4/data1/osinga/pipeline

or
/net/voorrijn/data2/osinga/pipeline/cubes

using casa

"""

if __name__ == '__main__':
    plt.ioff() # To not show plots immediately
    c = 299792458 # m / s
    nchan = 12
    print ("Assuming %i channels"%nchan)


# create linear polarization angle images: chi = 0.5 arctan(Q/U)
# polithresh set to 5 sigma of the 0000-Q-image.fits


def deltaLambdasquared(deltanu,nu_central):
    """
    Channel width in wavelength squared (units of m^2)
    Assuming a tophat channel bandpass (which is fine for deltanu<<nu_central).


    Inputs:
    deltanu    -- float -- channel width in Hertz (usually 8MHz)
    nu_central -- float -- channel central frequency in Hertz

    Returns:
    deltaLambdasquared -- float -- channel width in wavelength squared (m^2)

    """
    return 2*c**2*deltanu/(nu_central**3)*(1+0.5*(deltanu/nu_central)**2)

def phimax(dlambdasq):
    """
    The maximum Faraday depth to which one has more than 50% sensitivity

    Given in rad/m^2 (if dlambdasq is given in m^2)
    """

    return np.sqrt(3)/dlambdasq

def DeltaLambdasquared(nu_min,nu_max):
    """
    Not to be confused with deltaLambdaSquared
    This is the total bandwidth in wavelength squared, i.e.,
    lambda_max^2 - lambda_min^2

    Inputs
    nu_min -- float -- minimum frequency in Hz
    nu_max -- float -- maximum frequency in Hz

    Returns
    DeltaLambdasquared -- float -- Total bandwidth in wavelength squared (m^2)

    """
    c = 299792458 # m / s

    lambda_max = c/nu_min
    lambda_min = c/nu_max

    return lambda_max**2 - lambda_min**2

def FWHM_RMTF(Dlambdasq):
    """
    Full width halfmax of the RM transfer function 
    (i.e., the resolution in phi-space)

    Input
    Dlambdasq -- float -- total bandwidth in wavelength squared

    Returns
    """
    return 2*np.sqrt(3)/Dlambdasq

def max_scale(nu_max):
    """
    Largest scale in phi space to which we are sensitive is given by the shortest
    wavelength or highest frequency. 
    More precisely, this is the scale where the sensitivity has dropped to 50%

    nu_max -- float -- highest freq given in Hz

    """
    lambda_min = c/nu_max
    return np.pi/lambda_min**2

def plot_rms_vs_spw(title, savefig):
    """
    Determine RMS noise as the RMS of the entire (large) Q or U image.

    Calculate rms per spw.
    """
    all_noiseQ = np.empty(16)
    all_noiseU = np.empty(16)
    
    imagesQ = []
    imagesU = []
    exp_avg = '(IM0+IM1+IM2+IM3+IM4+IM5)/6'
    # Loop over all images, combine every 6 channels into one spw. 
    for channum in tqdm.tqdm(range(0,nchan),desc="Calculating rms"):
        # 6 channels per spw in my averaging setup
        if channum%6 == 0 and channum != 0:
            # Average all images in the same spw
            spw = int(channum/6-1)
            immath(outfile=('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/'
                            +'averaged_images/averaged_Q_spw%i'%spw)
                    ,mode='evalexpr'
                    ,imagename=imagesQ
                    ,expr=exp_avg)
            immath(outfile=('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/'
                            +'averaged_images/averaged_U_spw%i'%spw)
                    ,mode='evalexpr'
                    ,imagename=imagesU
                    ,expr=exp_avg)

            all_noiseQ[spw] = imstat('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/'
                                      +'averaged_images/averaged_Q_spw%i'%spw)['rms']
            all_noiseU[spw] = imstat('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/'
                                      +'averaged_images/averaged_U_spw%i'%spw)['rms']

            imagesQ = []
            imagesU = []

        imnameQ = ('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/Q_slices/'
                  +'Abell_85_Q_plane_freq'+(2-len(str((channum+1))))*'0'+str(channum+1)+'.fits')
        imnameU = ('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/U_slices/'
                  +'Abell_85_U_plane_freq'+(2-len(str((channum+1))))*'0'+str(channum+1)+'.fits')
        imagesQ.append(imnameQ)
        imagesU.append(imnameU)


    # Don't forget the last one.
    channum += 1
    spw = int(channum/6-1)
    immath(outfile=('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/'
                    +'averaged_images/averaged_Q_spw%i'%spw)
            ,mode='evalexpr'
            ,imagename=imagesQ
            ,expr=exp_avg)
    immath(outfile=('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/'
                    +'averaged_images/averaged_U_spw%i'%spw)
            ,mode='evalexpr'
            ,imagename=imagesU
            ,expr=exp_avg)
    all_noiseQ[spw] = imstat('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/'
                              +'averaged_images/averaged_Q_spw%i'%spw)['rms']
    all_noiseU[spw] = imstat('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/'
                              +'averaged_images/averaged_U_spw%i'%spw)['rms']
    all_spw = np.arange(16)

    plt.title(title)
    plt.scatter(all_spw,all_noiseQ*1e6,label='RMS noise Q',alpha=0.5)
    plt.scatter(all_spw,all_noiseU*1e6,label='RMS noise U',alpha=0.5,color='r')
    plt.xlabel("Spectral window index")
    plt.ylabel("RMS in entire image ($\mu$Jy/beam)")
    plt.legend(loc='lower left')
    plt.savefig(savefig)
    plt.show()
    plt.close()

def plot_rms_vs_channel(title, savefig):
    """
    Determine RMS noise as the central RMS I, Q or U image.

    Calculate rms per channel. If RMS > 2mJy define a channel as failed.
    """
    all_chan = np.arange(nchan)

    all_noiseQ = np.empty(nchan) #(90)
    all_noiseU = np.empty(nchan) #(90)
    all_noiseI = np.empty(nchan) #(90)
    
    # Use hinges-fences with fence=1.0 so we cut the highest and lowest values
    # from the img
    box = '1608,1608,1608,1608' # Calculate central rms, changed by WoutG to mkt images
    # algorithm = 'classic'
    algorithm = 'hinges-fences'

    for channum in tqdm.tqdm(range(0,nchan),desc='Calculating rms per channel'):
        rmsQ = imstat('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/Q_slices/Abell_85_Q_plane_freq'+(2-len(str((channum+1))))*'0'\
                      +str(channum+1)+'.fits',algorithm=algorithm,fence=1.0,box=box)['rms'][0]
        rmsU = imstat('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/U_slices/Abell_85_U_plane_freq'+(2-len(str((channum+1))))*'0'\
                      +str(channum+1)+'.fits',algorithm=algorithm,fence=1.0,box=box)['rms'][0]
        rmsI = imstat('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/I_slices/Abell_85_I_plane_freq'+(2-len(str((channum+1))))*'0'\
                      +str(channum+1)+'.fits',algorithm=algorithm,fence=1.0,box=box)['rms'][0]

        all_noiseQ[channum] = rmsQ
        all_noiseU[channum] = rmsU
        all_noiseI[channum] = rmsI

    # plt.scatter(all_chan,all_noiseQ*1e6,label='{}'.format(sourcename),alpha=0.5)
    plt.scatter(all_chan,all_noiseQ*1e6,alpha=0.5,label='Stokes Q',c='#1f77b4')
    plt.scatter(all_chan,all_noiseU*1e6,alpha=0.5,marker="^",label='Stokes U',c='#1f77b4')
    plt.scatter(all_chan,all_noiseI*1e6,alpha=0.5,marker="s",label='Stokes I',c='#ff7f0e')
    plt.axhline(np.nanmedian(all_noiseI)*1e6,ls='dashed',c='k',label='Stokes I median noise')

    np.save('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/all_noiseQ.npy',all_noiseQ)
    np.save('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/all_noiseU.npy',all_noiseU)
    np.save('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/all_noiseI.npy',all_noiseI)


    plt.xlabel("Channel index")
    plt.title(title)
    plt.ylabel("Central RMS ($\mu$Jy/beam)")
    plt.legend(loc='upper left',frameon=False)
    plt.savefig(savefig)
    plt.show()
    plt.close()

def create_inverse_variance_weights():
    """
    Given the rms noise levels that were just calculated. Translate these
    to weights.
    """
    all_noiseQ = np.load('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/all_noiseQ.npy')
    all_noiseU = np.load('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/all_noiseU.npy')
    # Remove failed channels if any
    nanmask = np.invert(np.isnan(all_noiseQ))
    
    all_noiseQ = all_noiseQ[nanmask]
    all_noiseU = all_noiseU[nanmask]

    averagerms = (all_noiseQ + all_noiseU)/2.

    weights = (1/averagerms)**2 # INVERSE VARIANCE WEIGHING

    # Normalize so that they sum to 1
    weights /= np.sum(weights)

    # save the weights to a file
    weightfile = open('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/inv_var_weights.txt','w')
    for i, weight in enumerate(weights):
        weightfile.write(str(weight))
        if i != len(weights)-1:
            weightfile.write('\n')
    weightfile.close()

def plotRMTF(rmtfile, savefig=None, show=False):
    rmtf = np.loadtxt(rmtfile,delimiter=' ')

    plt.plot(rmtf[:,0],rmtf[:,1],label='Real Part',ls='dashed')
    plt.plot(rmtf[:,0],rmtf[:,2],label='Imaginary Part',ls='dashed')
    plt.plot(rmtf[:,0],np.sqrt(rmtf[:,1]**2+rmtf[:,2]**2),label='Amplitude')
    plt.legend()
    if savefig is not None: plt.savefig(savefig)
    if show: plt.show()
    plt.close()

def fix_rmsynth_polint_header():
    """
    For PYBDSF to accept the file, need at least 3 axis where the third is freq
    """
    rmpolint = '/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/rmsynth/rmsynth_polint.fits'

    with fits.open(rmpolint,mode='update') as hdul:
        head = hdul[0].header
        head['NAXIS'] = 3
        head['NAXIS3'] = 1
        head['CTYPE3'] = 'FREQ'
        head['CRVAL3'] = 150e6
        head['CDELT3'] = 8e6
        head['CRPIX3'] = 1
        head['CUNIT3'] = 'Hz'

        hdul[0].header = head
        hdul.flush()

def create_parameterfile():
    
    # Open any image to get the image size
    img = glob.glob('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/I_slices/Abell_85_I_plane_freq01.fits')[0]
    with fits.open(img) as hdul:
        size = hdul[0].data[0][0].shape[1]

    with open(rmsynthpath+'rmsynth.par','w') as file:
        # This is a comment
        file.write(r'% Parameter file for rmsynthesis python code')
        file.write('\n')
        file.write('\n')
        file.write(r'% ra and dec min and max of the subimage to process, given in pixels')
        file.write('\n')
        file.write(r'% a value of -1 means to use the bound of the image')
        file.write('\n')
        if size == 3617:
            file.write(r'dec_min 0')
            file.write('\n')
            file.write(r'dec_max 3617')
            file.write('\n')
            file.write(r'ra_min 0')
            file.write('\n')
            file.write(r'ra_max 3617')
            file.write('\n')
        else:
            raise ValueError("SIZE %s NOT IMPLEMENTED YET"%size)

        file.write('\n')
        file.write(r'% Define the phi axis, dphi in rad/m/m')
        file.write('\n')
        file.write(r'phi_min -200')
        file.write('\n')
        file.write(r'nphi 40')
        file.write('\n')
        file.write(r'dphi 10')
        file.write('\n')
        file.write('\n')
        
        file.write(r'% Clean parameters. Gain is the loop gain, niter is the number of clean iterations')
        file.write('\n')
        file.write(r'do_clean False')
        file.write('\n')
        file.write(r'gain 0.05')
        file.write('\n')
        file.write(r'niter 50000')
        file.write('\n')
        file.write(r'cutoff 2e-5')
        file.write('\n')
        file.write('\n')

        file.write(r'% Weighting parameter. Give the name of the weight file (located in the input_dir). ')
        file.write('\n')
        file.write(r'do_weight inv_var_weights.txt')
        file.write('\n')
        file.write('\n')

        file.write(r'% Detection threshold on polarized intensity map')
        file.write('\n')
        file.write(r'threshold 5e-5')
        file.write('\n')
        file.write('\n')

        file.write(r'% output file')
        file.write('\n')
        file.write('outputfn %s'%rmsynthpath+'rmsynth')
        file.write('\n')
        file.write('\n')

        file.write(r'% directory where the input fits file can be found')
        file.write('\n')
        file.write('input_dir /net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/')
        file.write('\n')

def cleanup_rmsynth():
    """
    Remove the Faraday dispersion function cubes. F(\phi)
        (i.e., P(\phi),Q(\phi) and U(\phi))
    They are quite large. ~7GB for 400 steps in phi with 2736**2 pixels
    and I don't think we need em anymore after we found the maxima.

    """
    
    # todel = rmsynthpath+'rmsynth_di_p.fits'
    # print ("Deleting %s"%todel)
    # os.system('rm %s'%todel)

    todel = rmsynthpath+'rmsynth_di_q.fits'
    print ("Deleting %s"%todel)
    os.system('rm %s'%todel)
    todel = rmsynthpath+'rmsynth_di_u.fits'
    print ("Deleting %s"%todel)
    os.system('rm %s'%todel)


if __name__ == '__main__':

    # Call this script with .e.g., casa -c *.py
    
    print ("Making inverse variance weights and running RM synthesis.")

    figpath = r'/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/'
    if not os.path.exists(figpath):
        print ("Creating directory %s"%figpath)
        os.mkdir(figpath)

    rmsynthpath = r'/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/rmsynth/'
    if not os.path.exists(rmsynthpath):
        print ("Creating directory %s"%rmsynthpath)
        os.mkdir(rmsynthpath)

    ####### DISABLED STORING AVERAGE IMAGES PER SPW. NOT NEEDED.
    # avimgpath = './%s/%s/averaged_images/'%(directory,sourcename)
    # if not os.path.exists(avimgpath):
    #     print ("Creating directory %s"%avimgpath)
    #     os.mkdir(avimgpath)
    # title = 'RMS as function of spw. %s'%sourcename
    # savefig = figpath+'rms_vs_spw.png'
    # plot_rms_vs_spw(directory, sourcename, title, savefig)

    title = 'RMS as function of channel.'
    savefig = figpath+'rms_vs_channel.png'
    plot_rms_vs_channel(title, savefig)

    # Create rmsynth.par file in ./{directory}/{sourcename}/rmsynth/rmsynth.par
    create_parameterfile()

    create_inverse_variance_weights()
    # -p for plotting the RMTF as soon as its computed
    # runrmsynth = "python /net/bovenrijn/data1/digennaro/software/pyrmsynth/rmsynthesis.py -s -p "
    runrmsynth = "python /net/vdesk/data2/GoesaertW/Data_Analyis_Git/rmsynthesis.py -s "
    runrmsynth += rmsynthpath+'rmsynth.par'

    # Actually run RM synth
    os.system(runrmsynth)

    # Cleanup the cubes (they are quite big). Only need the 2D maps.
    cleanup_rmsynth()

    # Plot RMTF afterwards
    savefig = figpath+'RMTF.png'
    plotRMTF('/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/rmsynth/rmsynth_rmsf.txt',savefig,show=False)
    # plotRMTF('/net/voorrijn/data2/osinga/pipeline/cubes/81019/G134.73+48.89/rmsynth_inv_var/rmsynth_rmsf.txt')
