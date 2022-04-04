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
    nchan = 90#48
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

def plot_rms_vs_spw(directory, sourcename, title, savefig):
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
            immath(outfile=('./%s/%s/'%(directory,sourcename)
                            +'averaged_images/averaged_Q_spw%i'%spw)
                    ,mode='evalexpr'
                    ,imagename=imagesQ
                    ,expr=exp_avg)
            immath(outfile=('./%s/%s/'%(directory,sourcename)
                            +'averaged_images/averaged_U_spw%i'%spw)
                    ,mode='evalexpr'
                    ,imagename=imagesU
                    ,expr=exp_avg)

            all_noiseQ[spw] = imstat(('./%s/%s/'%(directory,sourcename)
                                      +'averaged_images/averaged_Q_spw%i'%spw))['rms']
            all_noiseU[spw] = imstat(('./%s/%s/'%(directory,sourcename)
                                      +'averaged_images/averaged_U_spw%i'%spw))['rms']

            imagesQ = []
            imagesU = []

        imnameQ = ('./%s/%s/'%(directory,sourcename)
                  +'stokes_q/{}'.format(sourcename)
                  +'_{0:04}-Q-image.pbcor.smoothed.fits'.format(channum))
        imnameU = ('./%s/%s/'%(directory,sourcename)
                  +'stokes_u/{}'.format(sourcename)
                  +'_{0:04}-U-image.pbcor.smoothed.fits'.format(channum))
        imagesQ.append(imnameQ)
        imagesU.append(imnameU)


    # Don't forget the last one.
    channum += 1
    spw = int(channum/6-1)
    immath(outfile=('./%s/%s/'%(directory,sourcename)
                    +'averaged_images/averaged_Q_spw%i'%spw)
            ,mode='evalexpr'
            ,imagename=imagesQ
            ,expr=exp_avg)
    immath(outfile=('./%s/%s/'%(directory,sourcename)
                    +'averaged_images/averaged_U_spw%i'%spw)
            ,mode='evalexpr'
            ,imagename=imagesU
            ,expr=exp_avg)
    all_noiseQ[spw] = imstat(('./%s/%s/'%(directory,sourcename)
                              +'averaged_images/averaged_Q_spw%i'%spw))['rms']
    all_noiseU[spw] = imstat(('./%s/%s/'%(directory,sourcename)
                              +'averaged_images/averaged_U_spw%i'%spw))['rms']   
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

def manually_add_failedchannels(sourcename, failedchannels):
    """
    While writing the paper, I noticed that some channels were affected by RFI
    but those channels were not flagged by the code below. So we flag it manually
    """
    manually = np.array([])
    if sourcename == 'G114.33+64.87':
        manually = np.array([20, 21, 22, 23, 25, 27, 28])
    if sourcename == 'G062.42-46.41':
        failedchannels = np.array([0,1,2,3,6,8,14,17,22])
    if sourcename == 'G186.39+37.25':
        failedchannels = np.array([19,21,22,48,49,50,51,52,53])
    if sourcename == 'G077.90-26.64':
        failedchannels = np.array([6,7,9,10,11,18,22,48,49,50,51,52])
    if sourcename == 'G072.63+41.46':
        failedchannels = np.array([20,21,22,23,48,50,51,52])

    failedchannels = np.append(failedchannels, manually)
    failedchannels = np.unique(failedchannels) # Just in case I manually add something that was already in ther
    return failedchannels

def plot_rms_vs_channel(directory, sourcename, title, savefig):
    """
    Determine RMS noise as the central RMS I, Q or U image.

    Calculate rms per channel. If RMS > 2mJy define a channel as failed.
    """
    all_chan = np.arange(nchan)

    all_noiseQ = np.empty(nchan) #(90)
    all_noiseU = np.empty(nchan) #(90)
    all_noiseI = np.empty(nchan) #(90)

    # Ignore the failed channels
    failedchannels = np.load('./%s/%s/failedchannels.npy'%(directory,sourcename))
    
    # Use hinges-fences with fence=1.0 so we cut the highest and lowest values
    # from the img
    box = '1100,1100,1500,1500' # Calculate central rms
    # algorithm = 'classic'
    algorithm = 'hinges-fences'

    for channum in tqdm.tqdm(range(0,nchan),desc="Calculating rms per channel for target %s"%sourcename):
        if channum in failedchannels:
            all_noiseQ[channum] = np.nan
            all_noiseU[channum] = np.nan
            all_noiseI[channum] = np.nan
        else:
            rmsQ = imstat(('./{}/{}/{}/{}'.format(
                                directory,sourcename,'stokes_q',sourcename)
                                +'_%04d-Q-image.pbcor.smoothed.fits'%channum)
                                ,algorithm=algorithm,fence=1.0,box=box)['rms'][0]
            rmsU = imstat(('./{}/{}/{}/{}'.format(
                        directory,sourcename,'stokes_u',sourcename)
                                +'_%04d-U-image.pbcor.smoothed.fits'%channum)
                                ,algorithm=algorithm,fence=1.0,box=box)['rms'][0]
            rmsI = imstat(('./{}/{}/{}/{}'.format(
                        directory,sourcename,'stokes_i',sourcename)
                                +'_%04d-image.pbcor.smoothed.fits'%channum)
                                ,algorithm=algorithm,fence=1.0,box=box)['rms'][0]

            if rmsQ > 2e-3 or rmsU > 2e-3: 
                # If RMS above 2 mJy then say it's a failed channel
                # Usually RMS is around 200 microJansky (in single channel)
                failedchannels = np.append(failedchannels,channum)

            else:
                all_noiseQ[channum] = rmsQ
                all_noiseU[channum] = rmsU
                all_noiseI[channum] = rmsI
        
    # Perhaps manually add channels affected by RFI as failedchannels
    failedchannels = manually_add_failedchannels(sourcename, failedchannels)
    # Save as RMS for failed channels as NaN so we're sure it doesnt get used
    all_noiseQ[np.array(failedchannels,dtype='int')] = np.nan
    all_noiseU[np.array(failedchannels,dtype='int')] = np.nan
    all_noiseI[np.array(failedchannels,dtype='int')] = np.nan

    # Save the new version of failed channels in case anything changed
    failedchannels = np.sort(failedchannels)
    np.save('./%s/%s/failedchannels.npy'%(directory,sourcename),failedchannels)

    # plt.scatter(all_chan,all_noiseQ*1e6,label='{}'.format(sourcename),alpha=0.5)
    plt.scatter(all_chan,all_noiseQ*1e6,alpha=0.5,label='Stokes Q',c='#1f77b4')
    plt.scatter(all_chan,all_noiseU*1e6,alpha=0.5,marker="^",label='Stokes U',c='#1f77b4')
    plt.scatter(all_chan,all_noiseI*1e6,alpha=0.5,marker="s",label='Stokes I',c='#ff7f0e')
    plt.axhline(np.nanmedian(all_noiseI)*1e6,ls='dashed',c='k',label='Stokes I median noise')

    np.save('./%s/%s/all_noiseQ.npy'%(directory,sourcename),all_noiseQ)
    np.save('./%s/%s/all_noiseU.npy'%(directory,sourcename),all_noiseU)
    np.save('./%s/%s/all_noiseI.npy'%(directory,sourcename),all_noiseI)

    plt.xlabel("Channel index")
    plt.title(title)
    plt.ylabel("Central RMS ($\mu$Jy/beam)")
    plt.legend(loc='upper left',frameon=False)
    plt.savefig(savefig)
    plt.show()
    plt.close()

def remove_failed_channels(directory, sourcename):
    """
    Because the function plot_rms_vs_channel() edits the failedchannels
    we have to remove the new failed channels from the stokes_{i,q,u} directories.
    Otherwise the RMsynth module will still use them
    """


    pre_imname = str(directory) +'/'+ sourcename + '/' 
    output_dir = pre_imname + 'stokes_'

    print ("Removing all pb corrected smoothed images that are failed from %s"%(output_dir+"(q,u,i)"))

    stokes = ['i','q','u']
    imagestokes = ['','Q-','U-']
    MFSimagestokes = ['I-', 'Q-', 'U-'] # naming is slightly different...

    failedchannels = np.load('./%s/%s/failedchannels.npy'%(directory,sourcename))

    for i, stok in enumerate(stokes):
        for channum in range(0,nchan):
            if channum in failedchannels:
                    imname = output_dir+stok + '/'
                    imname += sourcename+'_'+'{0:04}-'.format(channum) + imagestokes[i] +'image.pbcor.smoothed.fits'  
                    if os.path.exists(imname):
                        removecommand = 'rm ' + imname
                        print (removecommand)
                        os.system(removecommand)

def create_inverse_variance_weights(directory, sourcename):
    """
    Given the rms noise levels that were just calculated. Translate these
    to weights.
    """


    all_noiseQ = np.load('./%s/%s/all_noiseQ.npy'%(directory,sourcename))
    all_noiseU = np.load('./%s/%s/all_noiseU.npy'%(directory,sourcename))
    # Remove failed channels if any
    nanmask = np.invert(np.isnan(all_noiseQ))
    
    all_noiseQ = all_noiseQ[nanmask]
    all_noiseU = all_noiseU[nanmask]

    averagerms = (all_noiseQ + all_noiseU)/2.

    weights = (1/averagerms)**2 # INVERSE VARIANCE WEIGHING

    # Normalize so that they sum to 1
    weights /= np.sum(weights)

    # save the weights to a file
    weightfile = open('./%s/%s/inv_var_weights.txt'%(directory,sourcename),'w')
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

def fix_rmsynth_polint_header(directory, sourcename):
    """
    For PYBDSF to accept the file, need at least 3 axis where the third is freq
    """
    rmpolint = './%s/%s/rmsynth/rmsynth_polint.fits'%(directory,sourcename)

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

def create_parameterfile(directory, sourcename):
    
    # Open any image to get the image size
    img = glob.glob('./{}/{}/{}/{}_00*-Q-image.pbcor.smoothed.fits'.format(directory,sourcename,'stokes_q',sourcename))[0]
    with fits.open(img) as hdul:
        size = hdul[0].data.shape[2]

    with open(rmsynthpath+'rmsynth.par','w') as file:
        # This is a comment
        file.write(r'% Parameter file for rmsynthesis python code')
        file.write('\n')
        file.write('\n')
        file.write(r'% ra and dec min and max of the subimage to process, given in pixels')
        file.write('\n')
        file.write(r'% a value of -1 means to use the bound of the image')
        file.write('\n')
        if size == 2736:
            file.write(r'dec_min 605')
            file.write('\n')
            file.write(r'dec_max 2125')
            file.write('\n')
            file.write(r'ra_min 610')
            file.write('\n')
            file.write(r'ra_max 2120')
            file.write('\n')
        else:
            raise ValueError("SIZE %s NOT IMPLEMENTED YET"%size)

        file.write('\n')
        file.write(r'% Define the phi axis, dphi in rad/m/m')
        file.write('\n')
        file.write(r'phi_min -2000')
        file.write('\n')
        file.write(r'nphi 400')
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
        file.write('input_dir ./%s/%s/'%(directory,sourcename))
        file.write('\n')

def cleanup_rmsynth(directory, sourcename):
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

    # Call this script with .e.g., casa -c *.py 81019 G134.73+48.89
    try:
        directory, sourcename = int(sys.argv[3]), str(sys.argv[4])
    except:
        print("ERROR: Please call this script with arguments 'directory' and 'sourcename'")
        raise

    print ("Making inverse variance weights and running RM synthesis.")

    figpath = './%s/%s/figures/'%(directory,sourcename)
    if not os.path.exists(figpath):
        print ("Creating directory %s"%figpath)
        os.mkdir(figpath)

    rmsynthpath = './%s/%s/rmsynth/'%(directory,sourcename)
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

    title = 'RMS as function of channel. %s'%sourcename
    savefig = figpath+'rms_vs_channel.png'
    plot_rms_vs_channel(directory, sourcename, title, savefig)

    # Remove failed channels if they are new from the RMS cutoff
    remove_failed_channels(directory, sourcename)

    # Save weights as numpy array to disk
    create_inverse_variance_weights(directory, sourcename)

    # Create rmsynth.par file in ./{directory}/{sourcename}/rmsynth/rmsynth.par
    create_parameterfile(directory,sourcename)

    # -p for plotting the RMTF as soon as its computed
    # runrmsynth = "python /net/bovenrijn/data1/digennaro/software/pyrmsynth/rmsynthesis.py -s -p "
    runrmsynth = "python /net/lofar4/data1/osinga/software/pyrmsynth/rmsynthesis.py -s "
    runrmsynth += rmsynthpath+'rmsynth.par'

    # Actually run RM synth
    os.system(runrmsynth)

    # Cleanup the cubes (they are quite big). Only need the 2D maps.
    cleanup_rmsynth(directory, sourcename)

    # Plot RMTF afterwards
    savefig = figpath+'RMTF.png'
    plotRMTF('./%s/%s/rmsynth/rmsynth_rmsf.txt'%(directory,sourcename),savefig,show=False)
    # plotRMTF('/net/voorrijn/data2/osinga/pipeline/cubes/81019/G134.73+48.89/rmsynth_inv_var/rmsynth_rmsf.txt')
