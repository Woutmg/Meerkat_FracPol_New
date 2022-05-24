import numpy
import corner
import emcee
import matplotlib.pyplot # as plt doesnt work with parallel code
import matplotlib #as mpl
#import matplotlib.ticker as mtick
#import sys
#from tabulate import tabulate
import scipy.optimize
import sys


def polarization_fitting(wave, fluxI, err_fluxI, fluxQ, err_fluxQ, fluxU
  , err_fluxU, rm, x, y, cornerplot,curvature,sourcenum=None,plotdir='/net/vdesk/data2/GoesaertW/Meerkat_Data/Abell_85/',depol='ExtDepol',overdispersed_start=False):
  """
  ERIK ADDED: i: source number, plotdir='', where to put the cornerplot
  ERIK ADDED: check NaN values. Return NaN
  ERIK ADDED: whether to start as gabriella, with 200 chains at approximately the same value
              or to start with overdispersed starting values, only implemented for ExtDepol
  ERIK ADDED(10-01-2022): ExtDepol_RePol model which is ExtDepol that allows negative values for sigmaRM**2 
  """

  numpy.seterr(all='ignore')

  matplotlib.rcParams['xtick.direction']='in'
  matplotlib.rcParams['ytick.direction']='in'
  matplotlib.rcParams['xtick.labelsize']= 12
  matplotlib.rcParams['ytick.labelsize']= 12

  if depol != 'ExtDepol' and overdispersed_start == True:
    raise ValueError("TODO: not implemented yet for other models.")

  ##### CHECK WHETHER THERE ARE NANS IN THE DATA. IF SO RETURN NAN.
  if (numpy.isnan(fluxI).any() or numpy.isnan(fluxQ).any() or numpy.isnan(fluxU).any() 
     or numpy.isnan(rm)):
    if sourcenum is not None:
      print ("ERROR: Fitting for x,y=%i,%i (Sourcenum=%i) FAILED because of NAN"%(x,y,sourcenum))
      print ("RETURNING numpy.nan")

    labelname = ["norm","spix","curv"]
    fit_dataI = [[labelname[i], numpy.nan, numpy.nan, numpy.nan] for i in range(len(labelname))]
  
    if depol == "":
      labelname = ["p0","chi0","RM"]
      fit_data = [[labelname[i], numpy.nan, numpy.nan, numpy.nan] for i in range(len(labelname))]    
    elif depol == "ExtDepol" or depol == 'ExtDepol_RePol':
      labelname = ["p0","chi0","RM","sigma"]
      fit_data = [[labelname[i], numpy.nan, numpy.nan, numpy.nan] for i in range(len(labelname))]    
    elif depol == "IntDepol":
      labelname = ["p0","chi0","RM","sigma"]
      fit_data = [[labelname[i], numpy.nan, numpy.nan, numpy.nan] for i in range(len(labelname))]    
  
    return(fit_dataI, fit_data, x, y)

  # print depol
  ######################################################################################################
  # MCMC SPIX
  #def lnprior_spix(theta):
    #norm, a, curv = theta
  
    #return 0.0
    
  def lnprior_spix(theta, curvature):
    """
    Restrain parameters theta (norm,a,curv) to physical parts of parameter space

    curvature -- boolean -- whether to expect also the third (curvature) param
    """
    if curvature:
      norm, a, curv = theta
    else:
      norm, a = theta
      curv = 0

    if norm>=0. and -numpy.inf<a<numpy.inf and -numpy.inf<curv<numpy.inf:
      return 0.

    return -numpy.inf

  def lnlike_spix(theta,x,y,err,curvature): 
    if curvature:
      norm, a, curv = theta
    else:
      norm, a = theta
      curv = 0

    model = norm * x**(curv*numpy.log10(x) + a)  #see Massaro+04, A&A, 413, 489
  
    return -0.5*numpy.sum((y - model)**2/err**2)

  def lnprob_spix(theta,x,y,err,curvature):
    lp = lnprior_spix(theta, curvature)
    if not numpy.isfinite(lp):
      return -numpy.inf
  
    return lp + lnlike_spix(theta, x, y, err, curvature)

  # MCMC POLARIZATION 
  def lnprior_pol(theta):
    if depol == "ExtDepol" or depol == "IntDepol" or depol == "CombDepol":
      p0, chi0, rm, sigma_rm = theta
      if 0.<=p0<=1. and sigma_rm>=0. and (-200 <= rm <= 200):# and -numpy.pi<=chi0<numpy.pi
        return 0.	
    elif depol == 'ExtDepol_RePol': # For ExtDepol_Repol : sigma_RM unconstrained between -inf and +inf
      p0, chi0, rm, sigma_rm = theta
      if 0.<=p0<=1. and 0.<=chi0<numpy.pi and (-200 <= rm <= 200):
        return 0.       
    else: 
      p0, chi0, rm = theta
      if 0.<=p0<=1. and 0.<=chi0<numpy.pi and (-200 <= rm <= 200):
        return 0.

    return -numpy.inf


  def lnlike_pol(theta,x,y1,err1,y2,err2,norm,a,curv): 
    if depol == "ExtDepol" or depol == "IntDepol" or depol == "CombDepol" or depol == "ExtDepol_RePol":
      p0, chi0, rm, sigma_rm = theta
    
      freq = (3.e8/numpy.sqrt(x))*1.e-9
      stokes_i = norm * freq**(curv*numpy.log10(freq) + a)
    
      if depol == "ExtDepol" or depol == "ExtDepol_RePol":
        model1 = stokes_i * p0 * numpy.exp(-2*(sigma_rm)*(x**2)) * numpy.cos(2*(chi0 + rm*x))
        model2 = stokes_i * p0 * numpy.exp(-2*(sigma_rm)*(x**2)) * numpy.sin(2*(chi0 + rm*x))
      elif depol == "IntDepol":
        model1 = stokes_i * p0 * ((1.-numpy.exp(-2*(sigma_rm)*(x**2)))/(2*(sigma_rm)*(x**2))) * numpy.cos(2*(chi0 + rm*x))
        model2 = stokes_i * p0 * ((1.-numpy.exp(-2*(sigma_rm)*(x**2)))/(2*(sigma_rm)*(x**2))) * numpy.sin(2*(chi0 + rm*x))
      
    else:
      p0, chi0, rm,  = theta
    
      freq = (3.e8/numpy.sqrt(x))*1.e-9
      stokes_i = norm * freq**(curv*numpy.log10(freq) + a)
      
      model1 = stokes_i * p0 * numpy.cos(2*(chi0 + rm*x))
      model2 = stokes_i * p0 * numpy.sin(2*(chi0 + rm*x))    

    return -0.5 * numpy.sum( ((y1 - model1)/err1)**2 + ((y2 - model2)/err2)**2  )


  def lnprob_pol(theta,x,y1,err1,y2,err2,norm,a,curv):
    lp = lnprior_pol(theta)
    if not numpy.isfinite(lp):
      return -numpy.inf

    return lp + lnlike_pol(theta, x, y1, err1,y2,err2,norm,a,curv)

######################################################################################################



  ##################### STOKES I FIT
  freq = (3.e8/numpy.sqrt(wave))*1.e-9
  
  if curvature:
    # Eq 8 in Gabri paper
    func = lambda p, i: p[0] * i**(p[1]+p[2]*numpy.log10(i))
    guess= [1.0,-1.0,0] #norm, a, curv
  else:
    # Eq 8 without the curvature. 
    func = lambda p, i: p[0] * i**(p[1])
    guess= [1.0,-1.0] #norm, a
  
  errfunc = lambda p, i, j, err: (j - func(p, i)) / err
  out = scipy.optimize.leastsq(errfunc, guess, args=(freq, fluxI, err_fluxI), full_output=True)
           
  coeff = out[0]
  norm = coeff[0]
  a = coeff[1]
  if curvature:
    curv = coeff[2]
  else:
    # Always return something for curv, even when it's not wanted
    curv = 0
  
  #alpha = a + 2.*curv*numpy.log10(2)
  
  #print x, y
  
  if curvature:
    parms = [norm,a,curv]
  else:
    parms = [norm,a]


  ndim, nwalkers = len(parms), 200
  pos = [parms + 1.e-4*numpy.random.randn(ndim) for i in range(nwalkers)]
  
  runs = 1000
  sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_spix, args=(freq, fluxI, err_fluxI, curvature))
  
  
  sampler.run_mcmc(pos, runs)
  burn = 200
  samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
  

  # FIT SUMMARY
  lim_low = 50. - (68.27/2.)
  lim_upp = 50. + (68.27/2.)
  val_with_errs = [(v[1], v[1]-v[0], v[2]-v[1]) 
                   for v in zip(*numpy.percentile(samples, [lim_low, 50., lim_upp], axis=0))]
  
  labelname = ["norm","spix","curv"]
  fit_dataI = [[labelname[i], val_with_errs[i][0], -val_with_errs[i][1], val_with_errs[i][2]] for i in range(len(val_with_errs))]  
  
  norm = fit_dataI[0][1]
  a    = fit_dataI[1][1]
  if curvature:
    curv = fit_dataI[2][1]
  # otherwise it's set to 0 already
  
  if cornerplot:
    #CORNER PLOT
    label1 = '$I_0$ [mJy]'
    label2 = '$a$'
    label3 = '$b$'
    
    labelist = [label1, label2, label3]  
  
  
    fig = corner.corner(samples, labels=labelist, bins=[20]*ndim
      , label_kwargs={"fontsize": 14}, show_titles=True, title_kwargs={"fontsize": 8}
      , title_fmt='.3f',quantiles=[0.16, 0.5, 0.84])
  
    #numpy.savetxt("/net/bovenrijn/data1/digennaro/CIZAJ2242/polarization/5arcsec/pixel_"+str(x)+"-"+str(y)+"_fitI.txt", [norm, a, curv])
    if sourcenum is not None:
      # matplotlib.pyplot.savefig(plotdir+'source_%i'%sourcenum+"_cornerplotI.pdf")
      matplotlib.pyplot.savefig(plotdir+'source_%i'%sourcenum+"_cornerplotI.png")

    else:
      # matplotlib.pyplot.savefig(plotdir+str(x)+"-"+str(y)+"_cornerplotI.pdf")
      matplotlib.pyplot.savefig(plotdir+str(x)+"-"+str(y)+"_cornerplotI.png")
    matplotlib.pyplot.close()
  
  ##################### STOKES QU FIT

  # stokes I flux from model
  stokes_i = norm * (freq)**(a + curv*numpy.log10(freq))

    
  func = lambda p, x: 2.*(p[1] + p[2]*x) 


  
  if depol == "":
    guess = [1,1,rm] # p0, chi0, RM
    
    # Using Eq 5 in Gabri paper.
    errfunc = lambda p, x1, y1, err1, x2, y2, err2,stokes_i: \
      abs( (y1 - (stokes_i * p[0]*numpy.cos(func(p,x1))))/err1 ) + \
      abs( (y2 - (stokes_i * p[0]*numpy.sin(func(p,x2))))/err2 )

    out = scipy.optimize.leastsq(errfunc, guess, args=(wave,fluxQ,err_fluxQ,wave,fluxU,err_fluxU,stokes_i), full_output=True)
    coeff = out[0]
    covar = out[1]
    
    # boundaries on p0
    if coeff[0] < 0:
      coeff[0] = -1*coeff[0]
      coeff[1] =  numpy.pi/2 - coeff[1]
    elif coeff[0] > 1:
      coeff[0] = 1

    # boundaries on chi0    
    if coeff[1] >= numpy.pi or coeff[1] < 0:
      coeff[1] = coeff[1] % numpy.pi


    p0 = coeff[0]
    chi0 = coeff[1]
    rm_fit = coeff[2]
    sigma_rm = numpy.nan
    
    parms = [p0,chi0,rm_fit]
 
    ndim, nwalkers = len(parms), 200
    pos = [parms + 1.e-4*numpy.random.randn(ndim) for i in range(nwalkers)]
  
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_pol
      , args=(wave,fluxQ,err_fluxQ,fluxU,err_fluxU,norm,a,curv))
    
    runs = 1000
    sampler.run_mcmc(pos, runs)
    burn = 200
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    # if sourcenum is not None: 
    #   print ("Saving "+plotdir+'../'+'test_samples_%i.npy'%sourcenum)
    #   numpy.save(plotdir+'../'+'test_samples_%i.npy'%sourcenum,samples)

    # FIT SUMMARY
    lim_low = 50. - (68.27/2.)
    lim_upp = 50. + (68.27/2.)
    val_with_errs = [(v[1], v[1]-v[0], v[2]-v[1]) 
                     for v in zip(*numpy.percentile(samples, [lim_low, 50., lim_upp], axis=0))]
  
    labelname = ["p0","chi0","RM"]
    fit_data = [[labelname[i], val_with_errs[i][0], -val_with_errs[i][1], val_with_errs[i][2]] for i in range(len(val_with_errs))]    

    if cornerplot:
      # CORNER PLOT
      label1 = '$p_0$'
      label2 = r'$\chi_0$ [rad]'
      label3 = 'RM [rad m$^{-2}$]'
    
      labelist = [label1, label2, label3]  
  
      lims = [(samples[:,i].min(),samples[:,i].max()) for i in range(3)]      

      fig = corner.corner(samples, labels=labelist, bins=[20]*ndim
        , label_kwargs={"fontsize": 20}
        , show_titles=True, title_kwargs={"fontsize": 14}
        , title_fmt='.3f',quantiles=[0.16, 0.5, 0.84]
        , range=lims)

      matplotlib.pyplot.text(0.7,0.8,"Initial RM guess: %.2f [rad m$^{-2}$]"%rm
        ,transform=fig.axes[1].transAxes)

      #numpy.savetxt("/net/bovenrijn/data1/digennaro/CIZAJ2242/polarization/5arcsec/pixel_"+str(x)+"-"+str(y)+"_fitQU.txt", [fit_data[0][1], fit_data[1][1], fit_data[2][1], fit_data[3][1]])
      if sourcenum is not None:
        # matplotlib.pyplot.savefig(plotdir+'source_%i'%sourcenum+"_cornerplotQU_"+depol+".pdf")
        matplotlib.pyplot.savefig(plotdir+'source_%i'%sourcenum+"_cornerplotQU_"+depol+".png")
        matplotlib.pyplot.savefig(plotdir+'source_%i'%sourcenum+"_cornerplotQU_"+depol+".png")
      
      else:
        # matplotlib.pyplot.savefig(plotdir+str(x)+"-"+str(y)+"_cornerplotQU_"+depol+".pdf")
        matplotlib.pyplot.savefig(plotdir+str(x)+"-"+str(y)+"_cornerplotQU_"+depol+".png")
      #matplotlib.pyplot.show()
      matplotlib.pyplot.close()
    #sys.exit()
    
  elif depol == "ExtDepol":
    guess = [0.6,0,rm,1] # p0, chi0, RM, sigma_RM
    
    # Using Eq 6 in Gabri paper
    errfunc = lambda p, x1, y1, err1, x2, y2, err2, stokes_i: \
      abs( (y1 - (stokes_i * (p[0]*numpy.exp(-2*(p[3])*(x1**2))) * numpy.cos(func(p,x1))) )/err1 ) + \
      abs( (y2 - (stokes_i * (p[0]*numpy.exp(-2*(p[3])*(x1**2))) * numpy.sin(func(p,x2))) )/err2 )

    out = scipy.optimize.leastsq(errfunc, guess, args=(wave,fluxQ,err_fluxQ,wave,fluxU,err_fluxU,stokes_i))        
    coeff = out[0]

    # boundaries on p0
    if coeff[0] < 0:
      coeff[0] = -1*coeff[0]
      coeff[1] =  numpy.pi/2 - coeff[1]
    elif coeff[0] > 1:
      coeff[0] = 1

    # boundaries on chi0    
    if coeff[1] >= numpy.pi or coeff[1] < 0:
        coeff[1] = coeff[1] % numpy.pi
            
    p0 = coeff[0]
    chi0 = coeff[1]
    rm_fit = coeff[2]
    sigma_rm = abs(coeff[3])
    
    parms = [p0,chi0,rm_fit,sigma_rm]
    

    ndim, nwalkers = len(parms), 200
    if overdispersed_start:
        pos = numpy.array([parms + numpy.array([0.5,0.5*numpy.pi,100,100])*numpy.random.randn(ndim) for i in range(nwalkers)])
        pos[:,1] = [0 + 1.e-4*numpy.random.randn(1) for i in range(nwalkers)]
    else:
        pos = numpy.array([parms + 1.e-4*numpy.random.randn(ndim) for i in range(nwalkers)])
        pos[:,1] = [0 + 1.e-4*numpy.random.randn(1) for i in range(nwalkers)]
  
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_pol, args=(wave,fluxQ,err_fluxQ,fluxU,err_fluxU,norm,a,curv))
    
    runs = 1000
    sampler.run_mcmc(pos, runs)
    burn = 300
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    # if sourcenum is not None: 
    #   print ("Saving "+plotdir+'../'+'test_samples_%i.npy'%sourcenum)
    #   numpy.save(plotdir+'../'+'test_samples_%i.npy'%sourcenum,samples)

    samples[:,1] = samples[:,1] % numpy.pi # Range of chi0 was set to -pi to pi to avoid boundary problem -> need to mod samples
    # FIT SUMMARY
    lim_low = 50. - (68.27/2.)
    lim_upp = 50. + (68.27/2.)
    val_with_errs = [(v[1], v[1]-v[0], v[2]-v[1]) 
                     for v in zip(*numpy.percentile(samples, [lim_low, 50., lim_upp], axis=0))]
  
    labelname = ["p0","chi0","RM","sigma"]
    fit_data = [[labelname[i], val_with_errs[i][0], -val_with_errs[i][1], val_with_errs[i][2]] for i in range(len(val_with_errs))]   

    if cornerplot:
        fig, axes = matplotlib.pyplot.subplots(ncols=1, nrows=4)
        fig.set_size_inches(12,6)
        axes[0].plot(sampler.chain[:, :, 0].transpose(), color='black', alpha=0.05)
        axes[0].set_ylabel(labelname[0])
        axes[0].axvline(burn, ls='dashed', color='red', label = 'End of burn-in')
        axes[1].plot(sampler.chain[:, :, 1].transpose(), color='black', alpha=0.05)
        axes[1].set_ylabel(labelname[1])
        axes[1].axvline(burn, ls='dashed', color='red')
        axes[2].plot(sampler.chain[:, :, 2].transpose(), color='black', alpha=0.05)
        axes[2].set_ylabel(labelname[2])
        axes[2].axvline(burn, ls='dashed', color='red')
        axes[3].plot(sampler.chain[:, :, 3].transpose(), color='black', alpha=0.05)
        axes[3].set_ylabel(labelname[3])
        axes[3].axvline(burn, ls='dashed', color='red')
        axes[3].set_xlabel('Step number')
        fig.tight_layout()
        fig.suptitle('MCMC tracks trough parameter space')

    if cornerplot:
      # CORNER PLOT
      label1 = '$p_0$'
      label2 = r'$\chi_0$ [rad]'
      label3 = 'RM [rad m$^{-2}$]'
      label4 = r'$\sigma_{\rm RM}^2$ [rad m$^{-2}$]'
    
      labelist = [label1, label2, label3, label4]  
  
      lims = [(samples[:,i].min(),samples[:,i].max()) for i in range(4)]      


      # matplotlib.rc('figure', figsize=(6.64, 0.74*6.64), dpi=100)
      # fig = matplotlib.pyplot.figure(figsize=(6.64, 0.74*6.64), dpi=100)

      fig = corner.corner(samples, labels=labelist, bins=[20]*ndim
        , label_kwargs={"fontsize": 20}
        , show_titles=True, title_kwargs={"fontsize": 14}
        , title_fmt='.3f',quantiles=[0.16, 0.5, 0.84]
        , range=lims)
        # , figsize=(6.64, 0.74*6.64), dpi=100)
      # print ("Setting figure size smaller!")

      matplotlib.pyplot.text(0.7,0.8,"Initial RM guess: %.2f [rad m$^{-2}$]"%rm,transform=fig.axes[6].transAxes)

      # To save files with different names
      if overdispersed_start:
          odstr = '_ODstart'
      else:
          odstr = ''

      #numpy.savetxt("/net/bovenrijn/data1/digennaro/CIZAJ2242/polarization/5arcsec/pixel_"+str(x)+"-"+str(y)+"_fitQU.txt", [fit_data[0][1], fit_data[1][1], fit_data[2][1], fit_data[3][1]])
      if sourcenum is not None:
        # matplotlib.pyplot.savefig(plotdir+'source_%i'%sourcenum+"_cornerplotQU_"+depol+".pdf")
        # print ("Saving to %s"%plotdir+'pdfs/'+'source_%i'%sourcenum+"_cornerplotQU_"+depol+".pdf")
        matplotlib.pyplot.savefig(plotdir+'pdfs/'+'source_%i'%sourcenum+"_cornerplotQU_"+depol+"%s.pdf"%odstr)
        matplotlib.pyplot.savefig(plotdir+'source_%i'%sourcenum+"_cornerplotQU_"+depol+"%s.png"%odstr)
      
      else:
        # matplotlib.pyplot.savefig(plotdir+str(x)+"-"+str(y)+"_cornerplotQU_"+depol+".pdf")
        matplotlib.pyplot.savefig(plotdir+str(x)+"-"+str(y)+"_cornerplotQU_"+depol+"%s.png"%odstr)
      #matplotlib.pyplot.show()
      matplotlib.pyplot.close()


      # Save chain as well 
      if True:
        fig = corner.corner(samples, labels=labelist, bins=[20]*ndim
        , label_kwargs={"fontsize": 20}
        , show_titles=False, title_kwargs={"fontsize": 14}
        , quantiles=[0.16, 0.5, 0.84]
        , range=lims)

        ch = plotdir[:-8]+'chain_source_%i_QU%s.npy'%(sourcenum,odstr)
        print ("Saving chain to %s"%ch)
        numpy.save(ch, samples)

        ## For publication, uncomment this for a specific source
        # matplotlib.pyplot.savefig(plotdir+'pdfs/'+'no_title_source_%i'%sourcenum+"_cornerplotQU_"+depol+".pdf")
        # matplotlib.pyplot.close()



    #sys.exit()

  elif depol == "ExtDepol_RePol": # ExtDepol model with positive sigma_RM**2 and negative sigma_RM**2
    guess = [0.6,1,rm,1] # p0, chi0, RM, sigma_RM
    
    # Using Eq 6 in Gabri paper
    errfunc = lambda p, x1, y1, err1, x2, y2, err2, stokes_i: \
      abs( (y1 - (stokes_i * (p[0]*numpy.exp(-2*(p[3])*(x1**2))) * numpy.cos(func(p,x1))) )/err1 ) + \
      abs( (y2 - (stokes_i * (p[0]*numpy.exp(-2*(p[3])*(x1**2))) * numpy.sin(func(p,x2))) )/err2 )

    out = scipy.optimize.leastsq(errfunc, guess, args=(wave,fluxQ,err_fluxQ,wave,fluxU,err_fluxU,stokes_i))        
    coeff = out[0]

    # boundaries on p0
    if coeff[0] < 0:
      coeff[0] = -1*coeff[0]
      coeff[1] =  numpy.pi/2 - coeff[1]
    elif coeff[0] > 1:
      coeff[0] = 1

    # boundaries on chi0    
    if coeff[1] >= numpy.pi or coeff[1] < 0:
      coeff[1] = coeff[1] % numpy.pi
            
    p0 = coeff[0]
    chi0 = coeff[1]
    rm_fit = coeff[2]
    sigma_rm = coeff[3]
    
    parms = [p0,chi0,rm_fit,sigma_rm]
    

    ndim, nwalkers = len(parms), 200
    if overdispersed_start:
      pos = [parms + numpy.array([0.5,0.5*numpy.pi,100,100])*numpy.random.randn(ndim) for i in range(nwalkers)]
    else:
      pos = [parms + 1.e-4*numpy.random.randn(ndim) for i in range(nwalkers)]
  
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_pol, args=(wave,fluxQ,err_fluxQ,fluxU,err_fluxU,norm,a,curv))
    
    runs = 1000
    sampler.run_mcmc(pos, runs)
    burn = 200
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    # if sourcenum is not None: 
    #   print ("Saving "+plotdir+'../'+'test_samples_%i.npy'%sourcenum)
    #   numpy.save(plotdir+'../'+'test_samples_%i.npy'%sourcenum,samples)
      


    # FIT SUMMARY
    lim_low = 50. - (68.27/2.)
    lim_upp = 50. + (68.27/2.)
    val_with_errs = [(v[1], v[1]-v[0], v[2]-v[1]) 
                     for v in zip(*numpy.percentile(samples, [lim_low, 50., lim_upp], axis=0))]
  
    labelname = ["p0","chi0","RM","sigma"]
    fit_data = [[labelname[i], val_with_errs[i][0], -val_with_errs[i][1], val_with_errs[i][2]] for i in range(len(val_with_errs))]   

    if cornerplot:
      # CORNER PLOT
      label1 = '$p_0$'
      label2 = r'$\chi_0$ [rad]'
      label3 = 'RM [rad m$^{-2}$]'
      label4 = r'$\sigma_{\rm RM}^2$ [rad m$^{-2}$]'
    
      labelist = [label1, label2, label3, label4]  
  
      lims = [(samples[:,i].min(),samples[:,i].max()) for i in range(4)]      


      # matplotlib.rc('figure', figsize=(6.64, 0.74*6.64), dpi=100)
      # fig = matplotlib.pyplot.figure(figsize=(6.64, 0.74*6.64), dpi=100)

      fig = corner.corner(samples, labels=labelist, bins=[20]*ndim
        , label_kwargs={"fontsize": 20}
        , show_titles=True, title_kwargs={"fontsize": 14}
        , title_fmt='.3f',quantiles=[0.16, 0.5, 0.84]
        , range=lims)
        # , figsize=(6.64, 0.74*6.64), dpi=100)
      # print ("Setting figure size smaller!")

      matplotlib.pyplot.text(0.7,0.8,"Initial RM guess: %.2f [rad m$^{-2}$]"%rm,transform=fig.axes[6].transAxes)

      # To save files with different names
      if overdispersed_start:
          odstr = '_ODstart'
      else:
          odstr = ''

      #numpy.savetxt("/net/bovenrijn/data1/digennaro/CIZAJ2242/polarization/5arcsec/pixel_"+str(x)+"-"+str(y)+"_fitQU.txt", [fit_data[0][1], fit_data[1][1], fit_data[2][1], fit_data[3][1]])
      if sourcenum is not None:
        # matplotlib.pyplot.savefig(plotdir+'source_%i'%sourcenum+"_cornerplotQU_"+depol+".pdf")
        # print ("Saving to %s"%plotdir+'pdfs/'+'source_%i'%sourcenum+"_cornerplotQU_"+depol+".pdf")
        matplotlib.pyplot.savefig(plotdir+'pdfs/'+'source_%i'%sourcenum+"_cornerplotQU_"+depol+"%s.pdf"%odstr)
        matplotlib.pyplot.savefig(plotdir+'source_%i'%sourcenum+"_cornerplotQU_"+depol+"%s.png"%odstr)
      
      else:
        # matplotlib.pyplot.savefig(plotdir+str(x)+"-"+str(y)+"_cornerplotQU_"+depol+".pdf")
        matplotlib.pyplot.savefig(plotdir+str(x)+"-"+str(y)+"_cornerplotQU_"+depol+"%s.png"%odstr)
      #matplotlib.pyplot.show()
      matplotlib.pyplot.close()


      # Save chain as well 
      if True:
        fig = corner.corner(samples, labels=labelist, bins=[20]*ndim
        , label_kwargs={"fontsize": 20}
        , show_titles=False, title_kwargs={"fontsize": 14}
        , quantiles=[0.16, 0.5, 0.84]
        , range=lims)
        ch = plotdir[:-8]+'chain_source_%i_QU%s_%s.npy'%(sourcenum,odstr,depol)
        print ("Saving chain to %s"%ch)
        numpy.save(ch, samples)

        ## For publication, uncomment this for a specific source
        # matplotlib.pyplot.savefig(plotdir+'pdfs/'+'no_title_source_%i'%sourcenum+"_cornerplotQU_"+depol+".pdf")
        # matplotlib.pyplot.close()




  elif depol == "IntDepol":
    guess = [1,1,rm,1]
    
    errfunc = lambda p, x1, y1, err1, x2, y2, err2, stokes_i: \
      abs( (y1 - (stokes_i * p[0] * ((1. - numpy.exp(-2*(p[3])*(x1**2))) / (2*(p[3])*(x1**2))) * numpy.cos(func(p,x1))) )/err1 ) + \
      abs( (y2 - (stokes_i * p[0] * ((1. - numpy.exp(-2*(p[3])*(x1**2))) / (2*(p[3])*(x1**2))) * numpy.sin(func(p,x2))) )/err2 )
  
    out = scipy.optimize.leastsq(errfunc, guess, args=(wave,fluxQ,err_fluxQ,wave,fluxU,err_fluxU,stokes_i), full_output=True)
    coeff = out[0]

    # boundaries on p0
    if coeff[0] < 0:
      coeff[0] = -1*coeff[0]
      coeff[1] =  numpy.pi/2 - coeff[1]
    elif coeff[0] > 1:
      coeff[0] = 1

    # boundaries on chi0    
    if coeff[1] >= numpy.pi or coeff[1] < 0:
      coeff[1] = coeff[1] % numpy.pi

      
    p0 = coeff[0]
    chi0 = coeff[1]
    rm_fit = coeff[2]
    sigma_rm = abs(coeff[3])

    parms = [p0,chi0,rm_fit,sigma_rm]

    ndim, nwalkers = len(parms), 200
    pos = [parms + 1e-4*numpy.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_pol, args=(wave,fluxQ,err_fluxQ,fluxU,err_fluxU,norm,a,curv))
    

    runs = 1000
    sampler.run_mcmc(pos, runs)
    burn = 200
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))    


    # FIT SUMMARY
    lim_low = 50. - (68.27/2.)
    lim_upp = 50. + (68.27/2.)
    val_with_errs = [(v[1], v[1]-v[0], v[2]-v[1]) 
                     for v in zip(*numpy.percentile(samples, [lim_low, 50., lim_upp], axis=0))]
  
    labelname = ["p0","chi0","RM","sigma"]
    fit_data = [[labelname[i], val_with_errs[i][0], -val_with_errs[i][1], val_with_errs[i][2]] for i in range(len(val_with_errs))]

    if cornerplot:
      # CORNER PLOT
      label1 = '$p_0$'
      label2 = r'$\chi_0$ [rad]'
      label3 = 'RM [rad m$^{-2}$]'
      label4 = r'$\varsigma_{\rm RM}^2$ [rad m$^{-2}$]'
    
      labelist = [label1, label2, label3, label4]  
  
      fig = corner.corner(samples, labels=labelist, bins=[20]*ndim
        , label_kwargs={"fontsize": 20}, show_titles=True
        , title_kwargs={"fontsize": 14}, title_fmt='.3f',quantiles=[0.16, 0.5, 0.84])

      matplotlib.pyplot.text(0.7,0.7,"Initial RM guess: %.2f"%rm)

      #numpy.savetxt("/net/bovenrijn/data1/digennaro/CIZAJ2242/polarization/5arcsec/pixel_"+str(x)+"-"+str(y)+"_fitQU.txt", [fit_data[0][1], fit_data[1][1], fit_data[2][1], fit_data[3][1]])
      if sourcenum is not None:
        matplotlib.pyplot.savefig(plotdir+'pdfs/'+'source_%i'%sourcenum+"_cornerplotQU_"+depol+".pdf")
        matplotlib.pyplot.savefig(plotdir+'source_%i'%sourcenum+"_cornerplotQU_"+depol+".png")
      else:
        # matplotlib.pyplot.savefig(plotdir+str(x)+"-"+str(y)+"_cornerplotQU_"+depol+".pdf")
        matplotlib.pyplot.savefig(plotdir+str(x)+"-"+str(y)+"_cornerplotQU_"+depol+".png")
      matplotlib.pyplot.close()
    
  return(fit_dataI, fit_data, x, y)
  #return(fit_data, x, y)
  #return(fit_dataI,fit_data)

