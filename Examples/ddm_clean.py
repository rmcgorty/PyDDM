import sys
import glob
import numpy as np
import scipy
from scipy.optimize import leastsq, least_squares
from scipy.special import gamma
import pylab
import socket

comp_name = socket.gethostname()

if comp_name=='D1K6SR52' or comp_name=='D1MN8282':
    sys.path.append("C:\\Users\\rmcgorty\\Dropbox\\pycode")
elif comp_name=='yoga_mcg':
    sys.path.append("C:\\Users\\Ryan\\Dropbox\\pycode")
import tiff_file
import radiav
import mpfit

#This function is used to determine a new time when a distribution
# of decay times are present
newt = lambda t,s: (1./s)*gamma(1./s)*t

kbt = 1.38064852e-23 * (273+20) #thermal energy in SI units 

def genLogDist(start,stop,numPoints,base=4):
    '''
    This function will generate a logarithmically spaced set of numbers.
    This is for generating the times over which to calculate the intermediate
      scattering function.
    :param start: First time delay (usually 1)
    :param stop: Last time delay (often 598 or 998)
    :param numPoints: number of time delays (can be 60 for quick calculations)
    :param base: optional parameter, default is 4
    :return: unique list of numbers from start to stop, logarithmically spaced
    '''
    newpts = [] #empty list
    
    #first, will make list of numbers log-spaced longer than intended list.
    #This is due to fact that after converting list to integers, we will have
    # multiple duplicates. So to generate enough *unique* numbers we need to
    # first make a longer-than-desired list. 
    prepoints = 4*numPoints 
    numpts = prepoints
    while abs(numpts-numPoints)>0 and (prepoints>10):
        newstart = np.log(start)/np.log(base) #find log of start
        newstop = np.log(stop)/np.log(base) #find log of stop
        
        #generate long list of numbers
        pts = np.logspace(newstart, newstop, num=prepoints, base=base, dtype=np.int)
        
        #identify unique numbers in the list
        newpts = np.unique(pts)
        numpts = len(newpts) #how long is this list of unique numbers
        prepoints = prepoints-1
        if numpts==numPoints:
            return newpts
    return newpts

def getFFTDiffsAtTimes(imageFile, dts, limitImsTo=None, every=None, shiftAtEnd=False, noshift=False, submean=True):
    '''
    This code calculates the image structure function for the series of images
    in imageFile at the lag times specified in dts. 
    :param imageFile: Either a string specifying the location of the data or the data itself as a numpy array
    :param dts: 1D array of delay times
    :param limitImsTo: defaults to None
    :param every: defaults to None
    :param shiftAtEnd: defaults to False
    :param noshift: defaults to False
    :param submean: defaults to true
    :return: two numpy arrays: the fft'd data and the list of times
    '''
    if isinstance(imageFile, np.ndarray):
        ims = imageFile
    elif isinstance(imageFile, basestring):
        ims = tiff_file.imread(imageFile)
    else:
        print("Not sure what you gave for imageFile")
        return 0
    if limitImsTo is not None:
        ims = ims[:limitImsTo]
    if every is not None:
        ims = ims[::every]

    #Determines the dimensions of the data set (number of frames, x- and y-resolution in pixels
    ntimes, ndx, ndy = ims.shape

    #Initializes array for Fourier transforms of differences
    fft_diffs = np.zeros((len(dts), ndx, ndy),dtype=np.float)

    steps_in_diffs = np.ceil(dts/3.0).astype(np.int)


    j=0
    
    if noshift:
        shiftAtEnd=True

    #Loops over each delay time
    for k,dt in enumerate(dts):
        
        if dt%15 == 0:
            print("Running dt=%i...\n" % dt)

        #Calculates all differences of images with a delay time dt
        all_diffs = ims[dt:].astype(np.float) - ims[0:(-1*dt)].astype(np.float)

        #Rather than FT all image differences of a given lag time, only select a subset
        all_diffs_new = all_diffs[0:-1:steps_in_diffs[k],:,:]

        #Loop through each image difference and FT
        for i in range(0,all_diffs_new.shape[0]):
            if shiftAtEnd:
                temp = np.fft.fft2(all_diffs_new[i] - all_diffs_new[i].mean())
            else:
                if submean:
                    temp = np.fft.fftshift(np.fft.fft2(all_diffs_new[i]-all_diffs_new[i].mean()))
                else:
                    temp = np.fft.fftshift(np.fft.fft2(all_diffs_new[i]))

            fft_diffs[j] = fft_diffs[j] + abs(temp*np.conj(temp))/(ndx*ndy)

        #Divide the running sum of FTs to get the average FT of the image differences of that lag time
        fft_diffs[j] = fft_diffs[j] / (all_diffs_new.shape[0])
        
        if shiftAtEnd and not noshift:
            fft_diffs[j] = np.fft.fftshift(fft_diffs[j])

        #fft_diffs[j] = np.fft.fftshift(np.fft.fft2(all_diffs.mean(axis=0)-all_diffs.mean()))
        j = j+1

    return fft_diffs, dts

def getFFTDiffs(imageFile, times, limitImsTo=None, every=None, shiftAtEnd=False, noshift=False, submean=True,
                returncomp = False):
    if isinstance(imageFile, np.ndarray):
        ims = imageFile
    elif isinstance(imageFile, basestring):
        ims = tiff_file.imread(imageFile)
    else:
        print("Not sure what you gave for imageFile")
        return 0
    if limitImsTo is not None:
        ims = ims[:limitImsTo]
    if every is not None:
        ims = ims[::every]

    ntimes, ndx, ndy = ims.shape

    dts = np.arange(times,dtype=np.uint16)+1

    fft_diffs = np.zeros((len(dts), ndx, ndy),dtype='complex128')

    j=0
    
    if noshift:
        shiftAtEnd=True

    for dt in dts:
        
        if dt%15 == 0:
            print("Running dt=%i...\n" % dt)
        
        all_diffs = ims[dt:].astype(np.float) - ims[0:(-1*dt)].astype(np.float)

        for i in range(0,all_diffs.shape[0]):
            if shiftAtEnd:
                temp = np.fft.fft2(all_diffs[i] - all_diffs[i].mean())
            else:
                if submean:
                    temp = np.fft.fftshift(np.fft.fft2(all_diffs[i]-all_diffs[i].mean()))
                else:
                    temp = np.fft.fftshift(np.fft.fft2(all_diffs[i]))
            if returncomp:
                fft_diffs[j] = fft_diffs[j] + temp/(ndx*ndy)
            else:
                fft_diffs[j] = fft_diffs[j] + abs(temp*np.conj(temp))/(ndx*ndy)
        fft_diffs[j] = fft_diffs[j] / (all_diffs.shape[0])
        
        if shiftAtEnd and not noshift:
            fft_diffs[j] = np.fft.fftshift(fft_diffs[j])

        #fft_diffs[j] = np.fft.fftshift(np.fft.fft2(all_diffs.mean(axis=0)-all_diffs.mean()))
        j = j+1

    return fft_diffs
    
def getFFTDiffsMoreTimes(imageFile, time_start, time_stop, limitImsTo=None, every=None, shiftAtEnd=False, noshift=False, submean=True,
                returncomp = False):
    if isinstance(imageFile, np.ndarray):
        ims = imageFile
    elif isinstance(imageFile, basestring):
        ims = tiff_file.imread(imageFile)
    else:
        print("Not sure what you gave for imageFile")
        return 0
    if limitImsTo is not None:
        ims = ims[:limitImsTo]
    if every is not None:
        ims = ims[::every]

    ntimes, ndx, ndy = ims.shape

    dts = np.arange(time_start,time_stop,dtype=np.uint16)

    fft_diffs = np.zeros((len(dts), ndx, ndy),dtype='complex128')

    j=0
    
    if noshift:
        shiftAtEnd=True

    for dt in dts:
        
        if dt%15 == 0:
            print("Running dt=%i...\n" % dt)
        
        all_diffs = ims[dt:].astype(np.float) - ims[0:(-1*dt)].astype(np.float)

        for i in range(0,all_diffs.shape[0]):
            if shiftAtEnd:
                temp = np.fft.fft2(all_diffs[i] - all_diffs[i].mean())
            else:
                if submean:
                    temp = np.fft.fftshift(np.fft.fft2(all_diffs[i]-all_diffs[i].mean()))
                else:
                    temp = np.fft.fftshift(np.fft.fft2(all_diffs[i]))
            if returncomp:
                fft_diffs[j] = fft_diffs[j] + temp/(ndx*ndy)
            else:
                fft_diffs[j] = fft_diffs[j] + abs(temp*np.conj(temp))/(ndx*ndy)
        fft_diffs[j] = fft_diffs[j] / (all_diffs.shape[0])
        
        if shiftAtEnd and not noshift:
            fft_diffs[j] = np.fft.fftshift(fft_diffs[j])

        #fft_diffs[j] = np.fft.fftshift(np.fft.fft2(all_diffs.mean(axis=0)-all_diffs.mean()))
        j = j+1

    return fft_diffs

def dTheory(x,a1,t1,bg,s1,a2,t2,s2):
    g1 = np.exp(-1 * (x / t1)**s1)
    g2 = np.exp(-1 * (x / t2)**s2)
    d = a1 * (1 - g1) + a2 * (1 - g2) + bg
    return d
    
def dTheorySingle(x,a1,t1,bg,s1=1.0):
    g1 = np.exp(-1 * (x / t1))**s1
    d = a1 * (1 - g1) + bg
    return d

def dTheoryISF(x,c,t,s):
    '''
    x: independent variable: our lag time
    c: the nonergodicity parameter --> SET TO 0 here
    t: relaxation time (tau)
    s: stretching exponent ("p" in Cho et al 2020)
    '''
    g1 = np.exp(-1.0*(x/t)**s)
    d = g1
    return d
    

def dTheoryNonErgISF(x,c,t,s):
    '''
    x: independent variable: our lag time
    c: the nonergodicity parameter
    t: relaxation time (tau)
    s: stretching exponent ("p" in Cho et al 2020)
    '''
    g1 = np.exp(-1.0*(x/t)**s)
    d = ((1-c)*g1) + c
    return d

def dTheoryTwoModeISF(x,c,t1,s,t2,a,Z):
    '''
    From Wilson et. al PRL 2011
    
    x: independent variable: our lag time
    c: the nonergodicity parameter
    t1: subdiff. relaxation time (tau)
    s: stretching exponent ("p" in Cho et al 2020)
    t2: ballistic relaxation time, 1/qv
    a: proportion of population that is moving ballistically
    Z: Schulz distribution number
    '''
    theta = (x / t2)/(Z + 1.0)  
    VDist = ((Z + 1.0)/((Z * x)/t2)) * np.sin(Z*np.arctan(theta))/((1.0 + theta**2.0)**(Z/2.0))
    g1 = np.exp(-1.0*(x/t1)**s)
    d = ((1.0-c)*g1*((1.0-a)+a*VDist)) + c
    return d


def returnReasonableParams(d=None, fps=40.0, double=True, stretched=True, bg=100):
    '''
    Function to return reasonable parameters for fits
    The parameters are:
        * amplitude
        * decay time
        * background
        * alpha (stretching exponent)
    :param d:
    :param double:
    :param stretched:
    :return:
    '''
    params = np.array([1e8, 15.0, bg, 0.99, 1e6, 0.8, 1.0])
    if d is not None:
        params[2] = bg
        params[0] = (d.max() - bg)*0.85
    w = np.where((d-params[2])>(0.6*(d.max()-params[2])))
    if len(w[0])>0:
        params[1]=1*(w[0][0]/fps)
    else:
        params[1] = 1./fps
    if params[1]==0:
        params[1]=3./fps
    minpars = np.array([0, 1e-6, 0, 0.1, 0, 1e-6, 0.1])
    maxpars = np.array([1e22, 1e4, 1e18, 2.0, 1e12, 1e4, 2.0])
    fixed = np.repeat(False, len(params))
    limitedmin = np.repeat(True, len(params))
    limitedmax = np.repeat(True, len(params))
    if stretched==False:
        params[3]=1.0
        params[6]=1.0
        fixed[3] = True
        fixed[6] = True
    if double==False:
        params[4] = 0 #amplitude of 2nd exponential
        fixed[4] = True
        fixed[5] = True #tau of 2nd exponential
        fixed[6] = True #alpha (str exp) of 2nd exponential
    else:
        guess_amp = d[-50:].mean() - bg
        params[0] = 0.9*guess_amp
        params[4] = 0.1*guess_amp
        params[5] = 0.1*params[1]
    return params, minpars, maxpars, limitedmin, limitedmax, fixed


def twomodeFit_ISF(dData, times, params, minpars, maxpars, limitedmin, limitedmax, fixed, err=None, logfit=False,maxiter=600,
               factor=1e-3, quiet=False):
    parinfo = [
        {'n': 0, 'value': params[0], 'limits': [minpars[0], maxpars[0]], 'limited': [limitedmin[0], limitedmax[0]],
         'fixed': fixed[0], 'parname': "Nonerg", 'error': 0, 'step':0},
        {'n': 1, 'value': params[1], 'limits': [minpars[1], maxpars[1]], 'limited': [limitedmin[1], limitedmax[1]],
         'fixed': fixed[1], 'parname': "tau", 'error': 0, 'step':0},
        {'n': 2, 'value': params[2], 'limits': [minpars[2], maxpars[2]], 'limited': [limitedmin[2], limitedmax[2]],
         'fixed': fixed[2], 'parname': "Stretchexp", 'error': 0, 'step':0},
        {'n': 3, 'value': params[3], 'limits': [minpars[3], maxpars[3]], 'limited': [limitedmin[3], limitedmax[3]],
         'fixed': fixed[3], 'parname': "tau2", 'error': 0, 'step':0},
        {'n': 4, 'value': params[4], 'limits': [minpars[4], maxpars[4]], 'limited': [limitedmin[4], limitedmax[4]],
         'fixed': fixed[4], 'parname': "proportion", 'error': 0, 'step':0},
        {'n': 5, 'value': params[5], 'limits': [minpars[5], maxpars[5]], 'limited': [limitedmin[5], limitedmax[5]],
         'fixed': fixed[5], 'parname': "schulz", 'error': 0, 'step':0}
    ]

    def mpfitfun(x, y, err, logfit):
        if err is None:
            def f(p, fjac=None):
                if logfit:
                    return [0, (np.log(y) - np.log(dTheoryTwoModeISF(x, *p)))]
                else:
                    return [0, (y - dTheoryTwoModeISF(x, *p))]
        else:
            def f(p, fjac=None):
                return [0, (y - dTheoryTwoModeISF(x, *p)) / err]
        return f
    mp = mpfit.mpfit(mpfitfun(times, dData, err, logfit), parinfo=parinfo, quiet=quiet, maxiter=maxiter,factor=factor)

    if mp.status == 0:
        raise Exception(mp.errmsg)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm

    return mpp, dTheoryTwoModeISF(times,*mpp) ,mpperr,chi2

def newFit_ISF(dData, times, params, minpars, maxpars, limitedmin, limitedmax, fixed, err=None, logfit=False,maxiter=600,
               factor=1e-3, quiet=False):
    parinfo = [
        {'n': 0, 'value': params[0], 'limits': [minpars[0], maxpars[0]], 'limited': [limitedmin[0], limitedmax[0]],
         'fixed': fixed[0], 'parname': "Nonerg", 'error': 0, 'step':0},
        {'n': 1, 'value': params[1], 'limits': [minpars[1], maxpars[1]], 'limited': [limitedmin[1], limitedmax[1]],
         'fixed': fixed[1], 'parname': "tau", 'error': 0, 'step':0},
        {'n': 2, 'value': params[2], 'limits': [minpars[2], maxpars[2]], 'limited': [limitedmin[2], limitedmax[2]],
         'fixed': fixed[2], 'parname': "Stretchexp", 'error': 0, 'step':0},
    ]

    def mpfitfun(x, y, err, logfit):
        if err is None:
            def f(p, fjac=None):
                if logfit:
                    return [0, (np.log(y) - np.log(dTheoryNonErgISF(x, *p)))]
                else:
                    return [0, (y - dTheoryNonErgISF(x, *p))]
        else:
            def f(p, fjac=None):
                return [0, (y - dTheoryNonErgISF(x, *p)) / err]
        return f
    mp = mpfit.mpfit(mpfitfun(times, dData, err, logfit), parinfo=parinfo, quiet=quiet, maxiter=maxiter,factor=factor)

    if mp.status == 0:
        raise Exception(mp.errmsg)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm

    return mpp, dTheoryNonErgISF(times,*mpp) ,mpperr,chi2


def newFit(dData, times, params, minpars, maxpars, limitedmin, limitedmax, fixed, err=None, logfit=True,maxiter=600,
           factor=1e-3, quiet=False):
    parinfo = [
        {'n': 0, 'value': params[0], 'limits': [minpars[0], maxpars[0]], 'limited': [limitedmin[0], limitedmax[0]],
         'fixed': fixed[0], 'parname': "Amplitude1", 'error': 0, 'step':0},
        {'n': 1, 'value': params[1], 'limits': [minpars[1], maxpars[1]], 'limited': [limitedmin[1], limitedmax[1]],
         'fixed': fixed[1], 'parname': "Time1", 'error': 0, 'step':0.01},
        {'n': 2, 'value': params[2], 'limits': [minpars[2], maxpars[2]], 'limited': [limitedmin[2], limitedmax[2]],
         'fixed': fixed[2], 'parname': "Background", 'error': 0, 'step':0},
        {'n': 3, 'value': params[3], 'limits': [minpars[3], maxpars[3]], 'limited': [limitedmin[3], limitedmax[3]],
         'fixed': fixed[3], 'parname': "Alpha1", 'error': 0, 'step':0.005},
        {'n': 4, 'value': params[4], 'limits': [minpars[4], maxpars[4]], 'limited': [limitedmin[4], limitedmax[4]],
         'fixed': fixed[4], 'parname': "Amplitude2", 'error': 0, 'step':0},
        {'n': 5, 'value': params[5], 'limits': [minpars[5], maxpars[5]], 'limited': [limitedmin[5], limitedmax[5]],
         'fixed': fixed[5], 'parname': "Time2", 'error': 0, 'step':0.01},
        {'n': 6, 'value': params[6], 'limits': [minpars[6], maxpars[6]], 'limited': [limitedmin[6], limitedmax[6]],
         'fixed': fixed[6], 'parname': "Alpha2", 'error': 0, 'step':0.005}
    ]

    def mpfitfun(x, y, err, logfit):
        if err is None:
            def f(p, fjac=None):
                if logfit:
                    return [0, (np.log(y) - np.log(dTheory(x, *p)))]
                else:
                    return [0, (y - dTheory(x, *p))]
        else:
            def f(p, fjac=None):
                return [0, (y - dTheory(x, *p)) / err]
        return f
    mp = mpfit.mpfit(mpfitfun(times, dData, err, logfit), parinfo=parinfo, quiet=quiet, maxiter=maxiter,factor=factor)

    if mp.status == 0:
        raise Exception(mp.errmsg)
    mpp = mp.params
    mpperr = mp.perror
    chi2 = mp.fnorm

    return mpp, dTheory(times,*mpp) ,mpperr,chi2


    
def newFit_ALL(dData, times, params, minpars, maxpars, limitedmin, limitedmax, 
                      fixed, err=None, logfit=True,maxiter=600,
                      factor=0.3, single=True, startq=10):
    if params.ndim == 1:
        initParams = params.copy()
        allinitParams = False
    else:
        allinitParams = True
        params_all = params.copy()
        initParams = params[startq]
        params = initParams
    ntimes, nqs = dData.shape
    allFitResults = np.zeros((nqs, 7))
    allFitErrors = np.zeros_like(allFitResults)
    allTheory = np.zeros_like(dData)
    allchi2 = np.zeros((nqs))
    for q in range(startq,nqs):
        fit, theory, mpperr, chi2 = newFit(dData[:,q], times, params, minpars, maxpars, limitedmin, limitedmax, fixed, 
                      err=err, logfit=logfit,maxiter=maxiter,factor=factor, quiet=True)
        allFitResults[q] = fit
        allTheory[:,q] = theory
        allFitErrors[q] = mpperr
        allchi2[q] = chi2
        if allinitParams and (q+1<nqs):
            params = params_all[q+1]
        else:
            params = fit
    params = initParams
    for q in range(startq,1,-1):
        fit, theory, mpperr, chi2 = newFit(dData[:,q], times, params, minpars, maxpars, limitedmin, limitedmax, fixed, 
                      err=err, logfit=logfit,maxiter=maxiter,factor=factor, quiet=True)
        allFitResults[q] = fit
        allTheory[:,q] = theory
        allFitErrors[q] = mpperr
        allchi2[q] = chi2
        if allinitParams and (q-1>0):
            params = params_all[q-1]
        else:
            params = fit
    return allFitResults, allTheory, allFitErrors, allchi2
    
def newFitLeastsq(dData, times, params, minpars, maxpars, limitedmin, limitedmax, fixed, err=None, logfit=True,maxiter=600,
           factor=1e-3, single=True):
    parinfo = [
        {'n': 0, 'value': params[0], 'limits': [minpars[0], maxpars[0]], 'limited': [limitedmin[0], limitedmax[0]],
         'fixed': fixed[0], 'parname': "Amplitude1", 'error': 0, 'step':0},
        {'n': 1, 'value': params[1], 'limits': [minpars[1], maxpars[1]], 'limited': [limitedmin[1], limitedmax[1]],
         'fixed': fixed[1], 'parname': "Time1", 'error': 0, 'step':0.01},
        {'n': 2, 'value': params[2], 'limits': [minpars[2], maxpars[2]], 'limited': [limitedmin[2], limitedmax[2]],
         'fixed': fixed[2], 'parname': "Background", 'error': 0, 'step':0},
        {'n': 3, 'value': params[3], 'limits': [minpars[3], maxpars[3]], 'limited': [limitedmin[3], limitedmax[3]],
         'fixed': fixed[3], 'parname': "Alpha1", 'error': 0, 'step':0.005},
        {'n': 4, 'value': params[4], 'limits': [minpars[4], maxpars[4]], 'limited': [limitedmin[4], limitedmax[4]],
         'fixed': fixed[4], 'parname': "Amplitude2", 'error': 0, 'step':0},
        {'n': 5, 'value': params[5], 'limits': [minpars[5], maxpars[5]], 'limited': [limitedmin[5], limitedmax[5]],
         'fixed': fixed[5], 'parname': "Time2", 'error': 0, 'step':0.01},
        {'n': 6, 'value': params[6], 'limits': [minpars[6], maxpars[6]], 'limited': [limitedmin[6], limitedmax[6]],
         'fixed': fixed[6], 'parname': "Alpha2", 'error': 0, 'step':0.005}
    ]

    def leastsqfitfun(p, x, y, logfit, parinfo):
        newp = np.zeros((len(parinfo)))
        j = 0
        for i in range(len(parinfo)):
            if parinfo[i]['fixed']:
                newp[i] = parinfo[i]['value']
            else:
                newp[i] = p[j]
                j = j+1

        if logfit:
            diff = np.log(y) - np.log(dTheory(x, *newp))
            return diff
        else:
            diff = y - dTheory(x, *newp)
            return diff
            
    params_adjust = np.repeat(0,len(fixed)-fixed.sum()).astype(np.float)
    j=0
    for i in range(len(parinfo)):
        if not parinfo[i]['fixed']:
            params_adjust[j] = params[i]
            j=j+1
    
    plsq = leastsq(leastsqfitfun, params_adjust, args = (times, dData, logfit, parinfo))
    
    newplsq = np.zeros((len(parinfo)))
    j=0
    for i in range(len(parinfo)):
        if parinfo[i]['fixed']:
            newplsq[i]=parinfo[i]['value']
        else:
            newplsq[i]=plsq[0][j]
            j=j+1

    return newplsq, dTheory(times,*newplsq)
    
def newFitLeastsq_ALL(dData, times, params, minpars, maxpars, limitedmin, limitedmax, 
                      fixed, err=None, logfit=True,maxiter=600,
                      factor=1e-3, single=True, startq=10):
    ntimes, nqs = dData.shape
    allFitResults = np.zeros((nqs, 7))
    allTheory = np.zeros((ntimes, 7))
    for q in range(startq,nqs):
        fit, theory = newFitLeastsq(dData[:,q], times, params, minpars, maxpars, limitedmin, limitedmax, fixed, 
                      err=err, logfit=logfit,maxiter=maxiter,factor=factor, single=single)
    for q in range(startq,1,-1):
        fit, theory = newFitLeastsq(dData[:,q], times, params, minpars, maxpars, limitedmin, limitedmax, fixed, 
                      err=err, logfit=logfit,maxiter=maxiter,factor=factor, single=single)
                      
                      
                
    
def testingMask(im, angRange=None, rev=False):
    nx,ny = im.shape
    xx = np.arange(-(nx-1)/2., nx/2.)
    yy = np.arange(-(ny-1)/2., ny/2.)
    x,y = np.meshgrid(xx,yy)
    q = np.sqrt(x**2 + y**2)
    angles = np.arctan2(x,y)
    mask = np.ones_like(angles)
    if angRange is not None:
        w1 = np.where(angles>angRange[0])
    else:
        w1 = np.where(angles>(13*np.pi/14))
    mask[w1]=0
    mask = mask * np.rot90(np.rot90(mask))
    mask = mask * np.flipud(mask)
    mask[np.where(mask==0)] = np.nan
    if rev:
        mask = np.rot90(mask)
    return im*mask
    
def embedInNans(arr):
    im_dim_x = arr.shape[1]
    im_dim_y = arr.shape[2]
    larger_dim = max(im_dim_x, im_dim_y)
    mid = larger_dim/2 - 1
    new_arr = np.ones((arr.shape[0], larger_dim, larger_dim))*np.nan
    if im_dim_y>im_dim_x:
        new_arr[:,mid-im_dim_x/2:mid+im_dim_x/2,:] = arr
    else:
        new_arr[:,:,mid-im_dim_x/2:mid+im_dim_x/2] = arr
    return new_arr

def newRadav(im, limangles=False, angRange=None, mask=None, rev=False,
             debug_q = None):
    if mask is None:
        hasMask = False
    else:
        hasMask = True
    nx,ny = im.shape
    xx = np.arange(-(nx-1)/2., nx/2.)
    yy = np.arange(-(ny-1)/2., ny/2.)
    #x,y = np.meshgrid(xx,yy)
    x,y = np.meshgrid(yy,xx)
    q = np.sqrt(x**2 + y**2)
    angles = np.arctan2(x,y)
    
    qx = np.arange(-1*nx/2,nx/2)*(1./nx) * max(nx,ny)
    qy = np.arange(-1*ny/2,ny/2)*(1./ny) * max(nx,ny)
    qxx,qyy = np.meshgrid(qy,qx) #qy,qx is correct order
    q_new = np.sqrt(qxx**2 + qyy**2)
    
    if debug_q is not None:
        return q_new.round().astype(np.int)==debug_q
    
    if mask is None:
        mask = np.ones_like(angles)
    if angRange is not None:
        w1 = np.where(angles>angRange[0])
    else:
        w1 = np.where(angles>(13*np.pi/14))
    if mask is None:
        mask[w1]=0
        mask = mask * np.rot90(np.rot90(mask))
        mask = mask * np.flipud(mask)
        mask[np.where(mask==0)] = np.nan
        if rev:
            mask = np.rot90(mask)
    qr = q_new.round().astype(np.int)
    #rs = np.arange(0,(nx-1)/2)
    rs = np.arange(0,(max(nx,ny)-1)/2) 
    radav = np.zeros((len(rs)),dtype=np.float)
    for i in range(0,len(rs)):
        w = np.where(qr==rs[i])
        if len(w[0])>0:
            if limangles or hasMask:
                newim = im*mask
                radav[i] = np.nanmean(newim[w])
            else:
                radav[i] = np.nanmean(im[w])
        #else:
        #    print i
    return radav

def radialAvFFTs_v2(fft_diff, limangles=False, angRange=None, 
                    mask=None, rev=False):
    if isinstance(fft_diff, basestring):
        fft_diff_data = np.load(fft_diff)
    else:
        fft_diff_data = fft_diff.copy()
    radav = newRadav(fft_diff[0])
    ravs = np.zeros((fft_diff_data.shape[0], len(radav)))
    ravs[0] = radav
    for i in range(1,fft_diff_data.shape[0]):
        ravs[i] = newRadav(fft_diff_data[i], limangles=limangles, angRange=angRange,
                           mask=mask, rev=rev)
    return ravs



def radialAvFFTs(fft_diff, le, xc, yc, av4=False):
    if isinstance(fft_diff, basestring):
        fft_diff_data = np.load(fft_diff)
    else:
        fft_diff_data = fft_diff.copy()


    ravs = np.zeros((fft_diff_data.shape[0],le))

    if av4:
        for i in range(0,ravs.shape[0]):
            r1, nr = radiav.radav(abs(fft_diff[i]), xc, yc, le)
            r2, nr = radiav.radav(abs(fft_diff[i]), xc-1, yc-1, le)
            r3, nr = radiav.radav(abs(fft_diff[i]), xc-1, yc, le)
            r4, nr = radiav.radav(abs(fft_diff[i]), xc, yc-1, le)
            ravs[i] = (r1+r2+r3+r4)/4.0
    else:
        for i in range(0,ravs.shape[0]):
            ravs[i], nr = radiav.radav(abs(fft_diff[i]), xc, yc, le)

    return ravs


def fittingRadAvs(radav, times, p0, plot=True, pd = False, plotLog=True):
    if pd:
        if len(p0)==4:
            plsq = leastsq(dynamicStructFuncPDError, p0, args = (times, radav))
        elif len(p0)==5:
            plsq = leastsq(dynamicStructFuncPD3Error, p0, args = (times, radav))
    else:
        plsq = leastsq(dynamicStructFuncError, p0, args = (times, radav))
    #soln = least_squares(dynamicStructFuncError, p0, args=(times, radav), method='trf', loss='linear',
    #                     tr_solver='exact')
   # plsq = (soln.x, 1)
    #print "plsq: ", plsq
    if plot:
        pylab.figure()
        if plotLog:
            pylab.semilogx(times, radav, 'ro')
        else:
            pylab.plot(times, radav, 'ro')
        if pd:
            if plotLog:
                if len(p0)==4:
                    pylab.semilogx(times, dynamicStructFuncPD(plsq[0],times),'-k',lw=2)
                elif len(p0)==5:
                    pylab.semilogx(times, dynamicStructFuncPD3(plsq[0],times),'-k',lw=2)
            else:
                if len(p0)==4:
                    pylab.plot(times, dynamicStructFuncPD(plsq[0],times),'-k',lw=2)
                elif len(p0)==5:
                    pylab.plot(times, dynamicStructFuncPD3(plsq[0],times),'-k',lw=2)
        else:
            if plotLog:
                pylab.semilogx(times, dynamicStructFunc(plsq[0], times), '-k', lw=2)
            else:
                pylab.plot(times, dynamicStructFunc(plsq[0],times),'-k',lw=2)
        pylab.xlabel('Time')
        pylab.ylabel('D (q,dt)')
    return plsq

def fittingRadAvsDoubleExp(radav, times, p0, plot=True, plotLog=True, log=True, wts=None, fixedB=0):
    
    if fixedB>0:
        newp0 = np.zeros((4,))
        newp0[0] = p0[0]
        newp0[1] = p0[1]
        newp0[2] = p0[3]
        newp0[3] = p0[4]
    else:
        newp0 = p0
        
    plsq = leastsq(dynamicStructFuncDoubleError, newp0, args = (times, radav, log, wts, fixedB))
    
    if fixedB>0:
        new_plsq = np.zeros((5,))
        new_plsq[0] = plsq[0][0]
        new_plsq[1] = plsq[0][1]
        new_plsq[2] = fixedB
        new_plsq[3] = plsq[0][2]
        new_plsq[4] = plsq[0][3]
        plsq = (new_plsq,1)

    if plot:
        f, axarr = pylab.subplots(4, sharex=True)
        theory = dynamicStructFuncDouble(plsq[0], times)
        exp1, exp2, bg = dynamicStructFuncDouble_Each(plsq[0], times)
        err = radav - theory
        if plotLog:
            axarr[0].semilogx(times, radav, 'ro')
            axarr[0].semilogx(times, theory, '-k', lw=2)
            axarr[1].semilogx(times, abs(err), 'ro')
            #axarr[2].semilogx(times, radav-bg, 'bo')
            #axarr[2].semilogx(times, exp1, '-g', lw=2)
            #axarr[2].semilogx(times, exp2, '-m', lw=2)
        else:
            #axarr[0].plot(times, radav, 'ro')
            #axarr[0].plot(times, theory,'-k',lw=2)
            axarr[1].plot(times, err, 'ro')
            dif = radav-bg
            axarr[0].plot(times, abs(dif), 'bo')
            axarr[0].plot(times, abs(exp1), '-g', lw=2)
            axarr[0].plot(times, abs(exp2), '-m', lw=2)
            axarr[2].plot(times, abs(dif - exp1), 'mo')
            axarr[3].plot(times, abs(dif - exp2), 'go')
               
        axarr[1].set_xlabel('Time')
        axarr[1].set_ylabel('Error')
        axarr[0].set_ylabel('D (q,dt) - bg')
    return plsq

def fittingRadAvsDoubleExpPD(radav, times, p0, plot=True, plotLog=True, log=True, wts=None, fixedB=0):
    
    if fixedB>0:
        newp0 = np.zeros((6,))
        newp0[0:2] = p0[0:2]
        newp0[2:] = p0[3:]
        temp = leastsq(dynamicStructFuncDoublePDError, newp0, args = (times, radav, log, wts, fixedB))
        temp2 = np.zeros((7,))
        temp2[0:2] = temp[0][0:2]
        temp2[2] = fixedB
        temp2[3:] = temp[0][2:]
        plsq = (temp2,1)
    else:
        plsq = leastsq(dynamicStructFuncDoublePDError, p0, args = (times, radav, log, wts))

    if plot:
        f, axarr = pylab.subplots(2, sharex=True)
        theory = dynamicStructFuncDoublePD(plsq[0], times)
        err = radav - theory
        if plotLog:
            axarr[0].semilogx(times, radav, 'ro')
            axarr[0].semilogx(times, theory, '-k', lw=2)
            axarr[1].semilogx(times, abs(err), '-ro')
        else:
            axarr[0].plot(times, radav, 'ro')
            axarr[0].plot(times, theory,'-k',lw=2)
            axarr[1].plot(times, err, '-ro')
               
        axarr[1].set_xlabel('Time')
        axarr[0].set_ylabel('D (q,dt)')
        axarr[1].set_ylabel('Error')
    return plsq
    
def fittingRadAvsStretched(radav, times, p0, plot=True, plotLog=True, fixedalpha=0, log=False, wts=None, double=False, fixedB = 0):
    
    if double:
        if fixedalpha>0 and fixedB==0:
            temp = leastsq(dynamicStructFuncHoldStretchedDoubleError, p0, args=(times, radav, fixedalpha, fixedB, log, wts))
            temp2 = np.zeros((6,))
            temp2[0:5] = temp[0][0:5].copy()
            temp2[5] = fixedalpha
            plsq = (temp2, temp[1])
        elif fixedalpha>0 and fixedB>0:
            newp0 = np.hstack((p0[0:2],p0[3:]))
            temp = leastsq(dynamicStructFuncHoldStretchedDoubleError, newp0, args=(times, radav, fixedalpha, fixedB, log))
            temp2 = np.zeros((6,))
            temp2[0:2] = temp[0][0:2].copy()
            temp2[2] = fixedB
            temp2[3] = temp[0][2].copy()
            temp2[4] = temp[0][3].copy()
            temp2[5] = fixedalpha
            plsq = (temp2,temp[1])
        elif fixedalpha==0 and fixedB>0:
            newp0 = np.hstack((p0[0:2],p0[3:]))
            temp = leastsq(dynamicStructFuncStretchedDoubleError, newp0, args=(times, radav, log, wts, fixedB))
            temp2 = np.zeros((6,))
            temp2[0:2] = temp[0][0:2].copy()
            temp2[2] = fixedB
            temp2[3] = temp[0][2]
            temp2[4] = temp[0][3]
            temp2[5] = temp[0][4]
            plsq = (temp2, temp[1])
        else:
            plsq = leastsq(dynamicStructFuncStretchedDoubleError, p0, args=(times, radav, log, wts, fixedB))
        fitres = dynamicStructFuncStretchedDouble(plsq[0],times)
    else:
        if fixedalpha>0 and fixedB==0:
            temp = leastsq(dynamicStructFuncHoldStretchError, p0[0:3], args=(times, radav, fixedalpha, log, wts))
            temp2 = np.zeros((4,))
            temp2[0:3] = temp[0].copy()
            temp2[3] = fixedalpha
            plsq = (temp2, 1)
        elif fixedalpha==0 and fixedB>0:
            newp0 = np.hstack((p0[0:2],p0[3]))
            temp = leastsq(dynamicStructFuncStretchHoldBErr, newp0, args=(times, radav, fixedB, log))
            temp2 = np.zeros((4,))
            temp2[0:2] = temp[0][0:2].copy()
            temp2[2] = fixedB
            temp2[3] = temp[0][2].copy()
            plsq = (temp2,1)
        elif fixedalpha>0 and fixedB>0:
            newp0 = p0[0:2]
            temp = leastsq(dynamicStructFuncHoldStretchHoldBErr, newp0, args=(times, radav, fixedalpha, fixedB, log))
            temp2 = np.zeros((4,))
            temp2[0:2] = temp[0][0:2].copy()
            temp2[2] = fixedB
            temp2[3] = fixedalpha
            plsq = (temp2,1)
        elif fixedalpha==0 and fixedB==0:
            plsq = leastsq(dynamicStructFuncStretchedError, p0, args = (times, radav, log, wts),
                           maxfev=100000)
        fitres = dynamicStructFuncStretched(plsq[0], times)
                       
    if plot:
        f, axarr = pylab.subplots(2, sharex=True)
        err = radav-fitres
        if plotLog:
            axarr[0].semilogx(times, radav, 'ro')
            axarr[0].semilogx(times, fitres, '-k', lw=2)
            #axarr[1].semilogx(times, abs(err), 'ro')
        else:
            axarr[0].plot(times, radav,'ro')
            axarr[0].plot(times, fitres, '-k', lw=2)
            axarr[1].plot(times, err, 'ro')
    if plot:    
        axarr[1].set_xlabel('Time')
        axarr[1].set_ylabel('Error')
        axarr[0].set_ylabel('D (q,dt)')
    return plsq
    
def fittingRadAvsStretchedLog(radav, times, p0, plot=True, plotLog=True):
    plsq = leastsq(dynamicStructFuncStretchedLogError, p0, args = (times, radav))
    
    if plot:
        pylab.figure()
        pylab.semilogx(times, radav, 'ro')
        pylab.semilogx(times, dynamicStructFuncStretched(plsq[0], times), '-k', lw=2)
        pylab.xlabel('Time')
        pylab.ylabel('D (q,dt)')
    return plsq
    
def fittingRadAvsWithLog(radav, times, p0, plot=True, pd=False, plotLog=True, Z=False, S=False):
    newTimes = times.copy()
    newp0 = p0.copy()
    #newp0[1] = np.exp(p0[1])
    if pd and (not Z):
        if len(p0)==4:
            plsq = leastsq(dynamicStructFuncLogErrorPD, p0, args=(times, radav))
        if len(p0)==5:
            plsq = leastsq(dynamicStructFuncLogErrorPD3, p0, args=(times, radav))
    elif Z and (not pd) and (not S):
        plsq = leastsq(dynamicStructFuncZLogError, p0, args=(times, radav))
    elif Z and pd and (not S):
        plsq = leastsq(dynamicStructFuncZPDLogError, p0, args=(times, radav))
    elif not(pd) and Z and S:
        plsq = leastsq(dynamicStructFuncZSLogError, p0, args=(times, radav))
    else:
        temp = leastsq(dynamicStructFuncLogError, newp0, args = (newTimes, radav))
        plsq = temp
    #soln = least_squares(dynamicStructFuncLogError, p0, args=(times, radav), method='lm',
    #                     x_scale = np.array([1000,0.1,1000]))
    #plsq = (soln.x, 1)
    #plsq[0][1] = np.exp(temp[0][1])
    if plot:
        f, axarr = pylab.subplots(2, sharex=True)
        if plotLog:
            axarr[0].semilogx(times, radav, 'ro')
        else:
            axarr[0].plot(times, radav, 'ro')
        if pd and (not Z):
            if plotLog:
                if len(p0)==4:
                    theory = dynamicStructFuncPD(plsq[0],times)
                    axarr[0].semilogx(times, theory,'-k',lw=2)
                    err = radav - theory
                    axarr[1].semilogx(times, err,'-ro')
                elif len(p0)==5:
                    theory = dynamicStructFuncPD3(plsq[0],times)
                    axarr[0].semilogx(times, theory,'-k',lw=2)
                    err = radav - theory
                    axarr[1].semilogx(times, err,'-ro')
            else:
                if len(p0)==4:
                    theory = dynamicStructFuncPD(plsq[0],times)
                    axarr[0].plot(times, theory,'-k',lw=2)
                    err = radav - theory
                    axarr[1].plot(times, err,'-ro')
                elif len(p0)==5:
                    theory = dynamicStructFuncPD3(plsq[0],times)
                    axarr[0].plot(times, theory,'-k',lw=2)
                    err = radav - theory
                    axarr[1].plot(times, err,'-ro')
        elif Z and (not pd) and (not S):
            theory = dynamicStructFuncZ(plsq[0],times)
            err = radav - theory
            if plotLog:
                axarr[0].semilogx(times, theory,'-k',lw=2)
                axarr[1].semilogx(times, err,'-ro')
            else:
                axarr[0].plot(times, theory,'-k',lw=2)
                axarr[1].plot(times, err,'-ro')
        elif Z and pd and (not S):
            theory = dynamicStructFuncZPD(plsq[0],times)
            err = radav - theory
            if plotLog:
                axarr[0].semilogx(times, theory,'-k',lw=2)
                axarr[1].semilogx(times, err,'-ro')
            else:
                axarr[0].plot(times, theory,'-k',lw=2)
                axarr[1].plot(times, err,'-ro')
        elif Z and S and (not pd):
            theory = dynamicStructFuncZS(plsq[0],times)
            err = radav - theory
            if plotLog:
                axarr[0].semilogx(times, theory,'-k',lw=2)
                axarr[1].semilogx(times, err,'-ro')
            else:
                axarr[0].plot(times, theory,'-k',lw=2)
                axarr[1].plot(times, err,'-ro')
        else:
            theory = dynamicStructFunc(plsq[0], times)
            err = radav - theory 
            if plotLog:
                axarr[0].semilogx(times, theory, '-k', lw=2)
                axarr[1].semilogx(times, err,'-ro')
            else:
                axarr[0].plot(times, theory,'-k',lw=2)
                axarr[1].plot(times, err,'-ro')
        axarr[1].set_xlabel('Time')
        axarr[0].set_ylabel('D (q,dt)')
        axarr[1].set_ylabel('Error')
    return plsq
    
def fittingRadAvsWithLogHoldB(radav, times, p0, B, plot=True):
    newTimes = times.copy()
    newp0 = p0.copy()
    #newp0[1] = np.exp(p0[1])
    temp = leastsq(dynamicStructFuncLogErrorHoldB, newp0, args = (newTimes, radav, B))
    print("temp: ", temp)
    plsq = np.zeros((3,))
    plsq[0:2] = temp[0]
    plsq[2] = B
    #plsq[0][1] = np.exp(temp[0][1])
    if plot:
        pylab.figure()
        pylab.semilogx(times, radav, 'ro')
        pylab.semilogx(times, dynamicStructFunc(plsq, times), '-k', lw=2)
        pylab.xlabel('Time')
        pylab.ylabel('D (q,dt)')
    return (plsq,1)
    
def fittingAllRadAvs(radavs, times, p0, plotEvery=None, noreset=True, fitWithLog=False, holdBas = None,
                     pd=False, plotLog=True, stretched=False, fixedalpha=0, wts=None, double=False,
                     fixedB = 0, startingATB=None, resetAt = 2, Z=False):
    aParam = np.zeros((radavs.shape[1]))
    tParam = np.zeros_like(aParam)
    bParam = np.zeros_like(aParam)
    stretchParam = np.zeros_like(aParam)
    t2Param = np.zeros_like(aParam)
    a2Param = np.zeros_like(aParam)
    pd1Param = np.zeros_like(aParam)
    pd2Param = np.zeros_like(aParam)
    zParam = np.zeros_like(aParam)
    err = np.zeros_like(aParam)
    newp0 = p0.copy()
    '''
    #not used since first fit is usually nans
    if startingATB is not None:
        newp0[0] = startingATB[0][0]
        newp0[1] = startingATB[1][0]
        newp0[2] = startingATB[2][0]
    '''   
    if holdBas is not None:
        bparam_fixed = holdBas
    for i in range(0,radavs.shape[1]):
        if plotEvery is not None:
            if i%plotEvery == 0:
                toPlot=True
            else:
                toPlot=False
        else:
            toPlot=False
        if fitWithLog:
            if holdBas is not None:
                res = fittingRadAvsWithLogHoldB(radavs[:,i], times, newp0[:2], bparam_fixed, plot=toPlot)
            if Z and stretched and (not pd):
                res = fittingRadAvsWithLog(radavs[:,i], times, newp0, pd=pd, plot=toPlot, plotLog=plotLog, Z=Z, S=stretched)
                zParam[i] = res[0][3]
                stretchParam[i] = res[0][4]
            elif ((not stretched) and (not double)):
                res = fittingRadAvsWithLog(radavs[:,i], times, newp0, pd=pd, plot=toPlot, plotLog=plotLog, Z=Z)
                if pd:
                    if Z:
                        pd1Param[i] = res[0][4]
                        zParam[i] = res[0][3]
                    else:
                        pd1Param[i] = res[0][3]
                        if len(p0)==5:
                            pd2Param[i] = res[0][4]
                elif Z and (not pd):
                    zParam[i] = res[0][3]
            elif ((not stretched) and double and (not pd)):
                res = fittingRadAvsDoubleExp(radavs[:,i], times, newp0, plot=toPlot, plotLog=plotLog, log=fitWithLog, 
                                             wts=wts, fixedB=fixedB)
            if stretched and (not Z):
                res = fittingRadAvsStretched(radavs[:,i], times, newp0, plot=toPlot, plotLog=plotLog,
                                                fixedalpha=fixedalpha, log=fitWithLog, wts=wts, double=double,
                                                fixedB = fixedB)
            if double and pd:
                res = fittingRadAvsDoubleExpPD(radavs[:,i], times, newp0, plot=toPlot, plotLog=plotLog,
                                               log=fitWithLog, wts=wts, fixedB=fixedB)
        else:
            if stretched and (not Z):
                res = fittingRadAvsStretched(radavs[:,i], times, newp0, plot=toPlot, plotLog=plotLog,
                                             fixedalpha=fixedalpha, log=fitWithLog, wts=wts, double=double,
                                             fixedB=fixedB)
            elif (double and pd):
                res = fittingRadAvsDoubleExpPD(radavs[:,i], times, newp0, plot=toPlot, plotLog=plotLog,
                                               log=fitWithLog, wts=wts)
            else:
                res = fittingRadAvs(radavs[:,i], times, newp0, plot=toPlot, pd=pd, plotLog=plotLog)
                pd1Param[i] = res[0][3]
                if len(p0)==5:
                    pd2Param[i] = res[0][4]
        if toPlot:
            pylab.title("Fitting..." + str(i).zfill(2) + " ... a1=" + str(res[0][0]))
        print(str(i) + ": ", res)
        aParam[i] = res[0][0]
        tParam[i] = res[0][1]
        bParam[i] = res[0][2]
        if Z and stretched and (not pd):
            err[i] = np.sum(dynamicStructFuncZSError(res[0], times, radavs[:,i])**2)
        elif Z and (not pd) and (not stretched):
            err[i] = np.sum(dynamicStructFuncZError(res[0], times, radavs[:,i])**2)
        elif Z and pd and (not stretched):
            err[i] = np.sum(dynamicStructFuncZPDError(res[0], times, radavs[:,i])**2)
        elif double and (not pd):
            err[i] = np.sum(dynamicStructFuncDoubleError(res[0], times, radavs[:,i])**2)
        elif double and pd:
            err[i] = np.sum(dynamicStructFuncDoublePDError(res[0],times,radavs[:,i])**2)
        elif (not double) and (not stretched) and pd and (not Z):
            if len(p0)==4:
                err[i] = np.sum(dynamicStructFuncPDError(res[0],times,radavs[:,i])**2)
            elif len(p0)==5:
                err[i] = np.sum(dynamicStructFuncPD3Error(res[0],times,radavs[:,i])**2)
        elif (not double) and (not Z) and (not pd) and stretched:
            err[i] = np.sum(dynamicStructFuncStretchedError(res[0], times, radavs[:,i])**2)
        else:
            err[i] = np.sum(dynamicStructFuncError(res[0], times, radavs[:,i])**2)
        if i>resetAt and noreset:
            newp0 = res[0].copy()
        if (startingATB is not None) and (i<(radavs.shape[1]-1)):
            newp0[0] = startingATB[0][i+1]
            newp0[1] = startingATB[1][i+1]
            newp0[2] = startingATB[2][i+1]
            #newp0[5] = 0.85
        if ((len(res[0])>3) and stretched and (not double) and (not Z)):
            stretchParam[i] = res[0][3]
        if (not stretched):
            stretchParam[i]=0
        if double:
            t2Param[i] = res[0][4]
            a2Param[i] = res[0][3]
            if pd:
                pd1Param[i] = res[0][5]
                pd2Param[i] = res[0][6]
            elif stretched:
                stretchParam[i] = res[0][5]
    if Z and (not pd) and (not stretched):
        return aParam, tParam, bParam, zParam, err
    elif Z and stretched and (not pd):
        return aParam, tParam, bParam, zParam, stretchParam, err
    elif Z and pd and (not stretched):
        return aParam, tParam, bParam, zParam, pd1Param, err
    elif double and (not pd) and (not stretched):
        return aParam, tParam, bParam, a2Param, t2Param, err
    elif (not double) and (not stretched) and pd:
        if len(p0)==4:
            return aParam, tParam, bParam, pd1Param, err
        elif len(p0)==5:
            return aParam, tParam, bParam, pd1Param, pd2Param, err
    elif double and pd:
        return aParam, tParam, bParam, a2Param, t2Param, pd1Param, pd2Param, err
    elif double and stretched:
        return aParam, tParam, bParam, a2Param, t2Param, stretchParam, err
    return aParam,tParam,bParam,stretchParam,err

def fittingAllRadAvsWithLogs(radavs, times, p0, plotEvery=None, reset=True):
    aParam = np.zeros((radavs.shape[1]))
    tParam = np.zeros_like(aParam)
    bParam = np.zeros_like(aParam)
    err = np.zeros_like(aParam)
    newp0 = p0.copy()
    for i in range(0,radavs.shape[1]):
        if plotEvery is not None:
            if i%plotEvery == 0:
                toPlot=True
            else:
                toPlot=False
        else:
            toPlot=False
        res = fittingRadAvsWithLog(radavs[:,i], times, newp0, plot=toPlot)
        if toPlot:
            pylab.title("Fitting..." + str(i).zfill(2))
        print(str(i) + ": ", res)
        aParam[i] = res[0][0]
        tParam[i] = res[0][1]
        bParam[i] = res[0][2]
        err[i] = np.sum(dynamicStructFuncError(res[0], times, radavs[:,i])**2)
        if i>2 and reset:
            newp0 = res[0].copy()
    return aParam,tParam,bParam,err
    
def fittingAllRadAvsWithLogsHoldB(radavs, times, p0, B, plotEvery=None, reset=True):
    aParam = np.zeros((radavs.shape[1]))
    tParam = np.zeros_like(aParam)
    bParam = np.zeros_like(aParam)
    err = np.zeros_like(aParam)
    newp0 = p0.copy()
    for i in range(0,radavs.shape[1]):
        if plotEvery is not None:
            if i%plotEvery == 0:
                toPlot=True
            else:
                toPlot=False
        else:
            toPlot=False
        res = fittingRadAvsWithLogHoldB(radavs[:,i], times, newp0, B, plot=toPlot)
        if toPlot:
            pylab.title("Fitting..." + str(i).zfill(2))
        print(str(i) + ": ", res)
        aParam[i] = res[0][0]
        tParam[i] = res[0][1]
        bParam[i] = B
        err[i] = np.sum(dynamicStructFuncError(res[0], times, radavs[:,i])**2)
        if i>2 and reset:
            newp0 = res[0][0:2].copy()
    return aParam,tParam,bParam,err

def estimatingOffsets(data, times):

    #First get the t->0 offset (paramter C if y = A(1-e^-t/B)+C)
    firstSlope = (data(1)-data(0))/(times(1)-times(0))
    secondSlope = (data(2)-data(1))/(times(2)-times(1))
    diffSlope = firstSlope-secondSlope
    if diffSlope>0:
        slope = firstSlope+diffSlope
    else:
        slope = firstSlope
    paramC = data(0) - (slope*times(0))
   
    
    return paramC
    
    
def dynamicStructFunc(params, times):
    g = np.exp(-1*times/params[1])
    d = params[0]*(1-g) + abs(params[2])
    return d
    
def dynamicStructFuncDouble(params, times):
    g1 = np.exp(-1*(times/params[1]))
    g2 = np.exp(-1*(times/params[4]))
    d = params[0]*(1-g1) + params[3]*(1-g2) + params[2]
    return d
    
def dynamicStructFuncDouble_Each(params, times):
    g1 = np.exp(-1*(times/params[1]))
    g2 = np.exp(-1*(times/params[4]))
    d = params[0]*(1-g1) + params[3]*(1-g2) + params[2]
    return params[0]*(1-g1), params[3]*(1-g2), params[2]
    
def dynamicStructFuncDoublePD(params, times):
    g1 = np.exp(-1*(times/params[1]))*(1+(abs(params[5])*times*times/2.0))
    g2 = np.exp(-1*(times/params[4]))*(1+(abs(params[6])*times*times/2.0))
    d = params[0]*(1-g1) + params[3]*(1-g2) + params[2]
    return d
    
def dynamicStructFuncStretched(params, times):
    g = np.exp(-1*(times/params[1])**params[3])
    d = params[0]*(1-g) + abs(params[2])
    return d
    
def dynamicStructFuncStretchedDouble(params, times):
    g1 = np.exp(-1*(times/params[1])**params[5])
    #g1 = np.exp(-1*(times/params[1])**0.8)
    pow2 = 0.95
    g2 = np.exp(-1*(times/params[4])**pow2)
    d = params[0]*(1-g1) + params[3]*(1-g2) + params[2]
    return d
    
def dynamicStructFuncStretchedDoubleError(params, times, data, log=False, wts=None, fixedB=0):
    if wts is not None:
        if len(wts) != len(data):
            wts = wts[0:len(data)]
    else:
        wts = np.ones_like(data)
    penalty = 0
    
    if fixedB>0:
        newparams = np.zeros((6,))
        newparams[0:2] = params[0:2]
        newparams[2] = fixedB
        newparams[3] = params[2]
        newparams[4] = params[3]
        newparams[5] = params[4]
    else:
        newparams = params.copy()

    if log:
        return wts*(np.log(dynamicStructFuncStretchedDouble(newparams, times)) - np.log(data)) + penalty
    else:
        return wts*(dynamicStructFuncStretchedDouble(newparams, times) - data) + penalty

def dynamicStructFuncDoubleError(params, times, data, log=False, wts=None, fixedB=0):
    if wts is not None:
        if len(wts) != len(data):
            wts = wts[0:len(data)]
    else:
        wts = np.ones_like(data)
    penalty = 0
    
    if fixedB>0:
        newparams = np.zeros((5,))
        newparams[0] = params[0]
        newparams[1] = params[1]
        newparams[2] = fixedB
        newparams[3] = params[2]
        newparams[4] = params[3]
    else:
        newparams = params
        
    newparams[0] = abs(newparams[0])
    newparams[3] = abs(newparams[3])

    if log:
        return wts*(np.log(dynamicStructFuncDouble(newparams, times)) - np.log(data)) + penalty
    else:
        return wts*(dynamicStructFuncDouble(newparams, times) - data) + penalty

def dynamicStructFuncDoublePDError(params, times, data, log=False, wts=None, fixedB=0):
    if wts is not None:
        if len(wts) != len(data):
            wts = wts[0:len(data)]
    else:
        wts = np.ones_like(data)
    penalty = 0
    
    if fixedB>0:
        newparams = np.zeros((7,))
        newparams[0] = params[0]
        newparams[1] = params[1]
        newparams[2] = fixedB
        newparams[3] = params[2]
        newparams[4] = params[3]
        newparams[5] = params[4]
        newparams[6] = params[5]
    else:
        newparams = params
        
    newparams[0] = abs(newparams[0])
    newparams[3] = abs(newparams[3])

    if log:
        return wts*(np.log(dynamicStructFuncDoublePD(newparams, times)) - np.log(data)) + penalty
    else:
        return wts*(dynamicStructFuncDoublePD(newparams, times) - data) + penalty

def dynamicStructFuncHoldStretchedDoubleError(params, times, data, alpha, fixedB=0, log=False, wts=None):
    if fixedB>0:
        newparams = np.zeros((6,))
        newparams[0] = params[0]
        newparams[1] = params[1]
        newparams[2] = fixedB
        newparams[3] = params[2]
        newparams[4] = params[3]
        newparams[5] = alpha
    else:
        newparams = np.zeros((6,))
        newparams[0] = params[0]
        newparams[1] = params[1]
        newparams[2] = params[2]
        newparams[3] = params[3]
        newparams[4] = params[4]
        newparams[5] = alpha
    if wts is not None:
        if len(wts) != len(data):
            wts = wts[0:len(data)]
    else:
        wts = np.ones_like(data)
    penalty = 0
    if params[3]<-0.2:
        penalty = 1e10
    if log:
        return wts*(np.log(dynamicStructFuncStretchedDouble(newparams, times)) - np.log(data)) + penalty
    else:
        return wts*(dynamicStructFuncStretchedDouble(newparams, times) - data) + penalty
    
def dynamicStructFuncWithDrift(params, times):
    g = np.exp(-1*times/params[1])
    
def dynamicStructFuncPD(params, times):
    #Taking into account polydispersity
    g = np.exp(-1*times/params[1])*(1+(abs(params[3])*times*times/2.0))
    d = params[0]*(1-g) + abs(params[2])
    return d
    
def dynamicStructFuncPD3(params, times):
    #Taking into account polydispersity
    g = np.exp(-1*times/params[1])*(1+(abs(params[3])*times*times/2.0)-(abs(params[4]/6.0)*times*times*times))
    d = params[0]*(1-g) + abs(params[2])
    return d
    
def dynamicStructFuncZ(params, times):
    g = np.exp(-1*times/params[1])/(np.sqrt(1+times/params[3]))
    d = params[0]*(1-g) + abs(params[2])
    return d
    
def dynamicStructFuncZS(params, times):
    g = (np.exp(-1*times/params[1])**params[4])/(np.sqrt(1+(times/params[3])))
    d = params[0]*(1-g) + abs(params[2])
    return d

def dynamicStructFuncZPD(params, times):
    g = (np.exp(-1*times/params[1])/(np.sqrt(1+times/params[3])))*(1+(abs(params[4])*times*times/2.0))
    d = params[0]*(1-g) + abs(params[2])
    return d
    
def dynamicStructFuncError(params, times, data):
    return dynamicStructFunc(params, times) - data
    
def dynamicStructFuncStretchedError(params, times, data, log=False, wts=None):
    if wts is not None:
        if len(wts) != len(data):
            wts = wts[0:len(data)]
    else:
        wts = np.ones_like(data)
    penalty = 0
    if params[3]<0.01: #was 0.15; change to 0.01 on 8/16/2016 -rjm
        penalty = 1e10
    if log:
        return wts*(np.log(dynamicStructFuncStretched(params, times)) - np.log(data)) + penalty
    else:
        return wts*(dynamicStructFuncStretched(params, times) - data) + penalty
  
def dynamicStructFuncPDError(params, times, data):
    return dynamicStructFuncPD(params, times) - data  
    
def dynamicStructFuncPD3Error(params, times, data):
    return dynamicStructFuncPD3(params, times) - data  
    
def dynamicStructFuncZError(params, times, data):
    return dynamicStructFuncZ(params, times) - data
    
def dynamicStructFuncZPDError(params, times, data):
    return dynamicStructFuncZPD(params, times) - data
    
def dynamicStructFuncZSError(params, times, data):
    return dynamicStructFuncZS(params, times) - data
    
def dynamicStructFuncZSLogError(params, times, data):
    return np.log(dynamicStructFuncZS(params, times)) - np.log(data)
    
def dynamicStructFuncHoldStretchError(params, times, data, alpha, log=False,wts=None):
    newparams = np.zeros((4,))
    newparams[0] = params[0]
    newparams[1] = params[1]
    newparams[2] = params[2]
    newparams[3] = alpha
    if wts is not None:
        if len(wts) != len(data):
            wts = wts[0:len(data)]
    else:
        wts = np.ones_like(data)
    if log:
        return wts*(np.log(dynamicStructFuncStretched(newparams,times)) - np.log(data))
    else:
        return wts*(dynamicStructFuncStretched(newparams, times) - data)

def dynamicStructFuncLogErrorHoldB(params, times, data, B, log=False):
    newparams = np.zeros((3,))
    newparams[0] = params[0]
    newparams[1] = params[1]
    newparams[2] = B
    if log:
        return np.log(dynamicStructFunc(newparams,times))-np.log(data)
    else:
        return dynamicStructFunc(newparams,times) - data
    
def dynamicStructFuncStretchHoldBErr(params, times, data, B, log=False):
    newparams = np.zeros((4,))
    newparams[0] = params[0]
    newparams[1] = params[1]
    newparams[2] = B
    newparams[3] = params[2]
    if log:
        return np.log(dynamicStructFuncStretched(newparams,times)) - np.log(data)
    else:
        return dynamicStructFuncStretched(newparams,times) - data
        
def dynamicStructFuncHoldStretchHoldBErr(params, times, data, alpha, B, log=False):
    newparams = np.zeros((4,))
    newparams[0] = params[0]
    newparams[1] = params[1]
    newparams[2] = B
    newparams[3] = alpha
    if log:
        return np.log(dynamicStructFuncStretched(newparams,times)) - np.log(data)
    else:
        return dynamicStructFuncStretched(newparams,times) - data
        

def dynamicStructFuncLogError(params, times, data):
    return np.log(dynamicStructFunc(params, times)) - np.log(data)
    #err = abs(dynamicStructFunc(params, times) - data)
    #return np.log(1+err)
    
def dynamicStructFuncLogErrorPD(params, times, data):
    return np.log(dynamicStructFuncPD(params, times)) - np.log(data)
    #err = abs(dynamicStructFunc(params, times) - data)
    #return np.log(1+err)
    
def dynamicStructFuncLogErrorPD3(params, times, data):
    return np.log(dynamicStructFuncPD3(params, times)) - np.log(data)

def dynamicStructFuncZLogError(params, times, data):
    penalty = 0
    if params[3]>10000:
        penalty = 1e20
    if params[3]<0:
        penalty = 1e20
    return np.log(dynamicStructFuncZ(params, times)) - np.log(data) + penalty
    
def dynamicStructFuncZPDLogError(params, times, data):
    penalty = 0
    if params[3]>10000:
        penalty = 1e20
    if params[3]<0:
        penalty = 1e20
    return np.log(dynamicStructFuncZPD(params, times)) - np.log(data) + penalty
    
def dynamicStructFuncStretchedLogError(params, times, data):
    return np.log(dynamicStructFuncStretched(params, times)) - np.log(data)
    
def getDiff(tau, qs, qmin, qmax, power=0.5):
    w = np.where((qs>qmin)*(qs<qmax))
    pfit = np.polyfit(qs[w[0]], (1./tau[w[0]])**power,1)
    return pfit
    
def waveVectorRollOff(sigma, M, k0=(np.pi*2/0.525e-6)):
    #Equation 37
    #sigma is the NA of the objective
    #M is the incoherence parameter (NA of condenser over NA of obj)
    # k0 is 2pi/lambda
    m_factor = np.sqrt((1+2*M*M)/(1.0+M*M))
    return sigma*k0*m_factor

def qz_version1(q, k0):
    #Equation 12
    return (q**2)/(2*k0)
    
def Cq(q,qro,lamb,dlamb):
    #Equation 43
    qratio2 = (q/qro)**2
    lamb2 = (dlamb/lamb)**2
    numerator = np.exp(-0.5*qratio2*(1./(1+qratio2*lamb2)))
    denom = np.sqrt(1+(qratio2*lamb2))
    return (numerator/denom)

def qz(q,k0,M,lamb,dlamb,sigma):
    #Equation 41
    term1 = (q*q)/(2*k0)
    temp = (1./(sigma*sigma))
    temp = temp * ((q/k0)**2)
    temp = temp * ((dlamb/lamb)**2)
    term2 = 1 - (2*M*M) - temp
    return term1*term2

def dq(q, sigma, M, k0, lamb, dlamb):
    qk02 = (q/k0)**2
    lambratio = (dlamb/lamb)**2
    dq2 = q*q*((sigma*M)**2 + (0.25*qk02*lambratio))
    return np.sqrt(dq2)
    
def Aq(ap, Cq, dq, qz):
    term1 = (2*ap*ap)/np.sqrt(np.pi)
    term2 = (Cq*Cq)/dq
    term3 = 1 - np.exp(-1*(qz/dq)**2)
    return (term1*term2*term3)

#one that seems to work:
#aq = Aq_FromStart(1.0, q[1:63], 0.525e-6, 0.04e-6, 6e-8, 1.0)
def Aq_FromStart(ap, q, lamb, dlamb, M, sigma, dq_const=None):
    k0 = (2*np.pi)/lamb    
    if dq_const is not None:
        dqterm = dq_const * np.ones_like(q)
    else:
        dqterm = dq(q,sigma,M,k0,lamb,dlamb)    
    qro = waveVectorRollOff(sigma, M, k0=k0)
    cqterm = Cq(q,qro,lamb,dlamb)
    qzterm = qz(q,k0,M,lamb,dlamb,sigma)
    aq = Aq(ap, cqterm, dqterm, qzterm)
    return aq

def getDiffCoef(allRes, qs, qminmax):
    if allRes.has_key('Stretch-s') and allRes.has_key('Stretch-t'):
        s = allRes['Stretch-s']
        t = allRes['Stretch-t']
        nt = newt(t,s)
    elif allRes.has_key('SingleExp'):
        nt = allRes['t']
    else:
        return 0
    indx = (qs>qminmax[0]) & (qs<qminmax[1])
    ds = (qs[indx]**-2) / nt[indx]
    return ds.mean()

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def departureFromLineAll(allRes, qs, slope, good=0.1, c='0.6'):
    newDict = {}
    for k in allRes.keys():
        diff = departureFromLine(allRes[k],qs,slope,c=None)
        w = np.where(abs(diff[1:])<good)
        w2 = consecutive(w[0])
        lens = []
        for i in range(0,len(w2)):
            lens.append(len(w2[i]))
        newW = w2[np.argmax(lens)]
        newDict[k] = (qs[1:][newW[0]], qs[1:][newW[-1]])
    if c is not None:
        pylab.figure()
    for k in newDict.keys():
        if c is not None:
            pylab.bar(int(k), newDict[k][1]-newDict[k][0], bottom=newDict[k][0], width=32, color=c)
    return newDict

def departureFromLine(res, qs, slope, logdif=True, c='r'):
    nt = newt(res['Stretch-t'],res['Stretch-s'])
    onLine = (qs[:63]**-2) / slope
    if logdif:
        difference = np.log(nt)-np.log(onLine)
    else:
        difference = nt-onLine
    if c is not None:
        pylab.plot(qs[1:63], difference[1:], 'o', color=c)
    #pylab.ylim(-0.1,0.1)
    return difference
    
def plotTauVsQ(listOfRes, roiString, q, c='b', ms=3, elw=3, ecolor=None):
    numResults = len(listOfRes)
    ts = np.zeros((numResults, 63))
    for indx,res in enumerate(listOfRes):
        ts[indx] = newt(res[roiString]['Stretch-t'],res[roiString]['Stretch-s'])
    mean_ts = ts.mean(axis=0)
    error_ts = np.zeros((2,63))
    error_ts[1,:] = mean_ts - ts.min(axis=0)
    error_ts[0,:] = ts.max(axis=0) - mean_ts
    if ecolor is None:
        ecolor = c
    pylab.loglog(q[1:63], mean_ts[1:], 'o', color=c, ms=ms, mec=c)
    pylab.errorbar(q[1:63], mean_ts[1:], yerr=error_ts[:,1:], fmt=None, ecolor=ecolor, elinewidth=elw)
    
    
    
    