"""
This module contains the code for calculating and fitting the DDM matrix. Other 
modules defines classes which interface with the functions here. A select group of 
papers are listed below for those looking for a deeper understanding of these 
functions. [1]_ [2]_ [3]_

This code, and its earlier versions, have been used in several projects of the 
McGorty lab at the University of San Diego. 


References
----------
.. [1] Cerbino, R. & Trappe, V. Differential Dynamic Microscopy: Probing Wave Vector Dependent Dynamics with a Microscope. Phys. Rev. Lett. 100, 188102 (2008)
.. [2] Wilson, L. G. et al. Differential Dynamic Microscopy of Bacterial Motility. Phys. Rev. Lett. 106, 018101 (2011).
.. [3] Germain, D., Leocmach, M. & Gibaud, T. Differential dynamic microscopy to characterize Brownian motion and bacteria motility. American Journal of Physics 84, 202–210 (2016).



"""
###########################################################################
# File moved from other DDM repository on 5/21/2021                   #####
# Renamed ddm_calc.py from ddm.py from ddm_clean.py                   #####
# Authors:                                                            #####
#   Ryan McGorty (rmcgorty@sandiego.edu)                              #####
###########################################################################

import sys
import copy
import numpy as np
from scipy.optimize import least_squares, curve_fit
from scipy.special import gamma
from scipy.signal import blackmanharris #for Blackman-Harris windowing
import socket
import skimage
import fit_parameters_dictionaries as fpd
import logging
from IPython.core.display import clear_output

class IPythonStreamHandler(logging.StreamHandler):
    "A StreamHandler for logging that clears output between entries."
    def emit(self, s):
        clear_output(wait=True)
        print(s.getMessage())
    def flush(self):
        sys.stdout.flush()

logger = logging.getLogger("DDM Calculations")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
#ch = IPythonStreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

logger2 = logging.getLogger("DDM Analysis")
logger2.setLevel(logging.DEBUG)
ch2 = logging.StreamHandler()
#ch = IPythonStreamHandler()
ch2.setLevel(logging.DEBUG)
formatter2 = logging.Formatter('%(name)s - %(message)s')
ch2.setFormatter(formatter2)
logger2.addHandler(ch2)

comp_name = socket.gethostname()

#This function is used to determine a new time when a distribution
# of decay times are present
newt = lambda t,s: (1./s)*gamma(1./s)*t

#Sometimes it is helpful to apply a window function to the images (deals with beads/molecules leaving the field of view)
def window_function(im):
    r'''Applies windowing function to images.
    
    Particles moving outside the frame of the images can lead to artifacts in 
    the DDM analysis, especially for the higher wavevectors. Use of windowing 
    function described in Giavazzi 2017 [1]_. We 
    apply the Blackman-Harris windowing. This function creates a mask to 
    multiply the images by to implement the windowing. 

    
    Parameters
    ----------
    im : array
        Array of same size as one frame of the image sequences for analysis.
    
    Returns
    -------
    filter_func : array
        Multiply this array (of same size as `im`) to apply the windowing
        filter.
        
    References
    ----------
    .. [1] Giavazzi, F., Edera, P., Lu, P. J. & Cerbino, R. Image windowing mitigates edge effects in Differential Dynamic Microscopy. Eur. Phys. J. E 40, 97 (2017).

    '''
    if im.ndim==3:
        numPixels = im.shape[1]
    elif im.ndim==2:
        numPixels = im.shape[0]
    elif isinstance(im, int):
        numPixels = im
    x,y = np.meshgrid(blackmanharris(numPixels),blackmanharris(numPixels))
    filter_func = x*y
    return filter_func

def determining_A_and_B(im, use_BH_filter=False,
                        centralAngle=None, angRange=None):
    '''
    Calculates the 2D Fourier transform of each frame in the image series and takes the radial averages.
    This in order to find the amplitude and background.

    :param im: Images series
    :type im: numpy array
    :param use_BH_filter: Apply window filter
    :type use_BH_filter: bool

    :return:
            * radial_averages (*numpy array*)- radial average of all frames in provided image series


    '''
    av_fftsq_of_each_frame = np.zeros_like(im[0]*1.0) #initialize array
    nFrames,ndx,ndy = im.shape
    if use_BH_filter:
        filterfunction = window_function(im)
    else:
        filterfunction = np.ones_like(im[0])
    for i in range(nFrames):
        fft_of_image = np.fft.fft2(im[i]*filterfunction)
        sqr_of_fft = np.fft.fftshift(fft_of_image*np.conj(fft_of_image))
        av_fftsq_of_each_frame = av_fftsq_of_each_frame + abs(sqr_of_fft)
    av_fftsq_of_each_frame = av_fftsq_of_each_frame/(1.0*nFrames*ndx*ndy)
    rad_av_av_fftsq = radial_avg_ddm_matrix(av_fftsq_of_each_frame.reshape(1,ndx,ndy),
                                      centralAngle=centralAngle,
                                      angRange=angRange)
    return rad_av_av_fftsq

def generateLogDistributionOfTimeLags(start,stop,numPoints):
    '''
    This function will generate a logarithmically spaced set of numbers.
    This is for generating the lag times over which to calculate the intermediate scattering function.

    :param start: First time delay (usually 1)
    :type start: int
    :param stop: Last time delay (often 600 or 100 but can be ~20-50 for quick calculations)
    :type stop: int
    :param numPoints: number of time delays (can be 60 for quick calculations)


    :return:
        * listOfLagTimes (*List[int]*)- list of numbers from start to stop, logarithmically spaced

    '''
    newpts = [] #empty list

    #first, will make list of numbers log-spaced longer than intended list.
    #This is due to fact that after converting list to integers, we will have
    # multiple duplicates. So to generate enough *unique* numbers we need to
    # first make a longer-than-desired list.
    listOfLagTimes = np.geomspace(start, stop, num=numPoints, dtype=int)
    numberOfPoints = len(np.unique(listOfLagTimes))
    if numberOfPoints == numPoints:
        return np.unique(listOfLagTimes)
    else:
        newStartingNumberOfPoints = numPoints
        while numberOfPoints < numPoints:
            newStartingNumberOfPoints += 1
            listOfLagTimes = np.geomspace(start, stop, num=newStartingNumberOfPoints, dtype=int)
            numberOfPoints = len(np.unique(listOfLagTimes))
        return np.unique(listOfLagTimes)

def new_ddm_matrix(imageArray):
    r"""More experimental method for getting DDM matrix.
    
    More to come later. 
    
    """

    #First, generate Fourier transforms of all images
    fft_ims = np.zeros(imageArray.shape,dtype='complex128')
    ndx,ndy = imageArray[0].shape
    num_images = imageArray.shape[0]
    for i,im in enumerate(imageArray):
        fft_ims[i] = np.fft.fftshift(np.fft.fft2(im))/(ndx*ndy)

    #Getting the "d_c" term from the paper:
    #   Norouzisadeh, M., Chraga, M., Cerchiari, G. & Croccolo, F. The modern structurator:
    #    increased performance for calculating the structure function. Eur. Phys. J. E 44, 146 (2021).
    fft_in_times = np.fft.fftn(fft_ims, axes=(0,))
    new_matrix = np.conj(fft_in_times)*fft_in_times
    inverse_fft_in_times = np.fft.ifftn(new_matrix, axes=(0,))
    return np.real(inverse_fft_in_times)


def computeDDMMatrix(imageArray, dts, use_BH_windowing=False, fast_mode=False, quiet=False):
    r'''Calculates DDM matrix
    
    This function calculates the DDM matrix at the lag times provided by `dts`.  
    
    Parameters
    ----------
    imageArray : array
        3D array of images. First dimension should be time. 
    dts : array
        1D array of the lag times for which to calculate the DDM matrix
    use_BH_windowing : {True, False}, optional
        Apply Blackman-Harris windowing to the images if True. Default is False. 
    fast_mode : {True, False}, optional
        Calculates the DDM matrix using few Fourier transformed image pairs per lag time. 
        Default is False.
    quiet : {True, False}, optional
        If True, prints updates as the computation proceeds
        
        
    Returns
    -------
    ddm_mat : array
        The DDM matrix. First dimension is time lag. Other two are the x and y
        wavevectors.
    num_pairs_per_dt : array
        1D array. Contains the number of image pairs that went into calculating the 
        DDM matrix for each lag time. Used for weighting fits to the DDM matrix.
    
    
    '''

    ### TO-DO: check that imageArray is in fact a 3D array
    #
    #

    #Applies the Blackman-Harris window if desired
    if use_BH_windowing:
        filterfunction = window_function(imageArray)
    else:
        filterfunction = np.ones_like(imageArray[0])

    #Determines the dimensions of the data set (number of frames, x- and y-resolution in pixels
    ntimes, ndx, ndy = imageArray.shape

    #Initializes array for Fourier transforms of differences
    ddm_mat = np.zeros((len(dts), ndx, ndy),dtype=np.float)

    #We *don't* take the Fourier transform of *every* possible difference
    #of images separated by a given lag time. 
    steps_in_diffs = np.ceil(dts/3.0).astype(np.int)
    if fast_mode:
        w = np.where(steps_in_diffs < 20)
        steps_in_diffs[w] = 20

    #To record the number of pairs of images for each lag time
    num_pairs_per_dt = []

    #Loops over each delay time
    j=0
    for k,dt in enumerate(dts):

        if not quiet:
            if k%4 == 0:
                #print("Running dt=%i...\n" % dt)
                logger.info("Running dt = %i..." % dt)

        #Calculates all differences of images with a delay time dt
        all_diffs = filterfunction*(imageArray[dt:].astype(np.float) - imageArray[0:(-1*dt)].astype(np.float))

        #Rather than FT all image differences of a given lag time, only select a subset
        all_diffs_new = all_diffs[0::steps_in_diffs[k],:,:]

        #Loop through each image difference and take the fourier transform
        for i in range(0,all_diffs_new.shape[0]):
            temp = np.fft.fft2(all_diffs_new[i]) # - all_diffs_new[i].mean())
            ddm_mat[j] = ddm_mat[j] + abs(temp*np.conj(temp))/(ndx*ndy)

        num_pairs_per_dt.append(all_diffs_new.shape[0])

        #Divide the running sum of FTs to get the average FT of the image differences of that lag time
        ddm_mat[j] = ddm_mat[j] / (all_diffs_new.shape[0])
        ddm_mat[j] = np.fft.fftshift(ddm_mat[j])

        j = j+1
        
    num_pairs_per_dt = np.array(num_pairs_per_dt)

    return ddm_mat, num_pairs_per_dt


def get_FF_DDM_matrix(imageFile, dts, submean=True,
                       useBH_windowing=False):
    '''
    This code calculates the far-field DDM matrix for the series of images
    in imageFile at the lag times specified in dts.
    :param imageFile: Either a string specifying the location of the data or the data itself as a numpy array
    :param dts: 1D array of delay times
    :param limitImsTo: defaults to None
    :param every: defaults to None
    :param shiftAtEnd: defaults to False
    :param noshift: defaults to False
    :param submean: defaults to true
    :return: two numpy arrays: the fft'd data and the list of times

    For more on this far-field DDM method see:

    Buzzaccaro, S., Alaimo, M. D., Secchi, E. & Piazza, R. Spatially: resolved heterogeneous dynamics in a strong colloidal gel. J. Phys.: Condens. Matter 27, 194120 (2015).

    Philippe, A. et al. An efficient scheme for sampling fast dynamics at a low average data acquisition rate. J. Phys.: Condens. Matter 28, 075201 (2016).


    '''
    if isinstance(imageFile, np.ndarray):
        ims = imageFile
    elif isinstance(imageFile, basestring):
        ims = skimage.io.imread(imageFile)
    else:
        print("Not sure what you gave for imageFile")
        return 0

    if useBH_windowing:
        filterfunction = window_function(ims)
    else:
        filterfunction = np.ones_like(ims[0]*1.0)

    #Determines the dimensions of the data set (number of frames, x- and y-resolution in pixels
    ntimes, ndx, ndy = ims.shape

    #Initializes array for Fourier transforms of images
    fft_images = np.zeros((ntimes, ndx, ndy),dtype=np.complex128)

    ddm_matrix = np.zeros((len(dts),ndx,ndy),dtype=np.float)

    for i in range(ntimes):
        new_image = filterfunction*ims[i]
        if submean:
            new_image = new_image - new_image.mean()
        fft_images[i] = np.fft.fftshift(np.fft.fft2(new_image))/(ndx*ndy)

    for k,dt in enumerate(dts):
        all_pairs_1 = fft_images[dt:] * fft_images[0:(-1*dt)]
        norm_1 = np.mean(abs(fft_images[dt:] * np.conj(fft_images[dt:])),axis=0)
        norm_2 = np.mean(abs(fft_images[0:(-1*dt)] * np.conj(fft_images[0:(-1*dt)])),axis=0)
        all_pairs_2 = abs(all_pairs_1 * np.conj(all_pairs_1))
        all_pairs_3 = all_pairs_2.mean(axis=0) / (norm_1 * norm_2)
        ddm_matrix[k] = all_pairs_3

    return ddm_matrix, dts



def fit_ddm_all_qs(dData, times, param_dictionary,
                   amplitude_from_ims,
                   first_use_leastsq=True,
                   use_curvefit_method=False,
                   sigma=None,
                   update_tau_based_on_estimated_diffcoeff=False,
                   estimated_diffcoeff=None,
                   update_tau_based_on_estimated_velocity=False,
                   estimated_velocity=None,
                   update_tau2_based_on_estimated_diffcoeff=False,
                   estimated_diffcoeff2=None,
                   update_tau2_based_on_estimated_velocity=False,
                   estimated_velocity2=None,
                   update_limits_on_tau=False,
                   updated_lims_on_tau_fraction=0.1,
                   use_A_from_images_as_guess=False,
                   update_limits_on_A=False,
                   updated_lims_on_A_fraction=0.1,
                   err=None, logfit=False,maxiter=600,
                   factor=1e-3, quiet=False, quiet_on_method=True,
                   last_times = None, given_fit_method = None,
                   update_initial_guess_each_q = False,
                   debug=False):
    r"""Function to fit the DDM matrix or ISF for all wavevectors.
    
    This function fits the data from DDM (either the DDM matrix or the
    intermediate scattering function (ISF)) to a specified model. 
    
    .. math:: D(q,\Delta t) = A(q) [1 - ISF(q,\Delta t)] + B(q)
    
    Parameters
    ----------
    dData : array
        Array containing the DDM data to fit for. This will be either the
        DDM matrix or the ISF. This should be a 2D array; the first dimension 
        corresponds to the lag times, the second dimension corresponds to the
        wavevectors. 
    times : array_like
        1D array of the lagtimes
    param_dictionary : dict
        Dictionary corresponding to the model we will fit to. This dictionary
        contains the parameters, the initial guess for their values, their bounds,
        and the function of the model to fit to. See the module 
        :py:mod:`PyDDM.fit_parameters_dictionaries`
    amplitude_from_ims : array_like
        The DDM matrix is usually fit to something like 
        DDM_matrix = A(1-f)+B where A is the amplitude. But this A parameter can
        also be determined by taking the Fourier transforms of the images (rather
        than differences between images). This parameter is the amplitude found
        from the Fourier transforms of the images. We have the option of using
        these values as the initial guesses when fitting the amplitude of 
        the DDM matrix. 
    first_use_leastsq : {False}, optional
        If True, will use `scipy.optimize.least_squares` for fitting.
    use_curvefit_method : {True}, optional
        If True, will use `scipy.optimize.curve_fit` for fitting.
    sigma : {None}, optional
        If `scipy.optimize.curve_fit` is used, we can weight the data points by
        this array. If passed, it will need to be a 1D array of length equal to 
        the number of lag times. 
    
    Returns
    -------
    best_fit_params : dict
        Dictionary containing the best fit values
    theory : array
        Model evaluated using the best fit values. Will be of the same size as
        the passed parameter `dData`. 
    
    """

    #Find the number of lag times and number of wavevectors
    #based on shape of the data passed to the function
    num_times, num_qs = dData.shape

    #Initialize dictionary to store fitted values for parameters
    best_fit_params = {}

    #Find the number of parameters we will fit for
    number_of_parameters = len(param_dictionary['parameter_info'])

    for param in param_dictionary['parameter_info']:
        best_fit_params[param['parname']] = np.zeros((num_qs))

    theory = np.empty((num_times, num_qs)) #Empty array to store theoretical models calculated with best fits
    theory.fill(np.nan)

    #Loop through each wavevector
    for i in range(num_qs):
        if debug:
            print("Fitting for q index of %i..." % i)
            
        #If one does not want data for the longer lag times to be included
        #when fitting, one can pass the optional parameter `last_times`
        if last_times is not None:
            if np.isscalar(last_times):
                data_to_fit = dData[:last_times,i]
                times_to_fit = times[:last_times]
            else:
                data_to_fit = dData[:int(last_times[i]),i]
                times_to_fit = times[:int(last_times[i])]
        else:
            data_to_fit = dData[:,i]
            times_to_fit = times

        #For basing initial guess for 'Tau' on expected diffusion coefficient or velocity
        qvalue = dData.q[i].values
        if update_tau_based_on_estimated_diffcoeff and (estimated_diffcoeff is not None):
            for element in param_dictionary['parameter_info']:
                if (element['parname']=='Tau') and (i>0):
                    new_tau = 1./(qvalue*qvalue*estimated_diffcoeff)
                    if update_limits_on_tau:
                        element['limits'][0] = new_tau * (1-updated_lims_on_tau_fraction)
                        element['limits'][1] = new_tau * (1+updated_lims_on_tau_fraction)
                        element['value'] = new_tau
                    if (new_tau>=element['limits'][0]) and (new_tau<=element['limits'][1]):
                        element['value'] = new_tau
        elif update_tau_based_on_estimated_velocity and (estimated_velocity is not None):
            for element in param_dictionary['parameter_info']:
                if (element['parname']=='Tau') and (i>0):
                    new_tau = 1./(qvalue*estimated_velocity)
                    if update_limits_on_tau:
                        element['limits'][0] = new_tau * (1-updated_lims_on_tau_fraction)
                        element['limits'][1] = new_tau * (1+updated_lims_on_tau_fraction)
                        element['value'] = new_tau
                    elif (new_tau>=element['limits'][0]) and (new_tau<=element['limits'][1]):
                        element['value'] = new_tau
        if update_tau2_based_on_estimated_diffcoeff and (estimated_diffcoeff2 is not None):
            for element in param_dictionary['parameter_info']:
                if (element['parname']=='Tau2') and (i>0):
                    new_tau2 = 1./(qvalue*qvalue*estimated_diffcoeff2)
                    if update_limits_on_tau:
                        element['limits'][0] = new_tau2 * (1-updated_lims_on_tau_fraction)
                        element['limits'][1] = new_tau2 * (1+updated_lims_on_tau_fraction)
                        element['value'] = new_tau2
                    elif (new_tau2>=element['limits'][0]) and (new_tau2<=element['limits'][1]):
                        element['value'] = new_tau2
        elif update_tau2_based_on_estimated_velocity and (estimated_velocity2 is not None):
            for element in param_dictionary['parameter_info']:
                if (element['parname']=='Tau2') and (i>0):
                    new_tau2 = 1./(qvalue*estimated_velocity2)
                    if update_limits_on_tau:
                        element['limits'][0] = new_tau2 * (1-updated_lims_on_tau_fraction)
                        element['limits'][1] = new_tau2 * (1+updated_lims_on_tau_fraction)
                        element['value'] = new_tau2
                    elif (new_tau2>=element['limits'][0]) and (new_tau2<=element['limits'][1]):
                        element['value'] = new_tau2
        if use_A_from_images_as_guess:
            for element in param_dictionary['parameter_info']:
                if (element['parname']=='Amplitude') and (i>0):
                    new_A = amplitude_from_ims[i]
                    if new_A<0:
                        new_A=1
                    if update_limits_on_A:
                        element['limits'][0] = new_A * (1-updated_lims_on_A_fraction)
                        element['limits'][1] = new_A * (1+updated_lims_on_A_fraction)
                        element['value'] = new_A
                    elif (new_A>=element['limits'][0]) and (new_A<=element['limits'][1]):
                        element['value'] = new_A


        ret_params, theory[:len(times_to_fit),i], error, chi2 = fit_ddm(data_to_fit, times_to_fit, param_dictionary,
                                                                        first_use_leastsq=first_use_leastsq,
                                                                        use_curvefit_method=use_curvefit_method,
                                                                        sigma=sigma,
                                                                        err=err, logfit=logfit,maxiter=maxiter,
                                                                        factor=factor, quiet=quiet,
                                                                        quiet_on_method=quiet_on_method)

        for j, bf_param in enumerate(best_fit_params):
            best_fit_params[bf_param][i] = ret_params[j]

    return best_fit_params, theory



def fit_ddm(dData, times, param_dictionary,
            first_use_leastsq=True,
            use_curvefit_method=False,
            sigma=None,
            err=None, logfit=False,maxiter=600,
            factor=1e-3, quiet=False, quiet_on_method=True):
    r"""Function to fit the DDM matrix or ISF for one wavevector.
    
    This function fits the data from DDM (either the DDM matrix or the
    intermediate scattering function (ISF)) to a specified model. This function
    will fit the data for one q. 
    
    .. math:: D(q,\Delta t) = A(q) [1 - ISF(q,\Delta t)] + B(q)
    
    Parameters
    ----------
    dData : array
        Array containing the DDM data to fit for. This will be either the
        DDM matrix or the ISF. This should be a 2D array; the first dimension 
        corresponds to the lag times, the second dimension corresponds to the
        wavevectors. 
    times : array_like
        1D array of the lagtimes
    param_dictionary : dict
        Dictionary corresponding to the model we will fit to. This dictionary
        contains the parameters, the initial guess for their values, their bounds,
        and the function of the model to fit to. See the module 
        :py:mod:`PyDDM.fit_parameters_dictionaries`
    amplitude_from_ims : array_like
        The DDM matrix is usually fit to something like 
        DDM_matrix = A(1-f)+B where A is the amplitude. But this A parameter can
        also be determined by taking the Fourier transforms of the images (rather
        than differences between images). This parameter is the amplitude found
        from the Fourier transforms of the images. We have the option of using
        these values as the initial guesses when fitting the amplitude of 
        the DDM matrix. 
    first_use_leastsq : {False}, optional
        If True, will use `scipy.optimize.least_squares` for fitting.
    use_curvefit_method : {True}, optional
        If True, will use `scipy.optimize.curve_fit` for fitting.
    sigma : {None}, optional
        If `scipy.optimize.curve_fit` is used, we can weight the data points by
        this array. If passed, it will need to be a 1D array of length equal to 
        the number of lag times. 
    
    Returns
    -------
    params : array
        Array containing the best fit values
    theory : array
        Model evaluated using the best fit values. Will be of the same size as
        the passed parameter `dData`. 
    error : array
        Error between the fit and model.
    None
        Last return is `None`.
    
    """

    parameter_values = fpd.extract_array_of_parameter_values(param_dictionary)
    param_mins, param_maxs = fpd.extract_array_of_param_mins_maxes(param_dictionary)
    #print(param_mins)

    #If 'first_use_leastsq' is true, we will use the scipy.optimize leastsquares fitting method
    #  first (just to get initial parameters).
    if first_use_leastsq:
        lsqr_params, lsqr_theory, lsqr_error = execute_LSQ_fit(dData, times, param_dictionary, debug=False)
        which_params_should_be_fixed = fpd.extract_array_of_fixed_or_not(param_dictionary)
        for i,tofix in enumerate(which_params_should_be_fixed):
            if not tofix:
                parameter_values[i] = lsqr_params[i]

    #print(parameter_values)
    if use_curvefit_method:
        if first_use_leastsq:
            updated_param_dict = copy.deepcopy(param_dictionary)
            fpd.populate_intial_guesses(updated_param_dict, parameter_values)
            res = execute_ScipyCurveFit_fit(dData, times, updated_param_dict, sigma=sigma, debug=False)
        else:
            res = execute_ScipyCurveFit_fit(dData, times, param_dictionary, sigma=sigma, debug=False)
        return res[0], res[1], res[2], None


    else:
        return lsqr_params, lsqr_theory, lsqr_error, None




def execute_LSQ_fit(dData, times, param_dict, debug=True):
    r"""Performs least_squares fit.
    
    Using the `scipy.optimize.least_squares` function, the data is fit to 
    model specified within the parameter `param_dict`. [1]_
    
    Parameters
    ----------
    dData : array
        Array containing the DDM data to fit for. This will be either the
        DDM matrix or the ISF. This should be a 2D array; the first dimension 
        corresponds to the lag times, the second dimension corresponds to the
        wavevectors. 
    times : array_like
        1D array of the lagtimes
    param_dict : dict
        Dictionary corresponding to the model we will fit to. This dictionary
        contains the parameters, the initial guess for their values, their bounds,
        and the function of the model to fit to. See the module 
        :py:mod:`PyDDM.fit_parameters_dictionaries`
    debug : {True}, optional
        If True, will print out values of initial guesses and bounds (and other
        info).
        
    Returns
    -------
    lsqr_params : array
        Values found for the parameters. 
    theory : array
        Model evaluated using values of the best fit parameters. 
    fun : array
        Vector of residuals
        
        
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    
    """

    theory_function = param_dict['model_function']

    params_to_pass_to_lsqr = fpd.extract_array_of_parameter_values(param_dict)
    minimum_of_parameters, maximum_of_parameters = fpd.extract_array_of_param_mins_maxes(param_dict)

    #define the error function (difference between data and the model)
    error_function = lambda parameters: dData-theory_function(times,*parameters)

    if debug:
        print("Parameters going to lsqr fitting: ", params_to_pass_to_lsqr)
        print("Min parameters going to lsqr fitting: ", minimum_of_parameters)
        print("Max parameters going to lsqr fitting: ", maximum_of_parameters)
        print("Size of data going to lsqr fitting: %i" % len(dData))
        print("Number of lag times going to lsqr fitting: %i" % len(times))
    lsqr_results = least_squares(error_function, params_to_pass_to_lsqr, bounds=(minimum_of_parameters, maximum_of_parameters))
    lsqr_params = lsqr_results['x']

    return lsqr_params, theory_function(times,*lsqr_params), lsqr_results['fun']


def execute_ScipyCurveFit_fit(dData, times, param_dict, sigma=None, debug=True, method=None):
    r"""Performs curve_fit fit.
    
    Using the `scipy.optimize.curve_fit` function, the data is fit to 
    model specified within the parameter `param_dict`. [1]_
    
    Parameters
    ----------
    dData : array
        Array containing the DDM data to fit for. This will be either the
        DDM matrix or the ISF. This should be a 2D array; the first dimension 
        corresponds to the lag times, the second dimension corresponds to the
        wavevectors. 
    times : array_like
        1D array of the lagtimes
    param_dict : dict
        Dictionary corresponding to the model we will fit to. This dictionary
        contains the parameters, the initial guess for their values, their bounds,
        and the function of the model to fit to. See the module 
        :py:mod:`PyDDM.fit_parameters_dictionaries`
    sigma : {None}, optional
        Passed as `sigma` parameter to `scipy.optimize.curve_fit`
    debug : {True}, optional
        If True, will print out values of initial guesses and bounds (and other
        info).
    method : {None}, optional
        Passed as `method` to `scipy.optimize.curve_fit`. Can be `lm`, `trf`, 
        or `dogbox`. 
        
    Returns
    -------
    lsqr_params : array
        Values found for the parameters. 
    theory : array
        Model evaluated using values of the best fit parameters. 
    error : array
        Error of parameters
        
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    
    """
    theory_function = param_dict['model_function']

    params_to_pass_to_cf = fpd.extract_array_of_parameter_values(param_dict)
    minimum_of_parameters, maximum_of_parameters = fpd.extract_array_of_param_mins_maxes(param_dict)

    if debug:
        print("Parameters going to CurveFit fitting: ", params_to_pass_to_cf)
        print("Min parameters going to CurveFit fitting: ", minimum_of_parameters)
        print("Max parameters going to CurveFit fitting: ", maximum_of_parameters)
        print("Size of data going to CurveFit fitting: %i" % len(dData))
        print("Number of lag times going to CurveFit fitting: %i" % len(times))
    try:
        if method == 'lm':
            #With 'lm' method, must be unconstrained problem. So no bounds
            cf_results = curve_fit(theory_function, times, dData, p0=params_to_pass_to_cf,
                                   sigma=sigma, absolute_sigma=False, method='lm')
        elif method == None:
            cf_results = curve_fit(theory_function, times, dData, p0=params_to_pass_to_cf,
                                   bounds=(minimum_of_parameters, maximum_of_parameters),
                                   sigma=sigma, absolute_sigma=False)
        else:
            cf_results = curve_fit(theory_function, times, dData, p0=params_to_pass_to_cf,
                                   bounds=(minimum_of_parameters, maximum_of_parameters),
                                   sigma=sigma, absolute_sigma=False, method=method)
        cf_params = cf_results[0]
        errors_1stddev = np.sqrt(np.diag(cf_results[1]))
    except:
        cf_params = params_to_pass_to_cf
        errors_1stddev = np.zeros_like(cf_params)

    return cf_params, theory_function(times,*cf_params), errors_1stddev


def generate_mask(im, centralAngle, angRange):
    r"""Generates a mask of the same size as `im`.
    
    If the DDM matrix is not to be radially averaged, we can use a mask 
    to average values of the matrix only in some angular range around 
    a central angle.

    Parameters
    ----------
    im : ndarray
        DESCRIPTION.
    centralAngle : float
        DESCRIPTION.
    angRange : float
        DESCRIPTION.

    Returns
    -------
    mask : ndarray
        DESCRIPTION.

    """
    nx,ny = im.shape
    xx = np.arange(-(nx-1)/2., nx/2.)
    yy = np.arange(-(ny-1)/2., ny/2.)
    x,y = np.meshgrid(yy,xx)
    q = np.sqrt(x**2 + y**2)
    angles = np.arctan2(x,y)

    mask = np.ones_like(angles)
    if (angRange is not None) and (centralAngle is not None):
        mask = np.empty_like(q)
        mask.fill(0)
        centralAngleRadians = centralAngle * np.pi/180
        angRangeRadians = 0.5 * angRange * np.pi/180
        w = np.where(abs(angles-centralAngleRadians)<angRangeRadians)
        mask[w] = 1
        maskCopy = np.fliplr(np.flipud(mask))
        mask[maskCopy==1] = 1
    return mask


def find_radial_average(im, mask=None, centralAngle=None, angRange=None):
    '''
    For a single 2D matrix, finds radial average

    '''
    #From https://github.com/MathieuLeocmach/DDM/blob/master/python/DDM.ipynb
    nx,ny = im.shape

    if (centralAngle!=None) and (angRange!=None) and (mask==None):
        mask = generate_mask(im, centralAngle, angRange)
    elif mask==None:
        mask = np.ones_like(im)

    #dists = np.sqrt(np.fft.fftfreq(shape[0])[:,None]**2 +  np.fft.fftfreq(shape[1])[None,:]**2)

    dists = np.sqrt(np.arange(-1*nx/2, nx/2)[:,None]**2 + np.arange(-1*ny/2, ny/2)[None,:]**2)

    #because sometimes there is a "cross" shape in Fourier transform:
    #dists[0] = 0
    #dists[:,0] = 0

    bins = np.arange(max(nx,ny)/2+1)
    histo_of_bins = np.histogram(dists, bins)[0]
    h = np.histogram(dists, bins, weights=im*mask)[0]
    return h/histo_of_bins


def radial_avg_ddm_matrix(ddm_matrix, mask=None,
                          centralAngle=None, angRange=None,
                          remove_vert_line=True):
    r"""Radially averages DDM matrix. 
    
    For DDM analysis, if we can assume isotropic dynamics, we radially average 
    the DDM matrix so that the data is a function of the magnitude of the wavevector 
    q (rather than on the vector determined by q_x and q_y).
    
    Parameters
    ----------
    ddm_matrix : array
        DDM matrix to be radially averaged. This must be a 3D matrix. The first 
        dimension corresponds to lag time. The second and third dimensions are the
        x and y components of the wavevector. 
    mask : {None}, optional
        Array to be applied as mask to DDM matrix.
    centralAngle : {None}, optional
        DESC
    angRange : {None}, optional
        DESC
    remove_vert_line : {True}, optional
        DESC
        
    Return
    ------
    ravs : array
        Radially averaged DDM matrix. This will be a 2D array. The first dimension 
        corresponds to the lag time. The second dimension corresponds to the 
        magnitude of the wavevector. 
    
    
    """
    
    #From https://github.com/MathieuLeocmach/DDM/blob/master/python/DDM.ipynb
    nx,ny = ddm_matrix[0].shape
    dists = np.sqrt(np.arange(-1*nx/2, nx/2)[:,None]**2 + np.arange(-1*ny/2, ny/2)[None,:]**2)

    bins = np.arange(max(nx,ny)/2+1) - 0.5
    histo_of_bins = np.histogram(dists, bins)[0]

    if (centralAngle!=None) and (angRange!=None) and (mask==None):
        mask = generate_mask(ddm_matrix[0], centralAngle, angRange)
    elif mask==None:
        mask = np.ones_like(ddm_matrix[0])

    array_to_radial_avg = ddm_matrix[0]
    if remove_vert_line:
        array_to_radial_avg[:,int(ny/2)]=0
    h = np.histogram(dists, bins, weights=mask*array_to_radial_avg)[0]

    ravs = np.zeros((ddm_matrix.shape[0], len(h)))
    ravs[0] = h/histo_of_bins

    for i in range(1,ddm_matrix.shape[0]):
        array_to_radial_avg = ddm_matrix[i]
        if remove_vert_line:
            array_to_radial_avg[:,int(ny/2)]=0
        h = np.histogram(dists, bins, weights=mask*array_to_radial_avg)[0]
        ravs[i] = h/histo_of_bins
    return ravs


def get_MSD_from_DDM_data(q, A, D, B, qrange_to_avg):
    r'''
    Finds the mean squared displacement (MSD) from the DDM matrix as well as values
    for the amplitude (A) and background (B). Uses the method described in the papers below. [1]_ [2]_
    
    
    .. math:: MSD(\Delta t) = \frac{4}{q^2} \ln [\frac{A(q)}{A(q)-D(q,\Delta t)+B(q)}]
    
    
    
    Parameters
    ----------
    q : array
        1D array of the magnitudes of wavevectors
    A : array
        Array of same size as q of the amplitude
    D : array
        2D array containing the DDM matrix
    B : array or float
        Background
    qrange_to_avg : array_like
        2-element array or list. 
        
    Returns
    -------
    msd_mean : array
        Mean squared displacment, averaged over the range of q values specified
    msd_stddev : array
        Standard deviation of the mean squared displacements
        
    
    References
    ----------
    .. [1] Bayles, A. V., Squires, T. M. & Helgeson, M. E. Probe microrheology without particle tracking by differential dynamic microscopy. Rheol Acta 56, 863–869 (2017).
    .. [2] Edera, P., Bergamini, D., Trappe, V., Giavazzi, F. & Cerbino, R. Differential dynamic microscopy microrheology of soft materials: A tracking-free determination of the frequency-dependent loss and storage moduli. Phys. Rev. Materials 1, 073804 (2017).


    '''
    msd = (4./(q*q)) * np.log(A / (A-D+B))
    msd_mean = msd[qrange_to_avg[0]:qrange_to_avg[1],:].mean(axis=0)
    msd_stddev = msd[qrange_to_avg[0]:qrange_to_avg[1],:].std(axis=0)
    return msd_mean, msd_stddev



