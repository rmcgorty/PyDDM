"""
Preprocessing of data and core part of DDM analysis
"""
#import modules
import yaml
import os
import glob
import pickle
import numpy as np
import itertools
import ddm_calc as ddm
from scipy.special import gamma
from skimage import io 
from skimage.transform import downscale_local_mean #For binning
import socket
import time
import pandas as pd
import xarray as xr
import re
import copy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
try:
    from nd2reader import ND2Reader #https://github.com/Open-Science-Tools/nd2reader
    able_to_open_nd2 = True
except ModuleNotFoundError:
    print("nd2reader module not found. Reading of .nd2 files disabled.")
    able_to_open_nd2 = False
try:
    import dcimg_mod as dcimg
    able_to_open_dcimg = True
except ModuleNotFoundError:
    print("dcimg readiner not found. try 'pip install dcimg'")
    able_to_open_dcimg = False
try:
    import imageio #use 'conda install -c conda-forge imageio' and 'conda install -c conda-forge imageio-mmpeg'
    able_to_open_mp4 = True
except ModuleNotFoundError:
    print("imageio not installed.")
    able_to_open_mp4 = False
try:
    from readlif.reader import LifFile
    able_to_open_lif = True
except ModuleNotFoundError:
    print('readlif not installed')
    able_to_open_lif = False


import fit_parameters_dictionaries as fpd
import utils as hf #used to be called 'helper functions'
from sklearn.metrics import r2_score
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor
from IPython.display import display


def apply_binning(im, binsize):
    r"""Bin a series of images by a given factor

    :param im: The movie, a series of frames in ndarry format.
    :type im: ndarray

    :param binsize: the number, n gives a square of n x n dimension that should be combined to one pixel
    :type binsize: int


    :return:
        * binned_series (*ndarray*)- Binned time series

    """
    binned_series = downscale_local_mean(im, (1,binsize,binsize))

    return binned_series


def recalculate_ISF_with_new_background(ddm_dataset, 
                                        background_method = None,
                                        background_val = None):
    r"""
    The intermediate scattering function (ISF) is re-calculated from the DDM matrix, with the given background value.
    
    Recall the relationship between the DDM matrix (:math:`D(q,\Delta t)`) and the ISF (:math:`f(q, \Delta t)`): 
        
    .. math::  D(q, \Delta t) = A(q) \left[ 1 - f(q, \Delta t) \right] + B(q)
    
    We can estimate the amplitude,:math:`A(q)`, and background, :math:`B(q)`, terms by first calculating the Fourier 
    transforms of all images and averaging those together. See the function :py:func:`PyDDM.ddm_calc.determining_A_and_B`. With 
    that function and by assuming that the background is a constant with wavevector (i.e., independent of :math:`q`) we can determine 
    :math:`A(q)` and :math:`B`. With :math:`D(q,\Delta t)`, :math:`A(q)`, and :math:`B`, we can then find the ISF :math:`f(q, \Delta t)`. 
    This is done automatically when calculating the DDM matrix from the time series of images. 
    
    There are multiple methods for estimating the background. You may select one of four methods available by setting the 
    paramter `background_method` to 0, 1, 2 or 3. Alternatively, you may set the parameter `background_val` to the value of 
    the background that you want to use. 
    
    Parameters
    ----------
    ddm_dataset : xarray Dataset
        Dataset calculated with :py:meth:`PyDDM.ddm_analysis_and_fitting.DDM_Analysis.calculate_DDM_matrix`
    background_method : {0,1,2,3}, optional
        Method for calculating the background. 
        If 0, then we look at :math:`\left< | \tilde{I}(q, t) |^2 \right>_t` and assume that :math:`A(q)` goes to
        zero by the last 10% of q values. 
        If 1, then we estimate the background to be the minimum of the DDM matrix.
        If 2, then we estimate the background to be :math:`\left< DDM\_Matrix(q_{max}, \Delta t) \right>_{\Delta t}`
        If 3, then we estimate the background to be zero. 
    background_val : float, optional
        Default is None. If not None, then will use this value for :math:`B`.

    Returns
    -------
    ddm_dataset : xarray Dataset
        New Dataset with the variables `Amplitude` and `ISF` adjusted.

    """
    
            
    if background_method is not None:
        if background_method not in [0,1,2,3]:
            print("The `background_method` option must be either 0, 1, 2 or 3. Setting to 0.")
            background_method = 0
        if "avg_image_ft" in ddm_dataset:
            avg_ft = ddm_dataset['avg_image_ft']
        elif "av_fft_offrame" in ddm_dataset:
            avg_ft = ddm_dataset['av_fft_offrame']
        if background_method==0:
            number_of_hi_qs = int(0.1*len(ddm_dataset.q))
            ddm_dataset['B'] = 2*avg_ft[-1*number_of_hi_qs:].mean()
            ddm_dataset['B_std'] = 2*avg_ft[-1*number_of_hi_qs:].std()
        elif background_method==1:
            ddm_dataset['B'] = ddm_dataset.ddm_matrix[1:,1:].min()
        elif background_method==2:
            ddm_dataset['B'] = ddm_dataset.ddm_matrix[1:,-1].mean()
        elif background_method==3:
            ddm_dataset['B'] = 0
        ddm_dataset['Amplitude'] = 2 * avg_ft - ddm_dataset.B
        ddm_dataset['ISF'] = 1-(ddm_dataset.ddm_matrix-ddm_dataset.B)/ddm_dataset.Amplitude
        ddm_dataset.attrs['BackgroundMethod'] = background_method
        return ddm_dataset
            
    if background_val is not None:
        if ("avg_image_ft" in ddm_dataset) and ("ddm_matrix" in ddm_dataset):
            ddm_dataset['B'] = background_val
            ddm_dataset['Amplitude'] = 2 * ddm_dataset['avg_image_ft']-background_val
            ddm_dataset['ISF'] = 1-(ddm_dataset.ddm_matrix-background_val)/ddm_dataset.Amplitude
            ddm_dataset.attrs['BackgroundMethod'] = "None"
            return ddm_dataset
        elif ("av_fft_offrame" in ddm_dataset) and ("ravs" in ddm_dataset):
            ddm_dataset['B'] = background_val
            ddm_dataset['Amplitude'] = 2 * ddm_dataset['av_fft_offrame']-background_val
            ddm_dataset['ISF'] = 1-(ddm_dataset.ravs-background_val)/ddm_dataset.Amplitude
            ddm_dataset.attrs['BackgroundMethod'] = "None"
            return ddm_dataset
        else:
            print("No changes to ddm_dataset made. Returning same dataset as was passed.")
            return ddm_dataset


def newt(t,s):
    r"""
    This function is used to determine a new time when a distribution of decay times are present. The new time is the average over all the decay times.

    :param t: Decay time (seconds), e.g. tau
    :type t: float
    :param s: Stretching exponent
    :type s: float


    :return:
        * newt (*float*)- New, average decay time (seconds)

    Theory

    .. math::

        \langle \tau \rangle = \int_{0}^{\infty} e^{\left( \frac{-\Delta t}{\tau}\right)^{s}} \,d \Delta t = \frac{1}{s} \Gamma \left( \frac{1}{s} \right) \tau


     """

    newt = (1./s)*gamma(1./s)*t
    return newt

def print_fitting_models():
    fpd.return_possible_fitting_models()


class DDM_Analysis:
    """Performs preprossing of data, such as cropping, windowing etc. DDM calculations are performed on the processed time series
    to produce a xarray DataSet with DDM matrix, radial averages and ISF. The analysis parameters are provided by the user in 
    a YAML file: :doc:`More information here </Provide_info_for_analysis>` """

    def __init__(self, data_yaml, load_images=True):
        
        self.data_dir = None
        self.filename = None
        self.number_of_lag_times = None
        
        self.loaded_mp4 = False #if loading mp4, image data handled a bit differently 
        self.loaded_lif = False #Lif loading also handled a bit differently?
        
        if (isinstance(data_yaml, str)) or (isinstance(data_yaml, dict)):
            self.data_yaml=data_yaml
        else:
            print("Incorrect data type for analysis parameters. Argument must be filename or dictionary.")
        success = self.loadYAML()
        if success:
            self.setup(load_images)
            self.computer_name = socket.gethostname()


    def __str__(self):
        
        if isinstance(self.data_yaml, str):
            return f"""
            DDM Analysis:
                analysis parameters: {self.data_yaml}
                data directory: {self.data_dir}
                file name: {self.filename}
                number of lag times: {self.number_of_lag_times}"""
        else:
            return f"""
            DDM Analysis:
                data directory: {self.data_dir}
                file name: {self.filename}
                number of lag times: {self.number_of_lag_times}"""


    def loadYAML(self):
        r"""Opens yaml file and extracts parameter values.
        
        To analyze a recorded movie, there must be associated metadata. That can either 
        be in the form of a yaml file or string in the yaml format. The yaml file 


        """
        if isinstance(self.data_yaml, str):
            doesYAMLFileExist = os.path.exists(self.data_yaml)
            if doesYAMLFileExist:
                with open(self.data_yaml) as f:
                    self.content = yaml.safe_load(f)
            else:
                print("File %s does not exist. Check file name or path." % self.data_yaml)
                #ddm.logger2.error("File %s does not exist. Check file name or path." % self.data_yaml)
                return 0
        elif isinstance(self.data_yaml, dict):
            self.content = self.data_yaml.copy()

        # The YAML data *MUST* include the paramters 'DataDirectory' and 'FileName'
        self.data_dir=self.content['DataDirectory']
        self.filename= self.content['FileName']
        

        #Make sure the path to the movie exists before proceeding
        if os.path.exists(self.data_dir+self.filename):
            #print('File path to image data exists.')
            #ddm.logger.info("File path to image data exists.")
            if 'Metadata' in self.content:
                self.metadata = self.content['Metadata']
                self.pixel_size = self.metadata['pixel_size']
                self.frame_rate = self.metadata['frame_rate']
                if 'channel' in self.metadata:
                    self.channel = self.metadata['channel']
                else:
                    self.channel = None
            else:
                print("Image metadata not provided!!! Setting `pixel_size` and `frame_rate` to 1.")
                self.pixel_size = 1
                self.frame_rate = 1
                
            self.analysis_parameters = self.content['Analysis_parameters']
            if 'filename_for_saved_data' in self.analysis_parameters:
                self.filename_for_saving_data = self.analysis_parameters['filename_for_saved_data']
            else:
                self.filename_for_saving_data = self.filename[:-4]
                if self.channel is not None:
                    if self.channel in [0,1,2,3]:
                        self.filename_for_saving_data = "%s_c=%i" % (self.filename_for_saving_data, self.channel)
            if 'ending_frame_number' in self.analysis_parameters:
                self.last_frame = self.analysis_parameters['ending_frame_number']
            else:
                self.last_frame = None
            if 'starting_frame_number' in self.analysis_parameters:
                self.first_frame = self.analysis_parameters['starting_frame_number']
            else:
                self.first_frame = 0
            if 'number_lagtimes' in self.analysis_parameters:
                self.number_of_lag_times = self.analysis_parameters['number_lagtimes']
            elif 'number_lag_times' in self.analysis_parameters:
                self.number_of_lag_times = self.analysis_parameters['number_lag_times']
            else:
                print("Number of lag times not specified!!! Setting to 50.")
                self.number_of_lag_times = 50
            if 'first_lag_time' in self.analysis_parameters:
                self.first_lag_time = self.analysis_parameters['first_lag_time']
            else:
                self.first_lag_time = 1
            self.last_lag_time = self.analysis_parameters['last_lag_time']
            if 'central_angle' in self.analysis_parameters:
                self.central_angle = self.analysis_parameters['central_angle']
            else:
                self.central_angle = None
            if 'angle_range' in self.analysis_parameters:
                self.angle_range = self.analysis_parameters['angle_range']
            else:
                self.angle_range = None
            if 'number_differences_max' in self.analysis_parameters:
                self.num_dif_max = self.analysis_parameters['number_differences_max']
            else:
                self.num_dif_max = None
              
            self.crp_region = None
            if 'crop_to_roi' in self.analysis_parameters:
                if self.analysis_parameters['crop_to_roi'] is not None:
                    if len(self.analysis_parameters['crop_to_roi'])==4:
                        #if there is a list with pixel coordinates for cropping the new image will be cropped
                        self.crp_region = self.analysis_parameters['crop_to_roi']
            self.binsize = 1
            if 'binning' in self.analysis_parameters:
                if self.analysis_parameters['binning']:
                    if 'bin_size' in self.analysis_parameters:
                        self.binsize = self.analysis_parameters['bin_size']
            
            #These paramters (overlap_method and background_method) could also 
            #be set as passed parameter to the method `calculate_DDM_matrix`
            self.overlap_method = None
            self.background_method = None
            if 'overlap_method' in self.analysis_parameters:
                if self.analysis_parameters['overlap_method'] in [0,1,2,3]:
                    self.overlap_method = self.analysis_parameters['overlap_method']
                else:
                    print("Parameter 'overlap_method' must be 0, 1, 2, or 3.")
            if 'background_method' in self.analysis_parameters:
                if self.analysis_parameters['background_method'] in [0,1,2,3]:
                    self.background_method = self.analysis_parameters['background_method']
                else:
                    print("Parameter 'overlap_method' must be 0, 1, 2, or 3.")
                
            print(f'Provided metadata: {self.metadata}')
            #ddm.logger2.info(f'Provided metadata: {self.metadata}')
            
            return 1
        else:
            print('Error: check path to image file')
            return 0


    def set_filename_for_saving(self, filename, quiet=False):
        '''
        Change file name to save the data to disk. This is the file that will 
        store the `ddm_dataset` as a netCDF file (with extension .nc).

        :param filename: New file name
        :type filename: str


        '''
        if not quiet:
            print("Previous filename for saving ddm data was %s." % self.filename_for_saving_data)
        self.filename_for_saving_data = filename
        if not quiet:
            print("New filename for saving data will be %s." % self.filename_for_saving_data)


    def _openImage(self, load_images):
        '''
        Opens .nd2 file or .tif file and returns image series as multidimensional numpy array

        :return:
            * im (*numpy array*)- image series as numpy array

        '''
        if not load_images:
            return None

        if re.search(".\.nd2$", self.filename) is not None:
            if able_to_open_nd2:
                # Files with nd2 extension will be read using the package
                #  nd2reader. Nikon systems may save data with this file type.
                if self.channel is None:
                    print("Need to specify channel in yaml metadata. Defaulting to c=0.")
                    self.channel = 0
                with ND2Reader(self.data_dir+self.filename) as images:
                    # Metadata in nd2 file should have pixel size and other information.
                    #   However, data provided by user in yaml file will be used.
                    #   Perhaps later perform check that nd2 file metadata and yaml metadata agree.
                    print("According to nd2 file:")
                    print('\t%d x %d px' % (images.metadata['width'], images.metadata['height']))
                    print('\tPixel size of: %.2f microns' % images.metadata['pixel_microns'])
                    print('\tNumber of frames: %i' % images.sizes['t'])
                    im = np.zeros((images.sizes['t'], images.metadata['width'], images.metadata['height']), dtype=np.uint16)
                    for i in range(images.sizes['t']):
                        im[i] = images.get_frame_2D(t=i, c=self.channel)
            else:
                print("It seems you have an nd2 file to open. But nd2reader not installed!")
                return 

        if re.search(".\.lif$", self.filename) is not None:
            if able_to_open_lif:
                # Files with lif extension will be read using the package
                #  readlif. Leica systems may save data with this file type.
                if self.channel is None:
                    print("Need to specify channel/series in yaml metadata. Defaulting to c=0.")
                    self.channel = 0
                    
                lif_img = LifFile(self.data_dir+self.filename).get_image(self.channel)
                (x, y, z, t, m) = lif_img.dims
                
                scale_factor = (16 - lif_img.bit_depth[0]) ** 2
                if scale_factor == 0:
                    scale_factor = 1
                
                print('\t%d x %d px' % (x, y))
                print('\tPixel size of: %.2f microns' % lif_img.scale_n['X'])
                print('\tNumber of frames: %i' % t)
                                
                start_frame = self.first_frame
                if self.last_frame == None:
                    end_frame = t
                else:
                    end_frame = self.last_frame

                y1 = 0
                x1 = 0
                y2 = y
                x2 = x

                if 'crop_to_roi' in self.analysis_parameters:
                    if self.analysis_parameters['crop_to_roi'] is not None:
                        if len(self.analysis_parameters['crop_to_roi'])==4:
                            [y1,y2,x1,x2] = self.analysis_parameters['crop_to_roi']


                im = np.zeros(((end_frame-start_frame), x2-x1, y2-y1), dtype=np.uint16)
                
                for i in range(start_frame, end_frame):
                    im[i-start_frame] = lif_img.get_frame(z = 0, t = i, c = 0)[y1:y2, x1:x2]*scale_factor
                self.loaded_lif = True
                self.image_for_report = lif_img.get_frame(z=0, t=start_frame, c=0)
            else:
                print("It seems you have an lif file to open. But readlif not installed!")
                return 


        if (re.search(".\.tif$", self.filename) is not None) or (re.search(".\.tiff$", self.filename) is not None):
            im = io.imread(self.data_dir + self.filename)
            
        if (re.search(".\.dcimg$", self.filename) is not None):
            if able_to_open_dcimg:
                dcimg_loaded = dcimg.DCIMGFile(self.data_dir + self.filename)
                im = np.zeros(dcimg_loaded.shape, dtype=np.uint16)
                for i in range(im.shape[0]):
                    im[i] = dcimg_loaded[i]
            else:
                print("dcimg not loaded...")
                return
            
        if (re.search(".\.mp4$", self.filename) is not None):
            if able_to_open_mp4:
                vid = imageio.get_reader(self.data_dir + self.filename)
                numFrames = vid.count_frames()
                im1 = vid.get_data(0)
                if self.crp_region is not None:
                    im1 = im1[self.crp_region[0]:self.crp_region[1], self.crp_region[2]:self.crp_region[3],:]
                if self.binsize > 1:
                    im1 = downscale_local_mean(im1[:,:,0], (self.binsize, self.binsize))
                else:
                    im1 = im1[:,:,0]
                if self.last_frame is not None:
                    numFrames = self.last_frame - self.first_frame
                im = np.zeros((numFrames, im1.shape[0], im1.shape[1]), dtype=np.uint16)
                for i in range(self.first_frame, self.first_frame + numFrames):
                    temp = vid.get_data(i)
                    if self.crp_region is not None:
                        temp = temp[self.crp_region[0]:self.crp_region[1], self.crp_region[2]:self.crp_region[3],:]
                    if self.binsize > 1:
                        temp = downscale_local_mean(temp[:,:,0], (self.binsize, self.binsize))
                    else:
                        temp = temp[:,:,0]
                    im[i-self.first_frame] = temp
                self.loaded_mp4 = True
            else:
                print("mp4 opener not loaded...")
                return

        return im


    def setup(self, load_images):
        r"""Based off user-provided parameters, prepares images for the DDM analysis. 
        
        Using the parameters under the 'Analysis_parameters' section of the 
        YAML file, this will get the image stack ready for DDM analysis. Possible 
        things to do done include cropping the image to focus on a particular 
        region of interest (ROI), binnning the image, applying a windowing function. 

        """

        #??????????????????????????
        #Goal: in _openImage, only grab required frames/crop


        image_data = self._openImage(load_images)
        
        
        if image_data is None:
            print("Image not loaded.")
        else:
            print("Image shape: %i-by-%i-by-%i" % image_data.shape)
            
            if self.loaded_mp4:
                print("Loaded mp4 file.")
                self.im=image_data
                self.image_for_report = self.im[0]
                self.pixel_size = self.pixel_size*self.binsize
                return
    
            
            #Lif files do work beforehand like mp4s
            #still need to implement binning, 4rois
            if self.loaded_lif: #Lif only loads needed frames
                self.im=image_data
                print("Loaded lif file.")
                return
            
            #crops the number of frames based on given max frame numbers
            if self.last_frame is None:
                self.im=image_data[self.first_frame::,:,:]
            elif self.last_frame <= self.last_lag_time:
                print('The last frame number should be higher than the frame for the last lag time')
                self.last_lag_time = self.last_frame-1
                print('Setting last_lag_time to %i.' % self.last_lag_time)
            else:
                self.im=image_data[self.first_frame:self.last_frame,:,:]
            print('Number of frames to use for analysis: %i' % self.im.shape[0])
            print('Maximum lag time (in frames): %i' % self.last_lag_time)
            print('Number of lag times to compute DDM matrix: %i' % self.number_of_lag_times)
    
            self.image_for_report = self.im[0]

            if 'crop_to_roi' in self.analysis_parameters:
                if self.analysis_parameters['crop_to_roi'] is not None:
                    if len(self.analysis_parameters['crop_to_roi'])==4:
                        #if there is a list with pixel coordinates for cropping the new image will be cropped
                        crp_region = self.analysis_parameters['crop_to_roi']
                        self.im = self.im[:, crp_region[0]:crp_region[1], crp_region[2]:crp_region[3]]
                        print('New dimensions after cropping: %i-by-%i' % self.im.shape[1:])
                    else:
                        print("For cropping images, 'crop_to_roi' must be list of four integers.")
                        print("Using the full frame, dimensions: %i-by-%i." % self.im.shape[1:])
            else:
                #Just keep the whole frame
                print("Using the full frame, dimensions: %i-by-%i." % self.im.shape[1:])
    
            if 'split_into_4_rois' in self.analysis_parameters:
                if self.analysis_parameters['split_into_4_rois']:
                    print('Splitting into four tiles...')
                    #split image into four tiles
                    newarr = np.dsplit(self.im,2) #splits vertically
                    roi0, roi1 = np.hsplit(newarr[0],2) #split horizontally
                    roi2, roi3 = np.hsplit(newarr[1],2)
                    print(f'New dimensions for ROIs: {roi0.shape}')
                    
                    if 'use_windowing_function' in self.analysis_parameters:
                        if self.analysis_parameters['use_windowing_function']:
                            print("Applying windowing function to each ROI...")
                            roi0 = ddm.window_function(roi0)*roi0
                            roi1 = ddm.window_function(roi1)*roi1
                            roi2 = ddm.window_function(roi2)*roi2
                            roi3 = ddm.window_function(roi3)*roi3
        
                    self.im = [roi0, roi1, roi2, roi3]
    
            else:
                if 'use_windowing_function' in self.analysis_parameters:
                    if self.analysis_parameters['use_windowing_function']:
                        print("Applying windowing function...")
                        self.im=ddm.window_function(self.im)*self.im
    
            #After cropping, the images might be binned, this is done before splitting in tiles
            if 'binning' in self.analysis_parameters:
                if self.analysis_parameters['binning']:
                    if 'bin_size' in self.analysis_parameters:
                        self.binsize = self.analysis_parameters['bin_size']
                    else:
                        print("Bin size not set! Using 2x2 binning. Re-run with 'binning' as false if no binning desired.")
                        self.binsize = 2
                    if type(self.im) == list:
                        for i,im in enumerate(self.im):
                            self.im[i] = apply_binning(im, self.binsize)
                        dims_after_binning = self.im[0].shape
                    else:
                        self.im = apply_binning(self.im, self.binsize)
                        dims_after_binning = self.im.shape
    
                    #The number of pixels has been reduced by binning procedure, therefore the pixel size overwritten:
                    self.pixel_size = self.pixel_size*self.binsize
                    print("Applying binning...")
                    print(f'Dimensions after binning {dims_after_binning}, the new pixel size {self.pixel_size}')

        

    def calculate_DDM_matrix(self, quiet=False, velocity=[0,0],
                             bg_subtract_for_AB_determination=None, **kwargs):

        r"""Calculates the DDM matrix
        This function computes the DDM matrix. The radially averaged DDM matrix will
        then also be found, along with estimates of the background and amplitude. From 
        the amplitude and background, we can extract the intermediate scattering function 
        (ISF) from the DDM matrix. All of these computed variables will be stored as 
        an xarray Dataset. 

        Parameters
        ----------
        quiet : boolean (optional)
            If set to `False`, then when calculating the DDM matrix, a message will print out at 
            about every fourth time lag calculated (with a timestamp)
        velocity : array-like (optional)
            Velocity in x and y direction. 
        bg_subtract_for_AB_determination: {None, 'mode', 'median'}
            Deafault option is None. In method to find A(q) and B, one can subtract the mode or median of
            the images. 
        **overlap_method : {0,1,2,3}, optional
            Optional keyword argument. Will be set to 2 if not specified here nor in the YAML file. 
            Determines how overlapped the different image pairs are. Let's say you are finding all pairs 
            of images separated by a lag time of 10 frames. You'd have frame 1 and 11, 2 and 12, 3 and 13, etc. 
            One could use *each* possible pair when calculating the DDM matrix (overlap_method=3, maximally 
            overlapping pairs). One could only look at non-overlapping pairs, like frame 1 and 11, 11 and 21, 21 and 31, etc 
            (overlap_method=0, non-overlapping). Or one could do something in between. For overlap_method=2 (the DEFAULT), there will be 
            about 3 or 4 pairs (at most) between two frames. That is, if we have a lag time of 10 frames, we will 
            use the pairs of frame 1 and 11, 5 and 15, 9 and 19, etc. For overlap_method=1, we do something similar to
            overlap_method=2, but we compute *at most* `number_differences_max` (default:300) image differences per lag time. 
            So for example, with a lag time of 1 frame and 
            a movie that has 1000 frames, we could theoretically use 999 differences of images (and we would for the other methods). But 
            for overlap_method=2, we would only use 50 of those. This is for quickening up the computation. 
        **background_method : {0,1,2,3}, optional
            Optional keyword argument. Will be set to 0 if not specified here nor in the YAML file. 
            Determines how we estimate the :math:`B` parameter. This can be done by lookinag the average of the 
            Fourier transform of each image (not image difference!) squared (that's background_method=0). We could 
            also use the minimum of the DDM matrix (background_method=1). Or we could use the average DDM matrix for 
            the largest q (background_method=2). Or we could just set :math:`B` to zero (background_method=3).
        **number_lag_times : int
            Optional keyword argument. Must be set in the YAML file. 
            You may pass this optional keyword argument if you want to overwrite the value for the number of lag 
            times set in the YAML file. 
            
        Returns
        -------
        ddm_dataset : xarray Dataset
            Dataset containing the DDM matrix, the estimate of the background and amplitude, the ISF, etc. Coordinates 
            of these variables will be the wavevector (either as q_x, q_y or the magnitude q), lagtime, etc.


        """

        filename_to_be_used = f"{self.data_dir}{self.filename_for_saving_data}_ddmmatrix.nc"
        if os.path.exists(filename_to_be_used):
            print(f"The file {filename_to_be_used} already exists. So perhaps the DDM matrix was calculated already?")
            answer = input("Do you still want to calculate the DDM matrix? (y/n): ").lower().strip()
            if answer == "n" or answer=="no":
                with xr.open_dataset(filename_to_be_used) as ds:
                    self.ddm_dataset = ds.load()
                    self.ddm_dataset.close()
                return

        if 'overlap_method' in kwargs:
            self.overlap_method = kwargs['overlap_method']
        if 'background_method' in kwargs:
            self.background_method = kwargs['background_method']
        if 'number_lag_times' in kwargs:
            self.number_of_lag_times = kwargs['number_lag_times']
        if 'number_differences_max' in kwargs:
            self.num_dif_max = kwargs['number_differences_max']
        else:
            self.num_dif_max = 300

            
        #If 'overlap_method' or 'background_method' were *not* set in the YAML file, then
        # those will be None. So check if None, and if so, set to default value. 
        # These parameters must also be either 0, 1, 2, or 3. So check that as well. 
        if (self.overlap_method is None) or (self.overlap_method not in [0,1,2,3]):
            self.overlap_method = 2
        if (self.background_method is None) or (self.background_method not in [0,1,2,3]):
            self.background_method = 0
            
            
        self.lag_times_frames = ddm.generateLogDistributionOfTimeLags(self.first_lag_time, self.last_lag_time,
                                                                      self.number_of_lag_times)
        self.lag_times = self.lag_times_frames / self.frame_rate

        #print(f"Calculating the DDM matrix for {self.filename}...")
        self._computeDDMMatrix(quiet=quiet, velocity=velocity, bg_subtract_for_AB_determination=bg_subtract_for_AB_determination)

        return self.ddm_dataset
    

    #Do not call this function, instead call analysis_flow
    def _computeDDMMatrix(self, quiet=False, velocity=[0,0],
                          bg_subtract_for_AB_determination=None):
        '''
        Calculates the DDM matrix and radial averages then estimates
        amplitude (A) and background (B) based on direct fourier transform (FFT)
        of images (instead of FFT of difference images). Determines the ISF from
        A, B and radial averages. The data set is saved as netCDF file and a
        pdf report is produced
        '''
        #Calculate q_x and q_y and q which will function as coordinates
        if type(self.im)==list:
            self.q_y=np.sort(np.fft.fftfreq(self.im[0].shape[1], d=self.pixel_size))*2*np.pi
            self.q_x=self.q_y
            self.q=np.arange(0,self.im[0].shape[1]/2)*2*np.pi*(1./(self.im[0].shape[1]*self.pixel_size))
        else:
            self.q_y=np.sort(np.fft.fftfreq(self.im.shape[1], d=self.pixel_size))*2*np.pi
            self.q_x=self.q_y
            self.q=np.arange(0,self.im.shape[1]/2)*2*np.pi*(1./(self.im.shape[1]*self.pixel_size))
            
        if (type(self.im)==list) or (type(self.im)==np.ndarray):
            pass
        else:
            print("Image data not yet read!")
            return False
        
        start_time = time.time()
        self.ddm_matrix = []
        if (abs(velocity[0]) > 0) or (abs(velocity[1]) > 0):
            print("Will run DDM computation to correct for velocity...")
            print(velocity)
            vx = velocity[0] / self.frame_rate
            vy = velocity[1] / self.frame_rate
            self.ddm_matrix, self.num_pairs_per_dt = ddm.computeDDMMatrix_correctVelocityPhase(self.im, self.lag_times_frames, 
                                                                            [vx,vy], self.pixel_size, quiet=quiet,
                                                                            overlap_method=self.overlap_method, 
                                                                            number_differences_max=self.num_dif_max)
            end_time = time.time()
        else:
            try:
                if type(self.im)==list:
                    for i,im in enumerate(self.im):
                        print(f"Getting DDM matrix for {i+1} of {len(self.im)}...")
                        d_matrix, num_pairs = ddm.computeDDMMatrix(im, self.lag_times_frames, quiet=quiet,
                                                                   overlap_method=self.overlap_method,
                                                                   number_differences_max=self.num_dif_max)
                        self.ddm_matrix.append(d_matrix)
                    self.num_pairs_per_dt = num_pairs
                else:
                    self.ddm_matrix, self.num_pairs_per_dt = ddm.computeDDMMatrix(self.im, self.lag_times_frames, 
                                                                                  quiet=quiet,
                                                                                  overlap_method=self.overlap_method,
                                                                                  number_differences_max=self.num_dif_max)
    
                end_time = time.time()
            except:
                print("Unable to get DDM matrix.")
                return False

        print("DDM matrix took %s seconds to compute." % (end_time - start_time))

        if type(self.im)==list:
            self.ravs = []
            for i,d in enumerate(self.ddm_matrix):
                ravs = ddm.radial_avg_ddm_matrix(d, centralAngle=self.central_angle,
                                                 angRange=self.angle_range)
                self.ravs.append(ravs)
        else:
            self.ravs = ddm.radial_avg_ddm_matrix(self.ddm_matrix, centralAngle=self.central_angle,
                                                  angRange=self.angle_range)
            
            
        if type(self.im)==list:
            self.AF = []
            for i,d in enumerate(self.ddm_matrix):
                af,af_axis = self.find_alignment_factor(d)
                self.AF.append(af)
        else:
            self.AF,af_axis = self.find_alignment_factor(self.ddm_matrix)
        self.af_axis = af_axis
            

        #Determine Amplitude and Background from radial averages of directly fourier transformed images (not difference images)
        # Note: windowing (if applicable) already applied to self.im, so "use_BH_filter" should be False always
        if type(self.im)==list:
            self.ravfft = []
            for i,im in enumerate(self.im):
                r = ddm.determining_A_and_B(im, use_BH_filter=False,centralAngle=self.central_angle,
                                            angRange=self.angle_range,
                                            subtract_bg = bg_subtract_for_AB_determination)
                self.ravfft.append(r)
        else:
            self.ravfft = ddm.determining_A_and_B(self.im, use_BH_filter=False,
                                                  centralAngle=self.central_angle,
                                                  angRange=self.angle_range,
                                                  subtract_bg = bg_subtract_for_AB_determination)


        if type(self.im)==list:
            self.ddm_dataset = []
            for i in range(len(self.im)):
                filename = f"{self.data_dir}{self.filename_for_saving_data}_{i:02}"
                ds = self._create_dataset_and_report(filename, num=i)
                self.ddm_dataset.append(ds)
        else:
            filename = f"{self.data_dir}{self.filename_for_saving_data}"
            self.ddm_dataset = self._create_dataset_and_report(filename)



    def _create_dataset_and_report(self, file_name, num=None):
        r"""
        Creates the xarray Dataset and PDF report.

        Parameters
        ----------
        file_name : string
            This string with "_ddmmatrix.nc" appended at the end will 
            be the name of the file the xarray Dataset is saved for. 
        num : int, optional
            If divividing into multiple ROIs, thsi will be used. 
            The default is None.

        Returns
        -------
        ddm_dataset : xarray Dataset
            Dataset containing the DDM matrix and associated data
            and metadata.

        """
        if type(self.ddm_matrix)==list:
            if (num==None) or (len(self.ddm_matrix)<num):
                num=0
                print("Function '_create_dataset' needs 'num' parameter. Setting to 0.")
            ddm_matrix = self.ddm_matrix[num]
            ravfft = self.ravfft[num]
            ravs = self.ravs[num]
            image0 = self.im[num][0].astype(np.float64)
            AF = self.AF[num]
        else:
            ddm_matrix = self.ddm_matrix
            ravfft = self.ravfft
            ravs = self.ravs
            image0 = self.im[0].astype(np.float64)
            AF = self.AF

        #Put ddm_matrix and radial averages in a dataset:
        ddm_dataset=xr.Dataset({'ddm_matrix_full':(['lagtime', 'q_y','q_x'], ddm_matrix), #was 'ddm_matrix'
                                'ddm_matrix':(['lagtime', 'q'], ravs), #was 'ravs'
                                'first_image':(['y','x'], image0),
                                'alignment_factor':(['lagtime','q'], AF)},
                               coords={'lagtime': self.lag_times,
                                       'framelag':('frames', self.lag_times_frames),
                                       'q_y':self.q_y, 'q_x':self.q_x, 'q':self.q,
                                       'y':np.arange(image0.shape[0]), 'x':np.arange(image0.shape[1])},
                               attrs={'units':'Intensity', 'lagtime':'sec',
                                      'q':'μm$^{-1}$',
                                      'x':'pixels', 'y':'pixels',
                                      'info':'ddm_matrix is the averages of FFT difference images, ravs are the radial averages'})
        
        ddm_dataset.attrs['BackgroundMethod'] = self.background_method
        ddm_dataset.attrs['OverlapMethod'] = self.overlap_method
        ddm_dataset.attrs['AlignmentFactorAxis'] = self.af_axis

        ddm_dataset['avg_image_ft'] = (('q'), ravfft[0,:]) # av_fft_offrame=0.5*(A+B) #was 'av_fft_offrame'
        
        #Get last 10% of q. Average used for estimating background
        number_of_hi_qs = int(0.1*len(self.q))
        
        #Number of image differences used to calculate DDM matrix for each lagtime
        ddm_dataset['num_pairs_per_dt'] = (('lagtime'),self.num_pairs_per_dt)
        
        if self.background_method==0:
            ddm_dataset['B'] = 2*ddm_dataset.avg_image_ft[-1*number_of_hi_qs:].mean()
            ddm_dataset['B_std'] = 2*ddm_dataset.avg_image_ft[-1*number_of_hi_qs:].std()
            print(f" Background estimate ± std is {ddm_dataset.B.values:.2f} ± {ddm_dataset.B_std.values:.2f}")
        elif self.background_method==1:
            ddm_dataset['B'] = ddm_dataset.ddm_matrix[1:,1:].min()
            print(f" Background estimate is {ddm_dataset.B.values:.2f}")
        elif self.background_method==2:
            ddm_dataset['B'] = ddm_dataset.ddm_matrix[1:,-1].mean()
            print(f" Background estimate is {ddm_dataset.B.values:.2f}")
        elif self.background_method==3:
            ddm_dataset['B'] = 0
            print(f" Background estimate is {ddm_dataset.B.values:.2f}")
        

        # Calculate amplitude: av_fft_frame=0.5(A+B)->A=2*av_fft_frame-B
        ddm_dataset["Amplitude"] = (2 * ddm_dataset['avg_image_ft']) - ddm_dataset.B

        # calculate ISF with new amplitude and background
        ddm_dataset['ISF']=1-(ddm_dataset.ddm_matrix-ddm_dataset.B)/ddm_dataset.Amplitude

        ##write yaml file data to xarray attribute in format accepeted by net cdf (no multiple dimension or Booleans)
        for i in self.content:
            if not i.startswith('Fit'):
                try:
                    for j, k in self.content[i].items():
                        '''
                        if k==True:
                            ddm_dataset.attrs[j]= 'yes'
                        elif k==False or k==None:
                            ddm_dataset.attrs[j]= 'no'
                        else:
                            ddm_dataset.attrs[j]=k
                        '''
                        if k is None:
                            ddm_dataset.attrs[j] = 'None'
                        elif isinstance(k, bool):
                            if k==True:
                                ddm_dataset.attrs[j] = 'True'
                            elif k==False:
                                ddm_dataset.attrs[j] = 'False'
                        else:
                            ddm_dataset.attrs[j] = k
                except:
                    ddm_dataset.attrs[i]=self.content[i]


        ## write the ddm matrix to disk to the folder where the movie is located
        try:
            ddm_dataset.to_netcdf(f"{file_name}_ddmmatrix.nc", mode='w')
        except:
            print(f'Could not save data set as: {file_name}_ddmmatrix.nc')
            name_alternative=input("Provide an alternative name (do not include .nc suffix) [MUST SPECIFY FULL PATH!]: ")
            ddm_dataset.to_netcdf(f"{name_alternative}.nc", mode='w')
            print("note this name should be entered in the yaml file for fitting")

        with PdfPages(f"{file_name}_report.pdf") as pdf:
            self.generate_plots(ddm_dataset, pdf_to_save_to=pdf, num=num)

        return ddm_dataset
    
    def variationInDDMMatrix(self, lagtime, orientation_axis=0,
                             save_full_ddmmat=True,
                             velocity=[0,0]):
        r"""
        Creates the xarray Dataset and PDF report.

        Parameters
        ----------
        lagtime : int or list-like
            If an integer, will compute the DDM matrix for this
            lag time over all times. If an arrary or list, then 
            all lag times in that array/list will be used.
        orientation_axis : float, optional
            Axis for computing the alignment factor of the 2D
            DDM matrix for a given lag time and time
        save_full_ddmmat : bool, optional
            If True, then will save the DDM matrix as a function 
            of q_x and q_y for each lag time and time. If False, 
            then it will radially average that matrix so that the 
            DDM matrix is just a function of the magnitude of the 
            wavevector, the lagtime, and the time. If True, then 
            it may potentially use up a lot of memory. 
        velocity : list-like, optional
            Deafult is [0,0]. If not [0,0], then will use phiDM
            to correct for drift or ballistic motion. 

        Returns
        -------
        ddm_dataset : xarray Dataset
            Dataset containing the DDM matrix and associated data
            and metadata.

        """
        

        if type(self.im)==list:
            '''
            self.q_y=np.sort(np.fft.fftfreq(self.im[0].shape[1], d=self.pixel_size))*2*np.pi
            self.q_x=self.q_y
            self.q=np.arange(0,self.im[0].shape[1]/2)*2*np.pi*(1./(self.im[0].shape[1]*self.pixel_size))
            '''
            print("Not yet implemented for series of movies. Just use a single movie.")
        else:
            self.q_y=np.sort(np.fft.fftfreq(self.im.shape[1], d=self.pixel_size))*2*np.pi
            self.q_x=self.q_y
            self.q=np.arange(0,self.im.shape[1]/2)*2*np.pi*(1./(self.im.shape[1]*self.pixel_size))
            
        if np.isscalar(lagtime):
            if abs(velocity[0]>0) or abs(velocity[1]>0):
                print("Will run DDM computation to correct for velocity...")
                vx = velocity[0] / self.frame_rate
                vy = velocity[1] / self.frame_rate
                ddmmat, radav_ddmmat = ddm.temporalVarianceDDMMatrix(self.im, lagtime, vel_corr=[vx, vy, self.pixel_size])
                
            else:
                ddmmat, radav_ddmmat = ddm.temporalVarianceDDMMatrix(self.im, lagtime)
            
            number_of_times = ddmmat.shape[0]
            times = np.arange(number_of_times) / self.frame_rate
            
            AF,af_axis = self.find_alignment_factor(ddmmat, orientation_axis=orientation_axis)
                
            #Put ddm_matrix and radial averages in a dataset:
            ddm_dataset=xr.Dataset({'ddm_matrix_full':(['time', 'q_y','q_x'], ddmmat), 
                                    'ddm_matrix':(['time', 'q'], radav_ddmmat), 
                                    'alignment_factor':(['time','q'], AF),
                                    'lagtime_frames':(lagtime),
                                    'lagtime':(lagtime/self.frame_rate)},
                                   coords={'time': times,
                                           'q_y':self.q_y, 'q_x':self.q_x, 'q':self.q})
            
            ddm_dataset.attrs['AlignmentFactorAxis'] = af_axis
            
        else:
            number_of_lag_times = len(lagtime)
            number_of_frames = self.im.shape[0]
            if number_of_lag_times >= number_of_frames:
                lagtime = np.arange(1,number_of_frames-1)
                number_of_lag_times = len(lagtime)
            times = np.arange(number_of_frames-1) / self.frame_rate
            if save_full_ddmmat:
                ddmmat = np.empty((number_of_lag_times, number_of_frames-1, len(self.q_x), len(self.q_y)))
                ddmmat.fill(np.nan)
            radav_ddmmat = np.empty((number_of_lag_times, number_of_frames-1, len(self.q)))
            radav_ddmmat.fill(np.nan)
            AF = np.empty_like(radav_ddmmat)
            AF.fill(np.nan)
            
            for i,lag in enumerate(lagtime):
                if abs(velocity[0]>0) or abs(velocity[1]>0):
                    if i==0:
                        print("Will run DDM computation to correct for velocity...")
                    vx = velocity[0] / self.frame_rate
                    vy = velocity[1] / self.frame_rate
                    ddmmat_temp, radav_ddmmat_temp = ddm.temporalVarianceDDMMatrix(self.im, lag, vel_corr=[vx, vy, self.pixel_size])
                else:
                    ddmmat_temp, radav_ddmmat_temp = ddm.temporalVarianceDDMMatrix(self.im, lag)
                    
                AF_temp,af_axis = self.find_alignment_factor(ddmmat_temp, orientation_axis=orientation_axis)
                
                if save_full_ddmmat:
                    ddmmat[i,:ddmmat_temp.shape[0],:,:] = ddmmat_temp
                    
                radav_ddmmat[i,:radav_ddmmat_temp.shape[0],:] = radav_ddmmat_temp
                AF[i,:AF_temp.shape[0],:] = AF_temp
                
            #Put ddm_matrix and radial averages in a dataset:
            if save_full_ddmmat:
                ddm_dataset=xr.Dataset({'ddm_matrix_full':(['lagtime','time', 'q_y','q_x'], ddmmat), 
                                        'ddm_matrix':(['lagtime','time', 'q'], radav_ddmmat), 
                                        'alignment_factor':(['lagtime','time','q'], AF)},
                                       coords={'time': times,
                                               'lagtime': lagtime,
                                               'q_y':self.q_y, 'q_x':self.q_x, 'q':self.q})
            else:
                ddm_dataset=xr.Dataset({'ddm_matrix':(['lagtime','time', 'q'], radav_ddmmat), 
                                        'alignment_factor':(['lagtime','time','q'], AF)},
                                       coords={'time': times,
                                               'lagtime': lagtime,
                                               'q':self.q})
            
            ddm_dataset.attrs['AlignmentFactorAxis'] = af_axis
        
        return ddm_dataset
    
    
    def phiDM(self, lagt, halfsize, use_gf=True, gfsize=3, err_limit = 2e-5):
        r'''
        
        For more info, see Colin, R., Zhang, R. & Wilson, L. G. 
        Fast, high-throughput measurement of 
        collective behaviour in a bacterial population. 
        Journal of The Royal Society Interface 
        11, 20140486 (2014).
        https://royalsocietypublishing.org/doi/10.1098/rsif.2014.0486

        Parameters
        ----------
        lagt : int
            Lag time (in units of frames).
        halfsize : int
            Centered above zero frequency, size of region to fit to plane.
        use_gf : boolean, optional
            Use Gaussian filter or not. The default is True.
        gfsize : int, optional
            Size of Gaussian filter to apply. The default is 3.
        err_limit : float, optional
            If error between data and plane that is fit is greater than this value 
            per pixel, then won't be used to get avg velocity

        Returns
        -------
        phiDM_dataset : xarray Dataset
            Dataset contains the phase of the Fourier tranform of each image,
            the velocity (v_x and v_y), and error between the fit of plane and 
            the difference in phase.

        '''
        if type(self.im)==list:
            '''
            self.q_y=np.sort(np.fft.fftfreq(self.im[0].shape[1], d=self.pixel_size))*2*np.pi
            self.q_x=self.q_y
            self.q=np.arange(0,self.im[0].shape[1]/2)*2*np.pi*(1./(self.im[0].shape[1]*self.pixel_size))
            '''
            print("Not yet implemented for series of movies. Just use a single movie.")
            return None
        else:
            self.q_y=np.sort(np.fft.fftfreq(self.im.shape[1], d=self.pixel_size))*2*np.pi
            self.q_x=self.q_y
            self.q=np.arange(0,self.im.shape[1]/2)*2*np.pi*(1./(self.im.shape[1]*self.pixel_size))
            
        times = np.arange(self.im.shape[0]) / self.frame_rate
        
        phase = ddm.getPhase_phiDM(self.im, use_gf=use_gf, gfsize=gfsize)
        vx,vy,er = ddm.getVel_phiDM(phase, lagt, self.pixel_size, 
                                    self.frame_rate, halfsize=halfsize)
        vtimes = np.arange(len(vx)) / self.frame_rate
        w = np.where(er < err_limit)
        vx_mean = np.mean(vx[w])
        vy_mean = np.mean(vy[w])
        vx_median = np.median(vx[w])
        vy_median = np.median(vy[w])
        vx_std = np.std(vx[w])
        vy_std = np.std(vy[w])
        vx_stderr = vx_std / np.sqrt(len(w[0]))
        vy_stderr = vy_std / np.sqrt(len(w[0]))
        
        skew_vx = np.mean(((vx[w] - vx_mean)**3)/ vx_std**3)
        skew_vy = np.mean(((vy[w] - vy_mean)**3)/ vy_std**3)
        
        phiDM_dataset = xr.Dataset({'phase':(['time', 'q_y','q_x'], phase),
                                    'vx': (['vtime'], vx),
                                    'vy': (['vtime'], vy),
                                    'error': (['vtime'], er)},
                               coords={'time': times,
                                       'vtime': vtimes,
                                       'q_y':self.q_y, 'q_x':self.q_x})
        phiDM_dataset.attrs['lagtime'] = lagt
        phiDM_dataset.attrs['halfsize'] = halfsize
        phiDM_dataset.attrs['err_limit'] = err_limit
        phiDM_dataset.attrs['use_gf'] = int(use_gf)
        phiDM_dataset.attrs['gfsize'] = gfsize
        phiDM_dataset.attrs['vx_mean'] = vx_mean
        phiDM_dataset.attrs['vy_mean'] = vy_mean
        phiDM_dataset.attrs['vx_median'] = vx_median
        phiDM_dataset.attrs['vy_median'] = vy_median
        phiDM_dataset.attrs['vx_std'] = vx_std
        phiDM_dataset.attrs['vy_std'] = vy_std
        phiDM_dataset.attrs['vx_stderr'] = vx_stderr
        phiDM_dataset.attrs['vy_stderr'] = vy_stderr
        phiDM_dataset.attrs['vx_skew'] = skew_vx
        phiDM_dataset.attrs['vy_skew'] = skew_vy
        
        if 'binning' in self.analysis_parameters:
            phiDM_dataset.attrs['binning'] = int(self.analysis_parameters['binning'])
        else:
            phiDM_dataset.attrs['binning'] = 0
        if phiDM_dataset.attrs['binning']:
            phiDM_dataset.attrs['bin_size'] = self.binsize
        else:
            phiDM_dataset.attrs['bin_size'] = 1
        phiDM_dataset.attrs['filename'] = self.filename
        phiDM_dataset.attrs['data_dir'] = self.data_dir
        try:
            self.channel
        except NameError:
            pass
        else:
            if self.channel is not None:
                phiDM_dataset.attrs['channel'] = self.channel
        phiDM_dataset.attrs['pixel_size'] = self.pixel_size
        phiDM_dataset.attrs['frame_rate'] = self.frame_rate
        
        return phiDM_dataset
    
    
    
    def createTwoTimeCorr(self, ddm_var_dataset, qindex):
        r"""Create two-time correlation matrix
    
        After generating the DDM matrix as a function of time lag as well as 
        of time, can use this function to generate a 2D two-time correlation
        function for a particular wavenumber. 
        
    
        Parameters
        ----------
        ddm_variability : xarray dataset
            Results of function 'variationInDDMMatrix' in 'ddm_analysis_and_fitting' code
        q_index : int
            Index of the wavenumber array
    
        Returns
        -------
        twotimecorrelation : array
            Two time correlation function
    
        """
        number_of_frames = self.im.shape[0]
        if qindex >= len(self.q):
            print("qindex must be less than %i." % len(self.q))
            return None
        twotimecorr = hf.create_two_time_correlation_matrix(ddm_var_dataset,
                                                            number_of_frames,
                                                            qindex)
        return twotimecorr
        
    
    def find_alignment_factor_one_lagtime(self, ddmmatrix2d, orientation_axis=0, 
                                          remove_vert_line=True, remove_hor_line=True):
        r"""
        

        Parameters
        ----------
        orientation_axis : TYPE, optional
            DESCRIPTION. The default is np.pi/4.

        Returns
        -------
        None.

        """
        ddm_matrix_at_lagtime = ddmmatrix2d.copy()
        nx,ny = ddm_matrix_at_lagtime.shape
        if remove_vert_line:
            ddm_matrix_at_lagtime[:,int(ny/2)]=0
        if remove_hor_line:
            ddm_matrix_at_lagtime[int(nx/2),:]=0
        x = np.arange(-1*ny/2, ny/2, 1)
        y = np.arange(-1*nx/2, nx/2, 1)
        xx,yy = np.meshgrid(x,y)
        with np.errstate(divide='ignore', invalid='ignore'):
            cos2theta = np.cos(2*np.arctan(1.0*xx/yy) + orientation_axis)
        cos2theta[int(nx/2),int(ny/2)]=0
        
        dists = np.sqrt(np.arange(-1*nx/2, nx/2)[:,None]**2 + np.arange(-1*ny/2, ny/2)[None,:]**2)
    
        bins = np.arange(max(nx,ny)/2+1)
        histo_of_bins = np.histogram(dists, bins)[0]
        
        af_numerator = np.histogram(dists, bins, weights=ddm_matrix_at_lagtime*cos2theta)[0]
        af_denominator = np.histogram(dists, bins, weights=ddm_matrix_at_lagtime)[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            af = af_numerator / af_denominator
        return af
    
    
        
    def find_alignment_factor(self, ddmmatrix3d, orientation_axis=0, 
                              remove_vert_line=True, remove_hor_line=True):
        r"""
        

        Parameters
        ----------
        orientation_axis : TYPE, optional
            DESCRIPTION. The default is np.pi/4.

        Returns
        -------
        None.

        """
        AF = self.find_alignment_factor_one_lagtime(ddmmatrix3d[0], orientation_axis=orientation_axis,
                                                    remove_vert_line=remove_vert_line,
                                                    remove_hor_line=remove_hor_line)
        af_size = len(AF)
        num_lag_times = ddmmatrix3d.shape[0]
        all_af = np.zeros((num_lag_times, af_size))
        all_af[0] = AF
        for i in range(1,num_lag_times):
            all_af[i] = self.find_alignment_factor_one_lagtime(ddmmatrix3d[i], orientation_axis=orientation_axis,
                                                               remove_vert_line=remove_vert_line,
                                                               remove_hor_line=remove_hor_line)
        return all_af, orientation_axis
        
    
    
    def resave_ddm_dataset(self, ddmdataset, file_name = None):
        r"""
        Resave the DDM dataset. 

        Parameters
        ----------
        ddmdataset : xarray dataset
            DDM dataset
        file_name : string or None, optional
            Filename for saving data. The default is None. If None,
            then will use what is set in the attribute 'filename_for_saving_data'. 
            Note that '_ddmmatrix.nc' is appended to the end of the filename.


        """
        if file_name is None:
            file_name = f"{self.data_dir}{self.filename_for_saving_data}"
        try:
            ddmdataset.to_netcdf(f"{file_name}_ddmmatrix.nc", mode='w')
        except:
            print(f'Could not save data set as: {file_name}_ddmmatrix.nc')
            name_alternative=input("Provide an alternative name (do not include .nc suffix) [MUST SPECIFY FULL PATH!]: ")
            ddmdataset.to_netcdf(f"{name_alternative}.nc", mode='w')
            print("note this name should be entered in the yaml file for fitting")
        ddmdataset.close()


    def generate_plots(self, ddmdataset, pdf_to_save_to=None, q_to_see=1.5, num=None):
        '''
        Generates plot, which can be saved as PDF, to summarize the DDM matrix calulations. The region of interest is displayed;
        a Fourier transform of a difference image; a graph of radial averages of FFT of frames, used to determine amplitude and background
        and a plot of the intermediate scattering function at one given q-value.

        :param ddmdataset: DDM Dataset
        :type ddmdataset: Dataset
        :param pdf_to_save_to:
        :type pdf_to_save_to: TYPE, optional
        :param q_to_see: The magnitude of the wavector for which to plot the ISF units: μm$^{-1}$
        :type q_to_see: float
        :param num: The number of the ROI in case multiple ROIs have been analyzed
        :type num: int, optional

        q_to_see: float, optional
            q value to look at ISF


        '''
        plt.ion()
        plt.matshow(self.image_for_report, cmap=matplotlib.cm.gray)
        if 'crop_to_roi' in self.analysis_parameters:
            if self.analysis_parameters['crop_to_roi'] is not None:
                if len(self.analysis_parameters['crop_to_roi'])==4:
                    ax=plt.gca()

                    x1 = self.analysis_parameters['crop_to_roi'][0]
                    y1 = self.analysis_parameters['crop_to_roi'][2]
                    xsize = self.analysis_parameters['crop_to_roi'][1] - self.analysis_parameters['crop_to_roi'][0]
                    ysize = self.analysis_parameters['crop_to_roi'][3] - self.analysis_parameters['crop_to_roi'][2]
                    rect = Rectangle((y1,x1),xsize,ysize,linewidth=2,edgecolor='r',facecolor='none')

                    ax.add_patch(rect)
        if 'split_into_4_rois' in self.analysis_parameters:
            if self.analysis_parameters['split_into_4_rois'] and (type(self.im)==list):
                if num==None:
                    num=0
                ax=plt.gca()
                xsize,ysize = self.im[num][0].shape
                if num==0:
                    x1 = y1 = 0 #ROI "0" is top left
                if num==1:
                    x1,y1 = 0,ysize
                if num==2:
                    x1,y1 = xsize,0
                if num==3:
                    x1,y1 = xsize,ysize
                rect = Rectangle((x1,y1),xsize,ysize,linewidth=2,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
        plt.title(f"{self.filename_for_saving_data} \n Displayed is frame #1; time of {self.first_frame/self.frame_rate:.2f} sec", fontsize=9)
        plt.xlabel(f'1 pixel = {self.pixel_size} μm. Image will *not* reflect any applied windowing.', fontsize=8.5)
        if pdf_to_save_to != None:
            pdf_to_save_to.savefig()

        dt_to_show = 5
        if 'ddm_matrix_full' in ddmdataset.data_vars:
            ddm_mat_to_show = ddmdataset.ddm_matrix_full[dt_to_show]
        else:
            ddm_mat_to_show = ddmdataset.ddm_matrix[dt_to_show]
        plt.matshow(ddm_mat_to_show, cmap=matplotlib.cm.gray)
        plt.title(f"{self.filename_for_saving_data} \n DDM matrix for lag time of {self.lag_times[dt_to_show]:.2f} sec", fontsize=9)
        if pdf_to_save_to != None:
            pdf_to_save_to.savefig()

        ##Plot graph of rav FFT of frames, used to determine A  and B
        fig2=plt.figure(figsize=(6, 6/1.2))
        plt.semilogy(ddmdataset.coords['q'][3:], ddmdataset.avg_image_ft[3:],'ro')
        plt.xlabel("q μm$^{-1}$")
        plt.ylabel(r"$\left< | \tilde{I}(q, t) |^2 \right>_t$")
        plt.hlines(0.5*ddmdataset.B.values, ddmdataset.coords['q'][3], ddmdataset.coords['q'][-1], linestyles='dashed')
        plt.title(f"Used to estimate background (B) and amplitude (A). Dashed line at {0.5*ddmdataset.B.values:.2f}. \n Background ~ {ddmdataset.B.values:.0f}", fontsize=10)
        if pdf_to_save_to != None:
            pdf_to_save_to.savefig()

        #Plot ISF versus lagtime for one q-value
        fig3=plt.figure(figsize=(6, 6/1.2))
        isf_at_q=ddmdataset.ISF.sel(q=q_to_see, method='nearest')
        plt.semilogx(ddmdataset.coords['lagtime'], isf_at_q, 'o')
        plt.xlabel("lag time (s)")
        plt.ylabel("ISF")
        plt.title(f"ISF for q = {q_to_see:.2f} μm$^{-1}$")
        if pdf_to_save_to != None:
            pdf_to_save_to.savefig()

        if (self.central_angle != None) and (self.angle_range != None):

            if type(self.im)==list:
                im_0 = self.im[0][0]
            else:
                im_0 = self.im[0]
            mask = ddm.generate_mask(im_0, self.central_angle, self.angle_range)
            fig4=plt.figure(figsize=(6,6/1.2))
            plt.matshow(mask, fignum=False)
            plt.title("Applied mask to the DDM matrix")
            if pdf_to_save_to != None:
                pdf_to_save_to.savefig()


class DDM_Fit:
    """
    Set of functions to fit DDM matrix (image structure function) or ISF, the user can choose from a 
    variety of mathematical models :py:mod:`PyDDM.ISF_and_DDMmatrix_theoretical_models`. 
    Analysis paramters can be provided in a YAML file :doc:`More information here </Provide_info_for_analysis>`.
    Graphs of the fits are produced and can be saved as PDF.

    """

    def __init__(self, data_yaml, subimage_num=None, silent=False):
        self.subimage_num = subimage_num
        self.silent = silent
        self.model_dict = None
        self.data_yaml = data_yaml
        self.loadYAML()
        self.fittings = {} #Stores fit_model,parameters and data set, key-value provided by user
        display_table = not silent
        self.use_parameters_provided(display_params_table = display_table)
        self.load_data()
        
        
    def __str__(self):
        
        if isinstance(self.data_yaml, str):
            return f"""
            DDM Fit:
                analysis parameters: {self.data_yaml}
                data directory: {self.data_dir}
                file name: {self.filename}
                fitting model: {self.fit_model}"""
        else:
            return f"""
            DDM Fit:
                data directory: {self.data_dir}
                file name: {self.filename}
                fitting model: {self.fit_model}"""


    def loadYAML(self):
        r"""Loads information from .yml or .yaml file if file follows specified format
        
            The provided yaml (https://yaml.org/) file must contain the keys:
                * DataDirectory
                * FileName
                * Fitting_parameters

        """
        
        if isinstance(self.data_yaml, str):
            doesYAMLFileExist = os.path.exists(self.data_yaml)
            if doesYAMLFileExist:
                with open(self.data_yaml) as f:
                    self.content = yaml.safe_load(f)
            else:
                print("File %s does not exist. Check file name or path." % self.data_yaml)
                return 0
        elif isinstance(self.data_yaml, dict):
            self.content = self.data_yaml.copy()
            
        self.fit_options = self.content['Fitting_parameters']
        self.fit_model = self.fit_options['model']
        if self.fit_model not in fpd.fitting_models:
            print("Model not found! Here are list of possible models:")
            fpd.return_possible_fitting_models()
            return 0
        else:
            self.model_dict = copy.deepcopy(fpd.fitting_models[self.fit_model])


        self.data_dir = self.content['DataDirectory']
        self.filename = self.content['FileName']
        #Get file name without .extension for saving data
        self.filename_noext = self.filename[:-4]
        if self.subimage_num != None:
            self.filename_noext = f"{self.filename[:-4]}_{self.subimage_num:02}"
        elif 'split_into_4_rois' in self.content['Analysis_parameters']:
            if self.content['Analysis_parameters']['split_into_4_rois']:
                print("Images were split into 4 ROIs. However, ROI number (0 to 3) not specified.")
                answer = int(input("If you want to specify ROI #, enter now. If not, enter -1: "))
                if answer>-1:
                    self.subimage_num = answer
                    self.filename_noext = f"{self.filename[:-4]}_{self.subimage_num:02}"
        return 1

    def reload_fit_model_by_name(self, model_name, update_params=True):
        """Update the current fitting model, by providing the model named. If 
        invalid model name is provided a list with available options will be provided.


        :param model_name: Name of the model for fitting
        :type model_name: str

        :param update_params: If True updates fitting parameters according to provided parameters
        :type update_params: bool
        """
        if model_name not in fpd.fitting_models:
            print("Model not found! Here are list of possible models:")
            fpd.return_possible_fitting_models()
        else:
            self.fit_model = model_name
            self.model_dict = copy.deepcopy(fpd.fitting_models[self.fit_model])
        if update_params:
            self.use_parameters_provided()
            


    def use_parameters_provided(self, print_par_names=False, display_params_table=True):
        """Gives the parameters and their values, as provided in the initiation file (YAML file)

        :param print_par_names: Default is False
        :type print_par_names: bool

        """
        params_for_model = fpd.return_parameter_names(self.model_dict, print_par_names = print_par_names)
        for p in params_for_model:
            if p not in self.fit_options:
                print(f"Need to specify {p}.")
            elif len(self.fit_options[p])!=3:
                print(f"Length of list for parameter {p} must be 3 (initial guess, minimum, maximum).")
            else:
                fpd.set_parameter_guess_and_limits(self.model_dict,
                                                   p, self.fit_options[p])
        if display_params_table:
            fpd.turn_parameters_into_dataframe_for_display(self.model_dict['parameter_info'])

    def set_parameter_initial_guess(self, param_name, value):
        """Set the intitial value of parameter.

        :param param_name: Name of the parameter for which the intial guess should be set
        :type param_name: str
        :param value: Value for the parameter for which the intial guess should be set
        :type value: int or float

        """
        if self.model_dict != None:
            fpd.set_parameter_initial_guess(self.model_dict, param_name, value)
        else:
            print("Model not yet set.")

    def set_parameter_fixed(self, param_name, fix):
        if self.model_dict != None:
            fpd.set_parameter_fixed(self.model_dict, param_name, fix)
        else:
            print("Model not yet set.")

    def set_parameter_bounds(self, param_name, bounds):
        """Restrict the value of the parameter for the fitting by specifying bounds.

        :param param_name: Name of the parameter for which the intial guess should be set
        :type param_name: str
        :param bounds: Value for the parameter for which the intial guess should be set
        :type bounds: List[float]

        """
        if len(bounds)!=2:
            print("Must pass a list of two numbers for bounds -- lower and upper.")
        else:
            if self.model_dict != None:
                fpd.set_parameter_limits(self.model_dict, param_name, bounds)
            else:
                print("Model not yet set.")


    def load_data(self, data=None):
        """Loads data for fitting whether it is in memory or saved to disk.
        
        Parameters
        ----------
        data : xarray dataset
            Default is None, in which case it will look for saved file containing 
            the DDM dataset based on the filename and data directory. Otherwise, 
            one can pass the filename to the xarray dataset or the dataset itself. 

        """

        if type(data)==str:
            self._load_data_from_file(filename=data)
        elif (type(data)==xr.core.dataset.Dataset):
            self.ddm_dataset = data
        elif data==None:
            self._load_data_from_file()
        else:
            print("Optional argument must be filepath or xarray Dataset.")


    def _load_data_from_file(self, filename=None):
        #Open data set from disk
        if filename is not None:
            if os.path.exists(filename):
                print(f"Loading file {filename} ...")
                with xr.open_dataset(filename) as ds:
                    self.ddm_dataset = ds.load()
                    self.ddm_dataset.close()
            else:
                print(f"The file {filename} not found. Check for typos.")
        else:
            #file_name = re.search("\w+", self.filename).group()
            if 'filename_for_saved_data' in self.content['Analysis_parameters']:
                file_name = self.content['Analysis_parameters']['filename_for_saved_data']
                dataset_filename = f"{self.data_dir}{file_name}_ddmmatrix.nc"
            else:
                file_name = self.filename_noext
                dataset_filename = f"{self.data_dir}{file_name}_ddmmatrix.nc"
            if os.path.exists(dataset_filename):
                print(f"Loading file {dataset_filename} ...")
                with xr.open_dataset(dataset_filename) as ds:
                    self.ddm_dataset = ds.load()
                    self.ddm_dataset.close()
            else:
                print(f"File {dataset_filename} not found.")
                fls = glob.glob(f"{self.data_dir}{file_name}*.nc")
                if len(fls)>0:
                    print("Perhaps you meant to load one of...")
                    for flnm in fls:
                        print(f"\t{flnm}")
                    print(f"By default, loading {fls[0]}")
                    with xr.open_dataset(fls[0]) as ds:
                        self.ddm_dataset = ds.load()
                        self.ddm_dataset.close()
                    self.filename_noext = fls[0].split('\\')[-1][:-13]



    def fit(self, quiet=True, save=True, name_fit=None,
            use_lsqr_cf = [False,True],
            update_tau_based_on_estimated_diffcoeff = False,
            estimated_diffcoeff = None,
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
            debug=False,
            display_table=True):

        """
        Fits the DDM data

        :param save: save the fit in dictionary, called 'fittings', fit results are accescible fittings[name_fit]['fit'].
        :type save: bool
        :param name_fit: Name to save the fit results under in the fittings dictionary
        :type name_fit: None or str
        :param use_lsqr_cf: ???
        :type use_lsqr_cf: List[bool]
        :param update_tau_based_on_estimated_diffcoeff: Use an estimation of diffusion coeffcient to set initial value of tau (delay time) for fitting procedure
        :type update_tau_based_on_estimated_diffcoeff: bool
        :param estimated_diffcoeff: Estimated value of the diffusion constant (μm^2/s)
        :type estimated_diffcoeff: float
        :param update_tau_based_on_estimated_velocity: Use an estimation of velocity to set bounds to set initial value of tau (delay time) for fitting procedure
        :type update_tau_based_on_estimated_velocity: float
        :param estimated_velocity: Estimated value of the velocity (μm/s)
        :type estimated_velocity: float
        :param update_tau2_based_on_estimated_diffcoeff: Use an estimation of diffusion coeffcient to set initial value of the seconds tau for fitting procedure
        :type update_tau2_based_on_estimated_diffcoeff: bool
        :param estimated_diffcoeff2: Estimated value of the diffusion constant for second tau (μm^2/s)
        :type estimated_diffcoeff2: float
        :param update_tau2_based_on_estimated_velocity: Use an estimation of velocity to set bounds to set initial value of second tau for fitting procedure
        :type update_tau2_based_on_estimated_velocity: bool
        :param estimated_velocity2: Estimated value of the velocity (μm/s)
        :type estimated_velocity2: float
        :param update_limits_on_tau: Set bounds on tau based on the given estimation of the diffusion coefficient or velocity
        :type update_limits_on_tau: bool
        :param updated_lims_on_tau_fraction: Limits are set by adding and subtracting the given fraction of the estimated initial tau value, default is 0.1
        :type updated_lims_on_tau_fraction: float
        :param use_A_from_images_as_guess: If True the amplitude value, determined from the direct Fourier transform of the frames/images, is used as initial guess
        :type use_A_from_images_as_guess: bool
        :param update_limits_on_A: Set bounds on the amplitude based on the given estimation
        :type update_limits_on_A: bool
        :param updated_lims_on_A_fraction: Fraction that determines to what extend the amplitude is bounded during the fitting procedure, default is 0.1
        :type updated_lims_on_A_fraction: float
        :param display_table: Print table with fitted values
        :type display_table: bool

        """
        
        #get the radially averaged ddm matrix data 
        # note that an older version called this 'ravs'
        # but newer versiosn call this 'ddm_matrix'
        if 'ravs' in self.ddm_dataset.data_vars:
            ddm_matrix_data = self.ddm_dataset.ravs
        else:
            ddm_matrix_data = self.ddm_dataset.ddm_matrix

        #Either the DDM matrix or the ISF will be fit. If fitting the DDM 
        #matrix then the parameters A and B (amplitude and background) will 
        #be fit parameters. If fitting the ISF, A and B will be determined 
        #before doing the fit
        if self.model_dict['data_to_use'] == 'ISF':
            data_to_fit = self.ddm_dataset.ISF
        elif self.model_dict['data_to_use'] == 'DDM Matrix':
            data_to_fit = ddm_matrix_data

        if use_lsqr_cf[1]:
            sigma = 1./np.sqrt(self.ddm_dataset.num_pairs_per_dt)
        else:
            sigma = None

        best_fits, theories = ddm.fit_ddm_all_qs(data_to_fit, self.ddm_dataset.lagtime,
                                                 copy.deepcopy(self.model_dict),
                                                 self.ddm_dataset.Amplitude.values,
                                                 quiet=quiet,
                                                 first_use_leastsq = use_lsqr_cf[0],
                                                 use_curvefit_method = use_lsqr_cf[1],
                                                 sigma = sigma,
                                                 update_tau_based_on_estimated_diffcoeff = update_tau_based_on_estimated_diffcoeff,
                                                 estimated_diffcoeff = estimated_diffcoeff,
                                                 update_tau_based_on_estimated_velocity=update_tau_based_on_estimated_velocity,
                                                 estimated_velocity=estimated_velocity,
                                                 update_tau2_based_on_estimated_diffcoeff=update_tau2_based_on_estimated_diffcoeff,
                                                 estimated_diffcoeff2=estimated_diffcoeff2,
                                                 update_tau2_based_on_estimated_velocity=update_tau2_based_on_estimated_velocity,
                                                 estimated_velocity2=estimated_velocity2,
                                                 update_limits_on_tau=update_limits_on_tau,
                                                 updated_lims_on_tau_fraction=updated_lims_on_tau_fraction,
                                                 use_A_from_images_as_guess=use_A_from_images_as_guess,
                                                 update_limits_on_A=update_limits_on_A,
                                                 updated_lims_on_A_fraction=updated_lims_on_A_fraction,
                                                 debug=debug)

        bestfit_dataarray = xr.DataArray(data = [*best_fits.values()],
                                                 dims = ["parameter", "q"],
                                                 coords = [[*best_fits], self.ddm_dataset.q],
                                                 name = 'FitParameters')

        theory_dataarray = xr.DataArray(data = theories,
                                        dims = ["lagtime", "q"],
                                        coords = [self.ddm_dataset.lagtime, self.ddm_dataset.q],
                                        name = 'Theory')

        fit_results = xr.Dataset(dict(parameters=bestfit_dataarray, theory=theory_dataarray,
                                      isf_data=self.ddm_dataset.ISF,
                                      ddm_matrix_data=ddm_matrix_data,
                                      A=self.ddm_dataset.Amplitude,
                                      B=self.ddm_dataset.B))
        fit_results.attrs['model'] = self.fit_model
        fit_results.attrs['data_to_use'] = self.model_dict['data_to_use']
        
        #saving the initial parameters. but can't save xarray DataSet in netcdf format if
        #  we have a list of dictionaries as attribute. So, we convert the dictionary
        #  to a sring. 
        init_params = copy.deepcopy(self.model_dict['parameter_info'])
        init_params_as_string = []
        for thing in init_params:
            init_params_as_string.append(str(thing))
        fit_results.attrs['initial_params_dict'] = init_params_as_string 

        if 'Good_q_range' in self.content['Fitting_parameters']:
            force_q_range = self.content['Fitting_parameters']['Good_q_range']
        else:
            force_q_range = None
        if 'Auto_update_good_q_range' in self.content['Fitting_parameters']:
            if self.content['Fitting_parameters']['Auto_update_good_q_range']:
                update_good_q_range = True
            else:
                update_good_q_range = False
        else:
            update_good_q_range = True

        qrange, slope, d_eff, msd_alpha, msd_d_eff, d, d_std, v, v_std = get_tau_vs_q_fit(fit_results, forced_qs=force_q_range, 
                                                                                          update_good_q_range=update_good_q_range,
                                                                                          silent=self.silent)
        fit_results.attrs['effective_diffusion_coeff'] = d_eff
        fit_results.attrs['tau_vs_q_slope'] = slope
        fit_results.attrs['msd_alpha'] = msd_alpha
        fit_results.attrs['msd_effective_diffusion_coeff'] = msd_d_eff
        fit_results.attrs['diffusion_coeff'] = d
        fit_results.attrs['diffusion_coeff_std'] = d_std
        fit_results.attrs['velocity'] = v
        fit_results.attrs['velocity_std'] = v_std
        fit_results.attrs['good_q_range'] = qrange

        if ('Tau2' in fit_results.parameters.parameter):
            qrange, slope, d_eff, msd_alpha, msd_d_eff, d, d_std, v, v_std = get_tau_vs_q_fit(fit_results, use_tau2=True, 
                                                                                              forced_qs=force_q_range, update_good_q_range=update_good_q_range,
                                                                                              silent=self.silent)
            fit_results.attrs['tau2_effective_diffusion_coeff'] = d_eff
            fit_results.attrs['tau2_tau_vs_q_slope'] = slope
            fit_results.attrs['tau2_msd_alpha'] = msd_alpha
            fit_results.attrs['tau2_msd_effective_diffusion_coeff'] = msd_d_eff
            fit_results.attrs['tau2_diffusion_coeff'] = d
            fit_results.attrs['tau2_diffusion_coeff_std'] = d_std
            fit_results.attrs['tau2_velocity'] = v
            fit_results.attrs['tau2_velocity_std'] = v_std
            fit_results.attrs['tau2_good_q_range'] = qrange

        if save:
            name = self._save_fit(fit_results, name_fit = name_fit)
            if not self.silent:
                print(f"Fit is saved in fittings dictionary with key '{name}'.")

        if display_table:
            pd = hf.generate_pandas_table_fit_results(fit_results)
            display(pd)
            
        #Placing other metadata in the fit_results Dataset
        fit_results.attrs['DataDirectory'] = self.ddm_dataset.DataDirectory
        fit_results.attrs['FileName'] = self.ddm_dataset.FileName
        fit_results.attrs['pixel_size'] = self.ddm_dataset.pixel_size
        fit_results.attrs['frame_rate'] = self.ddm_dataset.frame_rate
        if "BackgroundMethod" in self.ddm_dataset.attrs:
            fit_results.attrs['BackgroundMethod'] = self.ddm_dataset.BackgroundMethod
        if "OverlapMethod" in self.ddm_dataset.attrs:
            fit_results.attrs['OverlapMethod'] = self.ddm_dataset.OverlapMethod

        return fit_results


    def _save_fit(self, fit_results, name_fit=None):

        #Save the fit and parameter settings to a dictionary with a key value provided by the user

        if name_fit == None:
            #name_fit=input("Under what name would you want to save the generated fit in the dictionary 'fittings': ")
            num_fits_so_far = len(self.fittings.keys())
            name_fit = "fit%02i" % (num_fits_so_far+1)

        if name_fit != None:
            self.fittings[name_fit]={'model':copy.deepcopy(self.fit_model),
                                     'settings':copy.deepcopy(self.model_dict['parameter_info'])}
            self.fittings[name_fit]['fit']=fit_results
            
        return name_fit




    def generate_fit_report(self, fit_results=None, PDF_save=True, forced_qs=None,
                            forced_qs_for_tau2=None, q_indices=[10,20,30,40], use_new_tau=True,
                            fit_report_name_end=None, show=True):
        r"""Generates fit report
        
        See :py:func:`PyDDM.ddm_analysis_and_fitting.fit_report`
        
        This method calls that function. For this method, you can refer to `fit_results` by 
        the key in the `fittings` dictionary of the :py:class:`PyDDM.ddm_analysis_and_fitting.DDM_Fit` class. 
        When calling the :py:func:`PyDDM.ddm_analysis_and_fitting.fit_report` function, 
        the filename of the image data and the image data directory will be used 
        when saving the PDF report. 
        
        Use the :py:func:`PyDDM.ddm_analysis_and_fitting.fit_report` if you 
        want the report saved somewhere not in the DataDirectory indicated in the 
        yaml data which initialized the DDM_Fit class. 
        

        Parameters
        ----------
        fit_results : xarray Dataset, str, or None
            Results of the DDM fitting to either the DDM matrix or the ISF. 
        PDF_save : bool, optional
            Saves plots to a PDF file. The default is True.
        forced_qs : list or None, optional
            Does power law fitting over this range of q. The default is None. If not 
            None, must be list-like with two numbers indicating the index of the 
            lower and upper q. 
        forced_qs_for_tau2 : list or None, optional
            Like `forced_qs` but for the second decay time
        q_indices : list, optional
            List of q indices to show when plotting the DDM matrix or ISF vs lag time 
            along with the theoretical model. The default is [10,20,30,40].
        use_new_tau : bool, optional
            If True, calculates a more representative decay time based on the value of 
            tau and the stretching exponent. The default is True.
        fit_report_name_end : str, optional
            String appended to the filename of the PDF generated. The default is None. 
        show : bool, optional
            Plots will appear if True. The default is True.
    
        Returns
        -------
        fit_results : xarray Dataset
            Results of DDM fitting with attributes possibly changed

        """

        if fit_report_name_end is None:
            pdf_report_filename = f'{self.filename_noext}'
        else:
            pdf_report_filename = f'{self.filename_noext}_{fit_report_name_end}'

        if show:
            plt.ion()
        else:
            plt.ioff()

        if isinstance(fit_results, str):
            if fit_results in self.fittings:
                fit_results = self.fittings[fit_results]['fit']
            else:
                print("Not in the list of keys: ", list(self.fittings.keys()))
                return
        elif fit_results is None:
            fit_keys = list(self.fittings)
            fit_results = self.fittings[fit_keys[-1]]['fit'] #gets the latest fit
            print(f"Using fit stored with key '{fit_keys[-1]}'.")
        elif isinstance(fit_results, xr.core.dataset.Dataset):
            pass
        else:
            print("Not a valid parameter passed to fit_results.")
            return
        
        new_fit_res = fit_report(fit_results, PDF_save=PDF_save, forced_qs=forced_qs, pdf_save_dir = self.data_dir, 
                                 forced_qs_for_tau2=forced_qs_for_tau2, q_indices=q_indices, use_new_tau=use_new_tau, 
                                 fit_report_name=pdf_report_filename, show=show)
        return new_fit_res

    def extract_MSD(self, fit=None, qrange=None):
        r"""
        

        Parameters
        ----------
        fit : TYPE, optional
            DESCRIPTION. The default is None.
        qrange : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        msd : TYPE
            DESCRIPTION.
        msd_std : TYPE
            DESCRIPTION.

        """
        if fit == None:
            fit_keys = list(self.fittings)
            fit = self.fittings[fit_keys[-1]]['fit'] #gets the latest fit
        if qrange == None:
            qrange = fit.good_q_range
        if 'Amplitude' in fit.parameter:
            amp = fit.parameters.loc['Amplitude']
        else:
            try:
                amp = fit.A
            except:
                amp = self.ddm_dataset.Amplitude
        if 'Background' in fit.parameter:
            bg = fit.parameters.loc['Background']
        else:
            try:
                bg = fit.B
            except:
                bg = self.ddm_dataset.B
        try:
            ddm_matrix_data = fit.ddm_matrix_data
        except:
            ddm_matrix_data = self.ddm_dataset.ddm_matrix
            
        msd, msd_std = ddm.get_MSD_from_DDM_data(fit.q, amp, ddm_matrix_data,
                                                 bg, qrange)
        try:
            msd = msd.drop('parameter')
            msd_std = msd_std.drop('parameter')
        except:
            pass
        fit['msd'] = msd
        fit['msd_std'] = msd_std
        return msd, msd_std


    def rheo_Mods(self, a, fit = None, tau = None, msd = None, 
                  width = 0.7, dim = 3, T=290, clip = 0.03):
        
        #ToDo: incorporate this data into fit variable/YAML file
        
        if fit == None:
            fit_keys = list(self.fittings)
            fit = self.fittings[fit_keys[-1]]['fit'] #gets the latest fit
        
        
        if (tau == None):
            try:
                fit_msd = fit['msd']
                
            except:
                fit_msd = extract_MSD()
        
            tau = np.array(fit_msd.lagtime)
        
        if (msd == None):
            try:
                fit_msd = fit['msd']
                
            except:
                fit_msd = extract_MSD()
        
            msd = np.array(fit_msd)
            
            
        omega, Gs, Gp, Gpp = ddm.micrheo(tau, msd, a, width = 0.7, dim = 3, T=290, clip = 0.03)
        
        return omega, Gs, Gp, Gpp
    
    
    

    def error_in_fit(self, fit=None, q_index=10, show_plot=True,
                     show_error_vs_q=False, use_isf=True):
        if (type(fit)!=xr.Dataset) and (type(fit)!=list) and (fit is None):
            fit_keys = list(self.fittings)
            fit = self.fittings[fit_keys[-1]]['fit'] #gets the latest fit
            print(f"Using fit stored with key '{fit_keys[-1]}'.")
        if show_plot:
            fig,ax = plt.subplots(figsize=(10,10/1.618))
        if show_error_vs_q:
            fig2,ax2 = plt.subplots(figsize=(10,10/1.618))
        if (type(fit)==list):
            colors = ['r','b','g','m']
            for i,f in enumerate(fit):
                e,m = self._error_in_one_fit(fit=f, q_index=q_index, axes=ax, show_plot=show_plot, label="%s"%(i+1),
                                             plot_color=colors[i%len(colors)], use_isf = use_isf)
                if show_error_vs_q:
                    ax2.loglog(f.q, m, 'o', color=colors[i%len(colors)], label="%s"%(i+1))
            if show_plot:
                ax.legend()
            if show_error_vs_q:
                ax2.legend()
        else:
            e,m = self._error_in_one_fit(fit=fit, q_index=q_index, axes=ax, show_plot=show_plot)
            if show_error_vs_q:
                ax2.loglog(fit.q, m, 'o', color='r')
        if show_error_vs_q:
            ax2.set_xlabel("q (μm$^{-1}$)")
            ax2.set_ylabel("Mean squared error")


    def _error_in_one_fit(self, fit=None, q_index=10, axes=None, show_plot=True,label=None, plot_color='b',
                          use_isf=True):
        if (type(fit)!=xr.Dataset) and (type(fit)!=list) and (fit is None):
            fit_keys = list(self.fittings)
            fit = self.fittings[fit_keys[-1]]['fit'] #gets the latest fit
            print(f"Using fit stored with key '{fit_keys[-1]}'.")

        theory_values = fit.theory
        if (fit.data_to_use == 'DDM Matrix'):
            if 'ravs' in fit.data_vars:
                data_values = fit.ravs
            elif 'ddm_matrix_data' in fit.data_vars:
                data_values = fit.ddm_matrix_data
            if use_isf:
                if "Background" in fit.parameter:
                    background = fit.parameters.loc['Background']
                else:
                    background = fit.B
                if "Amplitude" in fit.parameter:
                    amplitude = fit.parameters.loc['Amplitude']
                else:
                    amplitude = fit.A
                data_values = 1 - ((data_values - background) / amplitude)
                theory_values = 1 - ((theory_values-background)/amplitude)
        elif (fit.data_to_use == 'ISF'):
            data_values = fit.isf_data
        error_vs_lagtime = data_values - theory_values
        mean_squared_error = np.mean(error_vs_lagtime**2, axis=0) #mean square error as function of q
        if show_plot:
            if axes is None:
                fig,axes = plt.subplots(figsize=(10,10/1.618))
            axes.semilogx(fit.lagtime, error_vs_lagtime[:,q_index], 'o', color=plot_color, label=label)
            axes.set_xlabel("Lag time (s)")
            axes.set_ylabel("Error between data and fit.")
            axes.hlines(0, fit.lagtime[0], fit.lagtime[-1], 'k', linestyle=':')
            axes.set_title("q value of %.4f" % fit.q[q_index])
        print("Mean square error of %.8f" % mean_squared_error[q_index])
        return error_vs_lagtime, mean_squared_error


    def inspect_fit(self, q_at=5, fit_model=None):

        if fit_model == None:
            fit_model=[self.fit_model]

        if self.model_dict['data_to_use'] == 'ISF':
            data = self.ddm_dataset.ISF
        elif self.model_dict['data_to_use'] == 'DDM Matrix':
            data = self.ddm_dataset.ravs


        ##Only compare parameter settings of fits with the SAME model

        list_keys=list(self.fittings.keys())
        for n in range(0,len(list_keys)-1):

            if self.fittings[list_keys[n]]['model'] in fit_model and\
            self.fittings[list_keys[n]]['model']==self.fittings[list_keys[n+1]]['model']:

                try:
                    for i in range(0,10):
                        #Compare parameters sequentially
                        dict1 = self.fittings[f'{list_keys[n]}']['settings'][i]
                        dict2 = self.fittings[f'{list_keys[n+1]}']['settings'][i]

                        diffkeys = [k for k in dict1 if dict1[k] != dict2[k]]

                        for k in diffkeys:
                            print(f'{list_keys[n]}->{list_keys[n+1]}')
                            print(dict1['parname'],k, ':', dict1[k], '->', dict2[k], '\n')

                except:
                        pass

        #Plot all the different fits of model in one graph, for a given q-value


        plt.figure(figsize=(8,8./1.618))
        plt.semilogx(self.ddm_dataset.lagtime, data.sel(q=q_at, method='nearest'),'ko')

        for keys in list_keys:

            if self.fittings[keys]['model'] in fit_model:

                plt.semilogx(self.ddm_dataset.time,
                             self.fittings[keys]['fit'].theory.sel(q=q_at, method='nearest'),
                             '-', lw=3, label=keys+' '+self.fittings[keys]['model'])

                plt.legend()


        plt.xlabel("Time (s)")
        plt.ylabel(f"{self.model_dict['data_to_use']}")
        plt.title(f"Inspecting {self.fit_model} fit at q-vlaue of {q_at} (μm$^{-1}$)", fontsize=14)
        plt.show()


    def save_fits_disk(self, fit=None, save_directory=None, file_type = 'pickle', fit_fname_end=None):
        r"""Saves fit results
        
        Fit results are saved either as a pickle format or as a netcdf file. 
        

        Parameters
        ----------
        fit_name : string or xarray Dataset, optional
            The fit to save. The default is None. If None, then *all* fit results saved 
            in the dictionary self.fittings will be saved. If not all fits should be saved, 
            then set `fit_name` to the dictionary key (as string) of self.fittings *OR* set 
            `fit_name` to the xarray Dataset. 
        save_directory : string, optional
            Directory to save data. The default is None. If None, then the directory 
            where this will be saved will be the `DataDirectory` specified in the yaml 
            data used when the :py:class:`ddm_analysis_and_fitting.DDM_Fit` class was initialized. 
        file_type : string, optional
            Either 'pickle' or 'netcdf'. Default is 'pickle'.
        fit_fname_end : string, optional
            Saved filename will contain the image data filename with this string appended 
            at the end. Note that if `fit` was passed as a string, then that string will be 
            used in the filename (unless this parameter is not None, the default). If the 
            `fit` parameter is a Dataset, then the default is to use a filename with 'fit' 
            appended at the end (unless this parameter is not None). Note that if *all* fits 
            in the self.fittings dictionary are to be saved, then this parameter is irrevelant and 
            the keys of the dictionary will be used in the filenames. 

        """

        if save_directory is None:
            save_directory = self.data_dir

        if fit is not None:
            if isinstance(fit, xr.core.dataset.Dataset):
                fit_to_save = fit
                if fit_fname_end == None:
                    fit_fname_end = 'fit'
            elif isinstance(fit, str):
                if fit in self.fittings:
                    fit_to_save = self.fittings[fit]['fit']
                    if fit_fname_end == None:
                        fit_fname_end = fit
                else:
                    print("The key %s was not found in self.fittings." % fit)
                    print("Available keys are: ", list(self.fittings.keys()))
                    return 
            else:
                print("Must pass a fit of either type xarray.core.dataset.Dataset or as a string.")
                return
            if (file_type == 'pickle') or (file_type == 'Pickle'):
                saved_fit_filename = f'{save_directory}ddm_fit_results_{self.filename_noext}_{fit_fname_end}.pickle'
                try:
                    with open(saved_fit_filename, 'wb') as f:
                        pickle.dump(fit_to_save, f, pickle.HIGHEST_PROTOCOL)
                except:
                    print(f'Could not save fit results as: {saved_fit_filename}.')
            else:
                saved_fit_filename = f'{save_directory}ddm_fit_results_{self.filename_noext}_{fit_fname_end}.nc'
                fit_to_save.to_netcdf(path = saved_fit_filename)
            

        else:
            for name_of_fit in self.fittings:
                if (file_type == 'pickle') or (file_type == 'Pickle'):
                    saved_fit_filename = f'{save_directory}ddm_fit_results_{self.filename_noext}_{name_of_fit}.pickle'
                    try:
                        with open(saved_fit_filename, 'wb') as f:
                            pickle.dump(self.fittings[name_of_fit]['fit'], f, pickle.HIGHEST_PROTOCOL)
                    except:
                        print(f'Could not save fit results as: {saved_fit_filename}.')
                else:
                    saved_fit_filename = f'{save_directory}ddm_fit_results_{self.filename_noext}_{name_of_fit}.nc'
                    self.fittings[name_of_fit]['fit'].to_netcdf(path = saved_fit_filename)



##############################################################################
# FUNCTIONS FOR INSPECTING FITS                                              #
##############################################################################


def get_tau_vs_q_fit(fit_results, use_new_tau=True, use_tau2=False, 
                     forced_qs=None, update_good_q_range=True, silent=False):
    r"""From decay  time (tau) vs wavevector (q), gets effective diffusion coeff and scaling exponent
    
    This function looks at tau vs q and fits tau(q) to a powerlaw. From this we
    can determine an effective diffusion coefficient (or velocity). This function 
    will also try to estimate a 'good' range of q values for which a power law 
    relationship holds. To do so, it uses a linear model estimator. [1]_ 
    
    Parameters
    ----------
    fit_results : xarray Dataset
        Dataset containing fit results. Must have 'q' as a coordinate and have 'Tau' 
        (and/or 'Tau2') as a parameter. 
    use_new_tau : bool, optional
        If True (default), uses the stretching exponent to find more representative 
        decay time for each q value. 
    use_tau2 : bool, optional
        If False (default), uses 'Tau'. This is the case if there is just one decay 
        time in the model used.
    forced_qs : list or None, optional
        If None (default), then the range of good q values for which a power law 
        relationship holds will be estimated. If not None, must be a list of number. 
        The first number would be the smallest index of q values and the second, the 
        largest index. 
    update_good_q_range : bool, optional
        If True (default), then the range of good q values for which a power law 
        relationship is observed will be updated using a linear model estimator. 
    

    Returns
    -------
    good_q_range : list
        List of two numbers. Low and high indices of 'good' q values.
    slope : float
        Slope of log tau vs log q. If -2, then dynamics are diffusive. If -1, 
        then dynamics are ballistic. If less than -2, dynamics are subdiffusive. 
    effective_diffconst : float
        The effective diffusion coefficient
    MSD_alpha : float
        Estimated power in the relationship MSD ~ (lagtime)^power
    MSD_effective_diffconst : float
        Estimated coefficient in relationship MSD = (coeff)(lagtime^power)
    diffusion_coeff : float
        Diffusion coefficient found by forcing a tau ~ q^-2 power law
    diffusion_coeff_std : float
        Standard deviation of previous value
    velocity : float
        Velocity found by forcing a tau ~ q^-1 power law
    velocity_std : float
        Standard deviation of previous value
        
    
    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html
    

    """

    if use_tau2:
        tau = fit_results.parameters.loc['Tau2',:]
    else:
        tau = fit_results.parameters.loc['Tau',:]
    if use_new_tau and ('StretchingExp' in fit_results.parameters.parameter):
        stretch_exp = fit_results.parameters.loc['StretchingExp',:]
        if (use_tau2 and ('StretchingExp2' in fit_results.parameters.parameter)):
            stretch_exp = fit_results.parameters.loc['StretchingExp2',:]
        if not silent:
            print("In function 'get_tau_vs_q_fit', using new tau...")
        tau = newt(tau, stretch_exp)
        
    q = fit_results.q

    if forced_qs is not None:
        lowq,hiq = forced_qs
        good_q_range = forced_qs
    else:
        lowq = 1
        hiq = len(q)-2
        good_q_range = [lowq,hiq]

    '''
    for more on this linear model estimator
    see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html
    '''
    try:
        estimator = RANSACRegressor(random_state=42)
        logq = np.log(q[lowq:hiq]).values
        logt = np.log(tau[lowq:hiq]).values
        estimator.fit(logq[:,np.newaxis], logt)
        slope = estimator.estimator_.coef_
        coef1 = estimator.estimator_.intercept_
        inlier_mask = estimator.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
    
        if update_good_q_range:
            a=np.array([[len(list(g)),k] for k, g in itertools.groupby(inlier_mask)])
            largest_run_of_trues = 0
            index_start = 1
            index_end = np.nan
            index_to_mark = 0
            for i in range(a.shape[0]):
                if a[i,1]==1:
                    if a[i,0]>largest_run_of_trues:
                        index_start = index_to_mark
                        index_end = index_start + a[i,0]
                        largest_run_of_trues = a[i,0]
                index_to_mark += a[i,0]
            if index_end != np.nan:
                good_q_range = [index_start + lowq - 1, index_end + lowq - 1]
    
        MSD_alpha = 2./(-1*slope)
        MSD_effective_diffconst = (1.0/np.exp(coef1))**MSD_alpha
        effective_diffconst = 1.0/np.exp(coef1)
    except:
        print("RANSACRegressor failed")
        slope = np.nan
        MSD_alpha = np.nan
        MSD_effective_diffconst = np.nan
        effective_diffconst = np.nan

    '''
    If we force a tau ~ 1/q or tau ~ 1/q^2 dependence to get
     or a velocity a diffusion coeff, respectively.
    '''
    diffusion_coeff = np.mean(1./(tau[good_q_range[0]:good_q_range[1]]*(q[good_q_range[0]:good_q_range[1]]**2)))
    diffusion_coeff_std = np.std(1./(tau[good_q_range[0]:good_q_range[1]]*(q[good_q_range[0]:good_q_range[1]]**2)))
    velocity = np.mean(1./(tau[good_q_range[0]:good_q_range[1]]*(q[good_q_range[0]:good_q_range[1]])))
    velocity_std = np.std(1./(tau[good_q_range[0]:good_q_range[1]]*(q[good_q_range[0]:good_q_range[1]])))

    return good_q_range, slope, effective_diffconst, MSD_alpha, MSD_effective_diffconst, diffusion_coeff.values, diffusion_coeff_std.values, velocity.values, velocity_std.values


def fit_report(fit_results, PDF_save=True, forced_qs=None, pdf_save_dir = "./", 
               forced_qs_for_tau2=None, q_indices=[10,20,30,40], use_new_tau=True, 
               fit_report_name=None, show=True):
    r"""Creates a set of plots based on fit results
    
    Creates plots (saved in a single PDF file if `PDF_save` is True (the default)) 
    based on the DDM fit results. Plots of the DDM matrix or the ISF are shown 
    versus lag time along with the theoretical model calculated with the best fit 
    parameters. Also shown are the decay time (tau) vs wavevector (q) plots, the 
    amplitude vs the wavevector, and the background vs the wavevector. Depending on the 
    fit model used, the second decay time, fraction of dynamics with the first decay 
    time, the non-ergodicity parameter, and the Schulz number may also be plotted. 
    
    By fitting the tau vs q plot to a power law, one can get parameters describing 
    the type of dynamics. These quantities are stored as attributes to the 
    `fit_results` xarray Dataset which is returned. 
    

    Parameters
    ----------
    fit_results : xarray Dataset
        Results of the DDM fitting to either the DDM matrix or the ISF. 
    PDF_save : bool, optional
        Saves plots to a PDF file. The default is True.
    forced_qs : list or None, optional
        Does power law fitting over this range of q. The default is None. If not 
        None, must be list-like with two numbers indicating the index of the 
        lower and upper q. 
    pdf_save_dir : str, optional
        Location to save the PDF. The default is "./" (the current working directory).
    forced_qs_for_tau2 : list or None, optional
        Like `forced_qs` but for the second decay time
    q_indices : list, optional
        List of q indices to show when plotting the DDM matrix or ISF vs lag time 
        along with the theoretical model. The default is [10,20,30,40].
    use_new_tau : bool, optional
        If True, calculates a more representative decay time based on the value of 
        tau and the stretching exponent. The default is True.
    fit_report_name : str, optional
        String appended to the filename of the PDF generated. The default is None. 
        If None, the file will be named 'ddm_fit_report.pdf' in the directory 
        specified by the parameter `pdf_save_dir`. If not None, then it will 
        be named 'ddm_fit_report_{fit_report_name}.pdf'
    show : bool, optional
        Plots will appear if True. The default is True.

    Returns
    -------
    fit_results : xarray Dataset
        Results of DDM fitting with attributes possibly changed

    """

    if fit_report_name is None:
        pdf_report_filename = f'{pdf_save_dir}ddm_fit_report.pdf'
    else:
        pdf_report_filename = f'{pdf_save_dir}ddm_fit_report_{fit_report_name}.pdf'

    if show:
        plt.ion()
    else:
        plt.ioff()

    if isinstance(fit_results, xr.core.dataset.Dataset):
        pass
    else:
        print("Not a valid parameter passed to fit_results.")
        return

    list_of_all_figs = [] #initialize empty list to contain figures

    fig1 = hf.plot_to_inspect_fit(np.arange(q_indices[0],q_indices[-1],4), fit_results,
                                 show_legend=False, show_colorbar=True,
                                 print_params=False)
    list_of_all_figs.append(fig1)
    fig2 = hf.plot_to_inspect_fit_2x2subplot(q_indices, fit_results)
    list_of_all_figs.append(fig2)


    if forced_qs is None:
        forced_qs = fit_results.attrs['good_q_range']
    qrange, slope, d_eff, msd_alpha, msd_d_eff, d, d_std, v, v_std = get_tau_vs_q_fit(fit_results, use_new_tau=use_new_tau,
                                                                                      forced_qs=forced_qs, update_good_q_range=True)
    fit_results.attrs['effective_diffusion_coeff'] = d_eff
    fit_results.attrs['tau_vs_q_slope'] = slope
    fit_results.attrs['msd_alpha'] = msd_alpha
    fit_results.attrs['msd_effective_diffusion_coeff'] = msd_d_eff
    fit_results.attrs['diffusion_coeff'] = d
    fit_results.attrs['diffusion_coeff_std'] = d_std
    fit_results.attrs['velocity'] = v
    fit_results.attrs['velocity_std'] = v_std
    if forced_qs is not None:
        if (len(forced_qs)==2) and (forced_qs[0]<forced_qs[1]):
            fit_results.attrs['good_q_range']=forced_qs
    else:
        fit_results.attrs['good_q_range'] = qrange


    fig3 = hf.plot_one_tau_vs_q(fit_results, 'b', use_new_tau=use_new_tau)
    list_of_all_figs.append(fig3)

    if ('StretchingExp' in fit_results.parameters.parameter):
        fig4 = hf.plot_stretching_exponent(fit_results, 'g', 0.6)
        list_of_all_figs.append(fig4)


    if ('Tau2' in fit_results.parameters.parameter):

        qrange, slope, d_eff, msd_alpha, msd_d_eff, d, d_std, v, v_std = get_tau_vs_q_fit(fit_results, use_tau2=True, use_new_tau=use_new_tau,
                                                                                          forced_qs=forced_qs_for_tau2, update_good_q_range=True)
        fit_results.attrs['tau2_effective_diffusion_coeff'] = d_eff
        fit_results.attrs['tau2_tau_vs_q_slope'] = slope
        fit_results.attrs['tau2_msd_alpha'] = msd_alpha
        fit_results.attrs['tau2_msd_effective_diffusion_coeff'] = msd_d_eff
        fit_results.attrs['tau2_diffusion_coeff'] = d
        fit_results.attrs['tau2_diffusion_coeff_std'] = d_std
        fit_results.attrs['tau2_velocity'] = v
        fit_results.attrs['tau2_velocity_std'] = v_std
        #fit_results.attrs['tau2_good_q_range'] = qrange
        if forced_qs_for_tau2 != None:
            if (len(forced_qs_for_tau2)==2) and (forced_qs_for_tau2[0]<forced_qs_for_tau2[1]):
                fit_results.attrs['tau2_good_q_range']=forced_qs_for_tau2
            else:
                fit_results.attrs['tau2_good_q_range'] = qrange

        fig3b = hf.plot_one_tau_vs_q(fit_results, 'c', use_tau2=True, use_new_tau=use_new_tau)
        list_of_all_figs.append(fig3b)
        fig3c = hf.plot_one_tau_vs_q(fit_results, 'b', show_table=False, use_new_tau=use_new_tau)
        fig3c = hf.plot_one_tau_vs_q(fit_results, 'c', use_tau2=True, fig_to_use=fig3c, show_table=False, use_new_tau=use_new_tau)
        list_of_all_figs.append(fig3c)
        if ('StretchingExp2' in fit_results.parameters.parameter):
            fig4b = hf.plot_stretching_exponent(fit_results, 'limegreen', 0.6, use_s2=True)
            list_of_all_figs.append(fig4b)

    fig5 = hf.plot_amplitude(fit_results)
    list_of_all_figs.append(fig5)
    fig6 = hf.plot_background(fit_results)
    list_of_all_figs.append(fig6)
    #fig7 = hf.plot_amplitude_over_background(fit_results, self.ddm_dataset)
    #list_of_all_figs.append(fig7)
    if ('Fraction1' in fit_results.parameters.parameter) or ('FractionBallistic' in fit_results.parameters.parameter):
        fig8 = hf.plot_fraction(fit_results)
        list_of_all_figs.append(fig8)
    if ('SchulzNum' in fit_results.parameters.parameter):
        fig9 = hf.plot_schulz(fit_results)
        list_of_all_figs.append(fig9)
    if ('SchulzNum2' in fit_results.parameters.parameter):
        fig10 = hf.plot_schulz(fit_results, use2=True)
        list_of_all_figs.append(fig10)
    if 'NonErgodic' in fit_results.parameters.parameter:
        fig11 = hf.plot_nonerg(fit_results)
        list_of_all_figs.append(fig11)

    if PDF_save:
        with PdfPages(pdf_report_filename) as pdf:
            for fgr in list_of_all_figs:
                pdf.savefig(fgr, bbox_inches='tight')

    if not show:
        plt.close('all')
        
    return fit_results


def save_fit_results_to_excel(fit_results, also_save_data=True, file_name_end=None, save_dir=None):
    r"""Saves fit results to Excel file
    
    Metadata, fit results, and (optionally, if `also_save_data` is True (the default)) the data fit to 
    are saved to an Excel file. Data is first converted to Pandas dataframe. Then the 
    data is saved in a multi-sheet Excel (\*.xlsx) file. 
    

    Parameters
    ----------
    fit_results : xarray Dataset
        Result of DDM fitting. Returned from :py:meth:`PyDDM.ddm_analysis_and_fitting.DDM_Fit.fit`
    also_save_data : bool, optional
        Data fit to (either ISF or DDM matrix) is also saved to Excel file. The default is True.
    file_name_end : str, optional
        Append this string to the end of the saved filename. The default is None.
    save_dir : str, optional
        Directory to save Excel file. The default is None. If None, then will use the DataDirectory 
        which is (hopefully) an attribute of the fit_results Dataset. 

    Returns
    -------
    None.

    """

    bg = fit_results.B.values
    data_dir = fit_results.DataDirectory if 'DataDirectory' in fit_results.attrs else None
    data_filenm = fit_results.FileName if 'FileName' in fit_results.attrs else None
    pixel_size = fit_results.pixel_size if 'pixel_size' in fit_results.attrs else None
    frame_rate = fit_results.frame_rate if 'frame_rate' in fit_results.attrs else None
    

    about_fit_data = {'Parameters': ['Data directory', 'File', 'Pixel size', 'Framerate', 'B',
                               'tau vs q slope', 'Effective diffusion coeff',
                               'MSD alpha', 'MSD effective diffusion coeff',
                               'Diffusion coeff', 'Diffusion coeff std',
                               'Velocity', 'Velocity std', 'q range'],
                      'Values': [data_dir, data_filenm, pixel_size,
                               frame_rate, bg,
                               fit_results.tau_vs_q_slope, fit_results.effective_diffusion_coeff,
                               fit_results.msd_alpha, fit_results.msd_effective_diffusion_coeff,
                               fit_results.diffusion_coeff, fit_results.diffusion_coeff_std,
                               fit_results.velocity, fit_results.velocity_std,
                               fit_results.good_q_range]}

    if 'tau2_effective_diffusion_coeff' in fit_results:
        about_fit_data['Parameters'].append('2nd tau vs q slope')
        about_fit_data['Parameters'].append('2nd Effective diffusion coeff')
        about_fit_data['Parameters'].append('2nd MSD alpha')
        about_fit_data['Parameters'].append('2nd MSD effective diffusion coeff')
        about_fit_data['Parameters'].append('2nd Diffusion coeff')
        about_fit_data['Parameters'].append('2nd Diffusion coeff std')
        about_fit_data['Parameters'].append('2nd Velocity')
        about_fit_data['Parameters'].append('2nd Velocity std')
        about_fit_data['Parameters'].append('2nd q range')
        about_fit_data['Values'].append(fit_results.tau2_tau_vs_q_slope)
        about_fit_data['Values'].append(fit_results.tau2_effective_diffusion_coeff)
        about_fit_data['Values'].append(fit_results.tau2_msd_alpha)
        about_fit_data['Values'].append(fit_results.tau2_msd_effective_diffusion_coeff)
        about_fit_data['Values'].append(fit_results.tau2_diffusion_coeff)
        about_fit_data['Values'].append(fit_results.tau2_diffusion_coeff_std)
        about_fit_data['Values'].append(fit_results.tau2_velocity)
        about_fit_data['Values'].append(fit_results.tau2_velocity_std)
        about_fit_data['Values'].append(fit_results.tau2_good_q_range)

    about_fit_data['Parameters'].append('Fit model')
    about_fit_data['Parameters'].append('Date fit ran')
    about_fit_data['Parameters'].append('Computer ran on')
    about_fit_data['Values'].append(fit_results.model)
    about_fit_data['Values'].append(time.ctime())
    about_fit_data['Values'].append(socket.gethostname())

    dataframe_about = pd.DataFrame(data = about_fit_data)

    dataframe_bestfitparams = fit_results.parameters.to_dataframe()
    dataframe_theory = fit_results.theory.transpose().to_dataframe()
    dataframe_initialguess = pd.DataFrame(data = fit_results.initial_params_dict)
    if also_save_data:
        if fit_results.data_to_use == 'ISF':
            if 'isf_data' in fit_results:
                dataframe_data = fit_results.isf_data.transpose().to_dataframe()
            else:
                also_save_data = False
            dataframe_amplitude = fit_results.A.to_dataframe()
        elif fit_results.data_to_use == 'DDM Matrix':
            if 'ddm_matrix_data' in fit_results:
                dataframe_data = fit_results.ddm_matrix_data.transpose().to_dataframe()
            else:
                also_save_data = False

    if save_dir is None:
        save_dir = "./" if data_dir is None else data_dir
    if data_filenm is None:
        excel_file_name = f"fitresults_{fit_results.model}"
    else:
        excel_file_name = f"{data_filenm[:-4]}_fitresults_{fit_results.model}"
    if file_name_end is not None:
        excel_file_name = f"{excel_file_name}_{file_name_end}"
        
    full_path = f"{save_dir}{excel_file_name}.xlsx"

    with pd.ExcelWriter(full_path) as writer:
        dataframe_about.to_excel(writer, sheet_name='About')
        dataframe_bestfitparams.to_excel(writer, sheet_name='Best fit parameters')
        dataframe_initialguess.to_excel(writer, sheet_name='Initial guess')
        dataframe_theory.to_excel(writer, sheet_name='Theory')
        if also_save_data:
            dataframe_data.to_excel(writer, sheet_name='Data %s' % fit_results.data_to_use)
            if fit_results.data_to_use == 'ISF':
                dataframe_amplitude.to_excel(writer, sheet_name='Amplitude')




#see https://matplotlib.org/stable/gallery/event_handling/data_browser.html
class Browse_DDM_Fits:
    """
    Class for interactive inspection of DDM data fits. 
    
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    #def __init__(self, ax, ax2, q, t, ddm_matrix, lagtimes, thry):
    def __init__(self, fig, ax, ax2, fit):
        """
        Class for interactive inspection of fits.
        
        
        :param q: Name of the model for fitting
        :type q: ndarray

        ...
        """
        self.ax = ax
        self.ax2 = ax2
        self.fig = fig
        
        self.lastind = 0
        
        '''
        self.q = q
        self.t = t
        self.ddm_matrix = ddm_matrix
        self.lagtimes = lagtimes
        self.theory = thry
        '''
        
        if fit.data_to_use == "DDM Matrix":
            self.data = fit.ddm_matrix_data
        elif fit.data_to_use == "ISF":
            self.data = fit.isf_data
        self.lagtimes = fit.lagtime.values
        self.theory = fit.theory.values
        self.q = fit.q.values
        self.t = fit.parameters.loc['Tau'].values
        
        self.ax.set_title('Decay time vs wavevector')
        self.ax.set_xlabel("q ($\mu$m$^{-2}$)")
        self.ax.set_ylabel("Decay time (s)")
        self.line, = ax.loglog(self.q, self.t, 'o', picker=True, pickradius=150)

        self.text = self.ax.text(0.05, 0.95, 'selected: none',
                            transform=self.ax.transAxes, va='top')
        self.selected, = self.ax.plot([self.q[1]], [self.t[1]], 'o', ms=12, alpha=0.4,
                                 color='yellow', visible=False)
        
        print("Click on a point in the tau vs q plot to see a fit.")
        print("Or press 'N' or 'P' to display next or previous fit.")

    def on_press(self, event):
        if self.lastind is None:
            return
        if event.key not in ('n', 'p'):
            return
        if event.key == 'n':
            inc = 1
        else:
            inc = -1

        self.lastind += inc
        self.lastind = np.clip(self.lastind, 0, len(self.q) - 1)
        self.update()

    def on_pick(self, event):

        if event.artist != self.line:
            return True

        N = len(event.ind)
        if not N:
            return True

        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        distances = np.hypot(x - self.q[event.ind], y - self.t[event.ind])
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        self.lastind = dataind
        self.update()

    def update(self):
        if self.lastind is None:
            return

        dataind = int(self.lastind)

        self.ax2.cla()
        self.ax2.semilogx(self.lagtimes,self.data[:,dataind],'ro')
        self.ax2.semilogx(self.lagtimes,self.theory[:,dataind],'-b')
        self.ax2.set_xlabel("Lag time (s)")

        self.ax2.text(0.05, 0.9, f'q={self.q[dataind]:1.3f}',
                 transform=self.ax2.transAxes, va='top')
        #ax2.set_ylim(-0.5, 1.5)
        self.selected.set_visible(True)
        self.selected.set_data(self.q[dataind], self.t[dataind])

        self.text.set_text('selected: %d' % dataind)
        #print(dataind)
        self.fig.canvas.draw()
        

def interactive_fit_inspection(fit):
    r"""Interactive plot of tau vs q to inspect each fit.
    
    Generates a decay time (tau) vs wavevector (q) plot on a log-log scale. Click 
    a point to view a plot of the ISF or DDM matrix (whichever was used to fit) 
    versus lag time for chosen q value. Pressing the 'N' or 'P' keys will also 
    advance forward ('N'ext) or backwards ('P'revious) through the tau vs q points.
    

    Parameters
    ----------
    fit : xarray dataset
        Results of fit to either DDM matrix or ISF.

    Returns
    -------
    fig : matplotlib figure
        Interactive figure

    """
    
    fig, (ax, ax2) = plt.subplots(2, 1)
    ax.set_title('Decay time vs wavevector')
    ax.set_xlabel("q")
    ax.set_ylabel("tau (s)")
    #line, = ax.loglog(qvals, taus, 'o', picker=True, pickradius=150)
    
    browser = Browse_DDM_Fits(fig, ax, ax2, fit)# qvals, taus, ddm_matrix, lagtimes, thry)
    
    fig.canvas.mpl_connect('pick_event', browser.on_pick)
    fig.canvas.mpl_connect('key_press_event', browser.on_press)
    
    plt.show()
    return fig
