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
from nd2reader import ND2Reader #https://rbnvrw.github.io/nd2reader/index.html


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


def recalculate_ISF_with_new_background(ddm_dataset, background):
    r"""
    The intermediate scattering function (ISF) is re-calculated from the DDM matrix, with the given background value.
    
    Recall the relationship between the DDM matrix (:math:`D(q,\Delta t)`) and the ISF (:math:`f(q, \Delta t)`): 
        
    .. math::  D(q, \Delta t) = A(q) \left[ 1 - f(q, \Delta t) \right] + B(q)
    
    We can estimate the amplitude,:math:`A(q)`, and background, :math:`B(q)`, terms by first calculating the Fourier 
    transforms of all images and averaging those together. See the function :py:func:`PyDDM.ddm_calc.determining_A_and_B`. With 
    that function and by assuming that the background is a constant with wavevector (i.e., independent of :math:`q`) we can determine 
    :math:`A(q)` and :math:`B`. With :math:`D(q,\Delta t)`, :math:`A(q)`, and :math:`B`, we can then find the ISF :math:`f(q, \Delta t)`. 
    This is done automatically when calculating the DDM matrix from the time series of images. For that, we assume that the 
    amplitude goes to zero by the last 10% of the wavevectors. But, this might not be the case. This may over estimate the 
    background. Therefore, one may want to adjust the background value and recalculate the ISF. 
    
    Parameters
    ----------
    ddm_dataset : xarray Dataset
        Dataset calculated with :py:meth:`PyDDM.ddm_analysis_and_fitting.DDM_Analysis.calculate_DDM_matrix`
    background : float
        Esimate of the background parameter

    Returns
    -------
    ddm_dataset : xarray Dataset
        New Dataset with the variables `Amplitude` and `ISF` adjusted.

    """
    if ("avg_image_ft" in ddm_dataset) and ("ddm_matrix" in ddm_dataset):
        ddm_dataset['B'] = background
        ddm_dataset["Amplitude"] = 2 * ddm_dataset['avg_image_ft']-background
        ddm_dataset['ISF'] = 1-(ddm_dataset.ddm_matrix-background)/ddm_dataset.Amplitude
        return ddm_dataset
    elif ("av_fft_offrame" in ddm_dataset) and ("ravs" in ddm_dataset):
        ddm_dataset['B'] = background
        ddm_dataset["Amplitude"] = 2 * ddm_dataset['av_fft_offrame']-background
        ddm_dataset['ISF'] = 1-(ddm_dataset.ravs-background)/ddm_dataset.Amplitude
        return ddm_dataset
    else:
        return None
    
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



class DDM_Analysis:
    """Performs preprossing of data, such as cropping, windowing etc. DDM calculations are performed on the processed time series
    to produce a xarray DataSet with DDM matrix, radial averages and ISF. The analysis parameters are provided by the user in 
    a YAML file: :doc:`More information here </Provide_info_for_analysis>` """

    def __init__(self, data_yaml):
        
        self.data_dir = None
        self.filename = None
        self.number_of_lag_times = None
        
        if (isinstance(data_yaml, str)) or (isinstance(data_yaml, dict)):
            self.data_yaml=data_yaml
        else:
            print("Incorrect data type for analysis parameters. Argument must be filename or dictionary.")
        success = self.loadYAML()
        if success:
            self.setup()
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


        self.data_dir=self.content['DataDirectory']
        self.filename= self.content['FileName']
        

        #Make sure the path to the movie exists before proceeding
        if os.path.exists(self.data_dir+self.filename):
            #print('File path to image data exists.')
            #ddm.logger.info("File path to image data exists.")
            self.metadata = self.content['Metadata']
            self.pixel_size = self.metadata['pixel_size']
            self.frame_rate = self.metadata['frame_rate']
            self.analysis_parameters = self.content['Analysis_parameters']
            if 'filename_for_saved_data' in self.analysis_parameters:
                self.filename_for_saving_data = self.analysis_parameters['filename_for_saved_data']
            else:
                self.filename_for_saving_data = self.filename[:-4]
            if 'ending_frame_number' in self.analysis_parameters:
                self.last_frame = self.analysis_parameters['ending_frame_number']
            else:
                self.last_frame = None
            if 'starting_frame_number' in self.analysis_parameters:
                self.first_frame = self.analysis_parameters['starting_frame_number']
            else:
                self.first_frame = 0
            self.number_of_lag_times = self.analysis_parameters['number_lagtimes']
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
            print(f'Provided metadata: {self.metadata}')
            #ddm.logger2.info(f'Provided metadata: {self.metadata}')
            return 1
        else:
            print('Error: check path to image file')
            return 0


    def set_filename_for_saving(self, filename, quiet=False):
        '''
        Change file name to save the data to disk

        :param filename: New file name
        :type filename: str


        '''
        self.filename_for_saving_data = filename
        if not quiet:
            print("Previous filename for saving ddm data was %s." % self.filename_for_saving_data)
            print("New filename for saving data will be %s." % self.filename_for_saving_data)


    def _openImage(self):
        '''
        Opens .nd2 file or .tif file and returns image series as multidimensional numpy array

        :return:
            * im (*numpy array*)- image series as numpy array



        '''

        if re.search(".\.nd2$", self.filename) is not None:
            # Files with nd2 extension will be read using the package
            #  nd2reader. Nikon systems may save data with this file type.
            if 'channel' in self.metadata:
                channel = self.metadata['channel']
            else:
                print("Need to specify channel in yaml metadata. Defaulting to c=0.")
                channel = 0
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
                    im[i] = images.get_frame_2D(t=i, c=channel)

        if (re.search(".\.tif$", self.filename) is not None) or (re.search(".\.tiff$", self.filename) is not None):
            im = io.imread(self.data_dir + self.filename)

        return im


    def setup(self):
        '''This function uses the user provided data to prepare the movie for DDM analysis:
            -crops the number of frames based on given max frame numbers
            if wanted:
            -crops the frame size (splits it in four tiles)
            -splits it in four tiles
            -binning
            -applies window function'''

        image_data = self._openImage()
        print("Image shape: %i-by-%i-by-%i" % image_data.shape)

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

        self.lag_times_frames = ddm.generateLogDistributionOfTimeLags(self.first_lag_time, self.last_lag_time,
                                                                      self.number_of_lag_times)
        self.lag_times = self.lag_times_frames / self.frame_rate



    def calculate_DDM_matrix(self, fast_mode=False, quiet=False, maximal_overlap=False):
        '''Function that handels the computation of DDM_matrix for multiple (time or space) windows of same movie
            Calculates the DDM matrix and radial averages then estimates
            amplitude (A) and background (B) based on direct fourier transform (FFT)
            of images (instead of FFT of difference images). Determines the ISF from
            A, B and radial averages. The data set is saved as netCDF file and a
            pdf report is produced

            :param fast_mode: Fast computation of the DDM matrix, by setting all the lagtimes above a value to a maximum.
            :type fast_mode: bool
            :param quiet: Should be False
            :type quiet: bool

            :return:
                * ddm_dataset (*Dataset*)- xarray Dataset with:
                                            - DDM matrix, radial averages, intermediate scattering fucntion (ISF), Amplitude and background (B).
                                            - coordinates are: wavectors (q_x,q_y and magnitude:q), lagtime
                                            - attributes contains Meta data



            '''
        quit_now = False

        filename_to_be_used = f"{self.data_dir}{self.filename_for_saving_data}_ddmmatrix.nc"
        if os.path.exists(filename_to_be_used):
            print(f"The file {filename_to_be_used} already exists. So perhaps the DDM matrix was calculated already?")
            answer = input("Do you still want to calculate the DDM matrix? (y/n): ").lower().strip()
            if answer == "n" or answer=="no":
                self.ddm_dataset = xr.open_dataset(filename_to_be_used)
                self.ddm_dataset.close()
                return

        self.fast_mode=fast_mode
        if self.fast_mode:
            print("Calculating the DDM matrix in fast mode...")

        #print(f"Calculating the DDM matrix for {self.filename}...")
        self._computeDDMMatrix(quiet=quiet, maximal_overlap=maximal_overlap)



    #Do not call this function, instead call analysis_flow
    def _computeDDMMatrix(self, quiet=False, maximal_overlap=False):
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
        try:
            if type(self.im)==list:
                for i,im in enumerate(self.im):
                    print(f"Getting DDM matrix for {i+1} of {len(self.im)}...")
                    d_matrix, num_pairs = ddm.computeDDMMatrix(im, self.lag_times_frames, fast_mode = self.fast_mode, quiet=quiet,
                                                               maximal_overlap=maximal_overlap)
                    self.ddm_matrix.append(d_matrix)
                self.num_pairs_per_dt = num_pairs
            else:
                self.ddm_matrix, self.num_pairs_per_dt = ddm.computeDDMMatrix(self.im, self.lag_times_frames, 
                                                                              fast_mode = self.fast_mode, quiet=quiet,
                                                                              maximal_overlap=maximal_overlap)
            self.ddm_matrix_fastmode = self.fast_mode
            end_time = time.time()
        except:
            print("Unable to get DDM matrix")
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

        #Determine Amplitude and Background from radial averages of directly fourier transformed images (not difference images)
        # Note: windowing (if applicable) already applied to self.im, so "use_BH_filter" should be False always
        if type(self.im)==list:
            self.ravfft = []
            for i,im in enumerate(self.im):
                r = ddm.determining_A_and_B(im, use_BH_filter=False,centralAngle=self.central_angle,
                                            angRange=self.angle_range)
                self.ravfft.append(r)
        else:
            self.ravfft = ddm.determining_A_and_B(self.im, use_BH_filter=False,
                                                  centralAngle=self.central_angle,
                                                  angRange=self.angle_range)


        if type(self.im)==list:
            self.ddm_dataset = []
            for i in range(len(self.im)):
                filename = f"{self.data_dir}{self.filename_for_saving_data}_{i:02}"
                ds = self._create_dataset_and_report(filename, i)
                self.ddm_dataset.append(ds)
        else:
            filename = f"{self.data_dir}{self.filename_for_saving_data}"
            self.ddm_dataset = self._create_dataset_and_report(filename)

        return True


    def _create_dataset_and_report(self, file_name, num=None):

        if type(self.ddm_matrix)==list:
            if (num==None) or (len(self.ddm_matrix)<num):
                num=0
                print("Function '_create_dataset' needs 'num' parameter. Setting to 0.")
            ddm_matrix = self.ddm_matrix[num]
            ravfft = self.ravfft[num]
            ravs = self.ravs[num]
            image0 = self.im[num][0].astype(np.float64)
        else:
            ddm_matrix = self.ddm_matrix
            ravfft = self.ravfft
            ravs = self.ravs
            image0 = self.im[0].astype(np.float64)

        #Put ddm_matrix and radial averages in a dataset:
        ddm_dataset=xr.Dataset({'ddm_matrix_full':(['lagtime', 'q_y','q_x'], ddm_matrix), #was 'ddm_matrix'
                                'ddm_matrix':(['lagtime', 'q'], ravs), #was 'ravs'
                                'first_image':(['y','x'], image0)},
                               coords={'lagtime': self.lag_times,
                                       'framelag':('frames', self.lag_times_frames),
                                       'q_y':self.q_y, 'q_x':self.q_x, 'q':self.q,
                                       'y':np.arange(image0.shape[0]), 'x':np.arange(image0.shape[1])},
                               attrs={'units':'Intensity', 'lagtime':'sec',
                                      'q':'μm$^{-1}$',
                                      'x':'pixels', 'y':'pixels',
                                      'info':'ddm_matrix is the averages of FFT difference images, ravs are the radial averages'})

        ddm_dataset['avg_image_ft'] = (('q'), ravfft[0,:]) # av_fft_offrame=0.5*(A+B) #was 'av_fft_offrame'
        
        #Get last 10% of q. Average used for estimating background
        number_of_hi_qs = int(0.1*len(self.q))
        
        ddm_dataset['B'] = 2*ddm_dataset.avg_image_ft[-1*number_of_hi_qs:].mean()
        ddm_dataset['B_std'] = 2*ddm_dataset.avg_image_ft[-1*number_of_hi_qs:].std()
        ddm_dataset['num_pairs_per_dt'] = (('lagtime'),self.num_pairs_per_dt)
        print(f" Background estimate ± std is {ddm_dataset.B.values:.2f} ± {ddm_dataset.B_std.values:.2f}")

        # Calculate amplitude: av_fft_frame=0.5(A+B)->A=2*av_fft_frame-B
        ddm_dataset["Amplitude"] = 2 * ddm_dataset['avg_image_ft']-ddm_dataset.B

        # calculate ISF with new amplitude and background
        ddm_dataset['ISF']=1-(ddm_dataset.ddm_matrix-ddm_dataset.B)/ddm_dataset.Amplitude

        ##write yaml file data to xarray attribute in format accepeted by net cdf (no multiple dimension or Booleans)
        for i in self.content:

            if not i.startswith('Fit'):
                try:
                    for j, k in self.content[i].items():
                        if k==True:
                            ddm_dataset.attrs[j]= 'yes'
                        elif k==False or k==None:
                            ddm_dataset.attrs[j]= 'no'
                        else:
                            ddm_dataset.attrs[j]=k
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

        #Release any resources linked to this object.
        ddm_dataset.close()

        return ddm_dataset
    
    
    def resave_ddm_dataset(self, ddmdataset, file_name = None):
        r"""
        Resave the DDM dataset. 

        Parameters
        ----------
        ddmdataset : xarray dataset
            DDM dataset
        file_name : string or None, optional
            DESCRIPTION. The default is None.


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
                    rect = Rectangle((x1,y1),xsize,ysize,linewidth=2,edgecolor='r',facecolor='none')
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
        plt.ylabel("0.5 * (A+B)")
        plt.hlines(0.5*ddmdataset.B.values, ddmdataset.coords['q'][3], ddmdataset.coords['q'][-1], linestyles='dashed')
        plt.title(f"Estimation of background (B) and amplitude (A), based on average of \n FFT directly from frames. Dashed line at {0.5*ddmdataset.B.values:.2f}. \n So background ~ {ddmdataset.B.values:.0f}", fontsize=10)
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
    Set of functions to fit DDM matrix (image structure function) or ISF, the user can choose from a variety of mathematical models :py:mod:`PyDDM.ISF_and_DDMmatrix_theoretical_models`. 
    Analysis paramters can be provided in a YAML file :doc:`More information here </Provide_info_for_analysis>`.
    Graphs of the fits are produced and can be saved as PDF.

    """

    def __init__(self, data_yaml, subimage_num=None):
        self.subimage_num = subimage_num
        self.model_dict = None
        self.data_yaml = data_yaml
        self.loadYAML()
        self.fittings = {} #Stores fit_model,parameters and data set, key-value provided by user
        self.use_parameters_provided()
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


    def use_parameters_provided(self, print_par_names=False):
        """Gives the parameters and their values, as provided in the intiation file (YAML file)

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
                self.ddm_dataset=xr.open_dataset(filename)
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
                self.ddm_dataset=xr.open_dataset(dataset_filename)
                self.ddm_dataset.close()
            else:
                print(f"File {dataset_filename} not found.")
                fls = glob.glob(f"{self.data_dir}{file_name}*.nc")
                if len(fls)>0:
                    print("Perhaps you meant to load one of...")
                    for flnm in fls:
                        print(f"\t{flnm}")
                    print(f"By default, loading {fls[0]}")
                    self.ddm_dataset=xr.open_dataset(fls[0])
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
        if self.content['Fitting_parameters']['Auto_update_good_q_range']:
            update_good_q_range = True
        else:
            update_good_q_range = False

        qrange, slope, d_eff, msd_alpha, msd_d_eff, d, d_std, v, v_std = get_tau_vs_q_fit(fit_results, forced_qs=force_q_range, 
                                                                                          update_good_q_range=update_good_q_range)
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
                                                                                              forced_qs=force_q_range, update_good_q_range=update_good_q_range)
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
            print(f"Fit is saved in fittings dictionary with key '{name}'.")

        if display_table:
            pd = hf.generate_pandas_table_fit_results(fit_results)
            display(pd)
            
        #Placing other metadata in the fit_results Dataset
        fit_results.attrs['DataDirectory'] = self.ddm_dataset.DataDirectory
        fit_results.attrs['FileName'] = self.ddm_dataset.FileName
        fit_results.attrs['pixel_size'] = self.ddm_dataset.pixel_size
        fit_results.attrs['frame_rate'] = self.ddm_dataset.frame_rate

        return fit_results


    def _save_fit(self, fit_results, name_fit=None):

        #Save the fit and parameter settings to a dictionary with a key value provided by the user

        if name_fit == None:
            name_fit=input("Under what name would you want to save the generated fit in the dictionary 'fittings': ")

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
                     forced_qs=None, update_good_q_range=True):
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


    fig3 = hf.plot_one_tau_vs_q(fit_results, 'b', 0.2, y_position_of_text=0.2, use_new_tau=use_new_tau)
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

        fig3b = hf.plot_one_tau_vs_q(fit_results, 'c', 0.6, use_tau2=True, use_new_tau=use_new_tau)
        list_of_all_figs.append(fig3b)
        fig3c = hf.plot_one_tau_vs_q(fit_results, 'b', 0.6, show_table=False, use_new_tau=use_new_tau)
        fig3c = hf.plot_one_tau_vs_q(fit_results, 'c', 0.3, use_tau2=True, fig_to_use=fig3c, show_table=False, use_new_tau=use_new_tau)
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