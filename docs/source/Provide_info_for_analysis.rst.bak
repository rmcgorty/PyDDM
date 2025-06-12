Parameters for DDM analysis
****************************


Here is an explanation for all the entries which are in the iniation file needed for the DDM analysis. 

.. contents:: :local:

DataDirectory
=============
Provide the full link to the location of the movie to be analyzed. Note that, by default, much of the 
DDM analysis outputs will be saved to this folder. E.g., *'C:/Users/User_name/Folder/Subfolder/'* (note 
that the quotes are necessary).
	
FileName
========
Provide the full name of the file to be analyzed including the extention (either *.tif* or *.nd2*). E.g., 
*'images_nobin_40x_128x128_8bit.tif'*. Note that if 
you intend to analyze *.nd2* files (the format when using Nikon microscope software) then you 
will need the package `nd2reader`_.

.. _nd2reader: https://github.com/Open-Science-Tools/nd2reader

	
Metadata
========
The following information about the images acquired must be provided. 

pixel_size
-----------
Give the pixel size in μm, e.g., *0.242*.

frame_rate
------------
Provide the number of frames per second, e.g., *41.7*.


Analysis_parameters
====================
Parameters for determining the DDM matrix.
  
starting_frame_number
----------------------
The number of the first frame that should be analyzed, e.g., *0* if you want to analyze 
the movie from the beginning. If the first frame or two are bad, then you can adjust this 
parameter. You could also adjust if you want to break up a long movie into segments in time for analysis. 
  
ending_frame_number
--------------------
The last frame to be analyzed, e.g., *3000*. If the last frame of the movie is the last frame for analysis, give *null*
 
number_lag_times
----------------
The number of lag times to be samples, e.g., *40*.
 
first_lag_time
---------------
This is the shortest lag time for the calculated DDM matrix, provided in frame numbers. Typically, this will be *1*.

last_lag_time
--------------
This is the longest lag time, provided in frame numbers, for example a difference of *1000* frames. Of course, this 
cannot be larger than the number of frames in your movie. You also want to consider the fact that for long lag times, 
you will not have as much data going into the DDM matrix as you will for shorter lag times. 
 
crop_to_roi
------------
Select an region of interest in the orginal image by cropping it. Provide the pixel coordinates of the ROI in a list as follows: [y1,y2,x1,x2], 
e.g. *[0,256,0,256]*. Set to *null* to not crop and to analyze the whole image. Note that it currently requires a square ROI. 

split_into_4_rois
------------------
Split each frame into four separate ROIs. This could be handy if the orginal field of view is large and the calculation of the DDM matrix 
takes a long time to run. For example, you might have a 1024x1024 that you want to split into four 512x512 regions of interest. Give either *no* or *yes*. 
  
use_windowing_function
-----------------------
Use the windowing function, to mitgate edge effects. Either write *yes* for application or *no*, if not. 

More information: `Giavazzi, F., Edera, P., Lu, P.J. et al. Image windowing mitigates edge effects in Differential Dynamic Microscopy. Eur. Phys. J. E 40, 97 (2017). <https://link.springer.com/article/10.1140%2Fepje%2Fi2017-11587-3>`_
	

binning
--------
To bin the images, set this to *yes*. Otherwise, set as *no*. 

bin_size
---------
If binning, set to an integer value. For example, if set to *2*, then each 2x2 group of pixels will be averaged together. The resulting binned 
images will then be 2 times smaller in each dimension. 

central_angle
--------------
Set to a number to avoid radially averaging the DDM matrix over all angles. Rather, only average over a subset of angles centered on this one. 
If you do want to radially average the whole DDM matrix, then set to *null*. 
  
angle_range
------------
Set to a number to avoid radially averaging the DDM matrix over all angles. Rather, only average over a subset of angles spanning this range. 
If you do want to radially average the whole DDM matrix, then set to *null*. 

overlap_method
--------------
Use to select different methods for figuring out how many pairs of images should be used to calculate the DDM matrix for a given lag time. The options are 0, 1, 2, or 3. The default is *2*. Those correspond to:

* *0*: **Non-overlapping** image pairs will be used. For example, if the lag time is 10 frames, then differences between images 1 and 11, 11 and 21, 21 and 31, will be taken and Fourier transformed. Naturally, for long lag times, there will be few pairs of images that contribute to the DDM matrix and, therefore, one might see noisiness in the DDM matrix at these long lag times. 
* *1*: For each lag time, a maximum of XX image pairs will go into calculating the DDM matrix. By default, this number is 300. But the user may change this with the `number_differences_max` optional keyword argument passed to :py:meth:`PyDDM.ddm_analysis_and_fitting.DDM_Analysis.calculate_DDM_matrix` or specified here in this YAML file. 
* *2*: For each lag time, between images separated by the lag time, there will be ~3-4 image pairs used. So overlapping image pairs are considered but the amount of overlap is such that there will only be 3-4 pairs. For example, with a lag time of 10 frames, one might look at the image pairs 1 and 11, 4 and 14, 7 and 17, and 10 and 20. 
* *3*: For each lag time, the *maximum* number of image pairs are used. So, for example, with a lag time of 10 frames, one would consider pairs 1 and 11, 2 and 12, 3 and 13, 4 and 14, etc. 

background_method
-----------------
There are different methods for estimating the background paramater, *B*. The methods are selected by setting this parameter to 0, 1, 2, 3, 4, or 5. The default is *0*. Those correspond to:

* *0*: The average of the power spectrum of the images (not of the **differences** between images as is used to find the DDM matrix) is computed. We look at this function at the maximum 10% of wavevectors and take that value to be half the background. See Equation 6 in `Giavazzi, F., Malinverno, C., Scita, G. & Cerbino, R. Tracking-Free Determination of Single-Cell Displacements and Division Rates in Confluent Monolayers. Front. Phys. 6, (2018). <https://www.frontiersin.org/articles/10.3389/fphy.2018.00120/full>`_
* *1*: The background is taken to be the minimum of the DDM matrix.
* *2*: The background is taken to be the average (over all lag times) of the DDM matrix at the highest *q* value. 
* *3*: The background is estimated to be 0. 
* *4*: Similar to method *0* but we subtract from the power spectrum of the images, the power spectrum of the **average** image. So this method can be used if there is a signal coming from a static background. But be careful of using this method if there are objects or particles that move very little (e.g., are non-ergodic). 
* *5*: This does not assume a *q*-independent B. For each *q*, the first five data points of the DDM matrix vs lagtime are fit to a quadratic equation (second order polynomial) and the y-intercept of that function is used as *B*.

amplitude_method
-----------------
There are different methods for estimating the amplitude paramater, *A*. The methods are selected by setting this parameter to 0, 1, or 2. The default is *1*. Those correspond to:
* *0*: Like with background_method = *5*, we assume that the average of the Fourier transform squared of the images minus the Fourier transform squared of the average image is equal to 0.5*(A+B). 
* *1*: Like with background_method = *0*, we assume that the average of the Fourier transform squared of the images is equal to 0.5*(A+B). 
* *2*: Experimental. Don't use this.


Fitting_parameters
===================
Parameters about how the DDM data will fit are given here.
  
model
------
Provide the model to be used for fitting, the options are listed below:

* DDM Matrix - Single Exponential
* DDM Matrix - Double Exponential
* DDM Matrix - Exponential and Ballistic
* DDM Matrix - Ballistic
* DDM Matrix - Double Ballistic
* DDM Matrix - Single Exponential - NonErgodic
* ISF - Single Exponential
* ISF - Double Exponential
* ISF - Exponential and Ballistic
* ISF - Ballistic
* ISF - Double Ballistic
* ISF - Single Exponential - NonErgodic
* ISF - Double Exponential - NonErgodic

  
Tau
----
Decay time. Like all parameters, provide a list of three numbers corresponding to the intial guess,
the lower bound, and the upper bound. E.g., *[1.0, 0.001, 10]*. 

StretchingExp
--------------
Stretching exponential. E.g., *[1.0, 0.5, 1.1]*
 
Amplitude
---------
Amplitude. E.g., *[1e2, 1, 1e6]*
 
Background
-----------
Background. E.g., *[2.5e4, 0, 1e7]*
 
Good_q_range
------------
Range of wavevectors (provided by the indices of the list of q values) from which we can extract parameters 
like the diffusion coefficient or the velocity. Note that at low q and at high q, the data may be noisy and/or 
unreliable. We therefore usually pay attention to some middle region of wavevectors. E.g., *[5, 20]*.
 
Auto_update_good_q_range
------------------------
Set to *True* or *False*. If *True*, the range of 'good' q values will try to be determined automatically. 