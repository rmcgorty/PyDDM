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
In the image below the application of the parameters is illustrated

.. image:: ../Explanation_analysis_parameters.PNG
  
starting_frame_number
----------------------
The number of the first frame that should be analyzed, e.g., *0* if you want to analyze 
the movie from the beginning. If the first frame or two are bad, then you can adjust this 
parameter. You could also adjust if you want to break up a long movie into segments in time for analysis. 
  
ending_frame_number
--------------------
The last frame to be analyzed, e.g., *3000*. If the last frame of the movie is the last frame for analysis, give *null*
 
number_lagtimes
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