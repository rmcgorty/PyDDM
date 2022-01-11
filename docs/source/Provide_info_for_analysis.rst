Parameters for DDM analysis
****************************


Here is an explanation for all the entries which are in the iniation file needed for the DDM analysis. 

.. contents:: :local:

DataDirectory
=============
Provide the full link to the location of the movie to be analyzed

	*'C:/Users/User_name/Folder/Subfolder/'*
	
FileName
========
Provide the full name of the file including the extention (either *.tif* or *.nd2*)

	*'images_nobin_40x_128x128_8bit.tif'*
	
Metadata
========
pixel_size
-----------
	Give the pixel size in μm, for example: *0.242*

frame_rate
------------
	Provide the number of frames per second, e.g. *41.7*


Analysis_parameters
====================
In the image below the application of the parameters is illustrated

.. image:: ../Explanation_analysis_parameters.PNG
  
starting_frame_number
----------------------
	The number of the first frame that should be analyzed, e.g. *0*
  
ending_frame_number
--------------------
	The last frame to be analyzed, for example *3000*. If the last frame of the movie is the last frame for analysis, give: *null*
 
number_lagtimes
----------------
	The number of lagtimes to be samples  *40*
 
first_lag_time
---------------
	This is the shortest lagtime, provided in frame numbers, for example a difference of *1* frame

last_lag_time
--------------
	This is the longest lagtime, provided in frame numbers, for example a difference of *1000* frames
 
crop_to_roi
------------
	Select an region of interest in the orginal file by cropping it. Provide the pixel coordinates of the ROI in a list as follows: [y1,y2,x1,x2] 
	e.g. *[0,250,0,250]*
	Set to *null* for no cropping. 

split_into_4_rois
------------------
	Split each frame into four sperate ROIs. This could be handy if the orginal field of view is large and the calculation of the DDM matrix takes too long in that form 
	give either *no* or *yes*. 
  
use_windowing_function
-----------------------
	Use the windowing function, to mitgate edge effects. Either write *yes* for application or *no*, if not. 
	
	More information: `Giavazzi, F., Edera, P., Lu, P.J. et al. Image windowing mitigates edge effects in Differential Dynamic Microscopy. Eur. Phys. J. E 40, 97 (2017). <https://link.springer.com/article/10.1140%2Fepje%2Fi2017-11587-3>`_
	

binning
--------
yes

bin_size
---------
2 

central_angle
--------------
null
  
angle_range
------------
null

Fitting_parameters
===================
  
model
------
Provide the model to be used for fitting, the options are listed below:

* - DDM Matrix - Single Exponential
* - DDM Matrix - Double Exponential
* - DDM Matrix - Exponential and Ballistic
* - DDM Matrix - Ballistic
* - DDM Matrix - Double Ballistic
* - DDM Matrix - Single Exponential - NonErgodic
* - ISF - Single Exponential
* - ISF - Double Exponential
* - ISF - Exponential and Ballistic
* - ISF - Ballistic
* - ISF - Double Ballistic
* - ISF - Single Exponential - NonErgodic

  
Tau
----
[1.0, 0.001, 10]

StretchingExp
--------------
[1.0, 0.5, 1.1]
 
Amplitude
----------- 
[1e2, 1, 1e6]
 
Background
-----------
[2.5e4, 0, 1e7]
 
Good_q_range
-------------
[5, 20]
 
Auto_update_good_q_range
-------------------------
True