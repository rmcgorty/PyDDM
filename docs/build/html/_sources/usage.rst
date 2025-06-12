Usage
=========


Python Environment
------------------

It is recommended that you work with `Jupyter Notebook <https://jupyter.org/>`_ to 
go through the DDM analysis. The 'walkthroughs' and 'tutorials' here will be done 
in a Jupyter Notebook (these notebooks can be downloaded from the `GitHub repository <https://github.com/rmcgorty/PyDDM>`_).
	

Setting the Analysis Parameters
-------------------------------

Parameters such as the frame rate, pixel size, and file path to the data are all 
passed to the DDM functions using YAML. Image pre-processing can be done based off 
parameters such as ``binning`` and ``crop_to_roi``. Calculation of the DDM matrix requires 
specification of how many lag times to calculate and over what range. Fitting the DDM 
matrix requires a specified model and initial guesses for all parameters. These details and more can 
be included in a single YAML file. 

You can find an example YAML file for DDM analysis :download:`here <../example_data.yml>`.

More details on what goes into the YAML parameters file can be found here.

.. toctree::
	:maxdepth: 1

	What metadata to provide in the YAML file <Provide_info_for_analysis>


Analyzing your Data
-------------------

Working through the 'walkthroughs' is the best way to get acquainted  with the code and 
how to use it. Example data is provided in the GitHub repository which will allow you to 
follow along with the walkthrough and play around with it. 

But for a quick introduction, the analysis begins with importing the PyDDM code, 
initializing  the :py:class:`PyDDM.ddm_analysis_and_fitting.DDM_Analysis` class to load the 
data and :doc:`metadata <Provide_info_for_analysis>`, and running the :py:meth:`PyDDM.ddm_analysis_and_fitting.DDM_Analysis.calculate_DDM_matrix` 
method. Moving on to fitting of the DDM matrix, one then initializes the 
:py:class:`PyDDM.ddm_analysis_and_fitting.DDM_Fit` class and runs the 
:py:meth:`PyDDM.ddm_analysis_and_fitting.DDM_Fit.fit` method. 

.. code-block:: python

	import ddm_analysis_and_fitting as ddm
	ddm_calc = ddm.DDM_Analysis("my_ddm_parameters.yml")
	ddm_calc.calculate_DDM_matrix()
	ddm_fit = ddm.DDM_Fit(ddm_calc.data_yaml)
	fit01 = ddm_fit.fit()