Usage
=========


Installation
-------------

To use PyDDM, first download the following modules:
	
	**PyDDM**
	
	This contains two modules which are relevant for usage: 

	* ddm_analysis
	* ddm_fit
	

Enter analysis information
---------------------------

Make a file with information for analysis using YAML file. 

YAML is a data serialization language, more information: `YAML_website <https://yaml.org/>`_

Use default provided :download:`Example_YAML_file <../example_data.yml>`

.. toctree::
	:maxdepth: 1

	Provide_info_for_analysis


Ready to analyse
-----------------
The analysis is performed in two steps (reflected by the two Pythonmodules):

	#. DDM analysis to calculate the DDM matrix and ISF
	#. Fitting with multiple mathematical models

Initate instance of class

.. code-block:: console

	$ ddm_class=ddm_analysis('example.yaml')