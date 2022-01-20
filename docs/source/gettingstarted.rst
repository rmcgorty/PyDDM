Getting Started
===============


Installation
-------------

To use PyDDM, first download the **PyDDM** module found on 
`this GitHub repository <https://github.com/rmcgorty/PyDDM>`_.
	
This code makes use of a few Python packages. You likely already 
have most necessary packages if you installed a Python distribution 
like `Anaconda <https://www.anaconda.com/products/individual>`_. 

Some packages you are less likely to have are:

* `PyYAML <https://pyyaml.org/wiki/PyYAMLDocumentation>`_
* `xarray <https://xarray.pydata.org/en/stable/index.html>`_
* `nd2reader <https://github.com/Open-Science-Tools/nd2reader>`_


PyYAML
^^^^^^

YAML is a `"human-readable data-serialization language" <https://en.wikipedia.org/wiki/YAML>`_. 
It is often used in configuration or parameter files. In PyDDM, we use YAML to 
set the analysis and fitting parameters. Such parameters can be passed as either a 
string or text file as long as it is in the YAML format. PyYAML is a YAML parser for 
Python. 

xarray
^^^^^^

The xarray package makes working with multi-dimensional easier. One can use coordinates and 
attributes on top of array data. This helps to keep track of metadata. The DDM matrix that 
we compute and the results from fitting this matrix are all stored in xarray Datasets or 
DataArrays.

nd2reader
^^^^^^^^^

Image data can be read in either a TIFF format or nd2 format. The nd2 format is used by 
Nikon Instruments software. This nd2reader package allows us to open such files. 

