.. DDM_class_based_framework documentation master file, created by
   sphinx-quickstart on Wed Nov 10 16:49:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyDDM's documentation!
=====================================================

**PyDDM**  is a Python package for analysis of differential dynamic microscopy (DDM) data. 


.. note::

   This project is under active development.


.. toctree::
   :maxdepth: 1
   
   gettingstarted
   usage
   Basic walkthrough <Walkthrough - simple>
   Advanced walkthrough <Walkthrough - details>
   Adding a new DDM model <Walkthrough - creating a new DDM fitting function>

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   PyDDM


Note the two **walkthroughs**. The Jupyter Notebook files for both of these can be found in the `examples folder`_ of the GitHub PyDDM repository. They both use the same sample data of diffusing colloidal particles. The first walkthrough goes over the basics of using this code. In the second walkthrough, some additional features of the code are discussed. Importantly, the second walkthrough also goes over a few aspects of DDM that one should pay attention to: finding the background parameter and considering the range of wavevectors over which the data is reliable. It also briefly goes over how to slice and extract data from xarray Datasets. More on that can be `found here.`_ 

We also have instructions on how to add a new fitting function. This can be useful if you have a model for the intermediate scattering function that you want to employ.

.. _found here.: https://xarray.pydata.org/en/stable/user-guide/indexing.html

.. _examples folder: https://github.com/rmcgorty/PyDDM/tree/main/Examples


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
