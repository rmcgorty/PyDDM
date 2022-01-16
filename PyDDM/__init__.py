r""" This is a python package to perform Differential Dynamic Microscopy (DDM) Analysis

The functions in :py:mod:`PyDDM.ddm_calc` can be used to compute the DDM matrix and fit 
it to various models. The different theoretical models available to fit data are in the 
module :py:mod:`PyDDM.ISF_and_DDMmatrix_theoretical_models`. Details of the fitting for 
these models are found in :py:mod:`PyDDM.fit_parameters_dictionaries`. 

Within the module :py:mod:`PyDDM.ddm_analysis_and_fitting` are a couple classes for 
make working with the code easy and organized. 

For more on the principles of DDM, please see [1]_ [2]_ [3]_ .

This code, and its earlier versions, have been used in several projects of the 
McGorty lab at the University of San Diego. [4]_ [5]_ [6]_ [7]_ [8]_


References
----------
.. [1] Cerbino, R. & Trappe, V. Differential Dynamic Microscopy: Probing Wave Vector Dependent Dynamics with a Microscope. Phys. Rev. Lett. 100, 188102 (2008)
.. [2] Wilson, L. G. et al. Differential Dynamic Microscopy of Bacterial Motility. Phys. Rev. Lett. 106, 018101 (2011).
.. [3] Germain, D., Leocmach, M. & Gibaud, T. Differential dynamic microscopy to characterize Brownian motion and bacteria motility. American Journal of Physics 84, 202–210 (2016).
.. [4] Wulstein, D. M., Regan, K. E., Robertson-Anderson, R. M. & McGorty, R. Light-sheet microscopy with digital Fourier analysis measures transport properties over large field-of-view. Opt. Express, OE 24, 20881–20894 (2016).
.. [5] Anderson, S. J. et al. Filament Rigidity Vies with Mesh Size in Determining Anomalous Diffusion in Cytoskeleton. Biomacromolecules (2019) doi:10.1021/acs.biomac.9b01057.
.. [6] Wulstein, D. M., Regan, K. E., Garamella, J., McGorty, R. J. & Robertson-Anderson, R. M. Topology-dependent anomalous dynamics of ring and linear DNA are sensitive to cytoskeleton crosslinking. Science Advances 5, eaay5912 (2019).
.. [7] Wang, J. & McGorty, R. Measuring capillary wave dynamics using differential dynamic microscopy. Soft Matter 15, 7412–7419 (2019).
.. [8] You, R. & McGorty, R. Two-color differential dynamic microscopy for capturing fast dynamics. Review of Scientific Instruments 92, 023702 (2021).



"""
import sys, os

#import ddm_calc
#import ddm_analysis_and_fitting as ddm