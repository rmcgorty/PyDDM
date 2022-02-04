"""
This module contains a dictionary for each of the different models for fitting
the DDM matrix or the ISF. It also contains functions for setting the initial 
guesses and bounds for the fitting parameters for these models. 

.. data:: ddm_matrix_single_exponential

    :type: dict
    
    Single exponential model for the DDM matrix. This model fits the DDM matrix
    to :math:`D(q,\Delta t) = A(q) [1 - \exp(-(\Delta t/\\tau (q))^{s(q)})] + B(q)`. The 
    parameter :math:`s(q)` is the stretching exponent. This parameter can be fixed to 1 if 
    a simple exponential function is desired. 
    
    For this dictionary, the `model_function` key is set to :py:func:`PyDDM.ISF_and_DDMmatrix_theoretical_models.dTheorySingleExp_DDM`. 
    This dictionary also contains the key `data_to_use` which is equal to 'DDM Matrix'. The key `parameter_info` is set 
    to a list of dictionaries. This is a **4** element list corresponding to the parameters :math:`A`, :math:`\\tau`, :math:`B`, and 
    :math:`s`. These are given the parameter names, respectively: 'Amplitude', 'Tau', 'Background' and 'StretchingExp'.
    
    **Note:** To use this model, set the ``model`` parameter in your yaml file to **'DDM Matrix - Single Exponential'**. Or, once 
    you have initialized :py:class:`PyDDM.ddm_analysis_and_fitting.DDM_Fit`, you can switch to this model with::
    
        my_fit_class.reload_fit_model_by_name('DDM Matrix - Single Exponential')

.. data:: ddm_matrix_single_exponential_nonerg

    :type: dict
    
    Single exponential model for the DDM matrix with a non-ergodicity parameter, :math:`C(q)`. 
    This model fits the DDM matrix
    to :math:`D(q,\Delta t) = A(q) [1 - f(q, \Delta t)] + B(q)` where the ISF is equal to 
    :math:`f(q,\Delta t) = [1 - C(q)] \exp(-(\Delta t/\\tau (q))^{s(q)}) + C(q)`. 
    
    For this dictionary, the `model_function` key is set to :py:func:`PyDDM.ISF_and_DDMmatrix_theoretical_models.dTheorySingleExp_Nonerg_DDM`. 
    This dictionary also contains the key `data_to_use` which is equal to 'DDM Matrix'. The key `parameter_info` is set 
    to a list of dictionaries. This is a **5** element list corresponding to the parameters :math:`A`, :math:`\\tau`, :math:`B`, 
    :math:`s`, and :math:`C`. These are given the parameter names, respectively: 'Amplitude', 'Tau', 'Background', 'StretchingExp', and 
    'NonErgodic'. 
    
    **Note:** To use this model, set the ``model`` parameter in your yaml file to **'DDM Matrix - Single Exponential - NonErgodic'**. Or, once 
    you have initialized :py:class:`PyDDM.ddm_analysis_and_fitting.DDM_Fit`, you can switch to this model with::
    
        my_fit_class.reload_fit_model_by_name('DDM Matrix - Single Exponential - NonErgodic')

.. data:: ddm_matrix_double_exponential

    :type: dict
    
    Double exponential model for the DDM matrix. 
    This model fits the DDM matrix
    to :math:`D(q,\Delta t) = A(q) [1 - f(q, \Delta t)] + B(q)` where the ISF is equal to 
    :math:`f(q,\Delta t) = a(q) \exp(-(\Delta t/\\tau_1 (q))^{s_1(q)}) + (1-a(q)) \exp(-(\Delta t/\\tau_2 (q))^{s_2(q)})`. Here, 
    :math:`a(q)` is the fraction of the dynamics described with the first decay time, :math:`\\tau_1(q)`.
    
    For this dictionary, the `model_function` key is set to :py:func:`PyDDM.ISF_and_DDMmatrix_theoretical_models.dTheoryDoubleExp_DDM`. 
    This dictionary also contains the key `data_to_use` which is equal to 'DDM Matrix'. The key `parameter_info` is set 
    to a list of dictionaries. This is a **7** element list corresponding to the parameters :math:`A`, :math:`B`, :math:`a`, :math:`\\tau_1`,  
    :math:`s_1`, :math:`\\tau_2`, and :math:`s_2`. These are given the parameter names, respectively: 'Amplitude', 'Background', 'Fraction1', 
    'Tau', 'StretchingExp', 'Tau2', and 'StretchingExp2'. 
    
    **Note:** To use this model, set the ``model`` parameter in your yaml file to **'DDM Matrix - Double Exponential'**. Or, once 
    you have initialized :py:class:`PyDDM.ddm_analysis_and_fitting.DDM_Fit`, you can switch to this model with::
    
        my_fit_class.reload_fit_model_by_name('DDM Matrix - Double Exponential')
    
.. data:: ddm_matrix_exponential_ballistic

    :type: dict
    
    DDM matrix is modeled with an exponential term and a term to describe ballistic motion that 
    has a distribution of velocities modeled with a Shulz distribution. The DDM matrix is
    :math:`D(q,\Delta t) = A(q) [1 - f(q, \Delta t)] + B(q)` where the ISF is equal to 
    :math:`f(q,\Delta t) = \exp(-(\Delta t/\\tau_1 (q))^{s_1(q)}) \\times [(1-a) + a V(q, \Delta t)]`. Here, 
    :math:`V = \\frac{\\tau_2 \\times (Z+1)}{Z \\times \Delta t} \\frac{Z \\times \\tan^{-1}(\\theta)}{(1+\\theta^2)^{Z/2}}` and 
    :math:`\\theta = \\frac{\Delta t}{\\tau_2 \\times (Z+1)}`. The parameter :math:`Z(q)` is referred to as the 
    Schulz number and characterizes the distribution of velocities. 
    
    For this dictionary, the `model_function` key is set to :py:func:`PyDDM.ISF_and_DDMmatrix_theoretical_models.dTheoryExpAndBallistic_DDM`. 
    This dictionary also contains the key `data_to_use` which is equal to 'DDM Matrix'. The key `parameter_info` is set 
    to a list of dictionaries. This is a **7** element list corresponding to the parameters :math:`A`, :math:`B`, :math:`\\tau_1`,  
    :math:`s`, :math:`\\tau_2`, :math:`a`, and :math:`Z`. These are given the parameter names, respectively: 'Amplitude', 'Background',  
    'Tau', 'StretchingExp', 'Tau2', 'FractionBallistic', and 'SchulzNum'. 
    
    **Note:** To use this model, set the ``model`` parameter in your yaml file to **'DDM Matrix - Exponential and Ballistic'**. Or, once 
    you have initialized :py:class:`PyDDM.ddm_analysis_and_fitting.DDM_Fit`, you can switch to this model with::
    
        my_fit_class.reload_fit_model_by_name('DDM Matrix - Exponential and Ballistic')
    
.. data:: ddm_matrix_ballistic

    :type: dict
    
    DDM matrix is modeled with a term to describe ballistic motion that 
    has a distribution of velocities modeled with a Shulz distribution. The DDM matrix is
    :math:`D(q,\Delta t) = A(q) [1 - f(q, \Delta t)] + B(q)` where the ISF is equal to 
    :math:`f(q,\Delta t) = V(q, \Delta t)]`. Here, 
    :math:`V = \\frac{\\tau \\times (Z+1)}{Z \\times \Delta t} \\frac{Z \\times \\tan^{-1}(\\theta)}{(1+\\theta^2)^{Z/2}}` and 
    :math:`\\theta = \\frac{\Delta t}{\\tau \\times (Z+1)}`. The parameter :math:`Z(q)` is referred to as the 
    Schulz number and characterizes the distribution of velocities.
    
    For this dictionary, the `model_function` key is set to :py:func:`PyDDM.ISF_and_DDMmatrix_theoretical_models.dTheoryBallistic_DDM`. 
    This dictionary also contains the key `data_to_use` which is equal to 'DDM Matrix'. The key `parameter_info` is set 
    to a list of dictionaries. This is a **4** element list corresponding to the parameters :math:`A`, :math:`B`, :math:`\\tau`,  
    and :math:`Z`. These are given the parameter names, respectively: 'Amplitude', 'Background',  
    'Tau', and 'SchulzNum'. 
    
    **Note:** To use this model, set the ``model`` parameter in your yaml file to **'DDM Matrix - Ballistic'**. Or, once 
    you have initialized :py:class:`PyDDM.ddm_analysis_and_fitting.DDM_Fit`, you can switch to this model with::
    
        my_fit_class.reload_fit_model_by_name('DDM Matrix - Ballistic')
    
.. data:: isf_single_exponential

    :type: dict
    
    Single exponential model for the intermediate scattering function (ISF). This model fits the ISF to 
    to :math:`f(q,\Delta t) = \exp(-(\Delta t/\\tau (q))^{s(q)})`. The 
    parameter :math:`s(q)` is the stretching exponent. This parameter can be fixed to 1 if 
    a simple exponential function is desired. 
    
    For this dictionary, the `model_function` key is set to :py:func:`PyDDM.ISF_and_DDMmatrix_theoretical_models.dTheorySingleExp_ISF`. 
    This dictionary also contains the key `data_to_use` which is equal to 'ISF'. The key `parameter_info` is set 
    to a list of dictionaries. This is a **2** element list corresponding to the parameters :math:`\\tau` and 
    :math:`s`. These are given the parameter names, respectively: 'Tau' and 'StretchingExp'. 
    
    **Note:** To use this model, set the ``model`` parameter in your yaml file to **'ISF - Single Exponential'**. Or, once 
    you have initialized :py:class:`PyDDM.ddm_analysis_and_fitting.DDM_Fit`, you can switch to this model with::
    
        my_fit_class.reload_fit_model_by_name('ISF - Single Exponential')
    
.. data:: isf_single_exponential_nonerg

    :type: dict
    
    Single exponential model for the intermediate scattering function (ISF) with a non-ergodicity parameter, :math:`C(q)`. 
    This model fits the ISF to :math:`f(q,\Delta t) = [1 - C(q)] \exp(-(\Delta t/\\tau (q))^{s(q)}) + C(q)`. 
    
    For this dictionary, the `model_function` key is set to :py:func:`PyDDM.ISF_and_DDMmatrix_theoretical_models.dTheorySingleExp_Nonerg_ISF`. 
    This dictionary also contains the key `data_to_use` which is equal to 'ISF'. The key `parameter_info` is set 
    to a list of dictionaries. This is a **3** element list corresponding to the parameters :math:`\\tau`,  
    :math:`s`, and :math:`C`. These are given the parameter names, respectively: 'Tau', 'StretchingExp', and 'NonErgodic'. 
    
    **Note:** To use this model, set the ``model`` parameter in your yaml file to **'ISF - Single Exponential - NonErgodic'**. Or, once 
    you have initialized :py:class:`PyDDM.ddm_analysis_and_fitting.DDM_Fit`, you can switch to this model with::
    
        my_fit_class.reload_fit_model_by_name('ISF - Single Exponential - NonErgodic')

.. data:: isf_double_exponential

    :type: dict
    
    Double exponential model for the intermediate scattering function (ISF). 
    This model fits the ISF to 
    :math:`f(q,\Delta t) = a(q) \exp(-(\Delta t/\\tau_1 (q))^{s_1(q)}) + (1-a(q)) \exp(-(\Delta t/\\tau_2 (q))^{s_2(q)})`. Here, 
    :math:`a(q)` is the fraction of the dynamics described with the first decay time, :math:`\\tau_1(q)`.
    
    For this dictionary, the `model_function` key is set to :py:func:`PyDDM.ISF_and_DDMmatrix_theoretical_models.dTheoryDoubleExp_ISF`. 
    This dictionary also contains the key `data_to_use` which is equal to 'ISF'. The key `parameter_info` is set 
    to a list of dictionaries. This is a **5** element list corresponding to the parameters :math:`a`, :math:`\\tau_1`,  
    :math:`s_1`, :math:`\\tau_2`, and :math:`s_2`. These are given the parameter names, respectively: 'Fraction1', 'Tau', 'StretchingExp', 
    'Tau2', and 'StretchingExp2'. 
    
    **Note:** To use this model, set the ``model`` parameter in your yaml file to **'ISF - Double Exponential'**. Or, once 
    you have initialized :py:class:`PyDDM.ddm_analysis_and_fitting.DDM_Fit`, you can switch to this model with::
    
        my_fit_class.reload_fit_model_by_name('ISF - Double Exponential')
    
.. data:: isf_exponential_ballistic

    :type: dict
    
    The intermediate scattering function (ISF) is modeled with an exponential term and a term to describe ballistic motion that 
    has a distribution of velocities modeled with a Shulz distribution. The ISF is equal to 
    :math:`f(q,\Delta t) = \exp(-(\Delta t/\\tau_1 (q))^{s_1(q)}) \\times [(1-a) + a V(q, \Delta t)]`. Here, 
    :math:`V = \\frac{\\tau_2 \\times (Z+1)}{Z \\times \Delta t} \\frac{Z \\times \\tan^{-1}(\\theta)}{(1+\\theta^2)^{Z/2}}` and 
    :math:`\\theta = \\frac{\Delta t}{\\tau_2 \\times (Z+1)}`. The parameter :math:`Z(q)` is referred to as the 
    Schulz number and characterizes the distribution of velocities. 
    
    For this dictionary, the `model_function` key is set to :py:func:`PyDDM.ISF_and_DDMmatrix_theoretical_models.dTheoryExpAndBallistic_ISF`. 
    This dictionary also contains the key `data_to_use` which is equal to 'ISF'. The key `parameter_info` is set 
    to a list of dictionaries. This is a **5** element list corresponding to the parameters :math:`\\tau_1`,  
    :math:`s`, :math:`\\tau_2`, :math:`a`, and :math:`Z`. These are given the parameter names, respectively:   
    'Tau', 'StretchingExp', 'Tau2', 'FractionBallistic', and 'SchulzNum'. 
    
    **Note:** To use this model, set the ``model`` parameter in your yaml file to **'ISF - Exponential and Ballistic'**. Or, once 
    you have initialized :py:class:`PyDDM.ddm_analysis_and_fitting.DDM_Fit`, you can switch to this model with::
    
        my_fit_class.reload_fit_model_by_name('ISF - Exponential and Ballistic')
    
.. data:: isf_ballistic

    :type: dict
    
    The intermediate scattering function (ISF) is modeled with a term to describe ballistic motion that 
    has a distribution of velocities modeled with a Shulz distribution. The ISF is :math:`f(q,\Delta t) = V(q, \Delta t)]`. Here, 
    :math:`V = \\frac{\\tau \\times (Z+1)}{Z \\times \Delta t} \\frac{Z \\times \\tan^{-1}(\\theta)}{(1+\\theta^2)^{Z/2}}` and 
    :math:`\\theta = \\frac{\Delta t}{\\tau \\times (Z+1)}`. The parameter :math:`Z(q)` is referred to as the 
    Schulz number and characterizes the distribution of velocities.    
    
    For this dictionary, the `model_function` key is set to :py:func:`PyDDM.ISF_and_DDMmatrix_theoretical_models.dTheoryBallistic_ISF`. 
    This dictionary also contains the key `data_to_use` which is equal to 'ISF'. The key `parameter_info` is set 
    to a list of dictionaries. This is a **2** element list corresponding to the parameters :math:`\\tau`  and 
    :math:`Z`. These are given the parameter names, respectively: 'Tau', and 'SchulzNum'. 
    
    **Note:** To use this model, set the ``model`` parameter in your yaml file to **'ISF - Ballistic'**. Or, once 
    you have initialized :py:class:`PyDDM.ddm_analysis_and_fitting.DDM_Fit`, you can switch to this model with::
    
        my_fit_class.reload_fit_model_by_name('ISF - Ballistic')
    
"""

"""
Created on Thu Nov  4 11:02:14 2021

@author: RMCGORTY

File to hold the list of dictionaries for the different fit methods
Required for mpfit
"""
import numpy as np
import pandas as pd
from IPython.display import display
import ISF_and_DDMmatrix_theoretical_models as models


###############################################################################
# For each model that you might fit to, first create a dictionary.            #
# This dictionary should have the keys:                                       #
#    'parameter_info': list of dictionaries for each parameter (mpfit style)  #
#    'model_function': function for calculating the theoretical model         #
#    'data_to_use': either 'DDM Matrix' or 'ISF'                              #
###############################################################################

ddm_matrix_single_exponential = {}
ddm_matrix_single_exponential['parameter_info'] = [
        {'n': 0, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Amplitude", 'error': 0, 'step':0},
        {'n': 1, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau", 'error': 0, 'step':0},
        {'n': 2, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Background", 'error': 0, 'step':0},
        {'n': 3, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "StretchingExp", 'error': 0, 'step':0}]
ddm_matrix_single_exponential['model_function'] = models.dTheorySingleExp_DDM
ddm_matrix_single_exponential['data_to_use'] = 'DDM Matrix'

ddm_matrix_single_exponential_nonerg = {}
ddm_matrix_single_exponential_nonerg['parameter_info'] = [
        {'n': 0, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Amplitude", 'error': 0, 'step':0},
        {'n': 1, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau", 'error': 0, 'step':0},
        {'n': 2, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Background", 'error': 0, 'step':0},
        {'n': 3, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "StretchingExp", 'error': 0, 'step':0},
        {'n': 4, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "NonErgodic", 'error': 0, 'step':0}]
ddm_matrix_single_exponential_nonerg['model_function'] = models.dTheorySingleExp_Nonerg_DDM
ddm_matrix_single_exponential_nonerg['data_to_use'] = 'DDM Matrix'

ddm_matrix_double_exponential = {}
ddm_matrix_double_exponential['parameter_info'] = [
        {'n': 0, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Amplitude", 'error': 0, 'step':0},
        {'n': 1, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Background", 'error': 0, 'step':0},
        {'n': 2, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Fraction1", 'error': 0, 'step':0},
        {'n': 3, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau", 'error': 0, 'step':0},
        {'n': 4, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "StretchingExp", 'error': 0, 'step':0},
        {'n': 5, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau2", 'error': 0, 'step':0},
        {'n': 6, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "StretchingExp2", 'error': 0, 'step':0}]
ddm_matrix_double_exponential['model_function'] = models.dTheoryDoubleExp_DDM
ddm_matrix_double_exponential['data_to_use'] = 'DDM Matrix'


ddm_matrix_exponential_ballistic = {}
ddm_matrix_exponential_ballistic['parameter_info'] = [
        {'n': 0, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Amplitude", 'error': 0, 'step':0},
        {'n': 1, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Background", 'error': 0, 'step':0},
        {'n': 2, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau", 'error': 0, 'step':0},
        {'n': 3, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "StretchingExp", 'error': 0, 'step':0},
        {'n': 4, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau2", 'error': 0, 'step':0},
        {'n': 5, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "FractionBallistic", 'error': 0, 'step':0},
        {'n': 6, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "SchulzNum", 'error': 0, 'step':0}]
ddm_matrix_exponential_ballistic['model_function'] = models.dTheoryExpAndBallistic_DDM
ddm_matrix_exponential_ballistic['data_to_use'] = 'DDM Matrix'


isf_single_exponential = {}
isf_single_exponential['parameter_info'] = [
        {'n': 0, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau", 'error': 0, 'step':0},
        {'n': 1, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "StretchingExp", 'error': 0, 'step':0}]
isf_single_exponential['model_function'] = models.dTheorySingleExp_ISF
isf_single_exponential['data_to_use'] = 'ISF'


isf_single_exponential_nonerg = {}
isf_single_exponential_nonerg['parameter_info'] = [
        {'n': 0, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau", 'error': 0, 'step':0},
        {'n': 1, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "StretchingExp", 'error': 0, 'step':0},
        {'n': 2, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "NonErgodic", 'error': 0, 'step':0}]
isf_single_exponential_nonerg['model_function'] = models.dTheorySingleExp_Nonerg_ISF
isf_single_exponential_nonerg['data_to_use'] = 'ISF'


isf_double_exponential = {}
isf_double_exponential['parameter_info'] = [
        {'n': 0, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Fraction1", 'error': 0, 'step':0},
        {'n': 1, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau", 'error': 0, 'step':0},
        {'n': 2, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "StretchingExp", 'error': 0, 'step':0},
        {'n': 3, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau2", 'error': 0, 'step':0},
        {'n': 4, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "StretchingExp2", 'error': 0, 'step':0}]
isf_double_exponential['model_function'] = models.dTheoryDoubleExp_ISF
isf_double_exponential['data_to_use'] = 'ISF'

isf_exponential_ballistic = {}
isf_exponential_ballistic['parameter_info'] = [
        {'n': 0, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau", 'error': 0, 'step':0},
        {'n': 1, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "StretchingExp", 'error': 0, 'step':0},
        {'n': 2, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau2", 'error': 0, 'step':0},
        {'n': 3, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "FractionBallistic", 'error': 0, 'step':0},
        {'n': 4, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "SchulzNum", 'error': 0, 'step':0}]
isf_exponential_ballistic['model_function'] = models.dTheoryExpAndBallistic_ISF
isf_exponential_ballistic['data_to_use'] = 'ISF'

isf_ballistic = {}
isf_ballistic['parameter_info'] = [
        {'n': 0, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau", 'error': 0, 'step':0},
        {'n': 1, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "SchulzNum", 'error': 0, 'step':0}]
isf_ballistic['model_function'] = models.dTheoryBallistic_ISF
isf_ballistic['data_to_use'] = 'ISF'

ddm_matrix_ballistic = {}
ddm_matrix_ballistic['parameter_info'] = [
        {'n': 0, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Amplitude", 'error': 0, 'step':0},
        {'n': 1, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Background", 'error': 0, 'step':0},
        {'n': 2, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau", 'error': 0, 'step':0},
        {'n': 3, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "SchulzNum", 'error': 0, 'step':0}]
ddm_matrix_ballistic['model_function'] = models.dTheoryBallistic_DDM
ddm_matrix_ballistic['data_to_use'] = 'DDM Matrix'

isf_double_ballistic = {}
isf_double_ballistic['parameter_info'] = [
        {'n': 0, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau", 'error': 0, 'step':0},
        {'n': 1, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "SchulzNum", 'error': 0, 'step':0},
        {'n': 2, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau2", 'error': 0, 'step':0},
        {'n': 3, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "SchulzNum2", 'error': 0, 'step':0},
        {'n': 4, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Fraction1", 'error': 0, 'step':0}]
isf_double_ballistic['model_function'] = models.dTheoryTwoBallistic_ISF
isf_double_ballistic['data_to_use'] = 'ISF'

ddm_matrix_double_ballistic = {}
ddm_matrix_double_ballistic['parameter_info'] = [
        {'n': 0, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Amplitude", 'error': 0, 'step':0},
        {'n': 1, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Background", 'error': 0, 'step':0},
        {'n': 2, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau", 'error': 0, 'step':0},
        {'n': 3, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "SchulzNum", 'error': 0, 'step':0},
        {'n': 4, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Tau2", 'error': 0, 'step':0},
        {'n': 5, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "SchulzNum2", 'error': 0, 'step':0},
        {'n': 6, 'value': 0, 'limits': [0,0], 'limited': [True,True],
         'fixed': False, 'parname': "Fraction1", 'error': 0, 'step':0}]
ddm_matrix_double_ballistic['model_function'] = models.dTheoryTwoBallistic_DDM
ddm_matrix_double_ballistic['data_to_use'] = 'DDM Matrix'


fitting_models = {}
fitting_models['DDM Matrix - Single Exponential'] = ddm_matrix_single_exponential
fitting_models['DDM Matrix - Double Exponential'] = ddm_matrix_double_exponential
fitting_models['DDM Matrix - Exponential and Ballistic'] = ddm_matrix_exponential_ballistic
fitting_models['DDM Matrix - Ballistic'] = ddm_matrix_ballistic
fitting_models['DDM Matrix - Double Ballistic'] = ddm_matrix_double_ballistic
fitting_models['DDM Matrix - Single Exponential - NonErgodic'] = ddm_matrix_single_exponential_nonerg

fitting_models['ISF - Single Exponential'] = isf_single_exponential
fitting_models['ISF - Double Exponential'] = isf_double_exponential
fitting_models['ISF - Exponential and Ballistic'] = isf_exponential_ballistic
fitting_models['ISF - Ballistic'] = isf_ballistic
fitting_models['ISF - Double Ballistic'] = isf_double_ballistic
fitting_models['ISF - Single Exponential - NonErgodic'] = isf_single_exponential_nonerg

###############################################################################
# Below are functions for working with the fitting.                           #
###############################################################################

def turn_parameters_into_dataframe_for_display(parinfo):
    '''
    This function takes a list of dictionaries, one dictionary per parameter
    of a model for the ISF or DDM Matrix, and converts it to a Pandas dataframe.
    
    Parameters
    ----------
    parinfo : list
        List of dictionaries. A dictionary in this list will look something like this
        {'n': 0, 'value': 0, 'limits': [0,0], 'limited': [True,True], 'fixed': False, 'parname': "Amplitude", 'error': 0, 'step':0}
        See, for example, :py:data:`fit_parameters_dictionaries.ddm_matrix_single_exponential`
    initial_guesses : array
        1-D array containing the initial guesses for the parameters

    Returns
    -------
    Pandas dataframe
    
    
    '''
    parameter_data = {}
    for parameter in parinfo:
        parameter_data[parameter['parname']] = [parameter['value'], parameter['limits'][0], parameter['limits'][1]]
    pd_data = pd.DataFrame(data=parameter_data).transpose()
    pd_data.columns = ['Initial guess', 'Minimum', 'Maximum']
    #pd_data.style.format(thousands=",")
    display(pd_data)
    return pd_data

def return_possible_fitting_models():
    r"""
    Prints a list of the different fitting models available. 

    Returns
    -------
    None.

    """
    model_dictionary_keys = fitting_models.keys()
    for i,model_name in enumerate(model_dictionary_keys):
        print("%i: '%s'" % (i+1, model_name))
        
def return_parameter_names(parameter_dictionary, print_par_names=False):
    r"""
    Returns a list of the parameter names.

    Parameters
    ----------
    parameter_dictionary : dict
        Parameter dictionary for a given model.
    print_par_names : TYPE, optional
        Prints the names of the parameters in the model. 

    Returns
    -------
    param_names : list[str]
        List of parameter names

    """
    #print(parameter_dictionary.keys())
    param_info = parameter_dictionary['parameter_info']
    param_names = []
    for i,param in enumerate(param_info):
        if print_par_names:
            print("%i - Parameter name: %s" % (i+1, param['parname']))
        param_names.append(param['parname'])
    return param_names

def set_parameter_initial_guess(parameter_dictionary, param_name, value):
    r"""
    Sets the initial guess for a parameter. 

    Parameters
    ----------
    parameter_dictionary : dict
        Parameter dictionary for a given model
    param_name : str
        Name of the parameter
    value : float
        Initial guess for the  parameter


    """
    if 'parameter_info' not in parameter_dictionary:
        print("Provided parameter dictionary did not have 'parameter_info' as a key.")
        return
    param_info = parameter_dictionary['parameter_info']
    list_of_paramnames = []
    which_parameter = -1
    for i,param in enumerate(param_info):
        list_of_paramnames.append(param['parname'])
        if param_name==param['parname']:
            which_parameter = i
    if param_name in list_of_paramnames:
        parameter_dictionary['parameter_info'][which_parameter]['value'] = float(value)
        print(f"Parameter '{param_name}' set to {value}.")

def set_parameter_guess_and_limits(parameter_dictionary, param_name, guess_min_max):
    r"""
    Sets the initial guess *and* bounds for a parameter when doing the fitting

    Parameters
    ----------
    parameter_dictionary : dict
        Parameter dictionary for a given model
    param_name : str
        Name of the parameter
    guess_min_max : array_like
        *Three* element list or array. First element corresponds to the 
        initial guess; second element corresponds to lower bound; third 
        element corresponds to upper bound.


    """
    if 'parameter_info' not in parameter_dictionary:
        print("Provided parameter dictionary did not have 'parameter_info' as a key.")
        return
    param_info = parameter_dictionary['parameter_info']
    list_of_paramnames = []
    which_parameter = -1
    for i,param in enumerate(param_info):
        list_of_paramnames.append(param['parname'])
        if param_name==param['parname']:
            which_parameter = i
    if param_name in list_of_paramnames:
        parameter_dictionary['parameter_info'][which_parameter]['value'] = float(guess_min_max[0])
        #print(f"Parameter '{param_name}' set to {guess_min_max[0]}.")
        parameter_dictionary['parameter_info'][which_parameter]['limits'][0] = float(guess_min_max[1])
        #print(f"Parameter '{param_name}' lower limit set to {guess_min_max[1]}.")
        parameter_dictionary['parameter_info'][which_parameter]['limits'][1] = float(guess_min_max[2])
        #print(f"Parameter '{param_name}' upper limit set to {guess_min_max[2]}.")
    else:
        print(f"Parameter '{param_name}' not found in model 'parameter_info'. Check for typos.")
    #turn_parameters_into_dataframe_for_display(parameter_dictionary['parameter_info'])


def set_parameter_limits(parameter_dictionary, param_name, min_max):
    r"""
    Sets the initial guess *and* bounds for a parameter when doing the fitting

    Parameters
    ----------
    parameter_dictionary : dict
        Parameter dictionary for a given model
    param_name : str
        Name of the parameter
    min_max : array_like
        *Two* element list or array. First element corresponds to lower bound; second 
        element corresponds to upper bound.

    """
    if 'parameter_info' not in parameter_dictionary:
        print("Provided parameter dictionary did not have 'parameter_info' as a key.")
        return 0
    param_info = parameter_dictionary['parameter_info']
    list_of_paramnames = []
    which_parameter = -1
    for i,param in enumerate(param_info):
        list_of_paramnames.append(param['parname'])
        if param_name==param['parname']:
            which_parameter = i
    if param_name in list_of_paramnames:
        parameter_dictionary['parameter_info'][which_parameter]['limits'][0] = float(min_max[0])
        print(f"Parameter '{param_name}' lower limit set to {min_max[0]}.")
        parameter_dictionary['parameter_info'][which_parameter]['limits'][1] = float(min_max[1])
        print(f"Parameter '{param_name}' upper limit set to {min_max[1]}.")
    else:
        print(f"Parameter '{param_name}' not found in model 'parameter_info'. Check for typos.")


def set_parameter_fixed(parameter_dictionary, param_name, fix):
    r"""
    Sets a parameter to be fixed. CURRENTLY NOT SUPPORTED!

    Parameters
    ----------
    parameter_dictionary : dict
        Parameter dictionary for a given model
    param_name : str
        Name of the parameter
    fix : bool
        If True, parameter will be fixed. If False, parameter will 
        be allowed to vary. 

    """
    if 'parameter_info' not in parameter_dictionary:
        print("Provided parameter dictionary did not have 'parameter_info' as a key.")
        return
    if type(fix)!=bool:
        print("The 'fixed' parameter must be either True or False.")
        return
    param_info = parameter_dictionary['parameter_info']
    list_of_paramnames = []
    which_parameter = -1
    for i,param in enumerate(param_info):
        list_of_paramnames.append(param['parname'])
        if param_name==param['parname']:
            which_parameter = i
    if param_name in list_of_paramnames:
        parameter_dictionary['parameter_info'][which_parameter]['fixed'] = fix
        if fix:
            print(f"Parameter '{param_name}' will be fixed.")
        else:
            print(f"Parameter '{param_name}' will be allowed to vary.")
   

def populate_intial_guesses(parameter_dictionary, initial_guesses):
    r"""
    Set the initial guess for all parameters in a parameter dictionary.     
    
    Parameters
    ----------
    parameter_dictionary : dictionary
        Parameter dictionary for a given model
    initial_guesses : array
        1-D array containing the initial guesses for the parameters

    Note
    ----
    The array (or list) `initial_guesses` must have the same number of 
    elements as the `parameter_dictionary` has in the key `parameter_info`. 

    """
    if 'parameter_info' in parameter_dictionary:
        number_of_parameters_in_dictionary = len(parameter_dictionary['parameter_info'])
        length_of_initial_guesses = len(initial_guesses)
        if number_of_parameters_in_dictionary != length_of_initial_guesses:
            print("Not the correct number of initial guesses for paratmers. %i initial guesses but must have %i" % (length_of_initial_guesses, number_of_parameters_in_dictionary))
            return 
        else:
            for i,param in enumerate(parameter_dictionary['parameter_info']):
                param['value'] = initial_guesses[i]
    else:
        print("Paramter dictionary must have key of 'paramter_info'")
        return 
    
    
def populate_min_and_max_of_paramters(parameter_dictionary, minp, maxp):
    r"""
    Set the initial guess for all parameters in a parameter dictionary.   
    
    Parameters
    ----------
    parameter_dictionary : dict
        Parameter dictionary for a given model
    minp : array
        1-D array containing the minimum value each parameter can take
    maxp : array
        1-D array containing the minimum value each parameter can take

    Note
    ----
    The arrays (or lists) `minp` and `maxp` must have the same number of 
    elements as the `parameter_dictionary` has in the key `parameter_info`. 

    """
    if 'parameter_info' in parameter_dictionary:
        number_of_parameters_in_dictionary = len(parameter_dictionary['parameter_info'])
        length_of_mins = len(minp)
        length_of_maxs = len(maxp)
        if length_of_mins != length_of_maxs:
            print("Must have equal number of minimum values and of maximum values.")
            return 0
        elif number_of_parameters_in_dictionary != length_of_mins:
            print("Not the correct number of minimum or maximum values. %i minimums but must have %i" % (length_of_mins, number_of_parameters_in_dictionary))
            return 0
        else:
            for i,param in enumerate(parameter_dictionary['parameter_info']):
                param['limits'][0] = minp[i]
                param['limits'][1] = maxp[i]
    else:
        print("Paramter dictionary must have key of 'parameter_info'")
        return 0
    
def extract_array_of_parameter_values(parameter_dictionary):
    r"""
    From a parameter dictionary, return array of the initial guesses for 
    the parameters. 

    Parameters
    ----------
    parameter_dictionary : dict
        Parameter dictionary for a given model.

    Returns
    -------
    array
        Values of the parameters.

    """
    parameter_values = []
    if 'parameter_info' in parameter_dictionary:
        for i,param in enumerate(parameter_dictionary['parameter_info']):
            parameter_values.append(param['value'])
    else:
        print("Paramter dictionary must have key of 'parameter_info'")
        return 0
    return np.array(parameter_values)

def extract_array_of_param_mins_maxes(parameter_dictionary):
    r"""
    From a parameter dictionary, return two arrays: first corresponds to 
    the lower bounds; second to upper bounds. 

    Parameters
    ----------
    parameter_dictionary : dict
        Parameter dictionary for a given model.

    Returns
    -------
    lower_bounds : array
        Lower bounds for all parameters
    upper_bounds : array
        Upper bounds for all parameters

    """
    parameter_mins = []
    parameter_maxes = []
    if 'parameter_info' in parameter_dictionary:
        for i,param in enumerate(parameter_dictionary['parameter_info']):
            parameter_mins.append(param['limits'][0])
            parameter_maxes.append(param['limits'][1])
    else:
        print("Paramter dictionary must have key of 'parameter_info'")
        return 0
    return np.array(parameter_mins), np.array(parameter_maxes)


def extract_array_of_fixed_or_not(parameter_dictionary):
    r"""
    From a parameter dicionary, return an array specifying whether 
    the parameters are fixed (True) or not (False)

    Parameters
    ----------
    parameter_dictionary : dict
        Parameter dictionary for a given model.

    Returns
    -------
    array
        Array of type bool

    """
    fixed_parameters = []
    if 'parameter_info' in parameter_dictionary:
        for i,param in enumerate(parameter_dictionary['parameter_info']):
            fixed_parameters.append(param['fixed'])
    else:
        print("Paramter dictionary must have key of 'parameter_info'")
        return 0
    return np.array(fixed_parameters)