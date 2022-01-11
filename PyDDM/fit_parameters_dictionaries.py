# -*- coding: utf-8 -*-
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
    parameter_data = {}
    for parameter in parinfo:
        parameter_data[parameter['parname']] = [parameter['value'], parameter['limits'][0], parameter['limits'][1]]
    pd_data = pd.DataFrame(data=parameter_data).transpose()
    pd_data.columns = ['Initial guess', 'Minimum', 'Maximum']
    #pd_data.style.format(thousands=",")
    display(pd_data)
    return pd_data

def return_possible_fitting_models():
    model_dictionary_keys = fitting_models.keys()
    for i,model_name in enumerate(model_dictionary_keys):
        print("%i - %s" % (i+1, model_name))
        
def return_parameter_names(parameter_dictionary, print_par_names=False):
    #print(parameter_dictionary.keys())
    param_info = parameter_dictionary['parameter_info']
    param_names = []
    for i,param in enumerate(param_info):
        if print_par_names:
            print("%i - Parameter name: %s" % (i+1, param['parname']))
        param_names.append(param['parname'])
    return param_names

def set_parameter_initial_guess(parameter_dictionary, param_name, value):
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
        parameter_dictionary['parameter_info'][which_parameter]['value'] = float(value)
        print(f"Parameter '{param_name}' set to {value}.")

def set_parameter_guess_and_limits(parameter_dictionary, param_name, guess_min_max):
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
    if 'parameter_info' not in parameter_dictionary:
        print("Provided parameter dictionary did not have 'parameter_info' as a key.")
        return 0
    if type(fix)!=bool:
        print("The 'fixed' parameter must be either True or False.")
        return 0
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
    '''
    Parameters
    ----------
    parameter_dictionary : dictionary
        
    initial_guesses : array
        1-D array containing the initial guesses for the parameters

    Returns
    -------
    None.

    '''
    if 'parameter_info' in parameter_dictionary:
        number_of_parameters_in_dictionary = len(parameter_dictionary['parameter_info'])
        length_of_initial_guesses = len(initial_guesses)
        if number_of_parameters_in_dictionary != length_of_initial_guesses:
            print("Not the correct number of initial guesses for paratmers. %i initial guesses but must have %i" % (length_of_initial_guesses, number_of_parameters_in_dictionary))
            return 0
        else:
            for i,param in enumerate(parameter_dictionary['parameter_info']):
                param['value'] = initial_guesses[i]
    else:
        print("Paramter dictionary must have key of 'paramter_info'")
        return 0
    
    
def populate_min_and_max_of_paramters(parameter_dictionary, minp, maxp):
    '''
    Parameters
    ----------
    parameter_dictionary : dictionary
        
    minp : array
        1-D array containing the minimum value each parameter can take
    maxp : array
        1-D array containing the minimum value each parameter can take

    Returns
    -------
    None.

    '''
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
    '''
    

    Parameters
    ----------
    parameter_dictionary : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    parameter_values = []
    if 'parameter_info' in parameter_dictionary:
        for i,param in enumerate(parameter_dictionary['parameter_info']):
            parameter_values.append(param['value'])
    else:
        print("Paramter dictionary must have key of 'parameter_info'")
        return 0
    return np.array(parameter_values)

def extract_array_of_param_mins_maxes(parameter_dictionary):
    '''
    
    
    Parameters
    ----------
    parameter_dictionary : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
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
    '''
    
    
    Parameters
    ----------
    parameter_dictionary : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    fixed_parameters = []
    if 'parameter_info' in parameter_dictionary:
        for i,param in enumerate(parameter_dictionary['parameter_info']):
            fixed_parameters.append(param['fixed'])
    else:
        print("Paramter dictionary must have key of 'parameter_info'")
        return 0
    return np.array(fixed_parameters)