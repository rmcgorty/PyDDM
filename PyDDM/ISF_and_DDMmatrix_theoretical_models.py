# -*- coding: utf-8 -*-
"""
Mathematical models for the DDM matrix and the ISF

"""

#Created on Thu Nov  4 11:07:05 2021

#@author: RMCGORTY


import numpy as np

def dTheorySingleExp_DDM(lagtime,amplitude,tau,bg,s=1.0):
    r"""Theoretical model for the  DDM matrix with one exponential term
    
    Parameters
    ----------
    lagtime : array
        1D array of the lagtimes
    amplitude : float
        Amplitude, "A" in equation below
    tau : float
        The characteristic decay time
    bg : float
        Background term, "B" in equation below
    s : float
        Stretching exponent

    Returns
    -------
    ddm_matrix : array
        DDM matrix as shown in equation below

    Notes
    -----
    This model assumes a single exponential for the intermediate scattering 
    function. This is often used with diffusive or ballistic motion. If the 
    dynamics are subdiffusive, the stretching exponent may be less than 1. 

    .. math::
        f(q, \Delta t) = e^{\left( \frac{-\Delta t}{\tau}\right)^{s}} \\
        D(q, \Delta t) = A \times (1-f(q, \Delta t)) + B

    """
    g1 = np.exp(-1 * (lagtime / tau)**s)
    ddm_matrix = amplitude * (1 - g1) + bg
    return ddm_matrix

def dTheorySingleExp_Nonerg_DDM(lagtime,amplitude,tau,bg,s,C):
    r'''
    Theoretical model for the DDM matrix with an exponential term for the 
    intermediate scatting function. Also contains a non-ergodicity parameter. 
    With this, the ISF will decay to the non-ergodicity paramter (C), instead 
    of to zero as is the case with ergodic systems. 

    Parameters
    ----------
    lagtime : array
        1D array of the lagtimes
    amplitude : float
        Amplitude, "A" in equation below
    tau : float
        The characteristic decay time
    bg : float
        Background term, "B" in equation below
    s : float
        Stretching exponent
    C : float
        The non-ergodicity parameter

    Returns
    -------
    ddm_matrix : array
        DDM matrix as shown in equation below
        
    Notes
    -----
    .. math::
        f(q, \Delta t) = e^{\left( \frac{-\Delta t}{\tau}\right)^{s}} + C \\
        D(q, \Delta t) = A \times (1 - f(q, \Delta t)) + B
    
    A non-ergodic model was used in the paper below. [1]_
    
    References
    ----------
    .. [1] Cho, J. H., Cerbino, R. & Bischofberger, I. Emergence of Multiscale Dynamics in Colloidal Gels. Phys. Rev. Lett. 124, 088005 (2020).


    '''

    isf = ((1-C)*np.exp(-1.0*(lagtime/tau)**s)) + C
    ddm_matrix = amplitude * (1 - isf) + bg
    return ddm_matrix

def dTheoryDoubleExp_DDM(lagtime,amp,bg,f,t1,s1,t2,s2):
    r'''
    Theoretical model for the DDM matrix with two exponential terms

    :param lagtime: Time lag
    :type lagtime: float
    :param amp: Amplitude
    :type amp: float
    :param bg: Background
    :type bg: float
    :param f: Fraction with relaxation time t1, between 0-1
    :type f: float
    :param t1: Relaxation time 1
    :type t1: float
    :param s1: Stretching exponent 1
    :type s1: float
    :param t2: Relaxation time 2
    :type t2: float
    :param s2: Stretching exponent 2
    :type s2: float

    :return:
            * ddm_matrix (*float*)- DDM matrix (image structure function)

    .. math::
        f(q, \Delta t) = f \times e^{\left( \frac{-\Delta t}{\tau 1}\right)^{s1}} + (1-f) \times e^{\left( \frac{-\Delta t}{\tau 2}\right)^{s2}} \\
        D(q, \Delta t) = A \times (1-f(q, \Delta t)) + B


    '''
    isf = (f*np.exp(-1.0*(lagtime/t1)**s1)) + ((1-f)*np.exp(-1.0*(lagtime/t2)**s2))
    ddm_matrix = amp * (1 - isf) + bg
    return ddm_matrix


def dTheorySingleExp_ISF(lagtime,tau,s):
    r"""Theoretical model for the  DDM matrix with one exponential term
    
    Parameters
    ----------
    lagtime : array
        1D array of the lagtimes
    tau : float
        The characteristic decay time
    s : float
        Stretching exponent

    Returns
    -------
    ISF : array
        DDM matrix as shown in equation below

    Notes
    -----
    This model assumes a single exponential for the intermediate scattering 
    function. This is often used with diffusive or ballistic motion. If the 
    dynamics are subdiffusive, the stretching exponent may be less than 1. 
    
    If we use this model to fit our DDM data, we must be able to estimate the
    amplitude and background terms. 

    .. math::
        \\
        f(q, \Delta t) = e^{\left( \frac{-\Delta t}{\tau}\right)^{s}}
        
    """
    return np.exp(-1.0*(lagtime/tau)**s)

def dTheorySingleExp_Nonerg_ISF(lagtime,tau,s,C):
    r'''
    Theoretical model with an exponential term for the 
    intermediate scatting function. Also contains a non-ergodicity parameter. 
    With this, the ISF will decay to the non-ergodicity paramter (C), instead 
    of to zero as is the case with ergodic systems. 

    Parameters
    ----------
    lagtime : array
        1D array of the lagtimes
    tau : float
        The characteristic decay time
    s : float
        Stretching exponent
    C : float
        The non-ergodicity parameter

    Returns
    -------
    ddm_matrix : array
        DDM matrix as shown in equation below
        
    Notes
    -----
    .. math::
        f(q, \Delta t) = e^{\left( \frac{-\Delta t}{\tau}\right)^{s}} + C \\
        D(q, \Delta t) = A \times (1 - f(q, \Delta t)) + B
    
    A non-ergodic model was used in the paper below. [1]_
    
    References
    ----------
    .. [1] Cho, J. H., Cerbino, R. & Bischofberger, I. Emergence of Multiscale Dynamics in Colloidal Gels. Phys. Rev. Lett. 124, 088005 (2020).


    '''
    return ((1-C)*np.exp(-1.0*(lagtime/tau)**s)) + C


def dTheoryDoubleExp_ISF(lagtime,f,t1,s1,t2,s2):
    r'''
    lagtime: independent variable: our lag time
    f: fraction with relaxation time t1
    t1: relaxation time 1
    s1: stretching exponent
    t2: relaxation time 2
    s2: stretching exponent
    '''
    isf = (f*np.exp(-1.0*(lagtime/t1)**s1)) + ((1-f)*np.exp(-1.0*(lagtime/t2)**s2))
    return isf

###########################################################################
# Models that include a ballistic component (Schulz distributions)        #
#  along with an exponential term.                                        #
###########################################################################

def dTheoryExpAndBallistic_ISF(lagtime,tau1,s,tau2,a,Z):
    r'''Diffusive and ballistic motion
    
    This model for the ISF can be used to fit dynamics where there is diffusive 
    motion as well as ballistic motion. The ballistic component is assumed to 
    follow a Schulz distribution of velocities. This model was shown to work 
    for characterizing the motion of bacteria [1]_ . 
    
    Parameters
    ----------
    lagtime : array
        1D array of the lag times
    tau1 : float
        Characteristic decay time for the exponential term of the ISF
    s : float
        Stretching exponent for exponential term
    tau2 : float
        Characteristic decay time for ballistic component
    a : float
        Fraction of the dynamics that are ballistic
    Z : float
        Schulz distribution number
        
    Returns
    -------
    isf : array
        The intermediate scattering function
    
    References
    ----------
    .. [1] Wilson, L. G. et al. Differential Dynamic Microscopy of Bacterial Motility. *Phys. Rev. Lett.* 106, 018101 (2011). https://doi.org/10.1103/PhysRevLett.106.018101

    '''
    theta = (lagtime / tau2)/(Z + 1.0)
    VDist = ((Z + 1.0)/((Z * lagtime)/tau2)) * np.sin(Z*np.arctan(theta))/((1.0 + theta**2.0)**(Z/2.0))
    g1 = np.exp(-1.0*(lagtime/tau1)**s)
    isf = g1*((1.0-a)+a*VDist)
    return isf

def dTheoryExpAndBallistic_DDM(lagtime,amplitude,bg,tau1,s,tau2,a,Z):
    r'''Diffusive and ballistic motion
    
    This model for the DDM matrix can be used to fit dynamics where there is diffusive 
    motion as well as ballistic motion. The ballistic component is assumed to 
    follow a Schulz distribution of velocities. This model was shown to work 
    for characterizing the motion of bacteria [1]_ . 
    
    Parameters
    ----------
    lagtime : array
        1D array of the lag times
    amplitude : float
        Amplitude
    bg : float
        Background
    tau1 : float
        Characteristic decay time for the exponential term of the ISF
    s : float
        Stretching exponent for exponential term
    tau2 : float
        Characteristic decay time for ballistic component
    a : float
        Fraction of the dynamics that are ballistic
    Z : float
        Schulz distribution number
        
    Returns
    -------
    ddmmatrix : array
        The DDM matrix
    
    References
    ----------
    .. [1] Wilson, L. G. et al. Differential Dynamic Microscopy of Bacterial Motility. *Phys. Rev. Lett.* 106, 018101 (2011). https://doi.org/10.1103/PhysRevLett.106.018101

    '''
    theta = (lagtime / tau2)/(Z + 1.0)
    VDist = ((Z + 1.0)/((Z * lagtime)/tau2)) * np.sin(Z*np.arctan(theta))/((1.0 + theta**2.0)**(Z/2.0))
    g1 = np.exp(-1.0*(lagtime/tau1)**s)
    isf = g1*((1.0-a)+a*VDist)
    ddmmatrix = amplitude * (1-isf) + bg
    return ddmmatrix


###########################################################################
# Models that include a ballistic component (Schulz dist) ONLY!           #
#                                                                         #
###########################################################################

def dTheoryBallistic_ISF(lagtime,t1,Z):
    '''
    From Wilson et. al PRL 2011

    x: independent variable: our lag time

    t1: ballistic relaxation time, 1/qv
    Z: Schulz distribution number
    '''
    theta = (lagtime / t1)/(Z + 1.0)
    isf = ((Z + 1.0)/((Z * lagtime)/t1)) * np.sin(Z*np.arctan(theta))/((1.0 + theta**2.0)**(Z/2.0))
    return isf

def dTheoryBallistic_DDM(lagtime,amp,bg,t1,Z):
    '''
    From Wilson et. al PRL 2011

    x: independent variable: our lag time

    t1: ballistic relaxation time, 1/qv
    Z: Schulz distribution number
    '''
    theta = (lagtime / t1)/(Z + 1.0)
    isf = ((Z + 1.0)/((Z * lagtime)/t1)) * np.sin(Z*np.arctan(theta))/((1.0 + theta**2.0)**(Z/2.0))
    ddm_matrix = amp * (1 - isf) + bg
    return ddm_matrix

def dTheoryTwoBallistic_ISF(lagtime,t1,Z1,t2,Z2,f):
    '''
    From Wilson et. al PRL 2011

    x: independent variable: our lag time

    t1: ballistic relaxation time, 1/qv
    Z1: Schulz distribution number
    t2:
    Z2:
    f: fraction that is of type 1
    '''
    theta1 = (lagtime / t1)/(Z1 + 1.0)
    VDist1 = ((Z1 + 1.0)/((Z1 * lagtime)/t1)) * np.sin(Z1*np.arctan(theta1))/((1.0 + theta1**2.0)**(Z1/2.0))
    theta2 = (lagtime / t2)/(Z2 + 1.0)
    VDist2 = ((Z2 + 1.0)/((Z2 * lagtime)/t2)) * np.sin(Z2*np.arctan(theta2))/((1.0 + theta2**2.0)**(Z2/2.0))
    isf = (f*VDist1)+((1-f)*VDist2)
    return isf

def dTheoryTwoBallistic_DDM(lagtime,amp,bg,t1,Z1,t2,Z2,f):
    '''
    From Wilson et. al PRL 2011

    x: independent variable: our lag time

    t1: ballistic relaxation time, 1/qv
    Z1: Schulz distribution number
    t2:
    Z2:
    f: fraction that is of type 1
    '''
    theta1 = (lagtime / t1)/(Z1 + 1.0)
    VDist1 = ((Z1 + 1.0)/((Z1 * lagtime)/t1)) * np.sin(Z1*np.arctan(theta1))/((1.0 + theta1**2.0)**(Z1/2.0))
    theta2 = (lagtime / t2)/(Z2 + 1.0)
    VDist2 = ((Z2 + 1.0)/((Z2 * lagtime)/t2)) * np.sin(Z2*np.arctan(theta2))/((1.0 + theta2**2.0)**(Z2/2.0))
    isf = (f*VDist1)+((1-f)*VDist2)
    ddm_matrix = amp*(1-isf) + bg
    return ddm_matrix
