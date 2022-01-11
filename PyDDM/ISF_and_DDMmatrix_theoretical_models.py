# -*- coding: utf-8 -*-
"""

Mathematical models for the DDM matrix and the ISF

"""

#Created on Thu Nov  4 11:07:05 2021

#@author: RMCGORTY


import numpy as np

def dTheorySingleExp_DDM(lagtime,a1,t1,bg,s1=1.0):
    r"""
    Theoretical model for the  DDM matrix with one exponential term

    :param lagtime: Time lag
    :type lagtime: float
    :param a1: Amplitude for first exponential term
    :type a1: float
    :param t1: Decay time for first exponential term
    :type t1: float
    :param b1: Background
    :type b1: float
    :param s1: Stretching exponent for first exponetial term
    :type s1: float

    :return:
            * ddm_matrix (*float*)- DDM matrix (image structure function)

    .. math::
        \\
        f(q, \Delta t) = e^{\left( \frac{-\Delta t}{\tau}\right)^{s}} \\
        D(q, \Delta t) = A \times (1-f(q, \Delta t)) + B

    """
    g1 = np.exp(-1 * (lagtime / t1)**s1)
    ddm_matrix = a1 * (1 - g1) + bg
    return ddm_matrix

def dTheorySingleExp_Nonerg_DDM(lagtime,a1,t,bg,s,c):
    r'''
    Theoretical model for the DDM matrix with one exponential term for non ergodic systems

    :param lagtime: Time lag
    :type lagtime: float
    :param t: relaxation time (tau)
    :type t: float
    :param s: Stretching exponent  ("p" in Cho et al 2020)
    :type s: float
    :param c: non-ergodicity parameter
    :type c: float

    :return:
            * ddm_matrix (*float*)- DDM matrix (image structure function)

    .. math::

        f(q, \Delta t) = e^{\left( \frac{-\Delta t}{\tau}\right)^{s}} + C \\
        D(q, \Delta t) = A \times (1-f(q, \Delta t)) + B

    '''

    isf = ((1-c)*np.exp(-1.0*(lagtime/t)**s)) + c
    ddm_matrix = a1 * (1 - isf) + bg
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
        \\
        f(q, \Delta t) = f \times e^{\left( \frac{-\Delta t}{\tau 1}\right)^{s1}} + (1-f) \times e^{\left( \frac{-\Delta t}{\tau 2}\right)^{s2}} \\
        D(q, \Delta t) = A \times (1-f(q, \Delta t)) + B


    '''
    isf = (f*np.exp(-1.0*(lagtime/t1)**s1)) + ((1-f)*np.exp(-1.0*(lagtime/t2)**s2))
    ddm_matrix = amp * (1 - isf) + bg
    return ddm_matrix


def dTheorySingleExp_ISF(lagtime,t,s):
    r'''
    Theoretical model for the ISF with one exponential term

    :param lagtime: Time lag
    :type lagtime: float
    :param t: relaxation time (tau)
    :type t: float
    :param s: Stretching exponent  ("p" in Cho et al 2020)
    :type s: float

    :return:
            * ISF equation (*float*)- ISF (intermediate scattering function)

    .. math::
        \\
        f(q, \Delta t) = e^{\left( \frac{-\Delta t}{\tau}\right)^{s}}
        


    '''
    return np.exp(-1.0*(lagtime/t)**s)

def dTheorySingleExp_Nonerg_ISF(lagtime,t,s,c):
    r'''
    Theoretical model for the ISF with one exponential term

    :param lagtime: Time lag
    :type lagtime: float
    :param t: relaxation time (tau)
    :type t: float
    :param s: Stretching exponent  ("p" in Cho et al 2020)
    :type s: float
    :param c: non-ergodicity parameter
    :type c: float

    :return:
            * ISF equation (*float*)- ISF (intermediate scattering function)
    .. math::
        \\
        f(q, \Delta t) = e^{\left( \frac{-\Delta t}{\tau}\right)^{s}} + C


    '''
    return ((1-c)*np.exp(-1.0*(lagtime/t)**s)) + c


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

def dTheoryExpAndBallistic_ISF(lagtime,t1,s,t2,a,Z):
    '''
    From Wilson et. al PRL 2011

    x: independent variable: our lag time
    t1: subdiff. relaxation time (tau)
    s: stretching exponent ("p" in Cho et al 2020)
    t2: ballistic relaxation time, 1/qv
    a: proportion of population that is moving ballistically
    Z: Schulz distribution number
    '''
    theta = (lagtime / t2)/(Z + 1.0)
    VDist = ((Z + 1.0)/((Z * lagtime)/t2)) * np.sin(Z*np.arctan(theta))/((1.0 + theta**2.0)**(Z/2.0))
    g1 = np.exp(-1.0*(lagtime/t1)**s)
    isf = g1*((1.0-a)+a*VDist)
    return isf

def dTheoryExpAndBallistic_DDM(lagtime,amp,bg,t1,s,t2,a,Z):
    '''
    From Wilson et. al PRL 2011

    lagtime: independent variable: our lag time
    amp: amplitude
    bg: background
    t1: subdiff. relaxation time (tau)
    s: stretching exponent ("p" in Cho et al 2020)
    t2: ballistic relaxation time, 1/qv
    a: proportion of population that is moving ballistically
    Z: Schulz distribution number
    '''
    theta = (lagtime / t2)/(Z + 1.0)
    VDist = ((Z + 1.0)/((Z * lagtime)/t2)) * np.sin(Z*np.arctan(theta))/((1.0 + theta**2.0)**(Z/2.0))
    g1 = np.exp(-1.0*(lagtime/t1)**s)
    isf = g1*((1.0-a)+a*VDist)
    ddmmatrix = amp * (1-isf) + bg
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
