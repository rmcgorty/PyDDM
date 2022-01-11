# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:10:13 2021

@author: RMCGORTY
"""

import numpy as np #numerical python used for working with arrays, mathematical operations
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import scipy
from scipy.special import gamma

#This function is used to determine a new time when a distribution
# of decay times are present
newt = lambda t,s: (1./s)*gamma(1./s)*t

list_of_colors = ['r','b','g','m','c','k']

font_plt = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 10,
            }
font_plt_ax = {'family': 'serif',
               'color':  'black',
               'weight': 'normal',
               'size': 11,
              }

cmap = plt.get_cmap("viridis") #Others: "plasma", "cividis", "gnuplot", "jet", "rainbow", "turbo"

def view_colormap(cmap, axes=None, qminmax=None):
    colors = cmap(np.arange(cmap.N))
    if axes is None:
        fig = plt.figure(); axes = fig.gca();
    plt.yticks([])
    if qminmax is None:
        plt.xticks([])
    elif len(qminmax)==2:
        plt.xticks([0,10],["%.2f μm$^{-1}$" % qminmax[0], "%.2f μm$^{-1}$" % qminmax[1]], fontsize=8) 
    axes.imshow([colors], extent=[0,10,0,1])
    
    
def generate_pandas_table_fit_results(fit_results):
    data = {}
    data["q"] = fit_results.q
    for par in fit_results.parameter:
        data[str(par.values)] = fit_results.parameters.loc[par].values
    pd_data_frame = pd.DataFrame(data = data)
    cmap = plt.get_cmap('RdBu')
    if 'Tau' in fit_results.parameter:
        taus_over_qrange = fit_results.parameters.loc['Tau'][fit_results.good_q_range[0]:fit_results.good_q_range[1]]
        vmin = min(taus_over_qrange)
        vmax = max(taus_over_qrange)
    else:
        vmin=None
        vmax=None
    if type(fit_results.good_q_range) == list:
        left_q_range = fit_results.q.values[fit_results.good_q_range[0]]
        right_q_range = fit_results.q.values[fit_results.good_q_range[1]]
    else:
        left_q_range = None
        right_q_range = None
    return pd_data_frame.style.background_gradient(cmap, subset='Tau', axis=0)

        #\
        #.highlight_between(subset='q', left=left_q_range, right=right_q_range)
    

def plot_one_tau_vs_q(fit, ddmdata, plot_color, x_position_of_text, 
                      y_position_of_text=0.96, 
                      tau_v_q_slope = None, diffcoeff=None,
                      use_new_tau=True, fig_to_use=None, 
                      low_good_q=None, hi_good_q=None, ylim=None,
                      use_tau2=False, show_table=True):
    if fig_to_use is None:
        fig, ax = plt.subplots(nrows=1, figsize=(10,9))
    else:
        fig = fig_to_use
        ax = fig.gca()
    
    ylabel_str = 'tau (s)'
    q = fit.q
    if use_tau2 and ('Tau2' in fit.parameters.parameter):
        tau = fit.parameters.loc['Tau2',:]
        ylabel_str = 'Second tau (s)'
    elif use_tau2 and ('Tau2' not in fit.parameters.parameter):
        print("'Tau2' is not a parameter. Using 'Tau' instead.")
        tau = fit.parameters.loc['Tau']
    else:
        tau = fit.parameters.loc['Tau',:]
    if use_new_tau:
        if use_tau2 and ('StretchingExp2' in fit.parameters.parameter):
            stretch_exp = fit.parameters.loc['StretchingExp2',:]
            tau = newt(tau, stretch_exp)
        elif (not use_tau2) and ('StretchingExp' in fit.parameters.parameter):
            print("In hf.plot_one_tau_vs_q function, using new tau... ")
            stretch_exp = fit.parameters.loc['StretchingExp',:]
            tau = newt(tau, stretch_exp)
    
    ax.loglog(q[1:], tau[1:], 'o', color=plot_color)
    
    if low_good_q is None:
        lower_index_for_good_q = fit.good_q_range[0]
        if use_tau2:
            lower_index_for_good_q = fit.tau2_good_q_range[0]
    else:
        lower_index_for_good_q = low_good_q
    if hi_good_q is None:
        upper_index_for_good_q = fit.good_q_range[1]
        if use_tau2:
            upper_index_for_good_q = fit.tau2_good_q_range[1]
    else:
        upper_index_for_good_q = hi_good_q
    
    # We plot with '+' symbols where we have 'good' fits
    ax.plot(q[lower_index_for_good_q:upper_index_for_good_q],tau[lower_index_for_good_q:upper_index_for_good_q],'r+',label='good q range')
    
    # Finding the diffusion coefficient (or effective diffusion coefficient) and the scaling exponent (alpha)
    # For purely diffusive motion, alpha is equal to 1. If alpha is less than 1, then we have subdiffusion. 
    '''
    a = np.polyfit(np.log(q[lower_index_for_good_q:upper_index_for_good_q]),np.log(tau[lower_index_for_good_q:upper_index_for_good_q]), 1)
    slope = a[0]
    coef1 = np.exp(a[1])
    alpha = 2./(-1*slope)
    Dif = (1.0/coef1)**alpha
    tau_fit = coef1*(q**(-2.0/alpha))
    ax.plot(q, tau_fit, '-k')
    '''
    if (tau_v_q_slope is not None) and (diffcoeff is not None):
        alpha = 2./(-1*tau_v_q_slope)
        coef = diffcoeff**(-1./alpha)
        tau_fit = coef*(q**tau_v_q_slope)
        ax.plot(q, tau_fit, '-k')
    
    if use_tau2:
        tau_fit = (1./fit.tau2_effective_diffusion_coeff) * (q**fit.tau2_tau_vs_q_slope)
    else:
        tau_fit = (1./fit.effective_diffusion_coeff) * (q**fit.tau_vs_q_slope)
    ax.plot(q, tau_fit, '-k', lw=3, alpha=0.8)
    
    if use_tau2:
        ax.plot(q, (1./fit.tau2_velocity)*(q**-1), linestyle='dashdot')
        ax.plot(q, (1./fit.tau2_diffusion_coeff)*(q**-2), linestyle='dashed')
    else:
        ax.plot(q, (1./fit.velocity)*(q**-1), linestyle='dashdot')
        ax.plot(q, (1./fit.diffusion_coeff)*(q**-2), linestyle='dashed')
    
    ax.set_xlabel("q (μm$^{-1}$)", fontdict=font_plt_ax)
    ax.set_ylabel(ylabel_str, fontdict=font_plt_ax)
    ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))

    if ylim != None:
        ax.set_ylim(ymin=ylim[0], ymax=ylim[1])
    
    ax.tick_params(which='both', direction="in")
    ax.set_title("Decay time vs wavevector \nBallistic - dot/dash line; Diffusive - dashed line")
    
    if show_table:
        if not use_tau2:
            cellText = ["%.5f" % fit.tau_vs_q_slope]
            cellText.append("%.5f" % fit.effective_diffusion_coeff)
            cellText.append("%.5f" % fit.msd_alpha)
            cellText.append("%.5f" % fit.diffusion_coeff)
            cellText.append("%.5f" % fit.velocity)
            cols = ['Tau vs q slope', 'Effective diff coeff', 'MSD alpha', 'diffusion coeff', 'velocity']
        else:
            cellText = ["%.5f" % fit.tau2_tau_vs_q_slope]
            cellText.append("%.5f" % fit.tau2_effective_diffusion_coeff)
            cellText.append("%.5f" % fit.tau2_msd_alpha)
            cellText.append("%.5f" % fit.tau2_diffusion_coeff)
            cellText.append("%.5f" % fit.tau2_velocity)
            cols = ['Tau vs q slope', 'Effective diff coeff', 'MSD alpha', 'diffusion coeff', 'velocity']
        tables = ax.table(cellText=[cellText], colLabels=cols, bbox=[0, -0.18, 1, 0.1])
    
    return fig


def plot_taus_together(fit, ddmdata, colormap='virdris'):
    
    
    fig = plt.figure(figsize=(9,9./1.618))
    fractions=fit.parameters.loc['Fraction1',:].values
    plt.scatter(ddmdata.q, fit.parameters.loc['Tau',:],c=fractions,cmap=colormap, label='First decay time')
    plt.loglog()
    plt.ylim(fit.parameters.loc['Tau',:].min(),fit.parameters.loc['Tau',:].max()+100)
    plt.loglog(ddmdata.q, fit.parameters.loc['Tau2',:],'k+', label='Second decay time')
    cbar=plt.colorbar()
    cbar.set_label('Fraction of first exponent')
    plt.legend()
    plt.xlabel("q (μm$^-$$^1$)")
    plt.ylabel("tau (s)")
    
    plt.title("Decay time vs wavevector", fontsize=14)

    
    return fig
    


def plot_stretching_exponent(fit, ddmdata, plot_color, x_position_of_text,
                             axis_to_use=None,
                             low_good_q=None, hi_good_q=None, ylim=None,
                             use_s2=False):
    
    if axis_to_use is None:
        fig, ax = plt.subplots(nrows=1, figsize=(10,10/1.618))
    else:
        ax = axis_to_use
        
    q = fit.q
    if use_s2 and ('StretchingExp2' in fit.parameters.parameter):
        stretch_exp = fit.parameters.loc['StretchingExp2',:]
        title_str = "Second stretching exponent vs wavevector"
    else:
        stretch_exp = fit.parameters.loc['StretchingExp',:]
        title_str = "Stretching exponent vs wavevector"

    
    if low_good_q is None:
        lower_index_for_good_q = fit.good_q_range[0]
    else:
        lower_index_for_good_q = low_good_q
    if hi_good_q is None:
        upper_index_for_good_q = fit.good_q_range[1]
    else:
        upper_index_for_good_q = hi_good_q
    
    ax.semilogx(q[1:], stretch_exp[1:],'o', color=plot_color)
    ax.plot(q[lower_index_for_good_q:upper_index_for_good_q],stretch_exp[lower_index_for_good_q:upper_index_for_good_q],'r+',label='good q range')

    #Find the average stretching exponent over the q-range specified in previous code cell
    avg_stretching_exponent = stretch_exp[lower_index_for_good_q:upper_index_for_good_q].mean()
    
    plt.hlines(avg_stretching_exponent, q[lower_index_for_good_q], q[upper_index_for_good_q], linestyles='dashed')
    ax.set_xlabel("q (μm$^{-1}$)", fontdict=font_plt_ax)
    ax.set_ylabel("Stretching exponent", fontdict=font_plt_ax)
    ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.1f}'.format))
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
    ax.text(0.3,0.2,'avg stretching exp = %.2f' % avg_stretching_exponent, 
            fontdict=font_plt,horizontalalignment='left', 
            verticalalignment='center', transform=ax.transAxes)
    if ylim is not None:
        if len(ylim)==2:
            ax.set_ylim(ylim[0], ylim[1])
        else:
            print("the 'ylim' parameter needs to have length 2")
    plt.title(title_str)
    
    return fig

def plot_fraction(fit, ddmdata, color='m'):
    if ('FractionBallistic' in fit.parameter.values):
        frac = fit.parameters.loc['FractionBallistic']
        ylabel_str = "Fraction moving ballistically"
    elif ('Fraction1' in fit.parameter.values):
        frac = fit.parameters.loc['Fraction1']
        ylabel_str = "Fraction with dynamics of first time scale"
    else:
        print("No parameter called 'FractionBallistic' or 'Fraction1' found.")
        return None
    fig = plt.figure(figsize=(8,8./1.618))
    plt.semilogx(ddmdata.q[1:], frac[1:], color=color, marker='o', linestyle='')  
    plt.xlabel("q (μm$^{-1}$)")
    plt.ylabel(ylabel_str)
    plt.title("Fraction vs wavevector")
    return fig

def plot_schulz(fit, ddmdata, color='m', use2=False):
    fig = plt.figure(figsize=(8,8./1.618))
    if not use2:
        plt.loglog(ddmdata.q[1:], fit.parameters.loc['SchulzNum',:][1:], color=color, marker='o', linestyle='')
        plt.ylabel("Schulz number")
    else:
        plt.loglog(ddmdata.q[1:], fit.parameters.loc['SchulzNum2',:][1:], color=color, marker='o', linestyle='')
        plt.ylabel("Second Schulz number")
    plt.xlabel("q (μm$^{-1}$)")
    plt.title("Schulz number vs wavevector")
    return fig

def plot_background(fit, ddmdata, color='m', color2='k'):

    figB = plt.figure(figsize=(8,8./1.618))
    if ('Background' in fit.parameter.values):
        plt.semilogx(ddmdata.q[1:], fit.parameters.loc['Background',:][1:], color+'o', label="Background from fitting radial averages")  
    plt.hlines(ddmdata.B,ddmdata.q[1],ddmdata.q.max(),color2, linestyles='dashed', label="Background from direct FFT")
    plt.xlabel("q (μm$^{-1}$)")
    plt.ylabel("Background (a.u.)")
    plt.legend()
    plt.title("Background vs wavevector")
    return figB

def plot_amplitude(fit, ddmdata, color1='c', color2='k'):
    figA = plt.figure(figsize=(8,8./1.618))
    
    if ('Amplitude' in fit.parameter.values):
        plt.loglog(ddmdata.q[1:], fit.parameters.loc['Amplitude',:][1:],color1+'o', label='Amplitude from fit to DDM matrix')
    #determinded from direct FFT transforms
    plt.loglog(ddmdata.q[1:], ddmdata.Amplitude[1:], color2+'.', label='Amplitude from direct FFT')
    #plt.ylim(0,ddmdata.amplitude.max()+100)
    plt.xlabel("q (μm$^{-1}$)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()
    plt.title("Amplitude vs wavevector")
    return figA

def plot_nonerg(fit, ddmdata, plt_color='darkblue'):
    fig,axs = plt.subplots(2,1,figsize=(8,8./1.618))
    axs[0].semilogy(ddmdata.q[1:], fit.parameters.loc['NonErgodic'][1:],'o',color=plt_color)
    axs[0].set_xlabel("q (μm$^{-1}$)")
    axs[0].set_ylabel("Non-ergodicity parameter")
    axs[0].tick_params(axis="both",which="both",direction="in")
    axs[1].semilogy(ddmdata.q[1:]**2, fit.parameters.loc['NonErgodic'][1:],'o',color=plt_color)
    axs[1].set_xlabel("q$^2$ (μm$^{-2}$)")
    axs[1].set_ylabel("Non-ergodicity parameter")
    axs[1].tick_params(axis="both",which="both",direction="in")
    plt.suptitle("Non-ergodicity parameter")
    return fig


def plot_amplitude_over_background(fit, ddmdata, plt_color = 'g'):
    fig = plt.figure(figsize=(8,8./1.618))
    
    if ('Amplitude' in fit.parameter.values):
        amp = fit.parameters.loc['Amplitude',:][1:]
    else:
        amp = ddmdata.Amplitude[1:]
    if ('Background' in fit.parameter.values):
        bg = fit.parameters.loc['Background',:][1:]
    else:
        bg = np.ones_like(amp.values) * ddmdata.B.values
    
    plt.loglog(ddmdata.q[1:], amp/bg, plt_color+'o')

    plt.xlabel("q (μm$^{-1}$)")
    plt.ylabel("Amplitude / Background (a.u.)")
    plt.title("Amplitude/Background vs wavevector")
    
    return fig

def plot_to_inspect_fit(q_index_to_plot, fit, ddmdata, axis_to_use = None, ylim=None, 
                        oneplotcolor='r', show_legend=True, scale_by_q_to_power=0,
                        show_colorbar=False, print_params=True):
    if axis_to_use is None:
        fig, ax = plt.subplots(nrows=1, figsize=(10,10/1.618))
    else:
        ax = axis_to_use
    
    if type(fit) is not xr.core.dataset.Dataset:
        print("Must pass the fitting results as an xarray Dataset")
        return 0
    
    cellText = []
    
    times = fit.lagtime
    if fit.data_to_use == 'ISF':
        data = ddmdata.ISF
    elif fit.data_to_use == 'DDM Matrix':
        data = ddmdata.ravs
        
    xlabel_str = "Lag time (s)"
    ylabel_str = f"{fit.data_to_use}"
            
    if np.isscalar(q_index_to_plot):
        ax.semilogx(times, data[:,q_index_to_plot], 'o', color=oneplotcolor, label="Data for q index %i, q=%.4f μm$^{-1}$" % (q_index_to_plot, fit.q[q_index_to_plot]))
        ax.semilogx(times, fit.theory[:,q_index_to_plot], '-k', lw=3, alpha=0.8, label="Fit for q_index %i" % q_index_to_plot)
        if print_params:
            for param in fit.parameter:
                print("For q_index %i, the '%s' paramter is %.4f" % (q_index_to_plot, param.values, fit.parameters.loc[param][q_index_to_plot]))
            
        cellText.append(["%1.4f" % x for x in fit.parameters[:,q_index_to_plot]])
        cols = ["%s" % x.values for x in fit.parameter]
        rows = ["%i" % q_index_to_plot]
        tables = ax.table(cellText=cellText, rowLabels=rows, colLabels=cols, bbox=[0, -0.3, 1, 0.1])

    else:
        clrs = np.linspace(0,1,len(q_index_to_plot))
        for i,qv in enumerate(q_index_to_plot):
            plt_color = cmap(clrs[i])
            qv = int(qv)
            if scale_by_q_to_power:
                times = fit.lagtime * (fit.q[qv]**scale_by_q_to_power)
                xlabel_str = "Lag time multiplied by q to power of %.1f" % scale_by_q_to_power
            ax.semilogx(times, data[:,qv], 'o', color=plt_color, label="Data for q index %i, q=%.4f" % (qv, fit.q[qv]))
            ax.semilogx(times, fit.theory[:,qv], '-', color=plt_color, lw=3, alpha=0.8, label="Fit for q_index %i" % qv)
            ax.semilogx(times, fit.theory[:,qv], color='k', linestyle=(0, (1, 1)), lw=1)
            if print_params:
                for param in fit.parameter:
                    print("For q_index %i, the '%s' paramter is %.4f" % (qv, param.values, fit.parameters.loc[param][qv]))
                
            cellText.append(["%1.4f" % x for x in fit.parameters[:,qv]])
        cols = ["%s" % x.values for x in fit.parameter]
        rows = ["%i, %.3f μm$^{-1}$" % (qv,fit.q[int(qv)]) for qv in q_index_to_plot]
        table_height = 0.1*len(q_index_to_plot)
        tables = ax.table(cellText=cellText, rowLabels=rows, colLabels=cols, bbox=[0, -0.1-table_height, 1, table_height])
        
        if show_colorbar:
            if axis_to_use != None:
                fig = ax.get_figure()
            axes2 = fig.add_axes([0.65,0.65,0.2,0.3])
            view_colormap(cmap, axes=axes2, qminmax=[fit.q[int(q_index_to_plot[0])], fit.q[int(q_index_to_plot[-1])]])
            
                
    ax.tick_params(which='both', direction="in")
    ax.set_title(f"Fit model: {fit.model}")
    ax.set_xlabel(xlabel_str)
    ax.set_ylabel(ylabel_str)
    if show_legend:
        plt.legend()
            
    if ylim is not None:
        if len(ylim)==2:
            ax.set_ylim(ylim[0], ylim[1])
        else:
            print("the 'ylim' parameter needs to have length 2")
    
    return fig


def plot_to_inspect_fit_2x2subplot(q_index_to_plot, fit, ddmdata, ylim=None,
                                   oneplotcolor='r', print_params=True):

    if type(fit) is not xr.core.dataset.Dataset:
        print("Must pass the fitting results as an xarray Dataset.")
        return 0
    
    times = fit.lagtime
    if fit.data_to_use == 'ISF':
        data = ddmdata.ISF
        ylabel_str = "ISF"
        ISF=True
    elif fit.data_to_use == 'DDM Matrix':
        data = ddmdata.ravs
        ylabel_str = "Radially averaged DDM matrix"
        ISF=False
        
    xlabel_str = "Lag time (s)"
    
    fig= plt.figure(figsize=(12,12.))
    
    for n,q_at in enumerate(q_index_to_plot):
        if n>3:
            break
        ax= fig.add_subplot(2,2,n+1)
        ax.semilogx(times, data[:,q_at],'ro')
        ax.semilogx(times, fit.theory[:,q_at],'-k', lw=3)
        ax.set_xlabel(xlabel_str)
        ax.set_ylabel(ylabel_str)
        if ISF and (ylim is None): 
            ax.set_ylim((-0.15,1.15))
        ax.set_title('q index %i; q value %.3f μm$^{-1}$' % (q_at, fit.q[q_at]), fontsize=10)
        
        if ylim is not None:
            if len(ylim)==2:
                ax.set_ylim(ylim[0], ylim[1])
            else:
                print("The 'ylim' parameter needs to have length 2.")
                
    fig.suptitle(f"{fit.model}", fontsize=10)
    
    return fig


def get_velocity_variance(mean_velocity, schulz_number):
    '''
    Finds the *variance* of the Schulz velocity distribution.
    '''
    return (mean_velocity**2)/(schulz_number+1)

def get_schulz_dist(mean_velocity, schulz_num):
    Z = schulz_num
    f = lambda v: (v**Z / scipy.special.factorial(Z)) * (((Z+1)/mean_velocity)**(Z+1)) * np.exp(-1*(v/mean_velocity)*(Z+1))
    return f