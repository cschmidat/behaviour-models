"""
Common code across figures.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['font.family'] = 'Helvetica'
matplotlib.rcParams['svg.fonttype'] = 'none'  # To render as font rather than outlines


fontSizeLabels = 8
fontSizeTicks = 8
fontSizePanel = 12

TangoPalette = {\
    'Butter1'     : '#fce94f',\
    'Butter2'     : '#edd400',\
    'Butter3'     : '#c4a000',\
    'Chameleon1'  : '#8ae234',\
    'Chameleon2'  : '#73d216',\
    'Chameleon3'  : '#4e9a06',\
    'Orange1'     : '#fcaf3e',\
    'Orange2'     : '#f57900',\
    'Orange3'     : '#ce5c00',\
    'SkyBlue1'    : '#729fcf',\
    'SkyBlue2'    : '#3465a4',\
    'SkyBlue3'    : '#204a87',\
    'Plum1'       : '#ad7fa8',\
    'Plum2'       : '#75507b',\
    'Plum3'       : '#5c3566',\
    'Chocolate1'  : '#e9b96e',\
    'Chocolate2'  : '#c17d11',\
    'Chocolate3'  : '#8f5902',\
    'ScarletRed1' : '#ef2929',\
    'ScarletRed2' : '#cc0000',\
    'ScarletRed3' : '#a40000',\
    'Aluminium1'  : '#eeeeec',\
    'Aluminium2'  : '#d3d7cf',\
    'Aluminium3'  : '#babdb6',\
    'Aluminium4'  : '#888a85',\
    'Aluminium5'  : '#555753',\
    'Aluminium6'  : '#2e3436'
}

colors = {}
colors['activeOnly'] = TangoPalette['SkyBlue2']
colors['activePassive'] = TangoPalette['Chameleon3']
colors['passiveThenActive'] = TangoPalette['ScarletRed2']


def style_plot(ax: plt.Axes, num_it: int, params: dict) -> plt.Axes:
    """
    Generate style for Learning Rate plots.
    :param ax: Drawing axis
    :param num_it: Number of plotted iterations
    :param params: Parameter dict (see make_plots())
    :return: axis
    """
    #Remove top and right axes
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.legend(fontsize=params['fontSizeLabels'], loc='lower right')
    #Add line at 50%
    ax.hlines(50,0,num_it,color="silver", linestyles="--") 
    

    
    ax.set_xlabel("Trials", fontsize=params['fontSizeLabels'])
    ax.set_ylabel("Accuracy (%)", fontsize=params['fontSizeLabels'])
    ax.set_yticks(np.arange(30,110,10))
    
    ax.tick_params(axis='both', labelsize=params['fontSizeTicks'])
    
    ax.set_xlim((0, num_it))
    ax.set_ylim((25, 100))
    return ax


def make_plots(fig: plt.Figure, ax: plt.Axes, data: dict, params: dict):
    """
    Draw plot on the provided figure and axis objects.
    :param fig: Figure to draw to
    :param ax: Axis to draw to
    :param data: dict of three numpy arrays with accuracies
    :param params: dict with values for:
        title: Plot title
        fontSizeLabels: Legend and label font size
        fontSizeTicks: Ticks font size
        fontSizePanel: Plot title font size
    """
    num_it = data['an'].shape[-1]
    to_draw = [
    [data['an'], "A only", colors['activeOnly']],
    [data['ap'][:, ::10], "A + P", colors['activePassive']],
    [data['pta'][:, -num_it:], "P : A", colors['passiveThenActive']],
    ]
    
    for data_it, label, color in to_draw:
        data_mean = 100 * data_it.mean(axis=0)
        data_std = 100 * data_it.std(axis=0)
        #Plot mean
        ax.plot(data_mean, label=label, color=color, lw=3)
        #Make error bar
        ax.fill_between(
            np.arange(len(data_mean)),
            data_mean - data_std,
            data_mean + data_std,
            alpha=0.25,
            color=color,
        )
    #Call styling function
    ax = style_plot(ax, num_it, params)
    
    ax.set_title(params['title'], fontsize=params['fontSizePanel'])


