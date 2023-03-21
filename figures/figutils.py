"""
Common code across figures.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from jaratoolbox import extraplots
from jaratoolbox import extrastats
import studyutils
import scipy
import pandas as pd
import seaborn as sns


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


def significance_stars(xRange, yPos, yLength, color='k', starMarker='*', starSize=8, starString=None, gapFactor=0.1):
    """
    xRange: 2-element list or array with x values for horizontal extent of line.
    yPos: scalar indicating vertical position of line.
    yLength: scalar indicating length of vertical ticks
    starMarker: the marker type to use (e.g., '*' or '+')
    starString: if defined, use this string instead of a marker. In this case fontsize=starSize
    """
    nStars=1  # I haven't implemented plotting more than one star.
    # plt.hold(True)  # FIXME: Use holdState
    xGap = gapFactor*nStars
    xVals = [xRange[0], xRange[0],
             np.mean(xRange)-xGap*np.diff(xRange)[0], np.nan,
             np.mean(xRange)+xGap*np.diff(xRange)[0],
             xRange[1], xRange[1]]
    yVals = [yPos-yLength, yPos, yPos, np.nan, yPos, yPos, yPos-yLength]
    hlines, = plt.plot(xVals, yVals, color=color)
    hlines.set_clip_on(False)
    xPosStar = []  # FINISH THIS! IT DOES NOT WORK WITH nStars>1
    starsXvals = np.mean(xRange)
    if starString is None:
        hs, = plt.plot(starsXvals, np.tile(yPos, nStars),
                       starMarker, mfc=color, mec='None', clip_on=False)
        hs.set_markersize(starSize)
    else:
        hs = plt.text(starsXvals, yPos, starString, fontsize=starSize,
                      va='center', ha='center', color=color, clip_on=False)
    # plt.hold(False)
    return [hs, hlines]

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
    
def psychome_plot(fig: plt.Figure, ax: plt.Axes, psych: dict, params: dict):
    """
    Make plot for psychometric curves.
    :param fig: Figure to draw to
    :param ax: Axis to draw to
    :param psych: dict of three dicts each with 'fractions' of left choices
    :param params: dict with values for:
        idx_plot: Trial for which to plot curves
        title: Plot title
        fontSizeLabels: Legend and label font size
        fontSizeTicks: Ticks font size
        fontSizePanel: Plot title font size
    """
    dataEachCond = {}

    fontSizeLabels = params['fontSizeLabels']
    fontSizeTicks = params['fontSizeTicks']
    fontSizePanel = params['fontSizePanel']
    
    colorEachCond = {}
    colorEachCond['A only'] = TangoPalette['SkyBlue2']
    colorEachCond['A + P'] = TangoPalette['Chameleon3']
    colorEachCond['P : A'] = TangoPalette['ScarletRed2']
    plt.sca(ax)
    
    eachCond = list(psych.keys())
    possibleFMslopes = list(psych['A only'].keys())
    for key in eachCond:
        idx_plot = params['idx_plot']
        if key == 'A + P':
            idx_plot *= 10
        elif key == 'P : A':
            idx_plot += 45000
        dataEachCond[key] = np.empty((len(possibleFMslopes), len(psych['A only'][possibleFMslopes[0]])))
        for slope_idx, slope in enumerate(possibleFMslopes):
            dataEachCond[key][slope_idx] = psych[key][slope][:, idx_plot]

    xPad = 0.2 * (possibleFMslopes[-1] - possibleFMslopes[0])
    fitxval = np.linspace(possibleFMslopes[0]-xPad, possibleFMslopes[-1]+xPad, 40)
    lineEachCond = []

    for indcond, cond in enumerate(eachCond):
        fractionLeftEachValue = dataEachCond[cond].mean(axis=1)
        fractionLeftSE = dataEachCond[cond].std(axis=1)
        ciLeftEachValue = np.vstack((fractionLeftEachValue+fractionLeftSE,
                                     fractionLeftEachValue-fractionLeftSE))

        # -- Fit sigmoidal curve --
        par0 = [0, 0.5, 0, 0]
        bounds = [[-np.inf, 0.08, 0, 0], [np.inf, np.inf, 0.5, 0.5]]
        curveParams, pcov = scipy.optimize.curve_fit(extrastats.psychfun, possibleFMslopes,
                                                     fractionLeftEachValue, sigma=fractionLeftSE, p0=par0, bounds=bounds)
        fityval = extrastats.psychfun(fitxval, *curveParams)
        hfit, = ax.plot(fitxval, 100*fityval, '-', linewidth=2, color=colorEachCond[cond])
        lineEachCond.append(hfit)
        (pline, pcaps, pbars, pdots) = studyutils.plot_psychometric(possibleFMslopes,
                                                                    fractionLeftEachValue,
                                                                    ciLeftEachValue)
        plt.setp(pcaps, color=colorEachCond[cond])
        plt.setp(pbars, color=colorEachCond[cond])
        plt.setp(pdots, mfc=colorEachCond[cond], mec='none', ms=6)
        pline.set_visible(False)
    plt.xlim([-0.01, 1.01])
    plt.ylabel('Model Output Left (%)', fontsize=fontSizeLabels)
    plt.xlabel('Stimulus Parameter λ', fontsize=fontSizeLabels)
    extraplots.set_ticks_fontsize(plt.gca(), fontSizeTicks)
    extraplots.boxoff(plt.gca())
    plt.legend(lineEachCond, eachCond, loc='lower right', fontsize=fontSizeLabels, frameon=False)
    ax.set_title(params['title'], fontsize=params['fontSizePanel'])

def corr_plot(fig: plt.Figure, ax: plt.Axes, data: dict, params: dict):
    """
    Make box plot for correlations.
    :param fig: Figure to draw to
    :param ax: Axis to draw to
    :param data: dict of two numpy arrays with correlation values
    :param params: dict with values for:
        idx_plot: Trial for which to plot correlations
        title: Plot title
        fontSizeLabels: Legend and label font size
        fontSizeTicks: Ticks font size
        fontSizePanel: Plot title font size
    """
    plt.sca(ax)
    idx_plot = params['idx_plot']
    corr_pta = data['A + P'][:, ::10][:, idx_plot]
    corr_ap = data['P : A'][:, -5000:][:, idx_plot]
    df_corr = pd.DataFrame({'A + P': corr_ap, 'P : A': corr_pta})
    palette_box = {
        'A + P': TangoPalette['Chameleon2'],
        'P : A': TangoPalette['ScarletRed1']
    }
    palette_swarm = {
        'A + P': TangoPalette['Chameleon3'],
        'P : A': TangoPalette['ScarletRed3']
    }
    sns.boxplot(data=df_corr, palette=palette_box, showfliers=False, width=0.4, ax=ax, linewidth=0.8)
    plt.ylabel('$\cos(α)$', fontsize=params['fontSizeLabels'])
    sns.swarmplot(data=df_corr, palette=palette_swarm, ax=ax, size=2)
    sns.despine(ax=ax)
    hs, hl = significance_stars([0, 1], 0.8, yLength=0.01, gapFactor=0.2, starSize=6)
    plt.setp(hl, lw=0.75)
    plt.yticks(np.arange(-0.4,1,0.2))
    extraplots.set_ticks_fontsize(plt.gca(), params['fontSizeTicks'])
    ax.set_title(params['title'], fontsize=params['fontSizePanel'])