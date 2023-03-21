"""
Generic methods and classes used throughout.
"""

import os
import datetime
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from jaratoolbox import settings
from jaratoolbox import behavioranalysis
from jaratoolbox import extrastats








def get_slope(curveParams, xsep=0.1):
    """
    Estimate max slope of the psychometric curve.
    """
    slopePointSep = xsep
    slopePointsX = curveParams[0] + slopePointSep*np.array([-0.5, 0.5])
    slopePointsY = extrastats.psychfun(slopePointsX, *curveParams)
    slope = np.diff(slopePointsY)[0]/slopePointSep
    return slope

def plot_psychometric(possibleValues, fractionHitsEachValue, ciHitsEachValue):
    upperWhisker = ciHitsEachValue[1, :]-fractionHitsEachValue
    lowerWhisker = fractionHitsEachValue-ciHitsEachValue[0, :]
    (pline, pcaps, pbars) = plt.errorbar(possibleValues, 100*fractionHitsEachValue,
                                         yerr=[100*lowerWhisker, 100*upperWhisker], color='k')
    pdots, = plt.plot(possibleValues, 100*fractionHitsEachValue, 'o', mec='none', mfc='k', ms=8)
    plt.setp(pline, lw=2)
    ax = plt.gca()
    plt.ylim([0, 100])
    valRange = possibleValues[-1]-possibleValues[0]
    plt.xlim([possibleValues[0]-0.1*valRange, possibleValues[-1]+0.1*valRange])
    return pline, pcaps, pbars, pdots

def session_to_isodate(datestr):
    return datestr[0:4]+'-'+datestr[4:6]+'-'+datestr[6:8]

def days_around(isodate, outformat='%Y-%m-%d'):
    dateformat = '%Y-%m-%d'
    thisDay = datetime.datetime.strptime(isodate, dateformat)
    dayBefore =  thisDay - datetime.timedelta(days=1)
    dayAfter =  thisDay + datetime.timedelta(days=1)
    return (dayBefore.strftime(outformat), thisDay.strftime(outformat),
            dayAfter.strftime(outformat))

def gaussian(x, a, x0, sigma, y0):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+y0

def gaussians_mid_cross(mu1, mu2, std1, std2, amp1, amp2):
    '''
    Return the cross point between the means of the Gaussians
    https://stackoverflow.com/questions/22579434/
    python-finding-the-intersection-point-of-two-gaussian-curves
    '''
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = mu2/(std2**2) - mu1/(std1**2)
    c = mu1**2 /(2*std1**2) - mu2**2 / (2*std2**2) - np.log((std2*amp1)/(std1*amp2))
    allCrosses = np.roots([a,b,c])
    crossInd = np.logical_xor(allCrosses>mu1, allCrosses>mu2)
    midCross = float(allCrosses[crossInd])
    return midCross

