"""
Figure about one-layer model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import figutils

# If you want changes in figutils to reload, uncomment these lines:
#from importlib import reload
#reload(figutils)


# NOTE: We generally have the path set by things we can import, so people don't have to
#       change the script when running it on a different computer.
#figDataDir = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME, FIGNAME)
figDataDir = '../sim_data/' 
figDataFile = 'onel.npz'
figDataFullPath = os.path.join(figDataDir, figDataFile)

SAVE_FIGURE = 1
outputDir = '/tmp/'
figFilename = 'plots_model_one_layer' # Do not include extension
figFormat = 'pdf' # 'pdf' or 'svg'
figSize = [4, 2.5] # In inches

labelPosX = [0.03, 0.33]   # Horiz position for panel labels
labelPosY = [0.92, 0.48]    # Vert position for panel labels


# -- Load data --
#figData = np.load(figDataFullPath)

# -- Plot results --
fig = plt.gcf()
fig.clf()
fig.set_facecolor('w')

gsMain = gridspec.GridSpec(1, 2, width_ratios=[0.3,0.7])
gsMain.update(left=0.15, right=0.96, top=0.9, bottom=0.15, wspace=0.2, hspace=0.3)

# -- Panel labels for cartoons and plots --
fig.text(labelPosX[0], labelPosY[0], 'A', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[0], labelPosY[1], 'B', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[1], labelPosY[0], 'C', fontsize=figutils.fontSizePanel, fontweight='bold')

# -- Panel: learning curves for one-layer model --
ax1 = plt.subplot(gsMain[0, 1])
params = {
    'fontSizeLabels': figutils.fontSizeLabels,
    'fontSizeTicks': figutils.fontSizeTicks,
    'fontSizePanel': figutils.fontSizePanel,
    'title': '',
}
figutils.make_plots(fig, ax1, figDataFullPath, params)
    

plt.show()

if SAVE_FIGURE:
    figutils.save_figure(figFilename, figFormat, figSize, outputDir)
