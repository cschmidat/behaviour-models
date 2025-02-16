"""
Figure about one-layer model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import figutils
from PyPDF2 import PdfReader, PdfWriter, Transformation
# If you change figutils often, and want changes reloaded, uncomment lines below:
#from importlib import reload
#reload(figutils)


# NOTE: We generally have the path set by things we can import, so people don't have to
#       change the script when running it on a different computer.
#figDataDir = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME, FIGNAME)
figDataDir = '../sim_data/' 
figDataFile = 'onel.npz'
figDataFullPath = os.path.join(figDataDir, figDataFile)

SAVE_FIGURE = 1
outputDir = ''
figFilename = 'plots_model_one_layer.pdf'
figMergeFilename = 'merge-plots_model_one_layer.pdf'

figFullPath = os.path.join(outputDir, figFilename)
figMergeFullPath = os.path.join(outputDir, figMergeFilename)
figSize = [4, 2.5] # In inches

labelPosX = [0.0, 0.32]   # Horiz position for panel labels
labelPosY = [0.95, 0.48]    # Vert position for panel labels


# -- Load data --
figData = np.load(figDataFullPath)

# -- Plot results --
fig = plt.gcf()
fig.clf()
fig.set_facecolor('w')
fig.set_size_inches(figSize)

gsMain = gridspec.GridSpec(1, 2, width_ratios=[0.3,0.7])
gsMain.update(left=0.0, right=0.97, top=0.9, bottom=0.15, wspace=0.4, hspace=0.3)

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
figutils.make_plots(fig, ax1, figData, params)
    

plt.show()

if SAVE_FIGURE:
    fig.savefig(figFullPath, facecolor='none')

    # Get the data
    reader_base = PdfReader(figFullPath)
    page_base = reader_base.pages[0]

    reader = PdfReader("input_new.pdf")
    input_box = reader.pages[0]

    reader = PdfReader("model_onel.pdf")
    model_box = reader.pages[0]

    page_base.mergeScaledTranslatedPage(input_box, scale=1, tx=0*72., ty=0*72.)
    page_base.mergeScaledTranslatedPage(model_box, scale=1, tx=0.15*72., ty=1.4*72.)
    # Write the result back
    writer = PdfWriter()
    writer.add_page(page_base)
    with open(figMergeFullPath, "wb") as fp:
        writer.write(fp)
        print('Figure saved to {0}'.format(figMergeFullPath))

