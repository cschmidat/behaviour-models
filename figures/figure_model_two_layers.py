"""
Figure about two-layer models.
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
figData2File = 'doublel_sw.npz'
figData3File = 'doublel_sv.npz'
figData2FullPath = os.path.join(figDataDir, figData2File)
figData3FullPath = os.path.join(figDataDir, figData3File)

SAVE_FIGURE = 1
outputDir = ''
figFilename = 'plots_model_two_layer.pdf'
figMergeFilename = 'merge-plots_model_two_layer.pdf'

figFullPath = os.path.join(outputDir, figFilename)
figMergeFullPath = os.path.join(outputDir, figMergeFilename)
figSize = [7.8, 4.0] # In inches

labelPosX = [0.0, 0.16, 0.7]   # Horiz position for panel labels
labelPosY_right = [0.95, 0.48]    # Vert position for right panel labels
labelPosY_left = [0.95, 0.48-0.05]    # Vert position for left panel labels

ModelPosX = 0.03 #Horiz position of 'Model' label

# -- Load data --
fig2Data = np.load(figData2FullPath)
fig3Data = np.load(figData3FullPath)

# -- Plot results --
fig = plt.gcf()
fig.clf()
fig.set_facecolor('w')
fig.set_size_inches(figSize)

gsMain = gridspec.GridSpec(2, 2, width_ratios=[0.73,0.27])
gsMain.update(left=0.0, right=0.98, top=0.95, bottom=0.1, wspace=0.4, hspace=0.3)

# -- Panel labels for cartoons and plots --
fig.text(labelPosX[0], labelPosY_left[0], 'A', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[1], labelPosY_left[0], 'B', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[2], labelPosY_right[0], 'C', fontsize=figutils.fontSizePanel, fontweight='bold')

fig.text(labelPosX[0], labelPosY_left[1], 'D', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[1], labelPosY_left[1], 'E', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[2], labelPosY_right[1], 'F', fontsize=figutils.fontSizePanel, fontweight='bold')

fig.text(ModelPosX, labelPosY_left[0], "Model 2", fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(ModelPosX, labelPosY_left[1], "Model 3", fontsize=figutils.fontSizePanel, fontweight='bold')

# -- Panel: learning curves for two-layer models --
ax1 = plt.subplot(gsMain[0, 1])
ax2 = plt.subplot(gsMain[1, 1])
params = {
    'fontSizeLabels': figutils.fontSizeLabels,
    'fontSizeTicks': figutils.fontSizeTicks,
    'fontSizePanel': figutils.fontSizePanel,
    'title': '',
}
figutils.make_plots(fig, ax1, fig2Data, params)
figutils.make_plots(fig, ax2, fig3Data, params)

    

plt.show()

if SAVE_FIGURE:
    fig.savefig(figFullPath, facecolor='none')

    # Get the data
    reader_base = PdfReader(figFullPath)
    page_base = reader_base.pages[0]

    reader = PdfReader("model_two_dynamics-crop.pdf")
    model2dyn_box = reader.pages[0]
    
    reader = PdfReader("model2.pdf")
    model2_box = reader.pages[0]
    
    reader = PdfReader("model_three_dynamics-crop.pdf")
    model3dyn_box = reader.pages[0]

    reader = PdfReader("model3.pdf")
    model3_box = reader.pages[0]

    page_base.mergeScaledTranslatedPage(model2dyn_box, scale=1.15, tx=1.2*72., ty=2.3*72.)
    page_base.mergeScaledTranslatedPage(model3dyn_box, scale=1.15, tx=1.2*72., ty=0.3*72.)
    page_base.mergeScaledTranslatedPage(model2_box, scale=1.15, tx=0*72., ty=2.3*72.)
    page_base.mergeScaledTranslatedPage(model3_box, scale=1.15, tx=0*72., ty=0.3*72.)
    # Write the result back
    writer = PdfWriter()
    writer.add_page(page_base)
    with open(figMergeFullPath, "wb") as fp:
        writer.write(fp)
        print('Figure saved to {0}'.format(figMergeFullPath))

