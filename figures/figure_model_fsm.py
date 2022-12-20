"""
Figure about FSM models.
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
figData3File = 'doublel_sv_wrongpc.npz'
figData4File = 'doublel_sv_wrongpc_fsm.npz'
figData5File = 'doublel_svw_fsm.npz'
figData3FullPath = os.path.join(figDataDir, figData3File)
figData4FullPath = os.path.join(figDataDir, figData4File)
figData5FullPath = os.path.join(figDataDir, figData5File)

SAVE_FIGURE = 1
outputDir = ''
figFilename = 'plots_model_fsm.pdf'
figMergeFilename = 'merge-plots_model_fsm.pdf'

figFullPath = os.path.join(outputDir, figFilename)
figMergeFullPath = os.path.join(outputDir, figMergeFilename)
figSize = [4.0, 6.0] # In inches

labelPosX = [0.0, 0.43]   # Horiz position for panel labels
labelPosY = [0.975, 0.65, 0.32]    # Vert position for panel labels


# -- Load data --
fig3Data = np.load(figData3FullPath)
fig4Data = np.load(figData4FullPath)
fig5Data = np.load(figData5FullPath)

# -- Plot results --
fig = plt.gcf()
fig.clf()
fig.set_facecolor('w')
fig.set_size_inches(figSize)

gsMain = gridspec.GridSpec(3, 2, width_ratios=[0.5,0.5])
gsMain.update(left=0.0, right=0.969, top=0.96, bottom=0.06, wspace=0.3, hspace=0.5)

# -- Panel labels for cartoons and plots --
fig.text(labelPosX[0], labelPosY[0], 'A', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[1], labelPosY[0], 'B', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[0], labelPosY[1], 'C', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[1], labelPosY[1], 'D', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[0], labelPosY[2], 'E', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[1], labelPosY[2], 'F', fontsize=figutils.fontSizePanel, fontweight='bold')


# -- Panel: learning curves for two-layer models --
ax1 = plt.subplot(gsMain[0, 1])
ax2 = plt.subplot(gsMain[1, 1])
ax3 = plt.subplot(gsMain[2, 1])
params = {
    'fontSizeLabels': figutils.fontSizeLabels,
    'fontSizeTicks': figutils.fontSizeTicks,
    'fontSizePanel': figutils.fontSizePanel,
    'title': 'Model 3',
}
figutils.make_plots(fig, ax1, fig3Data, params)
params['title'] = 'Model 4'
figutils.make_plots(fig, ax2, fig4Data, params)
params['title'] = 'Model 5'
figutils.make_plots(fig, ax3, fig5Data, params)
    

plt.show()

if SAVE_FIGURE:
    fig.savefig(figFullPath, facecolor='none')

    # Get the data
    reader_base = PdfReader(figFullPath)
    page_base = reader_base.pages[0]

    reader = PdfReader("input_niso-crop.pdf")
    niso_box = reader.pages[0]
    
    reader = PdfReader("model4.pdf")
    model4_box = reader.pages[0]
    
    reader = PdfReader("input_naligned-crop.pdf")
    naligned_box = reader.pages[0]


    page_base.mergeScaledTranslatedPage(niso_box, scale=1, tx=0*72., ty=4.35*72.)
    page_base.mergeScaledTranslatedPage(model4_box, scale=1.15, tx=0.2*72., ty=2.4*72.)
    page_base.mergeScaledTranslatedPage(naligned_box, scale=1, tx=0*72., ty=0.3*72.)
    # Write the result back
    writer = PdfWriter()
    writer.add_page(page_base)
    with open(figMergeFullPath, "wb") as fp:
        writer.write(fp)
        print('Figure saved to {0}'.format(figMergeFullPath))

