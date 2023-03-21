"""
Figure about FSM models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import figutils
from PyPDF2 import PdfReader, PdfWriter, Transformation
import pickle
# If you change figutils often, and want changes reloaded, uncomment lines below:
#from importlib import reload
#reload(figutils)


# NOTE: We generally have the path set by things we can import, so people don't have to
#       change the script when running it on a different computer.
#figDataDir = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME, FIGNAME)
figDataDir = '../sim_data/' 
figData1File = 'doublel_svw_fsm.npz'
figData2AFile = 'corr_fsm_ap.npy'
figData2BFile = 'corr_fsm_pta.npy'
figData3AFile = 'psy_an_doublel_svw_fsm.npz'
figData3BFile = 'psy_ap_doublel_svw_fsm.npz'
figData3CFile = 'psy_pta_doublel_svw_fsm.npz'
figData1FullPath = os.path.join(figDataDir, figData1File)
figData2AFullPath = os.path.join(figDataDir, figData2AFile)
figData2BFullPath = os.path.join(figDataDir, figData2BFile)
figData3AFullPath = os.path.join(figDataDir, figData3AFile)
figData3BFullPath = os.path.join(figDataDir, figData3BFile)
figData3CFullPath = os.path.join(figDataDir, figData3CFile)

def extract_dict(npz):
    """
    Extract dict item from npz file.
    """
    return npz[npz.files[0]][()]

SAVE_FIGURE = 1
outputDir = ''
figFilename = 'plots_model_last.pdf'
figMergeFilename = 'merge-plots_model_last.pdf'

figFullPath = os.path.join(outputDir, figFilename)
figMergeFullPath = os.path.join(outputDir, figMergeFilename)
figSize = [6.0, 7.0] # In inches

labelPosX = [0.0, 0.54]   # Horiz position for panel labels
labelPosY = [0.975, 0.65, 0.32]    # Vert position for panel labels


# -- Load data --
fig1Data = np.load(figData1FullPath)
fig2AData = np.load(figData2AFullPath)
fig2BData = np.load(figData2BFullPath)
fig2Data = {'A + P': fig2AData, 'P : A': fig2BData}
fig3AData = extract_dict(np.load(figData3AFullPath, allow_pickle=True))
fig3BData = extract_dict(np.load(figData3BFullPath, allow_pickle=True))
fig3CData = extract_dict(np.load(figData3CFullPath, allow_pickle=True))
fig3Data = { 'A only': fig3AData, 'A + P': fig3BData, 'P : A': fig3CData,}
# -- Plot results --
fig = plt.gcf()
fig.clf()
fig.set_facecolor('w')
fig.set_size_inches(figSize)

gsMain = gridspec.GridSpec(3, 2, width_ratios=[0.5,0.5])
gsMain.update(left=0.08, right=0.969, top=0.96, bottom=0.06, wspace=0.5, hspace=0.5)

# -- Panel labels for cartoons and plots --
fig.text(labelPosX[0], labelPosY[0], 'A', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[0]+0.2, labelPosY[0], 'B', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[1], labelPosY[0], 'C', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[0], labelPosY[1]+0.05, 'D', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[1], labelPosY[1], 'E', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[0], labelPosY[2], 'F', fontsize=figutils.fontSizePanel, fontweight='bold')
fig.text(labelPosX[1], labelPosY[2], 'G', fontsize=figutils.fontSizePanel, fontweight='bold')

fig.text(0.085, 0.79, 'UL+\nSL', fontsize=figutils.fontSizeLabels, fontweight='bold')
fig.text(0.155, 0.835, 'SL', fontsize=figutils.fontSizeLabels, fontweight='bold')

fig.text(0.150, 0.710, 'active', fontsize=figutils.fontSizeLabels, fontweight='bold')
fig.text(0.160, 0.640, 'passive', fontsize=figutils.fontSizeLabels, fontweight='bold')

fig.text(0.150, 0.510, 'active', fontsize=figutils.fontSizeLabels, fontweight='bold')
fig.text(0.210, 0.430, 'passive', fontsize=figutils.fontSizeLabels, fontweight='bold')

fig.text(0.05, 0.70, 'A + P', fontsize=figutils.fontSizePanel, fontweight='bold', color=figutils.TangoPalette['Chameleon3'])
fig.text(0.05, 0.47, 'P : A', fontsize=figutils.fontSizePanel, fontweight='bold', color=figutils.TangoPalette['ScarletRed3'])

fig.text(0.11, 0.25, 'Trial 1500', fontsize=figutils.fontSizeLabels, fontweight='bold')
fig.text(0.65, 0.25, 'Trial 5000', fontsize=figutils.fontSizeLabels, fontweight='bold')

# -- Panel: learning curves for two-layer models --
ax1 = plt.subplot(gsMain[0, 1])
ax2 = plt.subplot(gsMain[1, 1])
ax3 = plt.subplot(gsMain[2, 0])
ax4 = plt.subplot(gsMain[2, 1])
params = {
    'fontSizeLabels': figutils.fontSizeLabels,
    'fontSizeTicks': figutils.fontSizeTicks,
    'fontSizePanel': figutils.fontSizePanel,
    'title': 'Model 5',
}
figutils.make_plots(fig, ax1, fig1Data, params)
params['title'] = None
params['idx_plot'] = 1500
figutils.corr_plot(fig, ax2, fig2Data, params)
params['idx_plot'] = 1500
figutils.psychome_plot(fig, ax3, fig3Data, params)
params['idx_plot'] = 4900
figutils.psychome_plot(fig, ax4, fig3Data, params)

    

plt.show()

if SAVE_FIGURE:
    fig.savefig(figFullPath, facecolor='none')

    # Get the data
    reader_base = PdfReader(figFullPath)
    page_base = reader_base.pages[0]

    reader = PdfReader("arrows.pdf")
    arrows_box = reader.pages[0]
    
    reader = PdfReader("model4.pdf")
    model4_box = reader.pages[0]
    
    reader = PdfReader("input_nonaligned.pdf")
    naligned_box = reader.pages[0]


    page_base.mergeScaledTranslatedPage(naligned_box, scale=1.10, tx=1.4*72., ty=5.35*72.)
    page_base.mergeScaledTranslatedPage(model4_box, scale=1.20, tx=0.2*72., ty=5.4*72.)
    page_base.mergeScaledTranslatedPage(arrows_box, scale=1.15, tx=0.5*72., ty=2.4*72.)
    # Write the result back
    writer = PdfWriter()
    writer.add_page(page_base)
    with open(figMergeFullPath, "wb") as fp:
        writer.write(fp)
        print('Figure saved to {0}'.format(figMergeFullPath))

