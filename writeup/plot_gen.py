"""
Running this script generates the three learning rate plots.

The data loaded is be specified by a command line argument.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def style_plot(ax: plt.Axes, num_it: int) -> plt.Axes:
    """
    Generate style for Learning Rate plots.
    :param ax: Drawing axis
    :param num_it: Number of plotted iterations
    :return: axis
    """
    #Remove top and right axes
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.legend()
    #Add line at 50%
    ax.hlines(50,0,num_it,color="silver", linestyles="--") 
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Correct trials (%)")
    
    ax.set_xlim((0, num_it))
    return ax

def make_plots(data: dict, title: str, num_it: int, outfile: str):
    """
    Make plot and save it.
    :param title: Plot title
    :param num_it: Number of plotted iterations
    """
    to_draw = [
    [data['an'], "A only", "blue"],
    [data['ap'][:, ::10], "A + P", "green"],
    [data['pta'][:, -num_it:], "P : A", "red"],
    ]

    fig, ax = plt.subplots(1,1)
    
    for data_it, label, color in to_draw:
        data_mean = 100 * data_it.mean(axis=0)
        data_std = 100 * data_it.std(axis=0)
        #Plot mean
        ax.plot(data_mean, label=label, color=color)
        #Make error bar
        ax.fill_between(
            np.arange(len(data_mean)),
            data_mean - data_std,
            data_mean + data_std,
            alpha=0.2,
            color=color,
        )
    #Call styling function
    ax = style_plot(ax, num_it)
    
    ax.set_title(title)
    fig.set_dpi(100)
    fig.savefig(outfile)
    
def main(infile: str, outfile: str, title: str):
    """
    Script that loads data and stores plots.
    :param filename: Data file name
    :param title: Plot title
    """
    data = np.load(infile)
    num_it =  len(data['an'][0])
    make_plots(data, title, num_it, outfile)
    
    
    
    
# Construct the argument parser
ap = argparse.ArgumentParser(description="Create learning rate plots from data")

# Add the arguments to the parser
ap.add_argument("title", help="Plot title", type=str)
ap.add_argument("infile", help="Filename of the input data file", type=str)
ap.add_argument("outfile", help="Filename of the output figure", type=str)

args = vars(ap.parse_args())


if __name__ == "__main__":
    main(args['infile'], args['outfile'], args['title'])