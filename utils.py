import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn 
from torch.nn import functional as F 

def plot(arrays:tuple, labels = None, xlabel = None, ylabel = None, title = None, grid = True, save = False):
    '''
    use to draw curves\\
    arrays: tuple of array\\
    labels: tuple of label\\
    '''
    assert type(arrays) == tuple
    plt.clf()
    for i in range(len(arrays)):
        if labels is not None:
            plt.plot(arrays[i], label = labels[i])
        else:
            plt.plot(arrays[i])
    if labels is not None:
        plt.legend()
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if grid:
        plt.grid()
    if save:
        plt.savefig('./' + title + '.png')


       
if __name__ == '__main__':
    a = np.arange(5)
    b = 2 * a 
    plot((a,), labels = ('a',))
    plt.show()