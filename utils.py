import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn 
from torch.nn import functional as F 

def plot(X, Y, labels = None, xlabel = None, ylabel = None, title = None, 
         grid = True, scatter = False, save = False):
    '''
    use to draw curves\\
    Y: tuple of array\\
    labels: tuple of label\\
    '''
    plt.clf()
    if scatter:
        draw_fn = plt.scatter
    else:
        draw_fn = plt.plot
        
    if type(Y) == tuple:
        for i in range(len(Y)):
            if labels is not None:
                draw_fn(X, Y[i], label = labels[i])
            else:
                draw_fn(X, Y[i])
    else:
        if labels is not None:
            draw_fn(X, Y, label = labels)
        else:
            draw_fn(X, Y)
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
    c = 3 * a
    plot(a, (b, c), labels = ('b', 'c'), scatter=True)
    plt.show()