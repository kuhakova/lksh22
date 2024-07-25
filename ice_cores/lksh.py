import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['font.size'] = 18

def load_data(filename, unpack=True):
    return np.loadtxt(filename, unpack=unpack)

def plot(x, y, xlabel=None, ylabel=None, label=None):
    plt.plot(x, y, label=label)

    if label is not(None):
        plt.legend()

    if xlabel is not(None):
        plt.xlabel(xlabel)
    
    if ylabel is not(None):
        plt.ylabel(ylabel)

def xlim(xmin, xmax):
    plt.xlim([xmin, xmax])

def ylim(ymin, ymax):
    plt.ylim([ymin, ymax])