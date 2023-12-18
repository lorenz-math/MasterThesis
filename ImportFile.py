import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms

import scipy.io
import sys
import json
import time
import pprint
import os

import math
import random


import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mc
import colorsys
from matplotlib import rc
import seaborn as sns
import pandas as pd
import itertools as it

from os import listdir
from os.path import isfile, join

#sets up imported packages and which PDE is approximated if there is more than one option

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#The equation that is approximated
import EquationModels.SemiLinWave as Ec


from ModelClassTorch2 import *
from DatasetTorch2 import *

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=False)
# plt.rc('text.latex', preamble=r'\usepackage{euscript}')
SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the tick labels
