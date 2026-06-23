import scipy
import skimage
import sys
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from numpy import matlib
import seaborn as sns
import math
import os
import scipy.optimize
import scipy.io as sio
from scipy import stats
import scipy.ndimage
# import h5py
sns.set_style("white")
from skimage.morphology import disk
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
import image_toolkit as imkit

sns.set(font_scale=1.5)
sns.set_style("whitegrid")

#Note: celltype will be the hue for plotting later.

# path = '/mnt/d/Documents_D/Rojas_lab/data/'
path = '/Volumes/data_ssd3/Barber_Lab/data/'
# path = '/Volumes/data_ssd1/Rojas_Lab/data/'
# path = '/Users/felixbarber/Documents/Rojas_Lab/data'

# expt_id, date = '/260415_bBL27_Tun_induction', '4/15/26'
# conditions = ['LB', '1h Tunicamycin treatment']
# expt = 'WT 4/15/26'
# label = 'mVenus'
# celltype = 'WT'

# expt_id, date = '/260603_yfp_pads', '6/3/26'
# conditions = ['WT LB','WT Tun', r'$\Delta lytE$ LB', r'$\Delta lytE$ Tun']
# expt = 'yocH 6/3/26'
# label = 'mVenus'
# celltypes = ['WT', 'WT', r'$\Delta lytE$', r'$\Delta lytE$']

# expt_id, date = '/260527_Tun_yocH_induction', '5/27/26'
# conditions = ['WT LB',r'$\Delta lytE$ LB', r'$\Delta lytE$ Tun']
# expt = 'yocH 5/27/26'
# label = 'mVenus'
# celltypes = ['WT', r'$\Delta lytE$', r'$\Delta lytE$']

expt_id, date = '/260521_bAH26', '5/21/26'
conditions = [r'$\Delta lytE$ LB', r'$\Delta lytE$ Tun']
expt = 'yocH 5/21/26'
label = 'mVenus'
celltypes = [r'$\Delta lytE$']

temp = {'id':expt_id, 'Conditions':conditions, 'Date':date, 'Celltype':celltypes, 'Label':label, 'expt':expt}
temp_path = path+expt_id+expt_id+'_condition_parameters.pkl'
with open(temp_path, 'wb') as output:  # Overwrites any existing file.
    pickle.dump(temp, output, pickle.HIGHEST_PROTOCOL)