import scipy
import sys
from scipy import io
import numpy as np
import numpy.matlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
from numpy import matlib
import seaborn as sns
import math
import os
import scipy.optimize
import scipy.io as sio
from scipy import stats
import scipy.ndimage
from skimage import io
import mat73

sns.set(font_scale=1.5)
sns.set_style("whitegrid")

# This script can be applied to generate a set of basic plots for an experiment with two, sequential antibiotic shocks.
# It is specifically designed so that it allows for different timesteps in different parts of the experiment.

remove_non_growing_cells = True
remove_fliers = True
corr=1.0 # To be redefined if lengthscales need to be corrected.
window = 1 # number of time points on either side with which to calculate the local slope
window_sa = 1 # note that the larger window here all but ensures that the measured SA growth rate will be less
scenes = None
pert_comp_ind=0
format73=False
lscale=0.065
########################################################################################################
# User inputs
########################################################################################################

# expt_id = '/211216_bFB8_Tun_gr'
# tsteps = 151
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['0.5ug/mL Tunicamycin added']
# t_pert = [10*60.0]
# max_time_truncation = 99*60.0
# scene_nums = 6
# max_width=4.0
# min_width=0.5
# min_length=2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd1/Rojas_Lab/data/'
# data_path='/Volumes/data_ssd1/Rojas_Lab/data/'
# format73=True
# im_shape=[1500,1500]
# ymin_pos,ymax_pos=im_shape[1]*0.8, im_shape[1]*1.0 # selects a portion of the image, with y increasing going down.
# corr=0.0929/0.065

expt_id = '/211020_bFB8_Tun'
tsteps = 166
dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
labels = ['0.5ug/mL Tunicamycin added']
t_pert = [10*60.0]
max_time_truncation = 99*60.0
scene_nums = 6
max_width=4.0
min_width=0.5
min_length=2.0
perform_ttest=False
base_path='/Volumes/data_ssd2/Rojas_Lab/data/'
data_path='/Volumes/data_ssd2/Rojas_Lab/data/'
format73=True
im_shape=[1500,1500]
ymin_pos,ymax_pos=im_shape[1]*0.8, im_shape[1]*1.0 # selects a portion of the image, with y increasing going down.
corr=0.0929/0.065
scenes=[1,2] # all other scenes were from row 4.
#
# expt_id = '/260612_bFB292_IPTG'
# tsteps = 361
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added']
# t_pert = [10*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 3  # Note that this timelapse had some growth in scene 5 but that these cells were very far away from the
# # inlet and likely received a lower dose of GlpQ based on CY5 channel
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data/'
# data_path='/Volumes/data_ssd3/Barber_Lab/data/'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.8, im_shape[1]*1.0 # selects a portion of the image, with y increasing going down.
# format73=True

# expt_id = '/260612_bFB295_IPTG_Mg'
# tsteps = 286
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added', r'MgCl$_2$ added']
# t_pert = [5*60.0, 65*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 3
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data'
# data_path='/Volumes/data_ssd3/Barber_Lab/data'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.9, im_shape[1]*1.0 # selects a portion of the image, with y increasing going down.
# format73=True
# # scenes=[1,4] # scenes 2 and 3 were overgrown

# expt_id = '/260612_bFB292_IPTG_Mg'
# tsteps = 286
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added', r'MgCl$_2$ added']
# t_pert = [5*60.0, 65*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data'
# data_path='/Volumes/data_ssd3/Barber_Lab/data'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.9, im_shape[1]*1.0 # selects a portion of the image, with y increasing going down.
# format73=True
# # scenes=[1,4] # scenes 2 and 3 were overgrown

# expt_id = '/260611_bFB292_IPTG_Mg'
# tsteps = 286
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added', r'MgCl$_2$ added']
# t_pert = [5*60.0, 65*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data'
# data_path='/Volumes/data_ssd3/Barber_Lab/data'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.9, im_shape[1]*1.0 # selects a portion of the image, with y increasing going down.
# format73=True
# # scenes=[1,4] # scenes 2 and 3 were overgrown

# expt_id = '/260531_bFB66_LB_long'
# tsteps = 361
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['LB Switched']
# t_pert = [60*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data/'
# data_path='/Volumes/data_ssd3/Barber_Lab/data/'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.9, im_shape[1]*1.0 # selects a portion of the image, with y increasing going down.
# format73=True

# expt_id = '/260531_bFB292_IPTG_Mg'
# tsteps = 286
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added', r'MgCl$_2$ added']
# t_pert = [5*60.0, 65*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data'
# data_path='/Volumes/data_ssd3/Barber_Lab/data'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.9, im_shape[1]*1.0 # selects a portion of the image, with y increasing going down.
# format73=True
# # scenes=[1,4] # scenes 2 and 3 were overgrown

# expt_id = '/260527_bFB66_LB_long'
# tsteps = 300
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['LB Switched']
# t_pert = [60*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data/'
# data_path='/Volumes/data_ssd3/Barber_Lab/data/'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.8, im_shape[1]*1.0 # selects a portion of the image, with y increasing going down.
# format73=True


# expt_id = '/260527_bFB292_IPTG_Mg'
# tsteps = 286
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added', r'MgCl$_2$ added']
# t_pert = [5*60.0, 65*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data'
# data_path='/Volumes/data_ssd3/Barber_Lab/data'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.0, im_shape[1]*0.2 # selects a portion of the image, with y increasing going down.
# format73=True

# expt_id = '/260521_bFB292_IPTG_Mg'
# tsteps = 286
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added', r'MgCl$_2$ added']
# t_pert = [5*60.0, 65*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data'
# data_path='/Volumes/data_ssd3/Barber_Lab/data'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.0, im_shape[1]*0.2 # selects a portion of the image, with y increasing going down.
# format73=True

# expt_id = '/260520_bFB66_NLS_PBS_Mg'
# tsteps = 97
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['PBS+5% NLS', r'PBS', r'PBS + 10mM MgCl$_2$', r'PBS', r'PBS + 10mM MgCl$_2$']
# t_pert = [(32-1)*20.0, (38-1)*20.0, (53-1)*20.0, (68-1)*20.0, (83-1)*20.0] # timepoints 32 and 40
# max_time_truncation = 140*60.0
# scene_nums = 3
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data/'
# data_path='/Volumes/data_ssd3/Barber_Lab/data/'
# im_shape=[2304,2304]
# window = 1 # number of time points on either side with which to calculate the local slope
# ymin_pos,ymax_pos=im_shape[1]*0.0, im_shape[1] # selects a portion of the image, with y increasing going down.
# format73=True

# expt_id = '/260519_bFB291_IPTG'
# tsteps = 361
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added']
# t_pert = [10*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4  # Note that this timelapse had some growth in scene 5 but that these cells were very far away from the
# # inlet and likely received a lower dose of GlpQ based on CY5 channel
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data/'
# data_path='/Volumes/data_ssd3/Barber_Lab/data/'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.8, im_shape[1]*1.0 # selects a portion of the image, with y increasing going down.
# format73=True

# expt_id = '/260519_bFB66_Tun_NLS_PBS_Mg'
# tsteps = 172
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['LB+0.5ug/mL Tunicamycin', 'PBS+5% NLS', r'PBS', r'PBS + 10mM MgCl$_2$', r'PBS', r'PBS + 10mM MgCl$_2$']
# t_pert = [(17-1)*20.0, (107-1)*20.0, (113-1)*20.0, (128-1)*20.0, (143-1)*20.0, (158-1)*20.0] # timepoints 32 and 40
# max_time_truncation = 140*60.0
# scene_nums = 4
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data/'
# data_path='/Volumes/data_ssd3/Barber_Lab/data/'
# im_shape=[2304,2304]
# window = 1 # number of time points on either side with which to calculate the local slope
# window_sa = 1 # note that the larger window here all but ensures that the measured SA growth rate will be less
# ymin_pos,ymax_pos=im_shape[1]*0.0, im_shape[1] # selects a portion of the image, with y increasing going down.
# # conservative=False
# format73=True

# expt_id = '/260519_bFB66_Tun_NLS_PBS_Mg_v2'
# tsteps = 172
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in fseconds
# labels = ['LB+0.5ug/mL Tunicamycin', 'PBS+5% NLS', r'PBS', r'PBS + 10mM MgCl$_2$', r'PBS', r'PBS + 10mM MgCl$_2$']
# t_pert = [(17-1)*20.0, (107-1)*20.0, (113-1)*20.0, (128-1)*20.0, (143-1)*20.0, (158-1)*20.0] # timepoints 32 and 40
# max_time_truncation = 140*60.0
# scene_nums = 4
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data/'
# data_path='/Volumes/data_ssd3/Barber_Lab/data/'
# im_shape=[2304,2304]
# window = 1 # number of time points on either side with which to calculate the local slope
# window_sa = 1 # note that the larger window here all but ensures that the measured SA growth rate will be less
# ymin_pos,ymax_pos=im_shape[1]*0.0, im_shape[1] # selects a portion of the image, with y increasing going down.
# # conservative=False
# format73=True

# expt_id = '/260306_bFB295_IPTG'
# tsteps = 361
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added']
# t_pert = [10*60.0]
# max_time_truncation = 140*60.0
# # scene_nums = 3
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data/'
# data_path='/Volumes/data_ssd3/Barber_Lab/data/'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.0, im_shape[1]*0.7 # selects a portion of the image, with y increasing going down.
# scenes=[2,3] # scene 1 lost focus and became overgrown.

# expt_id = '/260506_bFB66_Tun_NLS_PBS_Mg_v2'
# tsteps = 172
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['LB+0.5ug/mL Tunicamycin', 'PBS+5% NLS', r'PBS', r'PBS + 10mM MgCl$_2$', r'PBS', r'PBS + 10mM MgCl$_2$']
# t_pert = [(17-1)*20.0, (107-1)*20.0, (113-1)*20.0, (128-1)*20.0, (143-1)*20.0, (158-1)*20.0] # timepoints 32 and 40
# max_time_truncation = 140*60.0
# scene_nums = 4
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data/'
# data_path='/Volumes/data_ssd3/Barber_Lab/data/'
# im_shape=[2304,2304]
# window = 1 # number of time points on either side with which to calculate the local slope
# window_sa = 1 # note that the larger window here all but ensures that the measured SA growth rate will be less
# ymin_pos,ymax_pos=im_shape[1]*0.0, im_shape[1] # selects a portion of the image, with y increasing going down.
# # conservative=False


# expt_id = '/260505_bFB66_NLS_PBS_Mg'
# tsteps = 97
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['PBS+5% NLS', r'PBS', r'PBS + 10mM MgCl$_2$', r'PBS', r'PBS + 10mM MgCl$_2$']
# t_pert = [(32-1)*20.0, (38-1)*20.0, (53-1)*20.0, (68-1)*20.0, (83-1)*20.0] # timepoints 32 and 40
# max_time_truncation = 140*60.0
# scene_nums = 3
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data/'
# data_path='/Volumes/data_ssd3/Barber_Lab/data/'
# im_shape=[2304,2304]
# window = 1 # number of time points on either side with which to calculate the local slope
# ymin_pos,ymax_pos=im_shape[1]*0.0, im_shape[1] # selects a portion of the image, with y increasing going down.


# expt_id = '/260416_bFB295_IPTG_induction'
# tsteps = 361
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added']
# t_pert = [10*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data/'
# data_path='/Volumes/data_ssd3/Barber_Lab/data/'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.8, im_shape[1] # selects a portion of the image, with y increasing going down.

# expt_id = '/260505_bFB66_Tun_NLS_PBS_Mg'
# tsteps = 142
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['LB+tunicamycin', 'PBS+5% NLS', r'PBS', r'PBS + 10mM MgCl$_2$', r'PBS', r'PBS + 10mM MgCl$_2$']
# t_pert = [(17-1)*20.0, (77-1)*20.0, (83-1)*20.0, (98-1)*20.0, (113-1)*20.0, (128-1)*20.0] # timepoints 32 and 40
# max_time_truncation = 140*60.0
# scene_nums = 3
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data/'
# data_path='/Volumes/data_ssd3/Barber_Lab/data/'
# im_shape=[2304,2304]
# yboun=0.5 # fraction of the image to accept, measured along the y axis
# window = 1 # number of time points on either side with which to calculate the local slope
# ymin_pos,ymax_pos=im_shape[1]*0.0, im_shape[1]*1.0 # selects a portion of the image, with y increasing going down.

# expt_id = '/250408_bFB291_IPTG_induction'
# tsteps = 361
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added']
# t_pert = [10*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Rojas_Lab/data/'
# data_path='/Volumes/data_ssd2/Rojas_Lab/data/'
# im_shape=[1500,1500]
# ymin_pos,ymax_pos=im_shape[1]*0.8, im_shape[1] # selects a portion of the image, with y increasing going down.
# window = 1 # number of time points on either side with which to calculate the local slope
# window_sa = 1 # note that the larger window here all but ensures that the measured SA growth rate will be less


# expt_id = '/260504_bFB66_NLS_PBS_Mg_v2'
# tsteps = 97
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['PBS+5% NLS', r'PBS', r'PBS + 10mM MgCl$_2$', r'PBS', r'PBS + 10mM MgCl$_2$']
# t_pert = [(32-1)*20.0, (38-1)*20.0, (53-1)*20.0, (68-1)*20.0, (83-1)*20.0] # timepoints 32 and 40
# max_time_truncation = 140*60.0
# scene_nums = 4
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Barber_Lab/data/'
# data_path='/Volumes/data_ssd2/Barber_Lab/data/'
# im_shape=[2304,2304]
# yboun=0.5 # fraction of the image to accept, measured along the y axis
# window = 1 # number of time points on either side with which to calculate the local slope
# ymin_pos,ymax_pos=im_shape[1]*0.0, im_shape[1]*1.0 # selects a portion of the image, with y increasing going down.
# scenes=[1]

# expt_id = '/260504_bFB66_NLS_PBS_Mg'
# tsteps = 97
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['PBS+5% NLS', r'PBS', r'PBS + 10mM MgCl$_2$', r'PBS', r'PBS + 10mM MgCl$_2$']
# t_pert = [(32-1)*20.0, (38-1)*20.0, (53-1)*20.0, (68-1)*20.0, (83-1)*20.0] # timepoints 32 and 40
# max_time_truncation = 140*60.0
# scene_nums = 3
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Barber_Lab/data/'
# data_path='/Volumes/data_ssd2/Barber_Lab/data/'
# im_shape=[2304,2304]
# yboun=0.5 # fraction of the image to accept, measured along the y axis
# window = 1 # number of time points on either side with which to calculate the local slope
# ymin_pos,ymax_pos=im_shape[1]*0.0, im_shape[1] # selects a portion of the image, with y increasing going down.

# expt_id = '/260430_bFB66_15MSorb_noMg'
# tsteps = 76
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1.5M Sorbitol added', r'1.5M Sorbitol swapped']
# t_pert = [(32-1)*20.0, (40-1)*20.0] # timepoints 32 and 40
# max_time_truncation = 140*60.0
# scene_nums = 3
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Barber_Lab/data/'
# data_path='/Volumes/data_ssd2/Barber_Lab/data/'
# im_shape=[2304,2304]
# window = 1 # number of time points on either side with which to calculate the local slope

# expt_id = '/260430_bFB66_2MSorb_10mMMgCl2_LB'
# tsteps = 76
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['2M Sorbitol added', r'10mM MgCl$_2$ added', r'10mM MgCl$_2$ removed']
# t_pert = [(32-1)*20.0, (40-1)*20.0, (55-1)*20.0] # timepoints 32 and 40
# max_time_truncation = 140*60.0
# scene_nums = 3
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Barber_Lab/data/'
# data_path='/Volumes/data_ssd2/Barber_Lab/data/'
# im_shape=[2304,2304]
# window = 1 # number of time points on either side with which to calculate the local slope

# expt_id = '/260430_bFB66_15MSorb_10mMMgCl2'
# tsteps = 76
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1.5M Sorbitol added', r'10mM MgCl$_2$ added', r'10mM MgCl$_2$ removed']
# t_pert = [(32-1)*20.0, (40-1)*20.0, (55-1)*20.0] # timepoints 32 and 40
# max_time_truncation = 140*60.0
# scene_nums = 3
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Barber_Lab/data/'
# data_path='/Volumes/data_ssd2/Barber_Lab/data/'
# im_shape=[2304,2304]
# yboun=0.5 # fraction of the image to accept, measured along the y axis
# window = 1 # number of time points on either side with which to calculate the local slope

# expt_id = '/260428_bFB66_2MSorb_10mMMgCl2_noAF'
# tsteps = 52
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['2M Sorbitol added', r'10mM MgCl$_2$ added']
# t_pert = [(32-1)*20.0, (40-1)*20.0] # timepoints 32 and 40
# max_time_truncation = 140*60.0
# scene_nums = 2
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Barber_Lab/data/'
# data_path='/Volumes/data_ssd2/Barber_Lab/data/'
# im_shape=[2304,2304]
# window = 1 # number of time points on either side with which to calculate the local slope

# expt_id = '/260428_bFB66_2MSorb_20mMMgCl2'
# tsteps = 52
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['2M Sorbitol added', r'20mM $MgCl_2$ added']
# t_pert = [(32-1)*20.0, (40-1)*20.0] # timepoints 32 and 40
# max_time_truncation = 140*60.0
# scene_nums = 2
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Barber_Lab/data/'
# data_path='/Volumes/data_ssd2/Barber_Lab/data/'
# im_shape=[2304,2304]
# window = 1 # number of time points on either side with which to calculate the local slope

# expt_id = '/260428_bFB66_2MSorb_10mMMgCl2'
# tsteps = 68
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['2M Sorbitol added', r'10mM MgCl$_2$ added']
# t_pert = [(32-1)*20.0, (40-1)*20.0] # timepoints 32 and 40
# max_time_truncation = 140*60.0
# scene_nums = 2
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Barber_Lab/data/'
# data_path='/Volumes/data_ssd2/Barber_Lab/data/'
# im_shape=[2304,2304]
# window = 1 # number of time points on either side with which to calculate the local slope

# expt_id = '/250402_bFB66_LB'
# tsteps = 361
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['LB Switched']
# t_pert = [60*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 3
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Rojas_Lab/data/'
# data_path='/Volumes/data_ssd2/Rojas_Lab/data/'
# im_shape=[1500,1500]
# ymin_pos,ymax_pos=im_shape[1]*0.8, im_shape[1]*1.0 # selects a portion of the image, with y increasing going down.


# expt_id = '/250326_bFB295_IPTG_induction'
# tsteps = 361
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added']
# t_pert = [10*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 3  # Note that this timelapse had some growth in scene 5 but that these cells were very far away from the
# # inlet and likely received a lower dose of GlpQ based on CY5 channel
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Rojas_Lab/data/'
# data_path='/Volumes/data_ssd2/Rojas_Lab/data/'
# im_shape=[1500,1500]
# ymin_pos,ymax_pos=im_shape[1]*0.8, im_shape[1] # selects a portion of the image, with y increasing going down.


# expt_id = '/250324_bFB291_IPTG_induction'
# tsteps = 361
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added']
# t_pert = [10*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4  # Note that this timelapse had some growth in scene 5 but that these cells were very far away from the
# # inlet and likely received a lower dose of GlpQ based on CY5 channel
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Rojas_Lab/data/'
# data_path='/Volumes/data_ssd2/Rojas_Lab/data/'
# im_shape=[1500,1500]
# ymin_pos,ymax_pos=im_shape[1]*0.8, im_shape[1] # selects a portion of the image, with y increasing going down.

# expt_id = '/260216_bFB292_IPTG_Mg'
# tsteps = 286
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added', r'MgCl$_2$ added']
# t_pert = [5*60.0, 65*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Rojas_Lab/data'
# data_path='/Volumes/data_ssd2/Rojas_Lab/data'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.0, im_shape[1]*0.2 # selects a portion of the image, with y increasing going down.

# expt_id = '/260414_bFB292_IPTG'
# tsteps = 361
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added']
# t_pert = [10*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 2  # Note that this timelapse had some growth in scene 5 but that these cells were very far away from the
# # inlet and likely received a lower dose of GlpQ based on CY5 channel
# max_width = 2.5
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data/'
# data_path='/Volumes/data_ssd3/Barber_Lab/data/'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.8, im_shape[1]*1.0 # selects a portion of the image, with y increasing going down.
# format73=True

# expt_id = '/260316_bBF292_IPTG_Mg'
# tsteps = 331
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added', '10mM MgCl2 added']
# t_pert = [10*60.0, 80*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4  # Note that this timelapse had some growth in scene 5 but that these cells were very far away from the
# # inlet and likely received a lower dose of GlpQ based on CY5 channel
# max_width = 2.5
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Barber_Lab/data/'
# data_path='/Volumes/data_ssd2/Barber_Lab/data/'


# expt_id = '/260313_bBF292_IPTG'
# tsteps = 361
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added']
# t_pert = [10*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4  # Note that this timelapse had some growth in scene 5 but that these cells were very far away from the
# # inlet and likely received a lower dose of GlpQ based on CY5 channel
# max_width = 2.5
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd3/Barber_Lab/data/'
# data_path='/Volumes/data_ssd3/Barber_Lab/data/'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.0, im_shape[1]*0.2 # selects a portion of the image, with y increasing going down.
# format73=True


# expt_id = '/250325_bFB293_IPTG_induction'
# tsteps = 361
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added']
# t_pert = [10*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4  # Note that this timelapse had some growth in scene 5 but that these cells were very far away from the
# # inlet and likely received a lower dose of GlpQ based on CY5 channel
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Rojas_Lab/data/'
# data_path='/Volumes/data_ssd2/Rojas_Lab/data/'
# im_shape=[1500,1500]
# ymin_pos,ymax_pos=im_shape[1]*0.8, im_shape[1] # selects a portion of the image, with y increasing going down.

# expt_id = '/260305_bFB292_IPTG'
# tsteps = 361
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added']
# t_pert = [10*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4  # Note that this timelapse had some growth in scene 5 but that these cells were very far away from the
# # inlet and likely received a lower dose of GlpQ based on CY5 channel
# max_width = 2.5
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Barber_Lab/data/'
# data_path='/Volumes/data_ssd2/Barber_Lab/data/'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.0, im_shape[1]*0.2 # selects a portion of the image, with y increasing going down.



# expt_id = '/250403_bFB292_IPTG_induction'
# tsteps = 361
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added']
# t_pert = [10*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 2  # Note that this timelapse had some growth in scene 5 but that these cells were very far away from the
# # inlet and likely received a lower dose of GlpQ based on CY5 channel
# max_width = 2.5
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Rojas_Lab/data/'
# data_path='/Volumes/data_ssd2/Rojas_Lab/data/'
# corr=0.0929/0.065
# im_shape=[1500,1500]
# ymin_pos,ymax_pos=im_shape[1]*0.8, im_shape[1] # selects a portion of the image, with y increasing going down.


# expt_id = '/260217_bFB292_IPTG_Mg'
# tsteps = 286
# dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
# labels = ['1mM IPTG added', r'MgCl$_2$ added']
# t_pert = [5*60.0, 65*60.0]
# max_time_truncation = 140*60.0
# scene_nums = 4
# max_width = 4.0
# min_width = 0.5
# min_length = 2.0
# perform_ttest=False
# base_path='/Volumes/data_ssd2/Barber_Lab/data'
# data_path='/Volumes/data_ssd2/Barber_Lab/data'
# im_shape=[2304,2304]
# ymin_pos,ymax_pos=im_shape[1]*0.0, im_shape[1]*0.2 # selects a portion of the image, with y increasing going down.



# Generic code to be run.
initial_time_plotting = 15  # number of minutes for which to plot the growth rate values of each scene
thresh = 0.00005  # threshold for the average growth rate below which we preclude cells from analysis. This is set very
# low to make sure that cells are actually not growing at all.
flier_thresh = 2.0 # threshold for number of iqrs away from median growth rate beyond which we preclude cells from analysis.
w_flier_thresh = 3.0 # threshold for cell width relative rate of change beyond which we preclude cells from analysis.
tvec=np.cumsum(dt)
tvec=np.insert(tvec,0,0.0)
max_tstep_truncation = len(tvec) # this is the point at which the analysis truncates if you want it to be before the
# end of the dataset
# filtering to only include contigs of so many timepoints or more.
cutoff=15 # set to 50 for growth timecourses, 15 for lysis plots

if scenes is None:
    scenes = range(1, scene_nums+1)

# making file structures
if not os.path.exists('./outputs'+expt_id):
    os.mkdir('./outputs'+expt_id)
if not os.path.exists('./outputs' + expt_id+'/scene_growth_visualization'):
    os.mkdir('./outputs' + expt_id+'/scene_growth_visualization')
linestyles = ['-','-.','--', 'dotted','-','-.']
sys.stdout = open('./outputs'+expt_id+expt_id+'_outputs.txt', "w")
# Loading the data (same as antibiotic_growth_inhibition_tester_combined_shock_v2.py)
temp = []
temp1 = []
temp2 = []
temp3 = []
temp4 = []
temp5 = []
temp6 = []
print('hi')
# for i0 in range(1, scene_nums+1):
for i0 in scenes:
    temp_name = expt_id+'_s{:03d}'.format(i0)
    temp_path = data_path+expt_id+temp_name+'_1_a'+temp_name+'_BT_felix.mat'
    if format73:
        data=mat73.loadmat(temp_path)
    else:
        data=scipy.io.loadmat(temp_path)
    # print('hi')
    # print(data.keys(), i0)
    temp.append(np.asarray(data['lcell']))
    temp1.append(np.asarray(data['wcell']))
    temp2.append(np.asarray(data['sacell']))
    temp3.append(i0*np.ones(data['lcell'].shape[0]))
    ypos = np.nan * np.ones(temp[-1].shape)
    xpos = np.nan * np.ones(temp[-1].shape)
    a_proj = np.nan * np.ones(temp[-1].shape)
    for i0 in range(temp[-1].shape[0]): # cells
        for i1 in range(temp[-1].shape[1]): # timepoints
            if ~np.isnan(temp[-1][i0, i1]):
                temp_pix = data['pixels'][i0][i1].astype('int')
                temp_pix -= 1  # account for the fact that in matlab, these linear indices will start from 1 rather than 0
                pxls = np.unravel_index(temp_pix, im_shape)
                ypos[i0, i1] = np.mean(pxls[1])
                xpos[i0, i1] = np.mean(pxls[0])
                a_proj[i0, i1] = len(pxls[1])*lscale**2 # area in microns squared
    temp4.append(xpos)
    temp5.append(ypos)
    temp6.append(a_proj)
lcell = np.concatenate(temp,axis=0)*corr
wcell = np.concatenate(temp1,axis=0)*corr
sacell = np.concatenate(temp2,axis=0)*(corr**2)
xcents = np.concatenate(temp4,axis=0)
ycents = np.concatenate(temp5,axis=0)
saproj = np.concatenate(temp6,axis=0)*(corr**2)
scene_num_tracker = np.concatenate(temp3,axis=0)
# filtering the data based on dimensions
sacell=sacell[:, :max_tstep_truncation]
lcell=lcell[:, :max_tstep_truncation]
wcell=wcell[:, :max_tstep_truncation]
saproj=saproj[:, :max_tstep_truncation]
sacell[np.isnan(lcell)]=np.nan # making sure that the surface area doesn't include fixed objects that have been
saproj[np.isnan(lcell)]=np.nan # making sure that the surface area doesn't include fixed objects that have been
# filtered successfully within lcell and wcell.
print(lcell.shape,wcell.shape,sacell.shape, max_tstep_truncation)
time=tvec[:max_tstep_truncation]


###################################################################
cutoff_time = len(time)-1
time_truncated = time[:cutoff_time]
print("cutoff time = ", cutoff_time)

for i0 in range(lcell.shape[0]): # for each cell trace
    temp=(~np.isnan(lcell[i0,:])).astype(int) # 1 for cell, 0 for nan
    inds1=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==1) # beginning of each trace
    inds2=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==-1) # end point of each trace
    lens=(inds2[0]-inds1[0]).tolist() # gives the length of each contig
    for i1 in range(len(lens)):
        if lens[i1]<cutoff:
            lcell[i0,inds1[0][i1]:inds1[0][i1]+lens[i1]]=np.nan
            wcell[i0,inds1[0][i1]:inds1[0][i1]+lens[i1]]=np.nan
            sacell[i0,inds1[0][i1]:inds1[0][i1]+lens[i1]]=np.nan
            saproj[i0, inds1[0][i1]:inds1[0][i1] + lens[i1]] = np.nan
for i0 in range(wcell.shape[0]):  # for each cell trace
    temp=(~np.isnan(wcell[i0,:])).astype(int) # 1 for cell, 0 for nan
    inds1=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==1) # beginning of each trace
    inds2=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==-1) # end point of each trace
    lens=(inds2[0]-inds1[0]).tolist() # gives the length of each contig
    for i1 in range(len(lens)):
        if lens[i1]<cutoff:
            lcell[i0,inds1[0][i1]:inds1[0][i1]+lens[i1]]=np.nan
            wcell[i0,inds1[0][i1]:inds1[0][i1]+lens[i1]]=np.nan
            sacell[i0,inds1[0][i1]:inds1[0][i1]+lens[i1]]=np.nan
            saproj[i0, inds1[0][i1]:inds1[0][i1] + lens[i1]] = np.nan

# fitting to find the surface area growth rate at each point.

sgr_sa = np.zeros(sacell.shape)
sgr_sa[:, :] = np.nan
sgr_saproj = np.zeros(sacell.shape)
sgr_saproj[:, :] = np.nan
sgr_l = np.zeros(lcell.shape)
sgr_l[:, :] = np.nan
sgr_w = np.zeros(wcell.shape)
sgr_w[:, :] = np.nan
# time-sensitive to changes in growth rate.
for i0 in range(sacell.shape[0]):  # number of cells
    for i1 in range(sacell.shape[1]):  # timepoints
        if ~np.isnan(lcell[i0, i1]):  # if this timepoint is not a nan value
            # for specific length growth rate
            temp_x = time[max([i1 - window, 0]):min([i1 + window, len(time)])] # window of length 5 for all regions poss
            temp_y = lcell[i0, max([i1 - window, 0]):min([i1 + window, len(time)])]
            temp_x = temp_x[~np.isnan(temp_y)] # select only points that have actual data
            temp_y = temp_y[~np.isnan(temp_y)]
            temp_vals = scipy.stats.linregress(temp_x, temp_y)
            sgr_l[i0, i1] = temp_vals[0] / np.nanmean(temp_y)
            # for specific width growth rate
            temp_x = time[
                max([i1 - window, 0]):min([i1 + window, len(time)])]  # window of length 5 for all regions poss
            temp_y = wcell[i0, max([i1 - window, 0]):min([i1 + window, len(time)])]
            temp_x = temp_x[~np.isnan(temp_y)]  # select only points that have actual data
            temp_y = temp_y[~np.isnan(temp_y)]
            temp_vals = scipy.stats.linregress(temp_x, temp_y)
            sgr_w[i0, i1] = temp_vals[0] / np.nanmean(temp_y)
            # for specific SA growth rate
            temp_x = time[max([i1 - window_sa, 0]):min([i1 + window_sa, len(time)])]
            temp_y = sacell[i0, max([i1 - window_sa, 0]):min([i1 + window_sa, len(time)])]
            temp_x = temp_x[~np.isnan(temp_y)]
            temp_y = temp_y[~np.isnan(temp_y)]
            temp_vals = scipy.stats.linregress(temp_x, temp_y)
            sgr_sa[i0, i1] = temp_vals[0] / np.nanmean(temp_y)

            temp_x = time[max([i1 - window_sa, 0]):min([i1 + window_sa, len(time)])]
            temp_y = saproj[i0, max([i1 - window_sa, 0]):min([i1 + window_sa, len(time)])]
            temp_x = temp_x[~np.isnan(temp_y)]
            temp_y = temp_y[~np.isnan(temp_y)]
            temp_vals = scipy.stats.linregress(temp_x, temp_y)
            sgr_saproj[i0, i1] = temp_vals[0] / np.nanmean(temp_y)


###################################################################

# Filtering to remove cells that simply don't grow throughout the whole timelapse
if remove_non_growing_cells:
    filt_inds=np.nonzero(np.nanmean(sgr_l,axis=1)<thresh)
    sgr_l[filt_inds,:]=np.nan
    sgr_w[filt_inds, :] = np.nan
    wcell[filt_inds,:]=np.nan
    sgr_sa[filt_inds,:]=np.nan
    lcell[filt_inds,:]=np.nan

# Filtering to remove cells that change their width too fast
if remove_fliers:
    sgrmed = np.nanmedian(sgr_w,axis=0)
    sgrstd = scipy.stats.iqr(sgr_w,axis=0,nan_policy='omit')
    # fliers = np.absolute(sgr_w-np.tile(sgrmed,[sgr_w.shape[0],1]))/np.tile(sgrstd,[sgr_w.shape[0],1])>w_flier_thresh
    fliers = np.absolute(sgr_w)/np.maximum(np.absolute(np.tile(np.nanmedian(sgrmed),[sgr_w.shape[0],1])*w_flier_thresh),3.0*np.ones(sgr_w.shape)) > 1.0
    flier_inds = np.nonzero(np.amax(fliers, axis=1))
    sgr_w[flier_inds, :] = np.nan
    sgr_l[flier_inds, :] = np.nan
    sgr_w[flier_inds, :] = np.nan
    wcell[flier_inds, :] = np.nan
    sgr_sa[flier_inds, :] = np.nan
    sgr_saproj[flier_inds, :] = np.nan
    lcell[flier_inds, :] = np.nan

    # sgr_w[np.nonzero(fliers)]=np.nan
    # sgr_l[np.nonzero(fliers)] = np.nan
    # sgr_w[np.nonzero(fliers)] = np.nan
    # wcell[np.nonzero(fliers)] = np.nan
    # sgr_sa[np.nonzero(fliers)] = np.nan
    # lcell[np.nonzero(fliers)] = np.nan

    sgrmed = np.nanmedian(sgr_l,axis=0)
    sgrstd = scipy.stats.iqr(sgr_l,axis=0,nan_policy='omit')
    fliers = np.absolute(sgr_l-np.tile(sgrmed,[sgr_l.shape[0],1]))/np.tile(sgrstd,[sgr_l.shape[0],1])>flier_thresh
    flier_inds = np.nonzero(np.amax(fliers, axis=1))
    sgr_w[flier_inds, :] = np.nan
    sgr_l[flier_inds, :] = np.nan
    sgr_w[flier_inds, :] = np.nan
    wcell[flier_inds, :] = np.nan
    sgr_sa[flier_inds, :] = np.nan
    sgr_saproj[flier_inds, :] = np.nan
    lcell[flier_inds, :] = np.nan

    # sgr_w[np.nonzero(fliers)]=np.nan
    # sgr_l[np.nonzero(fliers)] = np.nan
    # sgr_w[np.nonzero(fliers)] = np.nan
    # wcell[np.nonzero(fliers)] = np.nan
    # sgr_sa[np.nonzero(fliers)] = np.nan
    # lcell[np.nonzero(fliers)] = np.nan

# Now we also filter to select cells growing within a certain range of y values. These values will be saved separately.

sel_pos=np.nonzero((ycents>ymin_pos)*(ycents<ymax_pos))
mult_val=np.nan*np.ones(sgr_w.shape)
mult_val[sel_pos]=1

# Next, we copy our data and put all outside values as nans
sgr_w_pos=np.copy(sgr_w)*mult_val
sgr_l_pos=np.copy(sgr_l)*mult_val
sgr_sa_pos=np.copy(sgr_sa)*mult_val
sgr_saproj_pos=np.copy(sgr_saproj)*mult_val
wcell_pos=np.copy(wcell)*mult_val
lcell_pos=np.copy(lcell)*mult_val

###################################################################
# plotting the cell length growth rate traces
# Paper-style figures.
fig = plt.figure(figsize=[2.5, 2])
sns.set(font_scale=1.15)
sns.set_style("white")
xv=time

for i0 in range(lcell.shape[0]):
    yv = lcell[i0, :]
    # print(yv)
    if len(t_pert)>0:
        plt.plot((xv-t_pert[pert_comp_ind])/60.0,yv,color='k',linewidth=0.5)
    else:
        plt.plot((xv) / 60.0, yv, color='k')
ax=plt.gca()
ymin, ymax = ax.get_ylim()
for i0 in range(len(t_pert)):
    plt.vlines((t_pert[i0]-t_pert[pert_comp_ind])/60.0,ymin=ymin,ymax=ymax,label=labels[i0],linestyle=linestyles[i0],color='k',linewidth=0.5,alpha=0.5)
plt.legend(loc=[1.02,0.0])
plt.ylabel(r'Length ($\mu m$)')
plt.xlabel('Time (s)')
fig.savefig('./outputs'+expt_id+expt_id+'_cell_lengths.pdf',dpi=150,bbox_inches='tight')
plt.clf()

# Plotting the sgr_w on a nice time axis
sgr=np.zeros(sgr_w.shape)
sgr[:,:]=sgr_w[:,:]
sgr*=3600.0
xv=time/60.0
fig=plt.figure(figsize=[8,5])
for i0 in range(sgr.shape[0]):
    temp=(~np.isnan(sgr[i0,:])).astype(int)
    inds1=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==1)[0] # beginning of each trace
    inds2=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==-1)[0] # end point of each trace
    for ind in range(len(inds1)):
        yv=sgr[i0,inds1[ind]:inds2[ind]]
        xv1 = xv[inds1[ind]:inds2[ind]]
        plt.xlim(left=0.0,right=np.amax(xv))
        # plt.plot(xv1,yv,alpha=0.2,color='b')
yv=np.nanmedian(sgr,axis=0)
err=np.nanstd(sgr,axis=0)/np.sqrt(np.sum(~np.isnan(sgr),axis=0))
plt.fill_between(xv,yv-err,yv+err,alpha=0.5,color='r')
plt.plot(xv,yv,label= 'Median',color='k',lw=3.0)
# plt.ylim(ymin=0)
ax=plt.gca()
vymin=ax.get_ylim()[0]
vymax=ax.get_ylim()[1]
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0]/60.0,ymin=vymin,ymax=vymax,label=labels[i0],linestyle=linestyles[i0])
plt.legend(loc=[1.02,0.0])
plt.xlabel('Time (min)')
plt.ylabel(r'$\frac{1}{w}\frac{dw}{dt}$ ($h^{-1}$)')
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_w_truncated.png',bbox_inches='tight',dpi=150)
plt.clf()


# Plotting the sgr on a nice time axis
sgr=np.zeros(sgr_l.shape)
sgr[:,:]=sgr_l[:,:]
sgr*=3600.0
xv=time/60.0
fig=plt.figure(figsize=[8,5])
for i0 in range(sgr.shape[0]):
    temp=(~np.isnan(sgr[i0,:])).astype(int)
    inds1=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==1)[0] # beginning of each trace
    inds2=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==-1)[0] # end point of each trace
    for ind in range(len(inds1)):
        yv=sgr[i0,inds1[ind]:inds2[ind]]
        xv1 = xv[inds1[ind]:inds2[ind]]
        plt.xlim(left=0.0,right=np.amax(xv))
        plt.plot(xv1,yv,alpha=0.2,color='b')
yv=np.nanmedian(sgr,axis=0)
err=np.nanstd(sgr,axis=0)/np.sqrt(np.sum(~np.isnan(sgr),axis=0))
plt.fill_between(xv,yv-err,yv+err,alpha=0.5,color='r')
plt.plot(xv,yv,label= 'Smoothed median',color='k',lw=3.0)
# plt.ylim(ymin=0)
ax=plt.gca()
vymin=ax.get_ylim()[0]
vymax=ax.get_ylim()[1]
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0]/60.0,ymin=vymin,ymax=vymax,label=labels[i0],linestyle=linestyles[i0])
plt.legend()
plt.xlabel('Time (min)')
plt.ylabel(r'$\frac{1}{l}\frac{dl}{dt}$ ($h^{-1}$)')
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_truncated.png',bbox_inches='tight',dpi=150)
plt.clf()


# Plotting the sgr on its original axis
sgr=np.zeros(sgr_l.shape)
sgr[:,:]=sgr_l[:,:]

sgrmed = np.nanmedian(sgr,axis=0)
sgrstd = scipy.stats.iqr(sgr,axis=0,nan_policy='omit')


xv=time
fig=plt.figure(figsize=[8,5])
for i0 in range(sgr.shape[0]):
    temp=(~np.isnan(sgr[i0,:])).astype(int)
    inds1=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==1)[0] # beginning of each trace
    inds2=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==-1)[0] # end point of each trace
    for ind in range(len(inds1)):
        yv=sgr[i0,inds1[ind]:inds2[ind]]
        xv1 = xv[inds1[ind]:inds2[ind]]
        xv1 = xv[inds1[ind]:inds2[ind]]
        plt.xlim(left=0.0,right=np.amax(xv))
        plt.plot(xv1,yv,alpha=0.2,color='b')
yv=np.nanmedian(sgr,axis=0)
err=np.nanstd(sgr,axis=0)/np.sqrt(np.sum(~np.isnan(sgr),axis=0))
plt.fill_between(xv,yv-err,yv+err,alpha=0.5,color='r')
plt.plot(xv,yv,label= 'Smoothed median',color='k',lw=3.0)
# plt.ylim(ymin=0)
ax=plt.gca()
vymin=ax.get_ylim()[0]
vymax=ax.get_ylim()[1]
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0],ymin=vymin,ymax=vymax,label=labels[i0],linestyle=linestyles[i0])
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel(r'$\frac{1}{l}\frac{dl}{dt}$ ($s^{-1}$)')
# plt.title('Growth rate response to antibiotic perturbation')
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr.png',bbox_inches='tight',dpi=150)
plt.xlim(xmax=xv[cutoff_time])
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_truncated_original.png',bbox_inches='tight',dpi=150)
plt.show()
plt.clf()

# Now let's plot the integral of that growth rate to show something like the "population-average relative length change".

xv=time
fig=plt.figure(figsize=[8,5])

yv=np.nanmedian(sgr,axis=0)*dt[0]
start_ind=np.nonzero(~np.isnan(yv))[0][0]
# print(start_ind)
# print(yv,yv.shape)
temp_xv=xv[start_ind:]
lvals=np.exp(np.cumsum(yv[start_ind:]))
# if not(lvals.shape==sgr.shape):
#     exit()
# print('lvals', lvals.shape, lvals)
plt.plot(temp_xv/60.0,lvals,label= 'Relative length',color='k',lw=3.0)
# plt.ylim(ymin=0)
ax=plt.gca()
vymin=ax.get_ylim()[0]
vymax=ax.get_ylim()[1]
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0]/60.0,ymin=vymin,ymax=vymax,label=labels[i0],linestyle=linestyles[i0])
plt.legend(loc=[1.01,0.03])
plt.xlabel('Time (min)')
plt.ylabel('Relative length (a.u.)')
# plt.title('Growth rate response to antibiotic perturbation')
fig.savefig('./outputs/'+expt_id+expt_id+'_rel_len.png',bbox_inches='tight',dpi=150)
plt.show()
plt.clf()

np.save('./outputs/'+expt_id+expt_id+'_sgr_l.npy',sgr)
print("DIMENSIONS", sgr.shape, scene_num_tracker.shape)
np.save('./outputs/'+expt_id+expt_id+'_scene_nums.npy',scene_num_tracker)
np.save('./outputs/'+expt_id+expt_id+'_time.npy',time)
np.save('./outputs/'+expt_id+expt_id+'_width.npy',wcell)
np.save('./outputs/'+expt_id+expt_id+'_sgr_l_pos.npy',sgr_l_pos*3600.0)
np.save('./outputs/'+expt_id+expt_id+'_width_pos.npy',wcell_pos)

# Now we just briefly plot the initial growth rate by chamber number for the first 10 mins
temp_cols=["Scene","Initial Growth Rate"]
temp_df=pd.DataFrame(columns=temp_cols)
temp_cutoff = np.nonzero(time > initial_time_plotting * 60)[0][0]
temp_lcutoff=np.nonzero(time>2*60)[0][0]
print(temp_cutoff)
for cell in range(sgr.shape[0]):
    temp_gr=np.nanmean(sgr[cell,temp_lcutoff:temp_cutoff])
    temp_df1=pd.DataFrame(columns=temp_cols, data=[[str(int(scene_num_tracker[cell])),temp_gr]])
    temp_df=pd.concat([temp_df,temp_df1])
print(temp_df.Scene.unique())
fig=plt.figure(figsize=[8,5])
temp_df=temp_df[[~np.isnan(obj) for obj in temp_df["Initial Growth Rate"]]]
sns.boxplot(data=temp_df,x="Scene",y="Initial Growth Rate")
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_initial_by_scene.png',bbox_inches='tight',dpi=150)
plt.clf()


# Plotting a hexbin version of specific cell growth rates

xv=time
fig=plt.figure(figsize=[8,5])
yv_out = np.array([])
for i0 in range(sgr.shape[0]):
    yv=sgr[i0,:]
    xv1 = xv[:]
    if i0 ==0:
        yv_out=yv
        xv_out=xv1
    else:
        yv_out=np.concatenate([yv,yv_out])
        xv_out = np.concatenate([xv1, xv_out])
plt.xlim(left=0.0,right=np.amax(xv_out))
plt.hexbin(xv_out,yv_out,cmap ='plasma')
yv=np.nanmedian(sgr,axis=0)
plt.plot(xv,yv,label= 'Median',color='g',lw=3.0)
# plt.ylim(ymin=np.nanmedian(sgrmed)-4*np.nanmedian(sgrstd),ymax=np.nanmedian(sgrmed)+6*np.nanmedian(sgrstd))
ax=plt.gca()
vymin=ax.get_ylim()[0]
vymax=ax.get_ylim()[1]
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0],ymin=vymin,ymax=vymax,label=labels[i0],linestyle=linestyles[i0],color='y')
plt.legend(loc=[0.5,0.6])
plt.xlabel('Time (s)')
plt.ylabel(r'$\frac{1}{l}\frac{dl}{dt}$ ($s^{-1}$)')
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_hexbin.png',bbox_inches='tight',dpi=150)
plt.clf()

###################################################################
# plotting the traces of cell widths

fig=plt.figure(figsize=[8,5])
xv=time
for i0 in range(lcell.shape[0]):
    yv=wcell[i0,:]
    plt.plot(xv,yv,alpha=0.4)
ax=plt.gca()
vymin=ax.get_ylim()[0]
vymax=ax.get_ylim()[1]
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0],ymin=vymin,ymax=vymax,label=labels[i0],linestyle=linestyles[i0])
plt.legend(loc=[1.02,0.2])
plt.ylabel(r'Width ($\mu m$)')
plt.xlabel('Time (s)')
fig.savefig('./outputs'+expt_id+expt_id+'_cell_widths.png',dpi=150,bbox_inches='tight')
plt.clf()

# plotting a nicer set of traces of cell widths

wmed = np.nanmedian(wcell,axis=0)
wstd = scipy.stats.iqr(wcell,axis=0)

xv=time[:]
fig=plt.figure(figsize=[8,5])
for i0 in range(wcell.shape[0]):
    temp=(~np.isnan(wcell[i0,:])).astype(int)
    inds1=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==1)[0] # beginning of each trace
    inds2=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==-1)[0] # end point of each trace
    for ind in range(len(inds1)):
        yv=wcell[i0,inds1[ind]:inds2[ind]]
        xv1 = xv[inds1[ind]:inds2[ind]]
        plt.xlim(left=0.0,right=np.amax(xv))
        plt.plot((xv1-t_pert[0])/60.0,yv,alpha=0.1,color='b')
plt.xlim(left=-t_pert[0]/60.0,right=np.amax(xv)/60.0)
plt.plot((xv-t_pert[0])/60.0,wmed,label= 'median',color='r',lw=3.0)
plt.fill_between((xv-t_pert[0])/60.0,wmed-wstd,wmed+wstd,alpha=0.2,color='r',lw=3.0)
# plt.ylim(ymin=np.nanmedian(sgrmed)-0.5*np.nanmedian(sgrstd),ymax=np.nanmedian(sgrmed)+1.0*np.nanmedian(sgrstd))
ax=plt.gca()
vymin=ax.get_ylim()[0]
vymax=ax.get_ylim()[1]
for i0 in range(len(t_pert)):
    plt.vlines((t_pert[i0]-t_pert[0])/60.0,ymin=vymin,ymax=vymax,label=labels[i0],linestyle=linestyles[i0])
plt.legend(loc=[1.02,0.2])
plt.xlabel('Time (min)')
plt.ylabel(r'Cell width ($\mu m$)')
# plt.title('Cell width in response to antibiotic perturbation')
fig.savefig('./outputs/'+expt_id+expt_id+'_widths_dist_v2.png',bbox_inches='tight',dpi=150)
plt.clf()


###################################################################
# plotting the traces of cell SA

fig=plt.figure(figsize=[8,5])
xv=time
for i0 in range(lcell.shape[0]):
    yv=sacell[i0,:]
    plt.plot(xv,yv,alpha=0.4)
ax=plt.gca()
vymin=ax.get_ylim()[0]
vymax=ax.get_ylim()[1]
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0],ymin=vymin,ymax=vymax,label=labels[i0],linestyle=linestyles[i0])
plt.legend(loc=[1.02,0.2])
# plt.ylim(ymin=0.0,ymax=np.nanmedian(sacell.flatten())+4*np.nanmedian(sacell.flatten()))
plt.ylabel(r'Surface Area ($\mu m^2$)')
plt.xlabel('Time (s)')
fig.savefig('./outputs'+expt_id+expt_id+'_cell_sa.png',dpi=150,bbox_inches='tight')
plt.clf()

###################################################################
sgrmed = np.nanmedian(sgr,axis=0)
sgrstd = np.nanstd(sgr,axis=0)
print('Perturbation index', int(t_pert[0]/dt[0]))
print('Cell length growth rate values')
temp=np.nanmean(sgr[:,:int(t_pert[0]/dt[0])],axis=1)*60**2
max_trunc=int(t_pert[0]/dt[0])+int(30*60.0/dt[0]) # gives the required time delay before calculating the final growth rate
print('Pre shock growth rate median cell average', np.around(np.nanmedian(temp),4))
print('Pre shock growth rate SD cell average', np.around(np.nanstd(temp),4))
print('Pre shock growth rate SEM', np.around(np.nanstd(temp)/np.sqrt(np.sum(~np.isnan(temp))),4))
temp=np.nanmean(sgr[:,max_trunc:],axis=1)*60**2
print('Post shock growth rate median cell average', np.around(np.nanmedian(temp),4))
print('Post shock growth rate SD cell average', np.around(np.nanstd(temp),4))
print('Post shock growth rate SEM', np.around(np.nanstd(temp)/np.sqrt(np.sum(~np.isnan(temp))),4))

###################################################################
# Plotting the SA SGR
flier_thresh=2.0
cutoff_thresh=0.0001
sgr=np.zeros(sgr_sa.shape)
sgr[:,:]=sgr_sa[:,:]

# Filtering to remove cells that simply don't grow throughout the whole timelapse
if remove_non_growing_cells:
    filt_inds=np.nonzero(np.nanmean(sgr,axis=1)<thresh)
    sgr[filt_inds,:]=np.nan
sgrmed = np.nanmedian(sgr,axis=0)
sgrstd = scipy.stats.iqr(sgr,axis=0,nan_policy='omit')

xv=time
fig=plt.figure(figsize=[8,5])
for i0 in range(sgr.shape[0]):
    temp=(~np.isnan(sgr[i0,:])).astype(int)
    inds1=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==1)[0] # beginning of each trace
    inds2=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==-1)[0] # end point of each trace
    for ind in range(len(inds1)):
        yv=sgr[i0,inds1[ind]:inds2[ind]]
        xv1 = xv[inds1[ind]:inds2[ind]]
        plt.xlim(left=0.0,right=np.amax(xv))
        plt.plot(xv1,yv,alpha=0.2,color='b')
yv=np.nanmedian(sgr,axis=0)
err=np.nanstd(sgr,axis=0)/np.sqrt(np.sum(~np.isnan(sgr),axis=0))
# plt.ylim(ymin=0)
plt.fill_between(xv,yv-err,yv+err,alpha=0.5,color='r')
plt.plot(xv,yv,label= 'Median',color='k',lw=3.0)

# plt.ylim(ymin=np.nanmedian(sgrmed)-4*np.nanmedian(sgrstd),ymax=np.nanmedian(sgrmed)+6*np.nanmedian(sgrstd))
ax=plt.gca()
vymin=ax.get_ylim()[0]
vymax=ax.get_ylim()[1]
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0],ymin=vymin,ymax=vymax,label=labels[i0],linestyle=linestyles[i0])
plt.legend(loc=[1.02,0.2])
plt.xlabel('Time (s)')
plt.ylabel(r'$\frac{1}{S}\frac{dS}{dt}$ ($s^{-1}$)')
# plt.title('Growth rate response to antibiotic perturbation')
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_SA.png',bbox_inches='tight',dpi=300)
plt.show()
np.save('./outputs/'+expt_id+expt_id+'_sgr_sa.npy',sgr)
plt.clf()

# Plotting a streamlined version of the data
fig=plt.figure(figsize=[8,5])
sgr_sa_smoothed=np.nanmedian(sgr_sa,axis=0)
sgr_sa_iqr_smoothed=scipy.stats.iqr(sgr_sa,axis=0,nan_policy='omit')
plt.plot(time,sgr_sa_smoothed,label='Smoothed Median',lw=3.0)
plt.fill_between(time,sgr_sa_smoothed-sgr_sa_iqr_smoothed,
                 sgr_sa_smoothed+sgr_sa_iqr_smoothed,alpha=0.2)
# plt.ylim(ymin=-0.0005,ymax=0.0015)
ax=plt.gca()
vymin=ax.get_ylim()[0]
vymax=ax.get_ylim()[1]
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0],ymin=vymin,ymax=vymax,label=labels[i0],linestyle=linestyles[i0])

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel(r'$\frac{1}{S}\frac{dS}{dt}$ ($s^{-1}$)')
# plt.title('Growth rate response to antibiotic perturbation')
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_SA_iqr.png',bbox_inches='tight',dpi=300)
plt.clf()

# Plotting growth rate on better axis
trunc_time=time[np.nonzero(time<=max_time_truncation)[0]]
trunc_sgr_sa = sgr_sa[:,np.nonzero(time<=max_time_truncation)[0]]*60*60
fig=plt.figure(figsize=[8,5])
sgr_sa_smoothed=np.nanmedian(trunc_sgr_sa,axis=0)
err=np.nanstd(trunc_sgr_sa,axis=0)/np.sqrt(np.sum(~np.isnan(trunc_sgr_sa),axis=0))
plt.plot(trunc_time/60,sgr_sa_smoothed,label='Median',lw=3.0)
plt.fill_between(trunc_time/60,sgr_sa_smoothed-err,
                 sgr_sa_smoothed+err,alpha=0.2)
# plt.ylim(ymin=0.0,ymax=3)
ax=plt.gca()
vymin=ax.get_ylim()[0]
vymax=ax.get_ylim()[1]
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0]/60.0,ymin=vymin,ymax=vymax,label=labels[i0],linestyle=linestyles[i0])

plt.legend()
plt.xlabel('Time (min)')
plt.ylabel(r'$\frac{1}{S}\frac{dS}{dt}$ ($h^{-1}$)')
# plt.title('Growth rate response to antibiotic perturbation')
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_SA_sem.png',bbox_inches='tight',dpi=300)



print('SA growth rate')
max_trunc=int(t_pert[0]/dt[0])+int(30*60.0/dt[0]) # gives the required time delay before calculating the final growth rate
temp=np.nanmean(trunc_sgr_sa[:,:int(t_pert[0]/dt[0])],axis=1)
print('Pre shock growth rate median cell average', np.around(np.nanmedian(temp),4))
print('Pre shock growth rate SD cell average', np.around(np.nanstd(temp),4))
print('Pre shock growth rate SEM', np.around(np.nanstd(temp)/np.sqrt(np.sum(~np.isnan(temp))),4))
temp=np.nanmean(trunc_sgr_sa[:,max_trunc:],axis=1)
print('Post shock growth rate median cell average', np.around(np.nanmedian(temp),4))
print('Post shock growth rate SD cell average', np.around(np.nanstd(temp),4))
print('Post shock growth rate SEM', np.around(np.nanstd(temp)/np.sqrt(np.sum(~np.isnan(temp))),4))
sys.stdout.close()
plt.clf()

# Plotting the growth rate over time as a function of position for each scene
binsize=31
for scene in scenes:
    # Binning into 30 timepoint windows
    for ind in range(int(len(tvec)/binsize)):
        subcell=np.nonzero(scene_num_tracker==scene)[0]
        subtime=(np.arange(ind*binsize,np.amin([(ind+1)*binsize,len(tvec)])))
        sgr_sel=np.nanmean(sgr[subcell[0]:subcell[-1]+1,subtime[0]:subtime[-1]+1],axis=1)
        # sgr_sel = np.nanmean(sgr_l_pos[subcell[0]:subcell[-1] + 1, subtime[0]:subtime[-1] + 1], axis=1)
        x_sel = np.nanmean(xcents[subcell[0]:subcell[-1]+1,subtime[0]:subtime[-1]+1],axis=1)
        y_sel = np.nanmean(ycents[subcell[0]:subcell[-1]+1,subtime[0]:subtime[-1]+1],axis=1)
        posns=np.nonzero(~np.isnan(sgr_sel))[0]
        fig=plt.figure(figsize=[4,4])
        ax=plt.gca()
        temp_im=np.log(io.imread(base_path+expt_id+expt_id+'_s'+str(scene).zfill(3)+'_1_a'+expt_id+'_s'+str(scene).zfill(3)+'_a'+str((ind+1)*binsize).zfill(4)+'.tif'))
        plt.imshow((temp_im-np.amin(temp_im.flatten()))/(np.amax(temp_im.flatten())-np.amin(temp_im.flatten()))*65536,cmap='Greys')
        plt.axis('off')
        # sns.set(fontscale=1.0)
        ax=sns.scatterplot(data=None,x=x_sel[posns],y=y_sel[posns],hue=np.around(sgr_sel[posns]*3600.0,decimals=2),hue_norm=(0, 2.5))
        ax.set_xlim(0,im_shape[0])
        ax.set_ylim(im_shape[0],0)
        plt.title('Scene: {0}, Time: {1} to {2}'.format(scene,subtime[0],subtime[-1]+1))
        plt.legend(loc=[1.1,0.0],title=r'$\lambda$=$\frac{1}{l}\frac{dl}{dt}$ [h$^{-1}$]')
        plt.axhline(ymin_pos,label='min Y',color='r')
        plt.axhline(ymax_pos,label='max Y',color='r')
        # plt.xlabel('x (pixels)')
        # plt.ylabel('y (pixels)')
        # fig.savefig('outputs'+expt_id+'
        fig.savefig('./outputs'+expt_id+'/scene_growth_visualization'+expt_id+'_s'+str(scene).zfill(3)+'_'+str(ind+1).zfill(2)+'.png',dpi=300,bbox_inches='tight')
        plt.show()
        plt.clf()

# Plotting the positional sgrs on a nice time axis
sgr=np.zeros(sgr_l.shape)
sgr[:,:]=sgr_l_pos[:,:]
sgr*=3600.0
xv=time/60.0
fig=plt.figure(figsize=[8,5])
for i0 in range(sgr.shape[0]):
    temp=(~np.isnan(sgr[i0,:])).astype(int)
    inds1=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==1)[0] # beginning of each trace
    inds2=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==-1)[0] # end point of each trace
    for ind in range(len(inds1)):
        yv=sgr[i0,inds1[ind]:inds2[ind]]
        xv1 = xv[inds1[ind]:inds2[ind]]
        plt.xlim(left=0.0,right=np.amax(xv))
        plt.plot(xv1,yv,alpha=0.2,color='b')
yv=np.nanmedian(sgr,axis=0)
err=np.nanstd(sgr,axis=0)/np.sqrt(np.sum(~np.isnan(sgr),axis=0))
plt.fill_between(xv,yv-err,yv+err,alpha=0.5,color='r')
plt.plot(xv,yv,label= 'Smoothed median',color='k',lw=3.0)
# plt.ylim(ymin=0)
ax=plt.gca()
vymin=ax.get_ylim()[0]
vymax=ax.get_ylim()[1]
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0]/60.0,ymin=vymin,ymax=vymax,label=labels[i0],linestyle=linestyles[i0])

plt.legend()
plt.xlabel('Time (min)')
plt.ylabel(r'$\frac{1}{l}\frac{dl}{dt}$ ($h^{-1}$)')
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_truncated_pos.png',bbox_inches='tight',dpi=150)
plt.clf()