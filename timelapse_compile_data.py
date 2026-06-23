# import scipy
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

out_path='/Users/barber.527/Documents/GitHub/WTA_hydrolysis/outputs/compiled_data/'
in_path='/Users/barber.527/Documents/GitHub/WTA_hydrolysis/outputs/'

##################### Experiment specifics

##################### Values
tstep=20.0
# Note: If t_pert<0 then normalization is not going to be very helpful.

expt_ids = ['/260612_bFB295_IPTG_Mg']
expt_label = 'bFB295_IPTG_induction_Mg_shock'
Celltype = r'$P_{spac} ponA$ LB->LB+1mM IPTG->LB+1mM IPTG + 10mM MgCl$_2$'
t_pert = [5*60.0,5*60.0]
t_pert1 = [65*60.0]

# expt_ids = ['/250324_bFB291_IPTG_induction', '/250408_bFB291_IPTG_induction', '/260519_bFB291_IPTG']
# expt_ids = ['/250324_bFB291_IPTG_induction', '/250408_bFB291_IPTG_induction']
# expt_label = 'bFB291_IPTG_induction'
# Celltype = r'$P_{spac} ponA$ LB->LB+1mM IPTG'
# t_pert = [10*60.0, 10*60.0]
# t_pert1 = []

# expt_ids = ['/260416_bFB295_IPTG_induction', '/260306_bFB295_IPTG']
# expt_label = 'bFB295_IPTG_induction'
# Celltype = r'$P_{spac} ponA$ $\Delta cwlO$ LB->LB+1mM IPTG'
# t_pert = [10*60.0,10*60.0]
# t_pert1 = []

# expt_ids = ['/260313_bBF292_IPTG', '/260414_bFB292_IPTG','/260305_bFB292_IPTG']
# expt_label = 'bFB292_IPTG_induction'
# Celltype = r'$P_{spac} ponA$ $\Delta lytE$ LB->LB+1mM IPTG'
# t_pert = [10*60.0,10*60.0,10*60.0]
# t_pert1 = []

# expt_ids = ['/260216_bFB292_IPTG_Mg', '/260217_bFB292_IPTG_Mg']
# expt_label = 'bFB292_IPTG_induction_Mg_shock'
# Celltype = r'$P_{spac} ponA$ LB->LB+1mM IPTG->LB+1mM IPTG + 10mM MgCl$_2$'
# t_pert = [5*60.0,5*60.0,5*60.0, 5*60.0]
# t_pert1 = [65*60.0,65*60.0,65*60.0,65*60.0]

# expt_ids = ['/260521_bFB292_IPTG_Mg', '/260527_bFB292_IPTG_Mg', '/260612_bFB292_IPTG_Mg', '/260611_bFB292_IPTG_Mg']
# expt_label = 'bFB292_IPTG_induction_Mg_shock_recent'
# Celltype = r'$P_{spac} ponA$ LB->LB+1mM IPTG->LB+1mM IPTG + 10mM MgCl$_2$'
# t_pert = [5*60.0,5*60.0,5*60.0,5*60.0,5*60.0]
# t_pert1 = [65*60.0,65*60.0,65*60.0,65*60.0,5*60.0]


# expt_ids = ['/250613_bFB66_s750_tun_gr', '/250612_bFB66_s750_tun_gr']
# expt_label = 'bFB66_s750_tun'
# Celltype = r'WT s750'
# t_pert = [20.0*60.0, 20.0*60.0]
# t_pert1 = []
# tstep=60.0

# expt_ids = ['/250508_bFB69_pad_LB', '/250514_bFB69_LB_pad_rep2', '/250514_bFB69_LB_pad']
# expt_label = 'bFB69_LB_pad'
# Celltype = r'$\Delta ponA$'
# t_pert = [-200.0, -200.0,-200.0]
# t_pert1 = []

# expt_ids = ['/250508_bFB66_pad_LB', '/250509_bFB66_pad_LB', '/250515_bFB66_LB_pad', '/250512_bFB66_LB_pad']
# expt_label = 'bFB66_LB_pad'
# Celltype = 'WT'
# t_pert = [-200.0, -200.0, -200.0,-200.0]
# t_pert1 = []

# expt_ids = ['/250507_bFB66_pad_tun_v2', '/250512_bFB66_tun_pad', '/250515_bFB66_tun_pad']
# expt_label = 'bFB66_tun_pad'
# Celltype = 'WT Tunicamycin'
# t_pert = [-500.0,-300.0, -200.0]
# t_pert1 = []

#
# expt_ids = ['/250402_bFB66_LB']
# expt_label = 'bFB66_long_time_LB'
# Celltype = r'WT'
# t_pert = [60.0]
# t_pert1 = []









# expt_ids = ['/250227_bFB87_Tun_PBS_long_incubation_45min']
# expt_label = 'bFB87_45minTun_PBS_long_incubation'
# Celltype = r'$\Delta ponA$ $\Delta cwlO$ 45min Tun->PBS+Buffer A'
# t_pert = [55*60.0]
# t_pert1 = [100*60.0]

# expt_ids = ['/250226_bFB66_Tun_PBS_long_incubation_45min']
# expt_label = 'bFB66_45minTun_PBS_long_incubation'
# Celltype = r'WT 45min Tun->PBS'
# t_pert = [55*60.0]
# t_pert1 = [100*60.0]

# expt_ids = ['/250225_bFB66_Tun_PBS_long_incubation_30min']
# expt_label = 'bFB66_30minTun_PBS_long_incubation'
# Celltype = r'WT 30min Tun->PBS'
# t_pert = [40*60.0]
# t_pert1 = [100*60.0]

# expt_ids = ['/250224_bFB66_Tun_BufferA_long_incubation']
# expt_label = 'bFB66_Tun_BufferA_long_incubation'
# Celltype = r'WT Tun->PBS+Buffer A'
# t_pert = [30*60.0]
# t_pert1 = [90*60.0]

# expt_ids = ['/250221_bFB69_Tun_PBS_long_incubation']
# expt_label = 'bFB69_Tun_PBS_60min_rec'
# Celltype = r'$\Delta ponA$ Tun->PBS'
# t_pert = [30*60.0]
# t_pert1 = [90*60.0]

# expt_ids = ['/250220_bFB66_Tun_PBS_long_incubation']
# expt_label = 'bFB66_Tun_PBS_60min_rec'
# Celltype = 'Tun->PBS'
# t_pert = [30*60.0]
# t_pert1 = [90*60.0]


# expt_ids = ['/250218_bFB66_PBS_long_incubation']
# expt_label = 'bFB66_PBS_60min_rec'
# Celltype = 'PBS treated'
# t_pert = [10*60.0]
# t_pert1 = [70*60.0]



# expt_ids = ['/250213_bFB8_PBS_BufferA_recovery']
# expt_label = 'bFB8_Buffer_rec'
# Celltype = 'PBS treated'
# t_pert = [10*60.0]
# t_pert1 = [25*60.0]

# expt_ids = ['/250213_bFB8_25uMGlpQ_recovery']
# expt_label = 'bFB8_25uMGlpQ_rec'
# Celltype = '25uM GlpQ treated'
# t_pert = [10*60.0]
# t_pert1 = [25*60.0]

# expt_ids = ['/241202_bFB7_25uMGlpQ_recovery']
# expt_label = 'bFB7_25uMGlpQ_rec'
# Celltype = '25uM GlpQ treated'
# t_pert = [10*60.0]
# t_pert1 = [25*60.0]


# expt_ids = ['/240920_bFB93_40uMGlpQ_recovery', '/240925_bFB93_40uMGlpQ_recovery_v2', '/240925_bFB93_40uMGlpQ_recovery',
#             '/241201_bFB93_40uMGlpQ_recovery']
# expt_label = 'bFB93_40uMGlpQ_rec'
# Celltype = '40uM GlpQ treated'
# t_pert = [10*60.0,10*60.0,10*60.0,10*60.0]
# t_pert1 = [25*60.0,25*60.0,25*60.0,25*60.0]

# expt_ids = ['/240920_bFB93_PBS_recovery']
# expt_label = 'bFB93_PBS_rec'
# Celltype = 'PBS treated'
# t_pert = [10*60.0]
# t_pert1 = [25*60.0]


# expt_ids = ['/240920_bFB93_GlpQ_Recovery', '/241130_bFB93_25uMGlpQ_recovery', '/241201_bFB93_25uMGlpQ_recovery']
# expt_label = 'bFB93_PBS_25uM_newGlpQ_rec'
# Celltype = 'GlpQ treated'
# t_pert = [10*60.0, 10*60.0, 10*60.0]
# t_pert1 = [25*60.0, 25*60.0, 25*60.0]


# expt_ids = ['/240724_bFB69_25uM_GlpQ_recovery']
# expt_label = 'bFB69_PBS_25uM_newGlpQ_rec'
# Celltype = 'GlpQ treated'
# t_pert = [10*60.0]
# t_pert1 = [25*60.0]

# expt_ids = ['/240724_bFB66_25uM_GlpQ_recovery']
# expt_label = 'bFB66_PBS_25uM_newGlpQ_rec'
# Celltype = 'GlpQ treated'
# t_pert = [10*60.0]
# t_pert1 = [25*60.0]

# expt_ids = ['/240918_bFB87_PBS_recovery']
# expt_label = 'bFB87_PBS_rec_25uM'
# Celltype = 'PBS treated'
# t_pert = [10*60.0]
# t_pert1 = [25*60.0]

# expt_ids = ['/240918_bFB87_GlpQ_recovery', '/241115_bFB87_GlpQ_recovery', '/241129_bFB87_25uMGlpQ_recovery']
# expt_label = 'bFB87_PBS_GlpQ_rec_25uM'
# Celltype = 'GlpQ treated'
# t_pert = [10*60.0, 10*60.0, 10*60.0]
# t_pert1 = [25*60.0, 25*60.0, 25*60.0]
#
#
# expt_ids = ['/231030_bFB66_PBS_BufferA_recovery_30min', '/240901_bFB66_30minPBS_recovery', '/240901_bFB66_30minPBS_recovery_run2']
# # expt_ids = ['/240901_bFB66_30minPBS_recovery', '/240901_bFB66_30minPBS_recovery_run2']
# expt_label = 'bFB66_30minPBS_rec_only'
# Celltype = 'PBS 30min'
# t_pert = [10*60.0, 10*60.0, 10*60.0]
# t_pert1 = [40*60.0, 40*60.0, 40*60.0]

# expt_ids = ['/240527_bFB66_GlpQ_tester', '/240725_bFB66_40uM_GlpQ_recovery', '/240807_bF66_40uMGlpQ_recovery']
# expt_label = 'bFB66_PBS_GlpQ_rec_hidose'
# Celltype = 'GlpQ treated'
# t_pert = [10*60.0,10*60.0,10*60.0]
# t_pert1 = [25*60.0,25*60.0,25*60.0]

# expt_ids = ['/231105_bFB69_PBS_BufferA_500mMSorbitol_recovery_15min',
#             '/231119_bFB69_PBS_BufferA_500mMSorbitol_recovery_15min_ZPBS',
#             '/240528_bFB69_BufferA_500mMSorbitol',
#             '/231129_bFB69_PBS_BufferA_500mMSorbitol_15min_recovery']
# expt_label = 'bFB69_PBS_bufferA_500mMSorb_rec'
# Celltype = 'Buffer, Sorbitol treated'
# t_pert = [10*60.0, 10*60.0, 10*60.0, 10*60.0]
# t_pert1 = [25*60.0, 25*60.0, 25*60.0, 25*60.0]

# expt_ids = ['/231110_bFB69_PBS_GlpQ_500mMSorbitol_recovery_15min_ZPBS', '/231103_bFB69_PBS_GlpQ_500mMSorbitol_recovery_15min',
#             '/231117_bFB69_PBS_GlpQ_500mMSorbitol_recovery_15min_ZPBS']
# expt_label = 'bFB69_PBS_GlpQ_500mMSorb_rec'
# Celltype = 'GlpQ, Sorbitol treated'
# t_pert = [10*60.0, 10*60.0, 10*60.0]
# t_pert1 = [25*60.0, 25*60.0, 25*60.0]


# expt_ids = ['/230921_bFB66_PBS_BufferA_recovery_15min']
# expt_label = 'bFB66_PBS_BufferA_rec_only'
# Celltype = 'Buffer treated only'
# t_pert = [10*60.0]
# t_pert1 = [25*60.0]

# expt_ids = ['/230926_bFB66_PBS_GlpQdenat_recovery_15min']
# expt_label = 'bFB66_PBS_GlpQdenat_rec_only'
# Celltype = 'Denatured treated only'
# t_pert = [10*60.0]
# t_pert1 = [25*60.0]

# expt_ids = ['/210916_bFB1_PBS_teichoicase_recovery']
# expt_label = 'bFB1_PBS_GlpQ_PhoD_rec_20min'
# Celltype = 'Teichoicase treated'
# t_pert = [2*60.0]
# t_pert1 = [22*60.0]

# expt_ids = ['/210916_bFB1_PBS_recovery']
# expt_label = 'bFB1_PBS_rec_20min'
# Celltype = 'PBS treated'
# t_pert = [10*60.0]
# t_pert1 = [30*60.0]

# expt_ids = ['/231004_bFB69_PBS_BufferA_recovery_15min', '/231027_bFB69_PBS_BufferA_recovery_15min',
#             '/231115_bFB69_PBS_BufferA_recovery_15min_ZPBS']
# expt_label = 'bFB69_PBS_Buffer_rec'
# Celltype = r'$\Delta ponA$ Buffer treated'
# t_pert = [10*60.0, 10*60.0, 10*60.0]
# t_pert1 = [25*60.0, 25*60.0, 25*60.0]

# expt_ids = ['/231012_bFB69_PBS_GlpQ_recovery_15min', '/231018_bFB69_PBS_GlpQ_recovery_15min', '/231108_bFB69_PBS_GlpQ_recovery_15min_ZPBS']
# expt_label = 'bFB69_PBS_GlpQ_rec'
# Celltype = 'GlpQ treated'
# t_pert = [10*60.0, 10*60, 10*60.0]
# t_pert1 = [25*60.0, 25*60.0, 25*60.0]

# expt_ids = ['/230921_bFB66_PBS_BufferA_recovery_15min', '/230926_bFB66_PBS_GlpQdenat_recovery_15min',
#             '/231103_bFB66_PBS_denatGlpQ_recovery_15min']
# expt_label = 'bFB66_PBS_GlpQdenat_rec'
# Celltype = 'Buffer treated'
# t_pert = [10*60.0,10*60.0, 10*60.0]
# t_pert1 = [25*60.0, 25*60.0, 25*60.0]

# expt_ids = ['/230921_bFB66_PBS_GlpQ_recovery_15min', '/230926_bFB66_PBS_GlpQ_recovery_15min',
#             '/231004_bFB66_PBS_GlpQ_recovery_15min', '/231011_bFB66_PBS_GlpQ_recovery_15min']
# expt_label = 'bFB66_PBS_GlpQ_rec'
# Celltype = 'GlpQ treated'
# t_pert = [10*60.0, 10*60.0, 10*60.0, 10*60.0]
# t_pert1 = [25*60.0, 25*60.0, 25*60.0, 25*60.0]

# expt_ids = ['/220121_bFB66_Tun_gr', '/211216_bFB66_Tun_gr', '/211215_bFB66_Tun_gr']
# expt_label = 'bFB66_Tun_gr'
# Celltype = 'WT'
# t_pert = [5*60.0,5*60.0,5*60.0]

# expt_ids = ['/210429_FB6_inducer_loss', '/210506_FB6_inducer_loss_timelapse', '/231110_bFB6_Xylose_depletion']
# expt_label = 'bFB6_Xylose_depletion'
# t_pert = [5*60.0, 2*60.0, 10*60.0]c
# Celltype = r'$P_{Xylose}-TagO$'
# #

# # expt_ids = ['/210723_FB7_Tun_Response','/211020_bFB7_Tun','/211215_bFB7_Tun_gr', '/240925_bFB7_Tun_gr', '/240926_bFB7_Tun_gr']
# # expt_label = 'bFB7_Tun_gr'
# # t_pert = [10*60.0,10*60.0,5*60.0,10*60.0,10*60.0]
# expt_ids = ['/240925_bFB7_Tun_gr','/240926_bFB7_Tun_gr', '/250519_bFB7_tun_gr']
# expt_label = 'bFB7_Tun_gr'
# t_pert = [10*60.0,10*60.0,10*60.0]
# Celltype = r'$\Delta cwlO$'
# #
# expt_ids = ['/211020_bFB8_Tun', '/211216_bFB8_Tun_gr', '/210723_FB8_Tun_Response'] # Note not to use 7/23 data for this since there was an air bubble
# expt_label = 'bFB8_Tun_gr'
# t_pert = [10*60.0,5*60.0, 10*60.0]
# Celltype = r'$\Delta lytE$'
# #
# expt_ids = ['/220204_bFB69_Tun_gr', '/220309_bFB69_Tun_gr', '/221012_bFB69_Tun_gr', '/240927_bFB69_Tun_gr', '/241126_bFB69_Tun_gr']
# # expt_ids = ['/240927_bFB69_Tun_gr']
# expt_label = 'bFB69_Tun_gr'
# t_pert = [5*60.0,5*60.0, 5*60.0, 10*60.0, 5.0*60.0]
# Celltype = r'$\Delta ponA$'
#
# expt_ids = ['/220303_bFB87_Tun_gr','/220311_bFB87_Tun_gr']
# expt_label = 'bFB87_Tun_gr'
# t_pert = [5*60.0,5*60.0]
# Celltype = r'$\Delta ponA$, $\Delta cwlO$'
#
# expt_ids = ['/220304_bFB93_Tun_gr', '/220310_bFB93_Tun_gr']
# expt_label = 'bFB93_Tun_gr'
# t_pert = [5*60.0, 5*60.0]
# Celltype = r'$\Delta ponA$, $\Delta lytE$'
#

#
# expt_ids = ['/220105_bFB79_Tun_response']
# expt_label = 'bFB79_Tun_gr'
# t_pert = [5*60.0]
# Celltype = r'$\Delta dacA$'
#
# expt_ids = ['/220909_bFB145_Tun_gr', '/220831_bFB145_Tun_gr', '/220916_bFB145_Tun_gr', '/221004_bFB145_Tun_gr']
# expt_label = 'bFB145_Tun_gr'
# t_pert = [5*60.0,5*60.0, 5*60.0, 5*60.0]
# Celltype = r'$PonA_{S390A}$'
#
# expt_ids = ['/220902_bFB149_Tun_gr', '/221005_bFB149_Tun_gr']
# expt_label = 'bFB149_Tun_gr'
# t_pert = [5*60.0, 5*60.0]
# Celltype = r'$PonA$'

# expt_ids = ['/221021_bFB113_Tun_gr', '/221025_bFB113_Tun_gr']
# expt_label = 'bFB113_Tun_gr'
# t_pert = [5*60.0, 5*60.0]
# Celltype = r'168 $\Delta pbpC$'

# expt_ids = ['/221021_bFB109_Tun_gr', '/221025_bFB109_Tun_gr']
# expt_label = 'bFB109_Tun_gr'
# t_pert = [5*60.0, 5*60.0]
# Celltype = r'168 $WT$'

# expt_ids = ['/221025_bFB118_Tun_gr']
# expt_label = 'bFB118_Tun_gr'
# t_pert = [5*60.0]
# Celltype = r'168 $\Delta pbpX$'

# expt_ids = ['/221110_bFB66_Amp_gr']
# expt_label = 'bFB66_Amp_gr'
# t_pert = [5*60.0]
# Celltype = r'PY79 Ampicillin'

# expt_ids = ['/221110_bFB66_Tun_Amp_gr']
# expt_label = 'bFB66_Tun_Amp_gr'
# t_pert = [5*60.0]
# Celltype = r'PY79 Tunicamycin and Ampicillin'

# expt_ids = ['/230314_bFB169_Mecillinam']
# expt_label = 'bFB169_Mec_gr'
# t_pert = [10*60.0]
# Celltype = r'PY79 $\Delta pbpH$ Mecillinam'

# expt_ids = ['/230314_bFB169_Mecillinam_Tun']
# expt_label = 'bFB169_Mec_Tun_gr'
# t_pert = [20*60.0]
# Celltype = r'PY79 $\Delta pbpH$ Mecillinam + Tunicamycin'

# expt_ids = ['/230331_bER47_Mecillinam_LB']
# expt_label = 'bER47_Mec'
# Celltype = 'Pxyl-FtsZ Mecillinam'
# t_pert = [10*60.0]

# expt_ids = ['/230331_bER47_Mecillinam_Tun_LB']
# expt_label = 'bER47_Mec_Tun'
# Celltype = 'Pxyl-FtsZ Mecillinam Tunicamycin'
# t_pert = [30*60.0]

# expt_ids = ['/230404_bER47_Tun']
# expt_label = 'bER47_Tun_gr'
# Celltype = 'Pxyl-FtsZ Tunicamycin'
# t_pert = [30*60.0]  #Note, this is not when tunicamycin is added, but rather the equivalent timepoint for
# comparison with bER47_Mec_Tun



var_ls=['sgr_l', 'sgr_sa', 'width', 'sgr_l_pos', 'width_pos']
yscale = [60*60.0, 60*60.0, 1.0, 1.0, 1.0]
# saving the experimental parameters
temp = {'expt_label':expt_label, 't_pert':t_pert, 'expt_ids':expt_ids, 'tstep':tstep,'vars':var_ls, 'Celltype':Celltype}
if 't_pert1' in locals():
    temp['t_pert1']=t_pert1
temp_path = out_path+expt_label+'_condition_parameters.pkl'
with open(temp_path, 'wb') as output:  # Overwrites any existing file.
    pickle.dump(temp, output, pickle.HIGHEST_PROTOCOL)

def compile_values_v1(temp_yvs, temp_xvs, show_out=False):
    # This function takes lists of arrays for each experiment and aligns them
    # this gives the maximal range for each data set around the perturbation timepoint. 
    # Assumes the same timestep in each case
    temp1=[temp[0] for temp in temp_xvs]
    temp2=[temp[-1] for temp in temp_xvs]
    temp_xv=np.linspace(np.amin(temp1), np.amax(temp2), int((np.amax(temp2)-np.amin(temp1))/tstep+1))
    temp_out=np.empty([np.sum([int(temp.shape[0]) for temp in temp_yvs]), len(temp_xv)])
    temp_out[:]=np.nan
    num_points=0
    for i0 in range(len(temp_yvs)):
        start_ind = np.nonzero(temp_xv==temp_xvs[i0][0])[0][0]
        end_ind = np.nonzero(temp_xv==temp_xvs[i0][-1])[0][0]
        if show_out:
            print(start_ind, end_ind)
            print(temp_out.shape)
            print(temp_yvs[i0].shape)
            print(num_points)
        temp_out[num_points:num_points+temp_yvs[i0].shape[0], start_ind:end_ind+1]=temp_yvs[i0][:,:]
        num_points+=temp_yvs[i0].shape[0]
    return(temp_out, temp_xv)


def compile_values_v2(temp_yvs, temp_xvs, show_out=False):
    # This function takes lists of arrays for each experiment and aligns them conservatively, excluding timepoints with
    # incomplete coverage
    # Assumes the same timestep in each case
    temp1=[temp[0] for temp in temp_xvs]
    temp2=[temp[-1] for temp in temp_xvs]
    temp_xv = np.linspace(np.amax(temp1), np.amin(temp2), int((np.amin(temp2) - np.amax(temp1)) / tstep + 1))
    temp_out=np.empty([np.sum([int(temp.shape[0]) for temp in temp_yvs]), len(temp_xv)])
    temp_out[:]=np.nan
    num_points=0
    for i0 in range(len(temp_yvs)):
        start_ind = np.nonzero(temp_xvs[i0]>=temp_xv[0])[0][0]
        end_ind = np.nonzero(temp_xvs[i0]>=temp_xv[-1])[0][0]
        if show_out:
            print(start_ind, end_ind)
            print(temp_out.shape)
            print(temp_yvs[i0].shape)
            print(num_points)
        temp_out[num_points:num_points+temp_yvs[i0].shape[0], :] = temp_yvs[i0][:, start_ind:end_ind+1]
        num_points += temp_yvs[i0].shape[0]
    return(temp_out, temp_xv)

# Code
for var in var_ls:
    xvs,yvs=[],[]
    for expt_id in expt_ids:
        temp=np.load(in_path+expt_id+expt_id+'_time.npy')
        temp -=t_pert[expt_ids.index(expt_id)]
        xvs.append(temp)
        temp1=np.load(in_path+expt_id+expt_id+'_'+var+'.npy')*yscale[var_ls.index(var)]
        if not(t_pert[expt_ids.index(expt_id)]<0): # if the normalization time is not less than zero
            temp1*=1.0/np.nanmedian(temp1[:,np.nonzero(temp==0)[0][0]]) # Normalization step for each timelapse separately
        else:
            temp1 *= 1.0 / np.nanmedian(
                temp1[:, 0])  # Normalization step for each timelapse separately
        yvs.append(temp1)
    yv, xv=compile_values_v1(yvs,xvs)
    np.save(out_path+expt_label+'_'+var+'_normed.npy', yv)
    del(xv, yv, xvs, yvs)

# Repeating the above but without the normalization step
for var in var_ls:
    xvs,yvs=[],[]
    for expt_id in expt_ids:
        temp=np.load(in_path+expt_id+expt_id+'_time.npy')
        temp -=t_pert[expt_ids.index(expt_id)]
        xvs.append(temp)
        temp1=np.load(in_path+expt_id+expt_id+'_'+var+'.npy')*yscale[var_ls.index(var)]
        yvs.append(temp1)
    yv, xv=compile_values_v1(yvs,xvs)
    np.save(out_path+expt_label+'_'+var+'.npy', yv)
np.save(out_path+expt_label+'_time.npy', xv)

# Generating conservative datasets
for var in var_ls:
    xvs,yvs=[],[]
    for expt_id in expt_ids:
        temp=np.load(in_path+expt_id+expt_id+'_time.npy')
        temp -=t_pert[expt_ids.index(expt_id)]
        xvs.append(temp)
        temp1=np.load(in_path+expt_id+expt_id+'_'+var+'.npy')*yscale[var_ls.index(var)]
        if not(t_pert[expt_ids.index(expt_id)] < 0):  # if the normalization time is not less than zero
            temp1 *= 1.0 / np.nanmedian(
                temp1[:, np.nonzero(temp == 0)[0][0]])  # Normalization step for each timelapse separately
        else:
            temp1 *= 1.0 / np.nanmedian(
                temp1[:, 0])  # Normalization step for each timelapse separately
        yvs.append(temp1)
    yv, xv=compile_values_v2(yvs,xvs)
    # Filtering out rows that were only tracked in the timepoints we have now removed.
    # temp_yv=np.sum(np.isnan(yv),axis=1)
    # yv1=yv[np.nonzero(temp_yv!=yv.shape[1])[0],:]
    # # print(yv1)
    # print(yv[np.nonzero(temp_yv==yv.shape[1])[0],:])
    # print(yv.shape, yv1.shape, xv.shape)
    np.save(out_path+expt_label+'_'+var+'_normed_conservative.npy', yv)

for var in var_ls:
    xvs,yvs=[],[]
    for expt_id in expt_ids:
        temp=np.load(in_path+expt_id+expt_id+'_time.npy')
        temp -=t_pert[expt_ids.index(expt_id)]
        xvs.append(temp)
        temp1=np.load(in_path+expt_id+expt_id+'_'+var+'.npy')*yscale[var_ls.index(var)]
        yvs.append(temp1)
    yv, xv=compile_values_v2(yvs,xvs)
    # Filtering out rows that were only tracked in the timepoints we have now removed.
    # temp_yv=np.sum(np.isnan(yv),axis=1)
    # yv1=yv[np.nonzero(temp_yv!=yv.shape[1])[0],:]
    # print(yv1)
    # print(yv[np.nonzero(temp_yv==yv.shape[1])[0],:])
    # print(yv.shape, yv1.shape, xv.shape)
    np.save(out_path+expt_label+'_'+var+'_conservative.npy', yv)
np.save(out_path+expt_label+'_time_conservative.npy', xv)