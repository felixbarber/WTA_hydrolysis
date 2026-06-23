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

##################### Experiment specifics
cutoff=None
plot_pert=True


expt_labels = ['bFB291_IPTG_induction', 'bFB292_IPTG_induction', 'bFB295_IPTG_induction']
group_label = 'WT_lytE_cwlO_ponA_OE'
pert = 'IPTG added'
plot_labels=['WT', r'$\Delta lytE$', r'$\Delta cwlO$']
legend_loc=[1.02,0.2]
# plot_pert=False
max_t=85.0

linestyles = ['-','-.','--', 'dotted','-','-.']
out_path = '/Users/barber.527/Documents/GitHub/WTA_hydrolysis/outputs/compiled_data/growth_rate_plots/'
in_path = '/Users/barber.527/Documents/GitHub/WTA_hydrolysis/outputs/compiled_data/'
# plot_labels = []
# expt_label = 'bFB7_Tun_gr'
# expt_label = 'bFB8_Tun_gr'
# expt_label = 'bFB69_Tun_gr'
# expt_label = 'bFB87_Tun_gr'
# expt_label = 'bFB93_Tun_gr'

# loading the experimental parameters
expt_vals=[]
for expt_label in expt_labels:
    temp_path = in_path+expt_label+'_condition_parameters.pkl'
    with open(temp_path, 'rb') as input:
        expt_vals.append(pickle.load(input))
    # plot_labels.append(expt_vals[-1]['Celltype'])
vars = expt_vals[-1]['vars'] # This part should be conserved across all compiled experiments
normed = ['_normed', '']
# ylabels = {'sgr_l': r'$\frac{1}{L}\frac{dL}{dt}$', 'sgr_sa': r'$\frac{1}{S}\frac{dS}{dt}$', 'width':'Width'}
ylabels = {'sgr_l': r'Growth rate $\lambda_l$', 'sgr_l_pos':r'Growth rate $\lambda_l$', 'sgr_sa': r'Growth rate $\lambda_S$', 'width':'Width', 'width_pos':'Width'}
abbrev_ylabels = {'sgr_l': r'$\lambda_l$', 'sgr_l_pos': r'$\lambda_l$', 'sgr_sa': r'$\lambda_S$', 'width':'Width', 'width_pos':'Width'}
yunits = {'sgr_l': r' $(h^{-1})$', 'sgr_sa': r' ($h^{-1}$)', 'width':r' ($\mu$m)','sgr_l_pos': r' $(h^{-1})$', 'width_pos':r' ($\mu$m)'}
sel_vars=['sgr_l', 'width', 'sgr_l_pos', 'width_pos']

with open(out_path + group_label+'.txt', 'w') as f:

    # # print(cutoff is None)
    # sns.set(font_scale=1.5)
    # sns.set_style("whitegrid")
    # for norm in normed:
    #     for var in vars:
    #         fig=plt.figure(figsize=[10,6])
    #         for expt_ind in range(len(expt_labels)):
    #             expt=expt_labels[expt_ind]
    #             xv=np.load(in_path+expt+'_time.npy')
    #             if not (cutoff is None):
    #                 temp_vals = np.nonzero(xv/60<cutoff)[0][-1]
    #             else:
    #                 temp_vals = len(xv)-1
    #             # print(temp_vals)
    #             # xv=xv[:temp_vals] - expt_vals[expt_ind]['t_pert'][-1]
    #             xv = xv[:temp_vals]
    #             yv=np.load(in_path+expt+'_'+var+norm+'.npy')[:,:temp_vals]
    #             sgr_sa_smoothed=scipy.ndimage.gaussian_filter(np.nanmedian(yv,axis=0),sigma=1)
    #             err=scipy.ndimage.gaussian_filter(np.nanstd(yv,axis=0),sigma=1)
    #             plt.plot(xv/60,sgr_sa_smoothed,label=plot_labels[expt_labels.index(expt)],lw=3.0)
    #             plt.fill_between(xv/60,sgr_sa_smoothed-err,sgr_sa_smoothed+err,alpha=0.4)
    #         # if norm==normed[0]:
    #             # plt.ylim(ymin=0.0,ymax=1.5)
    #
    #         ax=plt.gca()
    #         if plot_pert:
    #             plt.vlines(0,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=pert,linestyle=linestyles[0],colors='k')
    #         # plt.legend(loc=[1.02,0.2])
    #         plt.legend()
    #         plt.xlabel('Time (min)')
    #         temp_lab = ylabels[var]
    #         if norm==normed[0]:
    #             temp_lab+=' (norm.)'
    #         else:
    #             temp_lab+=yunits[var]
    #         plt.ylabel(temp_lab)
    #         plt.show()
    #         fig.savefig(out_path+group_label+ '_'+var+norm+'.png',dpi=300, bbox_inches='tight')
    #         plt.clf()
    #
    #
    # ## Now we do some more compact plots for paper figures
    #
    # sns.set(font_scale=2.0)
    # sns.set_style("white")
    # for norm in normed:
    #     for var in sel_vars:
    #         fig = plt.figure(figsize=[10, 6])
    #         for expt in expt_labels:
    #             xv = np.load(in_path + expt + '_time.npy')
    #             if not (cutoff is None):
    #                 temp_vals = np.nonzero(xv/60<cutoff)[0][-1]
    #             else:
    #                 temp_vals = len(xv)-1
    #             # xv=xv[:temp_vals] - expt_vals[expt_ind]['t_pert'][-1]
    #             xv = xv[:temp_vals]
    #             yv = np.load(in_path + expt + '_' + var + norm + '.npy')[:,:temp_vals]
    #             sgr_sa_smoothed = scipy.ndimage.gaussian_filter(np.nanmedian(yv, axis=0), sigma=1)
    #             err = scipy.ndimage.gaussian_filter(np.nanstd(yv, axis=0), sigma=1)
    #             plt.plot(xv / 60, sgr_sa_smoothed, label=plot_labels[expt_labels.index(expt)], lw=3.0)
    #             plt.fill_between(xv / 60, sgr_sa_smoothed - err, sgr_sa_smoothed + err, alpha=0.4)
    #         # if norm==normed[0]:
    #         # plt.ylim(ymin=0.0,ymax=1.5)
    #
    #         ax = plt.gca()
    #         if plot_pert:
    #             plt.vlines(0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], label=pert, linestyle=linestyles[0], colors='k')
    #         # plt.legend(loc=[1.02,0.2])
    #         plt.legend(loc=legend_loc)
    #         plt.xlabel('Time (min)')
    #         temp_lab = ylabels[var]
    #         if norm == normed[0]:
    #             temp_lab += ' (Normalized)'
    #         else:
    #             temp_lab += yunits[var]
    #         plt.ylabel(temp_lab)
    #         plt.show()
    #         fig.savefig(out_path + group_label + '_' + var + norm + '_compact.png', dpi=300, bbox_inches='tight')
    #         plt.legend(loc=[1.02,0.2])
    #         fig.savefig(out_path + group_label + '_' + var + norm + '_compact_legend.png', dpi=300, bbox_inches='tight')
    #         plt.clf()


    # # Paper-style figures.
    # sns.set(font_scale=0.9)
    # sns.set_style("white")
    #
    # for norm in normed:
    #     for var in sel_vars:
    #         fig = plt.figure(figsize=[2.5, 2])
    #         for expt in expt_labels:
    #             xv = np.load(in_path + expt + '_time.npy')
    #             if not (cutoff is None):
    #                 temp_vals = np.nonzero(xv/60<cutoff)[0][-1]
    #             else:
    #                 temp_vals = len(xv)-1
    #             # xv=xv[:temp_vals] - expt_vals[expt_ind]['t_pert'][-1]
    #             xv = xv[:temp_vals]
    #             yv = np.load(in_path + expt + '_' + var + norm + '.npy')[:,:temp_vals]
    #             sgr_sa_smoothed = scipy.ndimage.gaussian_filter(np.nanmedian(yv, axis=0), sigma=1)
    #             err = scipy.ndimage.gaussian_filter(np.nanstd(yv, axis=0), sigma=1)
    #             plt.plot(xv / 60, sgr_sa_smoothed, label=plot_labels[expt_labels.index(expt)], lw=0.5)
    #             plt.fill_between(xv / 60, sgr_sa_smoothed - err, sgr_sa_smoothed + err, alpha=0.4)
    #
    #             print('Generous cell tracks, ', expt, ', ', var, ' :',
    #                   np.sum(np.sum(np.isnan(yv), axis=1) != yv.shape[1]),
    #                   file=f)  # Python 3.x. We count a unique trace if it isn't all nan across the timecourse.
    #         # if norm==normed[0]:
    #         # plt.ylim(ymin=0.0,ymax=1.5)
    #
    #         ax = plt.gca()
    #         if plot_pert:
    #             plt.vlines(0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyle=linestyles[0], colors='k',lw=0.5)
    #         # plt.legend(loc=[1.02,0.2])
    #         # plt.legend()
    #         plt.xlabel('Time (min)')
    #         temp_lab = ylabels[var]
    #         if norm == normed[0]:
    #             temp_lab += ' (Normalized)'
    #         else:
    #             temp_lab += yunits[var]
    #         plt.ylabel(temp_lab)
    #         plt.show()
    #         fig.savefig(out_path + group_label + '_' + var + norm + '_compact.pdf', bbox_inches='tight')
    #         plt.legend(loc=legend_loc)
    #         fig.savefig(out_path + group_label + '_' + var + norm + '_compact_legend.pdf', dpi=300, bbox_inches='tight')
    #         plt.clf()

    # Paper-style figures with shortened axes.
    sns.set(font_scale=0.9)
    sns.set_style("white")

    for norm in normed:
        for var in sel_vars:
            fig = plt.figure(figsize=[2.5, 2])
            for expt in expt_labels:
                xv = np.load(in_path + expt + '_time.npy')
                if not (max_t is None):
                    temp_vals = np.nonzero(xv/60<max_t)[0][-1]
                else:
                    temp_vals = len(xv)-1
                # xv=xv[:temp_vals] - expt_vals[expt_ind]['t_pert'][-1]
                xv = xv[:temp_vals]
                yv = np.load(in_path + expt + '_' + var + norm + '.npy')[:,:temp_vals]
                sgr_sa_smoothed = scipy.ndimage.gaussian_filter(np.nanmedian(yv, axis=0),sigma=1)
                err = scipy.ndimage.gaussian_filter(np.nanstd(yv, axis=0),sigma=1)
                plt.plot(xv / 60, sgr_sa_smoothed, label=plot_labels[expt_labels.index(expt)], lw=0.5)
                plt.fill_between(xv / 60, sgr_sa_smoothed - err, sgr_sa_smoothed + err, alpha=0.4)

                print('Generous cell tracks, ', expt, ', ', var, ' :',
                      np.sum(np.sum(np.isnan(yv), axis=1) != yv.shape[1]),
                      file=f)  # Python 3.x. We count a unique trace if it isn't all nan across the timecourse.
            # if norm==normed[0]:
            # plt.ylim(ymin=0.0,ymax=1.5)

            ax = plt.gca()
            if plot_pert:
                plt.vlines(0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyle=linestyles[0], colors='k',lw=0.5)
            # plt.legend(loc=[1.02,0.2])
            # plt.legend()
            plt.xlabel('Time (min)')
            temp_lab = ylabels[var]
            if norm == normed[0]:
                temp_lab += ' (Normalized)'
            else:
                temp_lab += yunits[var]
            plt.ylabel(temp_lab)
            # plt.show()
            # fig.savefig(out_path + group_label + '_' + var + norm + '_compact_xlim.pdf', bbox_inches='tight')
            plt.legend(loc=legend_loc)
            fig.savefig(out_path + group_label + '_' + var + norm + '_compact_legend_xlim.pdf', dpi=300, bbox_inches='tight')
            plt.clf()


