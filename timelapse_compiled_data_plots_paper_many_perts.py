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

# expt_labels = ['bFB66_Tun_NLS_PBS_Mg']
# group_label = 'WT_Tun_Mg_lysis'
# perts = ['LB+0.5ug/mL Tunicamycin', 'PBS+5% NLS', r'PBS', r'PBS + 10mM MgCl$_2$', r'PBS', r'PBS + 10mM MgCl$_2$']
# t_pert = [(17-1)*20.0, (107-1)*20.0, (113-1)*20.0, (128-1)*20.0, (143-1)*20.0, (158-1)*20.0]
# plot_labels=[r'WT Tun']
# pert_comp_ind=1 # The index for the NLS perturbation
# post_lysis_ind=2 # The index for the first condition post NLS

# expt_labels = ['bFB66_NLS_PBS_Mg']
# group_label = 'WT_Mg_lysis'
# perts = ['PBS+5% NLS', r'PBS', r'PBS + 10mM MgCl$_2$', r'PBS', r'PBS + 10mM MgCl$_2$']
# t_pert = [(32-1)*20.0, (38-1)*20.0, (53-1)*20.0, (68-1)*20.0, (83-1)*20.0]
# plot_labels=[r'WT']
# pert_comp_ind=0 # The index for the NLS perturbation
# post_lysis_ind=1 # The index for the first condition post NLS


legend_loc=[1.02,0.2]
linestyles = ['-','-.','--', 'dotted','-','-.', '--']
out_path = '/Users/barber.527/Documents/GitHub/WTA_hydrolysis/outputs/compiled_data/growth_rate_plots/'
in_path = '/Users/barber.527/Documents/GitHub/WTA_hydrolysis/outputs/compiled_data/'


# loading the experimental parameters
expt_vals=[]
for expt_label in expt_labels:
    temp_path = in_path+expt_label+'_condition_parameters.pkl'
    with open(temp_path, 'rb') as input:
        expt_vals.append(pickle.load(input))
vars = expt_vals[-1]['vars'] # This part should be conserved across all compiled experiments
normed = [''] # No normalization for this.
ylabels = {'sgr_l': r'Growth rate $\lambda_l$', 'sgr_sa': r'Growth rate $\lambda_S$', 'width':'Width'}
abbrev_ylabels = {'sgr_l': r'$\lambda_l$', 'sgr_sa': r'$\lambda_S$', 'width':'Width'}
yunits = {'sgr_l': r' $(h^{-1})$', 'sgr_sa': r' ($h^{-1}$)', 'width':r' ($\mu$m)'}
sel_vars=['sgr_l', 'width']

print(expt_vals[-1])
with open(out_path + group_label+'.txt', 'w') as f:

    # print(cutoff is None)
    sns.set(font_scale=2.0)
    sns.set_style("whitegrid")
    for norm in normed:
        for var in vars:
            fig=plt.figure(figsize=[10,6])
            for expt_ind in range(len(expt_labels)):
                expt=expt_labels[expt_ind]
                xv=np.load(in_path+expt+'_time.npy')
                if not (cutoff is None):
                    temp_vals = np.nonzero(xv/60<cutoff)[0][-1]
                else:
                    temp_vals = len(xv)-1
                # print(temp_vals)
                # xv=xv[:temp_vals] - expt_vals[expt_ind]['t_pert'][-1]
                xv = xv[:temp_vals]
                yv=np.load(in_path+expt+'_'+var+norm+'.npy')[:,:temp_vals]
                sgr_sa_smoothed=scipy.ndimage.gaussian_filter(np.nanmedian(yv,axis=0),sigma=1)
                err=scipy.ndimage.gaussian_filter(np.nanstd(yv,axis=0),sigma=2)
                plt.plot(xv/60,sgr_sa_smoothed,label=plot_labels[expt_labels.index(expt)],lw=3.0)
                plt.fill_between(xv/60,sgr_sa_smoothed-err,sgr_sa_smoothed+err,alpha=0.4)
                print('Total cell count: ', expt, var , np.sum(np.amax(~np.isnan(yv), axis=1)), file=f)
            # if norm==normed[0]:
                # plt.ylim(ymin=0.0,ymax=1.5)

            ax=plt.gca()
            ymin,ymax=ax.get_ylim()
            for temp_ind in range(len(perts)):
                plt.vlines((expt_vals[-1]['t_perts'][temp_ind]-expt_vals[-1]['t_perts'][pert_comp_ind])/60.0,ymin=ymin,ymax=ymax,label=perts[temp_ind],linestyle=linestyles[temp_ind],colors='k',linewidth=0.5)
            # plt.legend(loc=[1.02,0.2])
            plt.legend()
            plt.xlabel('Time (min)')
            temp_lab = ylabels[var]+yunits[var]

            plt.ylabel(temp_lab)
            plt.show()
            fig.savefig(out_path+group_label+ '_'+var+norm+'.png',dpi=300, bbox_inches='tight')
            plt.clf()


    ## Now we do some more compact plots for paper figures

    sns.set(font_scale=2.0)
    sns.set_style("white")
    for norm in normed:
        for var in sel_vars:
            fig = plt.figure(figsize=[10,6])
            for expt in expt_labels:
                xv = np.load(in_path + expt + '_time.npy')
                if not (cutoff is None):
                    temp_vals = np.nonzero(xv/60<cutoff)[0][-1]
                else:
                    temp_vals = len(xv)-1
                # xv=xv[:temp_vals] - expt_vals[expt_ind]['t_pert'][-1]
                xv = xv[:temp_vals]
                yv = np.load(in_path + expt + '_' + var + norm + '.npy')[:,:temp_vals]
                sgr_sa_smoothed = scipy.ndimage.gaussian_filter(np.nanmedian(yv, axis=0), sigma=1)
                err = scipy.ndimage.gaussian_filter(np.nanstd(yv, axis=0), sigma=2)
                plt.plot(xv / 60, sgr_sa_smoothed, label=plot_labels[expt_labels.index(expt)], lw=3.0)
                plt.fill_between(xv / 60, sgr_sa_smoothed - err, sgr_sa_smoothed + err, alpha=0.4)
            # if norm==normed[0]:
            # plt.ylim(ymin=0.0,ymax=1.5)

            ax = plt.gca()
            ymin, ymax = ax.get_ylim()
            for temp_ind in range(len(perts)):
                plt.vlines((expt_vals[-1]['t_perts'][temp_ind] - expt_vals[-1]['t_perts'][pert_comp_ind]) / 60.0, ymin=ymin,
                           ymax=ymax, label=perts[temp_ind], linestyle=linestyles[temp_ind], colors='k',linewidth=0.5)
            # plt.legend(loc=[1.02,0.2])
            plt.legend(loc=legend_loc)
            plt.xlabel('Time (min)')
            temp_lab = ylabels[var]+yunits[var]
            plt.ylabel(temp_lab)
            plt.show()
            fig.savefig(out_path + group_label + '_' + var + norm + '_compact.png', dpi=300, bbox_inches='tight')
            plt.clf()


    # Paper-style figures.
    sns.set(font_scale=1.15)
    sns.set_style("white")

    for norm in normed:
        for var in sel_vars:
            fig = plt.figure(figsize=[2.5, 2])
            for expt in expt_labels:
                xv = np.load(in_path + expt + '_time.npy')
                if not (cutoff is None):
                    temp_vals = np.nonzero(xv/60<cutoff)[0][-1]
                else:
                    temp_vals = len(xv)-1
                # xv=xv[:temp_vals] - expt_vals[expt_ind]['t_pert'][-1]
                xv = xv[:temp_vals]
                yv = np.load(in_path + expt + '_' + var + norm + '.npy')[:,:temp_vals]
                sgr_sa_smoothed = np.nanmedian(yv, axis=0)
                err = np.nanstd(yv, axis=0)
                # plt.plot(xv / 60, sgr_sa_smoothed, label=plot_labels[expt_labels.index(expt)], lw=0.5)
                plt.plot(xv / 60, sgr_sa_smoothed, lw=0.5)
                plt.fill_between(xv / 60, sgr_sa_smoothed - err, sgr_sa_smoothed + err, alpha=0.4)
                print('Generous cell tracks, ', expt, ', ', var, ' :',
                      np.sum(np.sum(np.isnan(yv), axis=1) != yv.shape[1]),
                      file=f)  # Python 3.x. We count a unique trace if it isn't all nan across the timecourse.
            # if norm==normed[0]:
            # plt.ylim(ymin=0.0,ymax=1.5)

            ax = plt.gca()
            ymin, ymax = ax.get_ylim()
            for temp_ind in range(len(perts)):
                plt.vlines((expt_vals[-1]['t_perts'][temp_ind] - expt_vals[-1]['t_perts'][pert_comp_ind]) / 60.0, ymin=ymin,
                           ymax=ymax, label=perts[temp_ind], linestyle=linestyles[temp_ind], colors='k',linewidth=0.5)
            plt.legend(loc=[1.02,0.0])
            # plt.legend()
            plt.xlabel('Time (min)')
            temp_lab = ylabels[var]+yunits[var]
            plt.ylabel(temp_lab)
            plt.show()
            fig.savefig(out_path + group_label + '_' + var + norm + '_compact.pdf', bbox_inches='tight')
            plt.clf()


    # Now we generate a plot of relative elongation in each case and quantify the relative change in length at each mg pulse
    # event
    fig = plt.figure(figsize=[2.5, 2])
    var='sgr_l'
    xv = np.load(in_path + expt + '_time.npy')[:-1]
    print(xv)
    yv1 = np.load(in_path + expt + '_' + var + norm + '.npy')[:,:temp_vals]
    yv=np.nanmedian(yv1,axis=0)*(xv[1]-xv[0])/3600.0
    start_ind=np.nonzero(~np.isnan(yv))[0][0]
    temp_xv=xv[start_ind:]
    lvals=np.exp(np.cumsum(yv[start_ind:]))
    pert_ind=np.nonzero(temp_xv<0)[0][-1]
    print(pert_ind,lvals[pert_ind])
    plt.plot((temp_xv)/60.0,lvals/lvals[pert_ind],label= 'Relative length',color='k',lw=1.0)
    ax=plt.gca()
    ymin, ymax = ax.get_ylim()
    for temp_ind in range(len(perts)):
        plt.vlines((expt_vals[-1]['t_perts'][temp_ind] - expt_vals[-1]['t_perts'][pert_comp_ind]) / 60.0, ymin=ymin,
                   ymax=ymax, label=perts[temp_ind], linestyle=linestyles[temp_ind], colors='k', linewidth=0.5)
    plt.legend(loc=[1.01,0.0])
    plt.xlabel('Time (min)')
    plt.ylabel('Relative length (a.u.)')
    # plt.title('Growth rate response to antibiotic perturbation')
    fig.savefig(out_path + group_label + '_rel_len_compact.pdf', bbox_inches='tight')
    plt.clf()

    # Now we estimate and print the relative change in length for each cell at each shock event, bootstrapping across cells.
    var='sgr_l'
    xv = np.load(in_path + expt + '_time.npy')
    temp_vals = len(xv)-1
    sgr_l = np.load(in_path + expt + '_' + var + norm + '.npy')[:,:temp_vals]
    # print(sgr_l.shape)
    # First, let's calculate the average residual elongation in PBS.
    vals=np.array(t_pert)[post_lysis_ind:]/20.0+7 # Halfway through each PBS incubation
    residuals=[]
    for temp_ind in vals:
        temp_vals1=sgr_l[:,int(temp_ind)-1:int(temp_ind)+1].flatten()
        residuals.append(temp_vals1[np.nonzero(~np.isnan(temp_vals1))])
    residual_gr=np.concatenate(residuals)
    boot1 = np.random.choice(residual_gr, size=10000, replace=True)
    print('Residual growth rate +/- SD (h-1):',  np.around(np.mean(boot1),3),  np.around(np.std(boot1),3), file=f)  # Python 3.x)
    print('Total cell count: ', sgr_l.shape[0], file=f)
    for ind in range(len(t_pert)):
        pert=t_pert[ind]
        pert_ind=int(pert/20.0)
        # print(pert_ind)
        sgr_spec=sgr_l[:,pert_ind-2:pert_ind+6]
        # print(sgr_spec.shape, np.prod(sgr_spec,axis=1).shape)
        # print(sgr_spec)
        # print(np.prod(sgr_spec,axis=1))
        row_sel=np.nonzero(~np.isnan(np.prod(sgr_spec,axis=1)))[0]
        # print(len(row_sel))
        sgr_spec=sgr_spec[row_sel,:]-np.mean(residual_gr) # This is now all intact tracks across that entire perturbation. We now integrate this value.

        boot2=np.random.choice(np.sum(sgr_spec,axis=1).flatten(),size=10000,replace=True)
        lchangepercent=(np.exp((boot2-boot1*sgr_spec.shape[1])*(xv[1]-xv[0])/3600.0)-1.0)*100.0
        # lchangepercent = (np.exp(np.sum(sgr_spec,axis=1)*(xv[1]-xv[0])/3600.0)-1.0)*100.0
        # boot_sampled=np.random.choice(lchangepercent,size=10000,replace=True)
        # print(perts[ind],'% change: ', np.around(np.mean(lchangepercent),2), 'SE: ', np.std(lchangepercent)/np.sqrt(len(lchangepercent)), file=f)  # Python 3.x)
        print(perts[ind],'% change: ', np.around(np.mean(lchangepercent),2), 'SD: ', np.std(lchangepercent), file=f)  # Python 3.x)



