import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import scipy
from scipy import stats
import os
import pickle

# This script should be run with python 3. It assumes a file format where the expt_id is the unique identifier for a given
# experiment, and that this is conserved for all the files within the folder of the same name.

#############
# NOTE: data should be saved in UTF-8 csv format. Labels should be in regular csv format!
#############
out_path='/Users/barber.527/Documents/GitHub/WTA_hydrolysis/'

data_labels = []

max_time=400 # Maximum time for OD saturation plot.

plot_pert=False # Note: if using plot_pert, you will need to also define pert_time (in minutes), pert_label (string),
# and temp_xmax (maximum time for plotting)

# path = '/Users/barber.527/Documents/Rojas_Lab/data' # the base directory path
path = '/Volumes/data_ssd2/Barber_Lab/data' # the base directory path

# expt_id = '/260306_IPTG_lytE_cwlO' # the experiment title
expt_id = '/260313_IPTG_induction_plate_reader' # the experiment title

num_rep=4 # this is the number of repeats of each condition.

# max_time=None

num_groups=[0,30] # num conds + 1 # the number of conditions to be plotted + 1

# Note that this labels file should be filled as follows: blank corresponds to the blank wells (there should be num_rep of these), empty corresponds to empty wells, 
# and each unique condition should be labeled with its own name, that is the same for each of the repeats of that condition

# This part incorporates the cell labels, with the first row being the column numbers and the first entry of each row being
# the row letters.
with open(path+expt_id+expt_id+'_labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data_labels.append(row) # Data should be saved in UTF-8 format
        line_count += 1
print(data_labels)
# exit()
conditions = {}
conditions_inv={}
letters = ['A','B','C','D','E', 'F', 'G', 'H']
temp=[]
for i0 in range(len(data_labels)):
    temp+=data_labels[i0]
# temp1 gives the unique experimental conditions by taking a list of a set of a list (sets only have one copy of each unique item).
temp1=list(set(temp))
linestyles=['--','-.','-',':']
counter={}
for label in temp1:
    counter[label]=1 # initializing the counter for the repeats of each unique condition identifier at 1. This will increase.
# Now we build two dictionaries that map each unique condition to a letter and number combination on the 96 well plate
# conditions is a dict with entries of type 'A1':'Blank', i.e. letter+number=cond+numrep. 
# conditions_inv is a dict with entries of type 'Blank1':'A1', i.e. cond+numrep=letter+number
for row in data_labels[1:]:
    letter=row[0]
    for i0 in np.arange(1,len(row),1):
        if row[i0]!='Empty':
            conditions[letter+str(i0)]=[row[i0],counter[row[i0]]]
            conditions_inv[row[i0]+str(counter[row[i0]])]=letter+str(i0)
        else:
            conditions[letter+str(i0)]=['Empty',counter[row[i0]]]
            conditions_inv['Empty'+str(counter[row[i0]])]=letter+str(i0)
        counter[row[i0]]+=1
conds=list(set([conditions[key][0] for key in conditions.keys()]))
print([[conds[i0],i0] for i0 in range(len(conds))])
conds.sort()

# Now we open the actual data file.
data = []
with open(path+expt_id+expt_id+'.csv') as csv_file:  # Note this part should be in UTF-8 format.
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data.append(row)
        line_count += 1
print(line_count)
# Making a directory for the outputs
if not os.path.exists(out_path+'outputs'+expt_id):
    os.mkdir(out_path+'outputs'+expt_id)

df=pd.DataFrame(columns=['Growth Rate']) # This is a pandas dataframe that we will use to record the growth rate in each condition
tvec = np.asarray([float(data[1][i0]) for i0 in range(1,len(row)-1)]) # This is the vector of measurement times
fig=plt.figure(figsize=[10,10]) # generating a new figure in matplotlib
# Now we just do a raw plot of each and every vector.
for cond in conditions: # iterating through each and every condition in the plate (including repeats)
#     print(conditions[cond])
    if conditions[cond][0]!='Empty': # we only add the data for this condition if the plate is not listed as "empty"
        ind = np.nonzero([data[i0][0]==cond for i0 in range(len(data))])[0][0]
        yv=np.asarray([float(data[ind][i0]) for i0 in range(1,len(row)-1)])
        plt.plot(tvec/60.0,yv,label=conditions[cond][0])
plt.xlabel('Time (min)')
plt.legend()
fig.savefig(out_path+'outputs'+expt_id+expt_id+'_raw.png'.format(counter),bbox_inches='tight',dpi=300)
plt.clf()
del fig


# Now we find the average of the blank datasets. This will be subtracted from all datasets going forward. It's therefore
# important that the blank curves don't grow and hence this is why we add antibiotics to each.
blank_data = np.zeros([num_rep,len(data[0])-1])
for ind in range(1,num_rep+1):
        name=str('Blank')+str(ind)
        lab=conditions_inv[name]
        ind1 = np.nonzero([data[i0][0]==lab for i0 in range(len(data))])[0][0]
        yv=np.asarray([float(data[ind1][i0]) for i0 in range(1,len(row))])
        blank_data[ind-1,:]=yv[:]    
blank_av=np.mean(blank_data,axis=0)        

# Now we make plots where we do things a little nicer. for each condition that is in the range between conds_set[0] and conds_set[1] we
# plot the median of all repeats +/- the standard deviation of those repeats. Plotting the median corrects for big variations
# that may arise due to (e.g.) contamination
tvec = np.asarray([float(data[1][i0]) for i0 in range(1,len(row))])
groupings=[num_groups]
counter=1
for conds_set in groupings:
    fig=plt.figure(figsize=[5,5])
    temp_ind_tracker=0
    for cond in conds[1:conds_set[1]]:
        if not cond in ['Empty']:
            data_tun=np.zeros([num_rep,len(data[0])-1])
            for ind in range(1,num_rep+1):
                name=cond+str(ind)
                lab=conditions_inv[name]
                ind1 = np.nonzero([data[i0][0]==lab for i0 in range(len(data))])[0][0]
                yv=np.asarray([float(data[ind1][i0]) for i0 in range(1,len(row))])
                data_tun[ind-1,:]=yv[:]
            plt.plot(tvec/60.0,np.median(data_tun,axis=0)-blank_av,label=cond,linestyle=linestyles[np.mod(temp_ind_tracker,len(linestyles))])
            plt.fill_between(tvec/60.0,np.median(data_tun,axis=0)-blank_av-np.std(data_tun,axis=0),np.median(data_tun,axis=0)-blank_av+np.std(data_tun,axis=0),alpha=0.2)
            temp_ind_tracker+=1
            if temp_ind_tracker==19:
                temp_ind_tracker+=1 # This ensures that we don't recycle the same colors 
    plt.ylabel('Absorbance')
    plt.xlabel('Time (min)')
    ########
    if plot_pert:
        ax=plt.gca()
        plt.vlines(pert_time,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=pert_label, color='k', linestyle='--')
        plt.xlim(xmin=0,xmax=temp_xmax)
    ########
    plt.legend(loc=[1.02,0.0])
    fig.savefig(out_path+'outputs'+expt_id+expt_id+'_{0}.png'.format(counter),bbox_inches='tight',dpi=300)
    counter+=1
    plt.clf()
    del fig



# The same as above, just with a broader plot area
tvec = np.asarray([float(data[1][i0]) for i0 in range(1,len(row))])
groupings=[num_groups]
counter=1
for conds_set in groupings:
    fig=plt.figure(figsize=[10,5])
    temp_ind_tracker=0
    for cond in conds[conds_set[0]:conds_set[1]]:
        if not cond in ['Empty']:
            data_tun=np.zeros([num_rep,len(data[0])-1])
            for ind in range(1,num_rep+1):
                name=cond+str(ind)
                lab=conditions_inv[name]
                ind1 = np.nonzero([data[i0][0]==lab for i0 in range(len(data))])[0][0]
                yv=np.asarray([float(data[ind1][i0]) for i0 in range(1,len(row))])
                data_tun[ind-1,:]=yv[:]
            yinit=np.median(data_tun,axis=0)[0]-blank_av[0]
            plt.plot(tvec/60.0,(np.median(data_tun,axis=0)-blank_av),label=cond,linestyle=linestyles[np.mod(temp_ind_tracker,len(linestyles))])
            plt.fill_between(tvec/60.0,(np.median(data_tun,axis=0)-blank_av-np.std(data_tun,axis=0)),(np.median(data_tun,axis=0)-blank_av+np.std(data_tun,axis=0)),alpha=0.2)
            temp_ind_tracker+=1
            if temp_ind_tracker==19:
                temp_ind_tracker+=1 # This ensures that we don't recycle the same colors 
    plt.ylabel('Absorbance')
    plt.xlabel('Time (min)')
    ########
    if plot_pert:
        ax=plt.gca()
        plt.vlines(pert_time,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=pert_label, color='k', linestyle='--')
        plt.xlim(xmin=0,xmax=temp_xmax)
    ########
    plt.legend(loc=[1.02,0.0])
    fig.savefig(out_path+'outputs'+expt_id+expt_id+'_broad_{0}.png'.format(counter),bbox_inches='tight',dpi=300)
    counter+=1
    plt.clf()
    del fig
    

# The same as above, just with a broader plot area
tvec = np.asarray([float(data[1][i0]) for i0 in range(1,len(row))])
groupings=[num_groups]
counter=1
for conds_set in groupings:
    fig=plt.figure(figsize=[10,5])
    temp_ind_tracker=0
    for cond in conds[conds_set[0]:conds_set[1]]:
        if not cond in ['Empty']:
            data_tun=np.zeros([num_rep,len(data[0])-1])
            for ind in range(1,num_rep+1):
                name=cond+str(ind)
                lab=conditions_inv[name]
                ind1 = np.nonzero([data[i0][0]==lab for i0 in range(len(data))])[0][0]
                yv=np.asarray([float(data[ind1][i0]) for i0 in range(1,len(row))])
                data_tun[ind-1,:]=yv[:]
            yinit=np.median(data_tun,axis=0)[0]-blank_av[0]
            plt.plot(tvec/60.0,(np.median(data_tun,axis=0)-blank_av),label=cond,linestyle=linestyles[np.mod(temp_ind_tracker,len(linestyles))])
            plt.fill_between(tvec/60.0,(np.median(data_tun,axis=0)-blank_av-np.std(data_tun,axis=0)),(np.median(data_tun,axis=0)-blank_av+np.std(data_tun,axis=0)),alpha=0.2)
            temp_ind_tracker+=1
            if temp_ind_tracker==19:
                temp_ind_tracker+=1 # This ensures that we don't recycle the same colors 
    plt.ylabel('Absorbance')
    plt.xlabel('Time (min)')
    ########
    if plot_pert:
        ax=plt.gca()
        plt.vlines(pert_time,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=pert_label, color='k', linestyle='--')
        plt.xlim(xmin=0,xmax=temp_xmax)
    ########
    plt.legend(loc=[1.02,0.0])
    fig.savefig(out_path+'outputs'+expt_id+expt_id+'_broad_lim_range_{0}.png'.format(counter),bbox_inches='tight',dpi=300)
    counter+=1
    del fig
    plt.clf()

# Focusing on just the log phase behaviour
t_range=[150.0,400.0]
tvec = np.asarray([float(data[1][i0]) for i0 in range(1,len(row))])
plot_inds=np.nonzero(((tvec/60.0)>t_range[0])*((tvec/60.0)<t_range[1]))
print(plot_inds, tvec[plot_inds])
groupings=[num_groups]
counter=1
for conds_set in groupings:
    fig=plt.figure(figsize=[10,5])
    temp_ind_tracker=0
    for cond in conds[conds_set[0]:conds_set[1]]:
        if not cond in ['Empty']:
            data_tun=np.zeros([num_rep,len(data[0])-1])
            for ind in range(1,num_rep+1):
                name=cond+str(ind)
                lab=conditions_inv[name]
                ind1 = np.nonzero([data[i0][0]==lab for i0 in range(len(data))])[0][0]
                yv=np.asarray([float(data[ind1][i0]) for i0 in range(1,len(row))])
                data_tun[ind-1,:]=yv[:]
            yinit=np.median(data_tun,axis=0)[0]-blank_av[0]
            plt.plot(tvec[plot_inds]/60.0,(np.median(data_tun,axis=0)[plot_inds]-blank_av[plot_inds]),label=cond,linestyle=linestyles[np.mod(temp_ind_tracker,len(linestyles))])
            plt.fill_between(tvec[plot_inds]/60.0,(np.median(data_tun,axis=0)[plot_inds]-blank_av[plot_inds]-np.std(data_tun,axis=0)[plot_inds]),(np.median(data_tun,axis=0)[plot_inds]-blank_av[plot_inds]+np.std(data_tun,axis=0)[plot_inds]),alpha=0.2)
            temp_ind_tracker+=1
            if temp_ind_tracker==19:
                temp_ind_tracker+=1 # This ensures that we don't recycle the same colors
    plt.ylabel('Absorbance')
    plt.xlabel('Time (min)')
    ########
    if plot_pert:
        ax=plt.gca()
        plt.vlines(pert_time,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=pert_label, color='k', linestyle='--')
        plt.xlim(xmin=0,xmax=300)
    ########
    plt.legend(loc=[1.02,0.0])
    fig.savefig(out_path+'outputs'+expt_id+expt_id+'_broad_log_phase_{0}.png'.format(counter),bbox_inches='tight',dpi=300)
    counter+=1
    del fig
    plt.clf()

    # Paper figure
    t_range = [0.0, 600.0]
    tvec = np.asarray([float(data[1][i0]) for i0 in range(1, len(row))])
    plot_inds = np.nonzero(((tvec / 60.0) > t_range[0]) * ((tvec / 60.0) < t_range[1]))
    print(plot_inds, tvec[plot_inds])
    groupings = [num_groups]
    counter = 1
    sns.set(font_scale=2)
    sns.set_style("whitegrid")
    for conds_set in groupings:
        fig = plt.figure(figsize=[10, 5])
        temp_ind_tracker = 0
        for cond in conds[conds_set[0]:conds_set[1]]:
            if not cond in ['Empty']:
                data_tun = np.zeros([num_rep, len(data[0]) - 1])
                for ind in range(1, num_rep + 1):
                    name = cond + str(ind)
                    lab = conditions_inv[name]
                    ind1 = np.nonzero([data[i0][0] == lab for i0 in range(len(data))])[0][0]
                    yv = np.asarray([float(data[ind1][i0]) for i0 in range(1, len(row))])
                    data_tun[ind - 1, :] = yv[:]
                yinit = np.median(data_tun, axis=0)[0] - blank_av[0]
                plt.plot(tvec[plot_inds] / 60.0, (np.median(data_tun, axis=0)[plot_inds] - blank_av[plot_inds]),
                         label=cond, linestyle=linestyles[np.mod(temp_ind_tracker, len(linestyles))])
                plt.fill_between(tvec[plot_inds] / 60.0, (
                            np.median(data_tun, axis=0)[plot_inds] - blank_av[plot_inds] - np.std(data_tun, axis=0)[
                        plot_inds]), (np.median(data_tun, axis=0)[plot_inds] - blank_av[plot_inds] +
                                      np.std(data_tun, axis=0)[plot_inds]), alpha=0.2)
                temp_ind_tracker += 1
                if temp_ind_tracker == 19:
                    temp_ind_tracker += 1  # This ensures that we don't recycle the same colors
        plt.ylabel('Absorbance')
        plt.xlabel('Time (min)')
        ########
        if plot_pert:
            ax = plt.gca()
            plt.vlines(pert_time, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], label=pert_label, color='k',
                       linestyle='--')
            plt.xlim(xmin=0, xmax=300)
        ########
        plt.legend(loc=[1.02, 0.0])
        fig.savefig(out_path+'outputs' + expt_id + expt_id + '_broad_paper_{0}.png'.format(counter), bbox_inches='tight',
                    dpi=300)
        fig.savefig(out_path+'outputs' + expt_id + expt_id + '_broad_paper_{0}.pdf'.format(counter), bbox_inches='tight')
        counter += 1
        del fig
        plt.clf()

# The same as above, just with a broader plot area on a logy scale
tvec = np.asarray([float(data[1][i0]) for i0 in range(1,len(row))])
groupings=[num_groups]
counter=1
for conds_set in groupings:
    fig=plt.figure(figsize=[10,5])
    temp_ind_tracker=0
    for cond in conds[conds_set[0]:conds_set[1]]:
        if not cond in ['Empty']:
            data_tun=np.zeros([num_rep,len(data[0])-1])
            for ind in range(1,num_rep+1):
                name=cond+str(ind)
                lab=conditions_inv[name]
        #         print(lab,name)
                ind1 = np.nonzero([data[i0][0]==lab for i0 in range(len(data))])[0][0]
                yv=np.asarray([float(data[ind1][i0]) for i0 in range(1,len(row))])
                if len(yv)==12:
                    print(cond,yv)
        #         print(yv.shape,data_tun.shape)
                data_tun[ind-1,:]=yv[:]
        #         print(lab,yv)
            yinit=np.median(data_tun,axis=0)[0]-blank_av[0]
            plt.semilogy(tvec/60.0,(np.median(data_tun,axis=0)-blank_av),label=cond,linestyle=linestyles[np.mod(temp_ind_tracker,len(linestyles))])
            plt.fill_between(tvec/60.0,(np.median(data_tun,axis=0)-blank_av-np.std(data_tun,axis=0)),(np.median(data_tun,axis=0)-blank_av+np.std(data_tun,axis=0)),alpha=0.2)
            temp_ind_tracker+=1
            if temp_ind_tracker==19:
                temp_ind_tracker+=1 # This ensures that we don't recycle the same colors 
    plt.ylabel('Absorbance')
    plt.xlabel('Time (min)')
    ########
    if plot_pert:
        ax=plt.gca()
        plt.vlines(pert_time,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=pert_label, color='k', linestyle='--')
        plt.xlim(xmin=0,xmax=temp_xmax)
    ########
    plt.legend(loc=[1.02,0.0])
    # plt.show()
    fig.savefig(out_path+'outputs'+expt_id+expt_id+'_broad_logy_{0}.png'.format(counter),bbox_inches='tight',dpi=300)
    counter+=1
    plt.clf()
    del fig

window=2
tvec = np.asarray([float(data[1][i0]) for i0 in range(1,len(row))])
tvec_gr=tvec[window:len(tvec)-window]


# Now we plot the actual growth rates. 
growth_conds=list(set(conds))
df=pd.DataFrame(columns=['Growth Rate', 'Condition']) # This pandas dataframe will give the maximal growth rate of each condition
for conds_set in groupings:
    fig=plt.figure(figsize=[10,5])
    temp_ind_tracker=0
    for cond in conds[1:]:
        if not cond in ['Empty']:
            data_gr=np.zeros([num_rep,len(tvec_gr)])
            for ind in range(1,num_rep+1):
                name=cond+str(ind)
                lab=conditions_inv[name]
                ind1 = np.nonzero([data[i0][0]==lab for i0 in range(len(data))])[0][0]
                yv=np.asarray([float(data[ind1][i0]) for i0 in range(1,len(row))])
                yv-=blank_av
                # here is where we calculate the growth rate at each timepoint by looking at some time window to either side
                # and doing a linear regression
                for i0 in range(window,len(tvec)-window):
                    temp=scipy.stats.linregress(tvec[i0-window:i0+window+1]/60.0,np.log(yv[i0-window:i0+window+1]))[0]
                    data_gr[ind-1,i0-window]=temp
                temp_df=pd.DataFrame([[np.nanmax(data_gr[ind-1,:]),name[:-1]]],columns=['Growth Rate', 'Condition'])
                df=pd.concat((df,temp_df))
            plt.plot(tvec_gr/60.0,np.median(data_gr,axis=0),label=cond,linestyle=linestyles[np.mod(temp_ind_tracker,len(linestyles))])
            plt.fill_between(tvec_gr/60.0,(np.median(data_gr,axis=0)-np.std(data_gr,axis=0)),(np.median(data_gr,axis=0)+np.std(data_gr,axis=0)),alpha=0.2)
            temp_ind_tracker+=1
            if temp_ind_tracker==19:
                temp_ind_tracker+=1 # This ensures that we don't recycle the same colors 
    plt.ylabel('Growth rate (min-1)')
    plt.xlabel('Time (min)')
    ########
    if plot_pert:
        ax=plt.gca()
        plt.vlines(pert_time,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=pert_label, color='k', linestyle='--')
        plt.xlim(xmin=0,xmax=temp_xmax)
    ########
    plt.legend(loc=[1.02,0.0])
    # plt.show()
    fig.savefig(out_path+'outputs'+expt_id+expt_id+'_gr_{0}.png'.format(counter),bbox_inches='tight',dpi=300)
    counter+=1
    del fig

# Plotting bar graphs of the different maximal growth rates

temp_df=df.groupby('Condition').mean() # we group the maximal growth rates by their condition and take the mean with respect to each condition
temp_df1=df.groupby('Condition').std(ddof=1) # we group the maximal growth rates by their condition and take the standard deviation with respect to each condition
sns.set(font_scale=2)
sns.set_style("whitegrid")
current_palette = sns.color_palette()
temp_df.plot(kind='bar',yerr=temp_df1,capsize=10.0,grid=True,alpha=1.0,figsize=[10,8])
sns.stripplot(x='Condition', y='Growth Rate', data=df.reset_index(), size=7, jitter=True, linewidth=2.0,color='b',alpha=0.5)
plt.ylabel('Growth rate (min$^{-1}$)')
plt.gca().get_legend().remove()
fig=plt.gcf()
fig.savefig(out_path+'outputs'+expt_id+expt_id+'_growth_rate_bar_plot.png',bbox_inches='tight',dpi=300)
# plt.show()
plt.clf()

###########################
# Up until now we have simply been plotting all the data.
# However, you can also plot the same bar graphs without the data points for individual experiments in an order of your choosing
# Do this by creating a _curated.csv file where each line is a condition you want plotted (use the same names as in the main label file)
# Then simply run the whole piece of code and it will also generate the plots with the ordering you want.
# Remember to save as a comma separated file, NOT UTF8 as this adds a preamble that throws off the code.

# Now we plot the actual growth rates. 
growth_conds=list(set(conds))
df=pd.DataFrame(columns=['Growth Rate', 'Condition']) # This pandas dataframe will give the maximal growth rate of each condition
for conds_set in groupings:
    fig=plt.figure(figsize=[10,5])
    temp_ind_tracker=0
    for cond in conds[:]:
        if not cond in ['Empty']:
            data_gr=np.zeros([num_rep,len(tvec_gr)])
            for ind in range(1,num_rep+1):
                name=cond+str(ind)
                lab=conditions_inv[name]
                ind1 = np.nonzero([data[i0][0]==lab for i0 in range(len(data))])[0][0]
                yv=np.asarray([float(data[ind1][i0]) for i0 in range(1,len(row))])
                yv-=blank_av
                # here is where we calculate the growth rate at each timepoint by looking at some time window to either side
                # and doing a linear regression 
                for i0 in range(window,len(tvec)-window):
                    temp=scipy.stats.linregress(tvec[i0-window:i0+window+1]/60.0,np.log(yv[i0-window:i0+window+1]))[0]
                    data_gr[ind-1,i0-window]=temp
                temp_df=pd.DataFrame([[np.nanmax(data_gr[ind-1,:]),name[:-1]]],columns=['Growth Rate', 'Condition'])
                df=pd.concat((df,temp_df))
            temp_ind_tracker+=1
            if temp_ind_tracker==19:
                temp_ind_tracker+=1 # This ensures that we don't recycle the same colors 
    counter+=1
    del fig

to_plot=[]
with open(path+expt_id+expt_id+'_curated.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for temp_row in csv_reader:
        to_plot.append(temp_row[0])
        line_count += 1
        
print(to_plot)
temp_dfs=df.groupby('Condition').mean().loc[to_plot] # we group the maximal growth rates by their condition and take the mean with respect to each condition
temp_dfs1=df.groupby('Condition').std(ddof=1).loc[to_plot] # we group the maximal growth rates by their condition and take the standard deviation with respect to each condition
sns.set(font_scale=2)
sns.set_style("whitegrid")
current_palette = sns.color_palette()
temp_dfs.plot(kind='bar',yerr=temp_dfs1,capsize=10.0,grid=True,alpha=1.0,figsize=[10,8])
plt.ylabel('Growth rate (min$^{-1}$)')
plt.gca().get_legend().remove()
fig=plt.gcf()
fig.savefig(out_path+'outputs'+expt_id+expt_id+'_growth_rate_bar_plot_curated.png',bbox_inches='tight',dpi=300)
# plt.show()
plt.clf()
temp_path = out_path+'outputs'+expt_id+expt_id+'_growth_rate_df.pkl'
with open(temp_path, 'wb') as output:  # Overwrites any existing file.
    pickle.dump(df, output, pickle.HIGHEST_PROTOCOL)


# Calculating the maximal optical density for each condition

tvec = np.asarray([float(data[1][i0]) for i0 in range(1,len(row))])

growth_conds=list(set(conds))
df=pd.DataFrame(columns=['Max. density', 'Condition'])
for conds_set in groupings:
    for cond in conds[1:]:
        if not cond in ['Empty']:
            data_gr=np.zeros([num_rep,len(tvec_gr)])
            for ind in range(1,num_rep+1):
                name=cond+str(ind)
                lab=conditions_inv[name]
#                 print(lab,name)
                ind1 = np.nonzero([data[i0][0]==lab for i0 in range(len(data))])[0][0]
                yv=np.asarray([float(data[ind1][i0]) for i0 in range(1,len(row))])
                yv-=blank_av
                temp_df=pd.DataFrame([[np.nanmax(yv),name[:-1]]],columns=['Max. density', 'Condition'])
                # df=df.append(temp_df)
                df = pd.concat((df,temp_df))
    counter+=1
    del fig

# Plotting the saturation optical density. This works similarly to that for the maximal growth rate categorical bar graph plots above.

temp_df=df.groupby('Condition').mean()
temp_df1=df.groupby('Condition').std(ddof=1)
sns.set(font_scale=2)
sns.set_style("whitegrid")
current_palette = sns.color_palette()
temp_df.plot(kind='bar',yerr=temp_df1,capsize=10.0,grid=True,alpha=1.0,figsize=[10,8])
sns.stripplot(x='Condition', y='Max. density', data=df.reset_index(), size=7, jitter=True, linewidth=2.0,color='b',alpha=0.5)
plt.ylabel('Max. density')
plt.gca().get_legend().remove()
fig=plt.gcf()
fig.savefig(out_path+'outputs'+expt_id+expt_id+'_max_density_bar_plot.png',bbox_inches='tight',dpi=300)
# plt.show()
plt.clf()

print("I made it here")

# Plotting the saturation optical density. This works similarly to that for the maximal growth rate categorical bar graph plots above.

growth_conds=list(set(conds))
df=pd.DataFrame(columns=['Max. density', 'Condition'])
for conds_set in groupings:
    for cond in conds[:]:
        if not cond in ['Empty']:
            data_gr=np.zeros([num_rep,len(tvec_gr)])
            for ind in range(1,num_rep+1):
                name=cond+str(ind)
                lab=conditions_inv[name]
#                 print(lab,name)
                ind1 = np.nonzero([data[i0][0]==lab for i0 in range(len(data))])[0][0]
                yv=np.asarray([float(data[ind1][i0]) for i0 in range(1,len(row))])
                yv-=blank_av
                if not(max_time is None):
                    print(max_time, np.nonzero(tvec/60>max_time), tvec[np.nonzero(tvec/60>max_time)])
                    yv=yv[:np.nonzero(tvec/60>max_time)[0][0]]
                temp_df=pd.DataFrame([[np.nanmax(yv),name[:-1]]],columns=['Max. density', 'Condition'])
                # df=df.append(temp_df)
                df = pd.concat((df,temp_df))
    counter+=1
    del fig

temp_dfs=df.groupby('Condition').mean().loc[to_plot]
temp_dfs1=df.groupby('Condition').std(ddof=1).loc[to_plot]
sns.set(font_scale=2)
sns.set_style("whitegrid")
current_palette = sns.color_palette()
temp_dfs.plot(kind='bar',yerr=temp_dfs1,capsize=10.0,grid=True,alpha=1.0,figsize=[10,8])
plt.ylabel('Max. density')
plt.gca().get_legend().remove()
fig=plt.gcf()
fig.savefig(out_path+'outputs'+expt_id+expt_id+'_max_density_bar_plot_curated.png',bbox_inches='tight',dpi=300)
# plt.show()
plt.clf()

temp_path = out_path+'outputs'+expt_id+expt_id+'_sat_OD_df.pkl'
with open(temp_path, 'wb') as output:  # Overwrites any existing file.
    pickle.dump(df, output, pickle.HIGHEST_PROTOCOL)
