import scipy
import skimage
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
from skimage.morphology import disk
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
import matplotlib.patches as patches
from matplotlib import cm, colors


# Designed to be run in Python 3

def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')

# def image_brightness_props(temp_im):
#     # Takes as input a 2D, bw image and returns an approximation of the mode of that distribution and the fwhm of the
#     # noise distribution.
#
#     yv = data[i0, :, :].flatten()
#     out = np.histogram(yv, bins=100)
#     ind1 = np.argmax(out[0])
#     mode = (out[1][ind1] + out[1][ind1 + 1]) / 2.0
#
#     return mode

def import_data(expt_vals,label=False):
    # imports data from .m files and collates it in a nice fashion, combining fluorescence data from labeling images
    # that are stored in a systematic fashion
    col=['Integrated F{0}'.format(c) for c in expt_vals['channels']]+['Average F{0}'.format(c) for c in expt_vals['channels']]
    temp_df_out = pd.DataFrame(columns=col)
    if label:
        col+=['Labeled']
    for scene in range(1,expt_vals['num_scenes']+1):
        temp_path=expt_vals['base_path']+expt_vals['expt_id']+expt_vals['expt_id']+'_s{0}_1_a'.format(str(scene).zfill(3))+expt_vals['expt_id']+'_s{0}_BT_felix.mat'.format(str(scene).zfill(3))
        mat = scipy.io.loadmat(temp_path)
        num_cells,num_timepoints=mat['lcell'].shape
        t=num_timepoints-1 # timepoint to compare bf image segmentation towards
        # Now we load all the fluorescence images for this scene
        ims=[]
        for channel in expt_vals['channels']:
            fluo_im_path=expt_vals['base_path']+expt_vals['expt_id']+expt_vals['expt_id']+'_labels_a'+expt_vals['expt_id']+'_labels_s{0}_C{1}.tif'.format(str(scene).zfill(3),channel)
            ims.append(skimage.io.imread(fluo_im_path).transpose()) # transpose since rows and columns are swapped in matlab coords
            # background
            ims[expt_vals['channels'].index(channel)]=ims[expt_vals['channels'].index(channel)]-expt_vals['median_vals'][expt_vals['channels'].index(channel)]
            # subtract the median background fluorescence.
        for temp_cell in range(num_cells):
            if not(np.isnan(mat['lcell'][temp_cell,t])): # Only cells that have been selected based on the bf image filtering
                temp_df={}
                temp_pix=mat['pixels'][temp_cell,t].astype('int')
                temp_pix-=1 # account for the fact that in matlab, these linear indices will start from 1 rather than 0
                for channel in expt_vals['channels']:
                    temp_df['Integrated F{0}'.format(channel)]=np.sum(ims[expt_vals['channels'].index(channel)][np.unravel_index(temp_pix,ims[0].shape)])
                    temp_df['Average F{0}'.format(channel)]=np.mean(ims[expt_vals['channels'].index(channel)][np.unravel_index(temp_pix,ims[0].shape)])
                    if label and channel==expt_vals['class_channel']:
                        temp_df['Labeled'] = np.mean(ims[expt_vals['channels'].index(channel)][np.unravel_index(temp_pix,ims[0].shape)])>expt_vals['cutoff']
                temp_df_out=temp_df_out.append(temp_df,ignore_index=True)
    return temp_df_out


def bkgd_fluorescence_calculation(expt_vals):
    bkgd_med=[]
    for scene in range(1, expt_vals['num_scenes'] + 1):
        temp_path = expt_vals['base_path'] + expt_vals['expt_id'] + expt_vals['expt_id'] + '_s{0}_1_a'.format(
            str(scene).zfill(3)) + expt_vals['expt_id'] + '_s{0}_BT_felix.mat'.format(str(scene).zfill(3))
        mat = scipy.io.loadmat(temp_path)
        num_cells, num_timepoints = mat['lcell'].shape
        t = num_timepoints - 1  # timepoint to compare bf image segmentation towards
        mask = np.ones(expt_vals['im_shape'])
        temp_bkgd_med = []
        for temp_cell in range(num_cells):
            if not (len(mat['pixels'][temp_cell, t]) == 0):  # all segments
                temp_pix = mat['pixels'][temp_cell, t].astype('int')
                mask[np.unravel_index(temp_pix, expt_vals['im_shape'])] = np.nan
        # Now we load all the fluorescence images for this scene
        for channel in expt_vals['channels']:
            fluo_im_path = expt_vals['base_path'] + expt_vals['expt_id'] + expt_vals['expt_id'] + '_labels_a' + \
                           expt_vals['expt_id'] + '_labels_s{0}_C{1}.tif'.format(
                str(scene).zfill(3), channel)
            im = skimage.io.imread(
                fluo_im_path).transpose()  # transpose since rows and columns are swapped in matlab coords
            # background
            temp_bkgd_med.append((im * mask).flatten())
        bkgd_med.append(temp_bkgd_med)
    median_vals = []
    for i0 in range(len(expt_vals['channels'])):
        median_vals.append(np.nanmedian(np.concatenate([obj[i0] for obj in bkgd_med], axis=0)))
    return median_vals


def timepoint_bkgd_fluorescence_calculation(expt_vals):
    bkgd_med = []
    dir_path = expt_vals['base_path'] + expt_vals['expt_id']+'/'+expt_vals['cond']
    temp_path = dir_path +'/C1_blur'+ expt_vals['expt_id']+'_'+expt_vals['cond'] + '_BT_felix.mat'
    mat = scipy.io.loadmat(temp_path)
    median_vals = []
    num_cells, num_timepoints = mat['lcell'].shape
    for scene in range(1, expt_vals['num_scenes'] + 1):
        mask = np.ones(expt_vals['im_shape'])
        temp_bkgd_med = []
        for temp_cell in range(num_cells):
            if not (len(mat['pxls'][temp_cell, scene-1]) == 0):  # all segments
                temp_pix = mat['pxls'][temp_cell, scene-1].astype('int')
                mask[np.unravel_index(temp_pix, expt_vals['im_shape'])] = np.nan
        # Now we load all the fluorescence images for this scene
        for channel in expt_vals['channels']:
            fluo_im_path = dir_path + '/C{0}'.format(channel)+\
                           expt_vals['expt_id']+'_'+expt_vals['cond']+'_s{0}_C{1}.tif'.format(str(scene).zfill(3),channel)
            im = skimage.io.imread(
                fluo_im_path).transpose()  # transpose since rows and columns are swapped in matlab coords
            # background
            temp_bkgd_med.append(np.nanmedian((im * mask).flatten()))
        median_vals.append(temp_bkgd_med)
    return median_vals


def timepoint_import_data(expt_vals,label=False):
    # imports data from .m files and collates it in a nice fashion, combining fluorescence data from labeling images
    # that are stored in a systematic fashion
    col=['Integrated F{0}'.format(c) for c in expt_vals['channels']]+['Average F{0}'.format(c) for c in expt_vals['channels']]
    temp_df_out = pd.DataFrame(columns=col)
    if label:
        col+=['Labeled']
    dir_path = expt_vals['base_path'] + expt_vals['expt_id'] + '/' + expt_vals['cond']
    temp_path = dir_path + '/C1_blur' + expt_vals['expt_id'] + '_' + expt_vals['cond'] + '_BT_felix.mat'
    mat = scipy.io.loadmat(temp_path)
    num_cells, num_timepoints = mat['lcell'].shape
    # print('hi')
    for scene in range(1,expt_vals['num_scenes']+1):
        # Now we load all the fluorescence images for this scene
        ims=[]
        for channel in expt_vals['channels']:
            fluo_im_path = dir_path + '/C{0}'.format(channel) + \
                           expt_vals['expt_id'] + '_' + expt_vals['cond'] + '_s{0}_C{1}.tif'.format(str(scene).zfill(3),
                                                                                                    channel)
            ims.append(skimage.io.imread(
                fluo_im_path).transpose())  # transpose since rows and columns are swapped in matlab coords
            # background
            ims[expt_vals['channels'].index(channel)]=ims[expt_vals['channels'].index(channel)]-expt_vals['median_vals'][scene-1][expt_vals['channels'].index(channel)]
            # subtract the median background fluorescence.
        for temp_cell in range(num_cells):
            if not(np.isnan(mat['lcell'][temp_cell,scene-1])): # Only cells that have been selected based on the bf image filtering
                temp_df={}
                temp_pix=mat['pxls'][temp_cell,scene-1].astype('int')
                temp_pix-=1 # account for the fact that in matlab, these linear indices will start from 1 rather than 0
                for channel in expt_vals['channels']:
                    temp_df['Integrated F{0}'.format(channel)]=np.sum(ims[expt_vals['channels'].index(channel)][np.unravel_index(temp_pix,ims[0].shape)])
                    temp_df['Average F{0}'.format(channel)]=np.mean(ims[expt_vals['channels'].index(channel)][np.unravel_index(temp_pix,ims[0].shape)])
                    if label and channel==expt_vals['class_channel']:
                        temp_df['Labeled'] = np.mean(ims[expt_vals['channels'].index(channel)][np.unravel_index(temp_pix,ims[0].shape)])>expt_vals['cutoff']
                    # print(temp_df)
                temp_df_out=temp_df_out.append(temp_df,ignore_index=True)
    return temp_df_out


def timepoint_import_data_outline(expt_vals,label=False):
    # imports data from .m files and collates it in a nice fashion, combining fluorescence data from labeling images
    # that are stored in a systematic fashion
    # differs from previous version by also averaging fluorescence over cell outline
    col=['Integrated F{0}'.format(c) for c in expt_vals['channels']]+['Average F{0}'.format(c) for c in expt_vals['channels']]
    col+=['Average outline F{0}'.format(c) for c in expt_vals['channels']]
    col+=['Integrated outline F{0}'.format(c) for c in expt_vals['channels']]
    temp_df_out = pd.DataFrame(columns=col)
    if label:
        col+=['Labeled']
    dir_path = expt_vals['base_path'] + expt_vals['expt_id'] + '/' + expt_vals['cond']
    temp_path = dir_path + '/C1_blur' + expt_vals['expt_id'] + '_' + expt_vals['cond'] + '_BT_felix.mat'
    mat = scipy.io.loadmat(temp_path)
    num_cells, num_timepoints = mat['lcell'].shape
    # print('hi')
    
    # generating the connectivity mask that we will use to generate the cell outlines below
    conn_mask=np.zeros([5,5])
    conn_mask[2,2]=1
    conn_mask=dilation(dilation(conn_mask))

    
    for scene in range(1,expt_vals['num_scenes']+1):
        # Now we load all the fluorescence images for this scene
        ims=[]
        for channel in expt_vals['channels']:
            fluo_im_path = dir_path + '/C{0}'.format(channel) + \
                           expt_vals['expt_id'] + '_' + expt_vals['cond'] + '_s{0}_C{1}.tif'.format(str(scene).zfill(3),
                                                                                                    channel)
            ims.append(skimage.io.imread(
                fluo_im_path).transpose())  # transpose since rows and columns are swapped in matlab coords
            # background
            ims[expt_vals['channels'].index(channel)]=ims[expt_vals['channels'].index(channel)]-expt_vals['median_vals'][scene-1][expt_vals['channels'].index(channel)]
            # subtract the median background fluorescence.
        for temp_cell in range(num_cells):
            if not(np.isnan(mat['lcell'][temp_cell,scene-1])): # Only cells that have been selected based on the bf image filtering
                temp_df={}
                temp_pix=mat['pxls'][temp_cell,scene-1].astype('int')
                temp_pix-=1 # account for the fact that in matlab, these linear indices will start from 1 rather than 0
                
                # generating the cell outlines
                coords=np.unravel_index(temp_pix,expt_vals['im_shape'])
                temp_mat=np.zeros(expt_vals['im_shape'])
                temp_mat[coords]=1
                outline_coords=np.nonzero(dilation(temp_mat,selem=conn_mask)*(erosion(temp_mat,selem=conn_mask)==0))
                del temp_mat
                for channel in expt_vals['channels']:
                    temp_df['Integrated F{0}'.format(channel)]=np.sum(ims[expt_vals['channels'].index(channel)][coords])
                    temp_df['Average F{0}'.format(channel)]=np.mean(ims[expt_vals['channels'].index(channel)][coords])
                    temp_df['Integrated outline F{0}'.format(channel)]=np.sum(ims[expt_vals['channels'].index(channel)][outline_coords])
                    temp_df['Average outline F{0}'.format(channel)]=np.mean(ims[expt_vals['channels'].index(channel)][outline_coords])
                    if label and channel==expt_vals['class_channel']:
                        temp_df['Labeled'] = np.mean(ims[expt_vals['channels'].index(channel)][np.unravel_index(temp_pix,ims[0].shape)])>expt_vals['cutoff']
                    # print(temp_df)
                temp_df_out=temp_df_out.append(temp_df,ignore_index=True)
    return temp_df_out


def timepoint_import_data_outline_smart_bkgd(expt_vals,label=False,thresh_int=0):
    # imports data from .m files and collates it in a nice fashion, combining fluorescence data from labeling images
    # that are stored in a systematic fashion
    # differs from previous version by also averaging fluorescence over cell outline
    # Note that this script makes no attempt to implement tracking and that this is also not performed by the corresponding matlab script: timepoint_segmentation.m
    # thresh is the threshold for blob_log (set to 900 for a standard HADA exposure time of 100ms).

    col=['Integrated F{0}'.format(c) for c in expt_vals['channels']]+['Average F{0}'.format(c) for c in expt_vals['channels']]
    col+=['Average outline F{0}'.format(c) for c in expt_vals['channels']]
    col+=['Integrated outline F{0}'.format(c) for c in expt_vals['channels']]
    col+=['Outline vals']
    col+=['Scene']
    col+=['Cell spots']
    col+=['Wall spots']
    col += ['Length']
    col += ['Cell spots/length']
    col += ['Wall spots/length']
    temp_df_out = pd.DataFrame(columns=col)
    if label:
        col+=['Labeled']
    dir_path = expt_vals['base_path'] + expt_vals['expt_id'] + '/' + expt_vals['cond']
    temp_path = dir_path + '/C1_blur' + expt_vals['expt_id'] + '_' + expt_vals['cond'] + '_BT_felix.mat'
    if not os.path.exists('./outputs'+expt_vals['expt_id'] + '/cell_segments'):
        os.makedirs('./outputs'+expt_vals['expt_id'] +  '/cell_segments')
    mat = scipy.io.loadmat(temp_path)
    num_timepoints = mat['lcell'].shape[1]
    # Only cells that have been selected based on the bf image filtering

    # parameters for spot detection
    gam = 1.0
    min_sig = 2.0
    max_sig = 4
    ovrlap = 0.0
    vmin, vmax = 0.0, 10000
    # thresh=expt_vals['thresh_peak'][expt_vals['cond']]
    thresh = expt_vals['thresh_peak']
    rescale = expt_vals['rescale']
    min_thresh=expt_vals['min_thresh']
    # generating the connectivity mask that we will use to generate the cell outlines below
    conn_mask=np.zeros([5,5])
    conn_mask[2,2]=1
    conn_mask=dilation(dilation(conn_mask))
    # print(expt_vals.keys())
    sig=100
    saved_ims=[]

    for scene in range(1,expt_vals['num_scenes']+1):
        # print(expt_vals['excl_scenes'])
        if not(scene in expt_vals['excl_scenes']): # if we don't just need to exclude this scene from consideration
            print(scene)
            num_cells=np.sum(~np.isnan(mat['lcell'][:,scene-1]))
            # Now we load all the fluorescence images for this scene
            ims=[]
            for channel in expt_vals['channels']:
                fluo_im_path = dir_path + '/C{0}'.format(channel) + \
                               expt_vals['expt_id'] + '_' + expt_vals['cond'] + '_s{0}_C{1}.tif'.format(str(scene).zfill(3),
                                                                                                        channel)
                fluo_im_out_path = dir_path + '/C{0}'.format(channel) + \
                           expt_vals['expt_id'] + '_' + expt_vals['cond'] + '_s{0}_C{1}_adj'.format(str(scene).zfill(3),
                                                                                                            channel)

                if os.path.exists(fluo_im_out_path):
                    im_adj = np.load(fluo_im_out_path)
                else:
                    # background correction using the mask of cells
                    
                    im=skimage.io.imread(fluo_im_path).transpose()  # transpose since rows and columns are swapped in matlab coords
                    
                    mask = np.ones(expt_vals['im_shape'])
                    for temp_cell in np.nonzero(~np.isnan(mat['lcell'][:,scene-1]))[0]:
                        if not (len(mat['pxls'][temp_cell, scene-1]) == 0):  # all segments
                            temp_pix = mat['pxls'][temp_cell, scene-1].astype('int')
                            temp_pix-=1 # account for the fact that in matlab, these linear indices will start from 1 rather than 0
                            mask[np.unravel_index(temp_pix, expt_vals['im_shape'])] = np.nan
                    # New version to focus on non-masked pixels
                    cutoff = np.median(im[~np.isnan(mask)].flatten()) + 2 * scipy.stats.iqr(
                        im[~np.isnan(mask)])  # high background
                    # cutoff = np.mean(im[~np.isnan(mask)].flatten()) + 2*np.std(im[~np.isnan(mask)]) # high background
                    # values
                    # cutoff=np.mean(im[np.isnan(mask)].flatten())-np.std(im[np.isnan(mask)])
                    
                    # Now we create a mask for pixels with high intensity:
                    im1=im.copy()
                    im1[np.nonzero(im>cutoff)]=0 # this mask sets to zero any pixels with intensity greater than the
                    #cutoff (i.e. similar to the segmented cells)
                    umax=np.amax(im1.flatten())
                    im1=im1/umax # normalizing prior to smoothing
                    temp_out1=skimage.filters.gaussian(im1, sigma=sig) # gaussian filter to smooth over these areas
                    
                    im2=im<cutoff # binary image to correct for zero values in mask
                    temp_out2=skimage.filters.gaussian(im2, sigma=sig) # normalizing gaussian filter 
                    
                    bkgd=umax*temp_out1/temp_out2 # normalized background values
                    
                    im_adj=im-bkgd
                    im_adj[np.nonzero(im_adj<0)]=0
                    np.save(fluo_im_out_path+'.npy', im_adj)
                    skimage.io.imsave(fluo_im_out_path + '.tif', im_adj.transpose())
                ims.append(im_adj)
                saved_ims.append(im_adj)
            print(im_adj.shape)
            temp_im = np.copy(ims[-1])
            # norm_vals = [np.percentile(temp_im[np.isnan(mask)], q=98), min_thresh]
            for temp_cell in np.nonzero(~np.isnan(mat['lcell'][:,scene-1]))[0]:
                if (len(mat['pxls'][temp_cell, scene-1]) != 0) & ~np.isnan(mat['lcell'][temp_cell,scene-1]):  # all segments
                    temp_pix = mat['pxls'][temp_cell, scene-1].astype('int')
                    temp_pix-=1 # account for the fact that in matlab, these linear indices will start from 1 rather than 0
                    temp_df={}
                    coords=np.unravel_index(temp_pix,expt_vals['im_shape'])
                    # generating the cell outlines
                    # temp_mat=np.zeros(expt_vals['im_shape'])
                    # temp_mat[coords]=1
                    # outline_coords=np.nonzero(dilation(temp_mat,selem=conn_mask)*(erosion(temp_mat,selem=conn_mask)==0))
                    temp_pix_o = mat['boun'][temp_cell, scene-1].astype('int')
                    temp_pix_o-=1 # account for the fact that in matlab, these linear indices will start from 1 rather than 0

                    if np.median(temp_im[coords]) > thresh_int:
                        # we only include this in the cutoff calculation if it is fluorescent
                        temp_im[coords] = 2500  # for visualization purposes
                        temp_im[temp_pix_o[:, 0], temp_pix_o[:, 1]] = 5000  # for visualization purposes
                    # del temp_mat
                    for channel in expt_vals['channels']:
                        temp_vec_cell = ims[expt_vals['channels'].index(channel)][coords]
                        temp_df['Integrated F{0}'.format(channel)]=np.sum(temp_vec_cell)
                        temp_df['Average F{0}'.format(channel)]=np.mean(temp_vec_cell)
                        # outline calculations
                        temp_vec_outline=ims[expt_vals['channels'].index(channel)][temp_pix_o[:,0],temp_pix_o[:,1]]
                        temp_df['Integrated outline F{0}'.format(channel)]=np.sum(temp_vec_outline)
                        temp_df['Average outline F{0}'.format(channel)]=np.mean(temp_vec_outline)
                        temp_df['CV outline F{0}'.format(channel)]=np.std(temp_vec_outline)/np.mean(temp_vec_outline)
                        temp_df['SD outline F{0}'.format(channel)]=np.std(temp_vec_outline)
                        temp_df['Skew outline F{0}'.format(channel)]=scipy.stats.skew(temp_vec_outline)
                        # This was added later
                        temp_df['Outline vals']=temp_vec_outline
                        if label and channel == expt_vals['class_channel']:
                            temp_df['Labeled'] = np.mean(ims[expt_vals['channels'].index(channel)][np.unravel_index(temp_pix,ims[0].shape)])>expt_vals['cutoff']
                        temp_df['Scene'] = scene

                    # Now we calculate the number of bright spots within this cell for the HADA channel!
                    temp_im1 = ims[expt_vals['channels'].index(expt_vals['HADA_channel'])].copy()
                    ymin, ymax, xmin, xmax = np.amin(coords[0]), np.amax(coords[0]), np.amin(
                        coords[1]), np.amax(coords[1]) # These are the boundaries of the cell in question
                    # temp_im1[temp_pix_o[:, 0], temp_pix_o[:, 1]] = np.amax(
                    #     temp_im1[ymin:ymax, xmin:xmax])  # for visualization purposes
                    temp_im2 = temp_im1[ymin:ymax, xmin:xmax]
                    if rescale:
                        temp_im2 *= 65535.0/np.amax(temp_im1)
                        # Option 1:  we rescale to the max of the image as a whole, to avoid
                        # spurious additional spots in extra bright cells.
                        # input_data = skimage.exposure.adjust_gamma(
                        #     ims[expt_vals['channels'].index(expt_vals['HADA_channel'])][
                        #         ymin:ymax, xmin:xmax] * 65535.0 / np.amax(norm_vals), gam)
                        # Option 2: we rescale to the median of the cell in question, provided it's above a certain
                        # brightness level
                        input_data = skimage.exposure.adjust_gamma(
                            ims[expt_vals['channels'].index(expt_vals['HADA_channel'])][
                                ymin:ymax, xmin:xmax] / np.amax([np.amin(temp_im1[coords]), min_thresh]), gam)
                        vmin,vmax=0,np.amax(np.amax(temp_im2))

                    else:
                        temp_im2 *= 65535.0 / np.amax(temp_im1)
                        # Option 1:  we rescale to the max of the image as a whole, to avoid
                        # spurious additional spots in extra bright cells.
                        # input_data = skimage.exposure.adjust_gamma(
                        #     ims[expt_vals['channels'].index(expt_vals['HADA_channel'])][
                        #         ymin:ymax, xmin:xmax] * 65535.0 / np.amax(norm_vals), gam)
                        # Option 2: we rescale to the median of the cell in question, provided it's above a certain
                        # brightness level
                        # print(skimage.__version__)
                        # print(ymin, ymax, xmin, xmax)
                        input_data = skimage.exposure.adjust_gamma(
                            ims[expt_vals['channels'].index(expt_vals['HADA_channel'])][
                                ymin:ymax, xmin:xmax] / np.amax([np.mean(temp_im1[coords]), min_thresh]), gam)
                        vmin, vmax = 0, np.amax(np.amax(temp_im2))
                    out = skimage.feature.blob_log(input_data, min_sigma=min_sig, max_sigma=max_sig, num_sigma=10,
                                                   threshold=thresh, overlap=ovrlap, log_scale=False)
                    # Plotting the cell + outline
                    fig = plt.figure(figsize=[10, 5])
                    ax=plt.subplot(1, 3, 1)
                    # plt.imshow(temp_im2,vmin=vmin,vmax=vmax)
                    plt.imshow(temp_im2, norm=colors.PowerNorm(gamma=0.5))
                    ax.grid(False)
                    yv, xv = np.array([int(val[0]) for val in out]), np.array([int(val[1]) for val in out])
                    plt.plot(xv, yv, linestyle='', marker=',', color='w')
                    coords_comb = [(yv[ind1], xv[ind1]) for ind1 in np.arange(len(yv))]
                    plt.title('Spots={0}'.format(len(xv)), fontsize=10.0)

                    # Now we make a binary image that we will then dilate to filter for the correct spots.
                    bin_im = np.zeros(ims[expt_vals['channels'].index(expt_vals['HADA_channel'])].shape)
                    bin_im[coords] = 1
                    bin_im = bin_im[ymin:ymax, xmin:xmax]
                    bin_im = scipy.ndimage.binary_dilation(bin_im, iterations=2)
                    # Detecting this new mask
                    lab_im = skimage.measure.label(bin_im)
                    props = skimage.measure.regionprops(lab_im)
                    plt.subplot(1, 3, 2)
                    plt.imshow(bin_im)
                    coords_comb2 = [(temp[0], temp[1]) for temp in props[0].coords]
                    # Now we check if the detected spots are in the cell interior
                    detected_spots = [val in coords_comb2 for val in coords_comb]
                    ax = plt.gca()
                    ax.grid(False)
                    for temp_ind in np.nonzero(detected_spots)[0]:
                        # Define the circle's properties
                        center_x = xv[temp_ind]
                        center_y = yv[temp_ind]
                        radius = out[temp_ind, 2]
                        # Create the Circle patch object
                        circle = patches.Circle((center_x, center_y), radius,
                                                edgecolor='g', facecolor='none', linewidth=3)
                        # Add the circle to the axes
                        ax.add_patch(circle)
                    plt.title('Cell spots={0}'.format(np.sum(detected_spots)), fontsize=10.0)

                    # Now we look at the cell periphery
                    bin_im1 = np.zeros(ims[expt_vals['channels'].index(expt_vals['HADA_channel'])].shape)
                    bin_im1[temp_pix_o[:, 0], temp_pix_o[:, 1]] = 1
                    bin_im1 = bin_im1[ymin:ymax, xmin:xmax]
                    bin_im1 = scipy.ndimage.binary_dilation(bin_im1, iterations=2)

                    plt.subplot(1, 3, 3)
                    plt.imshow(bin_im1)
                    # Detecting this new mask
                    lab_im1 = skimage.measure.label(bin_im1)
                    props = skimage.measure.regionprops(lab_im1)
                    coords_comb3 = [(temp[0], temp[1]) for temp in props[0].coords]
                    detected_spots_periphery = [val in coords_comb3 for val in coords_comb]
                    ax = plt.gca()
                    ax.grid(False)
                    for temp_ind in np.nonzero(detected_spots_periphery)[0]:
                        # Define the circle's properties
                        center_x = xv[temp_ind]
                        center_y = yv[temp_ind]
                        radius = out[temp_ind, 2]
                        # Create the Circle patch object
                        circle = patches.Circle((center_x, center_y), radius,
                                                edgecolor='g', facecolor='none', linewidth=3)
                        # Add the circle to the axes
                        ax.add_patch(circle)
                    # plt.plot(xv,yv,linestyle='',marker='x', color='w')
                    plt.title('Wall spots={0}'.format(np.sum(detected_spots_periphery)), fontsize=10.0)
                    spot_num = sum(detected_spots)
                    plt.suptitle(
                        expt_vals['expt_id'][1:] + ', ' + expt_vals['cond'] + ', scene:' + str(scene) + ', cell:' + str(
                            temp_cell))
                    fig.savefig(fluo_im_out_path)
                    temp_df['Cell spots'] = float(np.sum(detected_spots))
                    temp_df['Wall spots'] = float(np.sum(detected_spots_periphery))
                    temp_df['Length']=mat['lcell'][temp_cell, scene-1]
                    temp_df['Cell spots/length']=temp_df['Cell spots']/temp_df['Length']
                    temp_df['Wall spots/length'] = temp_df['Cell spots'] / temp_df['Length']
                    cell_im_out_path = './outputs'+expt_vals['expt_id'] +  '/cell_segments'+expt_vals['expt_id'] + '_' + expt_vals['cond'] + '_s{0}_cell{1}_adj'.format(
                        str(scene).zfill(3),temp_cell)
                    fig.savefig(cell_im_out_path,bbox_inches='tight')
                    plt.clf()
                    temp_df_out = temp_df_out.append(temp_df,ignore_index=True)
            fig = plt.figure(figsize=[8, 8])
            plt.imshow(temp_im.transpose())
            plt.axis('off')
            fig.savefig(fluo_im_out_path + '_adj.png', dpi=300, bbox_inches='tight')
            plt.clf()
    return temp_df_out, saved_ims


def hydrolysis_outline_smart_bkgd(expt_vals, label=False, thresh_int=2000, redo=False, temp_df_out=None, timelapse_file_storage=False):
    # imports data from .m files and collates it in a nice fashion, combining fluorescence data from labeling images
    # that are stored in a systematic fashion
    # differs from previous version by also averaging fluorescence over cell outline
    col = ['Integrated F{0}'.format(c) for c in expt_vals['channels']]+['Average F{0}'.format(c) for c in expt_vals['channels']] + ['Average outline F{0}'.format(c) for c in expt_vals['channels']] +['Integrated outline F{0}'.format(c) for c in expt_vals['channels']]
    if label:
        col += ['Labeled']
    col += ['Time (secs)', 'Scene', 'Cellnum', 'Length (microns)', 'Condition', 'Date']
    fresh_df = temp_df_out is None  # boolean for whether we have previous conditions or not
    prev_num = 0
    if fresh_df:
        temp_df_out = pd.DataFrame(columns=col)
    else:
        prev_num += np.amax(temp_df_out.Cellnum)
    temp_df_out1 = pd.DataFrame(columns=temp_df_out.columns)  # we use this to save all the info from the current
    # print(temp_df_out1.head(10))
    # condition
    num_timepoints = expt_vals['num_timepoints']
    # Only cells that have been selected based on the bf image filtering
    
    # generating the connectivity mask that we will use to generate the cell outlines below
    conn_mask = np.zeros([5, 5])
    conn_mask[2, 2] = 1
    conn_mask = dilation(dilation(conn_mask))

    sig = 100

    for scene in range(1, expt_vals['num_scenes']+1):
        if timelapse_file_storage:  # this part allows this code to be used both by hyrolysis_timelapses (False) and
            # lysis_timelapses (True) for file-storage type
            dir_path = expt_vals['base_path'] + expt_vals['expt_id'] + expt_vals[
                'expt_id'] + '_s{0}_1_a'.format(str(scene).zfill(3))
            temp_path = dir_path + expt_vals['expt_id'] + '_s{0}'.format(
                str(scene).zfill(3)) + '_BT_felix.mat'
        else:
            dir_path = expt_vals['base_path'] + expt_vals['expt_id']+'/'+expt_vals['condition'] + expt_vals['expt_id'] + '_' + expt_vals['condition']+'_s{0}_1_a'.format(str(scene).zfill(3))
            temp_path = dir_path + expt_vals['expt_id'] + '_'+expt_vals['condition']+'_s{0}'.format(str(scene).zfill(3)) + '_BT_felix.mat'
        mat = scipy.io.loadmat(temp_path)
        for time in range(1, num_timepoints+1):
            print('reached condition', expt_vals['condition'], 'scene {0}, timepoint {1}'.format(scene, time))
            num_cells = np.sum(~np.isnan(mat['lcell'][:, time-1]))
            # Now we load all the fluorescence images for this scene
            ims = []
            for channel in expt_vals['channels']:
                if timelapse_file_storage:  # this part allows this code to be used both by hyrolysis_timelapses (False)
                    # and lysis_timelapses (True) for file-storage type
                    fluor_base_path = expt_vals['base_path'] + expt_vals['expt_id'] + \
                                      expt_vals['expt_id'] + '_s{0}_{1}_a'.format(
                        str(scene).zfill(3), str(channel))
                    fluo_im_path = fluor_base_path + expt_vals['expt_id'] + \
                                   '_s{0}_a{1}.tif'.format(str(scene).zfill(3), str(time).zfill(4))
                    fluo_im_out_path = fluor_base_path + expt_vals['expt_id'] + \
                                       '_s{0}_a{1}_adj'.format(str(scene).zfill(3), str(time).zfill(4))
                else:
                    fluor_base_path = expt_vals['base_path'] + expt_vals['expt_id']+'/'+expt_vals['condition'] + expt_vals['expt_id'] + '_' + expt_vals['condition']+'_s{0}_{1}_a'.format(str(scene).zfill(3),str(channel))
                    fluo_im_path = fluor_base_path + expt_vals['expt_id']+'_'+expt_vals['condition']+'_s{0}_a{1}.tif'.format(str(scene).zfill(3), str(time).zfill(4))
                    fluo_im_out_path = fluor_base_path + expt_vals['expt_id']+'_'+expt_vals['condition']+'_s{0}_a{1}_adj'.format(str(scene).zfill(3), str(time).zfill(4))
                if os.path.exists(fluo_im_out_path+'.npy') and not(redo):
                    im_adj = np.load(fluo_im_out_path+'.npy').transpose()
                else:
                    # background correction using the mask of cells
                    im = skimage.io.imread(fluo_im_path).transpose()  # transpose since rows and columns are swapped in matlab coords and we want to make it line up fine.

                    mask = np.ones(expt_vals['im_shape'])
                    for temp_cell in np.nonzero(~np.isnan(mat['lcell'][:, time-1]))[0]:
                        # if not (len(mat['pixels'][temp_cell, time-1]) == 0):  # all segments
                            # temp_pix = mat['pixels'][temp_cell, time-1].astype('int')
                            # temp_pix-=1 # account for the fact that in matlab, these linear indices will start from 1 rather than 0
                            # if np.median(im[np.unravel_index(temp_pix, expt_vals['im_shape'])])>1000: # we only include this in the cutoff calculation if it is fluorescent
                                # mask[np.unravel_index(temp_pix, expt_vals['im_shape'])] = np.nan
                        if not (len(mat['B'][temp_cell, time-1]) == 0):
                            # We use the boundary values to calculate the cutoff since this is the labeled part.
                            temp_pix_o = mat['B'][temp_cell, time-1].astype('int')
                            temp_pix_o -= 1
                            # account for the fact that in matlab, these linear indices will start from 1 rather than 0
                            if np.median(im[temp_pix_o[:, 0], temp_pix_o[:, 1]]) > thresh_int:
                                # we only include this in the cutoff calculation if it is fluorescent
                                mask[temp_pix_o[:, 0], temp_pix_o[:, 1]] = np.nan
                    cutoff = np.nanmax([thresh_int, np.mean(im[np.isnan(mask)].flatten())-np.std(im[np.isnan(mask)])])
                    print('cutoff = {0}'.format(cutoff))
                    # Now we create a mask for pixels with high intensity:
                    im1 = im.copy()
                    im1[np.nonzero(im > cutoff)] = 0  # this mask sets to zero any pixels with intensity greater than the
                    # cutoff (i.e. similar to the segmented cells)
                    umax = np.amax(im1.flatten())
                    im1 = im1/umax  # normalizing prior to smoothing
                    temp_out1 = skimage.filters.gaussian(im1, sigma=sig)  # gaussian filter to smooth over these areas

                    im2 = im < cutoff  # binary image to correct for zero values in mask
                    temp_out2 = skimage.filters.gaussian(im2, sigma=sig)  # normalizing gaussian filter

                    bkgd = umax*temp_out1/temp_out2  # normalized background values

                    im_adj = im-bkgd
                    im_adj[np.nonzero(im_adj < 0)] = 0
                    skimage.io.imsave(fluo_im_out_path+'.tif', im_adj.transpose())
                    np.save(fluo_im_out_path+'.npy', im_adj.transpose())
                    ###
                ims.append(im_adj)  # channel images for this timepoint

                # saved_ims.append(im_adj)
            temp_figs = []
            fig = plt.figure(figsize=[8, 8])
            temp_im = np.copy(ims[-1])
            print('Number of cells = {0}'.format(np.sum(~np.isnan(mat['lcell'][:, time-1]))))

            cells = np.nonzero(np.nanmax(~np.isnan(mat['lcell']), axis=1))  # total number of cells in this scene
            for temp_cell in np.nonzero(~np.isnan(mat['lcell'][:, time-1]))[0]:
                if not ((len(mat['pixels'][temp_cell, time-1]) == 1) or (len(mat['B'][temp_cell, time-1]) == 1)):
                    cellnum = np.nonzero(cells[0] == temp_cell)[0][0] + 1  # unique cell number starting from one

                    temp_pix = mat['pixels'][temp_cell, time-1].astype('int')
                    temp_pix -= 1
                    # account for the fact that in matlab, these linear indices will start from 1 rather than 0
                    temp_df = {}
                    coords = np.unravel_index(temp_pix, expt_vals['im_shape'])

                    # generating the cell outlines
                    # temp_mat=np.zeros(expt_vals['im_shape'])
                    # temp_mat[coords]=1
                    # outline_coords=np.nonzero(dilation(temp_mat,selem=conn_mask)*(erosion(temp_mat,selem=conn_mask)==0))
                    temp_pix_o = mat['B'][temp_cell, time-1].astype('int')
                    temp_pix_o -= 1
                    # account for the fact that in matlab, these linear indices will start from 1 rather than 0

                    # print('temp pix', len(temp_pix), len(mat['pxls'][temp_cell, time-1]))
                    # print(temp_cell)
                    # if not (len(mat['boun'][temp_cell, time-1].astype('int')) == 0):

                    if np.median(temp_im[coords]) > thresh_int:
                        # we only include this in the cutoff calculation if it is fluorescent
                        temp_im[coords] = 25000  # for visualization purposes
                        temp_im[temp_pix_o[:, 0], temp_pix_o[:, 1]] = 50000  # for visualization purposes

                    # del temp_mat
                    for channel in expt_vals['channels']:
                        temp_vec_cell = ims[expt_vals['channels'].index(channel)][coords]
                        temp_df['Integrated F{0}'.format(channel)] = [np.sum(temp_vec_cell)]
                        temp_df['Average F{0}'.format(channel)] = [np.mean(temp_vec_cell)]
                        # outline calculations
                        temp_vec_outline = ims[expt_vals['channels'].index(channel)][temp_pix_o[:, 0], temp_pix_o[:, 1]]
                        temp_df['Integrated outline F{0}'.format(channel)] = [np.sum(temp_vec_outline)]
                        temp_df['Average outline F{0}'.format(channel)] = [np.mean(temp_vec_outline)]
                        temp_df['CV outline F{0}'.format(channel)] = [np.std(temp_vec_outline)/np.mean(temp_vec_outline)]
                        temp_df['SD outline F{0}'.format(channel)] = [np.std(temp_vec_outline)]
                        if label and channel == expt_vals['class_channel']:
                            temp_df['Labeled'] = [np.mean(ims[expt_vals['channels'].index(channel)][np.unravel_index(temp_pix, ims[0].shape)]) > expt_vals['cutoff']]
                    temp_df['Time (secs)'] = [(time-1)*expt_vals['timestep']]
                    temp_df['Scene'] = [scene]
                    temp_df['Condition'] = [expt_vals['condition']]
                    temp_df['Date'] = [expt_vals['date']]
                    if scene == 1:
                        temp_df['Cellnum'] = [cellnum+prev_num]
                        # counting from prev_num+1 to prev_num+N in the first scene. We accumulate over all
                        # scenes and conditions. Note that this extra counter starts from 1 with each timepoint.
                    else:
                        temp_df['Cellnum'] = [np.amax(temp_df_out1[temp_df_out1.Scene != scene].Cellnum) + cellnum]
                        # print(temp_df)
                    temp_df['Length (microns)'] = [mat['lcell'][temp_cell, time-1]]
                    temp_df1 = pd.DataFrame(temp_df)
                    # print(temp_df1.head())
                    if timelapse_file_storage == 0 and (temp_df1['Average F{0}'.format(expt_vals['FDAA_channel'])].loc[0] > thresh_int):
                        # We are only including data for which the cells are fluorescent!
                        # print("Hello sailor")
                        temp_df_out1 = pd.concat([temp_df_out1, temp_df1])
            # print(temp_df_out1.head(10))
            # print(np.amax(temp_im))
            plt.imshow(temp_im.transpose())
            fig.savefig(fluo_im_out_path+'_adj.png', dpi=300, bbox_inches='tight')
            plt.clf()
    temp_df_out = pd.concat([temp_df_out, temp_df_out1], ignore_index=True)
    return temp_df_out