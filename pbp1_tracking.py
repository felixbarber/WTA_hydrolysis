import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
import seaborn as sns
import scipy
from scipy import stats
import sklearn
from sklearn.neighbors import KernelDensity
from skimage import feature
from skimage import measure
from skimage import exposure
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
import weakref
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import cm
import pickle
import os
from numpy.matlib import repmat

expt_id = '/260415_bFB323_Tun_Mg'
conds=['10uM_IPTG',"50min_Tun", "0min_Mg","10min_Mg","20min_Mg"]
# Note skipping 40min timepoint here because I only have one replicate for that one.
thresholds=list(0.0003*np.ones(len(conds)))
timepoints=[-5.0,50.0, 60.0, 70.0,80.0]
scene_nums=[4,4,3,4,5]
condition=['$PBP1-mNeonGreen$ Tun -> MgCl2, 4/13/26']
path = '/Volumes/data_ssd2/Barber_Lab/data/'
lscale=0.065
dt = 0.053  # frame rate

# expt_id = '/260413_bFB323_10uMIPTG_Mg'
# conds=["10uM_IPTG", "0min_Mg","10min_Mg","20min_Mg","30min_Mg","30min_10uM_IPTG"]
# # Note skipping 40min timepoint here because I only have one replicate for that one.
# thresholds=list(0.0003*np.ones(len(conds)))
# timepoints=[-5.0,0.0, 10.0, 20.0,30.0, 40.0]
# scene_nums=[2,2,3,3,3,3]
# condition=['$PBP1-mNeonGreen$ LB->MgCl2, 4/13/26']
# path = '/Volumes/data_ssd2/Barber_Lab/data/'
# lscale=0.065
# dt = 0.053  # frame rate

# expt_id = '/260413_bFB323_10uMIPTG_Tun'
# conds=['10uM_IPTG',"50min_Tun", "0min_Mg","10min_Mg","20min_Mg"]
# # Note skipping 40min timepoint here because I only have one replicate for that one.
# thresholds=list(0.0003*np.ones(len(conds)))
# timepoints=[-5.0,50.0, 60.0, 70.0,80.0]
# scene_nums=[3,3,2,2,3]
# condition=['$PBP1-mNeonGreen$ Tun -> MgCl2, 4/13/26']
# path = '/Volumes/data_ssd2/Barber_Lab/data/'
# lscale=0.065
# dt = 0.053  # frame rate

# expt_id = '/260324_bFB323_10uMIPTG_Mg'
# conds=['10uM_IPTG',"50min_Tun", "0min_Mg","10min_Mg","20min_Mg"]
# # Note skipping 40min timepoint here because I only have one replicate for that one.
# thresholds=list(0.0003*np.ones(len(conds)))
# timepoints=[-5.0,50.0, 60.0, 70.0,80.0]
# scene_nums=[4,4,4,4,4]
# condition=['$PBP1-mNeonGreen$ Tun->MgCl2, 3/24/26']
# path = '/Volumes/data_ssd2/Barber_Lab/data/'
# lscale=0.065
# dt = 0.045  # frame rate

# Saving the experimental parameters
temp = {id: expt_id, 'conds': conds, 'scene_nums': scene_nums, 'timepoints': timepoints, 'condition': condition,
        'thresholds': thresholds}
temp_path = path + expt_id + expt_id + '_condition_parameters.pkl'
with open(temp_path, 'wb') as output:  # Overwrites any existing file.
    pickle.dump(temp, output, pickle.HIGHEST_PROTOCOL)
# exit()
perform_tracking = True
save_image_output = True
prog_path = path + expt_id + expt_id + '_progress_tracker.pkl'


# exit()

def save_object(obj, filename):
    # Code taken from:
    # https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


class trace(object):  # we will define an object called a trace for each MreB filament.
    trackCount = 0  # total number of tracks

    def __init__(self, parameters):  # parameters = [coords(y,x), frame]
        self.coords = [parameters[0]]
        self.frames = [parameters[1]]
        self.displacement = []
        trace.trackCount += 1
        self.id = trace.trackCount

    def timepoint(self, parameters):  # parameters = [coords(y,x), frame]
        self.coords = self.coords + [parameters[0]]
        self.frames = self.frames + [parameters[1]]
        self.displacement = self.displacement + [parameters[2]]


def filter_range(temp_vals):
    return temp_vals1


# Progress tracker to check whether a given scene has already been analyzed or not
if os.path.exists(prog_path):
    with open(prog_path, "rb") as f:
        done_scenes = pickle.load(f)
else:
    done_scenes = {}
    for cond in conds:
        for scene_num in range(1, scene_nums[conds.index(cond)] + 1):
            done_scenes[cond + '_s' + str(scene_num)] = False
            # if cond==conds[0] and scene_num==1:
            # done_scenes[cond+'_s'+str(scene_num)]=True
# Now we begin the analysis
# done_scenes['20min_s1']=False
# done_scenes['20min_s2']=False
for cond in conds:
    for scene_num in range(1, scene_nums[conds.index(cond)] + 1):
        if not (done_scenes[cond + '_s' + str(scene_num)]):
            # Image analysis parameters. Optimized for 210128_FB2_TIRF_LB experiment scene 2.
            temp_path = path + expt_id + '/' + cond + expt_id + '_' + cond + '_s{0}'.format(
                str(scene_num).zfill(3))
            gam = 1.0
            thresh = thresholds[conds.index(cond)]
            slope_cutoff, R2_cutoff = 0.8, 0.8
            lin_reg_thresh = 0.9
            # thresh=0.002 # for mNeonGreen
            min_sig = 2.0
            max_sig = 4
            ovrlap = 0.6
            # thresh_rel = 0.5
            max_timepoints = 45
            print(cond + ', Scene {0}'.format(scene_num))

            data = io.imread(temp_path + '.tif')

            ####################
            # Now we try to figure out what the foreground area is.
            temp_edges = np.zeros([data.shape[1], data.shape[2]])
            for timepoint in range(data.shape[0]):
                temp_edges += skimage.feature.canny(data[timepoint, :, :], sigma=6, low_threshold=0.2,
                                                    high_threshold=0.8, use_quantiles=True)
            # Distance transforming from the edges map
            temp_out = scipy.ndimage.distance_transform_cdt(temp_edges == 0, metric='chessboard')
            # Filtering to find the background based on size.
            label_objects, nb_labels = ndi.label(temp_out > 5)
            sizes = np.bincount(label_objects.ravel())
            sizes[0] = 0
            mask_sizes = sizes > 1000
            background = mask_sizes[label_objects]
            # expanding the background
            background = skimage.morphology.binary_dilation(background, footprint=np.ones((15, 15)))
            # removing small holes in the background
            label_objects, nb_labels = ndi.label(background == 0)
            sizes = np.bincount(label_objects.ravel())
            sizes[0] = 0
            mask_sizes = sizes > 100
            foreground = mask_sizes[label_objects]
            background = foreground == 0
            rescaling = None  # This gives the extent to which we expand the image before finding puncta. This may be helpful
            # since the spot detection is limited to individual pixel coordinates so we are losing some accuracy.
            # To avoid using this, just set rescaling to None
            if not (rescaling is None):
                rescaled_min_sig = rescaling * min_sig
                rescaled_max_sig = rescaling * max_sig
            # plotting the end result
            fig = plt.figure(figsize=[15, 15])
            plt.subplot(1, 3, 1)
            im = np.amax(data, axis=0)
            plt.imshow(im)
            plt.subplot(1, 3, 2)
            plt.imshow(background)
            plt.subplot(1, 3, 3)
            temp_im_out = im.copy()
            temp_im_out[np.nonzero(background)] = 0.0
            plt.imshow(temp_im_out)
            fig.savefig(temp_path + '_background_detection.png', bbox_inches='tight')
            plt.clf()
            skimage.io.imsave(temp_path + '_background.tif', 255 * np.ones(background.shape) * background)

            cell_area = np.sum(background == 0)
            ###################

            num_timepoints = np.amin([data.shape[0], max_timepoints])
            # num_timepoints = 10  # This is an option when you want to troubleshoot this, to decrease the number of
            # timepoints.
            if perform_tracking:
                # Turns out we didn't actually use the background data, since all I'm looking at for the time being is filament
                # tracking, irrespective of what cell that filament is in.
                # bkgd_data = io.imread(path+expt_id+expt_id+'_s{0}_crop_bkgd.tif'.format(str(scene_num).zfill(3)))

                # STEP 1: First we actually do the spot finding for each pixel.
                outputs = []
                for i0 in range(num_timepoints):
                    input_data = skimage.exposure.adjust_gamma(data[i0, :, :], gam)
                    # first we simply try to find a good threshold to keep the number of points more or less the same. We expect that the
                    # threshold must go down from where it was previously, but not too far with each timestep. Varies between ~0.005 for timepoint
                    # 1 and 0.002 for timepoint 30
                    if not (rescaling is None):
                        rescaled_input_data = skimage.transform.resize(input_data,
                                                                       rescaling * np.asarray(input_data.shape))
                        out = skimage.feature.blob_log(rescaled_input_data, min_sigma=rescaled_min_sig,
                                                       max_sigma=rescaled_max_sig, num_sigma=10, threshold=thresh,
                                                       overlap=ovrlap, log_scale=False)
                    else:
                        out = skimage.feature.blob_log(input_data, min_sigma=min_sig, max_sigma=max_sig, num_sigma=10,
                                                       threshold=thresh, overlap=ovrlap, log_scale=False)
                    if i0 != 0:
                        while out.shape[0] < outputs[0].shape[0]:
                            # if we are measuring fewer points than the previous timestep and the first timestep.
                            thresh -= 0.01 * thresh  # exponentially decreasing threshold
                            # thresh-=0.000025
                            out = skimage.feature.blob_log(input_data, min_sigma=min_sig, max_sigma=max_sig,
                                                           num_sigma=10, threshold=thresh, overlap=ovrlap,
                                                           log_scale=False)
                        if not (rescaling is None):
                            rescaled_input_data = skimage.transform.resize(input_data,
                                                                           rescaling * np.asarray(input_data.shape))
                            out = skimage.feature.blob_log(rescaled_input_data, min_sigma=rescaled_min_sig,
                                                           max_sigma=rescaled_max_sig, num_sigma=10, threshold=thresh,
                                                           overlap=ovrlap, log_scale=False)
                    if not (rescaling is None):
                        outputs.append(out / rescaling)
                    else:
                        outputs.append(out)
                    if i0 % 4 == 0:
                        print('Completed {0} reps, threshold={1}, num spots = {2}'.format(i0 + 1, thresh, out.shape[0]))

                # STEP 2: Now we do the tracking for each trace

                tracks = []
                # for i0 in range(data.shape[0]):
                dist_thresh = 2.0
                for i0 in range(num_timepoints):  # note that frame number starts counting from zero.
                    coords = np.copy(outputs[i0][:, :2])  # y,x coords
                    full_ids = [id(obj) for obj in tracks]
                    if i0 > 0:  # in this case we can start doing the actual tracking.
                        dist_map = euclidean_distances(coords_prev, coords)
                        mapping = np.argmin(dist_map, axis=1)
                        min_dist = np.amin(dist_map, axis=1)
                        # Correct for multiple initial traces mapping to the same current coordinate

                        to_remove = []  # list of indices to remove from coords before generating completely new traces
                        for i1 in range(coords_prev.shape[0]):
                            #             print(dist_map[i1,mapping[i1]])
                            if dist_map[i1, mapping[i1]] < dist_thresh and np.argmin(dist_map[:, mapping[i1]]) == i1:
                                # if this distance is below the threshold and the point in the previous timepoint is indeed the closest to
                                # our current one
                                update_vars = [coords[mapping[i1], :], i0, dist_map[i1, mapping[i1]]]
                                tracks[full_ids.index(ids[i1])].timepoint(update_vars)  # find the relevant trace and
                                # add a new timepoint to it
                                to_remove.append(mapping[i1])  # this tracks the new spots which have now been assigned
                        #         print(coords.shape[0],len(to_remove))
                        coords = np.delete(coords, to_remove, axis=0)  # coords now corresponds to only unassigned new
                        # spots.
                    #         print(coords.shape[0])

                    for ind in range(coords.shape[0]):
                        tracks.append(trace([coords[ind, :], i0]))  # generating trace instances for each new,
                        # unassigned spot
                    coords_prev = np.asarray(
                        [obj.coords[-1] for obj in tracks if obj.frames[-1] == i0])
                    # making an array of all previous objects
                    #     print(coords[0,:],coords_prev[0,:])
                    ids = [id(obj) for obj in tracks if obj.frames[-1] == i0]
                ########################################################################################
                # Now we filter to find only the tracks that have a length greater than or equal to 3.
                tracks_filt = [obj for obj in tracks if len(obj.frames) > 2]
                # tracks_filt = [obj for obj in tracks]
                ########################################################################################
                # Now we filter to find only the tracks that are well described by a linear fit.

                # lscale=0.08 # Microns per pixel. ############## should update this.

                for obj in tracks_filt:
                    temp_coords = np.copy(obj.coords)
                    temp_frames = np.copy(obj.frames)
                    # This generates the distances between our initial point and each subsequent point over time
                    temp_dists = np.linalg.norm(
                        temp_coords - repmat(np.reshape(temp_coords[0, :] + 0.001 * np.random.normal(size=2),
                                                        [1, 2]), temp_coords.shape[0], 1), axis=1)
                    xv = dt * (temp_frames - temp_frames[0] + 0.001)
                    yv = temp_dists * lscale
                    temp_vals1 = scipy.stats.linregress(xv, yv)
                    temp_vals = scipy.stats.linregress(np.log(xv), np.log(yv))
                    obj.linear_fit = temp_vals
                    obj.speed_fit = temp_vals1
                    if temp_vals1[2] > lin_reg_thresh:
                        obj.lin = True
                    else:
                        obj.lin = False
                    if (temp_vals[0] > slope_cutoff) and (temp_vals1[2] > lin_reg_thresh):
                        obj.ballistic = True
                    else:
                        obj.ballistic = False

                save_object(tracks_filt, temp_path + '_tracked_filt.pkl')

            # if this part is run, then we save the fluorescence timelapse in RGB format with the MreB filaments tracked.
            if save_image_output:
                # convert image to grayscale
                # with open(path + expt_id + expt_id + '_s{0}_tracked_filt.pkl'.format(str(scene_num).zfill(3))) as input:
                # tracks_filt=pickle.load(input)

                theta = np.linspace(0.0, 6.28, 8)
                temp_im1 = np.repeat(data[:, :, :, np.newaxis], 3, axis=3) / 65535.0
                # temp_im1=skimage.exposure.adjust_gamma(temp_im1,gam)
                for frame in range(num_timepoints):
                    ########################################################################################
                    # frame_tracks = [obj for obj in tracks_filt if frame in obj.frames and obj.lin]
                    frame_tracks = [obj for obj in tracks_filt if frame in obj.frames]
                    ########################################################################################
                    for obj in frame_tracks:
                        temp_rgb_vals = cm.tab20(obj.id % 20)
                        temp_mask = np.zeros(data.shape[1:])
                        cent = obj.coords[obj.frames.index(frame)]
                        vals = np.array([cent[0] + 1.5 * np.sin(theta), cent[1] + 1.5 * np.cos(theta)]).astype(int)
                        # This is an array of values that will be filled in the image.
                        # Now we filter these vals to make sure they appear within our grid
                        ym = data.shape[1] - 1
                        xm = data.shape[2] - 1
                        to_remove = [vals[0, i0] > ym or vals[0, i0] < 0 or vals[1, i0] > xm or vals[1, i0] < 0 for i0
                                     in
                                     range(vals.shape[1])]
                        # print(vals.shape)
                        # print(vals)
                        vals = np.delete(vals, np.nonzero(to_remove), axis=1)
                        # print(vals)
                        # print(vals)
                        # print(zip(vals))
                        #####################################################
                        for temp_ind in range(vals.shape[1]):
                            temp_mask[vals[0, temp_ind], vals[1, temp_ind]] = 1
                        # temp_mask[zip(vals)] = 1  # problematic for python 3.
                        #####################################################
                        temp_mask1 = np.repeat(temp_mask[:, :, np.newaxis], 3, axis=2)
                        temp_mask_inv = temp_mask1 == 0
                        temp_im1[frame, :, :, :] = temp_im1[frame, :, :, :] * temp_mask_inv
                        temp_im1[frame, :, :, :] += temp_mask1 * np.asarray(temp_rgb_vals[:3])
                    if frame % 10 == 0:
                        print('Finished frame {0}'.format(frame))
                # saving the image output
                skimage.io.imsave(temp_path + '_tracked_filt.tif', temp_im1, metadata={'axes': 'TYXC'})
                del temp_im1


                fig = plt.figure(figsize=[20, 20])
                plt.imshow(data[0, :, :])
                for track in tracks_filt:
                    if track.ballistic:
                        temp_track = np.reshape(track.coords, [len(track.coords), 2])
                        temp_rgb_vals = cm.tab20(track.id % 20)
                        plt.plot(temp_track[:, 1], temp_track[:, 0], color=temp_rgb_vals, marker=None, linewidth=1)
                plt.axis('off')
                fig.savefig(temp_path + '_tracks_vis_ballistic.png', bbox_inches='tight')
                plt.clf()
                fig = plt.figure(figsize=[20, 20])
                plt.imshow(data[0, :, :])
                for track in tracks_filt:
                    if not (track.ballistic):
                        temp_track = np.reshape(track.coords, [len(track.coords), 2])
                        temp_rgb_vals = cm.tab20(track.id % 20)
                        plt.plot(temp_track[:, 1], temp_track[:, 0], color=temp_rgb_vals, marker=None, linewidth=1)
                plt.axis('off')
                fig.savefig(temp_path + '_tracks_vis_nb.png', bbox_inches='tight')
                plt.clf()

            done_scenes[cond + '_s' + str(scene_num)] = True
            save_object(done_scenes, prog_path)
            del data, frame_tracks, tracks, tracks_filt
        else:
            print('Done ' + cond + ', scene {0} already'.format(scene_num))