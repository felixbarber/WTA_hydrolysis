# Performs omnipose segmentation on aligned and background-corrected images.

# Import dependencies
import numpy as np
import omnipose

# set up plotting defaults
from omnipose.plot import imshow
# omnipose.plot.setup()
import cellpose_omni
from cellpose_omni import io
from cellpose_omni import models
from cellpose_omni.models import MODEL_NAMES
# This checks to see if you have set up your GPU properly.
# CPU performance is a lot slower, but not a problem if you
# are only processing a few images.
from omnipose.gpu import use_gpu
use_GPU = use_gpu()
import os
import matplotlib.pyplot as plt
import time

# User input:

expt_id="/260617_Tun_Spo0A_act"
conds=["bBL40_LB", "bBL40_Tun", "bBL41_LB", "bBL41_Tun","bBL42_LB", "bBL42_Tun"]
num_scenes = [10,9,10,10,7,7]
base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# expt_id="/260603_yfp_pads"
# conds=["27_tun", "27_LB", "26a_LB", "26a_tun"]
# num_scenes = [16,16, 14, 14]
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# expt_id="/260527_Tun_yocH_induction"
# conds=["bAH26_Tun", "bAH26_LB", "bBL27_LB"]
# num_scenes = [10,10, 9]
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# expt_id="/260415_bBL27_Tun_induction"
# conds=["1h_Tun", "LB"]
# num_scenes = [10,10]
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# expt_id="/260521_bAH26"
# conds=["Tun", "LB"]
# num_scenes = [14,10]
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

for cond in conds:
    directory = base_path+expt_id+"/"+cond + "/C1_blur"
    print(directory)
    tif_files = io.get_image_files(directory)
    # tif_files = [f for f in os.listdir(directory) if f.lower().endswith(".tif")]
    print(tif_files, directory)
    # imgs = [io.imread(directory+"/"+f) for f in tif_files]
    imgs = [io.imread(f) for f in tif_files]
    print("Found .tif files:")
    # for file in tif_files:
    #     print(file)

    model_name = 'bact_phase_affinity'
    # model_name = 'bact_fluor_omni'
    model = models.CellposeModel(gpu=use_GPU, model_type=model_name)

    n = [-1]  # make a list of integers to select which images you want to segment
    # n = range(nimg) # or just segment them all
    # print(n)
    # define parameters
    params = {'channels': None,  # always define this if using older models, e.g. [0,0] with bact_phase_omni
              'rescale': None,  # upscale or downscale your images, None = no rescaling
              'mask_threshold': -2,  # erode or dilate masks with higher or lower values between -5 and 5
              'flow_threshold': 0,
              # default is .4, but only needed if there are spurious masks to clean up; slows down output
              'transparency': True,  # transparency in flow output
              'omni': True,  # we can turn off Omnipose mask reconstruction, not advised
              'cluster': True,  # use DBSCAN clustering
              'resample': True,  # whether or not to run dynamics on rescaled grid or original grid
              'verbose': False,  # turn on if you want to see more output
              'tile': False,  # average the outputs from flipped (augmented) images; slower, usually not needed
              'niter': None,
              # default None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation
              'augment': False,  # Can optionally rotate the image and average network outputs, usually not needed
              'affinity_seg': True,  # new feature, stay tuned...
              }

    tic = time.time()
    masks, flows, styles = model.eval([imgs[i] for i in range(len(imgs))], **params)

    net_time = time.time() - tic

    print('total segmentation time: {}s'.format(net_time))
    io.save_masks(imgs, masks, flows, tif_files,
                  png=False,
                  tif=True,  # whether to use PNG or TIF format
                  suffix='',  # suffix to add to files if needed
                  save_flows=False,  # saves both RGB depiction as *_flows.png and the raw components as *_dP.tif
                  save_outlines=False,  # save outline images
                  dir_above=0,
                  # save output in the image directory or in the directory above (at the level of the image directory)
                  in_folders=True,  # save output in folders (recommended)
                  save_txt=False,  # txt file for outlines in imageJ
                  save_ncolor=False)  # save ncolor version of masks for visualization and editing