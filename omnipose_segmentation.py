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
# num_scenes = 8
# expt_id="/210723_FB8_Tun_Response"
# base_path="/Volumes/data_ssd2/Rojas_Lab/data/"

# num_scenes = 6
# expt_id="/211216_bFB8_Tun_gr"
# base_path="/Volumes/data_ssd1/Rojas_Lab/data/"

# num_scenes = 6
# expt_id="/211020_bFB8_Tun"
# base_path="/Volumes/data_ssd2/Rojas_Lab/data/"

# num_scenes = 3
# expt_id="/260612_bFB292_IPTG"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

num_scenes = 3
expt_id="/260612_bFB295_IPTG_Mg"
base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/260612_bFB292_IPTG_Mg"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/260611_bFB292_IPTG_Mg"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/260531_bFB66_LB_long"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/260531_bFB292_IPTG_Mg"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/260527_bFB66_LB_long"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/260527_bFB292_IPTG_Mg"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/260521_bFB292_IPTG_Mg"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/260520_bFB66_NLS_PBS_Mg"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/260520_bFB66_NLS_PBS_Mg"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/260519_bFB291_IPTG"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/260519_bFB66_Tun_NLS_PBS_Mg_v2"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/260519_bFB66_Tun_NLS_PBS_Mg"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/260306_bFB295_IPTG"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/260506_bFB66_Tun_NLS_PBS_Mg_v2"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 3
# expt_id="/260505_bFB66_NLS_PBS_Mg"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"


# num_scenes = 4
# expt_id="/260416_bFB295_IPTG_induction"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 3
# expt_id="/260505_bFB66_Tun_NLS_PBS_Mg"
# base_path="/Volumes/data_ssd3/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/260504_bFB66_NLS_PBS_Mg_v2"
# base_path="/Volumes/data_ssd2/Barber_Lab/data/"

# num_scenes = 3
# expt_id="/260504_bFB66_NLS_PBS_Mg"
# base_path="/Volumes/data_ssd2/Barber_Lab/data/"

# num_scenes = 3
# expt_id="/260430_bFB66_15MSorb_noMg"
# base_path="/Volumes/data_ssd2/Barber_Lab/data/"

# num_scenes = 3
# expt_id="/260430_bFB66_2MSorb_10mMMgCl2_LB"
# base_path="/Volumes/data_ssd2/Barber_Lab/data/"

# num_scenes = 3
# expt_id="/260430_bFB66_15MSorb_10mMMgCl2"
# base_path="/Volumes/data_ssd2/Barber_Lab/data/"

# num_scenes = 2
# expt_id="/260428_bFB66_2MSorb_10mMMgCl2_noAF"
# base_path="/Volumes/data_ssd2/Barber_Lab/data/"

# num_scenes = 2
# expt_id="/260428_bFB66_2MSorb_10mMMgCl2"
# base_path="/Volumes/data_ssd2/Barber_Lab/data/"

# num_scenes = 2
# expt_id="/260428_bFB66_2MSorb_20mMMgCl2"
# base_path="/Volumes/data_ssd2/Barber_Lab/data/"

# num_scenes = 4
# expt_id="/250402_bFB66_LB"
# base_path="/Volumes/data_ssd2/Rojas_Lab/data/"


# num_scenes = 3
# expt_id="/250326_bFB295_IPTG_induction"
# base_path="/Volumes/data_ssd2/Rojas_Lab/data/"

# num_scenes = 4
# expt_id="/250324_bFB291_IPTG_induction"
# base_path="/Volumes/data_ssd2/Rojas_Lab/data/"

# base_path="/Volumes/data_ssd2/Rojas_Lab/data"
# expt_id="/250408_bFB291_IPTG_induction"
# num_scenes=4

# base_path="/Volumes/data_ssd2/Rojas_Lab/data"
# expt_id="/260216_bFB292_IPTG_Mg"
# num_scenes=4


# base_path="/Volumes/data_ssd2/Rojas_Lab/data"
# expt_id="/250324_bFB291_IPTG_induction"
# num_scenes=4

# base_path="/Volumes/data_ssd2/Barber_Lab/data"
# expt_id="/260414_bFB292_IPTG"
# num_scenes=3

# base_path="/Volumes/data_ssd2/Barber_Lab/data"
# expt_id="/260316_bBF292_IPTG_Mg"
# num_scenes=4

# base_path="/Volumes/data_ssd2/Barber_Lab/data"
# expt_id="/260313_bBF292_IPTG"
# num_scenes=4

# base_path="/Volumes/data_ssd2/Rojas_Lab/data"
# expt_id="/250325_bFB293_IPTG_induction"
# num_scenes=4

# base_path="/Volumes/data_ssd2/Barber_Lab/data"
# expt_id="/260305_bFB292_IPTG"
# num_scenes=4

# base_path="/Volumes/data_ssd2/Rojas_Lab/data"
# expt_id="/250403_bFB292_IPTG_induction"
# num_scenes=2

# base_path="/Volumes/data_ssd2/Barber_Lab/data"
# expt_id="/260217_bFB292_IPTG_Mg"
# num_scenes=4

for scene in range(1,num_scenes+1): # Iterate through scenes

    directory = base_path+expt_id+expt_id+'_s'+str(scene).zfill(3)+'_1_a'
    print(directory)
    tif_files = io.get_image_files(directory)

    # tif_files = [f for f in os.listdir(directory) if f.lower().endswith(".tif")]
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