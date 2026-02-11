# WTA_hydrolysis
This repository comprises code from the paper: Wall teichoic acids confer hydrolase redundancy by regulating peptidoglycan synthesis in Bacillus subtilis

This repository contains scripts written in FIJI, MatLab and Python. All python code should be run with python 3.11.13 using the following libraries:
Matplotlib 3.7.1
Numpy 1.24.3
Scipy 1.10.1
Scikit-learn 1.2.2
Scikit-image 0.20.0
Pandas 1.5.3
Seaborn 0.13.2

The general workflow uses FIJI scripts (*.ijm) to import and build image file structures, uses MatLab to segment and track phase contrast images either for single-timepoint staining or timelapse tracking, and applies Python scripts for fluorescent quantification, single molecule tracking and plotting.

Examples of script workflows for different applications:

## Single-cell growth rate and width tracking (e.g. Figs. 1B, S2A,B,C,D):
- Image_process_updated.ijm
- Alignment_updated.ijm
- Imagealign_felix.m
- Eraseimagepart_felix_updated_blurry.m
- Bactrack_felix_tester.m
- Timelapse_growth_rate_script_truncated_time.py
### Run the steps above for each individual experiment. Then run the step(s) below to compile data across multiple experiments:
- Timelapse_compile_data.py
- Timelapse_compiled_data_plots.py


## EDA-DA labeling (Fig. S1A):
- Timepoint_imaging_v1.ijm
- Timepoint_blur_background_automated_repeat.m
- Timepoint_segmentation.m
- Cell_staining_timepoint_EDADA.py
- Cell_staining_EDADA_reformatting.py
- Cell_staining_label_saving.py
### Run the steps above for each individual experiment. Then run the step(s) below to compile data across multiple experiments:
- EDADA_plotting_paper.py 

## Single-cell growth rate and width tracking on pads (Figs. S2E, F):

- Image_process_updated.ijm
- Alignment_updated.ijm
- Imagealign_felix_pad.m
- Bactrack_felix_tester.m
- Timelapse_growth_rate_script_truncated_time.py
### Run the steps above for each individual experiment. Then run the step(s) below to compile data across multiple experiments:
- Timelapse_compile_data.py
- Timelapse_compiled_data_plots.py

## Timepoint staining imaging (Figs. 1E, S5A):
- timepoint_imaging_scenes_single_file.ijm (Fiji)
- timepoint_blur_background_automated_repeat.m
- timepoint_segmentation.m
- cell_staining_timepoint.py
- cell_staining_label_saving.py
### Run the steps above for each individual experiment. Then run the step(s) below to compile data across multiple experiments:
- cell_staining_timepoint_compilation_paper.py

## MreB tracking (Figs. 1C, 2F, S3, S9 etc.):
- tirf_processing.ijm
- Mreb_tracking_v3_py3.py
- Mreb_consolidate_timepoints.py (needed to combine finer time-resolution points into coarser timepoints for some experiments)
### Run the steps above for each individual experiment. Then run the step(s) below to compile data across multiple experiments:
- Mreb_plotting_compilation_v2_py3.py

## Single Molecule Tracking (Figs. 2D, S6C)
- tirf_processing.ijm
- single_molecule_tracking_v1.py
- pbp1_tracking_compilation.ipynb

## Bulk growth curve data
- growth_curves_reordering.py

## Lysis microscopy data
- lysis_annotation_lysed_cells_updated.ijm
- lysis_annotation_plain_cells_updated.ijm
- cell_lysis_timelapse_compiled.py
- lysis_stat_tests.ipynb (to calculate statistical significance)

## Permeability assay — see dedicated ReadMe file.
## PBP1-HADA colocalization
- FDAA_PBP1_cross_correlation.ipynb
