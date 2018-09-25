# The Saliency Learning ROS package
*v0.1.0*

## Description
This software provides a platform to run and evaluate the saliency learning as well as the exploration strategy based on RL-IAC. 
The software can be used in several modes:
* RL_IAC: basic usage
* offline_saliency: Evaluate saliency with an offline model
* offline_learning: learn an offline model of saliency
* bottom_up: evaluate bottom-up saliency
* segmentation_only: segment and create output masks

## Documentation
See below

## Folder structure
* bottom_up: bottom_up saliency library
* feature_extractor: feature extractor library
* foveal_segmentation: depth-based segmentation library
* learning_module: incremental saliency learning library
* data: deep learning & random forest models
* params: parameters definitions
* Matlab: matlab files for evaluation and plots display

## Output
An executable able to run and evaluate saliency and RL-IAC exploration.

## Usage

### Download data folder
smb://diskstation/data_thales/GitResources/personals/celine_craye/saliency_learning/data

### Preparing dataset
To get started, you need a dataset correctly formated. Currently two datasets are available at 
[link] smb://diskstation/data_thales/GitResources/personals/celine_craye/rosbag-IROS
and
[link] smb://diskstation/data_thales/GitResources/personals/celine_craye/RGBD-scebes-RLIAC

A dataset should contain at least:
* an image_data folder with rgb and depth images. RGB images should be png format (img000.png for example), and corresponding depth images should have the same name with **_depth** at the end (img000_depth.png). You can either compute the segmentation online, or use pre-computed one. For that, segmentation mask should have the same name as rgb images with _GT extension (img000_GT.png).
* a file **input_region_map.dat** containing the region associated with each image. This file is a list of elements formatted this way [image_data/imgxxx.png] = k, where imgxxx.png is the rgb image name and k is the region index.
* a file **region_map.dat** containing the navigation graph of the experiment. Each row of this file is a node and associated childs, and should be formatted this way region=[up, down, left, right, stay],[timeup, timedown,  ...],[modelidx], where region is the region index, up, down, ... are the upper, lower, ... neighbors indices. timeup, timedown is the time required to move to the neighbor region. modelidx is the index of the model used in the region.


### Parameters list
Below is the list of parameters used by the software.
#### Param files and dataset
* [inputPath] path to the images folder
* [inputRegionPath] path to the input-name -> region correspondance table
* [regionMapPath] path to the region map and action costs
* [segParamPath] path to the segmentation parameters
* [learningParamPath] path to the learning parameters
* [bottomupParamPath] path to the bottom up saliency parameters

#### Related to the experiment
* [nbSteps] number of steps in the experiments (int)
* [initialPosition] initial frame index (int)

#### related to segmentation
* [useSegImages] use segmented images : 1 calculate mask at each step : 0
* [useFloorTracker] use floor tracker when segmenting at each step : 0 no, 1 yes

#### Related to feature extraction
* [featureType] type of features extract : "Make3D", "Deep" or "Itti"

#### related to learning
* [intr_motivation_type] intrinsic motivation criterion (int) see RegionsM.cpp
* INTR_MOTIV_PROGRESS = 0; INTR_MOTIV_NOVELTY = 1; INTR_MOTIV_UNCERTAINTY = 2; INTR_MOTIV_ERROR = 3;
* [useBackward] use forward or backward for error estimation (backward is better !). 1:yes, 0:no
* [evaluationMetrics] metrics to estimate error (int) see Evaluation.h  F1_SCORE = 0; ACCURACY = 1; HARMONIC_MEAN = 2;
* [usePerFrameEval] Calculate average error by averaging the score of each frame (instead of score of each pixel) 1:yes, 0:no
* [useLongTerm] Use backward long term evaluation (by the way, please do not use it) 1:yes, 0:no
* [modelName] file name where to save the learned model

#### related to action selection and displacement
* [actionType] type of action selection strategy (int). See ActionSelectionM.cpp SELECT_CHRONO = 0; SELECT_RANDOM = 1; SELECT_IAC = 2; SELECT_RLIAC = 3;
* [nbLearnPerStay] number of input to acquire when "learning" action is selected (int)
* [learnAtEachStep] do you want to keep learning during robot displacement (instead, wait for target to be reached) 1:yes, 0:no
* [RNDActionSelection] percentage of actions selected randomly (float [0:1])
* [RNDPositionSelection] percentage of positions selected randomly (float [0:1])
* [learnMoveRatio] arbitrary increase the time difference beetween learning and moving (int)

#### display options
* [displayFrames] if string contains "rgb" and/or "seg" and/or "sal", displays rgb input/segmentation mask/saliency map
* [displayRegionState] draw internal learning state 1:yes, 0:no
* [displayWorldMap] draw world map and associated reward 1:yes, 0:no

#### related to evaluation
* [usePerRegionEval] 1:yes, 0:no
* [evalSubsamplingRate] = N. speed up evaluation by evaluating 1 pixel out of N (int)
* [evalInputPath] same as [inputPath], but for evaluation set. if empty, takes [inputPath]
* [evalRegionPath] same as [inputRegionPath], but for evaluation set. if empty, takes [inputRegionPath]
* [outputLogFile] name of the output log file. If empty, takes param filename root
* [outputSaliencyDir] directory to save saliency maps or segmentation. Please create it before running


### Launch the executable
The executable can be launched with the following syntax
```
./RL_IAC_saliencyLearning <param_file> <run_type>
```
with 
* <param_file> the link to the param.dat file
* <run_type>
- RL_IAC : basic usage
- offline_saliency : evaluate saliency with offline model
- offline_learning : learn an offline model of saliency
- bottom_up : evaluate bottom-up saliency
- segmentation_only : segment and create output masks

## Installation (linux only)

### Install dependencies
Install caffe library (take realease rc2, because it's not compiling with rc3 yet). With or without cuda, you also need OpenCV2.x (mandatory because OpenCV3 re-implemented random forests) and PCL1.7 or PCL1.8


### Copy 

## Liscences
Unknown
