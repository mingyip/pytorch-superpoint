# Superpoint for Event Camera data


<p align="center">
  <img src="https://github.com/mingyip/pytorch-superpoint/blob/master/images/header_office.png?raw=true" width="720px">
</p>

## Abstract
We developed an interest point detector to detect interest point location from **event camera data**. This idea was adapted from the literature *"SuperPoint: Self-Supervised Interest Point Detection and Description."* Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich. [ArXiV 2018](https://arxiv.org/abs/1712.07629). [1] This code is partially based on the pytorch implementation. https://github.com/eric-yyjau/pytorch-superpoint.

Our algorithm mainly consists of two parts:
1. We first train our detector network with synthetic event data. 
2. We iteratively use the trained detector network to label interest points of a real-world dataset and 
    then train the network with the labeled dataset.


## Installation
The environment is run in python 3.6, Pytorch 1.5.0 and ROS. We ran our code with Ubuntu 18.04 and ROS Melodic. Installation instructions for *ROS* can be found [here](http://wiki.ros.org/melodic/Installation/Ubuntu). To generate syntheic event data, we used "ESIM: an Open Event Camera Simulator". You may find installation details of *ESIM* [here](https://github.com/uzh-rpg/rpg_esim).

#### To install conda env
```
conda create --name py36-sp python=3.6
conda activate py36-sp
pip install -r requirements.txt
pip install -r requirements_torch.txt # install pytorch
```

#### To install Ros Melodic 
```
sudo apt-get update
sudo apt-get install ros-melodic-desktop-full
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
```
After installed Ros, don't forget to install the Event Camera Simulator. 


## Dataset
In step 2, we used data sequences (in ros format) from [MVSEC](https://daniilidis-group.github.io/mvsec/) [2] and [IJRR](http://rpg.ifi.uzh.ch/davis_data.html) (ETH event dataset) [4] to further train our network. This code processes the events in HDF5 format. To convert the rosbags to this format, open a new terminal and source a ROS workspace. We command to use packages from https://github.com/TimoStoff/event_cnn_minimal
```
source /opt/ros/kinetic/setup.bash
python events_contrast_maximization/tools/rosbag_to_h5.py <path/to/rosbag/or/dir/with/rosbags> --output_dir <path/to/save_h5_events> --event_topic <event_topic> --image_topic <image_topic>
```



## Usage
All commands should be executed within the `pytorch-sp` folder.

#### 1) Generating Synthetic Event Data

Modify the output path in the bash script to set your synthetic data output location. We developed a python event simulator *pyV2E*, which mimics the existing ESIM simulator and generatesevent data from video/image input.
```
cd generate_data
bash generate.sh
```
#### 2) Train MagicPoint on Synthetic Shapes
```
python train.py train_base configs/magicpoint_shapes_pair.yaml magicpoint_synth --eval
```

#### 3) Export detections on Dataset
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```

#### 4) Train Superpoint on self-labeled real world dataset
```
python train4.py <train task> <config file> <export folder> --eval
```


## Results

<table style="width:100%">
<thead>
  <tr>
    <th><h3>Dataset</h3></th>
    <th><h3>Intensity Image</h3></th>
    <th><h3>&nbsp;&nbsp;Event Data&nbsp;&nbsp;</h3></th>
    <th><h3>&nbsp;&nbsp;&nbsp;&nbsp;Heamap&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</h3></th>
    <th><h3>Interest Points</h3></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>boxes_6dof</td>
    <td colspan="4"><img src="https://github.com/mingyip/pytorch-superpoint/blob/master/images/box_final.gif?raw=true" width="960px"></td>
  </tr>
  <tr>
    <td>dynamic_6dof</td>
    <td colspan="4"><img src="https://github.com/mingyip/pytorch-superpoint/blob/master/images/office_1_final.gif?raw=true" width="960px"></td>
  </tr>
  <tr>
    <td>office_spiral</td>
    <td colspan="4"><img src="https://github.com/mingyip/pytorch-superpoint/blob/master/images/office_2_final.gif?raw=true" width="960px"></td>
  </tr>
  <tr>
    <td>urban</td>
    <td colspan="4"><img src="https://github.com/mingyip/pytorch-superpoint/blob/master/images/urban_final.gif?raw=true" width="960px"></td>
  </tr>
</tbody>
</table>

## Reference
[1] DeTone, D., Malisiewicz, T., & Rabinovich, A. (2018). SuperPoint: Self-Supervised Interest Point Detection and Description. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 337-33712.

[2] Zhu, A.Z., Thakur, D., Özaslan, T., Pfrommer, B., Kumar, V., & Daniilidis, K. (2018). The Multivehicle Stereo Event Camera Dataset: An Event Camera Dataset for 3D Perception. IEEE Robotics and Automation Letters, 3, 2032-2039.

[3] Mueggler, E., Rebecq, H., Gallego, G., Delbrück, T., & Scaramuzza, D. (2017). The event-camera dataset and simulator: Event-based data for pose estimation, visual odometry, and SLAM. The International Journal of Robotics Research, 36, 142 - 149.

[4] Rebecq, H., Gehrig, D., & Scaramuzza, D. (2018). ESIM: an Open Event Camera Simulator. CoRL.

[5]. Stoffregen, T., Scheerlinck, C., Scaramuzza, D., Drummond, T., Barnes, N., Kleeman, L., & Mahony, R. (2020). Reducing the Sim-to-Real Gap for Event Cameras. ECCV.



