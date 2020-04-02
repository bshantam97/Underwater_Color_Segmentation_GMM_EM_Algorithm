# Code for Project 3 of ENPM 673-Perception for Autonomous Robots


### Authors
Shantam Bajpai
Arpit Aggarwal


### Instructions for running the code
To run the code for the files in Code folder, follow the following commands:

```
cd Code
python yellow_buoy_1D_gauss.py 'video_path(in .mp4 format)' 'train_file_path'
```
where, video_path and train_file_path are the paths for input video and training data of the buoy. For example, running the python file on my local setup was:

```
cd Code/
python yellow_buoy_1D_gauss.py /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/detectbuoy.avi /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/buoy1/train
```

```
cd Code/
python orange_buoy_3D_gauss.py /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/detectbuoy.avi /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/buoy2/train
```

### Software Required
To run the .py files, use Python 3. Standard Python 3 libraries like OpenCV, Numpy, Scipy and matplotlib are used.
