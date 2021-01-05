# VisualEyeTracker
A visual eye tracker based on a small deep learning models capable of running smoothly on modern microcomputers like Raspberry Pi 4. Currently it's only possible to train new models with included data (see below) and run it locally on your machine on a video that meets certain requirements described in the "Running locally on your PC" section.

# Setup
A correctly configured python 3.8 environment is assumed. The code was tested on Ubuntu 20.04. Written with PyCharm.

Run pip install -r requirements.txt from the repository root to install the required modules.

# WIP: Training new models
For efficient training, you might want to consider getting a Nvidia GPU (even a lower tier like 1050 will be more than enough). It takes less than an hour on 1080Ti to train the default model which offers fairly solid performance.
Download the training set from here: https://drive.google.com/file/d/1lYoeLhNQT6oqqtzQQsDPucslfSniXHyi/view?usp=sharing

# Running locally on your PC
A default pre-trained model that's a variation on the original ResNet is provided in the repository in trained_models/resnet_v1 folder. You can run this model on a video in real time by running the following command from the project root:
./tester.py trained_models/resnet_v1 PATH_TO_YOUR_VIDEO
Where PATH_TO_YOUR_VIDEO is path to some mp4 file containing an eye. The video must have 320x240 resolution and the eye should fill the entire frame. Various lighting conditions, skin colors and camera angles are supported. If the model turns out to be racist, please report it, I will do my best to make it better.

# Running on Raspberry Pi or Banana Pi
TODO
