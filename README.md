# MKZ Intent: A package for pedestrian intent recognition

## Authors
Gaetano Evangelista, Houston Lucas, Cayler Miley, Jamie Poston

## Summary
MKZ Intent is a machine learning module to identify future locations based on 3D positions of each pedestrian. This project was created during the 2017-2018 school year for the University of Nevada, Reno Senior Project course. A model is trained on BVH data from the [CMU MOCAP dataset](http://mocap.cs.cmu.edu/) which uses pose data as input and outputs the projected future location of the hips. The projection is frame rate independent.

## Package Installation
PyTorch as well as NVIDIA CUDA and cuDNN are required to build and train the model.

### NVIDIA CUDA and cuDNN
For full installation instructions please see [CUDA](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A) and [cuDNN](https://developer.nvidia.com/cudnn).

### PyTorch
Follow the instructions at [PyTorch](http://pytorch.org/). Make sure to specify your package manager and the version of CUDA and Python you are using. We used CUDA 8.0 and Python 2.7.

### Python Dependencies
```bash
sudo pip install bvh matplotlib
sudo apt-get install msttcorefonts -qq
```