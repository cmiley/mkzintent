# I'm Walking Here!: A package for pedestrian intent recognition

## Authors
Gaetano Evangelista, Houston Lucas, Cayler Miley, Jamie Poston

## Summary
I'm Walking Here! is a machine learning module to predict the future position and pose of a person. This project was created during the 2017-2018 school year for the University of Nevada, Reno Senior Project course. The code trains a model on data from the [CMU MOCAP dataset](http://mocap.cs.cmu.edu/) which uses pose data as input and outputs the projected future location of the hips. The projection is frame rate independent.

## Package Installation
A virtual environment using Python 3.6 is recommended as well as NVIDIA GPU support. Instructions for both GPU and CPU are provided. 

### GPU Supported Installation
Install CUDA and CudNN from the [Gist](https://gist.github.com/zhanwenchen/e520767a409325d9961072f666815bb8) by Phil Chen. This guide does a much better job of reducing errors than the installation guide provided by NVIDIA. At the time of this writing, we used CUDA 9.0 and CudNN 7.4. 

Next set up a virtual environment and clone the repo: 

```bash
virtualenv -p python3.6 pir
cd pir
source bin/activate
git clone https://github.com/cmiley/mkzintent.git
```

Next find the necessary way to install [PyTorch](http://pytorch.org/) based on your CUDA configuration. If you are using CUDA 9.0 the command is simply: 

```bash
pip install torch torchvision
```

Lastly install the dependencies for graphs and BVH: 

```bash
pip install bvh matplotlib
```

### CPU-Only Installation
Create a Python virtual environment and install packages as follows (if you have trouble, then find the most up to date wheel at [PyTorch](http://pytorch.org/)): 

```bash
virtualenv -p python3.6 pir
cd pir
source bin/activate
git clone https://github.com/cmiley/mkzintent.git
pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl
pip install torchvision bvh matplotlib
```

## Running the Project
First download the dataset (in BVH format) from [cgspeed](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/cmu-bvh-conversion). It will take a while, but grab all of the .zip files and place them into a single directory. Once you have the training data directory set up, modify the MKZIntentConf.py with the directory path. 

Next run the driver: 

```bash
python driver.py
```

This creates a whitelist of the BVH files that can be used for training. After the whitelist is created run the model trainer: 

```bash
python ModelTrainer.py
```

The output will be two graphs and a model (all placed in directory based on the start time). The output to the terminal is simply logging information on how long the model should take to train. One graph is the training and test loss, the other is the time taken for each Epoch. Additionally, a log and serialized plotting data file is created for use for future users. 