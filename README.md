# Overview
This project focuses on compacting the [Gaussian Head Avatar](https://yuelangx.github.io/gaussianheadavatar/) model using methods in [Compact3D](https://ucdvision.github.io/compact3d/) to significantly reduce memory usage while preserving rendering quality. The original model requires storing a large number of
3D Gaussians, which takes up most of the storage. By quantizing key attributes of rotation, scale, and features, this project achieve
a 8x reduction in storage with extreme minimal impact
on performance. 


# Environment Setting
git clone https://github.com/YuelangX/Gaussian-Head-Avatar.git --recursive
The environment setting is as describe in repo of Gaussian Head Avatar

# Dataset
Mini-dataset used in this project is removed based on the regulation of NeRSemble dataset.
However, it can still be found by the above repo.

# Methods
1. Mesh Head Initialization
python train_meshhead.py --config config/train_meshhead_N031.yaml

2. Compacted Gaussian Head Avatar
python train_gaussianhead.py --config config/train_gaussianhead_N031.yaml

3. Reenactment task
python reenactment.py --config config/reenactment_N031.yaml

# Pre-trained Models
Pre-trained Models is stored in checkpoints, which included:
1. Mesh Model
2. Compacted model with codebook size 2000, training epoch=200
For using the pre-trained model, rename the model into the format showing in #Methods and run the command

# Code
Baseline code is from Gaussian Head Avatar, as mentioned above.
Compacted code is adapted from kmeans_quantize.py of Compact3D, and stored as lib/compact3d/compact3d.py
Source code: https://github.com/UCDvision/compact3d.git

The default setting of the models is Compacted Gaussian Head Avatar with compact starting from epoch=100.

# Parameter settings
Training Epoch:
change train funciont of the main function in /train_gaussianhead.py

Adjust the compacting epoch:
1. change epoch's setting of state_dict function in /lib/module/GaussianHeadModule.py
2. change compact_ite of generate function in /lib/trainer/GaussianHeadTrainer.py

Adjust K-means:
The initialzation function in GaussianHeadModule



