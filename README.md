Tangent space-free Lorentz Spatial Temporal Graph Convolution Networks (LSTGCN)
======================================================

This repository includes the implementations of LSTGCN for traffic flow forecasing in PyTorch. 

To get the data: 
The **PEMS03**, **PEMS04**, **PEMS07** and **PEMS08** datasets can be downloaded from (https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw) with password: `p72z` and uncompress data file [STSGCN_data.tar.gz] using`tar -zxvf data.tar.gz` 

The **METR-LA**, **PEMS_BAY** and **PEMSD07_M** can be downloaded from the repository https://github.com/hazdzz/STGCN

To train LSTGCN , use, for example:

python train.py --dataset PEMS03 --task nr --dim 4 --models_dim [4] --lr 0.1 --num_of_vertices 358 --num_layers 3

python train.py --dataset PEMS04 --task nr --dim 4 --models_dim [4] --lr 0.1 --num_of_vertices 307 --num_layers 4


#### Directory: 
       data                     datasets files 
       layers                   includes a centroid-based classification and layers 
       log                      path to save logs  
       manifolds                includes the Lorentz manifold 
       model_save               path to save trained models  
       models                   encoder for graph embedding and decoder for post-processing  
       utils                    utility modules and functions  
       config.py                config file  
       train.py                 run this file to start the training  
       requirements.txt         requirements file  
       README.md                README file  



