Tangent space-free Lorentz Spatial Temporal Graph Convolution Networks (LSTGCN)
======================================================

This repository includes the implementations of LSTGCN for traffic flow forecasing in PyTorch. 

To train a model, use, for example:

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



