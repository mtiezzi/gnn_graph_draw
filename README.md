# **Graph Neural Networks for Graph Drawing**

This repository contains the implementation for the paper [**Graph Neural Networks for Graph
Drawing**](https://ieeexplore.ieee.org/document/9810169), accepted for publication in the IEEE Transactions on Neural
Networks and Learning Systems.
[Technical Report](https://arxiv.org/abs/2109.10061)

*Authors:*  [Matteo Tiezzi](https://mtiezzi.github.io/), Gabriele Ciravegna and Marco Gori.

Make sure to have Python dependencies by running:

```
pip install -r requirements.txt
```

We tested the code with PyTorch 1.10 and Deep Graph Library (DGL). Follow the [instructions](https://pytorch.org/get-started/) on the official
websites for further details.




REPOSITORY DESCRIPTION
----------------------

The folder structure is the following:

    data :                                   folder with dataset and utilities to generate datasets
    viz_utils:                               folder containing utilities/loss functions                
    crossing_dataset_creator.py :            utility to create the dataset for training the Neural Aesthete on edge-crossing
    crossing_learning_mlp.py :               script to train the Neural Aesthete
    crossing_test_algorithm.py :             utilities for the Neural Aesthete
    gd_stress.py :                           utilities to Draw Graphs using Graph Drawing force-directed packages ("neato", "pivotmds", "forceatlas2") 
    graph_draw_main.py :                     script to draw graphs by iteratively optimizing a loss function with SGD (standard GD approaches). 
    graph_neural_drawers.py :                script to train GNNs for Graph Drawing (datasets from the paper)
    inference_gnn_stress_huge.py :           script to draw huge graphs with a pretrained Graph Neural Drawer
    random_graph_factory.py :                script to generate the Sparse dataset from the paper
    split_rome_dataset.py :                  script to create train/val/test split (Rome dataset)

DATASETS
===========

The `Rome` dataset can be found at  [this link](http://www.graphdrawing.org/download/rome-graphml.tgz) and unpacked into
the `data` folder. 

The `Sparse` dataset can be generated with the `random_graph_factory.py` script. 

To obtain train/val/test splits, please exploit the `split_rome_dataset.py` script (similar code can be used to split the other
datasets).


HOW TO TRAIN THE NEURAL AESTHETE
========================
Launch the `crossing_dataset_creator.py` script to create the synthetic dataset to train the Aesthete. 
Then, launch the `crossing_learning_mlp`  script to train a Neural Aesthete. 
The model will be saved in the `saved_models` folder.


Standard Graph Drawing with Neural Aesthete
=====================================================================

Use the `graph_draw_main.py` script to draw graphs (please refer to the Arguments and the paper for further details)
using standard SGD and the Neural Aesthete.

Graph Neural Networks for Graph Drawing
=================================================================

Use the `graph_neural_drawers.py` script to draw graphs (please refer to the Arguments and the paper for further
details ) using GNNs and the Neural Aesthete.


Draw graphs with Graph Drawing packages
==============================================

Please install the followig dependendencies before running the `gd_packages.py` script:

    pip install networkx graphviz  pydot
    pip install networkit 
    # for windows 
    winget install graphviz
  
    # install ForceAtlas2  
    git clone https://github.com/cvanelteren/forceatlas2.git  
    python setup.py install







