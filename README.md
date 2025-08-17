# DRMAHGC
Drug Repositioning by Multi-Aspect Heterogeneous Graph Contrastive Learning and Positive-Fusion Negative Sampling Strategy

# Requirements:
- python 3.9.13
- cudatoolkit 11.3.1
- pytorch 1.10.0
- dgl 0.9.0
- networkx 2.8.4
- numpy 1.23.1
- scikit-learn 0.24.2

# Data:
The data files needed to run the model.

# Code:
- data_preprocess.py: Methods of data processing
- metric.py: Metrics calculation
- pos_contrast.py: Get positive and negative samples for contrastive learning
- Contrast.py: Code of graph contrastive learning
- model.py: Model of DRMAHGC
- train_DDA.py: Train the model

# Usage:
Execute ```python train.py``` 


<img width="4383" height="2475" alt="overall architecture" src="https://github.com/user-attachments/assets/c9244cbf-6972-4445-a6b7-28990a9c453d" />
