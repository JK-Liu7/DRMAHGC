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
The data files needed to run the model, which contain B-dataset, C-dataset and F-dataset.
- DrugFingerprint, DrugGIP: The similarity measurements of drugs to construct the similarity network
- DiseasePS, DiseaseGIP: The similarity measurements of diseases to construct the similarity network
- Protein_sequence, ProteinGIP_Drug, ProteinGIP_Disease: The similarity measurements of proteins to construct the similarity network
- DrugDiseaseAssociationNumber: The known drug disease associations
- DrugProteinAssociationNumber: The known drug protein associations
- ProteinDiseaseAssociationNumber: The known disease protein associations

# SGMAE:
- Implementation of SGMAE
- Embedding/: The high-order feature embeddings of drugs and diseases obtained by SGMAE
- Usage: Execute ```python train_GAE.py``` 

# Code:
- data_preprocess.py: Methods of data processing
- metric.py: Metrics calculation
- pos_contrast.py: Get positive and negative samples for contrastive learning
- Contrast.py: Code of graph contrastive learning
- model.py: Model of DRMAHGC
- train_DDA.py: Train the model

# Usage:
Execute ```python train_DDA.py``` 
