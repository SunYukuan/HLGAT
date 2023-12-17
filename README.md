# HLGAT
Open-source code for "High-frequency and Low-frequency Dual-channel Graph Attention Network"
# Main Requirements
pandas==1.4.4  
Pillow==9.2.0  
PyYAML==6.0  
torch=2.0.1  
scipy==1.9.1  
dgl=1.1.2+cu117  
# Description
+ train.py
  + main() -- Train a new model for node classification task on the Cora, Citeseer, Pubmed, Texas, Cornell, Film, Chameleon, Squirrel
+ model.py
  + HLGAT() -- The Heterogeneous Label-aware Graph Attention Network (HLGAT) model is designed for node classification tasks.
    It uses graph attention mechanisms for effective feature aggregation.
+ utils.py
  + accuracy() --The accuracy function calculates the classification accuracy given model logits and true labels.
  + preprocess_data() -- Load data of selected dataset
# Running the code
Training example: 

python train.py --dataset cornell
# Citation
The manuscript is under review for Pattern Recognition.
