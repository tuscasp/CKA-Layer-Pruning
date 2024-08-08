# Effective-Layer-Pruning

## Env Setup

``` bash
conda create --name tf215
conda activate tf215
conda install python==3.9
pip install tensorflow==2.15.1
pip install scikit-learn
```

## Experiments instruction

This repository provides code examples of our CKA criterion for layer pruning, including some of our pruned models. <br />

To observe and understand the functionality of our method, we simplify many training/fine-tuning parameters. If you are interested in reproducing our results, please follow the steps below: <br />
1 - Put debug=false <br />
2 - Increase the number of epochs in the fine-tuning function to 200. <br />
3 - Divide the learning rate by 10 at epochs 100 and 150. <br />
4 - Use data augmentation (details in the paper)
