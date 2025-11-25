# SVM-BM for ncRNA Classification

This repository contains the full implementation of the SVM-BM (Support Vector Machine Boosting with Markov Resampling) algorithm developed for binary classification of ncRNA and non-ncRNA sequence pairs. The project includes data preprocessing, PCA-based analysis, model training, and evaluation against classical ensemble baselines.



## 📌 Project Overview

The goal of this project is to compare the performance of:

- **SVM-BM**
- **AdaBoost with SVM weak learners**
- **Random-subset SVM ensemble**

All methods are trained under identical subsampling constraints (very small subsets of size N = 10–) to evaluate robustness and generalization.


## 📊 Dataset

The original dataset is from:

> *“Detection of non-coding RNAs on the basis of predicted secondary structure formation free energy change”*  
> **BMC Bioinformatics**
https://www.csie.ntu.edu.tw/%7Ecjlin/libsvmtools/datasets/

The dataset includes paired RNA features (folding free energy, nucleotide frequencies, sequence lengths).  







