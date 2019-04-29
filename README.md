# DeepFE-PPI
DeepFE-PPI: An integration of deep learning with feature embedding for protein-protein interaction prediction
we have reported a novel predictor DeepFE-PPI，a protein-protein interaction prediction method that integrates deep learning and feature embedding.For a given protein-protein pair, we first use Res2Vec to repressnt all residues, and then we regard the feature vector as the input of the network and apply a branch of dense layers to automatically learn diverse hierarchical features. Finally, we use a neural network with four hidden layers to connect their outputs for PPIs prediction.

===============================

DeepFE-PPI uses the following dependencies:

•Python 3.5.2
• Numpy 1.14.1
• Gensim 3.4.0
• HDF5 and h5py 
• Pickle 
• Scikit-learn 0.19
• Tensorflow 1.2.0
• keras 1.2.0

===============================

Datasets

The main directory contains the directories of S.cerevisiae, Human and five species-specific protein-protein interaction datasets. In each directory, there are:

 * The S.cerevisiae core dataset: 5594 positive protein pairs and 5594 negative protein pairs. 
 * The human dataset: 3899 positive protein pairs and 4262 negative protein pairs.
 * Each of five species-specific protein interaction datasets (C. elegans, E. coli, H. sapiens, M. musculus, and H. pylori) contains 4013, 6954, 1412, 313 and 1420 positive protein pairs.

===============================

model

The model directory contains 5 subfolders, with a folder for word2vec models and four folders for deep learning models, and structured as follows: 
 * c1c2c3: This folder contains a model corresponds to Section " Park and Marcotte’s evaluation scheme".
 * dl: This folder have two subfolders: 
       ** 11188 folder contains the model that the 5-flod cross validation methods executed on the S.cerevisiae core dataset.
	   ** human folder contains the model that the 5-flod cross validation methods executed on the human dataset.
 * rewrite: This folder contains a model corresponds to 'redo_cv_code.py'.
 * train_11188_test_5_special: This folder contains a model corresponds to 'train_11188_test_5_special.py'.
 * word2vec: This folder contains all models when Parameter Selection. 
 
===============================

*.py

 * 5cv_11188.py: 5-flod cross validation methods on the S.cerevisiae core dataset.
 * 5cv_human.py: 5-flod cross validation methods on the human dataset.
 * c1c2c3_11188.py: The code corresponds to Section " Park and Marcotte’s evaluation scheme". We redo the special multiple cross-validation method proposed by Park & Marcotte [Park Y, Marcotte EM. Nat Methods. 2012; 9 (12)]
 * redo_cv_code.py: We rewrite the cross validation method without any libraries.
 * swiss_Res2vec_val_11188.py: Parameter Selection for residue reprsetation and deep learning.
 * train_11188_test_5_special.py: Codes that trains on the S.cerevisiae core dataset and tests on five species-specific protein interaction datasets.

===============================
Usage: 

Run these file from command line. 

For example:
>python train_11188_test_5_special.py
output:  
accuracy_test_Celeg = 100, 
accuracy_test_Ecoli = 100, 
accuracy_test_Hpylo = 100, 
accuracy_test_Hsapi = 100, 
accuracy_test_Mmusc = 100


===============================

Contact us:
Any questions about DeepFE-PPI, please email to dxqllp@163.com.


===============================

This dataset was used in the paper 'DeepFE-PPI: An integration of deep learning with feature embedding for protein-protein interaction prediction' for PPI prediction. For more details, please refer to the paper.

