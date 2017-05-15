# Classification of Clathrin Structures using Stochastic Optical Reconstruction Microscopy (STORM)

This repository employs unsupervised learning methods to classify clathrin
"cup-like" structures given microscopies for proteins of interest.

# Install

We begin by navigating to the root of the repository, which we will call `$STORM`.

    cd $STORM
    
(optional) We recommend setting up a virtual environment first. This project uses Python3.

    virtualenv ../env --python=python3
    source ../env/bin/activate

Install all Python requirements.

    pip install -r requirements.txt

# Classify

This repository includes saved models for the PCA with SVM models. To classify all samples in a provided matlab file, run

    bash storm.sh classify <path_to_matlab_file>

# Train

Alternatively, you can toy with various hyperparameters and attempt training on your own.

## Featurize

Start by picking a featurization technique.

    cd $STORM
    bash storm.sh encode_(ae|kmeans) <data_class>
    
## Classify

We then train a support vector machine (SVM) using the featurizations. For the below command, make sure to featurize both the `train.mat` and `test.mat` datasets, specified above.

    bash storm.sh svm <data_class>
