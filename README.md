# Semi-Supervised Deep Learning for Molecular Structures using Stochastic Optical Reconstruction Microscopy (STORM)

By [Alvin Wan](http://alvinwan.com) and [Allen Guo](http://aguo.us)

During clathrin-mediated endocytosis (CME), clathrin surrounds molecules awaiting transport, forming a spherical coat. Our goal was to pick out clathrin undergoing this process. This repository employs semi-supervised learning methods to classify clathrin "cup-like" structures given microscopies for proteins of interest. See the problem formulation and approach specifics in our report [Semi-Supervised Deep Learning for Molecular Structures](https://github.com/alvinwan/storm/blob/master/storm.pdf).

The clathrin data was provided by the [Ke Xu lab](http://www.cchem.berkeley.edu/xuklab/) in UC Berkeley's College of Chemistry, whose research work we are supporting. If you find this work useful for your research, please consider citing:

```
@citation{storm,
    Author = {Alvin Wan and Allen Guo},
    Title = {Semi-Supervised Deep Learning for Molecular Structures},
    Year = {2017}
}
```

# Install

This project requires Python3. We begin by navigating to the root of the repository, which we will call `$STORM`.

    cd $STORM
    
(optional) We recommend setting up a virtual environment first. This project uses Python3.

    virtualenv ../env --python=python3
    source ../env/bin/activate

Install all Python requirements.

    pip install -r requirements.txt

# Train

Alternatively, you can toy with various hyperparameters and attempt training on your own. We approached the problem using a two-step pipeline. First, find a latent representation in a lower-dimensional space. Then, run a simple classifier on the encoded data.

> If your data is located at `data/train_molecules.mat` and `data/test_molecules.mat`, the `<data_class>` mentioned below would be `molecules`.

## Featurize

Start by picking a featurization technique.

    cd $STORM
    bash storm.sh encode_(ae|kmeans|pca) <data_class>
    
## Classify

We then train a support vector machine (SVM) using the featurizations. For the below command, make sure to featurize both the `train.mat` and `test.mat` datasets, specified above.

    bash storm.sh svm <data_class>
