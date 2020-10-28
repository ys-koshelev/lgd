# LGD
Repository implementing Learnable Gradient Descent method for image restoration via recurrent neural networks (RNNs). 
Project is mainly based on [this paper](https://arxiv.org/pdf/1706.04008.pdf). 

# Requirements
All requirements are stored in [requirements.txt](https://github.com/ys-koshelev/lgd/blob/master/requirements.txt). 
To install them, just run `pip intall requirements.txt`

# Pretrained models
All pretrained models can be downloaded 
[from here](https://drive.google.com/file/d/1Jpm9wMTtt4lP988eWTAepha21L0xSrbv/view?usp=sharing). 
You can use them to run inference in corresponding 
[Jupyter Notebooks](https://github.com/ys-koshelev/lgd/blob/master/jupyter) 

# Experiments description
For convenience all inference experiments are given as corresponding Jupyter Notebooks:
| Path to notebook                                                                                                          | Description                                                                                |
|---------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| [RNN_denoising.ipynb](https://github.com/ys-koshelev/lgd/blob/master/jupyter/RNN_denoising.ipynb)                         | Denoising, using learned gradient descent network                                          |
| [RNN_deblurring.ipynb](https://github.com/ys-koshelev/lgd/blob/master/jupyter/RNN_deblurring.ipynb)                       | Deblurring, using learned gradient descent network                                         |
| [RNN_super-resolution.ipynb](https://github.com/ys-koshelev/lgd/blob/master/jupyter/RNN_super-resolution.ipynb)           | Super-Resolution, using learned gradient descent network                                   |
| [TV_denoising_LBFGS.ipynb](https://github.com/ys-koshelev/lgd/blob/master/jupyter/TV_denoising_LBFGS.ipynb)               | Denoising, using total-variation restoration with L-BFGS minimizer                         |
| [TV_deblurring_LBFGS.ipynb](https://github.com/ys-koshelev/lgd/blob/master/jupyter/TV_deblurring_LBFGS.ipynb)             | Deblurring, using total-variation restoration with L-BFGS minimizer                        |
| [TV_super-resolution_LBFGS.ipynb](https://github.com/ys-koshelev/lgd/blob/master/jupyter/TV_super-resolution_LBFGS.ipynb) | Super-Resolution, using total-variation restoration with L-BFGS minimizer                  |
| [TV_segsynthesis_LBFGS.ipynb](https://github.com/ys-koshelev/lgd/blob/master/jupyter/TV_segsynthesis_LBFGS.ipynb)         | Semantic Synthesis, using total-variation restoration with L-BFGS minimizer (unsuccessful) |

# Datasets
[BSD500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) for training all linear problems

[BSD68](https://github.com/clausmichele/CBSD68-dataset/archive/master.zip) for testing all linear problems

[ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) for playing with semantic synthesis

# Additional information
This work is related to final project on 2020 Bayesian Methods of Machine Learning course at Skoltech. 
Initially I am trying to reproduce a restoration method, whcih is now commonly known as a learnable gradient descent. 
More detailed description is given in [this document](https://github.com/ys-koshelev/lgd/blob/master/description.pdf).

