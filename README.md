# Reinforcement-Learning-for-Neural-Network-Pruning

Implimentation of 5 multi-armed bandit algorithms to prune a neural network. The method can be chosen from UCB1 (upper confidence bound), KL_UCB (Kullback-Leibler UCB),
TS_Beta (Thompson Sampling with Bernoulli prior), TS_Normal (Thompson Sampling with Gaussian prior) and Bayes_UCB (Upper cofidence using bayes apesteriori for Bernoulli priors)

For this test a Convolutional Neural Network has been used, which is trained for classification of human activity recognition based on FMCW radar-signals. This can be replaced by any dataset for any task.

## Parameter

**method**: chosing between one of the methods above <br/>
**horizon**: default is on 7000<br/>
**model_path**: if None a network gets trained on the given dataset

## Dependencies

  - python=3.7.6
  - dask=2.11.0
  - dask-jobqueue=0.7.0
  - distributed=2.11.0
  - jupyterlab=1.2.6
  - matplotlib=3.1.3
  - numpy=1.18.1
  - pandas=1.0.1
  - python=3.7.6
  - tensorflow-gpu=2.1.0
