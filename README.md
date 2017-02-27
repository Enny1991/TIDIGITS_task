# Spoken DIGITS classification task using Deep Recurrent Neural Networks (RNNs).

This is a basic task to get to know RNNs and a modern framework to 
program Deep Learning in python.

## The Task
You are given a dataset which consist in spoken digits, i.e. different 
people saying the numbers from zero to 9, where there are two versions 
of the digit 0 namely 'zero' and 'O' (letter o). The dataset is called
[TIDIGITS](https://catalog.ldc.upenn.edu/ldc93s10), you are gonna use 
only a subset of this dataset since a part of it contains also 
'connected digits', that is people say a string of number connected.
The dataset is in 2 mat files: one is for the 
[train](https://www.dropbox.com/s/r68ct7mhlblrz26/tidigits_mfccs_train.mat?dl=0),
the other is for the [test](https://www.dropbox.com/s/x3mdibr27aiyzr5/tidigits_mfccs_test.mat?dl=0)
You can split the training set in a training and validation set if 
you would like to do some cross validation on your model.
The matlab files contain a structure with all the data you need (and more).

The features are already extracted. We are using [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
which is a temporal frequency representation of the audio samples.
These features are 2-D matrices of size (F, T) where F is 39 features and T is the temporal length of the samples.
The ones that you will be used are store in the 'mffc_third' cell array.
The labels (from 0 to 10) for the corresponding samples are stored in 'idx_labels'.

To load matlab structures in python you can use loadmat from scipy.io.
```python
from scipy.io import loadmat

data = loadmat('path/to/data/data_file.mat')['structure_name']
```

The data structure in python is a dictionary with nested set of arrays. 
So you'll have to sort it to pull out the digits and the labels
```python
samples = data['mffc_third'][0][0][0]  # <- dictionary access plus array access
labels = data['idx_labels'][0][0]
```

You'll have to load train and test separately.

---

## RNNs
Story of development of RNNs goes way back and there is a lot of material you could read.
I tried to put together some interesting/important papers to understand 
the theory behind them [HowToRNN](HowToRNN.md).

Don't have to understand everything perfectly but this will give you an overview.
The [Wikipedia page](https://en.wikipedia.org/wiki/Recurrent_neural_network) is also a nice start.

The main idea is that you have a network with multiple layers but every layer has
recurrent connection which means that the activation of the current time
depends both on the current input and on the previous time step activation of the network.

 You use these networks to analyze temporal sequences. In order to train them
 you use backpropagation through time [BPTT](https://en.wikipedia.org/wiki/Backpropagation_through_time),
 which is a modified version of backpropagation adapted for RNNs, where 
 you basically run the network and then unroll it in time such that you 
 have a sort of fully connect very deep network (basically a layer per time step) 
 and then you do normal backpropagation.

For sequence classification usually people do the following. You have an 
RNN which is fed with the sequence. Then you take the last hidden activation 
of the network and use this as features to feed to a fully connected layer
for the classification. When using neural networks for classification you 
should use an output layer with a number of units that matches the number 
of classes, use the [softmax activation](https://en.wikipedia.org/wiki/Softmax_function)
(a variant of logistic regression) and use [categorical cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) 
as a cost function. This will help the network learn a probability 
distribution over your output classes.

---

## Tensorflow
There are many frameworks for Deep Learning in python. We are gonna be 
using [Tensorflow](https://www.tensorflow.org/). Tensorflow is an API 
for symbolic computation. It basically means that instead of doing direct 
computation, you create a so called 'computational graph' that defines 
your operations, compile it and evaluate it. This allows you to avoid 
doing manually backpropagation since the gradients in your architecture 
are calculated by the framework and this makes life easier!
Refer to the documentation for installation and first tutorials.
I suggest you use a machine with Linux or MacOS. You could use Windows 
with Anaconda (a platform to run python) but it's a bit more complicated to set up.

You task is to use Tensorflow to train a RNN with LSTM or GRU units (see [HowToRNN](HowToRNN.md)) 
to solve the digit classification problem.
With the data you are given you should obtain around 97/98 % accuracy on the test set.

---
## TIPS

- Tensorflow might be a bit overwhelming in the beginning, but there are
 a lot of examples in the documentation and people have done a lot of work 
 with it so you can find examples on GitHub.

- The whole RNN framework is also quite wide, try to understand the basic 
concepts about RNN computation and please feel free to ask me questions about it.

- The network architecture does not have to be too complicated, this is a simple task.

- If you are not satisfied with the results you can always try some of the 
tricks that people use in the community:
  - Dropout or other kind of weight regularization
  - Change optimization method (SGD, RMS, AdaGrad, Adam, ...)
  - Data augmentation
  - ...
