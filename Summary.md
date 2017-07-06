## Data
Not sure why the scipy.io does not work properly, but I found a way to cheese through.
#### Features
The data are loaded and whitened. I.e. the means of each feature over all the data are substracted from the corresponding feature and I divide the feature with the corresponding standard deviation. The features are padded on the n_step axis in the begining.
#### Lables
Just converted the lables to one-hot vector.

## Network Organization
- 3 layers recurrent neural network with GRU units.
- Each layer's dropout rate is 0.25.
- Each layers is composed of 120 hidden unit.
- Softmax is the final layer.
- Used cross entropy as the cost function.

## Training
- Batch size is 128
- Learning rate is 1e-4
- AdamOptimizer is used

## Results
Reference-style: 
![alt text][logo]

[logo]: ./figures/Figure1
## Something still bothers me
- Why the sequence length parameter does not work?
- Why padding to the end does not work?