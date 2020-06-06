# DeepLearning Models

## Algorithms
### Deep Neural Network
Implemented using Numpy vectorization. 

Xavier initialization for network weights.

Activation Functions:
- Hidden layers : Leaky ReLU
- Output layer: Sigmoid

Includes brackpropagation using Gradient Descent or Adam optimization.
The latter uses mini-batch.

### TensorFlow Deep Neural Network
Implemented with TensorFlow 1

Xavier initialization for network weights, Adam optimization and mini-batch.

## Models
The constructor of the models receive:
- `model ( layers_dimension, num_features, num_classes, learning_rate, num_iterations, beta1, beta2)`
  - **layers_dimension** is a list with the amount of units in each layer. e.g. \[784, 800, 300, 10\] is the dimension of a NN with a 1 input layer, 2 hidden layers and 1 output layer.
  - **beta1** and **beta2** are Adam parameters

The main public methods of the models are:

- `fit ( training_inputs, training_labels, optimizer )` - Trains the Neural Network and saves it's parameters in a .npy file
- `predict ( test_inputs )` - Classify new data
- `get_accuracy ( test_inputs, test_labels, type )` - Get accuracy of predictions made on a labeled dataset

## Datasets
The included utility functions can fetch datasets from OpenML. The default dataset is:
### MNIST 
The [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) is a large database of handwritten digits that contain 70,000 images. 

## Setup

Global requirements are Python 3.6+ and [Virtualenv](https://virtualenv.pypa.io/en/latest/)

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then the is available through `python main.py`
