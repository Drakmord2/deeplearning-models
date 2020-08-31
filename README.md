# DeepLearning Models
This repository is a collection of [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) models implemented with knowlegde from
DeepLearning.ai Specialization on [Coursera](https://www.coursera.org/specializations/deep-learning).

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

### Keras Convolutional Network
Implemented with Keras 2

Convolutional Layer:
- Zero Padding
- Batch Normalization
- ReLU Activation
- Max Pooling

Dense layer:
- Flatten
- Softmax

### TensorFlow Deep Q Network
Reinforcement Learning heavily based on [lufficc/dqn](https://github.com/lufficc/dqn).

Intended to be used as an agent on OpenAI Gym.


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

Global requirements are Python 3.6+ and [Virtualenv](https://virtualenv.pypa.io/en/latest/). 

Configure the system by executing the following commands on the project's root folder:

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

When inside the virtual environment, the code can be run using `python main.py`
