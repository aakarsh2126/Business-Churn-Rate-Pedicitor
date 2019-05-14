# Business-Churn-Rate-Pedicitor
Artificial Neural Network Model to Predict whether the customer will leave the bank or not.
## Libraries used
1. keras(tensorflow as backend)
2. pandas
3. numpy
4. sklearn
5. matplotlib
## Artificial Neural Network
The Artifial Neural Network is one of the methods of supervised deep learning where we try to mimic the working of actual human brain cell.We try to find and analyse the output from previous neuron and forward it to next neuron.
## Description
This Neural Network model consists of 11 nodes in input layer,2 hidden layers each with 6 nodes and 1 output layer with 1 node.
The model is trained with 8000 data and is tested in 2000 test data with 83 percent accuracy.
## Structure of ANN
A typical neural network consists of 3 layers- the input layer,the hidden layer and the output layer.
### Input Layer
This layer consists the first layer of neural network.The number of nodes(neurons) in this layer depends on the size of feature vector.
### Hidden Layer
This layer does the most working of the neural network.It may have one to many layers with each layer having any number of neurons increasing the complexity of the network.The more the complex network is the better is the feature extraction and pattern recognition.
### ht=sigma(wt*xt)
ht=output of the current node
wt=weight matrix for taking input from one node to other
xt=current input vector
sigma=sigmoid activation function.
Having too much layers in hidden layer or too much number of neurons can lead to overfitting problem(good accuracy on train data but bad accuracy on test data).
To avoid overfitting we add dropout regularization(i.e. ignoring some perecetage of neurons).
### Output layer
This layer is the last layer of the neural network.Generally it consists of one node for regression type problem or multiple node for classification type of problem.
## Forward Propagation
The forward propagation algorithm is passing of data from input layer to output layer.
### ht=sigma(wt*xt)
ht=output of the current node
wt=weight matrix for taking input from one node to other
xt=current input vector
sigma=sigmoid activation function.
