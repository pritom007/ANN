import numpy as np
import random

class Network(object):

    def __init__(self, sizes):
        """
        :param: sizes: a list containing the number of neurons in the respective layers of the network.
                See project description.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def inference(self, x):
        """
        :param: x: input of ANN
        :return: the output of ANN with input x, a 1-D array
        """
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x)+b)
        return x
    def training(self, trainData, T, n, alpha, test_data=None):
        """
        trains the ANN with training dataset using stochastic gradient descent
        :param trainData: a list of tuples (x, y) representing the training inputs and the desired outputs.
        :param T: total number of iteration
        :param n: size of mini-batches
        :param alpha: learning rate
        """
        training_data = list(trainData)
        l = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(T):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + n]
                for k in range(0, l, n)]
            for mini_batch in mini_batches:
                self.updateWeights(mini_batch, alpha)
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test));
            else:
                print("Epoch {} complete".format(j))


    def updateWeights(self, batch, alpha):
        """
        called by 'training', update the weights and biases of the ANN
        :param batch: mini-batch, a list of pair (x, y)
        :param alpha: learning rate
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (alpha / len(batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (alpha / len(batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """
        called by 'updateWeights'
        :param: (x, y): a tuple of batch in 'updateWeights'
        :return: a tuple (nablaW, nablaB) representing the gradient of the empirical risk for an instance x, y
                nablaW and nablaB follow the same structure as self.weights and self.biases
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = dSquaredLoss(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, data):
        """
        :param data: dataset, a list of tuples (x, y) representing the training inputs and the desired outputs.
        :return: the number of correct predictions of the current ANN on the input dataset.
                The prediction of the ANN is taken as the argmax of its output
        """
        test_results = [(np.argmax(self.inference(x)), y)
                        for (x, y) in data]
        correct, wrong = 0, 0
        for x, y in test_results:
            if int(x == y):
                correct += 1
            else:
                wrong += 1
        return correct, wrong

    def evaluatePercent(self, data):
        correct , wrong = self.evaluate(data)
        total = correct+wrong
        print(correct,wrong)
        return (correct*100)/(total)

# activation functions together with their derivative functions:
def dSquaredLoss(a, y):
    """
    :param a: vector of activations output from the network
    :param y: the corresponding correct label
    :return: the vector of partial derivatives of the squared loss with respect to the output activations
    """
    return 2 * (a - y)

def sigmoid(z):
    """The sigmoid function"""
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))
