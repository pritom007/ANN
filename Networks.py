import numpy as np


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

    def training(self, trainData, T, n, alpha):
        """
        trains the ANN with training dataset using stochastic gradient descent
        :param trainData: a list of tuples (x, y) representing the training inputs and the desired outputs.
        :param T: total number of iteration
        :param n: size of mini-batches
        :param alpha: learning rate
        """
        for i in range(T):
            error = 0
            for td in trainData:
                x = td[0]
                y = td[1]
                self.updateWeights(x)
                error = error + self.backprop(y, alpha, n)  # change

            if i % 100 == 0:
                print("error %-.5f" % error)

    def updateWeights(self, batch, alpha):
        """
        called by 'training', update the weights and biases of the ANN
        :param batch: mini-batch, a list of pair (x, y)
        :param alpha: learning rate
        """

    def backprop(self, x, y):
        """
        called by 'updateWeights'
        :param: (x, y): a tuple of batch in 'updateWeights'
        :return: a tuple (nablaW, nablaB) representing the gradient of the empirical risk for an instance x, y
                nablaW and nablaB follow the same structure as self.weights and self.biases
        """

    def evaluate(self, data):
        """
        :param data: dataset, a list of tuples (x, y) representing the training inputs and the desired outputs.
        :return: the number of correct predictions of the current ANN on the input dataset.
                The prediction of the ANN is taken as the argmax of its output
        """
        correct, wrong = 0, 0
        for i in range(len(data)):
            res = self.inference(data[0])
            res_max = res.argmax()
            if res_max == data.labels[i]:  # check the data inputs var
                correct += 1
            else:
                wrong += 1
        return correct, wrong


# activation functions together with their derivative functions:
def dSquaredLoss(a, y):
    """
    :param a: vector of activations output from the network
    :param y: the corresponding correct label
    :return: the vector of partial derivatives of the squared loss with respect to the output activations
    """


def sigmoid(z):
    """The sigmoid function"""
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))
