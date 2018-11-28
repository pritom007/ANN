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

        # The first element of sizes is the input layer.
        # The last element of the sizes is the output layer
        # Other than input and output layer, rest is hidden layer
        self.ni = sizes[0] # 1 for bayes
        self.nh = sizes[1]
        self.no = sizes[2]
        self.input_layer = [1.0] * self.ni +1
        self.output_layer = [1.0] * self.no
        self.hidden_layers = [1.0] * self.nh
        print(self.input_layer, self.hidden_layers)
        # dynamic case
        # self.output_layer = np.asarray(sizes[self.num_layers - 1])
        # self.hidden_layers =[range(self.num_layers-2)]

        # for h in self.hidden_layers:
        #    self.hidden_layers[h] = np.asarray()


    def inference(self, x):
        """
        :param: x: input of ANN
        :return: the output of ANN with input x, a 1-D array
        """
        input = np.array(x)
        output = self.sigmoid(np.dot(input, self.weights))
        return output

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
            res_max = np.argmax(res)
            if res_max == data[1]:  # check the data inputs var
                correct += 1
            else:
                wrong += 1
        return correct, wrong

    def activation(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation


# activation functions together with their derivative functions:
def dSquaredLoss(a, y):
    """
    :param a: vector of activations output from the network
    :param y: the corresponding correct label
    :return: the vector of partial derivatives of the squared loss with respect to the output activations
    """
    return 2 * (a-y)

def sigmoid(z):
    """The sigmoid function"""
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))
nn = Network([4,4,1])