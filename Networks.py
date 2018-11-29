import numpy as np


class Network(object):

    def __init__(self, sizes):
        """
        :param: sizes: a list containing the number of neurons in the respective layers of the network.
                See project description.
        """
        # not needed
        self.num_layers = len(sizes)
        # for ex: [3,4,4,1]
        self.sizes = sizes
        # we are considering there are only the layers
        self.inputLayerWidth = sizes[0]
        self.inputLayerHeight = sizes[1]
        self.hiddenLayerSize = sizes[2]
        self.outputLayerSize =  sizes[3]
        # For 3 layered network there should be 2 set of biases and weights
        #self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # baises
        self.biases0 = np.random.rand(self.inputLayerHeight, self.hiddenLayerSize)
        self.biases1 = np.random.rand(self.inputLayerHeight, self.outputLayerSize)
        # weights
        self.W1 = np.random.rand(self.inputLayerWidth, self.hiddenLayerSize)
        self.W2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize)


    def inference(self, x):
        """
        :param: x: input of ANN
        :return: the output of ANN with input x, a 1-D array
        """
        input = np.asarray(x)
        output = self.forward(input)
        return output
    def training(self, trainData, T, n, alpha):
        """
        trains the ANN with training dataset using stochastic gradient descent
        :param trainData: a list of tuples (x, y) representing the training inputs and the desired outputs.
        :param T: total number of iteration
        :param n: size of mini-batches
        :param alpha: learning rate
        """
        for t in range(T):
            error = 0
            x = trainData[0]
            y = trainData[1]
            self.forward(x)
            self.backprop(x, y)
            for k in range(len(y)):
                error += 0.5 *(y[k] - self.ao[k])**2
           # if t % 1000 == 0:
               # print('error %-.5f' % error)


    def updateWeights(self, batch, alpha):
        """
        called by 'training', update the weights and biases of the ANN
        :param batch: mini-batch, a list of pair (x, y)
        :param alpha: learning rate
        """

    def forward(self, X):
        # forward propagation through our network
        self.ai = X
        self.z = np.dot(self.ai, self.W1) + self.biases0  # dot product of X (input) and first set of 3x2 weights
        self.ah = sigmoid(self.z)  # activation function

        self.z3 = np.dot(self.ah,
                         self.W2) + self.biases1  # dot product of hidden layer (z2) and second set of 3x1 weights
        self.ao = sigmoid(self.z3)  # final activation function

        return self.ao

    def backprop(self, x, y):
        """
        called by 'updateWeights'
        :param: (x, y): a tuple of batch in 'updateWeights'
        :return: a tuple (nablaW, nablaB) representing the gradient of the empirical risk for an instance x, y
                nablaW and nablaB follow the same structure as self.weights and self.biases
        """
        self.ao_error = y - self.ao
        self.ao_delta = self.ao_error * sigmoid_prime(self.ao)

        self.ah_error = self.ao_error.dot(self.W2.T)
        self.ah_delta = self.ah_error * sigmoid_prime(self.ah)

        ah_change = self.ao_delta.T.dot(self.ah)
        alphasize = np.ones(shape=(3,1)) * 0.5 #self.aplha

        self.W1 += x.T.dot(self.ah_delta) + alphasize.dot(ah_change)
        ao_change = self.ah_delta.T.dot(self.ai)
        self.W2 += self.ah.T.dot(self.ao_delta) + ao_change.dot(alphasize)

    def evaluate(self, data):
        """
        :param data: dataset, a list of tuples (x, y) representing the training inputs and the desired outputs.
        :return: the number of correct predictions of the current ANN on the input dataset.
                The prediction of the ANN is taken as the argmax of its output
        """
        correct , wrong = 0, 0
        for i in range(len(data[1])):
            res = self.inference(data[0][i])
            res_max = argmax(res)[0]
            if res_max >= 0.5:
                res_max = 1
            else:
                res_max = 0
            if res_max == data[1][i]:
                correct +=1
            else:
                wrong += 1
        return correct

    def evaluatePercent(self, data):
        return self.evaluate(data)/len(data[1]) *100
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
def argmax(array):
  max = 0
  for x in array:
    if x > max:
        max = x
  return max
nn = Network([3, 6, 4, 1])
data = {}
X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 0],
              [1, 0, 0],
              [1, 0, 1]])
print("X  ",X.shape[1])
Y = np.array([[0], [0], [0], [0], [0],[1]])
data[0] = X
data[1] = Y

ava = 0
for i in range(5):
    nn.training(data, 10000, 10, 0.5)
    ava += nn.evaluatePercent(data)
print("Percentage of correct prediction on an average: ", ava/5)
newData = np.asarray(([0,1,1]),dtype=float)
out = [x for x in nn.inference(newData)]
print("new predict ", argmax(out))