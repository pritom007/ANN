import numpy as np
import math
import random

def rand(a, b):
    return (b-a)*random.random() + a
def sig(x):
    return 1 / (1 + np.exp(-x))
def inference(x):
    """
     :param: x: input of ANN
     :return: the output of ANN with input x, a 1-D array
    """
    input = np.array(x)
    output = sig(np.dot(input, 2 * np.random.random((3, 1)) - 1))
    return output

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

def update(inputs):
    if len(inputs) != ni - 1:
        raise ValueError('wrong number of inputs')

    # input activations
    for i in range(ni - 1):
        # self.ai[i] = sigmoid(inputs[i])
        ai[i] = inputs[i]

    # hidden activations
    for j in range(nh):
        sum = 0.0
        for i in range(ni):
            sum = sum + ai[i] * wi[i][j]
        ah[j] = sig(sum)

    # output activations
    for k in range(no):
        sum = 0.0
        for j in range(nh):
            sum = sum + ah[j] * wo[j][k]
        ao[k] = sig(sum)

    return ao[:]
# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

def evaluate(data):
    """
    :param data: dataset, a list of tuples (x, y) representing the training inputs and the desired outputs.
    :return: the number of correct predictions of the current ANN on the input dataset.
    The prediction of the ANN is taken as the argmax of its output
   """
    correct, wrong = 0, 0
    for i in range(len(data[1])):
        res = inference(data[0][i])
        #res = np.argmax(res)
        print("res ", res)
        if np.round(res) == data[1][i]:  # check the data inputs var
            correct += 1
        else:
            wrong += 1

    return correct, wrong
data = {}
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [0, 0, 0],
              [1, 1, 0]])
print("X  ",X.shape[1])
Y = np.array([[1], [0], [1], [1], [1]])
data[0] = X
data[1] = Y
ni = 5 +1
nh = 6
no = 1
ai = [1.0] * ni
ah = [1.0] * nh
ao = [1.0] * no
# create weights
wi = makeMatrix(ni, nh)
wo = makeMatrix(nh, no)
# set them to random vaules
for i in range(ni):
    for j in range(nh):
        wi[i][j] = rand(-0.2, 0.2)
for j in range(nh):
    for k in range(no):
        wo[j][k] = rand(-2.0, 2.0)
print("b ",ai, " ao ",ao)
update(X)
print("f ", ai," ao ", ao)
res1, res2 = evaluate(data)
print(res1, res2)
sizes = [4, 1]
num_layers = len(sizes)
sizes = sizes
biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

print(biases)
print("----------")
print(weights)
