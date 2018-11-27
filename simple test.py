import numpy as np
import math

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

def evaluate(data):
    """
    :param data: dataset, a list of tuples (x, y) representing the training inputs and the desired outputs.
    :return: the number of correct predictions of the current ANN on the input dataset.
    The prediction of the ANN is taken as the argmax of its output
   """
    correct, wrong = 0, 0
    for i in range(len(data[1])):
        res = inference(data[0][i])
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

Y = np.array([[1], [1], [1], [0], [1]])
data[0] = X
data[1] = Y

res1, res2 = evaluate(data)
print(res1, res2)

