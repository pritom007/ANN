import numpy as np


class NN():
    def __init__(self):
        np.random.seed(1)

        self.weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self , x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return  x * (1 - x)

    def train(self, data, training_iterations):

        for i in range(training_iterations):

            output = self.think(data[0])
            error = data[1] - output
            adjustment = np.dot(np.array(data[0]).T, error * self.sigmoid_deriv(output))
            self.weights += adjustment

    def think(self, data):

        input = np.array(data[0]).astype(float)
        output = self.sigmoid(np.dot(input, self.weights))

        return output

if __name__ == "__main__":

    nn = NN()
    data = {'x':[], 'y':[]}
    print("Random weight: ")
    print(nn.weights)
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [0, 0, 0],
                  [1, 1, 0]])

    Y = np.array([[1], [1], [1], [0], [1]])
    data[0] = X
    data[1]= Y
    nn.train(data,10000)
    print("weights after training: ")
    print(nn.weights)

    A = str(input("Input 1:"))
    B = str(input("Input 2:"))
    C = str(input("Input 3:"))

    print("New Situation:  input data - ",A, B , C)
    print("Output data: ")
    data[0] = [A, B, C]
    print(nn.think(data))
