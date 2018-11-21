import numpy as np


# Sigmod function
# Keeps the data value between 0 to 1
def sigmoid(x, deriv= False):
    if (deriv == True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

x = np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1],
             [0,0,0]])

y = np.array([[1], [1], [1], [1], [0]])

# seed good for debugging

np.random.seed(1)

# synapses

syn0 = 2*np.random.random((3, 4)) - 1
syn1 = 2*np.random.random((4, 1)) - 1

# training
for j in range(60000):
    #layers
    l0 = x
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    # backward propagation
    l2_error = y -l2 # arbitrary at first

    if(j % 10000) == 0:
        print("Error: ", str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * sigmoid(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * sigmoid(l1, deriv=True)

    # update synapses/ weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
print("Output after training ")
print(l2)


