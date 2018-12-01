import database_loader
import Networks
import matplotlib.pyplot as plt
# read data:


# mini-batch size:
mini_batch_size = 10
new_nn = Networks.Network([784, 30, 10])
x_axis=[]
y_axis=[]
for x in range(1, 30):
    training_data, validation_data, test_data = database_loader.load_data()
    new_nn.training(training_data, x, mini_batch_size, .5)
    correct, wrong = new_nn.evaluate(test_data)
    print(correct, wrong)
    total = (correct *100) /(correct + wrong)
    x_axis.append(x)
    y_axis.append(total)
    print("--------------")
plt.plot(x_axis,y_axis, 'ro')
plt.show()