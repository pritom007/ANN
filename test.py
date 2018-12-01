import database_loader
import experiment
import Networks
import matplotlib.pyplot as plt


# mini-batch size:
mini_batch_size = 10
# read data:
training_data, validation_data, test_data = database_loader.load_data()

# Test for the experiment network
exp_nn = experiment.Network([784, 30, 10], cost=experiment.CrossEntropyCost)

''''
exp_nn.weight_initializer()
exp_nn.training(training_data, 4, 10, 0.5, evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True)
'''


# Test for the given network
new_nn = Networks.Network([784, 30, 10])
new_nn.training(training_data, 12, mini_batch_size, .5)

'''
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
'''