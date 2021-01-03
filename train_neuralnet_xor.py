# XOR
import numpy as np
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

# Dataset : XOR, no test set
x_train = np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
t_train_result = np.logical_xor(x_train[:,0],x_train[:,1]).astype(int)

# convert to one hot labeled result
t_train=np.zeros((4,2))
print(t_train.shape)

for i in range(t_train.shape[0]):
    t_train[(i, 0)] = 1-t_train_result[i]
    t_train[(i,1)] = t_train_result[i]

x_test = x_train # Train & Test set is equal
t_test = t_train

network = TwoLayerNet(input_size=2, hidden_size=3, output_size=2)

iters_num = 50000
cnt=0
train_size = x_train.shape[0]
learning_rate = 0.1

batch_size=4 # train & batch is equal
print("Train Size:",train_size)

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = train_size/batch_size

for i in range(iters_num):
    # No batch
    grad = network.numerical_gradient(x_train, t_train)
    #grad = network.gradient(x_batch, t_batch)
    
    # Learning
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_train, t_train)

    
    if i % (iters_num/100) == 0:
        cnt+=1
        print("#",cnt, end=" ")

        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(loss)

        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# Graph
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_loss_list))
#plt.plot(x, train_acc_list, label='train acc')
#plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.plot(x, train_loss_list, label='train loss')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()