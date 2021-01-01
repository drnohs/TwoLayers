# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net_quiz import TwoLayerNet
import time

# Dataset
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#iters_num = 10000
iters_num=1

train_size = x_train.shape[0]

#batch_size = 100

learning_rate = 0.1

print("Train Size:",train_size)

#train_loss_list = []
#train_acc_list = []
#test_acc_list = []
batch_list = []
time_list = []

#iter_per_epoch = max(train_size / batch_size, 1)

#for i in range(iters_num):
for batch_size in range(10,101,10):
    print("Batch Size:", batch_size)
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    sTime=time.time()

    grad, cnt = network.numerical_gradient(x_batch, t_batch)
    #grad = network.gradient(x_batch, t_batch)

    eTime=time.time()
    tTime=(eTime-sTime) # sec, 1970.1.1 ~ now

    print("Numerical Differentiation Count : ",cnt) # 39710
    print("Elapsed time : ",tTime)

    # Learning
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    #train_loss_list.append(loss)
    
    #if i % iter_per_epoch == 0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    #train_acc_list.append(train_acc)
    #test_acc_list.append(test_acc)
    print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    batch_list.append(batch_size)
    time_list.append(tTime)


# Graphs
"""markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')"""

plt.plot(batch_list,time_list)
plt.xlabel("batch size")
plt.ylabel("time(sec)")
plt.ylim(0, 40)
plt.show()