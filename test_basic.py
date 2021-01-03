import numpy as np

# Batch for 10, 20, ..., 100
for i in range(10,101,10):
    print(i)

# Data for XOR
x_train = np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
#t_train = np.logical_xor(x_train[:,0],x_train[:,1])*1
t_train_result = np.logical_xor(x_train[:,0],x_train[:,1]).astype(int) # dtype err

t_train=np.zeros((4,2))
print(t_train.shape)

for i in range(t_train.shape[0]):
    t_train[(i, 0)] = 1-t_train_result[i]
    t_train[(i,1)] = t_train_result[i]


print(x_train)
print(t_train)