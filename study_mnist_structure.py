import numpy as np
import random
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train,t_train), (x_test,t_test) = load_mnist(flatten=True,normalize=False)
#(x_train,t_train), (x_test,t_test) = load_mnist(flatten=False,normalize=False)
#(x_train,t_train), (x_test,t_test) = load_mnist(flatten=True,normalize=True)

print(x_train.shape) # (60000,784(28x28))
print(t_train.shape) # (60000,)
print(x_test.shape) # (10000,784(28x28))
print(t_test.shape) # (10000,)

count=x_train.shape[0]
print("Study Image Count :",count)

num=random.randint(0,count)
img=x_train[num]
label=t_train[num]
print("#{} label:{}".format(num,label))

img=img.reshape(28,28)
print("Image Data : ", img)

img_show(img)
