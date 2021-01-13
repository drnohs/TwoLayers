import numpy as np
import matplotlib.pylab as plt
from common.functions import *

# Graphs for step_function, relu, sigmoid

x=np.arange(-5.0,5.0,0.1)
y1=step_function(x)
y2=relu(x)
y3=sigmoid(x)

plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)

plt.show()