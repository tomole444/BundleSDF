import numpy as np
import matplotlib.pyplot as plt

import time

if __name__ == "__main__":
    
    y = np.loadtxt("/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/ownBuchPoseShift/movement/2340.txt")
    x = range(0,len(y))
    #plt.hist(a)
    plt.plot(x,y)
    plt.show()
