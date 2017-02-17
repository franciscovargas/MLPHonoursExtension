from scipy.signal import correlate
from mlp.convlin import myconv_tense
from time import time
import numpy as np

if __name__ == "__main__":
    scipy = list()
    me = list()
    for i in range(100,9000, 10):
        print i, "iteration"
        I = np.ones((100,i))
        k = np.ones(5)
        k2 = k.reshape(1,-1)
        t1 = time()
        correlate(I,k2)
        t2 = time()
        scipy.append(t2 - t1)
        t1 = time()
        myconv_tense(I,k)
        t2 = time()
        me.append(t2 - t1)
