import matplotlib.pyplot as plt
import numpy as np
import cPickle as p
import numpy

"""
Error bar plots just in case
"""



f1 = p.load(open("seeds_conv_fft_final.pkl"))
print np.array(f1)[0].shape

print numpy.array(f1[0][0]).shape ,  numpy.array((f1[0][1])).shape
f12 = np.array([np.array(x[1]) for x in f1])
f1 = np.array([np.array(x[1]) for x in f1])
print len(f12)

f2 = p.load(open("seeds_conv_fft_feat.pkl"))
f2 = np.array([np.array(x[1]) for x in f2])
print f12.shape, f1.shape
test1 = np.mean(f1, axis=0)
# test2 = np.mean(f12, axis=0)
test2 = np.mean(f2, axis=0)
# print  np.std(f1, axis=0)[:,1]
var1 = np.std(f1, axis=0)[:,1]
var2 = np.std(f2, axis=0)[:,1]
print var1 > var2
fig2, ax = plt.subplots()


ax.plot(test1[:,1], 'b')
ax.errorbar(range(len(test1[:,1])),test1[:,1], yerr=var1 )
ax.plot(test2[:,1] , 'r')
ax.errorbar(range(len(test2[:,1])),test2[:,1], yerr=var2 )



plt.show()
