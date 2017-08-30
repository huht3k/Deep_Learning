# -*- coding: utf-8 -*-
"""
Created on Mon May 22 08:00:15 2017

@author: huht
"""
"""
refer: cs231n, lecture5
"""
import numpy as np
import matplotlib.pyplot as plt

## assume some unit gaussian 10-D input data
D = np.random.randn(1000, 500)
hidden_layer_sizes = [500] * 10
#nonlinearities = ['tanh'] * len(hidden_layer_sizes)
nonlinearities = ['relu'] * len(hidden_layer_sizes)

act = {'relu' : lambda x : np.maximum(0,x), 'tanh' : lambda x : np.tanh(x)}
Hs = {}

for i in xrange(len(hidden_layer_sizes)) :
    X = D if i == 0 else Hs[i-1]    ## input for layer0, 1, ...
    fan_in = X.shape[1]
    fan_out = hidden_layer_sizes[i]
    
    #W = np.random.randn(fan_in, fan_out) * 0.01
    #W = np.random.randn(fan_in, fan_out) * 1.0
    #W = np.random.randn(fan_in, fan_out) /np.sqrt(fan_in)
    W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in / 2.0)
    
    H = np.dot(X, W)
    H = act[nonlinearities[i]](H)
    Hs[i] = H
    
print 'input layer had mean %f and std %f' % (np.mean(D), np.std(D))
layer_means = [np.mean(H) for i, H in Hs.iteritems()]
layer_stds = [np.std(H) for i, H in Hs.iteritems()]
for i, H in Hs.iteritems():
    print 'hidden layer %d had mean %f and std %f' % ((i+1), layer_means[i], layer_stds[i])
    

# plot the means and standard deviations 
plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(Hs.keys(), layer_means, 'ob-')
plt.title('layer mean')
plt.subplot(1, 2, 2)
plt.plot(Hs.keys(), layer_stds, 'or-')
plt.title('layer std')



# plot the raw distributions
plt.figure(2)
for i, H in Hs.iteritems():
    plt.subplot(1, len(Hs), i+1)
    plt.hist(H.ravel(), 30, range=(-1, 1))
plt.show()


