import matplotlib.pyplot as plt
from PattRecClasses.GaussD import GaussD
from PattRecClasses.MarkovChain import MarkovChain
from PattRecClasses.HMM import HMM
from PattRecClasses.DiscreteD import DiscreteD
from gauss_logprob import gauss_logprob
import numpy as np
# Example: Define and use a simple infinite-duration HMM

# State generator
#mc = MarkovChain( np.array( [[ 0.5, 0.5 ]] ), np.array( [ [ 0.9, 0.1 ], [ 0.05, 0.95 ] ] )) #Infinite
#mc = MarkovChain( np.array( [[ 0.75, 0.25]] ), np.array( [ [ 0.8, 0.1, 0.1], [ 0.05, 0.85, 0.1] ] )) #Finite
mc = MarkovChain( np.array([[1, 0]]), np.array([[0.9, 0.1, 0], [ 0.1, 0.8, 0.1 ] ] ) )

g1 = GaussD( means=[0], stdevs=[1] )   # Distribution for state = 1
g2 = GaussD( means=[0], stdevs=[3] )   # Distribution for state = 2
h  = HMM( mc, [g1, g2])                # The HMM

# Generate an output sequence
# x_len_list = np.zeros([1, 10])
# for i in range(len(x_len_list[0])):
#   x,s = h.rand(20)
#   x_len_list[0,i] = len(x)

x,s = h.rand(500)

print(x)
print(s)

# # plt.subplot(2,1,1)
# plt.plot(x)
# # plt.xlabel('t')
# # plt.ylabel('X_t')
# # plt.title('behaviour of HMM with 500 samples')
# # plt.ylim((-5, 5))
#
#
# print(np.shape(s))
# # plt.subplot(2,1,2)
# # plt.plot(s)
# # plt.xlabel('t')
# # plt.ylabel('S_t')
# # plt.title('behaviour of HMM with 500 samples')