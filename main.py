from PattRecClasses.GaussD import GaussD
from PattRecClasses.MarkovChain import MarkovChain
from PattRecClasses.HMM import HMM
from PattRecClasses.DiscreteD import DiscreteD
from gauss_logprob import gauss_logprob
import numpy as np

print(GaussD)

## TESTCASES

#One-dimensional Gaussian
# g = []
# nsemitones=12
# for i in range(nsemitones):
#   g.append(GaussD( means=np.array( [i] ) , stdevs=np.array( [1.0] ) ))


g1 = GaussD( means=np.array( [0] ) , stdevs=np.array( [1.0] ) )
g2 = GaussD( means=np.array( [3] ) , stdevs=np.array( [2.0] ) )
g = [g1, g2]
x_Seq = np.array([[-0.2, 2.6, 1.3]])
logP = gauss_logprob(g,x_Seq)
P = np.exp(logP)
for i in range(len(P[0])):
    P[:,i] =P[:,i] / P[:,i].max()
print(P)


#mc = MarkovChain( np.array( [[ 1, 0]] ), np.array( [ [ 0.9, 0.1], [ 0.1, 0.9 ] ] ) )
mc = MarkovChain( np.array([[1, 0]]), np.array([[0.9, 0.1, 0], [ 0, 0.9, 0.1 ] ] ) ) #concatenating [0,0,1]
h = HMM(mc,g)
# T = len(logP[0,:])
# N= len(logP) #num states
# print (T)
# print(N)

alpha_hat, norms = mc.forward(P)
print(alpha_hat)
print(norms)

logprob = h.logprob(P)
print(logprob)



