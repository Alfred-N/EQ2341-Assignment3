from PattRecClasses.GaussD import GaussD
from PattRecClasses.MarkovChain import MarkovChain
from PattRecClasses.HMM import HMM
from PattRecClasses.DiscreteD import DiscreteD
from gauss_logprob import gauss_logprob
from GetMusicFeatures import GetMusicFeatures
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

P = np.e**(logP-np.max(logP, axis=0))# normalized
# P = np.e**logP #non normalized

#mc = MarkovChain( np.array( [[ 1, 0]] ), np.array( [ [ 0.9, 0.1], [ 0.1, 0.9 ] ] ) ) #infinite
mc = MarkovChain( np.array([[1, 0]]), np.array([[0.9, 0.1, 0], [ 0, 0.9, 0.1 ] ] ) ) #finite
h = HMM(mc,g)

alpha_hat, norms = mc.forward(P)
print(norms)
logprob = h.logprob(P) # = -4.8527641993012445 for normalized P, = -9.187726979475208 for non normalized P
print(logprob)
beta_hat = mc.backward(x_Seq, P, norms)
print(beta_hat)


