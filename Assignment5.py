import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from PattRecClasses.GaussD import GaussD
from PattRecClasses.MarkovChain import MarkovChain
from PattRecClasses.HMM import HMM
from PattRecClasses.DiscreteD import DiscreteD
from gauss_logprob import gauss_logprob
from GetMusicFeatures import GetMusicFeatures
from GetSemitones import GetSemitones
import numpy as np


fileName = "melody_1"
sr, mel1 = wavfile.read("Songs/"+fileName + ".wav")
_, mel2 = wavfile.read("Songs/"+fileName + ".wav")
_, mel3 = wavfile.read("Songs/"+fileName + ".wav")

frIsequence1, pprd1 = GetMusicFeatures(mel1, sr, winlength=0.03)
frIsequence2, pprd2 = GetMusicFeatures(mel2, sr, winlength=0.03)
frIsequence3, pprd3 = GetMusicFeatures(mel3, sr, winlength=0.03)

semi1 = GetSemitones(frIsequence1)
sample_mean1 = np.mean(semi1)
sample_var1 = np.var(semi1)
semi2 = GetSemitones(frIsequence2)
sample_mean2 = np.mean(semi2)
sample_var2 = np.var(semi2)
semi3 = GetSemitones(frIsequence3)
sample_mean3 = np.mean(semi3)
sample_var3 = np.var(semi3)

g1 = GaussD( means=np.array( [sample_mean1] ) , stdevs=np.array( [np.sqrt(sample_var1)] ) )
#g2 = GaussD( means=np.array( [sample_mean1] ) , stdevs=np.array( [np.sqrt(sample_var1)] ) )
#g2 = GaussD( means=np.array( [sample_mean2] ) , stdevs=np.array( [np.sqrt(sample_var2)] ) )
g2 = GaussD( means=np.array( [sample_mean3] ) , stdevs=np.array( [np.sqrt(sample_var3)] ) )
g = [g1, g2]
#g = [g2, g1]

plt.figure(1)
plt.plot(semi1, 'b', label='Semitone', linestyle='dashed')
plt.title(fileName)
plt.ylabel("semitone")
plt.xlabel("t")

#plt.savefig("Figures/semitones_" + str(fileName) + ".png")

semi1 = semi1.reshape(1,len(semi1))
semi2 = semi2.reshape(1,len(semi2))
semi3 = semi3.reshape(1,len(semi3))


logP1 = gauss_logprob(g, semi1)
logP2 = gauss_logprob(g, semi2)
logP3 = gauss_logprob(g, semi3)

P1 = np.e**(logP1-np.max(logP1, axis=0))
P2 = np.e**(logP2-np.max(logP2, axis=0))
P3 = np.e**(logP3-np.max(logP3, axis=0))

mc = MarkovChain(np.array([[1, 0]]), np.array([[0.9, 0.1], [0.1, 0.9]]))  # infinite
h = HMM(mc, g)

alpha_hat1, norms1 = mc.forward(P1)
logprob1 = h.logprob(P1)
print(logprob1)
alpha_hat2, norms2 = mc.forward(P2)
logprob2 = h.logprob(P2)
print(logprob2)
alpha_hat3, norms3 = mc.forward(P3)
logprob3 = h.logprob(P3)
print(logprob3)

beta_hat1 = mc.backward(semi1, P1, norms1)
# print(np.shape(beta_hat1))
# print(beta_hat1[:, 1:5])

gamma1 = mc.condStateProb(alpha_hat1, beta_hat1, norms1)
print(np.shape(gamma1))
plt.plot(gamma1[0, :]*sample_var1/3 + 0*sample_mean1, 'r', label='$P(S_1|X_{1:t})$')
plt.plot(gamma1[1, :]*sample_var1/3 + 0*sample_mean1, 'g', label='$P(S_2|X_{1:t})$')
plt.legend()

plt.show()




