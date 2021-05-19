from numpy.core.numeric import tensordot
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from PattRecClasses.GaussD_v2 import GaussD
from PattRecClasses.DiscreteD import DiscreteD
from gauss_logprob import gauss_logprob
from GetMusicFeatures import GetMusicFeatures
from GetSemitones import GetSemitones
import numpy as np
import numpy.matlib

from BaumWelch import HMM
from BaumWelch import multigaussD

fileName1 = "melody_1"
fileName2 = "melody_2"
fileName3 = "melody_3"
sr, mel1 = wavfile.read("Songs/"+fileName1 + ".wav")
_, mel2 = wavfile.read("Songs/"+fileName2 + ".wav")
_, mel3 = wavfile.read("Songs/"+fileName3 + ".wav")

frIsequence1, pprd1 = GetMusicFeatures(mel1, sr, winlength=0.03)
frIsequence2, pprd2 = GetMusicFeatures(mel2, sr, winlength=0.03)
frIsequence3, pprd3 = GetMusicFeatures(mel3, sr, winlength=0.03)


semi1 = GetSemitones(frIsequence1)
semi2 = GetSemitones(frIsequence2)
semi3 = GetSemitones(frIsequence3)
max_len = min(len(semi1),len(semi2),len(semi3))
semi1 = semi1[0:max_len]
semi2 = semi2[0:max_len]
semi3 = semi3[0:max_len]
# # semi1 = np.matlib.repmat(semi1,2,1).transpose()
semi1 = np.concatenate((np.reshape(semi1,(len(semi1),1)),np.zeros([len(semi1),1])),axis=1)
semi2 = np.concatenate((np.reshape(semi2,(len(semi2),1)),np.zeros([len(semi2),1])),axis=1)
semi3 = np.concatenate((np.reshape(semi3,(len(semi3),1)),np.zeros([len(semi3),1])),axis=1)

# Define a HMM
n_states=5
end_state=0
# q_old = np.array([0.8, 0.2])
q = np.random.uniform(size=(n_states,))
q[0]=1
q = q/sum(q)
# print(np.shape(q),np.shape(q_old))
A = np.random.uniform(size=(n_states,n_states))
for i in range(n_states):
    A[i,:] = A[i,:]/sum(A[i,:])

# A = np.array([[0.95, 0.05],
#               [0.30, 0.70]])

# means_test = np.array( [[0, 0], [2, 2]] )
# covs_test  = np.array( [[[0.01, 0],[0, 0.01]], 
#                    [[0.01, 0],[0, 0.01]]] )


means = np.array([[i, 0.] for i in range(n_states)])

covs = np.array([[[1, 0],[0, 10**(-2)]] for i in range(n_states)])

# print(means)
# # print(np.shape(covs_test))
# print(np.shape(covs))
# print(covs)
# # print(covs_test)


B = np.array([multigaussD(means[i],covs[i]) for i in range(n_states)])


hm  = HMM(q, A, B)
# obs = np.array([ hm.rand(100)[0] for _ in range(10) ])

np.set_printoptions(suppress=True)
# print(obs[0,0:10])
# # print(np.shape(obs))
# # print(np.shape(semi1))
# # print(semi1[0:10])

song_1 = np.array([semi1, semi2])
song_1_2 = np.array([semi2])
song_2 = np.array([semi3])

# # print('True HMM parameters:')
# # print('q:')
# # print(q)
# # print('A:')
# # print(A)
# # print('B: means, covariances')
# # print(means)
# # print(covs)

# # # Estimate the HMM parameters from the obseved samples
# # # Start by. assigning initial HMM parameter values,
# # # then refine these iteratively

qstar = np.random.uniform(size=(n_states,))
qstar = qstar/sum(qstar)

Astar = np.random.uniform(size=(n_states-end_state,n_states))
for i in range(n_states-end_state):
    if i==0:
        Astar[0,0]=1
    else:
        Astar[i,0]=0
    Astar[i,:] = Astar[i,:]/sum(Astar[i,:])
print(Astar)
meansstar = np.zeros([n_states-end_state,2])
covsstar =  np.array([[[1, 0],[0, 10**(-0)]] for i in range(n_states-end_state)])
Bstar = np.array([multigaussD(meansstar[i],covsstar[i]) for i in range(n_states-end_state)])

hm_learn = HMM(qstar, Astar, Bstar)


print("Running the Baum Welch Algorithm...")
# hm_learn.baum_welch(song_1_2, 20, prin=1, uselog=False) #melody 2
hm_learn.baum_welch(song_2, 20, prin=1, uselog=False) #melody 3


# print("True States:\n",true_states[0:10] )
# Test the Viterbi algorithm
# print("Running the Viterbi Algorithm...")
# predicted_states = hm_learn.viterbi(semi1)
# np.save("pred_states",predicted_states)
# print("Predicted States:\n",predicted_states[150:200] )

print("Calculation likelihoods")
_, c1 = hm_learn.alphahat(semi1)
_, c2 = hm_learn.alphahat(semi2)
_, c3 = hm_learn.alphahat(semi3)
log_likelihood1 = np.sum(np.log(c1))
log_likelihood2 = np.sum(np.log(c2))
log_likelihood3 = np.sum(np.log(c3))
print("likelihood of melody 1 = ", log_likelihood1)
print("likelihood of melody 2 = ", log_likelihood2)
print("likelihood of melody 3 = ", log_likelihood3)




