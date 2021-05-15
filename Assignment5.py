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

semi1 = GetSemitones(frIsequence)














# plt.figure(1)
# plt.plot(semi1)
# plt.title(fileName)
# plt.ylabel("semitone")
# plt.xlabel("t")
# plt.savefig("Figures/semitones_" + str(fileName) + ".png")
