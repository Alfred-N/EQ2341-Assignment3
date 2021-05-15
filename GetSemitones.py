import numpy as np
def GetSemitones(frIsequence):
  f_log = np.log(frIsequence[0])
  f = frIsequence[0]
  mean = np.mean(f_log)
  stdev = np.std(f_log)
  f_max = mean + stdev
  f_min = mean - stdev

  log_I = np.log(frIsequence[2])
  r_thresh = np.mean(frIsequence[1])
  I_thresh = np.mean(log_I)

  r = frIsequence[1]

  noise = np.zeros(np.shape(f))
  #print(noise.shape)
  for i in range(len(f)):
    if f_log[i]>f_max or f_log[i]<f_min or (r[i]<r_thresh and log_I[i]<I_thresh):
      noise[i] = 1
  #f_clean = f_log[np.where(noise==0)]
  f_clean = f[np.where(noise==0)]
  base_freq=np.min(f_clean)
  #print(base_freq)
  semitones = 12*np.log2(f/base_freq) + 1

  for i in range(len(noise)):
    if noise[i] ==1:
      semitones[i] = np.random.rand(1)*0.5
  return semitones