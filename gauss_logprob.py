'''
logP = gauss_logprob(pDs,x)
method to give the probability of a data sequence,
assumed to be drawn from given Gaussian Distribution(s).

Input:
pD=    GaussD object or array of GaussD objects
x=     row vector with data assumed to be drawn from a Gaussian Distribution

Result:
logP=  array with log-probability values for each element in x,
       for each given GaussD object
       size(p)== [length(pDs),size(x,2)], if pDs is one-dimensional vector
       size(p)== [size(pDs),size(x,2)], if pDs is multidim array
'''
import numpy as np
def gauss_logprob(pDs, x):
    nObj = len(pDs) # Number of GaussD Objects
    nx = x.shape[1] # Number of observed vectors
    logP = np.zeros((nObj, nx))

    for i, pD in enumerate(pDs):
        dSize = pD.dataSize
        assert dSize == x.shape[0]

        z = np.dot(pD.covEigen, (x-np.matlib.repmat(pD.means, 1, nx)))

        z /= np.matlib.repmat(np.expand_dims(pD.stdevs, 1), 1, nx)

        logP[i, :] = -np.sum(z*z, axis=0)/2 
        logP[i, :] = logP[i, :] - sum(np.log(pD.stdevs)) - dSize*np.log(2*np.pi)/2

    return logP
