import numpy as np
from .DiscreteD import DiscreteD


class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """

    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  # InitialProb(i)= P[S(1) = i]

        self.A = transition_prob  # TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]

        self.nStates = transition_prob.shape[0]

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True
            extra_row = [np.eye(self.A.shape[1])[self.A.shape[1] - 1]]
            self.A = np.concatenate((self.A, extra_row), axis=0)
            self.q = np.concatenate((self.q, [[0]]), axis=1)
            self.nStates = self.nStates + 1
            print(self.A)

    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates) - self.A.T) @ self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T @ pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)

        logpD = np.log(aii) * t + np.log(1 - aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1 / (1 - np.diag(self.A))

    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        # t_rand = np.random.randint(1,high=tmax,size=1,dtype=int)
        # *** Insert your own code here and remove the following error message

        if self.is_finite == False:
            S = np.zeros([1, tmax], dtype=int)
            S[0][0] = DiscreteD(self.q[0]).rand(1)
            for i in range(1, tmax):
                # print(S.dtype)
                # pS_t = np.dot(pS_t,self.A)
                S[0][i] = int(DiscreteD(self.A[S[0, i - 1], :]).rand(1))
            return S

        elif self.is_finite:

            S = np.array([[int(0)]])

            S[0][0] = int(DiscreteD(self.q[0]).rand(1))
            for i in range(1, tmax):

                s_t = int(DiscreteD(self.A[S[0, i - 1], :]).rand(1))
                if s_t == self.nStates - 1:
                    break
                else:
                    S = np.concatenate((S, np.array([[s_t]])), axis=1)
                    # S[0].append(s_t)
            return S

    def viterbi(self):
        pass

    def stationaryProb(self):
        pass

    def stateEntropyRate(self):
        pass

    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass

    def initErgodic(self):
        pass

    def forward(self, P):
        T = len(P[0, :])
        if self.is_finite:
            P = np.concatenate((P, np.zeros([1, T])), axis=0)
            norms = np.zeros([T + 1, 1])
        else:
            norms = np.zeros([T, 1])

        N = len(P)  # num states
        alpha = np.zeros([N, T])
        alpha_hat = np.zeros(np.shape(alpha))

        alpha[:, 0] = np.multiply(self.q, P[:, 0])
        alpha_hat[:, 0], norms[0, 0] = self.normalize(alpha[:, 0])

        for t in range(1, T):
            alpha_a_product = np.dot(self.A.transpose(), alpha_hat[:, t - 1])
            b_kt = P[:, t]
            alpha[:, t] = np.multiply(b_kt, alpha_a_product)
            alpha_hat_t, norms_t = self.normalize(alpha[:, t])
            alpha_hat[:, t] = alpha_hat_t
            norms[t, 0] = norms_t
        if self.is_finite:
            norms[T,0] = np.dot(self.A.transpose()[N-1,:], alpha_hat[:,t])

        return alpha_hat, norms

    def finiteDuration(self):
        pass

    def backward(self):
        pass

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass

    def normalize(self, alpha_temp):

        norm = np.sum(alpha_temp)
        alpha_hat = alpha_temp / norm
        return alpha_hat, norm
