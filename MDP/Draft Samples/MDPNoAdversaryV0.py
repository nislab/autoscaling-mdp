#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Marmote and MarmoteMDP and pyMarmoteMDP are free softwares: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#Marmote is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with MarmoteMDP. If not, see <http://www.gnu.org/licenses/>.

#Copyright 2019 Emmanuel Hyon, Alain Jean-Marie

"""
 @brief Class to implement MDP for Kubernetes autoscaling decision 
 @author jdchambo
 @date June 2024
 @version 0.1

 This MDP Considers a secnario where a microservice running on Kubernetes is subject to autoscaling based on the amount of traffic present

 This is a naive coding based on translation to unichain methods per claims in Tournaire 2021, 2023 

 This model has three actions:
  -1, remove a service unit
  0, keep number of service units the same
  1, add a service unit

 The number of states is variable and controlled by two variables:
    M, the limit on the number of Service Units
    N, a buffer for the maximum number of requests
"""

# import the library
import marmote.core as mc
import marmote.mdp as mmdp
import numpy as np

'''
MDP Tuning parameters
It is necessary to specify the following:
criteria, min(imization) or max(imization) of values; typcially depending on whether dealing with cost or reward function driving state transitions
Discount factor 0 <= gamma <= 1, where gamma closer to 0 is a greedy user seeking to optimize short term gain, and gamma closer to 1 is seeking to optimize longer term gain
epsilon, the threshold for precision between old and new values
max(imium )Iter(ations), the maximum number of iterations before the system halts

For certain methods, an analogous delta and secondary inner loop maximum is necessary to specify as well, although these can also potentially be set to be the same as epsilon and maxIter 
'''
criterion = "min" # want to minimize costs
gamma = 0.95 # want to emphasize future rewards
epsilon = 0.0001
maxIter = 1000


# transition rates
lam = 500 # arrival rate of requests (500)
mu = 100 # per-SU service rate (100)

'''
Cost Model
Under the Tournaire model, the stage costs consist of five components. Three are instantaneous and related to actions, two accumulate each time unit:
An activation cost Ca
A deactivation cost Cd
A cost for dropped requests Cr
An operation cost per SU Cs
A holding cost per request Ch

The first three are the instantaneous costs, the last two are the accumlated costs. 

If X_a is the rate for state X taking action a, L is the maximum transition rate, Num(m+a) is the number of SUs after taking action a, and lambda is the arrival rate of requests, the formula for staging costs is

(Ca*1_(a=1)+Cd*1_(a=-1))*X_a/L + lambda/L * Cr * 1_(n=N) + (Num(m+a)*Cs+n*Ch)/L
'''   
Ca = 2 # Activation Cost 
Cd = 2 # Deactivation Cost 
Cs = 5 # SU Operation Cost 
Ch = 5 # Holding Cost 
Cr = 10 # Dropped Request Cost 
INF = 2*(10**34) # Cost of invalid action (defined equal to 20 Decillion Units, which in dollars is much greater than real aggregate GDP and is therefore more than sufficent, usually)

'''
Creating state space
States are in the form (m,n) where m is the number of machines (in terms of Service Units), and n is the number of requests

We define a cap on the number of Service Units which can be created, and a buffer on the number of requests

M is the cap on service units, and m is drawn from the ragne [1,M], however due to python indexing the corresponding array index is always m-1
N is the buffer size, and n is drawn from the range [0,N]; due to the same indexing consideration it is necessary to specify a buffer size of N+1 when generating the array 
'''
M = 16 # Service Unit Cap (16)
N = 100 # Request Buffer (100)
LAM = lam + min(M,N)*mu # maximum transition rate, used for normalization
boxdims = np.array([M, N+1]) # dimensions of the state space Marmote Box definition 
stateSpace = mc.MarmoteBox(boxdims) # creates state space [0, ..., M-1] x [0, N] (thus, SUs require +1 added to compensate for python indexing)
dimSS = stateSpace.Cardinal()

'''
Creating action space

Provider actions take the form of modifying the number of machines, adding or subtracting at most one service unit at a time.
The actions are taken when requests enter or exit the system

Because of how python indexing works, the actual action space is defined as the number of machines added/subtracted +1:

0 - subtract one SU
1 - keep same number of SUs
2 - add one SU

'''
actionSpace = mc.MarmoteInterval(0,2) # create action space as the interval of actions [0,1,2]
dimAS = actionSpace.Cardinal()

'''
NumSUs

Determine the number of Service Units resulting from action a.

This is the real number of active service units in the range [1,M] after taking the action.
Thus, a provider could hypothetically attempt to deactivate when at the minimum or activate when at the cap but would be blocked from doing so;
the result being that the number of machines remains the same. This helper function accounts for this.

inputs
su - number of Service Units (note that index offseting must be accounted for before passing in)
act - action taking by provider (index offseting must be accounted for here as well)

output
new - the new number of SUs after accounting for the defined system bounds and the action taken
'''
def NumSUs(su,act):
    new = min(max(1,su+act),M)
    return new

'''
StateRate

Generates the rate per state given state (m,n) and action a, for the request arrival rate lambda and per-SU service rate mu defined above under transition rates

inputs 
su - number of SUs active
req - number of requests in the system
act - action taken

output

rate - the state rate for the given state and action

'''
def StateRate(su,req,act):
    suNew = NumSUs(su,act)
    if req == N:
        arr = 0 # arrival cannot happen, accounted for in dropped request charge, not part of net system state rate
    else:
        arr = lam # buffer not full, arrival possible, include in system state rate
    rate = arr + mu*min(req,suNew)
    return rate

'''
CostGen

Generates the stage cost to pass into the cost matrix for the given state and action, using the Tournaire model above 

inputs

su - number of sus active
req - number of requests in the system
act - action taken

output 

cost - the stage cost associated with the given cost and action

'''
def CostGen(su,req,act):
    # compute accumlated costs
    suops = NumSUs(su,act)*Cs # SU operating costs
    holding = req*Ch # request holding costs
    # determine if instantaneous costs apply
    match act:
        case 1:
            activation = Ca*StateRate(su,req,act) # cost to activate SU
        case -1: 
            activation = Cd*StateRate(su,req,act) # cost to deactivate SU; reuse variable as inverse of same action
        case 0:
            activation = 0    
    if req == N:
        drop = lam*Cr # cost due to dropped request
    else:
        drop = 0
    cost = (suops+holding+activation+drop)/LAM # total cost is sum of instantaneous and accumulated costs, with uniformalization factor 
    return cost

'''
Cost Matrix

Sets the stage costs associated with each state and action

'''
CostMat = mc.FullMatrix(dimSS,dimAS)
etat = np.array([0,0]) # get initial state space
for k in range(dimSS):
    # compute state index
    indexO = stateSpace.Index(etat)
    m = etat[0]+1 # number of SUs; add 1 to offset 0 index
    n = etat[1] # number of requests
    # define the cost for each action
    for a in range(dimAS):
        act = a-1
        # determine cost; if scaling invalid, then cost = infinity
        if ((m == 1 and act == -1) or (m == M and act == 1)):
            cost = INF
        else:
            cost = CostGen(m,n,act)
        CostMat.setEntry(indexO,a,cost)
    stateSpace.NextState(etat)

'''
print("********************************")
print("Cost Matrix")
print(CostMat)
'''

'''
Transition Matricies

Computes the probability of transitioning from state O(rigin) to state D(estination), given action a. Thus, a matrix is created for each action:

P0 - transitions associated with action -1 to deactivate SUs
P1 - transitions associated with action 0 to keep SUs constant
P2 - transitions associated with action 1 to activate SUs

A structure trans is created to hold the three matricies

In the MDP approach, the agent behavior can be broadly described in the form of:

1) Observe State
2) Take Action
3) Observe (arrival/depature)

Regardless of action, the only possible "real" transitions are to states which are adjacent in number of requests (n-1) or (n+1) as arrivals and depatures are the 
triggers for the SU scaling action

The state space probabilities are moderated by the normalization factor LAM representing the maximum transition rate

Because of the normailzation factor, a psuedo-transition event is required for self transitions; as it is a self transition, this will thus always return to the same state despite
arrivals/depatures being the trigger states for scaling.

It is additionally necessary to handle the following special cases (and combinations of special cases):

Empty Buffer - depatures physically impossible
Full Buffer - further arrivals dropped and not recognized (but do accumulate costs as registered by the Cost Matrix)
Attempted Deactivation when 1 SU - not regocgnized as a valid action
Attempted Activation when at Max SUs - not recognized as a valid action

Combinations of the above scenarios result in edge/corner cases to consider.

'''

# Compute transition value for each state.
def arrivalProb(su,req,act):
    '''
    Compute flag to determine whether positive probability cannot be computed
    Return True if either
    a) the buffer is full and therefore requets get dropped thus additional arrivals do not register, or
    b) scaling is invalid due to already being at min/max SUs
    Otherwise return False
    '''
    match act:
        case -1:
            # subtracting 1 SU, cannot do so if at minimum; also cannot accept request with full buffer
            invalid = (su == 1 or req == N)
        case 0:
            # neither adding nor removing SUs, reverts to standard birth-death model for finite chain
            invalid = (req == N)
        case 1:
            # adding 1 SU, cannot do so if at maximum; also cannot accept request with full buffer
            invalid = (su == M or req == N)
    if invalid:
        # determined arrival not possible or conditioned action cannot be triggered
        prob = 0
    else:
        # uniformized probability of arrival to current state
        prob = lam/LAM
    return prob

def depatureProb(su,req,act):
    '''
    Compute flag to determine whether positive probability cannot be computed
    Return True if either
    a) the buffer is empty and therefore depatures physically impossible, or 
    b) scaling is invalid due to already being at min/max SUs
    Otherwise return False
    '''
    match act:
        case -1:
            # subtracting 1 SU, cannot do so if at minimum; also cannot have depature with empty buffer 
            invalid = (su == 1 or req == 0)
        case 0:
            # neither adding nor removing SUs, reverts to standard birth-death model for finite chain
            invalid = (req == 0)
        case 1:
            # adding 1 SU, cannot do so if at maximum; also cannot have depature with empty buffer
            invalid = (su == M or req == 0)
    if invalid:
        # determined depature not possible or conditioned action cannot be triggered
        prob = 0
    else:
        # uniformized probability of depature from current state
        prob = mu*min(req,NumSUs(su,act))/LAM
    return prob

#Create matrix corresponding to each action 
P0 = mc.SparseMatrix(dimSS) # action -1
P1 = mc.SparseMatrix(dimSS) # action 0
P2 = mc.SparseMatrix(dimSS) # action 1
trans = [P0,P1,P2]
etat = np.array([0,0]) # get initial state space
sortie = np.array([0,0]) # array to represent the end state following transition, intialize to dummy state
for k in range(dimSS):
    # compute state index
    indexO = stateSpace.Index(etat)
    m = etat[0] + 1 # active SUs; add 1 to offset 0 indexing
    n = etat[1] # number of requests in the system
    for a in range(3):
        act = a-1
        # determine arrival probability
        p = arrivalProb(m,n,act)
        if p > 0:
            sortie[0] = NumSUs(m,act) - 1 #index offset by 1
            sortie[1] = n + 1
            indexD = stateSpace.Index(sortie)
            trans[a].setEntry(indexO,indexD,p)
        # determine depature probability
        q = depatureProb(m,n,act)
        if q > 0:
            sortie[0] = NumSUs(m,act) - 1 # index offset by 1
            sortie[1] = n - 1
            indexD = stateSpace.Index(sortie)
            trans[a].setEntry(indexO,indexD,q)
        # self transition psuedo-event; this is guarenteed for any state
        self = 1 - (p+q)
        trans[a].setEntry(indexO,indexO,self)          
    stateSpace.NextState(etat)
'''
print("********************************")
print("Probability Matrix Action = -1")
print(P0)
print("Probability Matrix Action = 0")
print(P1)
print("Probability Matrix Action = 1")
print(P2)
'''


print("********************************") 
print("Building MDP")
mdp = mmdp.DiscountedMDP(criterion, stateSpace, actionSpace, trans, CostMat, gamma)
#mdp = mmdp.AverageMDP(criterion, stateSpace, actionSpace, trans, CostMat)

#call the function to solve the MDP.
'''
print("Solving using Value Iteration")
optimum = mdp.ValueIteration(epsilon, maxIter)
'''
print("Solving using modified Policy Iteration")
optimum = mdp.PolicyIterationModified(epsilon, maxIter, epsilon, maxIter)


print("********************************")
print("Printing Solution")
line = optimum.SolutionByDim(1,stateSpace)
print(line)

