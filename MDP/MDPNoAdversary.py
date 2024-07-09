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

 This model has three actions:
  -1, remove a service unit
  0, keep number of service units the same
  1, add a service unit

 The number of states is variable and controlled by two variables:
    N, the limit on the number of Service Units
    B, a buffer for the maximum number of requests
"""

# import the library
import marmote.core as mc
import marmote.mdp as mmdp
import numpy as np

# We want to minimize the costs at each stage
critere = "min"
# here is the discount factor
#beta=0.95
#here are the parameters for the value iteration
epsilon = 0.0001
maxIter = 700

# Define costs (in cents) 
Ca =  2 # Activation Cost
Cd =  2 # Deactivation Cost
Cs = 5 # Service Unit operation Cost
Ch =  5 # Request Holding cost 
Cr =  10 # Cost of Dropped Request
INF = 10**10 # Arbitrary cost for invalid actions

# creating the state space
N =  3 # limit on number of Service Units
B = 5 # request Buffer
boxdims = np.array([N, B+1]) # dimensions of the Marmote Box defining the state space
stateSpace = mc.MarmoteBox(boxdims) # creates state space [0, ..., N-1] x [0, B] due to python indexing
dimSS = stateSpace.Cardinal()

# creating the action space as interval of the possible actions labeled 0, 1, 2
actionSpace = mc.MarmoteInterval(0,2)
dimAS = actionSpace.Cardinal()

def NumSUs(su,act):
    # determine the number of Service Units resulting from action
    new = min(max(1,su+act),N)
    return new

# transition rates
lam = 2 # arrival rate of requests, in requests per second
mu = 3 # departure rate of requests, in requests per second
LAM = lam + N*mu # maximum transition rate, used for normalization

# define Cost Matrix
CostMat = mc.FullMatrix(dimSS,dimAS)
etat = np.array([0,0]) # get initial state space
for k in range(dimSS):
    # compute state index
    indexO = stateSpace.Index(etat)
    n = etat[0]+1 # number of nodes; add 1 to offset 0 index
    r = etat[1] # requests
    # define the cost for each action
    if (r == B):
        # account for full buffer
        if (n == 1):
            # subtraction invalid at min nodes
            CostMat.setEntry(indexO,0,INF)
        else:
            CostMat.setEntry(indexO,0,(Cd+NumSUs(n,-1)*Cs+r*Ch+lam*Cr)/LAM)
        CostMat.setEntry(indexO,1,(NumSUs(n,0)*Cs+r*Ch+lam*Cr)/LAM)
        if (n == N):
            # addition invalid at max nodes
            CostMat.setEntry(indexO,2,INF)
        else:
            CostMat.setEntry(indexO,2,(Ca+NumSUs(n,1)*Cs+r*Ch+lam*Cr)/LAM)
    else:
        if (n == 1):
            # subtraction invalid at min nodes
            CostMat.setEntry(indexO,0,INF)
        else:
            CostMat.setEntry(indexO,0,(Cd+NumSUs(n,-1)*Cs+r*Ch)/LAM)
        CostMat.setEntry(indexO,1,(Cs+NumSUs(n,0)*r*Ch)/LAM)
        if (n == N):
            # addition invalid at max nodes
            CostMat.setEntry(indexO,2,INF)
        else:
            CostMat.setEntry(indexO,2,(Ca+NumSUs(n,1)*Cs+r*Ch)/LAM)
    stateSpace.NextState(etat)




# Compute transition value for each state.

#Create matrix corresponding to action -1
P0 = mc.SparseMatrix(dimSS) 
etat = np.array([0,0]) # get initial state space
sortie = np.array([0,0]) # array to represent the end state following transition, intialize to dummy state
for k in range(dimSS):
    # compute state index
    indexO = stateSpace.Index(etat)
    n = etat[0] + 1 # number of nodes; add 1 to offset 0 index
    r = etat[1] # requests
    #condition on special cases
    # no valid transition when n = 1
    if n > 1:
        if r == 0: 
            # empty queue, arrivals only
            p = lam/LAM
            sortie[0] = NumSUs(n,-1) - 1 # index offset by 1
            sortie[1] = r + 1
            indexD = stateSpace.Index(sortie)
            P0.setEntry(indexO,indexD,p)
            P0.setEntry(indexO,indexO,1-p)
        elif r == B:
            # full queue, depatures only 
            q = mu*min(r,NumSUs(n,-1))/LAM
            sortie[0] = NumSUs(n,-1) - 1 # index offset by 1
            sortie[1] = r - 1
            indexD = stateSpace.Index(sortie)
            P0.setEntry(indexO,indexD,q)
            P0.setEntry(indexO,indexO,1-q)
        else:
            # arrival
            p = lam/LAM
            sortie[0] = NumSUs(n,-1) - 1 # index offset by 1
            sortie[1] = r + 1
            indexD = stateSpace.Index(sortie)
            P0.setEntry(indexO,indexD,p)
            # departure 
            q = mu*min(r,NumSUs(n,-1))/LAM
            sortie[0] = NumSUs(n,-1) - 1 # index offset by 1
            sortie[1] = r - 1
            indexD = stateSpace.Index(sortie)
            P0.setEntry(indexO,indexD,q)
            P0.setEntry(indexO,indexO,1-p-q)
    stateSpace.NextState(etat)


#Create matrix corresponding to action 0
P1 =mc.SparseMatrix(dimSS)
etat = np.array([0,0]) # get initial state space
sortie = np.array([0,0]) # array to represent the end state following transition, intialize to dummy state
for k in range(dimSS):
    # compute state index
    indexO = stateSpace.Index(etat)
    n = etat[0] + 1 # number of nodes; add 1 to offset 0 index
    r = etat[1] # requests
    #condition on special cases
    if r == 0: 
        # empty queue, arrivals only
        p = lam/LAM
        sortie[0] = NumSUs(n,0) - 1 # index offset by 1
        sortie[1] = r + 1
        indexD = stateSpace.Index(sortie)
        P1.setEntry(indexO,indexD,p)
        P1.setEntry(indexO,indexO,1-p)
    elif r == B:
        # full queue, depatures only 
        q = mu*min(r,NumSUs(n,0))/LAM
        sortie[0] = NumSUs(n,0) - 1 # index offset by 1
        sortie[1] = r - 1
        indexD = stateSpace.Index(sortie)
        P1.setEntry(indexO,indexD,q)
        P1.setEntry(indexO,indexO,1-q)
    else:
        # arrival
        p = lam/LAM
        sortie[0] = NumSUs(n,0) - 1 # index offset by 1
        sortie[1] = r + 1
        indexD = stateSpace.Index(sortie)
        P1.setEntry(indexO,indexD,p)
        # departure 
        q = mu*min(r,NumSUs(n,0))/LAM
        sortie[0] = NumSUs(n,0) - 1 # index offset by 1
        sortie[1] = r - 1
        indexD = stateSpace.Index(sortie)
        P1.setEntry(indexO,indexD,q)
        P1.setEntry(indexO,indexO,1-p-q)
    stateSpace.NextState(etat)


#Create matrix corresponding to action 1
P2 =mc.SparseMatrix(dimSS)
etat = np.array([0,0]) # get initial state space
sortie = np.array([0,0]) # array to represent the end state following transition, intialize to dummy state
for k in range(dimSS):
    # compute state index
    indexO = stateSpace.Index(etat)
    n = etat[0] + 1 # number of nodes; add 1 to offset 0 index
    r = etat[1] # requests
    #condition on special cases
    # no valid transition when n = N
    if n < N: 
        if r == 0: 
            # empty queue, arrivals only
            p = lam/LAM
            sortie[0] = NumSUs(n,1) - 1 # index offset by 1
            sortie[1] = r + 1
            indexD = stateSpace.Index(sortie)
            P2.setEntry(indexO,indexD,p)
            P2.setEntry(indexO,indexO,1-p)
        elif r == B:
            # full queue, depatures only 
            q = mu*min(r,NumSUs(n,1))/LAM
            sortie[0] = NumSUs(n,1) - 1 # index offset by 1
            sortie[1] = r - 1
            indexD = stateSpace.Index(sortie)
            P2.setEntry(indexO,indexD,q)
            P2.setEntry(indexO,indexO,1-q)
        else:
            # arrival
            p = lam/LAM
            sortie[0] = NumSUs(n,1) - 1 # index offset by 1
            sortie[1] = r + 1
            indexD = stateSpace.Index(sortie)
            P2.setEntry(indexO,indexD,p)
            # departure 
            q = mu*min(r,NumSUs(n,1))/LAM
            sortie[0] = NumSUs(n,1) - 1 # index offset by 1
            sortie[1] = r - 1
            indexD = stateSpace.Index(sortie)
            P2.setEntry(indexO,indexD,q)
            P2.setEntry(indexO,indexO,1-p-q)
    stateSpace.NextState(etat)



trans = [P0,P1,P2]
print("#") 
print("Building MDP")
mdp = mmdp.AverageMDP(critere, stateSpace, actionSpace, trans, CostMat)

print("Solving using modified Policy Iteration")
#call the function to solve the MDP.
optimum = mdp.PolicyIterationModified(epsilon, maxIter, 0.001, 100)

print("********************************")
print("Printing Solution")
line = optimum.SolutionByDim(1,stateSpace)
print(line)


#call the function to solve the MDP.
#optimum3 = mdp.ValueIterationInit(epsilon,200,optimum2)

#optimum4 = 
#0.001 and 100 are the inner loop parameters
