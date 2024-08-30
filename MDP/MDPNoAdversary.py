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
#here are the parameters for the modified iteration
epsilon = 0.0001
# delta = 0.0001
maxIter = 1000

# Define costs (in cents) 
Ca =  2 # Activation Cost
Cd =  2 # Deactivation Cost
Cs = 5 # Service Unit operation Cost
Ch =  5 # Request Holding cost 
Cr =  10 # Cost of Dropped Request

# creating the state space
N = 16 # limit on number of Service Units
B = 100 # request Buffer
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
lam = 0.356 # arrival rate of requests, in requests per second; corresponds to ~ interarrival time of 2.81s
mu = 0.0003 # departure rate of requests, in requests per second; corresponds to mean serice time of ~ 0.95h or 3420s
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
        CostMat.setEntry(indexO,0,(Cd+NumSUs(n,-1)*Cs+r*Ch+lam*Cr)/LAM)
        CostMat.setEntry(indexO,1,(NumSUs(n,0)*Cs+r*Ch+lam*Cr)/LAM)        
        CostMat.setEntry(indexO,2,(Ca+NumSUs(n,1)*Cs+r*Ch+lam*Cr)/LAM)
    else:
        CostMat.setEntry(indexO,0,(Cd+NumSUs(n,-1)*Cs+r*Ch)/LAM)
        CostMat.setEntry(indexO,1,(Cs*NumSUs(n,0)+r*Ch)/LAM)
        CostMat.setEntry(indexO,2,(Ca+NumSUs(n,1)*Cs+r*Ch)/LAM)
    stateSpace.NextState(etat)

#print("********************************")
#print("Cost Matrix")
#print(CostMat)


# Compute transition value for each state.

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
    n = etat[0] + 1 # number of nodes; add 1 to offset 0 index
    r = etat[1] # requests
    for a in range(3):
        act = a-1
        # arrivals
        if r < B:
            p = lam/LAM
            sortie[0] = NumSUs(n,act) - 1 #index offset by 1
            sortie[1] = r + 1
            indexD = stateSpace.Index(sortie)
            trans[a].setEntry(indexO,indexD,p)
        else:
            # special case full buffer, no arrivals
            p = 0
        # departures
        if r > 0:
            q = mu*min(r,NumSUs(n,act))/LAM
            sortie[0] = NumSUs(n,act) - 1 # index offset by 1
            sortie[1] = r - 1
            indexD = stateSpace.Index(sortie)
            trans[a].setEntry(indexO,indexD,q)
        else:
            # special case empty buffer, no depatures
            q = 0
        # self transition psuedo event
        trans[a].setEntry(indexO,indexO,1-p-q)
    stateSpace.NextState(etat)


#print("********************************")
#print("Probability Matrix Action = -1")
#print(P0)
#print("Probability Matrix Action = 0")
#print(P1)
#print("Probability Matrix Action = 1")
#print(P2)
#print("********************************") 
print("Building MDP")
mdp = mmdp.AverageMDP(critere, stateSpace, actionSpace, trans, CostMat)

print("Solving using Value Iteration")
optimum = mdp.ValueIteration(epsilon,maxIter)
print("********************************")
print("Printing Solution")
line = optimum.SolutionByDim(1,stateSpace)
print(line)

#print("Solving using modified Policy Iteration")
#call the function to solve the MDP.
#optimum2 = mdp.PolicyIterationModified(epsilon, maxIter, delta, maxIter)

#print("********************************")
#print("Printing Solution")
#line2 = optimum2.SolutionByDim(1,stateSpace)
#print(line2)
