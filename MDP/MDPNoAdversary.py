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

# We want to minimize the costs at each stage
critere = "min"
# here is the discount factor
beta=0.95
#here are the parameters for the value iteration
epsilon = 0.0001
maxIter = 700

# Define costs 
Ca =  # Activation Cost
Cd = 2 # Deactivation Cost
Cs = 1 # Service Unit operation Cost
Ch = 1 # Request Holding cost 
Cr = 5 # Cost of Dropped Request
INF = 10**10 # Arbitrary cost for invalid actions

# creating the state space
N = 2 # limit on number of Service Units
B = 5 # request Buffer
dimSS = N*B #defining dimension
stateSpace = mc.MarmoteInterval(0,dimSS-1)
# we just created an interval from 0 to dimSS-1.

# creating the action space
dimSA = 3
actionSpace =mc.MarmoteInterval(0,dimSA-1)

# transition rates
lam = 6 # arrival rate of requests, in requests per second
mu = 6 # departure rate of requests, in requests per second
LAM = lam + N*mu # maximum transition rate, used for normalization

print("#")
trans=list()

#Create the first matrix P0 - corresponding to action -1
P0 = mc.SparseMatrix(dimSS)
# Compute transition value for each state.
# Skip n=0 states, cannot subtract nodes when at minimum.
for n in range(1,N):
    for r in range(B):
        IndexO = n*B + r # mapping index of type (n,r) to matrix coordinate
        # arrival case
        if r < B-1:
            p = lam/LAM
            IndexD = (n-1)*B+(r+1)
            P0.setEntry(IndexO,IndexD,p)
        else:
            p = 0 # arrivals not possible, handle self transition appropriately
        # departure case; service rate depends on requests, number of nodes after subtracting
        if r > 0:
            q = mu*min(n,r)/LAM # nodes off by one due to indexing 
            IndexD = (n-1)*B+(r-1)
            P0.setEntry(IndexO,IndexD,q)
        else:
            q = 0 # departures not possible, handle self transition appropriately
        P0.setEntry(IndexO,IndexO,1-p-q) # psuedo-event self transition
trans.append(P0) # add the matrix to the list

#Create the second matrix P1 - corresponding to action 0
P1 =mc.SparseMatrix(dimSS)
# Compute transition value for each state.
for n in range(N):
    for r in range(B):
        IndexO = n*B + r # mapping index of type (n,r) to matrix coordinate
        # arrival case
        if r < B-1:
            p = lam/LAM
            IndexD = n*B+(r+1)
            P1.setEntry(IndexO,IndexD,p)
        else:
            p = 0 # arrivals not possible, handle self transition appropriately
        # departure case; service rate depends on requests, number of nodes after subtracting
        if r > 0:
            q = mu*min(n,r)/LAM # nodes off by one due to indexing 
            IndexD = n*B+(r-1)
            P1.setEntry(IndexO,IndexD,q)
        else:
            q = 0 # departures not possible, handle self transition appropriately
        P1.setEntry(IndexO,IndexO,1-p-q) # psuedo-event self transition
trans.append(P1) # add the matrix to the list

#Create the third matrix P2 - corresponding to action 1
P2 =mc.SparseMatrix(dimSS)
# Compute transition value for each state.
# Skip n=0 states, cannot subtract nodes when at minimum.
for n in range(N-1):
    for r in range(B):
        IndexO = n*B + r # mapping index of type (n,r) to matrix coordinate
        # arrival case
        if r < B-1:
            p = lam/LAM
            IndexD = (n+1)*B+(r+1)
            P2.setEntry(IndexO,IndexD,p)
        else:
            p = 0 # arrivals not possible, handle self transition appropriately
        # departure case; service rate depends on requests, number of nodes after subtracting
        if r > 0:
            q = mu*min(n,r)/LAM # nodes off by one due to indexing 
            IndexD = (n+1)*B+(r-1)
            P2.setEntry(IndexO,IndexD,q)
        else:
            q = 0 # departures not possible, handle self transition appropriately
        P2.setEntry(IndexO,IndexO,1-p-q) # psuedo-event self transition
trans.append(P2) # add the matrix to the list

#Create the reward matrix
Reward  = mc.FullMatrix(dimSS, dimSA)
for n in range(N):
    for r in range(B):
        IndexO = n*B + r
        for a in range(dimSA):
            Cost = (Cs*(n+a-1)+r*Ch)/LAM # Base costs based on operating nodes, holding
            if a == 0:
                # subtracting node, deactivation cost factored in
                Cost += Cd*(lam+n*mu)/LAM
            elif a == 2:
                # adding node, activation cost factored in
                Cost += Ca*(lam+n*mu)/LAM
            if r == B:
                # full buffer, cost of dropped request factored in
                Cost += Cr*lam/LAM
            Reward.setEntry(IndexO,a,Cost)

print("Begining of MDP building")
mdp = mmdp.DiscountedMDP(critere, stateSpace, actionSpace, trans, Reward,beta)
print("End of MDP building\n")

print("Print MDP")
print(mdp)
print("End of Printing MDP")

print("Call of  value iteration")
#call the function to solve the MDP.
optimum = mdp.ValueIteration(epsilon, maxIter)

print("********************************")
print("Print value iteration Solution")
print(optimum)

print("\nCall Gauss Seidel value iteration")
optimum2 = mdp.ValueIterationGS(epsilon,10)
print("Print Gauss Seidel value iteration Solution")
print(optimum2)


print("Call of  value iteration with init")
#call the function to solve the MDP.
optimum3 = mdp.ValueIterationInit(epsilon,200,optimum2)
print("********************************")
print("Print value iteration Solution with Init")
print("Optimum 3",optimum3)

print("Call of Policy Iteration Modified")
optimum4 = mdp.PolicyIterationModified(epsilon, maxIter, 0.001, 100)
#0.001 and 100 are the inner loop parameters
print("Print Policy Iteration Modified Solution") 
print(optimum4)
print("Policy With Str",str(optimum4))

print("\nCall of Policy Iteration Modified GS")
optimum5 = mdp.PolicyIterationModifiedGS(epsilon, maxIter, 0.001, 100)
print("last test",optimum5)
