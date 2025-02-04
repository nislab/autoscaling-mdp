#!/usr/bin/env python
# coding: utf-8

# # Stocastic Game MDP for single-queue Kubernetes Cluster with both attacker and defender present

# Code adapted from sample code for Marmote Application Lesson 2 for Application to the Control of a Tandem Multi-Server System
# https://marmote.gitlabpages.inria.fr/marmote/pytutos/App_Lesson2.html

# In[ ]:

import marmote.core as mco
import marmote.mdp as md
import marmote.markovchain as mch
import numpy as np


# ## The model definition

# We consider a single queue, multi-server K8s model with auto scaling

# **Parameters**
# 
# Size of the systems: N - maximum size of the buffer
# Number of Service Units: M (for Machines)
# Number of defender (cluster scaling) actions: D
# Number of attacker actions: A
#  
# Statistics:  
# lam - Poisson arrival rate to the system    
# mu - homogenized per Service Unit service rate 
# K - attacker strength
# r - scaling randomization factor in [0,1)  
# 
# Costs:
# Ca - deadweight billing costs of activating/deacivating machine 
# Cr - cost of rejecting request due to full buffer (instantaneous based on prob. of arrival)   
# Cp - SLA request hold penalty for exceeding expected wait time - based on queue size
# Cs - cost of operating each additional service unit
# Ck - cost of launching an attack (proportional to attack strength)
# 
# Other parameters:
# 
# B - Total attack budget, determines probability of attack
# W - wait time threshold for SLA
# beta - factor for discounted MDP
# epsilon - stopping factor for optimal VI solution
# maxIter - maximum number of iterations to run solution over
# INF - definition of "infinity" for cost of invalid action; here defined equal to 20 Decillion Units. As this is greater than total possible global GDP if expressed in USD, this is more than sufficient.

# Build the model with a python dictionary

model=dict()
model['N'] = 101 
model['M'] = 12
model['D'] = 3  
model['A'] = 2
model['lam'] = 50
model['mu'] =  5
model['K'] = 5 #10 #20
model['Ca'] =  0 #0.00002686 #0.00316
model['Cr'] = 2.11  
model['Cs'] = 0.00316
model['Cp'] = 0.0211
model['Ck'] = 0.34
model['W'] = 0.083 #0.167 #0.25
model['beta'] = 0.95 
model['epsilon'] = 0.0001 
model['maxIter'] = 10000 
model['INF'] = 2*(10**34) 

print(model)


# ## Build a discrete time discounted MDP

# ### Build the state and action spaces

# States are of the form (m,n) with m from [1,M] being the number of Machines/Service Units; n from [0,N] being the number of requests
# Zero indexing means that machines are indexed [0,M-1] and are an off-by-1 offset; while a buffer of N requires model['N'] = N+1 to represent the number of possible states, including n=0 
statedims=np.array([model['M'],model['N']])
# print(statedims) 
states= mco.MarmoteBox(statedims)

# Actions are of the form (d,a) representing the number of machines to scale by, and the strength of the attack.
# by default, d indexes into the set {-1,0,1}, thus removing or adding at most one SU
# by default, a indexes into the set {0,1} representing remaining idle, or attacking by transmitting strength K*lambda rate traffic
actiondims = np.array([model['D'],model['A']])
# print(actiondims)
actions=mco.MarmoteBox(actiondims)

# print("Number of states",states.Cardinal())
# print(states)
# print("Number of actions",actions.Cardinal())
# print("actions",actions)


# ### Build matrices

# #### Transition matrices

# We begin by defining a function which computes the transition matrix associated with an action such that the action index is: index_action.
# 
# In a state, there are two events - arrvials, and depatures. We also account for a third, pseudo-event where a self-transition occurs resulting from normalization
# While Marmote has built in normalization methods, we normalize manually to ensure consistency 
#

# In[ ]:

# name coordinate indicies for ease of reading
# state coordinates
SU = 0
QUEUE = 1
# action coordiantes
DEF = 0
ATT = 1


def new_su(su,act,modele):
    # get number of K8s nodes associated with a given state, action index
    n = su + 1 # state space index is offset by 1 from actual number of nodes 
    a = act - 1 # action space index is offset by 1 from action (in opposite direction, e.g. index 0 = subtract 1 node)
    return max(min(n+a,modele['M']),1)

# define normalization factor, which equal to the maximum transition rate possible for the system at large
NORM = (model['K']+1)*model['lam']+model['M']*model['mu']

def fill_in_matrix(index_action,modele,ssp,asp):
    # retrieve the action asscoiated with index
    action_buf = asp.DecodeState(index_action)
    #*#print("index action",index_action,"action",action_buf)
    #define the states
    etat=np.array([0,0])
    afteraction=np.array([0,0])
    jump=np.array([0,0])
    # define transition matrix
    P=mco.SparseMatrix(ssp.Cardinal()) 
    # browsing state space
    ssp.FirstState(etat)
    for k in range(ssp.Cardinal()):
        # compute the index of the state
        indexL=ssp.Index(etat)
        # compute the state after the action
        afteraction[QUEUE]=etat[QUEUE]
        afteraction[SU] = new_su(etat[SU],action_buf[0],modele) - 1 # new_su returns actual number of nodes, need corresponding index
        #*# print("####index State=",k,"State",etat,"State after action",afteraction)
        # tdetail all the possible transitions
        ## Arrival (increases the number of customer in first coordinate with rate lambda)
        if (afteraction[QUEUE]<modele['N']-1):
            jump[QUEUE]=afteraction[QUEUE]+1
            jump[SU]=afteraction[SU]
            #compute the index of the jump
            indexC=ssp.Index(jump)
            # compute the (normalized) rate
            rateArr = modele['lam']/NORM
            #fill in the entry
            #*# print("*Event: Arrival. Index=",indexC,"Jump State=",jump,"rate=",rateArr)
            P.setEntry(indexL,indexC,rateArr)
        else: 
            rateArr = 0
        #
        ##departure (decreases number of customer in first coordinate with rate mu)
        if (afteraction[QUEUE]>0):
            jump[QUEUE]= afteraction[QUEUE]-1
            jump[SU]= afteraction[SU]
            #compute the index of the jump
            indexC=ssp.Index(jump)
            # compute the (nromalized) rate
            rateDep=min(afteraction[QUEUE],afteraction[SU]+1)*modele['mu']/NORM
            #fill in the entry
            #*# print("*Event: Departure. Index=",indexC,"Jump State=",jump,"rate=",rateDep)
            P.setEntry(indexL,indexC,rateDep)
        else:
            rateDep = 0
        #
        ## pseudo-event self-transition; result of normalization factors
        P.setEntry(indexL,indexL,1-rateArr-rateDep)
        # get next state
        ##
        ssp.NextState(etat)
    # print(P)
    return P


# #### Cost Matrix

# The Cost Matrix represents the costs in the zero-sum-game played by the two players.
# The defender "pays" the attacker 
#
# Accumlated Costs are summed and then normalized

def calc_def_cost(modele,etat,action):
    normalizedcosts = 0.0
    nodes = new_su(etat[SU],action[DEF],modele)
    # SU operating costs
    normalizedcosts += nodes*modele['Cs'] 
    # Minimum SU charges applied to unused time
    if (action[DEF] != 1):
        normalizedcosts += modele['Ca']*(modele['lam']+nodes*modele['mu'])
    # SLA delay penalty costs
    if (modele['W'] < etat[QUEUE]/(modele['lam']*nodes)):
        normalizedcosts += modele['Cp']*(etat[QUEUE]-modele['lam']*modele['W']*nodes)
    # SLA rejected request costs
    if ((modele['N']-1)==etat[QUEUE]):
        normalizedcosts += modele['lam']*modele['Cr']
    normalizedcosts /= NORM
    return normalizedcosts

def calc_att_cost(modele,action):
    normalizedcosts = 0.0
    if action[ATT] == 1:
        normalizedcosts += modele['K']*modele['Ck']
    normalizedcosts /= NORM
    return normalizedcosts


def fill_in_cost(modele,ssp,asp):
    R= mco.FullMatrix(ssp.Cardinal(),asp.Cardinal())
    #define the states
    etat=np.array([0,0])
    #define the actions
    acb=asp.StateBuffer()
    ssp.FirstState(etat)
    for k in range(ssp.Cardinal()):
        # compute the index of the state
        indexS=ssp.Index(etat)
        #*#print("##State",etat)
        asp.FirstState(acb)
        for j in range(asp.Cardinal()):
            #*#print("---Action",acb,end='  ')
            action=acb[0]
            if (action == 0 and etat[SU] == 0) or (action == 2 and etat[SU] == modele['M']-1):
                R.setEntry(indexS,indexA,modele['INF']) # invalid action, cost = infinity
            else:
                # set cost as normal
                defcosts = calc_def_cost(modele,etat,action)
                attcosts = calc_att_cost(modele,action)
                R.setEntry(indexS,indexA,defcosts-attcosts)
            asp.NextState(acb)
        ssp.NextState(etat)
    return R

# ### Build the continuous time MDP

# Build transition matrices

trans=list()

action_buf = actions.StateBuffer()
actions.FirstState(action_buf)
for k in range(actions.Cardinal()):
    trans.append(fill_in_matrix(k,model,states,actions))
    # print("---Transition Matrix ",k, "filled in")

# Fill cost matrix

# print("Cost Matrix")
Costs=fill_in_cost(model,states,actions)
# print(Costs)

# Build the MDP, specifying minimum criteria as want cost minimization for scaling policy

#print("Begining of Building MDP")
mdp=md.DiscountedMDP("max",states,actions,trans,Costs,model['beta'])
# mdp = md.AverageMDP("min",states,actions,trans,Costs)
# print(mdp)

# ### Solve using Value Iteration
optimum = mdp.ValueIteration(model['epsilon'],model['maxIter'])
print("Value iteration solution")
line = optimum.SolutionByDim(1,states)
print(line)

