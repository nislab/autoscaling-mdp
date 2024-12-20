#!/usr/bin/env python
# coding: utf-8

# # Control of K8s autoscaler with single queue

# Code adapted from sample code for Marmote Application Lesson 2 for Application to the Control of a Tandem Multi-Server System
# https://marmote.gitlabpages.inria.fr/marmote/pytutos/App_Lesson2.html

# In[ ]:


import marmote.core as mco
import marmote.mdp as md
import numpy as np


# ## The model definition

# We consider a single queue, multi-server K8s model with auto scaling

# **Parameters**
# 
# Size of the systems: B (for request Buffer)
# 
# Number of servers: N (for K8s Nodes)
#
# Number of actions: A
#  
# Statistics:  
# lam - Poisson arrival rate to the system    
# mu - homogenized per Service Unit service rate   
# 
# Costs:  
# Cr - cost of rejecting request due to full buffer (instantaneous based on prob. of arrival)   
# Ch - cost of request hold time; factor of SLA and expected response time based on current queue size and number of service units    
# Cs - cost of operating each additional service unit
# 
# Other parameters:
# 
# beta - factor for discounted MDP
# epsilon - stopping factor for optimal VI solution
# maxIter - maximum number of iterations to run solution over
# INF - definition of "infinity" for cost of invalid action; here defined equal to 20 Decillion Units. As this is greater than total possible global GDP if expressed in USD, this is more than sufficient.

# Build the model with a python dictionary

model=dict()
model['B'] = 10 
model['N'] = 5 
model['A'] = 3
model['lam'] = 50 
model['mu'] = 500 
model['Cr'] = 10  
model['Cs'] = 0.013  
model['Ch'] = 0.25 
model['beta'] = 0.95 
model['epsilon'] = 0.0001 
model['maxIter'] = 700 
model['INF'] = 2*(10**34) 

print(model)


# ## Build a discrete time discounted MDP

# ### Build the states

# The state is *(n,b)* with n from 0 to N-1 (which is off-by-1 offset for the actual number of nodes) and b from 0 to B-1 representing the number of requests. 
# An action is *a* with a from 0 to 2 representing the index into the set {-1,0,1} representing the number of nodes to scale by


dims=np.array([model['N'],model['B']])
# print(dims) 
states= mco.MarmoteBox(dims)
#
actions=mco.MarmoteBox([model['A']])

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
SU = 0
QUEUE = 1

def new_su(su,act,modele):
    # get number of K8s nodes associated with a given state, action index
    n = su + 1 # state space index is offset by 1 from actual number of nodes 
    a = act - 1 # action space index is offset by 1 from action (in opposite direction, e.g. index 0 = subtract 1 node)
    return max(min(n+a,modele['N']),1)

# define normalization factor, which equal to the maximum transition rate possible for the system at large
NORM = model['lam']+model['N']*model['mu']

def set_transition_entries(modele,ssp,etat,indexL,act, P):
    # define transition states
    afteraction=np.array([0,0])
    jump=np.array([0,0])
    # get state after action - which either decreases SUs by 1, increases SUs by 1, or keeps same number of SUs
    afteraction[QUEUE]=etat[QUEUE]
    afteraction[SU] = new_su(etat[SU],act,modele) - 1 # new_su returns actual number of nodes, need corresponding index
    #*# print("####index State=",k,"State",etat,"State after action",afteraction)
    ### detail possible transitions
    # Arrival (increases the number of customers by 1 with rate lambda)
    if (afteraction[QUEUE]<modele['B']-1):
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
    # Departure (decreases number of customers by 1 with rate mu)
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
    ## pseudo-event self-transition; result of normalization factors
    P.setEntry(indexL,indexL,1-rateArr-rateDep)

def fill_in_matrix(index_action,modele,ssp,asp):
    # define transition matrix
    P=mco.SparseMatrix(ssp.Cardinal()) 
    # retrieve the action asscoiated with index
    action_buf = asp.DecodeState(index_action)
    #*#print("index action",index_action,"action",action_buf)
    #define the states
    etat=np.array([0,0])
    # browsing state space
    ssp.FirstState(etat)
    for k in range(ssp.Cardinal()):
        # compute the index of the state
        indexL=ssp.Index(etat)
        # compute and set the corresponding entries
        set_transition_entries(modele,ssp,etat,indexL,action_buf[0],P)
        # get next state
        ssp.NextState(etat)
    # print(P)
    return P


# #### Cost Matrix

# We define now a function to fill in the cost matrix. 
#
# If an action is invalid, it is defined as infinity, otherwise costs are defined as below
#     
# rejection cost= Cr*lam/NORM in states where *b=B* .  
# 
# Accumulated Costs are:  
# (SLA Cost of Customer Throughput) = (b/(lam*nodes_after_action)*Ch   
# (number of activated Service Units) = (nodes_after_action)*Cs
# 
# Where nodes_after_action depends on the current number of service unit nodes n, and the action a
# (e.g. if nodes are already at minimum, attempted subtraction has no effect)  
#
# Accumlated Costs are summed and then normalized


def fill_in_cost(modele,ssp,asp):
    R= mco.FullMatrix(ssp.Cardinal(),asp.Cardinal())
    #define the states
    etat=np.array([0,0])
    #define the actions
    acb=asp.StateBuffer()
    ssp.FirstState(etat)
    for k in range(ssp.Cardinal()):
        # compute the index of the state
        indexL=ssp.Index(etat)
        #*#print("##State",etat)
        asp.FirstState(acb)
        for j in range(asp.Cardinal()):
            #*#print("---Action",acb,end='  ')
            action=acb[0]
            if (action == 0 and etat[SU] == 0) or (action == 2 and etat[SU] == modele['N']-1):
                R.setEntry(indexL,j,modele['INF']) # invalid action, cost = infinity
            else:
                # set cost as normal
                nodes = new_su(etat[SU],action,modele)
                rejectioncosts=0.0
                if ((modele['B']-1)==etat[QUEUE]):
                    rejectioncosts+=(modele['lam']*modele['Cr'])/NORM 
                accumulatedcosts = 0.0
                accumulatedcosts=((etat[QUEUE]/(modele['lam']*nodes))*modele['Ch'] + nodes*modele['Cs'])/NORM
                #*#print("Rejection Costs=",rejectioncosts,end= ' ')
                #*#print("Accumulated Costs=",accumulatedcosts)
                R.setEntry(indexL,j,accumulatedcosts+rejectioncosts)
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
# mdp=md.DiscountedMDP("min",states,actions,trans,Costs,model['beta'])
mdp = md.AverageMDP("min",states,actions,trans,Costs)
# print(mdp)

# ### Solve the MDP using Value Iteration

# optimum=mdp.ValueIteration(model['epsilon'],model['maxIter'])
# print("Value iteration solution")
# line = optimum.SolutionByDim(1,states)
# print(line)

# ### Solve using Policy Iteration
optimum2 = mdp.PolicyIterationModified(model['epsilon'],model['maxIter'],0.001,100)
print("Policy iteration solution")
line2 = optimum2.SolutionByDim(1,states)
print(line2)

def fill_in_policy_matrices(modele,ssp,mdpopt,mdpcost):
    # define transition and cost matricies
    P = mco.SparseMatrix(ssp.Cardinal()) 
    C = mco.FullMatrix(states.Cardinal(),1)
    # browse state space
    etat=np.array([0,0])
    ssp.FirstState(etat)
    # Flags for defining when to compute probabilities
    scaledown = True
    downscaleindex = -1
    scaleup = False
    for k in range(ssp.Cardinal()):
        # compute the index of the state
        indexL=ssp.Index(etat)
        # Compute the opimal action, resetting flags as necessary
        Q = etat[QUEUE]
        if Q == 0:
            # at next level, reset flags
            scaledown = True
            downscaleindex = -1
            scaleup = False
        opt_action = mdpopt.getActionIndex(indexL)
        # compute the state after the action
        if opt_action == 0:
            downscaleindex = indexL
            # get asscoaited cost
        if opt_action == 1:
            if scaledown:
                # at scaling threshold for scale down
                scaledown = False
                if downscaleindex > -1:
                    # downscaling indicated for queue size >= 0 (should effectively occur any time SU > 1, but may not always be case)
                    set_transition_entries(modele,ssp,etat,indexL,opt_action,P)
                    scalecost =  mdpcost.getEntry(downscaleindex,0)
                    C.setEntry(downscaleindex,0,scalecost)
            set_transition_entries(modele,ssp,etat,indexL,opt_action,P)
            scalecost = mdpcost.getEntry(indexL,1)
            C.setEntry(indexL,0,scalecost)
        if opt_action == 2 and not scaleup:
            # at scaling threshold for scale up
            scaleup = True
            set_transition_entries(modele,ssp,etat,indexL,opt_action,P)
            scalecost = mdpcost.getEntry(indexL,2)
            C.setEntry(indexL,0,scalecost)
        ssp.NextState(etat)
    return P, C
    

    '''
        afteraction=np.array([0,0])
    jump=np.array([0,0])
        afteraction[QUEUE]=etat[QUEUE]
        
        afteraction[SU] = new_su(etat[SU],action_buf[0],modele) - 1 # new_su returns actual number of nodes, need corresponding index
        #*# print("####index State=",k,"State",etat,"State after action",afteraction)
        # tdetail all the possible transitions
        ## Arrival (increases the number of customer in first coordinate with rate lambda)
        if (afteraction[QUEUE]<modele['B']-1):
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
'''

# Build Policy From Optimal Solution
bbuf = states.StateBuffer()
policy, policycost = fill_in_policy_matrices(model,states,optimum2, Costs)
print(policy)
print(policycost)
'''
states.FirstState(bbuf)
scaledown = True
downscaleindex = -1
scaleup = False
for k in range(states.Cardinal()):
    indexS = states.Index(bbuf)
    Q = bbuf[QUEUE] # get current size of queue
    S = bbuf[SU] # get current SUs of system
    if Q == 0:
        # at next level, reset flags
        scaledown = True
        downscaleindex = -1
        scaleup = False
    opt_action = optimum2.getActionIndex(indexS)
    if (opt_action == 0):
        downscaleindex = indexS
    if (opt_action == 1):
        if scaledown:
            scaledown = False
            if (downscaleindex > -1):
                print("state = ",downscaleindex," optimal action =", 0)
        print("state = ",indexS," optimal action =", 1)
        cbuf = states.StateBuffer()
        states.FirstState(cbuf)
        for l in range(states.Cardinal()):
            indexL = states.Index(cbuf)
            #t = trans[opt_action][indexS,indexL]
            #policy.setEntry(indexS,indexL,t)
            states.NextState(cbuf)
    if (opt_action == 2 and not scaleup):
        scaleup = True
        print("state = ",indexS," optimal action =", 2)
    states.NextState(bbuf)

print(trans[0])
'''