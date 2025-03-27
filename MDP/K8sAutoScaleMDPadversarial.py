#!/usr/bin/env python
# coding: utf-8

# # Control of K8s autoscaler with single queue, and an adversary present

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
#
# Number of actions: A
#  
# Statistics:  
# lam - Poisson arrival rate to the system    
# mu - homogenized per Service Unit service rate 
# K - attacker strength  
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
# W - wait time threshold for SLA
# beta - factor for discounted MDP
# epsilon - stopping factor for optimal VI solution
# maxIter - maximum number of iterations to run solution over
# INF - definition of "infinity" for cost of invalid action; here defined equal to 20 Decillion Units. As this is greater than total possible global GDP if expressed in USD, this is more than sufficient.

# Build the model with a python dictionary

model=dict()
model['N'] = 101 
model['M'] = 12 
model['A'] = 3
model['lam'] = 50
model['mu'] =  5
model['K'] = 5 #10 #20
model['Ca'] =  0 #0.0017 #0.01
model['Cr'] = 7.47  
model['Cs'] = 0.01
model['Cp'] = 0.075
model['Ck'] = 2
model['W'] = 0.083 #0.167 #0.25
model['beta'] = 0.95 
model['epsilon'] = 0.0001 
model['maxIter'] = 10000 
model['INF'] = 2*(10**34) 

print(model)


# ## Build a discrete time discounted MDP

# ### Build the states

# The state is *(n,b)* with n from 0 to N (which is off-by-1 offset for the actual number of nodes) and b from 0 to B-1 representing the number of requests. 
# An action is *a* with a from 0 to 2 representing the index into the set {-1,0,1} representing the number of nodes to scale by


dims=np.array([model['M'],model['N']])
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
    return max(min(n+a,modele['M']),1)

# define normalization factor, which equal to the maximum transition rate possible for the system at large
NORM = model['lam']+model['M']*model['mu']

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
            if (action == 0 and etat[SU] == 0) or (action == 2 and etat[SU] == modele['M']-1):
                R.setEntry(indexL,j,modele['INF']) # invalid action, cost = infinity
            else:
                # set cost as normal
                nodes = new_su(etat[SU],action,modele)
                activationcosts = 0.0
                if (action != 1):
                    activationcosts += modele['Ca']*(modele['lam']+nodes*modele['mu'])
                rejectioncosts = 0.0
                if ((modele['N']-1)==etat[QUEUE]):
                    rejectioncosts += modele['lam']*modele['Cr'] 
                penaltycosts = 0.0
                if (modele['W'] < etat[QUEUE]/(modele['lam']*nodes)):
                    penaltycosts += modele['Cp']*(etat[QUEUE]-modele['lam']*modele['W']*nodes)
                servercosts = 0.0
                servercosts += nodes*modele['Cs']
                normalizedcosts=(activationcosts+rejectioncosts+penaltycosts+servercosts)/NORM
                #*#print("Rejection Costs=",rejectioncosts,end= ' ')
                #*#print("Accumulated Costs=",accumulatedcosts)
                R.setEntry(indexL,j,normalizedcosts)
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
mdp=md.DiscountedMDP("min",states,actions,trans,Costs,model['beta'])
# mdp = md.AverageMDP("min",states,actions,trans,Costs)
# print(mdp)

# ### Solve using Value Iteration
optimum = mdp.ValueIteration(model['epsilon'],model['maxIter'])
print("Value iteration solution")
line = optimum.SolutionByDim(1,states)
print(line)



# ### Adversarial MDP

# Adversary assumes the optimal action is taken by the application owner, 
# Makes their move in relation to this
#
# Action space is to not attack, 0, or attack, 1

actionsadv = mco.MarmoteBox([2])
# define updated normalization factor for adversarial case
NORMadv = model['lam']*(model['K']+1)+model['M']*model['mu']

# define the adversarial transition and reward matricies

def set_transition_matrix_adv_entry(modele,ssp,act_buf,etat,indexL,opt_act,P):
    #define the states
    afteraction=np.array([0,0])
    jump=np.array([0,0])
    # compute the state after the optimal action
    afteraction[QUEUE]=etat[QUEUE]
    afteraction[SU] = new_su(etat[SU],opt_act,modele) - 1 # new_su returns actual number of nodes, need corresponding index
    ## Arrival (increases the number of customer in first coordinate)
    if (afteraction[QUEUE]<modele['N']-1):
        jump[QUEUE]=afteraction[QUEUE]+1
        jump[SU]=afteraction[SU]
        #compute the index of the jump
        indexC=ssp.Index(jump)
        # compute the (normalized) rate
        if act_buf == 0:
            rateArr = modele['lam']/NORMadv
        elif act_buf == 1:
            # attack in progress, + 1 for standard arrivals
            rateArr = (modele['lam']*(modele['K']+1))/NORMadv
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
        rateDep=min(afteraction[QUEUE],afteraction[SU]+1)*modele['mu']/NORMadv
        #fill in the entry
        #*# print("*Event: Departure. Index=",indexC,"Jump State=",jump,"rate=",rateDep)
        P.setEntry(indexL,indexC,rateDep)
    else:
        rateDep = 0
    #
    ## pseudo-event self-transition; result of normalization factors
    P.setEntry(indexL,indexL,1-rateArr-rateDep)


def fill_in_matrix_adv(index_action,modele,ssp,asp,mdpopt):
    # retrieve the action asscoiated with index
    action_buf = asp.DecodeState(index_action)
    # define transition matrix
    P=mco.SparseMatrix(ssp.Cardinal())
    # browse state space
    etat = np.array([0,0])
    ssp.FirstState(etat)
    for k in range(ssp.Cardinal()):
        # compute index of the state
        indexL = ssp.Index(etat)
        # flags for marking whether action corresponds to scale up/down threshold, reset at each level
        if etat[QUEUE] == 0:
            scaledown = True
            downthresh = -1
            scaleup = False
        #Grab optimal auto-scaling action
        opt_action = mdpopt.getActionIndex(indexL)
        # determine whether to set entries or not
        match opt_action:
            case 0:
                downthresh += 1
                etatdown = etat
                indexdown = indexL
                # psuedo-event not actually part of state space, self transition to self
                P.setEntry(indexL,indexL,1)
            case 1:
                if scaledown:
                    scaledown = False
                    if downthresh > -1:
                        # threshold for scale-down exists
                        set_transition_matrix_adv_entry(modele,ssp,action_buf,etatdown,indexdown,0,P)
                set_transition_matrix_adv_entry(modele,ssp,action_buf,etat,indexL,1,P)
            case 2: 
                if not scaleup:
                    scaleup = True
                    set_transition_matrix_adv_entry(modele,ssp,action_buf,etat,indexL,2,P)
                else:
                    # psuedo-event not actually part of state space, self transition to self
                    P.setEntry(indexL,indexL,1)
        # get next state
        ssp.NextState(etat)
    # print (P)
    return P
        

# Reward matrix - reward is equal to the cost incurred by the provider 
# the cost is equal to the cost of the electricity spend, proportional to traffic

def attack_reward(modele,etat,action,opt_action):
    nodes = new_su(etat[SU],opt_action,modele)
    activationcosts = 0.0
    if (opt_action != 1):
        activationcosts += modele['Ca']*(modele['lam']*(1+action*modele['K'])+nodes*modele['mu'])
    rejectioncosts = 0.0
    if ((modele['N']-1)==etat[QUEUE]):
        rejectioncosts += modele['lam']*(1+action*modele['K'])*modele['Cr']
    penaltycosts = 0.0
    if (modele['W'] < etat[QUEUE]/(modele['lam']*(1+action*modele['K'])*nodes)):
        penaltycosts += modele['Cp']*(etat[QUEUE]-modele['lam']*(1+action*modele['K'])*modele['W']*nodes)
    servercosts = 0.0
    servercosts += nodes*modele['Cs']
    attackcosts = 0.0
    if action == 1:
        attackcosts += modele['K']*modele['Ck']
    reward = (activationcosts+rejectioncosts+penaltycosts+servercosts-attackcosts)/NORMadv
    return reward

def fill_in_reward_adv(modele,ssp,asp, mdpopt):
    R= mco.FullMatrix(ssp.Cardinal(),asp.Cardinal())
    #define the states
    etat=np.array([0,0])
    #define the actions
    acb=asp.StateBuffer()
    ssp.FirstState(etat)
    for k in range(ssp.Cardinal()):
        # compute the index of the state
        indexL=ssp.Index(etat)
        # flags for marking whether action corresponds to scale up/down threshold, reset at each level
        if etat[QUEUE] == 0:
            scaledown = True
            downthresh = -1
            scaleup = False
        #Grab optimal auto-scaling action
        opt_action = mdpopt.getActionIndex(indexL)
        # determine whether to set entries or not
        asp.FirstState(acb)
        for j in range(asp.Cardinal()):
            #*#print("---Action",acb,end='  ')
            action=acb[0]
            match opt_action:
                case 0:
                    downthresh += 1
                    etatdown = etat
                    indexdown = indexL
                    # psuedo-event not actually part of state space, self transition to self
                    if action == 0:
                        R.setEntry(indexL,j,0.0)
                    else:
                        R.setEntry(indexL,j,-1*modele['INF'])
                case 1:
                    if scaledown:
                        scaledown = False
                        if downthresh > -1:
                            # threshold for scale-down exists
                            reward = attack_reward(modele,etatdown,action,opt_action)
                            R.setEntry(indexdown,j,reward)
                    reward = attack_reward(modele,etat,action,opt_action)
                    R.setEntry(indexL,j,reward)
                case 2: 
                    if not scaleup:
                        scaleup = True
                        reward = attack_reward(modele,etat,action,opt_action)
                        R.setEntry(indexL,j,reward)
                    else:
                        # psuedo-event not actually part of state space, self transition to self
                        if action == 0:
                            R.setEntry(indexL,j,0.0)
                        else:
                            R.setEntry(indexL,j,-1*modele['INF'])
            # get next action
            asp.NextState(acb)
        # get next state
        ssp.NextState(etat)
    return R


# Build transition matricies
trans_adv=list()
adv_action_buf = actionsadv.StateBuffer()
actionsadv.FirstState(adv_action_buf)
for k in range(actionsadv.Cardinal()):
    trans_adv.append(fill_in_matrix_adv(k,model,states,actionsadv,optimum))

# Fill in rewards

Reward = fill_in_reward_adv(model,states,actionsadv,optimum)
# print(Reward)


# solve MDP, using maximum criteria as attacker wants to maximize the reward

mdpadv = md.DiscountedMDP("max",states,actionsadv,trans_adv,Reward,model['beta'])

# ### Solve using Policy Iteration
advoptimum = mdpadv.ValueIteration(model['epsilon'],model['maxIter'])
print("Policy iteration solution, Adversarial Attack Decision")
advline = advoptimum.SolutionByDim(1,states)
print(advline)

# ### Print Thresholds
print("Attack thresholds")
etat = np.array([0,0])
states.FirstState(etat)
for k in range(states.Cardinal()):
    indexS = states.Index(etat)
    if etat[QUEUE] == 0:
        idle = True
    adv_opt_action = advoptimum.getActionIndex(indexS)
    if (adv_opt_action == 1) and idle:
        print("m = %d, n = %d", etat[SU], etat[QUEUE])
        idle = False
    if etat[QUEUE] == N and idle:
        print("m = %d no threshold exists", etat[SU])