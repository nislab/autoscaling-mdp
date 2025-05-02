# autoscaling-mdp

Code to generate numerical solutions to Markov Decision Processes modeling the Kubernetes autoscaling process and corresponding leader-follower security game.

Within Kubernetes, there exist concerns related to autoscaling and in particular _Economic Denial of Sustainability_ which can result in cloud bills being driven up to excessive levels. However, the autoscaling problem more generally is an open question as optimizations are largely dependant on the workload type, and are not necessarily approached in a formalized way.

Here, we strive to develop Markov Decision Process based models to formalize the autoscaling process, and to determine the incentives for attacking a cluster on the basis of EDoS. While some work exists on the subject of MDP based scaling models<sup>[1]</sup>, our work here features a leader-follower (Stackelberg) game in which the defender optimizes their policy, upon which the attacker determines the incentive to launch the attack.

The MDPs consist of:
- State spaces, here ordered pairs (m,n) of Service Units of computing power, and requests in the system
- Action spaces, determining the decisions which can be made at a given time.
- A utility function to optimize, either some cost function to minimize for cluster operations or a net revenue funciton to maximize.
- A probability transition matrix associated with each action determining where states can transition between.

Here, transitions are triggered by a single arrival or depature. The defender determines whether to add or remove a SU (or keep the SUs the same). The attacker determines whether to attack or not. The incentives depend on SLA penalties, the ongoing cost of operating the SUs, and any minimum charges imposed by the billing structure. The attacker is also subject to the cost of attacks reducing the total reward.


[1] Tournaire, Thomas,  Hind Castel-Taleb, and Emmanuel Hyon. 2023. "Efficient Computation of Optimal Thresholds in Cloud Auto-scaling Systems". *ACM Transactions on Modeling and Performance Evaluation of Computing Systems* (December 2023) Vol 8, Issue 4.  

------------

# Usage

To run the routine, run the script directly from the command line for the desired scenario:

- K8sAutoScaleMDP.py runs the MDP solver solely to optimize the autoscaling problem based on our formulation of the problem, without consideration of the attacker.
- K8sAutoScaleMDPadversarial.py conducts the full two-stage game, solving the defender's solution first and using this as input for the attacker strategy.

## Dependencies

The code utilizes [Marmote](https://marmote.gitlabpages.inria.fr/marmote/index.html)<sup>[2]</sup>, a package developed for solving MDPs within C++ which utilizes a Python-based wrapper for simplified syntax. The published version of our code uses the Marmote Python API.

Marmote installation requires the Anaconda or [miniconda](https://conda.io/miniconda.html) package manager to be installed 

Once installed, the following installs the Marmote packages (if using Windows, first open an `anaconda prompt` or a `conda powershell` via the Start menu):

'''
conda create -n marmote-use
conda activate marmote-use
conda install -c marmote -c conda-forge marmote
'''

In addition, NumPy is required for array generation if not already installed. Conda enviornments can also be used for this:

'''
conda create -n my-env
conda activate my-env
conda install numpy
'''

Alternatively, if pip is preferred:

'''
pip install numpy
'''
[2] Jean-Marie, Alain and Emmanuel Hyon. "Marmote's Documentation" https://marmote.gitlabpages.inria.fr/marmote/about.html, Last Accessed 2025-05-02


## Inputs

The inputs are controlled via the Dict structures at the top of the scripts, and correspond to the system parameters governing the MDP models as described in "Exploiting Kubernetes Autoscaling for Economic Denial of Sustainability."

* N - maximum size of the request buffer (in reality set to N+1 due to Python 0 indexing; e.g. a buffer of 100 should have a corresponding N value of 101).
* M - maximum size of the Service Unit pool; again, due to zero-indexing, this results in a given index "m" representing the state of m+1 SUs being active, as M >= 1 by design. 
* A - the number of available actions; that is, the size of the action space available for scaling machines - by default this is 3 as the actions are to scale up or down by at most 1. Note that if this is to be updated, code must be updated to ensure that indexing is correctly accounted for as currently the scaling action is derived by subtracting 1 from the action index. ([0,2] -> [-1,1]).
* lam - arrival rate of legitimate traffic.
* mu - service rate of a single SU.
* Ca - the cost related to minimum charges incurred when machines are activated/deactivated; i.e. the "deadweight cost" incurred by committing to activating the resource for a minimum period, and/or deactivating it with time remaining in the current billing cycle.
* Cs - The cost to run a SU 
* Cr - SLA penalty for rejecting requests due to a full buffer.
* Cp - SLA penalty for requests being held for excessive periods. 
* W - the threshold of time before the SLA penalty is applied
* epsilon - the stopping factor for applying the optimal solution. By default we let this be 0.0001.
* beta - the discount factor for weighting greedy vs. patient solutions. Beta is restricted between [0,1]; by default we let beta = 0.95, which heavily favors patient solutions.
* maxInter - the maximum number of iterations before the algorithm aborts with the current solution; used to prevent an infinite loop scenario.
* INF - a defined value of Infinity to prevent invalid actions from being chosen by heavily penalizing them. This is set to 20 Decillion such that even if arbitrarily large values are utilized for the other values, INF should remain multiple orders of magnitude greater than any reasonable value which could be picked to model at scale, as this value exceeds total global GDP as expressed in most stable currencies.

The following parameters are specific to the adversarial MDP as they only apply to the adversary's (potential) presence:

* K - the strength of the attack; i.e. the proportionally greater level of traffic the attacker sends, such that the adversarial rate of traffic lam_a = Klam, and by extension total traffic under attack conditions is (K+1)lam.
* Ck - the cost to launch an attack, which is proportional to the attack strength. E.g. in the POMACS/SIGMETRICS work, we assumed access to a 2kW power source server farm to generate the traffic rate for each K of attack power, thus an attack of K=5 times the rate of traffic would cost 5K or 10kW under this assumption. 


## MDP Solvers

The solvers leverage the discounted Value Iteration method for solving MDPs - this is based on solving a single step Bellman optimization to determine the update; an initial policy is chosen to populate a vector of values $V^0$. Subsequently, at each step the next vector $V^{n+1}$ is populated for each state by determining the optimal value at each state:

$V^{n+1}(s) = \max_{a \in \mathcal{A}} \left(\mathcal{R}(s,a) + \sum_{t\in\mathcal{S}}\beta\mathcal{P}(t|s,a)V^n(t)\right)$.

Once $||V^{n+1} - V^n|| < \epsilon(1-\beta)/2\beta$, the solution halts (unless maxIter is reached first); the corresponding policy are the set of actions chosen at the current step $n+1$, denoted by $\pi$:

$\pi_a(s) = \arg \max_{a \in \mathcal{A}} \left(\mathcal{R}(s,a) + \sum_{t\in\mathcal{S}}\beta\mathcal{P}(t|s,a)V^{n+1}(t)\right)$.

To accomplish this, the code generates the discretized Semi-MDP associated with the configuration.

First, by filling in the transition matrix for each action and calculating the probability of an arrival or depature event given the starting state and given action, as well as the corresponding normalization factor NORM, which is the maximum per-state transition rate. This will simply be the arrival rate plus the maximum service rate, itself a product of the service rate and the maximum number of SUs.

The code then fills in the cost matrix, computing the cost to operate the cluster at each stage according to the following formula:

$\mathcal{C}(s,a) = \Big((m+a)C_s + \Lambda(s,a)\mathbbm{1}_{d \neq 0}C_a + \lambda\mathbbm{1}_{n=N}C_r + (n - \lambda (m+a) W)\mathbbm{1}_{\frac{n}{(m+a)\lambda} > W}C_p  \Big)\Big/ \tilde{\Lambda}$,

where $\Lambda$ is the state transition rate at the given state, and $\tilde{\Lambda}$ is the normalization factor NORM. Invalid actions are assigned a cost of INF.

An additional helper function computes the number of SUs following an action given the minimum and maximum SU restrictions as well as the zero-index offset.

In the case of the defender, the criteria for the Value Iteration is set to minimize the cost.

If running the adversarial script, the output solution is used as the input for the adversarial MDP, which scans over the state space to determine the states where the defender scales up or down. The transition and cost matricies are otherwise filled in analogously, save for the fact that the actions are to attack or remain idle, and thus rates of arrival depend on the attack being active. This also results in a new normalization factor NORMadv which accounts for the attacker arrivals as part of the maximum tranistion rate.

The "cost" matrix is now a net reward defined as follows:
$\mathcal{R}(s,a) = \Big((m+\pi(s))C_s + \Lambda_{adv}(s,\pi(s))\mathbbm{1}_{d \neq 0}C_a + (1+K\mathbbm{1}_{a=1})\lambda\mathbbm{1}_{n=N}C_r + (n - \lambda (m+\pi(s)) W)\mathbbm{1}_{\frac{n}{(m+\pi(s))(K\mathbbm{1}_{a=1}+1)\lambda} > W}C_p + K\mathbbm{1}_{a=1}C_k  \Big)\Big/ \tilde{\Lambda_{adv}}$, 

 where $\Lambda_{adv}$ is the state transition rate at the given state, and $\tilde{\Lambda_{adv}}$ is the normalization factor NORMadv. Invalid actions are assigned a cost of -INF.

The resulting MDP is then run through a discounted Value Iteration of its own to determine the attacker incentive, this time based on the maximum value.

Marmote supports alternative solution methods, such as directly implementing CT Discounted MDPs, the use of Policy Iteration based solutions, Gauss Sidel improvement based algorithms, and Average MDPs. The [API](https://marmote.gitlabpages.inria.fr/marmote/python_index.html) contains the full list; to change solution methods the optimum and advoptimum lines must be updated at a minimum; changing to a different MDP type (e.g. Discounted to Average) requires changing the mdp and mdpadv lines to reflect the new structure, and may require additional rewrites to accomodate differences in how structures process inputs.

## Outputs

By default, the output is printed to screen; this can be redirected on the command line or code can be modified to specify the file destination.

The code prints:
The Dict structure - to validate the inputs in the solution
The Value Iteration solution for the initial scaling problem, this is printed by dimension. For instance:

'''
etat : 4	(   0,   4)	       0.0052574   1
etat : 5	(   0,   5)	       0.0056945   2
'''

Represents an ouput where at state (1,4), that is 1 SU, 4 requests the optimal action is to keep the number of SUs the same, with corresponding cost 0.0053, but at state (1,5) the optimal action is to scale up by 1 with corresponding cost 0.0057.

If running the adversarial MDP, this is then followed by the solution for the adversarial MDP:

'''
etat : 845	(   8,  37)	       0.0158669   0
etat : 846	(   8,  38)	       0.0245403   0
etat : 847	(   8,  39)	               0   0
'''

In the above example, at 9 SUs the net reward is highest when the attacker remains idle, and thus the attack is never launched; in this scenario the cluster scales up by the time 39 requests enter the system and therefore the state is not visited.

'''
etat : 1155	(  11,  44)	               0   0
etat : 1156	(  11,  45)	         0.24872   0
etat : 1157	(  11,  46)	         1.08587   1
'''

Conversely at 12 SUs the cluster scales to 12 at 45 reuqests, but at that point there is no incentive to attack, whereas one does exist if 46 requests are present.

The final output is the minimum threshold to launch the attack at each SU level. Note that this prints the actual number of SUs as this is a custom routine and not a full printout of the MDP solution from the Marmote structure, thus 

'''
Attack thresholds
m = 10, n = 40
m = 11, n = 42
m = 12, n = 46
'''

Correspond to states (9,40), (10,42), and (11,46) in the actual MDP. 

------------------------

# License


These materials may be freely used and distributed, provided that attribution to this original source is acknowledged. If you reuse the code in this repository, we kindly ask that you refer to the relevant work (cf. the included bib files in the citation directory of this repository):

* Chamberlain, Jonathan, Jilin Zheng, Zeying Zhu, Zaoxing Liu, and David Starobinski. 2025. "Exploiting Kubernetes Autoscaling for Economic Denial of Sustainability." *Proceedings of the ACM on Measurement and Analysis of Computing Systems* Vol. 9, Issue. 2, Article 22 (June 2025). 
-----------------
# References

If you are leveraging this code in your work and would like to be featured in this list, kindly create an issue in this repository and provide us with the reference.

-----------------
# Acknowledgment

Support by the US National Science Foundation is greatfully acknoweldged, with these scripts developed in support of work which is supported in part by the following grants: 

* AST-2229104 (BU)
* SaTC-2415754 (UMD)
* CNS-2431093  (UMD) 