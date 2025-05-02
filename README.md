# autoscaling-mdp

Code to generate numerical solutions to Markov Decision Processes modeling the Kubernetes autoscaling process and corresponding leader-follower security game.


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

The solvers leverage the discounted Value Iteration method for solving MDPs - this is based on solving a single step Bellman optimization to determine the update:

\begin{equation}
    \max_{a \in \mathcal{A}} \left(\mathcal{R}(s,a) + \sum_{t\in\mathcal{S}}\beta\mathcal{P}(t|s,a)V^n(t)\right)
\end{equation}

## Outputs

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