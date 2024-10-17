/* Marmote and MarmoteMDP are free softwares: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Marmote is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Marmote. If not, see <http://www.gnu.org/licenses/>.

Copyright 2022 Emmanuel Hyon, Alain Jean-Marie*/

/**
 * @class to implement Medium arrival case from Tournaire paper 
 * @author jdchambo.
 * @version 0.1
 * @date September 2023
 *
 * This model is of a Kubernetes cluster with autoscaling, where scaling is based upon cost aware metrics.
 * 
 * The state space has two dimensions, number of requests x number of active nodes [0,B]*[1,N]
 * 
 * Where B is the maximum size of the buffer, and N is the limit on the number of nodes
 * 
 * The objective is to minimize costs, thus rewards are negative based on cost to activate/deactiate/run machines,
 * costs of request delay, and cost of dropping requests due to insufficient capacity
 * 
 * There are three actions:
 *  0 - subtract a node
 *  1 - keep number of nodes constant
 *  2 - add a node
 * 
 *  This is so the number of nodes added by a given action is always a-1,
 * (negative action values are not permitted as they are used as indicies to matricies)
 * 
 * Original MDP is continuous, thus leverage uniformization to discretize the MDP
 * 
 * MDP is average cost utilizing relative value iteration as the basis for solution, using an average cost model
**/



#include <iostream>
#include <list>
#include <vector>
#include <string>

using namespace std;

#include "marmoteCore/marmoteInterval.h"
#include "marmoteCore/marmoteBox.h"
#include "marmoteCore/marmoteSparseMatrix.h"
#include "marmoteMDP/marmoteAverageMDP.h"
#include "marmoteMDP/marmoteFeedbackSolutionMDP.h"
#include "marmoteMDP/marmoteSolutionMDP.h"

int CostCalc(int Ca, int Cd, int Cs, int Ch, int Cr, int a, int LAMu, int lam, int mu, int n, int N, int r, int B) {
    int C = 0;
    int nNew = max(min(n+a,N),0); // compute number of new nodes, needed for both (de)activation and running nodes costs
    /* First consider cost of (de)activation if relevant */
    if (a == 1) {
        C += Ca;
    }
    else if (a == -1) {
        C += Cd;
    }
    C *= lam + mu * min(r, nNew);
    /* If buffer full, consider cost of dropped request */
    if (r == B) {
        C += lam * Cr;
    }
    /* Consider running costs of held requests and running nodes */
    C += nNew * Cs + r * Ch;
    /* normalize cost */
    C /= LAMu;
    return -1 * C;
}


int main( int argc, char** argv )
{
    
    /* create indices */
    stateType k; /* to iterate on states */
    stateType indexO; /* index of current/origin state */
    stateType indexD; /* index of destination state */ 
    int r,n; /* get the indicies of the number of requests (columns) and number of nodes (rows) */
    int nNew; /* number of nodes after action a taken */

    /* denote parameters of K8s cluster*/
    int B = 100; // the maximum buffer size for requests
    int N = 16; // the maximum number of nodes which may be active
    int lam = 500; // the arrival rate of new requests
    int mu = 100; // the service rate of requests
    int Ca = 2; // the cost of activating a node
    int Cd = 2; // the cost of deactivating a node
    int Cs = 5; // cost of running a node per unit of time
    int Ch = 5; // cost of holding requests in the queue
    int Cr = 10; // cost of dropped requests

    int LAMu = lam + N*mu; // the uniformized rate, used for computing transition probabilities
    int p = lam/LAMu; // (uniformized) probability of an arrival, constant w/r/t the active nodes
    int q; // (uniformized) probability of a departure, depends on the active nodes following an action
    
    /* create the state space as a marmoteBox */
    /* definitions of the size of the two dimensions: dimension 1 is queue length and dimension 2 is number of nodes */
    stateType dims[2]={B+1,N}; // add 1 to buffer to account for 0 indexing
    /* create the box*/
    MarmoteBox *stateSpace = new MarmoteBox(2,dims);

    // create the action space as an interval between 0 and 2
    MarmoteInterval *actionSpace = new MarmoteInterval(0,2);               
          
    // compute the size of the cardinal and store it in variable dim_SS
    stateType dim_SS = stateSpace->Cardinal();
    stateType dim_AS = actionSpace->Cardinal();
    
    /*Allocate buffers (to be used to now the index of the state) */
    /* allows to manage origin state (before transition) */
    MarmoteState statebufferO = stateSpace->StateBuffer();
    /* allows to manage destination state (after transition) */
    MarmoteState statebufferD = stateSpace->StateBuffer();
    
    
    /* create the matrix for the Costs. This is a dim_SS * dim_AS matrix */
    SparseMatrix *CostMat  = new SparseMatrix(dim_SS, dim_AS);
    /* fill in the matrix */
    /* costs depend on action taken, number of active nodes, and number of requests */
    /* initialize statebufferO such that it is the first state in the state space */
    stateSpace->FirstState(statebufferO);
    int Cost;
    for(k=0;  k < dim_SS; k++) {
        /* computing the index of the state */
        indexO=stateSpace->Index(statebufferO);
        r = statebufferO[0];
        n = statebufferO[1] + 1; // number of nodes is offset by 1 due to zero index
        for(int a=0;a<dim_AS;a++){
            /* Compute cost from action taken and state information */
            Cost = CostCalc(Ca, Cd, Cs, Ch, Cr, a - 1, LAMu, lam, mu, n, N, r, B);
            CostMat->setEntry(indexO,a,Cost);
        }
        stateSpace->NextState(statebufferO);
    }
    
    
    /* create and initialize elements of MDP */
    string critere("min");
    
    /* create and initialize elements of solving */
    double epsilon = 0.0001;
    int maxIter=250;
    
    cout<<"Begining of building MDP"<<endl;
    /* create the MDP */
    AverageMDP *mdpSSP = new AverageMDP(critere, stateSpace, actionSpace, CostMat);
    cout << "Fin of building MDP" <<endl;

    cout<<"Add matrices"<<endl;

    // Define transition matricies for each action 
    
    /* Define matrix for action 0 (SUBTRACT Node) */
    SparseMatrix *P0 = new SparseMatrix(dim_SS);
    /* if buffer is neither empty nor full, can have arrival or departure event */
    for(r = 1 ; r < B ; r++) {
        statebufferO[0]=r; /* initialize the value of the first dim */
        /* handle n = 1 corner case - cannot subtract a node if only one node present 
        As probability of events is otherwise 0, uniformization results in pesudo-event transition to current state with probability 1 */
        statebufferO[1] = 0; // index of nodes offset by one
        indexO=stateSpace->Index(statebufferO);
        P0->setEntry(indexO,indexO,1);
        for(n = 2 ; n <= N ; n++) {    
            /* define a state and get its index */            
            statebufferO[1]=n-1; /* initialize the value of the second dim; offset by one due to indexing */
            indexO=stateSpace->Index(statebufferO);
            nNew = n-1; /* Action is to subtract node */
            statebufferD[1] = nNew-1; // index of nodes offset by one
            q = mu*min(r,nNew)/LAMu; // service rate depends on number of nodes, but buffer may be smaller than number of active nodes
            /* If arrival */
            statebufferD[0] = r+1;
            indexD=stateSpace->Index(statebufferD); //compute index of destination
            P0->setEntry(indexO,indexD,p);
            /* If departure */
            statebufferD[0] = r-1;
            indexD=stateSpace->Index(statebufferD); //compute index of destination
            P0->setEntry(indexO,indexD,q);
            /* pesudo-event transition to current state */
            P0->setEntry(indexO,indexO,1-p-q);
        }
    }
    /* If buffer empty, can only have arrivals, adjust probabilities accordingly for first row */
    statebufferO[0] = 0;
    statebufferD[0] = 1;
    /* handle n = 1 corner case */
    statebufferO[1] = 0; // index offset by 1
    indexO=stateSpace->Index(statebufferO);
    P0->setEntry(indexO,indexO,1);
    for (n = 2; n <= N; n++) {
        statebufferO[1]=n-1; /* initialize the value of the second dim, offset by 1 from number of nodes */
        indexO=stateSpace->Index(statebufferO);
        nNew = n-1; /* Action is to subtract node */
        statebufferD[1] = nNew-1; // index of nodes offset by one
        indexD=stateSpace->Index(statebufferD); //compute index of destination
        /* If arrival */
        P0->setEntry(indexO,indexD,p);
        /* pesudo-event transition to current state */
        P0->setEntry(indexO,indexO,1-p);
    }
    /* If buffer full, can only have departures, adjust probabilities accordingly for last row */
    statebufferO[0] = B;
    statebufferD[0] = B-1;
    /* handle n = 1 corner case */
    statebufferO[1] = 0; // index offset by 1
    indexO=stateSpace->Index(statebufferO);
    P0->setEntry(indexO,indexO,1);
    for (n = 2; n <= N; n++) {
        statebufferO[1]=n-1; /* initialize the value of the second dim, offset by 1 from number of nodes */
        indexO=stateSpace->Index(statebufferO);
        nNew = n-1; /* Action is to subtract node */
        statebufferD[1] = nNew-1; // index of nodes offset by one
        q = mu*min(B,nNew)/LAMu; // service rate depends on number of nodes, but buffer may be smaller than number of active nodes
        indexD=stateSpace->Index(statebufferD); //compute index of destination
        /* If departure */
        P0->setEntry(indexO,indexD,q);
        /* pesudo-event transition to current state */
        P0->setEntry(indexO,indexO,1-q);
    }
    
    mdpSSP->AddMatrix(0,P0);
    cout<<"Added matrix (action 0)"<< endl;
    
    SparseMatrix *P1 = new SparseMatrix(dim_SS);
    /* Define matrix for action 1 (MAINTAIN Nodes) */
    /* if buffer is neither empty nor full, can have arrival or departure event */
    for(r = 1 ; r < B ; r++) {
        for(n = 1 ; n <= N ; n++) {   
            /* define a state and get its index */
            statebufferO[0]=r; /* initialize the value of the first dim */
            statebufferO[1]=n-1; /* initialize the value of the second dim, offset by 1 from number of nodes */
            indexO=stateSpace->Index(statebufferO);
            nNew = n; /* Action is to keep current number of nodes */
            statebufferD[1] = nNew - 1; // index is offset by 1 from number of nodes 
            q = mu*min(r,nNew)/LAMu; // service rate depends on number of nodes, but buffer may be smaller than number of active nodes
            /* If arrival */
            statebufferD[0] = r+1;
            indexD=stateSpace->Index(statebufferD); //compute index of destination
            P1->setEntry(indexO,indexD,p);
            /* If departure */
            statebufferD[0] = r-1;
            indexD=stateSpace->Index(statebufferD); //compute index of destination
            P1->setEntry(indexO,indexD,q);
            /* pesudo-event transition to current state */
            P1->setEntry(indexO,indexO,1-p-q);
        }
    }
    /* If buffer empty, can only have arrivals, adjust probabilities accordingly for first row */
    statebufferO[0] = 0;
    statebufferD[0] = 1; 
    for (n = 1; n <= N; n++) {
        statebufferO[1]=n-1; /* initialize the value of the second dim, offset by 1 from number of nodes */
        indexO=stateSpace->Index(statebufferO);
        nNew = n; /* Action is to keep current number of nodes */
        statebufferD[1] = nNew - 1; // index is offset by 1 from number of nodes 
        indexD=stateSpace->Index(statebufferD); //compute index of destination
        /* If arrival */
        P1->setEntry(indexO,indexD,p);
        /* pesudo-event transition to current state */
        P1->setEntry(indexO,indexO,1-p);
    }
    /* If buffer full, can only have departures, adjust probabilities accordingly for last row */
    statebufferO[0] = B;
    statebufferD[0] = B-1;
    for (n = 1; n <= N; n++) {
        statebufferO[1]=n-1; /* initialize the value of the second dim, offset by 1 from number of nodes */
        indexO=stateSpace->Index(statebufferO);
        nNew = n; /* Action is to keep current number of nodes */
        statebufferD[1] = nNew - 1; // index is offset by 1 from number of nodes
        q = mu*min(B,nNew)/LAMu; // service rate depends on number of nodes, but buffer may be smaller than number of active nodes
        indexD=stateSpace->Index(statebufferD); //compute index of destination
        /* If departure */
        P1->setEntry(indexO,indexD,q);
        /* pesudo-event transition to current state */
        P1->setEntry(indexO,indexO,1-q);
    }
    
    mdpSSP->AddMatrix(1,P1);
    cout<<"Added matrix (action 1)"<< endl;
    
    /* Define matrix for action 2 (ADD Node) */
    SparseMatrix *P2 = new SparseMatrix(dim_SS);
     /* if buffer is neither empty nor full, can have arrival or departure event */
    for(r = 1 ; r < B ; r++) {
        statebufferO[0]=r; /* initialize the value of the first dim */
        /* handle n = N corner case - cannot add a node if already at the limit 
        As probability of events is otherwise 0, uniformization results in pesudo-event transition to current state with probability 1 */
        statebufferO[1] = N-1; // index of nodes offset by one
        indexO=stateSpace->Index(statebufferO);
        P2->setEntry(indexO,indexO,1);
        for(n = 1 ; n <= N-1 ; n++) {    
            statebufferO[1]=n-1; /* initialize the value of the second dim, index offset by 1 */
            indexO=stateSpace->Index(statebufferO);
            nNew = n+1; /* Action is to add node */
            statebufferD[1] = nNew - 1; // index offset by 1 
            q = mu*min(r,nNew)/LAMu; // service rate depends on number of nodes, but buffer may be smaller than number of active nodes
            /* If arrival */
            statebufferD[0] = r+1;
            indexD=stateSpace->Index(statebufferD); //compute index of destination
            P2->setEntry(indexO,indexD,p);
            /* If departure */
            statebufferD[0] = r-1;
            indexD=stateSpace->Index(statebufferD); //compute index of destination
            P2->setEntry(indexO,indexD,q);
            /* pesudo-event transition to current state */
            P2->setEntry(indexO,indexO,1-p-q);
        }
    }
    /* If buffer empty, can only have arrivals, adjust probabilities accordingly for first row */
    statebufferO[0] = 0;
    statebufferD[0] = 1;
    /* handle n = N corner case */
    statebufferO[1] = N-1; // index offset by 1
    indexO=stateSpace->Index(statebufferO);
    P2->setEntry(indexO,indexO,1);
    for (n = 1; n <= N-1; n++) {
        statebufferO[1]=n-1; /* initialize the value of the second dim, offset by 1 from number of nodes */
        indexO=stateSpace->Index(statebufferO);
        nNew = n+1; /* Action is to add node */
        statebufferD[1] = nNew-1; // index of nodes offset by one
        indexD=stateSpace->Index(statebufferD); //compute index of destination
        /* If arrival */
        P2->setEntry(indexO,indexD,p);
        /* pesudo-event transition to current state */
        P2->setEntry(indexO,indexO,1-p);
    }
    /* If buffer full, can only have departures, adjust probabilities accordingly for last row */
    statebufferO[0] = B;
    statebufferD[0] = B-1;
    /* handle n = N corner case */
    statebufferO[1] = N-1; // index offset by 1
    indexO=stateSpace->Index(statebufferO);
    P2->setEntry(indexO,indexO,1);
    for (n = 2; n <= N; n++) {
        statebufferO[1]=n-1; /* initialize the value of the second dim, offset by 1 from number of nodes */
        indexO=stateSpace->Index(statebufferO);
        nNew = n+1; /* Action is to add node */
        statebufferD[1] = nNew-1; // index of nodes offset by one
        q = mu*min(B,nNew)/LAMu; // service rate depends on number of nodes, but buffer may be smaller than number of active nodes
        indexD=stateSpace->Index(statebufferD); //compute index of destination
        /* If departure */
        P2->setEntry(indexO,indexD,q);
        /* pesudo-event transition to current state */
        P2->setEntry(indexO,indexO,1-q);
    }
    
    mdpSSP->AddMatrix(2,P2);
    cout<<"Added matrix (action 2)"<< endl;
    
    
    
    cout<<"Finishing Adding matrices MDP"<< endl;
    cout<<"Writing MDP"<<endl;
    //mdpSSP->WriteMDP();
    
    cout<<endl<<endl<<"###############################"<<endl;

    cout<<endl<<endl<<"###############################"<<endl;

    cout<<"Printing solution from value iteration"<<endl;
    cout<<"Done : path length  "<< maxIter <<endl;
    SolutionMDP *optimum2 = mdpSSP->ValueIteration(epsilon, maxIter);
    cout<<"Print solution"<< endl;
    optimum2->WriteSolution();

    cout<<endl<<endl<<"################################"<<endl;
    

    /* to get FeedbackPolicy properties we should make a cast */
    FeedbackSolutionMDP * policy;
    if ( dynamic_cast <FeedbackSolutionMDP *> (optimum2)  != NULL ){
        policy = dynamic_cast <FeedbackSolutionMDP *> (optimum2);
    }
    /* Now policy is of FeedbackSolutionMDP type */
    cout<<"Print solution by dimension (line by line)"<< endl;
    optimum2->WriteSolutionByDim(1,stateSpace);
    
    cout<<endl<<endl<<"################################"<<endl;
    
    
    
    cout<<"Printing State Space Path and value function"<<endl;
    /* initial state */
    stateSpace->FirstState(statebufferO);
    /* path */
    for(k=0; k<stateSpace->Cardinal();k++){
        /* getting the index of the state */
        indexO = stateSpace->Index(statebufferO);
        /* the different values of the states are in the array */
        r=statebufferO[0]; /* getting value of the first dimension of the box */
        n=statebufferO[1]; /* getting value of the second dimension of the box */
        cout<< "--request=" << r << " --nodes=" << n;
        /* getting the values and the action at the index of the state */
        cout<< " --Optimal action=" << policy->getActionIndex(indexO) << " --Optimal Value=" << policy->getValueIndex(indexO)  <<endl;
        /* Move to next state */
        stateSpace->NextState(statebufferO);
    }
    
    cout<<endl<<"********************************"<<endl;

    cout<<"Deleting"<<endl;
    
    mdpSSP->ClearRew();
    for(int i=2;i>=0;i--){
       cout<<"Deleting Matrix"<<endl; 
       mdpSSP->DeleMatrix(i); 
    }
    

    cout<<"Deleting 2 (MDP)"<<endl;
    delete mdpSSP;
    delete optimum2;
    
    /* delete buffer */
    delete[] statebufferO;
    delete[] statebufferD;

    return 0;
}