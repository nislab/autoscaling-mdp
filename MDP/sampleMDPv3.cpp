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


#include "marmoteCore/marmoteInterval.h"
#include "marmoteCore/marmoteSparseMatrix.h"
#include "marmoteCore/marmoteBox.h"
#include "marmoteMDP/marmoteAverageMDP.h"
#include "marmoteMDP/marmoteFeedbackSolutionMDP.h"
#include "marmoteMDP/marmoteSolutionMDP.h"

#include <list>
#include <vector>
#include <string>

using namespace std;
/**
 * Class to impliment example of MDP for autoscaling with 2 servers and no buffer excess
 * @author Chamberlain, Jonathan
 * @version 3
 * @date January 2024
 * 
 * Creating numerical solver for basic policy iteration of MDP modeling server with 2 available pods
 * and max traffic equal to 2
 * 
 * Version 1: hard code all aspects to validate against hand calculated solution
 * Version 2: define variables for transitions and costs
 * Version 3: define Marmote Box for state spaces
 */
int main( int argc, char** argv )
{
    
    string critere("max");
        
    //create and initialize epsilon.
    double epsilon = 0.000001;

    //create and initialize the maximum number of iterations allowed.
    int maxIter = 150;
    
    // define transition rates
    double lambda = 1;
    double mu = 0.75;
    double LAM = 1 + 2*0.75; // maximum transition rate, used for normalization
    double p,q; // probabilities of arrivals, departures

    // define costs
    double CA = -1; // activation cost
    double CD = -1; // deactiavtion cost
    double CS = -1.5; // cost of operating each node
    double CH = -2; // cost of holding each request
    double INF = -100000; // arbitrary cost for invalid action

    // Create MDP object
    
    // define parameters for states to iterate on with MarmoteBox
    int r, n; // indices indicating number of requests, number of active nodes
    int B = 3; // maximum buffer for number of requests
    int N = 3; // maximum number of nodes
    stateType indexO; // index of current/origin state 
    stateType indexD; // index of destination state 
    stateType k; // ordinal indicies of states
    // create box such that indcies k map to states of type (r,n)
    stateType dims[2]={B+1,N}; // c++ is zero indexed.
    MarmoteBox *stateSpace = new MarmoteBox(2,dims);

    // three actions, subtract node (0), maintain nodes (1), add node (2)
    MarmoteSet *actionSpace = new MarmoteInterval(0,2);    
    
   
     // compute size of cardinals for state, action spaces
    stateType dim_SS = stateSpace->Cardinal();
    stateType dim_AS = actionSpace->Cardinal();
    // Allocate buffers for state spaces
    // manage origin/pre-transition state
    MarmoteState statebufferO = stateSpace->StateBuffer();
    // manage destination/post-transition state
    MarmoteState statebufferD = stateSpace->StateBuffer();

    // use an iterator on the action space
    MarmoteState actionbuffer = actionSpace->StateBuffer();
    
    vector<TransitionStructure*> trans(dim_AS);
    
    // Fill in transtion probailities for each action

    // Action 0 - subtract node; invalid action if already at minimum number of nodes
    actionSpace->FirstState(actionbuffer);
    SparseMatrix *P0 = new SparseMatrix(dim_SS); 
    for(r = 0; r <= B ; r++) {
        statebufferO[0]=r; // initialize the value request dimension
        for (n = 1; n < N; n++) {
            // skip index n=0 as cannot subtract nodes when at minimum
            statebufferO[1]=n;
            indexO=stateSpace->Index(statebufferO);
            if (r < B) {
                // arrival case
                p = lambda/LAM;
                statebufferD[0] = r+1;
                statebufferD[1] = n-1;
                indexD = stateSpace->Index(statebufferD);
                P0->addToEntry(indexO,indexD,p);
            } else {
                p = 0; // no arrival possible, handle self transition appropriately
            }
            if (r > 0) {
                // depature case
                // service rate depends on requests, number of nodes after subtracting; number of nodes off by 1 due to indexing
                q = mu*std::min(r,n)/LAM;  
                statebufferD[0] = r-1;
                statebufferD[1] = n-1;
                indexD = stateSpace->Index(statebufferD);
                P0->addToEntry(indexO,indexD,q);
            } else {
                q = 0; // no departure possible, handle self transition appropriately
            }
            // psuedo-event self transition
            P0->addToEntry(indexO,indexO,1-p-q);
        }
    }
    trans.at(actionSpace->Index(actionbuffer)) = P0;

    // Action 1 - maintain nodes; always valid, creates B-D chain at each level
    actionSpace->NextState(actionbuffer);
    SparseMatrix *P1 = new SparseMatrix(dim_SS);
     for(r = 0; r <= B ; r++) {
        statebufferO[0]=r; // initialize the value request dimension
        for (n = 0; n < N; n++) {
            statebufferO[1]=n;
            indexO=stateSpace->Index(statebufferO);
            if (r < B) {
                // arrival case
                p = lambda/LAM;
                statebufferD[0] = r+1;
                statebufferD[1] = n;
                indexD = stateSpace->Index(statebufferD);
                P1->addToEntry(indexO,indexD,p);
            } else {
                p = 0; // no arrival possible, handle self transition appropriately
            }
            if (r > 0) {
                // depature case
                // service rate depends on requests, number of nodes; number of nodes off by 1 due to indexing
                q = mu*std::min(r,n+1)/LAM; 
                statebufferD[0] = r-1;
                statebufferD[1] = n;
                indexD = stateSpace->Index(statebufferD);
                P1->addToEntry(indexO,indexD,q);
            } else {
                q = 0; // no departure possible, handle self transition appropriately
            }
            // psuedo-event self transition
            P1->addToEntry(indexO,indexO,1-p-q);
        }
    }  
    trans.at(actionSpace->Index(actionbuffer)) = P1;
    
    // Action 2 - add node; invalid if already at max nodes
    actionSpace->NextState(actionbuffer);
    SparseMatrix *P2 = new SparseMatrix(dim_SS);
     for(r = 0; r <= B ; r++) {
        statebufferO[0]=r; // initialize the value request dimension
        for (n = 0; n < N-1; n++) {
            // skip index n=N-1 as cannot add nodes when at maximum
            statebufferO[1]=n;
            indexO=stateSpace->Index(statebufferO);
            if (r < B) {
                // arrival case
                p = lambda/LAM;
                statebufferD[0] = r+1;
                statebufferD[1] = n+1;
                indexD = stateSpace->Index(statebufferD);
                P2->addToEntry(indexO,indexD,p);
            } else {
                p = 0; // no arrival possible, handle self transition appropriately
            }
            if (r > 0) {
                // depature case
                // service rate depends on requests, number of nodes after adding; number of nodes off by 1 due to indexing
                q = mu*std::min(r,n+2)/LAM; 
                statebufferD[0] = r-1;
                statebufferD[1] = n+1;
                indexD = stateSpace->Index(statebufferD);
                P2->addToEntry(indexO,indexD,q);
            } else {
                q = 0; // no departure possible, handle self transition appropriately
            }
            // psuedo-event self transition
            P2->addToEntry(indexO,indexO,1-p-q);
        }
    }
    trans.at(actionSpace->Index(actionbuffer)) = P2;
    
    // define rewards; done in terms of running costs, activation costs, deactivation costs, holding costs
    // arbitrary large cost defined for invalid actions

    // Reward = CA*(activation == true)+CD*(deactivation==true)+CS*(numactiveNodes)+CH*(numreqs)
    SparseMatrix *Reward  = new SparseMatrix(dim_SS,dim_AS);
    for (r = 0; r <= B; r++) {
        for (n = 0; n < N; n++) {
            statebufferO[0] = r;
            statebufferO[1] = n;
            indexO = stateSpace->Index(statebufferO);
            Reward->addToEntry(indexO,1,(n+1)*CS+r*CH);
            if (n > 0) {
                Reward->addToEntry(indexO,0,CS*n+CD+r*CH);
            } else {
                Reward->addToEntry(indexO,0,INF);
            }
            if (n < N-1) {
                Reward->addToEntry(indexO,2,CS*(n+2)+CA+r*CH);
            } else {
                Reward->addToEntry(indexO,2,INF);
            }
        }
    }

    cout << "Size :\t" << stateSpace->Cardinal() << std::endl;
    cout << "Begining building of MDP" << std::endl;
    AverageMDP *mdp = new AverageMDP(critere, stateSpace, actionSpace, trans, Reward);
    cout << "End of building MDP" << std::endl;

    cout << "Writing MDP" << endl;
    mdp->WriteMDP();
    
    cout << endl << "Writing solution modified policy iteration" << endl ;
    //call the function to solve the MDP.
    SolutionMDP *optimum = mdp->PolicyIterationModified(epsilon, maxIter,0.01,150);
    optimum->WriteSolution();
        
    cout <<endl << endl << "Checking solutions" << endl ;
    double *sol = mdp->PolicyCost(optimum,epsilon, maxIter);
    for(int i=0;i<stateSpace->Cardinal();i++){
           cout <<"i= " << i << "  sol= " << sol[i] << endl;   
    }
    
    cout<< endl << endl << "********************************" << endl ;
        
    cout << "Destruction"<< endl;
    mdp->ClearRew();
    delete mdp;
    delete optimum;
    
    cout << "Destruction 2" << endl;
    delete P0;
    delete P1;
    delete P2;

    delete[] sol;
    delete[] actionbuffer;

    return 0;
}


