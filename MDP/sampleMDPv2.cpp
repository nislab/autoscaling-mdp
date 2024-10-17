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
 * @version 2
 * @date January 2024
 * 
 * Creating numerical solver for basic policy iteration of MDP modeling server with 2 available pods
 * and max traffic equal to 2
 * 
 * Version 1: hard code all aspects to validate against hand calculated solution
 * Version 2: define variables for transitions and costs
 */
int main( int argc, char** argv )
{
    
    string critere("max");
        
    //create and initialize epsilon.
    double epsilon = 0.000001;

    //create and initialize the maximum number of iterations allowed.
    int maxIter = 150;
    
    // define transition probabilities
    double lambda = 1;
    double mu = 0.75;
    double LAM = 1 + 2*0.75; // maximum transition rate, used for normalization

    // define costs
    double CA = -1; // activation cost
    double CD = -1; // deactiavtion cost
    double CS = -1.5; // cost of operating each node
    double CH = -2; // cost of holding each request
    double INF = -100000; // arbitrary cost for invalid action

    // Create MDP object
    // six states maping from (r,n) in 0<=r<=2, 1<=n<=2 to [1,6]
    int dimPB=6;
    MarmoteSet *stateSpace = new MarmoteInterval(1,6);
    // three actions, subtract node (0), maintain nodes (1), add node (2)
    MarmoteSet *actionSpace = new MarmoteInterval(0,2);    
    
    // use an iterator on the action space
    MarmoteState actionbuffer = actionSpace->StateBuffer();
    
    vector<TransitionStructure*> trans(actionSpace->Cardinal());
    
    // Fill in transtion probailities for each action
    // Action 0 - subtract node; invalid action if already at minimum number of nodes
    actionSpace->FirstState(actionbuffer);
    SparseMatrix *P0 = new SparseMatrix(dimPB); /* matrix for the a_11 action*/
    P0->addToEntry(3,1,lambda/LAM); // arrival
    P0->addToEntry(3,3,1-lambda/LAM); // pseudo-event self transition
    P0->addToEntry(4,0,mu/LAM); // departure
    P0->addToEntry(4,2,lambda/LAM); // arrival
    P0->addToEntry(4,4,1-(lambda+mu)/LAM); // pseudo-event self transition
    P0->addToEntry(5,1,mu/LAM); // departure
    P0->addToEntry(5,5,1-mu/LAM); // pseudo-event self transition
    trans.at(actionSpace->Index(actionbuffer)) = P0;

    // Action 1 - maintain nodes; always valid, creates B-D chain at each level
    actionSpace->NextState(actionbuffer);
    SparseMatrix *P1 = new SparseMatrix(dimPB);
    P1->addToEntry(0,1,lambda/LAM); // arrival
    P1->addToEntry(0,0,1-lambda/LAM); // pseduo-event self transition
    P1->addToEntry(1,0,mu/LAM); // departure
    P1->addToEntry(1,2,lambda/LAM); // arrival
    P1->addToEntry(1,1,1-(lambda+mu)/LAM); // pseudo-event self transition
    P1->addToEntry(2,1,mu/LAM); // departure
    P1->addToEntry(2,2,1-mu/LAM); // pseudo-event self transition
    P1->addToEntry(3,4,lambda/LAM); // arrival
    P1->addToEntry(3,3,1-lambda/LAM); // pseudo-event self transition
    P1->addToEntry(4,3,mu/LAM); // departure
    P1->addToEntry(4,5,lambda/LAM); // arrival
    P1->addToEntry(4,4,1-lambda/LAM); // pseudo-event self transition
    P1->addToEntry(5,4,(2*mu)/LAM); // departure
    P1->addToEntry(5,5,1-(2*mu)/LAM); // psuedo-event self transition   
    trans.at(actionSpace->Index(actionbuffer)) = P1;
    
    // Action 2 - add node; invalid if already at max nodes
    actionSpace->NextState(actionbuffer);
    SparseMatrix *P2 = new SparseMatrix(dimPB);
    P2->addToEntry(0,4,lambda/LAM); // arrival
    P2->addToEntry(0,0,1-lambda/LAM); // pseduo-event self transition
    P2->addToEntry(1,3,mu/LAM); // departure
    P2->addToEntry(1,5,lambda/LAM); // arrival
    P2->addToEntry(1,1,1-(mu+lambda)/LAM); // pseudo-event self transition
    P2->addToEntry(2,4,(2*mu)/LAM); // departure
    P2->addToEntry(2,2,1-(2*mu)/LAM); // pseudo-event self transition
    trans.at(actionSpace->Index(actionbuffer)) = P2;
    
    // define rewards; done in terms of running costs, activation costs, deactivation costs, holding costs
    // arbitrary large cost defined for invalid actions
    SparseMatrix *Reward  = new SparseMatrix(dimPB,actionSpace->Cardinal());
    Reward->addToEntry(0,0,INF);
    Reward->addToEntry(0,1,CS);
    Reward->addToEntry(0,2,2*CS+CA);
    
    Reward->addToEntry(1,0,INF);
    Reward->addToEntry(1,1,CS+CH);
    Reward->addToEntry(1,2,2*CS+CA+CH);
    
    Reward->addToEntry(2,0,INF);
    Reward->addToEntry(2,1,CS+2*CH);
    Reward->addToEntry(2,2,2*CS+CA+2*CH);

    Reward->addToEntry(3,0,CS+CD);
    Reward->addToEntry(3,1,2*CS);
    Reward->addToEntry(3,2,INF);

    Reward->addToEntry(4,0,CS+CD+CH);
    Reward->addToEntry(4,1,2*CS+CH);
    Reward->addToEntry(4,2,INF);

    Reward->addToEntry(5,0,CS+CD+2*CH);
    Reward->addToEntry(5,1,2*CS+2*CH);
    Reward->addToEntry(5,2,INF);
        
    cout << "Size :\t" << stateSpace->Cardinal() << std::endl;
    cout << "Begining building of MDP" << std::endl;
    AverageMDP *mdp = new AverageMDP(critere, stateSpace, actionSpace, trans,Reward);
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


