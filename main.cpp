/*

Snake end value training with Monte Carlo Tree Search.
Uses MARL framework.

*/

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <ctime>

//environment details

#define boardx 6
#define boardy 6
#define maxTime 180

#define numAgentActions 4
#define numChanceActions (boardx*boardy)
#define maxNumActions (boardx*boardy)

//network details

#define numlayers 2
#define maxNodes 60
#define startingParameterRange 0.01

//training deatils

#define learnRate 0.01
#define momentum 0.9
#define regRate 0.0001
#define scoreNorm 5
#define batchSize 2000
#define numBatches 2
#define queueSize 40000

#define exploitationMultiplier 2 // unused
#define exploitationChange 0.001 // unused
#define numGames 1501
#define numPaths 120
#define maxStates (maxTime*2*numPaths)
#define evalPeriod 100
#define numEvalGames 20
#define evalZscore 2

using namespace std;

ofstream fout ("snakeTree.out");
ifstream netIn ("snakeTree_net.in");
ofstream netOut ("snakeTree_net.out");

int numActions[2] = {numAgentActions, numChanceActions};

int numNodes[numlayers+1] = {45, 60, 1};

class Agent{
public:
    double weights[numlayers][maxNodes][maxNodes];
    double bias[numlayers][maxNodes];
    double activation[numlayers+1][maxNodes];
    double inter[numlayers][maxNodes];
    double output;
    
    double expectedOutput;
    double Dbias[numlayers][maxNodes];
    double Dactivation[numlayers+1][maxNodes];
    double Sbias[numlayers][maxNodes];
    double Sweights[numlayers][maxNodes][maxNodes];
    
    void randomize(){
        int l,i,j;
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l]; i++){
                for(j=0; j<numNodes[l+1]; j++){
                    weights[l][i][j] = randVal();
                }
            }
        }
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l+1]; i++){
                bias[l][i] = randVal();
            }
        }
    }
    
    void copy(Agent* net){
        int l,i,j;
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l]; i++){
                for(j=0; j<numNodes[l+1]; j++){
                    weights[l][i][j] = net->weights[l][i][j];
                }
            }
        }
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l+1]; i++){
                bias[l][i] = net->bias[l][i];
            }
        }
    }
    
    void pass(){
        
        int l,i,j;
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l+1]; i++){
                inter[l][i] = bias[l][i];
                for(j=0; j<numNodes[l]; j++){
                    inter[l][i] += weights[l][j][i] * activation[l][j];
                }
                activation[l+1][i] = nonlinear(inter[l][i], l);
            }
        }
        output = activation[numlayers][0];
        
    }
    
    void resetGrad(){
        int l,i,j;
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l]; i++){
                for(j=0; j<numNodes[l+1]; j++){
                    Sweights[l][i][j] = 0;
                }
            }
        }
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l+1]; i++){
                Sbias[l][i] = 0;
            }
        }
    }
    
    void backProp(double weight){ //weight=1 means default learn rate
        double mult = weight * learnRate / batchSize;
        pass();
        int l,i,j;
        Dactivation[numlayers][0] = 2 * (output - expectedOutput);
        for(l=numlayers-1; l>=0; l--){
            for(i=0; i<numNodes[l+1]; i++){
                Dbias[l][i] = Dactivation[l+1][i] * dnonlinear(inter[l][i], l);
                Sbias[l][i] -= Dbias[l][i] * mult;
            }
            for(i=0; i<numNodes[l]; i++){
                Dactivation[l][i] = 0;
                for(j=0; j<numNodes[l+1]; j++){
                    Dactivation[l][i] += Dbias[l][j] * weights[l][i][j];
                }
            }
            for(i=0; i<numNodes[l+1]; i++){
                for(j=0; j<numNodes[l]; j++){
                    Sweights[l][j][i] -= Dbias[l][i] * activation[l][j] * mult;
                }
            }
        }
    }
    
    void train(){
        int l,i,j;
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l]; i++){
                for(j=0; j<numNodes[l+1]; j++){
                    weights[l][i][j] += Sweights[l][i][j];
                    weights[l][i][j] *= 1 - regRate;
                    Sweights[l][i][j] *= momentum;
                }
            }
        }
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l+1]; i++){
                bias[l][i] += Sbias[l][i];
                bias[l][i] *= 1 - regRate;
                Sbias[l][i] *= momentum;
            }
        }
    }
    
    void saveNet(){
        int l,i,j;
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l]; i++){
                for(j=0; j<numNodes[l+1]; j++){
                    netOut<<weights[l][i][j]<<' ';
                }
                netOut<<'\n';
            }
            netOut<<'\n';
        }
        netOut<<'\n';
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l+1]; i++){
                netOut<<bias[l][i]<<' ';
            }
            netOut<<'\n';
        }
    }
    
    void readNet(){
        int l,i,j;
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l]; i++){
                for(j=0; j<numNodes[l+1]; j++){
                    netIn>>weights[l][i][j];
                }
            }
        }
        for(l=0; l<numlayers; l++){
            for(i=0; i<numNodes[l+1]; i++){
                netIn>>bias[l][i];
            }
        }
    }
    
private:
    double randVal(){
        return (((double)rand() / RAND_MAX)*2-1) * startingParameterRange;
    }
    /*
    double nonlinear(double x){
        return 1/(1+exp(-x));
    }
    
    double dnonlinear(double x){
        return nonlinear(x) * (1-nonlinear(x));
    }
    */
    
    double nonlinear(double x, int l){
        if(x>0 || l==numlayers-1) return x;
        return 0;
    }
    
    double dnonlinear(double x, int l){
        if(x>0 || l==numlayers-1) return 1;
        return 0;
    }
};

int dir[4][2] = {{0,1}, {1,0}, {0,-1}, {-1,0}};

class Environment{
public:
    int timer;
    double score;
    int actionType; // 0 = action state, 1 = reaction state.
    
    int snakeSize;
    int headx,heady;
    int tailx, taily;
    int applex,appley;
    int snake[boardx][boardy]; // -1 = not snake. 0 to 3 = snake unit pointing to next unit. 4 = head.
    
    void initialize(){
        timer = 0;
        score = 0;
        actionType = 0;
        snakeSize = 2;
        headx = boardx/2;
        heady = 2;
        tailx = headx;
        taily = 1;
        
        int i,j;
        for(i=0; i<boardx; i++){
            for(j=0; j<boardy; j++){
               snake[i][j] = -1;
            }
        }
        snake[headx][heady] = 4;
        snake[tailx][taily] = 0;
        
        while(true){
            applex = rand()%boardx;
            appley = rand()%boardy;
            if(snake[applex][appley] == -1){
                break;
            }
        }
    }
    
    bool isEndState(){
        if(timer == maxTime){
            return true;
        }
        int newx,newy;
        for(int d=0; d<4; d++){
            newx = headx + dir[d][0];
            newy = heady + dir[d][1];
            if(newx == -1 || newx == boardx){
                continue;
            }
            if(newy == -1 || newy == boardy){
                continue;
            }
            if(snake[newx][newy] == -1){
                return false;
            }
        }
        return true;
    }
    
    bool validAction(int actionIndex){ // returns whether the action is valid.
        if(actionType == 0){
            return validAgentAction(actionIndex);
        }
        else{
            return validChanceAction(actionIndex);
        }
    }
    
    bool validAgentAction(int d){
        int newHeadx = headx + dir[d][0];
        int newHeady = heady + dir[d][1];
        if(newHeadx == -1 || newHeadx == boardx){
            return false;
        }
        if(newHeady == -1 || newHeady == boardy){
            return false;
        }
        return snake[newHeadx][newHeady] == -1;
    }
    
    bool validChanceAction(int pos){
        int newApplex = pos / boardy;
        int newAppley = pos % boardy;
        return snake[newApplex][newAppley] == -1;
    }
    
    void setAction(Environment* currState, int actionIndex){
        if(currState->actionType == 0){
            setAgentAction(currState, actionIndex);
        }
        if(currState->actionType == 1){
            setChanceAction(currState, actionIndex);
        }
    }
    
    void setAgentAction(Environment* currState, int actionIndex){
        timer = currState->timer + 1;
        score = currState->score;
        actionType = 0;
        snakeSize = currState->snakeSize;
        headx = currState->headx;
        heady = currState->heady;
        tailx = currState->tailx;
        taily = currState->taily;
        applex = currState->applex;
        appley = currState->appley;
        int i,j;
        for(i=0; i<boardx; i++){
            for(j=0; j<boardy; j++){
                snake[i][j] = currState->snake[i][j];
            }
        }
        
        int newHeadx = headx + dir[actionIndex][0];
        int newHeady = heady + dir[actionIndex][1];
        snake[headx][heady] = actionIndex;
        headx = newHeadx;
        heady = newHeady;
        snake[newHeadx][newHeady] = 4;
        
        if(headx == applex && heady == appley){
            score += 1;
            //score += 1 - timer*0.5/maxTime;
            snakeSize++;
            actionType = 1;
        }
        else{
            int tailDir = snake[tailx][taily];
            snake[tailx][taily] = -1;
            tailx += dir[tailDir][0];
            taily += dir[tailDir][1];
        }
    }
    
    void setChanceAction(Environment* currState, int actionIndex){
        timer = currState->timer;
        score = currState->score;
        actionType = 0;
        snakeSize = currState->snakeSize;
        headx = currState->headx;
        heady = currState->heady;
        tailx = currState->tailx;
        taily = currState->taily;
        
        applex = actionIndex / boardy;
        appley = actionIndex % boardy;
        
        int i,j;
        for(i=0; i<boardx; i++){
            for(j=0; j<boardy; j++){
               snake[i][j] = currState->snake[i][j];
            }
        }
    }
    
    void inputAgent(Agent* a){
        a->activation[0][0] = (double) timer / maxTime;
        a->activation[0][1] = score / scoreNorm;
        a->activation[0][2] = actionType;
        a->activation[0][3] = headx;
        a->activation[0][4] = heady;
        a->activation[0][5] = tailx;
        a->activation[0][6] = taily;
        a->activation[0][7] = applex;
        a->activation[0][8] = appley;
        int i,j;
        for(i=0; i<boardx; i++){
            for(j=0; j<boardy; j++){
                a->activation[0][9 + (i*boardy + j)] = snake[i][j];
            }
        }
    }
    
    void inputSymmetric(Agent* a, int t){
        int m = boardx-1;
        int sym[8][2][3] = {
            {{ 1, 0, 0},{ 0, 1, 0}},
            {{ 0,-1, m},{ 1, 0, 0}},
            {{-1, 0, m},{ 0,-1, m}},
            {{ 0, 1, 0},{-1, 0, m}},
            {{ 0, 1, 0},{ 1, 0, 0}},
            {{ 1, 0, 0},{ 0,-1, m}},
            {{ 0,-1, m},{-1, 0, m}},
            {{-1, 0, m},{ 0, 1, 0}}
        };
        int symDir[8][2] = {
            { 1,0},
            { 1,3},
            { 1,2},
            { 1,1},
            {-1,1},
            {-1,2},
            {-1,3},
            {-1,0}
        };
        a->activation[0][0] = (double) timer / maxTime;
        a->activation[0][1] = score / scoreNorm;
        a->activation[0][2] = actionType;
        a->activation[0][3] = sym[t][0][0]*headx + sym[t][0][1]*heady + sym[t][0][2];
        a->activation[0][4] = sym[t][1][0]*headx + sym[t][1][1]*heady + sym[t][1][2];
        a->activation[0][5] = sym[t][0][0]*tailx + sym[t][0][1]*taily + sym[t][0][2];
        a->activation[0][6] = sym[t][1][0]*tailx + sym[t][1][1]*taily + sym[t][1][2];
        a->activation[0][7] = sym[t][0][0]*applex + sym[t][0][1]*appley + sym[t][0][2];
        a->activation[0][8] = sym[t][1][0]*applex + sym[t][1][1]*appley + sym[t][1][2];
        
        int i,j,x,y;
        for(i=0; i<boardx; i++){
            for(j=0; j<boardy; j++){
                x = sym[t][0][0]*i + sym[t][0][1]*j + sym[t][0][2];
                y = sym[t][1][0]*i + sym[t][1][1]*j + sym[t][1][2];
                if(snake[i][j] == -1 || snake[i][j] == 4){
                    a->activation[0][9 + (x*boardy + y)] = snake[i][j];
                }
                else{
                    a->activation[0][9 + (x*boardy + y)] = (symDir[t][0]*snake[i][j] + symDir[t][1] + 4) % 4;
                }
            }
        }
    }
    
    void copyEnv(Environment* e){
        timer = e->timer;
        score = e->score;
        actionType = e->actionType;
        snakeSize = e->snakeSize;
        headx = e->headx;
        heady = e->heady;
        tailx = e->tailx;
        taily = e->taily;
        applex = e->applex;
        appley = e->appley;
        int i,j;
        for(i=0; i<boardx; i++){
            for(j=0; j<boardy; j++){
                snake[i][j] = e->snake[i][j];
            }
        }
    }
    
    void print(){ // optional function for debugging
        fout<<"Timer: "<<timer<<'\n';
        fout<<"Score: "<<score<<'\n';
        fout<<"Action type: "<<actionType<<'\n';
        fout<<"Snake size: "<<snakeSize<<'\n';
        int i,j;
        for(i=0; i<boardx; i++){
            for(j=0; j<boardy; j++){
                if(i == applex && j == appley){
                    fout<<'A'<<' ';
                }
                else{
                    if(snake[i][j] == -1){
                        fout<<". ";
                    }
                    else{
                        fout<<snake[i][j]<<' ';
                    }
                }
            }
            fout<<'\n';
        }
    }
};

class Data{
public:
    Environment e;
    double expectedValue;
    
    Data(Environment* givenEnv, double givenExpected){
        e.copyEnv(givenEnv);
        expectedValue = givenExpected;
    }
    
    void trainAgent(Agent* a){
        e.inputSymmetric(a, rand()%8);
        a->expectedOutput = expectedValue;
        a->backProp(1);
    }
};

class DataQueue{
public:
    Data* queue[queueSize];
    int index;
    
    DataQueue(){
        index = 0;
    }
    
    void enqueue(Data* d){
        queue[index%queueSize] = d;
        index++;
    }
    
    void trainAgent(Agent* a){
        int i,j;
        for(i=0; i<numBatches; i++){
            for(j=0; j<batchSize; j++){
                queue[rand() % min(index,queueSize)]->trainAgent(a);
            }
            a->train();
        }
    }
};

double squ(double x){
    return x*x;
}

Environment states[maxStates];
DataQueue dq;

class Trainer{
public:
    Agent a;
    
    double exploitationFactor;
    
    Trainer(){
        a.randomize();
        a.resetGrad();
        exploitationFactor = 1;
    }
    
    //Storage for the tree:
    int outcomes[maxStates][maxNumActions];
    int size[maxStates];
    double sumScore[maxStates];
    int index;
    
    //For executing a training iteration:
    int roots[maxStates];
    int currRoot;
    double actionProbs[numAgentActions];
    
    void initializeNode(int currNode){
        for(int i=0; i<numActions[states[currNode].actionType]; i++){
            if(!states[currNode].validAction(i)){
                outcomes[currNode][i] = -2;
            }
            else{
                outcomes[currNode][i] = -1;
            }
        }
        size[currNode] = 0;
        sumScore[currNode] = 0;
    }
    
    void trainTree(){
        states[0].initialize();
        initializeNode(0);
        currRoot = 0;
        roots[0] = 0;
        index = 1;
        
        int s = 0;
        int chosenAction;
        int i;
        while(!states[currRoot].isEndState()){
            if(states[currRoot].actionType == 0){
                for(i=0; i<numPaths; i++){
                    expandPath();
                }
                computeActionProbs();
                chosenAction = sampleActionProbs();
            }
            if(states[currRoot].actionType == 1){
                chosenAction = getRandomChanceAction(&states[currRoot]);
                if(outcomes[currRoot][chosenAction] == -1){
                    outcomes[currRoot][chosenAction] = index;
                    states[index].setAction(&states[currRoot], chosenAction);
                    initializeNode(index);
                    index++;
                }
            }
            currRoot = outcomes[currRoot][chosenAction];
            s++;
            roots[s] = currRoot;
        }
        int numStates = s+1;
        double finalScore = states[currRoot].score;
        Data* newData;
        for(i=0; i<numStates; i++){
            newData = new Data(&states[roots[i]], finalScore);
            dq.enqueue(newData);
        }
        dq.trainAgent(&a);
        
        exploitationFactor = exploitationChange * (finalScore * exploitationMultiplier) + (1-exploitationChange) * exploitationFactor;
    }
    
    int evalGame(){ // return index of the final state in states.
        states[0].initialize();
        initializeNode(0);
        currRoot = 0;
        roots[0] = 0;
        index = 1;
        
        int s = 0;
        int chosenAction;
        int i;
        while(!states[currRoot].isEndState()){
            if(states[currRoot].actionType == 0){
                for(i=0; i<numPaths; i++){
                    expandPath();
                }
                computeActionProbs();
                chosenAction = sampleActionProbs();
            }
            if(states[currRoot].actionType == 1){
                chosenAction = getRandomChanceAction(&states[currRoot]);
                if(outcomes[currRoot][chosenAction] == -1){
                    outcomes[currRoot][chosenAction] = index;
                    states[index].setAction(&states[currRoot], chosenAction);
                    initializeNode(index);
                    index++;
                }
            }
            currRoot = outcomes[currRoot][chosenAction];
            s++;
            roots[s] = currRoot;
        }
        return currRoot;
    }
    
    void printGame(){
        states[0].initialize();
        initializeNode(0);
        currRoot = 0;
        roots[0] = 0;
        index = 1;
        
        int s = 0;
        int chosenAction;
        int i;
        while(!states[currRoot].isEndState()){
            states[currRoot].print();
            if(states[currRoot].actionType == 0){
                for(i=0; i<numPaths; i++){
                    expandPath();
                }
                computeActionProbs();
                fout<<"Action probabilities: ";
                for(i=0; i<numAgentActions; i++){
                    fout<<actionProbs[i]<<' ';
                }
                fout<<"\n\n";
                chosenAction = optActionProbs();
            }
            if(states[currRoot].actionType == 1){
                chosenAction = getRandomChanceAction(&states[currRoot]);
                if(outcomes[currRoot][chosenAction] == -1){
                    outcomes[currRoot][chosenAction] = index;
                    states[index].setAction(&states[currRoot], chosenAction);
                    initializeNode(index);
                    index++;
                }
            }
            currRoot = outcomes[currRoot][chosenAction];
            s++;
            roots[s] = currRoot;
        }
    }
    
    double evaluate(){
        int endState;
        double scoreSum = 0;
        double sizeSum = 0;
        double scoreSquareSum = 0;
        for(int i=0; i<numEvalGames; i++){
            endState = evalGame();
            scoreSum += states[endState].score;
            sizeSum += states[endState].snakeSize;
            scoreSquareSum += squ(states[endState].score);
        }
        double averageScore = scoreSum / numEvalGames;
        double variance = scoreSquareSum / numEvalGames - squ(averageScore);
        double SE = sqrt(variance / numEvalGames) * evalZscore;
        cout<<"Average snake size: "<<(sizeSum/numEvalGames)<<'\n';
        cout<<"Average score: "<<averageScore<<'\n';
        cout<<"Confidence interval: (" << (averageScore - SE) << ", " << (averageScore + SE) << ")\n";
        //exploitationFactor = averageScore * exploitationMultiplier;
        cout<<"Exploitation factor is now: "<<exploitationFactor<<'\n';
        cout<<'\n';
        return averageScore;
    }
    
    int path[maxStates];
    
    void expandPath(){
        //fout<<"New path at "<<currRoot<<'\n';
        int currNode = currRoot;
        int nextNode,nextAction;
        int count = 0;
        int currType;
        int maxIndex;
        double maxVal,candVal;
        int i;
        while(currNode != -1 && !states[currNode].isEndState()){
            //fout<<"Checking state:\n";
            //states[currNode].print();
            path[count] = currNode;
            count++;
            maxVal = -1000000;
            currType = states[currNode].actionType;
            //fout<<"Checking actions:\n";
            for(i=0; i<numActions[currType]; i++){
                nextNode = outcomes[currNode][i];
                if(nextNode == -2){
                    //fout<<"No ";
                    continue;
                }
                if(nextNode == -1){
                    candVal = 1000 + (double)rand() / RAND_MAX;
                }
                else{
                    if(currType == 0){
                        candVal = sumScore[nextNode] / size[nextNode] + 0.5 * log(size[currNode]) / sqrt(size[nextNode]);
                    }
                    if(currType == 1){
                        candVal = (double)rand() / RAND_MAX - size[nextNode];
                    }
                }
                //fout<<candVal<<' ';
                if(candVal > maxVal){
                    maxVal = candVal;
                    maxIndex = i;
                }
            }
            //fout<<"Best Action: "<<maxIndex<<"\n\n";
            nextAction = maxIndex;
            currNode = outcomes[currNode][maxIndex];
        }
        double newVal;
        if(currNode == -1){
            outcomes[path[count-1]][nextAction] = index;
            states[index].setAction(&states[path[count-1]], nextAction);
            initializeNode(index);
            //fout<<"New state:\n";
            //states[index].print();
            states[index].inputSymmetric(&a, rand()%8);
            a.pass();
            newVal = a.output;
            path[count] = index;
            index++;
            count++;
            /*
            Attempt at voiding very bad scores. Crashes.
            if(size[currRoot] > 0 && newVal < (sumScore[currRoot] / size[currRoot]) - 2){
                cout<<sumScore[currRoot] <<' '<< size[currRoot]<<'\n';
                outcomes[path[count-1]][nextAction] = -2;
                return;
            }
            else{
                outcomes[path[count-1]][nextAction] = index;
                path[count] = index;
                index++;
                count++;
            }
            */
        }
        else{
            newVal = states[currNode].score;
            path[count] = currNode;
            count++;
        }
        //fout<<"Evaluated at "<<newVal<<'\n';
        for(i=0; i<count; i++){
            size[path[i]]++;
            sumScore[path[i]] += newVal;
        }
    }
    
    void printTree(){
        for(int i=0; i<index; i++){
            fout<<"State "<<i<<'\n';
            states[i].print();
            fout<<"Outcomes: ";
            for(int j=0; j<numActions[states[i].actionType]; j++){
                fout<<outcomes[i][j];
            }
            fout<<'\n';
            fout<<"Size: "<<size[i]<<'\n';
            fout<<"Sum score: "<<sumScore[i]<<'\n';
            fout<<'\n';
        }
    }
    
    void computeActionProbs(){
        int i;
        int nextIndex;
        for(i=0; i<numAgentActions; i++){
            nextIndex = outcomes[currRoot][i];
            if(nextIndex != -2){
                actionProbs[i] = squ(size[nextIndex]);
            }
            else{
                actionProbs[i] = -1;
            }
        }
    }
    
    int optActionProbs(){
        int i;
        int maxIndex = 0;
        for(i=1; i<numAgentActions; i++){
            if(actionProbs[i] > actionProbs[maxIndex]){
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    int sampleActionProbs(){
        double propSum = 0;
        int i;
        for(i=0; i<numAgentActions; i++){
            if(actionProbs[i] == -1){
                continue;
            }
            propSum += actionProbs[i];
        }
        double parsum = 0;
        double randReal = (double)rand() / RAND_MAX * propSum;
        
        int actionIndex = -1;
        for(i=0; i<numAgentActions; i++){
            if(actionProbs[i] == -1){
                continue;
            }
            parsum += actionProbs[i];
            if(randReal <= parsum){
                actionIndex = i;
                break;
            }
        }
        return actionIndex;
    }
    
    int getRandomChanceAction(Environment* e){
        int i;
        int possibleActions[numChanceActions];
        int numPossibleActions = 0;
        for(i=0; i<numChanceActions; i++){
            if(e->validAction(i)){
                possibleActions[numPossibleActions] = i;
                numPossibleActions++;
            }
        }
        return possibleActions[rand() % numPossibleActions];
    }
};


int main()
{
    srand((unsigned)time(NULL));
    
    Trainer t;
    //t.evaluate();
    
    for(int i=0; i<numGames; i++){
        if(i%evalPeriod == 0){
            cout<<"Game "<<i<<'\n';
            t.evaluate();
        }
        t.trainTree();
    }
    for(int i=0; i<10; i++){
        fout<<"Printed game "<<i<<'\n';
        t.printGame();
    }
    
    t.a.saveNet();
    
    return 0;
}



