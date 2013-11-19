#include <map>
#include <tuple>
#include <time.h>
#include <stdlib.h>
#include <cmath>

std::map<std::tuple<int, int>, double> getDistribution(int c_x, int c_y, float stdDev, int width, int height){
    //c_x and c_y are means, q_x and q_y are queries

}

std::tuple<int, int> sampleDistr(std::map<std::tuple<int, int>, double> *distr){
    srand(time(NULL));
    double r = (double)rand() / (double)RAND_MAX;
    typedef std::map<std::tuple<int, int>, double>::iterator it_type;
    for(it_type iterator = distr.begin(); iterator != distr.end(); iterator++) {
        if(r < iterator->second){
            return iterator->first;
        }
        r = r - iterator->second;
    }
}

void initializeUniformly(int numParticles, int width, int height, std::tuple<int, int> *particles){ //particles is already loaded with zeros
	srand(time(NULL));
	for (int i = 0; i < numParticles; i++){
		x = rand() % width;
        y = rand() % height;
		particles[i] = std::make_tuple(x,y);
	}
}

void observe(int* particles, std::map<std::tuple<int, int>, double> &edistr){
    /*
	noisyDistance = observation
    emissionModel = busters.getObservationDistribution(noisyDistance)
    pacmanPosition = gameState.getPacmanPosition()

    beliefs = util.Counter()
    for p in self.particles:
        trueDistance = util.manhattanDistance(p, pacmanPosition)
        beliefs[p] += emissionModel[trueDistance]
    beliefs.normalize()
    if beliefs.totalCount() == 0:
        self.initializeUniformly(gameState)
        beliefs = self.getBeliefDistribution()
    self.particles = []
    for i in range(self.numParticles):
        self.particles += [util.sample(beliefs)]
    return self.getBeliefDistribution()
    */
}

void elapseTime(std::tuple<int, int> *oldParticles, int numParticles, int width, int height){
    std::tuple<int, int> newParticles[numParticles];
    std::tuple<int, int> particle;
    for(int i = 0; i < numParticles; i++){
        particle = oldParticles[i];
        distr = getDistribution(std::get<0>(particle), std::get<1>(particle), 1.0f, width, height);
        newParticles[i] = sampleDistr(&distr);
    }
    self.particles = newParticles
}


void parFilterMatrix(int* particles, std::map<std::tuple<int, int>, double> &edistr, int width, int height){
	/*int numParticles = height*width/4;
	int particles[numParticles] = {0};
	initializeUniformly(numParticles, particles);
	*/
}


