#include <cstdlib>
#include <iostream>
#include <tuple>
#include <vector>
#include <map>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <sys/time.h>
#include <trng/yarn5.hpp>
#include <trng/discrete_dist.hpp>
#include <trng/truncated_normal_dist.hpp>
#include <getopt.h>
#include <omp.h>

#include "tracker.h"
#include "parFilter.h"
#include "ImageHelper.hpp"

void ParticleFilter::initializeUniformly(){
    int x,y;
    srand(time(NULL));
    for (int i = 0; i < numParticles; i++){
        x = (int) (rand() % width);
        y = (int) (rand() % height);
        particles[i] = std::make_tuple(x,y); //initialize velocities to zero
    }
}

void ParticleFilter::telapse(std::tuple<int,int> *oldParticle) {
    /* given an old particle, the dimensions of the frame, and the std deviation,
    returns a new particle sampled from a 2D truncated normal distribution. */ 

    //init and unpack variables
    int x, y;
    std::tie(x, y) = *oldParticle;

    //init distributions
    trng::truncated_normal_dist<> X(std::max(std::min(x,width),0), sigma, 0, (float) width - 1);
    trng::truncated_normal_dist<> Y(std::max(std::min(y,height),0), sigma, 0, (float) height - 1);

    //replace old particle
    *oldParticle = std::make_tuple((int) X(r), (int) Y(r));
}

void ParticleFilter::observe(){
	/* Uses edistr matrix to weight particles, then resamples using discrete
     * distribution according to (#particles * weight). If total weight is
     * zero, the particles are uniformly distributed */
    std::map<std::tuple<int, int>, double> beliefs;
    std::tuple<int,int> t;
    double frameGivenPos, total;

    #if TIME
        timeval start, end;
        gettimeofday(&start, 0);
    #endif

    for(int i = 0; i < numParticles; i++){
        t = particles[i];
        frameGivenPos = imageHelper->similarity(std::get<0>(t), std::get<1>(t));
        beliefs[std::make_tuple(std::get<0>(t), std::get<1>(t))] += frameGivenPos;
        total += frameGivenPos;
    }

    #if TIME
        gettimeofday(&end, 0);
        timer.timeToComputeDistances += end.tv_sec + 1e-6*end.tv_usec - start.tv_sec - 1e-6*start.tv_usec;
    #endif

    // If all the weights are zero (i.e. no location fulfills evidence)
    if(total == 0){
        if(verbose){std::cout << "Reinitializing..." << std::endl;};
        initializeUniformly();
        for(int i = 0; i < numParticles; i++){
            t = particles[i];
            beliefs[std::make_tuple(std::get<0>(t), std::get<1>(t))] += 1;
        }
        total = numParticles;
    }

    int size = (int)beliefs.size();
    std::vector<double> p;  // stores relative probabilities
    std::tuple<int,int> *locations = new std::tuple<int,int>[size];
   
    // populate vector with relative probabilities and keep track of particle locations
    int i = 0;
    int count = 0;
    for(it_type it = beliefs.begin(); it != beliefs.end(); it++) {
        p.push_back(it->second);                
        locations[i++] = it->first;
        beliefs.erase(it->first); //should take care of memory
        count++;
    }

    // discrete distribution object
    trng::discrete_dist dist(p.begin(), p.end());
    // random generator
    trng::yarn5 r;

    std::tuple<int, int> newPos;
    std::tuple<int, int> oldParticle;

    // resample
    for(int a = 0; a < numParticles; a++){          	
        newPos = locations[dist(r)];
    	oldParticle = particles[a];

    	// positions of newPos, dx = newPos(x) - oldParticle(x), etc.
        particles[a] = std::make_tuple(std::get<0>(newPos), 
                                    std::get<1>(newPos));
    }

    delete[] locations;
    p.clear();
}      

ParticleFilter::ParticleFilter (int np, double sig, bool verb, ImageHelper& _imageHelper){
    if(verb){std::cout << "Starting particle filter..." << std::endl;}
    imageHelper = &_imageHelper;
    width = imageHelper->image_width();
    height = imageHelper->image_height();
    numParticles = np;
    sigma = sig;
    verbose = verb;

    dims = std::make_tuple(width,height);
    std::srand(std::time(0));
    r.seed(std::rand());  

    if(verb){std::cout << "Initializing particles..." << std::endl;}
    particles = new std::tuple<int, int>[numParticles];
    ParticleFilter::initializeUniformly();
}

std::tuple<int,int> ParticleFilter::bestGuess(){
    double x,y;
    for(int i=0; i<numParticles;i++){
        x += std::get<0>(particles[i]);
        y += std::get<1>(particles[i]);
    }
    x = x/numParticles;
    y = y/numParticles;
    return std::make_tuple((int)x, (int)y);
}

void ParticleFilter::printFirstN(int n) {
    for (int i = 0; i < n; i++) {
        std::cout << "particles[" << i << "]: ";
        std::cout << std::get<0>(particles[i]) << ", " << std::get<1>(particles[i]);
        std::cout << std::endl;
    }
}

void ParticleFilter::printParticleN(int n){
    std::cout << "particles[" << n << "]: ";
    std::cout << std::get<0>(particles[n]) << ", " << std::get<1>(particles[n]);
    std::cout << std::endl;
}

void ParticleFilter::parFilterIterate(){
    /* Does one iteration of particle filter: elapse time, then observe */
    
    #if TIME
        timeval start, end;
        gettimeofday(&start, 0);
    #endif

    for(int i = 0; i < numParticles; i++){
        telapse(&particles[i]);
    }

    #if TIME
        gettimeofday(&end, 0);
        timer.timeToElapseTime += end.tv_sec + 1e-6*end.tv_usec - start.tv_sec - 1e-6*start.tv_usec;
    #endif

    #if TIME
            gettimeofday(&start, 0);
    #endif

    observe();

    #if TIME
        gettimeofday(&end, 0);
        timer.timeToObserve += end.tv_sec + 1e-6*end.tv_usec - start.tv_sec - 1e-6*start.tv_usec;
    #endif
}

void ParticleFilter::printParams(){
    std::cout << "Width: " << width << std::endl;
    std::cout << "Height: " << height << std::endl;
    std::cout << "numParticles: " << numParticles << std::endl;
    std::cout << "Sigma: " << sigma << std::endl;
    std::cout << "Verbose: " << verbose << std::endl;
}


//dummy observation model
double normProb(int x, int y){
    trng::truncated_normal_dist<> X(1280/2, 200, 0, (float) 1280 - 1);
    trng::truncated_normal_dist<> Y(1024/2, 200, 0, (float) 1024 - 1);
    double prob = X.pdf(x) * Y.pdf(y);
    return prob;
    //return (double)x/1280.0;
}

double ParticleFilter::histProb(int x, int y) {
    return imageHelper->similarity(x, y);
}

/*int main(int argc, char *argv[]) {

    bool verbose = false;
    int c;
    int width = 1280;
    int height = 1024;
    int numParticles = 200;
    double sigma = 20.0;
    int numIter = 200;
    
    while((c = getopt(argc, argv, "v:w:h:n:s:i:"))!=-1){
        switch(c){
            case 'v':
                verbose = atoi(optarg) == 1;
                break;
    	    case 'w':
                width = atoi(optarg);
                break;
    	    case 'h':
                height = atoi(optarg);
                break;
    	    case 'n':
        		numParticles = atoi(optarg);
        		break;
    	    case 's':
        		sigma = atoi(optarg);
        		break;
    	    case 'i':
        		numIter = atoi(optarg);
        		break;
        }
    }

    ParticleFilter pf (width, height, numParticles, sigma, verbose);
    pf.printParams();

    struct timeval time_start, time_end;
    double timetaken;
    gettimeofday(&time_start, 0);    

    for (int i=0; i< numIter; i++) {
        pf.parFilterIterate(normProb);
        if(verbose){pf.printFirstN(numParticles);};
    }

    gettimeofday(&time_end, 0);
    timetaken = time_end.tv_sec + 1e-6*time_end.tv_usec - time_start.tv_sec - 1e-6*time_start.tv_usec;
    std::cout << "Time taken for " << numIter << " iterations with ";
    std::cout << numParticles << " particles is " << timetaken << "s." << std::endl;

    //test particleList
    std::tuple<int,int,int,int> firstP = (pf.particleList())[0];
    std::cout << "First particle: ";
    std::cout << std::get<0>(firstP) << " " << std::get<1>(firstP) << " " << std::get<2>(firstP) << " " << std::get<3>(firstP) << " " <<std::endl;

    //test bestGuess
    std::tuple<int,int> best = pf.bestGuess();
    std::cout << "Best guess: ";
    std::cout << std::get<0>(best) << " " << std::get<1>(best) << std::endl;

    pf.destroy();
    return 1;
}*/
