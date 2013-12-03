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
#include <trng/uniform_dist.hpp>
#include <trng/discrete_dist.hpp>
#include <trng/truncated_normal_dist.hpp>
#include <getopt.h>
#include <omp.h>

#include "tracker.h"
#include "parFilter.h"
#include "ImageHelper.hpp"

void ParticleFilter::initializeUniformly(){
    int x,y;
    for (int i = 0; i < numParticles; i++){
        trng::uniform_dist<> X(0, width - 1);
        trng::uniform_dist<> Y(0, height - 1);
        x = (int) X(r);
        y = (int) Y(r);
        particles[i] = std::make_tuple(x,y,0,0); //initialize velocities to zero
    }
}

void ParticleFilter::telapse(std::tuple<int,int,int,int> *oldParticle) {
    /* given an old particle, the dimensions of the frame, and the std deviation,
     * returns a new particle sampled from a 2D truncated normal distribution. */ 

    //init and unpack variables
    int x, y, dx, dy, x1, y1;
    std::tie(x, y, dx, dy) = *oldParticle;

    //init distributions, setting mean to value projected by last velocity
    //particles in motion tend to stay in motion
    trng::truncated_normal_dist<> X(x+dx, sigma, (float) std::min(0, x+dx), (float) std::max(width - 1, x+dx));
    trng::truncated_normal_dist<> Y(y+dy, sigma, (float) std::min(0, y+dy), (float) std::max(height - 1, y+dy));

    //sample next point
    x1 = (int) X(r); y1 = (int) Y(r);

    if (x1 >= width || x1 < 0 || y1 >= height || y1 < 0) {
        trng::uniform_dist<> X(0, width - 1);
        trng::uniform_dist<> Y(0, height - 1);
        x1 = (int) X(r);
        y1 = (int) Y(r);
        *oldParticle = std::make_tuple(x1,y1,0,0);
    } else {
        *oldParticle = std::make_tuple(x1, y1, x1-x, y1-y);
    }
}

void ParticleFilter::observe(){
	/* computes probability for each particle using basic image processing
     * and resamples all particles from generated distribution. */

    #if TIME
        timeval start, end, end1;
        gettimeofday(&start, 0);
    #endif

    double* probmaxes;
    std::tuple<int,int,int,int>* parmaxes;
    //generate probability distribution across particles

    #if USE_PARALLEL
    #pragma omp parallel
    #endif
    {

    #if USE_PARALLEL
    #pragma omp critical
    #endif
    {
    probmaxes = new double[omp_get_num_threads()];
    parmaxes = new std::tuple<int,int,int,int>[omp_get_num_threads()];
    }

    std::tuple<int,int,int,int> t;
    int thrd_num = omp_get_thread_num();

    #if USE_PARALLEL
    #pragma omp for
    #endif
    for(int i = 0; i < numParticles; i+=125) {
        for (int j = i; j < i + 125; j++) {
            t = particles[j];
            probs[j] = imageHelper->similarity(std::get<0>(t), std::get<1>(t));

            //retain max p per thread...
            if (probmaxes[thrd_num] != max(probmaxes[thrd_num], probs[j])) {
                probmaxes[thrd_num] = probs[j];
                parmaxes[thrd_num] = t;
            }
        }
    }
    #if USE_PARALLEL
    #pragma omp critical
    #endif
    {
    double best = 0.0;
    for (int i=0; i < omp_get_num_threads(); i++) {
        if (probmaxes[i] > best) {
            best = probmaxes[i];
            bestGuess1 = parmaxes[i];
        }
    }

    }
    //end parallel block
    }




    #if TIME
        gettimeofday(&end, 0);
        timer.timeToComputeDistances += end.tv_sec + 1e-6*end.tv_usec - start.tv_sec - 1e-6*start.tv_usec;
    #endif

    //convert probability matrix to vector for trng
    std::vector<double> p(probs, probs + numParticles);
   
    // create discrete dist from probability matrix
    trng::discrete_dist dist(p.begin(), p.end());

    // resample
    for(int a = 0; a < numParticles; a++){          	
    	// positions of newPos, dx = newPos(x) - oldParticle(x), etc.
        newparticles[a] = particles[dist(r)];
    }

    // swap pointers to particle buffers 
    // newparticles will now contain oldparticles to be overwritten on next iteration
    std::swap(particles, newparticles);

    p.clear();

    #if TIME
        gettimeofday(&end1, 0);
        timer.timetoResample += end1.tv_sec + 1e-6*end1.tv_usec - end.tv_sec - 1e-6*end.tv_usec;
    #endif
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

    //seed trng random generator w/ time
    std::srand(std::time(0));
    r.seed(std::rand());  

    if(verb){std::cout << "Initializing particles..." << std::endl;}

    // initialize memory for particle filter data
    particles = new std::tuple<int, int, int, int>[numParticles];
    // buffer for observation resample step
    newparticles = new std::tuple<int, int, int, int>[numParticles];
    probs = new double[numParticles];
    ParticleFilter::initializeUniformly();
}

std::tuple<int,int,int,int> ParticleFilter::bestGuess(){
    double x=0,y=0,dx=0,dy=0;
    for(int i=0; i<numParticles;i++){
        x += std::get<0>(particles[i]);
        y += std::get<1>(particles[i]);
        dx += std::get<2>(particles[i]);
        dy += std::get<3>(particles[i]);
    }
    x = x/numParticles;
    y = y/numParticles;
    dx = dx/numParticles;
    dy = dy/numParticles;
    return std::make_tuple((int)x, (int)y, (int)dx, (int)dy);

    
    /*// double x=0,y=0;
    // for(int i=0; i<numParticles;i++){
    //     x += std::get<0>(particles[i]);
    //     y += std::get<1>(particles[i]);
    // }
    // x = x/numParticles;
    // y = y/numParticles;
    return bestGuess1; //std::make_tuple((int)x, (int)y);*/
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
