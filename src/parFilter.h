#ifndef _PARFILTER_H // must be unique name in the project
#define _PARFILTER_H

#include "ImageHelper.hpp"
#include <trng/yarn5.hpp>

//Map iterator type
//typedef std::map<std::tuple<int,int>, double>::iterator it_type;

class ParticleFilter {
        ImageHelper* imageHelper;
        bool verbose;
        int width, height, numParticles, numIter;
        double sigma;
        std::tuple<int, int> *particles;
        double *probs;
        std::tuple<int,int> dims;
        trng::yarn5 r;

        void initializeUniformly();

        void telapse(std::tuple<int,int> *oldParticle);

        void observe();

    public:
        ParticleFilter (int np, double sig, bool verb, ImageHelper& imageHelper);

        std::tuple<int,int>* particleList(){ return particles; }

        std::tuple<int,int> bestGuess();

        void printFirstN(int n);

        void printParticleN(int n);

        void parFilterIterate();

        void printParams();

        void destroy(){ delete[] particles; }

        double histProb(int x, int y);
};


//dummy observation model
double normProb(int x, int y);

#endif
