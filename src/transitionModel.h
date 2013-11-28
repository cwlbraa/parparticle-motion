#ifndef _TRANSITIONMODEL_H // must be unique name in the project
#define _TRANSITIONMODEL_H

void telapse(std::tuple<int,int,int,int> *oldParticle,
                             std::tuple<int,int> dims, 
                             float sigma, trng::yarn5 *random);

#endif 
