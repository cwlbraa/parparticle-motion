#include <cstdlib>
#include <iostream>
#include <tuple>
#include <ctime>

#include <math.h>
#include <trng/truncated_normal_dist.hpp>
#include <trng/yarn5.hpp>
void telapse(std::tuple<int,int,int,int> *oldParticle,
                             std::tuple<int,int> dims, 
                             float sigma, trng::yarn5 *random) {
    /* given an old particle, the dimensions of the frame, and the std deviation,
    returns a new particle sampled from a 2D truncated normal distribution. */ 

    //init and unpack variables
    int width, height, x, y, dx, dy;
    std::tie(width, height) = dims;
    std::tie(x, y, dx, dy) = *oldParticle;

    //init distributions
    trng::truncated_normal_dist<> X(std::min(std::max(x+dx,width),0), sigma, 0, (float) width - 1);
    trng::truncated_normal_dist<> Y(std::min(std::max(y+dy,height),0), sigma, 0, (float) height - 1);

    //replace old particle
    *oldParticle = std::make_tuple((int) X(*random), (int) Y(*random), dx, dy);
}

/*
int main() {
    int width = 1280;
    int height = 1024;
    std::tuple<int, int> dims (width,height);

    std::tuple<int, int> topLeft (0,0);
    std::tuple<int, int> topRight (width,0);
    std::tuple<int, int> bottomLeft (0,height);
    std::tuple<int, int> bottomRight (width,height);
    std::tuple<int, int> middle (width/2,height/2);
    
    //init generator
    trng::yarn5 r;
    //seed based on time
    std::srand(std::time(0));
    r.seed(std::rand());

    for (int i=0; i< 20; i++) {
        //move 5 particles 20 times, printing locations
        telapse(&topLeft, dims, 20.0, &r);
        telapse(&topRight, dims, 20.0, &r);
        telapse(&bottomLeft, dims, 20.0, &r);
        telapse(&bottomRight, dims, 20.0, &r);
        telapse(&middle, dims, 20.0, &r);

        std::cout << "topLeft': ";
        std::cout << std::get<0>(topLeft) << ", " << std::get<1>(topLeft);
        std::cout << std::endl;

        std::cout << "topRight': ";
        std::cout << std::get<0>(topRight) << ", " << std::get<1>(topRight);
        std::cout << std::endl;

        std::cout << "bottomLeft': ";
        std::cout << std::get<0>(bottomLeft) << ", " << std::get<1>(bottomLeft);
        std::cout << std::endl;

        std::cout << "bottomRight': ";
        std::cout << std::get<0>(bottomRight) << ", " << std::get<1>(bottomRight);
        std::cout << std::endl;

        std::cout << "middle': ";
        std::cout << std::get<0>(middle) << ", " << std::get<1>(middle);
        std::cout << std::endl;
        std::cout << std::endl;
    }

    return 1;
}
*/
