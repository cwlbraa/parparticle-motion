#include <sys/time.h>  
#include <stdio.h>

#define TIME false
#define DEBUG false
#define FPS false

struct Timer {
    double start, end;
    double timeToConvertColor;
    double timeToComputeDistances;
    double timeToElapseTime;
    double timeToObserve;
    double timeToLoadFrame;
    double timeToDrawStuff;
    double timeToPFIterate;
    double timePerFrame;
    unsigned int numIterations;
};

extern struct Timer timer;