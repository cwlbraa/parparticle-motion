#include <sys/time.h>  
#include <stdio.h>

#define TIME false
#define DEBUG false
#define FPS true

struct Timer {
    double start, end;
    double timeToConvertColor;
    double timeToComputeDistances;
    double timeToElapseTime;
    double timetoResample;
    double timeToObserve;
    double timeToLoadFrame;
    double timeToDrawStuff;
    double timeToPFIterate;
    double timePerFrame;
    unsigned int numIterations;
};

extern struct Timer timer;