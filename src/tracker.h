#include <sys/time.h>
#include <stdio.h>

#define TIME false
#define DEBUG false
#define FPS false
#define USE_SERIAL false
#define USE_PARALLEL true
// If both USE_SERIAL and USE_PARALLEL are set to false, openCV's methods will be used

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
