#include <sys/time.h>
#include <stdio.h>

#define TIME true
#define DEBUG false
#define FPS false
#define USE_SERIAL false
#define USE_PARALLEL true

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