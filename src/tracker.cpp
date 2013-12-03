#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <getopt.h>
#include <ctime>
#include <sys/time.h>
#include <tuple>
#include <omp.h>

#include "parFilter.h"
#include "ImageHelper.hpp"
#include "tracker.h"

using namespace std;
using namespace cv;

// Evil variables
Timer timer;
timeval time_start, time_stop;
double totalTime;
int numIterations;

// Arguments for comparing histograms
int hist_size[] = {64, 60}; // corresponds to {hue_bins, saturation_bins}
int channels[] = {0, 1};
float hue_range[] = {0, 180};
float saturation_range[] = {0, 256};
const float* ranges[] = {hue_range, saturation_range};

// Arguments for the the particle filter
bool verbose = false;
int numParticles = 1000;
double sigma = 3;

void RGB_TO_HSV_PARALLEL(Mat& source, Mat& destination) {
    destination.create(source.rows, source.cols, CV_8UC3);
    uint8_t* data = (uint8_t*)source.data;
    uint8_t* target = (uint8_t*)destination.data;

    /*int I_STEP_SIZE = 104;
    int J_STEP_SIZE = 104;

    #pragma omp parallel for
    for (int i_step = 0; i_step < source.rows; i_step += I_STEP_SIZE) {
        for (int j_step = 0; j_step < source.cols; j_step += J_STEP_SIZE) {
            for (int i = i_step; i < min(i_step + I_STEP_SIZE, source.rows); i++) {
                for (int j = j_step; j < min(j_step + J_STEP_SIZE, source.cols); j++) {*/
    #pragma omp parallel for
    for (int i = 0; i < source.rows; i++) {
        for (int j = 0; j < source.cols; j++) {
                    float b, g, r, h, s, v, min, max;

                    r = data[source.cols * 3 * i + 3 * j] / 255.0f;
                    g = data[source.cols * 3 * i + 3 * j + 1] / 255.0f;
                    b = data[source.cols * 3 * i + 3 * j + 2] / 255.0f;
                    max = std::max(r, std::max(g, b));
                    min = std::min(r, std::min(g, b));
                    
                    v = max;

                    if (v != 0.0f)
                        s = (v - min) / v;
                    else
                        s = 0.0f;

                    if (v == r)
                        h = 60.0f * (g - b) / (v - min);
                    else if (v == g)
                        h = 120.0f + 60.0f * (b - r) / (v - min);
                    else if (v == b)
                        h = 240.0f + 60.0f * (r - g) / (v - min);

                    if (h < 0.0f)
                        h = h + 360.0f;

                    target[destination.cols * 3 * i + 3 * j] = h / 2.0f; // h is actually 0-360, but obviously that doesn't fit in a uchar
                    target[destination.cols * 3 * i + 3 * j + 1] = s * 255.0f;
                    target[destination.cols * 3 * i + 3 * j + 2] = v * 255.0f;
                /*}
            }*/
        }
    }
}

void RGB_TO_HSV_SERIAL(Mat& source, Mat& destination) {
    destination.create(source.rows, source.cols, CV_8UC3);
    uint8_t* data = (uint8_t*)source.data;
    uint8_t* target = (uint8_t*)destination.data;

    for (int i = 0; i < source.rows; i++) {
        for (int j = 0; j < source.cols; j++) {
            float b, g, r, h, s, v, min, max;

            r = data[source.cols * 3 * i + 3 * j] / 255.0f;
            g = data[source.cols * 3 * i + 3 * j + 1] / 255.0f;
            b = data[source.cols * 3 * i + 3 * j + 2] / 255.0f;
            max = std::max(r, std::max(g, b));
            min = std::min(r, std::min(g, b));
            
            v = max;

            if (v != 0.0f)
                s = (v - min) / v;
            else
                s = 0.0f;

            if (v == r)
                h = 60.0f * (g - b) / (v - min);
            else if (v == g)
                h = 120.0f + 60.0f * (b - r) / (v - min);
            else if (v == b)
                h = 240.0f + 60.0f * (r - g) / (v - min);

            if (h < 0.0f)
                h = h + 360.0f;

            target[destination.cols * 3 * i + 3 * j] = h / 2.0f; // h is actually 0-360, but obviously that doesn't fit in a uchar
            target[destination.cols * 3 * i + 3 * j + 1] = s * 255.0f;
            target[destination.cols * 3 * i + 3 * j + 2] = v * 255.0f;
        }
    }
}

double bhattacharyya_distance(MatND& src, MatND& ref) {
    return 1.0 - compareHist(src, ref, 3);
}

void genAndDisplayHeatMap(Mat& source, Mat& reference, Mat& source_hsv, Mat& reference_hsv) {
    int width = source.size().width;
    int height = source.size().height;
    int refWidth = reference.size().width;
    int refHeight = reference.size().height;
    MatND referenceHist;
    MatND tempHist;
    Mat heatMat = Mat(height, width, CV_8UC1);
    
    calcHist(&reference_hsv, 1, channels, Mat(), referenceHist, 2, hist_size, ranges, true, false);
    int halfW = refWidth/2;
    int halfH = refHeight/2;
    
    for (int i = 0; i < width; i++) {
        int colStart, colEnd;
        colStart = max(i-halfW,0);
        colEnd = min(i+halfW, width);
        Range colRange = Range(colStart, colEnd);
    
        for (int j = 0; j < height; j++) {
            int rowStart, rowEnd;
            double distance;
            
            rowStart = max(j-halfH,0);
            rowEnd = min(j+halfH,height);
            Range rowRange = Range(rowStart, rowEnd);
            
            Mat sourceCompareMat = Mat(source_hsv, rowRange, colRange); 
            calcHist(&sourceCompareMat, 1, channels, Mat(), tempHist, 2, hist_size, ranges, true, false);
            distance = bhattacharyya_distance(tempHist, referenceHist);
            heatMat.at<uchar>(j, i) = static_cast<unsigned char>((distance)*255); 
        }
    }
    Mat heatMat2;
    
    //applyColorMap(heatMat, heatMat2, COLORMAP_HOT);
    imshow("Source", source);
    imshow("Reference", reference);
    //imshow('HeatMap', heatMat2);
    imshow("HeatMap", heatMat);
    
    waitKey(0);
}

void track_image(std::string source_location, std::string reference_location) {
    Mat source, // This is the image to be analyzed
        source_hsv, // This is the source in HSV color space
	reference,
        reference_hsv; // This is the region in HSV color space

    source = imread(source_location, CV_LOAD_IMAGE_COLOR);
    reference = imread(reference_location, CV_LOAD_IMAGE_COLOR);

    if (!source.data || !reference.data) {
        cout << "Could not load an image file" << endl;
        return;
    }

    cvtColor(source, source_hsv, CV_RGB2HSV);
    cvtColor(reference, reference_hsv, CV_RGB2HSV);

    if (DEBUG) {     
        genAndDisplayHeatMap(source, reference, source_hsv, reference_hsv);
    }
    else {
        ImageHelper imageHelper(source_hsv, reference_hsv, hist_size, ranges, channels);
        ParticleFilter pf(numParticles, sigma, verbose, imageHelper);
        pf.printParams();

        //struct timeval time_start, time_end;
        //double timetaken;
        //gettimeofday(&time_start, 0); 

        pf.parFilterIterate();

        //gettimeofday(&time_end, 0);
        //timetaken = time_end.tv_sec + 1e-6*time_end.tv_usec - time_start.tv_sec - 1e-6*time_start.tv_usec;
        //std::cout << "Time taken for " << numIter << " iterations with ";
        //cout << numParticles << " particles is " << timetaken << "s." << endl;

        tuple<int, int, int, int>* particles = pf.particleList();
        for (int i = 0; i < numParticles; i++) {
            circle(source, Point(get<0>(particles[i]), get<1>(particles[i])), 3, Scalar(0, 0, 255), -1, 8);
        }

		//test bestGuess
		std::tuple<int,int,int,int> best = pf.bestGuess();
		std::cout << "Best guess: ";
		std::cout << std::get<0>(best) << " " << std::get<1>(best) << std::endl;

		circle(source, Point(get<0>(best), get<1>(best)), 5, Scalar(0, 255, 0), -1, 8);

		namedWindow("Particle Tracker", CV_WINDOW_AUTOSIZE);
        imshow("Particle Tracker", source);

		waitKey(0);

		pf.destroy();    

	}

	
}

void drawOverlay(Mat& frame, Mat& reference, std::tuple<int, int,int,int> bestGuess) {
    int refWidth = reference.size().width;
    int refHeight = reference.size().width;
    Point vertex1 = Point(get<0>(bestGuess) - int(refWidth/2), get<1>(bestGuess) - int(refHeight/2));
    Point vertex2 = Point(get<0>(bestGuess) + int(refWidth/2), get<1>(bestGuess) + int(refHeight/2));

    rectangle(frame, vertex1, vertex2, Scalar(255, 255, 255), 3, 8, 0);
}

// video file: type=1, webcam: type=0
void track_video(string video_location, string reference_location) {
    Mat frame,
        frame_hsv,
        reference,
        reference_hsv;
    VideoCapture video(video_location);
	reference = imread(reference_location, CV_LOAD_IMAGE_COLOR);

    if (!reference.data) {
        cout << "Could not load an image file" << endl;
        return;
    }
    if (!video.isOpened()) {
        cout << "Could not load a video" << endl;
    }

    #if USE_SERIAL
    RGB_TO_HSV_PARALLEL(reference, reference_hsv);
    #elif USE_PARALLEL
    RGB_TO_HSV_SERIAL(reference, reference_hsv);
    #else
    cvtColor(reference, reference_hsv, CV_RGB2HSV);
    #endif


    namedWindow("Video Tracker", CV_WINDOW_AUTOSIZE);
    //imshow("Video Tracker", reference_hsv);
    //waitKey(0);

    #if TIME
        timeval temp, tv;
        gettimeofday(&temp, 0);;
    #endif

	video >> frame;

    #if TIME
            gettimeofday(&tv, 0);
            timer.timeToLoadFrame += tv.tv_sec + 1e-6*tv.tv_usec - temp.tv_sec - 1e-6*temp.tv_usec;
    #endif

	cvtColor(frame, frame_hsv, CV_RGB2HSV);

	ImageHelper imageHelper(frame_hsv, reference_hsv, hist_size, ranges, channels);
    ParticleFilter pf(numParticles, sigma, verbose, imageHelper);
    pf.printParams();

    pf.parFilterIterate();

    #if TIME
        timer.numIterations = 0;
        gettimeofday(&tv, 0);
        timer.start = tv.tv_sec + 1e-6*tv.tv_usec;
    #endif

    #if FPS
        gettimeofday(&time_start, 0);
        numIterations = 0;
    #endif

    while (true) {
        #if FPS
            numIterations++;
        #endif
 
        #if TIME
            timeval start, end;
            gettimeofday(&start, 0);
        #endif
 
        #if TIME
            timer.numIterations += 1;
        #endif
 
        #if TIME
            gettimeofday(&temp, 0);
        #endif
 
        video >> frame;
 
        #if TIME
            gettimeofday(&tv, 0);
            timer.timeToLoadFrame += tv.tv_sec + 1e-6*tv.tv_usec - temp.tv_sec - 1e-6*temp.tv_usec;
        #endif
 
        if (frame.empty()) {
            #if FPS
                gettimeofday(&time_stop, 0);
                totalTime = time_stop.tv_sec + 1e-6*time_stop.tv_usec - time_start.tv_sec - 1e-6*time_start.tv_usec;
            #endif
            break;
        }
 
        #if TIME
            gettimeofday(&temp, 0);
        #endif
 
        #if USE_PARALLEL
        RGB_TO_HSV_PARALLEL(frame, frame_hsv);
        #elif USE_SERIAL
        RGB_TO_HSV_SERIAL(frame, frame_hsv);
        #else
        cvtColor(frame, frame_hsv, CV_RGB2HSV);
        #endif
 
        #if TIME
            gettimeofday(&tv, 0);
            timer.timeToConvertColor += tv.tv_sec + 1e-6*tv.tv_usec - temp.tv_sec - 1e-6*temp.tv_usec;
        #endif
 
        #if TIME
            gettimeofday(&temp, 0);
        #endif
 
        pf.parFilterIterate();
 
        #if TIME
            gettimeofday(&tv, 0);
            timer.timeToPFIterate += tv.tv_sec + 1e-6*tv.tv_usec - temp.tv_sec - 1e-6*temp.tv_usec;
        #endif
 
        //gettimeofday(&time_end, 0);
        //timetaken = time_end.tv_sec + 1e-6*time_end.tv_usec - time_start.tv_sec - 1e-6*time_start.tv_usec;
        //std::cout << "Time taken for " << numIter << " iterations with ";
        //cout << numParticles << " particles is " << timetaken << "s." << endl;
 
        #if TIME
            gettimeofday(&temp, 0);
        #endif
 
        tuple<int, int, int, int>* particles = pf.particleList();
 
        // Draw the particles
        for (int i = 0; i < numParticles; i++) {
            circle(frame, Point(get<0>(particles[i]), get<1>(particles[i])), 2, Scalar(0, 0, 255), -1, 8);
        }
 
        std::tuple<int,int,int,int> best = pf.bestGuess();
 
        // Draw the best guess particle
                circle(frame, Point(get<0>(best), get<1>(best)), 2, Scalar(0, 255, 0), -1, 8);
 
        // Draw the rectangle
        Point vertex1 = Point(get<0>(best) - reference.cols / 2, get<1>(best) - reference.rows / 2);
        Point vertex2 = Point(get<0>(best) + reference.cols / 2, get<1>(best) + reference.rows / 2);
        rectangle(frame, vertex1, vertex2, Scalar(0, 255, 0), 2, 8, 0);
        line(frame, Point(get<0>(best), get<1>(best)), Point(get<0>(best) + 5*get<2>(best), get<1>(best) + 5*get<3>(best)), Scalar(255, 0, 0), 2);
       
        imshow("Video Tracker", frame);
        //waitKey(1);
 
        #if TIME
            gettimeofday(&tv, 0);
            timer.timeToDrawStuff += tv.tv_sec + 1e-6*tv.tv_usec - temp.tv_sec - 1e-6*temp.tv_usec;
        #endif
 
        #if TIME
            gettimeofday(&end, 0);
            timer.timePerFrame += end.tv_sec + 1e-6*end.tv_usec - start.tv_sec - 1e-6*start.tv_usec;
        #endif
    }

    #if TIME
        gettimeofday(&tv, 0);
        timer.end = tv.tv_sec + 1e-6*tv.tv_usec;

        double timeToLoadFrame = timer.timeToLoadFrame / timer.numIterations;
        double timeToConvertColor = timer.timeToConvertColor / timer.numIterations;
        double timeToPFIterate = timer.timeToPFIterate / timer.numIterations;
        double timeToDrawStuff = timer.timeToDrawStuff / timer.numIterations;
        double timePerFrame = timeToLoadFrame + timeToConvertColor + timeToPFIterate + timeToDrawStuff;

        cout << endl;

		#if USE_PARALLEL
		cout << "Number of threads: " << pf.numThreads << endl;
		#endif

        cout << "Average time per frame: " << timePerFrame << "s" << endl;
        cout << "\tAverage time to load frame: " << timeToLoadFrame << "s" << endl;
        cout << "\tAverage time to convert color: " << timeToConvertColor << "s" << endl;
        cout << "\tAverage time to iterate particle filter: " << timeToPFIterate << "s" << endl;
        cout << "\t\tAverage time to observe: " << timer.timeToObserve / timer.numIterations << "s" << endl;
        cout << "\t\t\tAverage time to compute distances: " << timer.timeToComputeDistances / timer.numIterations << "s" << endl;
        cout << "\t\t\tAverage time to resample: " << timer.timetoResample / timer.numIterations << "s" << endl;
        cout << "\t\tAverage time to elapse time: " << timer.timeToElapseTime / timer.numIterations << "s" << endl;
        cout << "\tAverage time to draw stuff: " << timeToDrawStuff << "s" << endl;
    #endif

    #if FPS
        cout << "Average FPS: " << numIterations / totalTime << endl;
    #endif

	pf.destroy();
}

int main(int argc, char* argv[]) {
    string source_file;
    string reference_file;
    
    /*char c;
    while((c = getopt(argc, argv, "v:w:h:n:s:i:"))!=-1){
        switch(c){
            case 'v':
                verbose = atoi(optarg) == 1;
                break;
            case 'n':
                numParticles = atoi(optarg);
                break;
            case 's':
                sigma = atoi(optarg);
                break;
        }
    }*/

	source_file = argv[1];
	reference_file = argv[2];

	//track_image(source_file, reference_file);
	track_video(source_file, reference_file);
}

/* DEPRECATED STUFF

void genAndDisplayHeatMap(Mat& source, Mat& reference, Mat& source_hsv, Mat& reference_hsv) {
    int width = source.size().width;
    int height = source.size().height;
    int refWidth = reference.size().width;
    int refHeight = reference.size().height;
    MatND referenceHist;
    MatND tempHist;
    Mat heatMat = Mat(height, width, CV_8UC1);
    
    calcHist(&reference_hsv, 1, channels, Mat(), referenceHist, 2, hist_size, ranges, true, false);
    int halfW = refWidth/2;
    int halfH = refHeight/2;
    
    for (int i = 0; i < width; i++) {
        int colStart, colEnd;
        colStart = max(i-halfW,0);
        colEnd = min(i+halfW, width);
        Range colRange = Range(colStart, colEnd);
    
        for (int j = 0; j < height; j++) {
            int rowStart, rowEnd;
            double distance;
            
            rowStart = max(j-halfH,0);
            rowEnd = min(j+halfH,height);
            Range rowRange = Range(rowStart, rowEnd);
            
            Mat sourceCompareMat = Mat(source_hsv, rowRange, colRange); 
            calcHist(&sourceCompareMat, 1, channels, Mat(), tempHist, 2, hist_size, ranges, true, false);
            distance = bhattacharyya_distance(tempHist, referenceHist);
            heatMat.at<uchar>(j, i) = static_cast<unsigned char>((distance)*255); 
        }
    }
    Mat heatMat2;
    
    //applyColorMap(heatMat, heatMat2, COLORMAP_HOT);
    imshow("Source", source);
    imshow("Reference", reference);
    //imshow('HeatMap', heatMat2);
    imshow("HeatMap", heatMat);
    
    waitKey(0);
}

void track_image(std::string source_location, std::string reference_location) {
    Mat source, // This is the image to be analyzed
        source_hsv, // This is the source in HSV color space
    reference,
        reference_hsv; // This is the region in HSV color space

    source = imread(source_location, CV_LOAD_IMAGE_COLOR);
    reference = imread(reference_location, CV_LOAD_IMAGE_COLOR);

    if (!source.data || !reference.data) {
        cout << "Could not load an image file" << endl;
        return;
    }

    cvtColor(source, source_hsv, CV_RGB2HSV);
    cvtColor(reference, reference_hsv, CV_RGB2HSV);

    if (DEBUG) {     
        genAndDisplayHeatMap(source, reference, source_hsv, reference_hsv);
    }
    else {
        ImageHelper imageHelper(source_hsv, reference_hsv, hist_size, ranges, channels);
        ParticleFilter pf(numParticles, sigma, verbose, imageHelper);
        pf.printParams();

        //struct timeval time_start, time_end;
        //double timetaken;
        //gettimeofday(&time_start, 0); 

        pf.parFilterIterate();

        //gettimeofday(&time_end, 0);
        //timetaken = time_end.tv_sec + 1e-6*time_end.tv_usec - time_start.tv_sec - 1e-6*time_start.tv_usec;
        //std::cout << "Time taken for " << numIter << " iterations with ";
        //cout << numParticles << " particles is " << timetaken << "s." << endl;

        tuple<int, int, int, int>* particles = pf.particleList();
        for (int i = 0; i < numParticles; i++) {
            circle(source, Point(get<0>(particles[i]), get<1>(particles[i])), 3, Scalar(0, 0, 255), -1, 8);
        }

        //test bestGuess
        std::tuple<int,int> best = pf.bestGuess();
        std::cout << "Best guess: ";
        std::cout << std::get<0>(best) << " " << std::get<1>(best) << std::endl;

        circle(source, Point(get<0>(best), get<1>(best)), 5, Scalar(0, 255, 0), -1, 8);

        namedWindow("Particle Tracker", CV_WINDOW_AUTOSIZE);
        imshow("Particle Tracker", source);

        waitKey(0);

        pf.destroy();    

    }

    
}
*/
