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

#include "parFilter.h"
#include "ImageHelper.hpp"

#define DEBUG false

int hist_size[] = {32, 30}; // corresponds to {hue_bins, saturation_bins}
int channels[] = {0, 1};
float hue_range[] = {0, 256};
float saturation_range[] = {0, 180};
const float* ranges[] = {hue_range, saturation_range};
bool verbose = false;
int numParticles = 200;
double sigma = 12.0;

using namespace std;
using namespace cv;

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

		for(int i = 0; i < 5; i++){
        	pf.parFilterIterate();
		}

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

void drawOverlay(Mat& frame, Mat& reference, std::tuple<int, int> bestGuess) {
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

    cvtColor(reference, reference_hsv, CV_RGB2HSV);

    namedWindow("Video Tracker", CV_WINDOW_AUTOSIZE);

	video >> frame;
	cvtColor(frame, frame_hsv, CV_RGB2HSV);

	ImageHelper imageHelper(frame_hsv, reference_hsv, hist_size, ranges, channels);
    ParticleFilter pf(numParticles, sigma, verbose, imageHelper);
    pf.printParams();

    while (true) {
        //struct timeval time_start, time_end;
        //double timetaken;
        //gettimeofday(&time_start, 0); 

		for(int i = 0; i < 5; i++){
        	pf.parFilterIterate();
		}

        //gettimeofday(&time_end, 0);
        //timetaken = time_end.tv_sec + 1e-6*time_end.tv_usec - time_start.tv_sec - 1e-6*time_start.tv_usec;
        //std::cout << "Time taken for " << numIter << " iterations with ";
        //cout << numParticles << " particles is " << timetaken << "s." << endl;

        tuple<int, int, int, int>* particles = pf.particleList();
        for (int i = 0; i < numParticles; i++) {
            circle(frame, Point(get<0>(particles[i]), get<1>(particles[i])), 2, Scalar(0, 0, 255), -1, 8);
        }

		//test bestGuess
		std::tuple<int,int> best = pf.bestGuess();
		//std::cout << "Best guess: ";
		//std::cout << std::get<0>(best) << " " << std::get<1>(best) << std::endl;

		circle(frame, Point(get<0>(best), get<1>(best)), 2, Scalar(0, 255, 0), -1, 8);
        drawOverlay(frame, reference, best);

        imshow("Video Tracker", frame);

        if (waitKey(30) >= 0)
            break;

		video >> frame;

		if (frame.empty())
			break;
		cvtColor(frame, frame_hsv, CV_RGB2HSV);
		
    }

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
