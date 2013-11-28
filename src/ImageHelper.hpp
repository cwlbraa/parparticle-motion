#ifndef _IMAGEHELPER_HPP
#define _IMAGEHELPER_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

class ImageHelper {
public:
	Mat image;
	MatND ref_hist;
    int* hist_size;
    int* channels;
    const float** ranges;
    int default_radius_x;
    int default_radius_y;

    ImageHelper(Mat _image, Mat _ref, int _hist_size[], const float* _ranges[], int _channels[]);

    //double similarity(int x, int y, int radius_x, int radius_y);

    double similarity(int x, int y);

    int image_width();

    int image_height();

    void advance_frame(Mat _image_hsv);

    MatND compute_histogram(Mat image);

    double bhattacharyya_distance(MatND src, MatND ref);
};

#endif