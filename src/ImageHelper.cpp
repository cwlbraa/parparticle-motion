#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

#include "ImageHelper.hpp"
#include "tracker.h"

using namespace cv;

ImageHelper::ImageHelper(Mat& _image, Mat& _ref, int _hist_size[], const float* _ranges[], int _channels[]) {
    image = _image;
    hist_size = _hist_size;
    channels = _channels;
    ranges = _ranges;
    default_radius_x = _ref.cols % 2 == 0? _ref.cols / 2 - 1 : _ref.cols / 2;
    default_radius_y = _ref.rows % 2 == 0? _ref.rows / 2 - 1 : _ref.rows / 2;
    ref_hist = compute_histogram(_ref);
}

/*
    x - the x coordinate of the center of the region
    y - the y coordinate of the center of the region
    radius_x - the radius of the region in the x direction
    radius_y - the radius of the region in the y direction
*/
/*double ImageHelper::similarity(int x, int y, int radius_x, int radius_y) {
    Mat region(image,
               Range(max(y - radius_y, 0), min(y + radius_y, image.rows)),
               Range(max(x - radius_x, 0), min(x + radius_x, image.cols)));
    MatND region_hist;

    region_hist = compute_histogram(region);
    return bhattacharyya_distance(region_hist, ref_hist);
}*/

/*
    x - the x coordinate of the center of the region
    y - the y coordinate of the center of the region
*/
double ImageHelper::similarity(int x, int y) {
    Mat region(image,
               Range(max(y - default_radius_y, 0), min(y + default_radius_y, image.rows)),
               Range(max(x - default_radius_x, 0), min(x + default_radius_x, image.cols)));
    MatND region_hist;

    region_hist = compute_histogram(region);

    #if USE_SERIAL
    return bhatta_distance_serial(region_hist, ref_hist);
    #elif USE_PARALLEL
    return bhatta_distance_parallel(region_hist, ref_hist);
    #else
    return bhattacharyya_distance(region_hist, ref_hist);
    #endif
}

int ImageHelper::image_width() {
    return image.cols;
}

int ImageHelper::image_height() {
    return image.rows;
}

void ImageHelper::advance_frame(Mat& _image_hsv) {
    image = _image_hsv;
}

MatND ImageHelper::compute_histogram(Mat& image) {
    MatND histogram;
    calcHist(&image, 1, channels, Mat(), histogram, 2, hist_size, ranges, true, false);
    return histogram;
}

double ImageHelper::bhattacharyya_distance(MatND& src, MatND& ref) {
    return 1.0 - compareHist(src, ref, 3);
}

double ImageHelper::bhatta_distance_serial(MatND& src, MatND& ref) {
    uint8_t* src_data = (uint8_t*)src.data;
    uint8_t* ref_data = (uint8_t*)ref.data;


    double bhattacharyya = 0.0;
    double h1 = 0.0;
    double h2 = 0.0;

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            h1 += src_data[src.cols * i + j];
            h2 += ref_data[src.cols * i + j];
            bhattacharyya += std::sqrt(src_data[src.cols * i + j] * ref_data[src.cols * i + j]);
        }
    }

    return 1 - std::sqrt(1.0 - 1.0 / std::sqrt(h1 * h2) * bhattacharyya);
}

double ImageHelper::bhatta_distance_parallel(MatND& src, MatND& ref) {
    uint8_t* src_data = (uint8_t*)src.data;
    uint8_t* ref_data = (uint8_t*)ref.data;


    double bhattacharyya = 0.0;
    double h1 = 0.0;
    double h2 = 0.0;

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            h1 += src_data[src.cols * 3 * i + 3 * j];
            h2 += ref_data[src.cols * 3 * i + 3 * j];
            bhattacharyya += std::sqrt(src_data[src.cols * 3 * i + 3 * j] * ref_data[src.cols * 3 * i + 3 * j]);
        }
    }

    return 1 - std::sqrt(1.0 - 1.0 / std::sqrt(h1 * h2) * bhattacharyya);
}