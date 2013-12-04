#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include "nmmintrin.h"

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
    normalize(histogram, histogram, 0, 1, NORM_MINMAX, -1, Mat());
    return histogram;
}

double ImageHelper::bhattacharyya_distance(MatND& src, MatND& ref) {
    return 1.0 - compareHist(src, ref, 3);
}

double ImageHelper::bhatta_distance_serial(MatND& src, MatND& ref) {
    float* src_data = (float*)src.data;
    float* ref_data = (float*)ref.data;


    double bhattacharyya = 0.0;
    double h1 = 0.0;
    double h2 = 0.0;

    //std::cout << src.rows << ", " << src.cols << std::endl;

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
    float* src_data = (float*)src.data;
    float* ref_data = (float*)ref.data;


    double bhattacharyya = 0.0;
    double h1 = 0;
    double h2 = 0;
    float results[4];

    __m128 p1;
    __m128 p2;
	__m128 p3;
	__m128 p4;
    __m128 q1;
    __m128 q2;
	__m128 q3;
	__m128 q4;
    __m128 tmp;
	__m128 tmp2;

    int max_sse = src.rows * src.cols - ((src.rows * src.cols) % 16);

    for (int i = 0; i < max_sse; i+=16) {
        p1 = _mm_loadu_ps((src_data + i));
        p2 = _mm_loadu_ps((src_data + i + 4));
		p3 = _mm_loadu_ps((src_data + i + 8));
        p4 = _mm_loadu_ps((src_data + i + 12));

        tmp = _mm_add_ps(p1, p2);
		tmp2 = _mm_add_ps(p3, p4);
		tmp = _mm_add_ps(tmp, tmp2);
        tmp = _mm_hadd_ps(tmp, tmp);
        tmp = _mm_hadd_ps(tmp, tmp);
        _mm_storeu_ps(results, tmp);
        h1 += results[0];

        q1 = _mm_loadu_ps((ref_data + i));
        q2 = _mm_loadu_ps((ref_data + i + 4));
		q3 = _mm_loadu_ps((ref_data + i + 8));
        q4 = _mm_loadu_ps((ref_data + i + 12));

        tmp = _mm_add_ps(q1, q2);
		tmp2 = _mm_add_ps(q3, q4);
		tmp = _mm_add_ps(tmp, tmp2);
        tmp = _mm_hadd_ps(tmp, tmp);
        tmp = _mm_hadd_ps(tmp, tmp);
        _mm_storeu_ps(results, tmp);
        h2 += results[0];

        p1 = _mm_mul_ps(p1, q1);
        p2 = _mm_mul_ps(p2, q2);
		p3 = _mm_mul_ps(p3, q3);
		p4 = _mm_mul_ps(p4, q4);

        p1 = _mm_sqrt_ps(p1);
        p2 = _mm_sqrt_ps(p2);
		p3 = _mm_sqrt_ps(p3);
		p4 = _mm_sqrt_ps(p4);

        p1 = _mm_add_ps(p1, p2);
		p3 = _mm_add_ps(p3, p4);
		p1 = _mm_add_ps(p1, p3);

        p1 = _mm_hadd_ps(p1, p1);
        p1 = _mm_hadd_ps(p1, p1);

        _mm_storeu_ps(results, p1);
        bhattacharyya += results[0];
    }

    int dim = src.rows * src.cols;

    for (int i = max_sse; i < dim; i++) {
        h1 += src_data[i];
        h2 += ref_data[i];
        bhattacharyya += std::sqrt(src_data[i] * ref_data[i]);
    }

    return 1 - std::sqrt(1.0 - 1.0 / std::sqrt(h1 * h2) * bhattacharyya);
}
