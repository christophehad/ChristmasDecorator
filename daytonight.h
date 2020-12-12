#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/intensity_transform.hpp>

#include "ChristmasDecorator.h"

void quantizeImageWithKmeans(const Mat& image, Mat quantized_image, int nmb_clusters);
Mat increaseColor(const Mat& image, double scale, int channel);
Mat changeHSVchannel(const Mat& image, double scale, int channel);
Mat changeHSVchannel(const Mat& image, double scale, int channel);
Mat increaseColor(const Mat& image, double scale, int channel);

Mat darkenSkyOfImage(const Mat& image);
Mat dayToNightTransfer(const Mat& image);