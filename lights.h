#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <stdlib.h>

#include "ChristmasDecorator.h"

Mat getMaskAsGuirlandes(const Mat3b& mask, const Mat& image, Mat& lights, vector<Vec3b> lights_color, bool crop_to_mask);

Mat getMaskAsLights(const Mat3b& mask, const Mat& image, Mat& lights, vector<Vec3b> lights_color, bool crop_to_mask, bool window_glow);