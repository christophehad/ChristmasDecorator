#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <algorithm>
#include "ChristmasDecorator.h"

using namespace cv;
using namespace std;

struct RectRegion {
	Point corner;
	int height;
	int width;
};

// 1 % seems the best
const double cornersMarginThresh = 0.01;

// sometimes a small one (150) detects many duplicate corners in small images
const int HARRIS_THRESH = 170;

// allow for rect regions to intersect
const bool WITH_INTERS = true;

// important for debugging
string type2str(int type);

// main function that returns a vector of the regions corresponding to the label with color
//	color should be in BGR
vector<RectRegion> getRegions(const Mat& I, Vec3b color);

Mat selectColor(const Mat& Ic, Vec3b color);

float convol(const Mat& input, vector<vector<float>>& mask, int k, int i, int j, float weight);

Mat threshold(const Mat& Ic, float s, bool denoise);

Mat canny(const Mat& Ic, float s1, float s2);

vector<Point> harris(const Mat& Ic);



