#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <iostream>
#include "ChristmasDecorator.h"

using namespace cv;
using namespace std;

const int WARP_OFFSET = 100;

struct DataAlign {
	Mat I, tmp;
	vector<Point2f> selection;
	string inPath, outPath;
};

// main interactive function for selecting the input region to be warped
void interactivePerspTransform(const Mat& input, string inPath, string outPath);
