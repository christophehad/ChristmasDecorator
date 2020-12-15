#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <iostream>
#include "ChristmasDecorator.h"

using namespace cv;
using namespace std;

const bool WARP_OFFSET_RATIO = 0.1;

struct DataAlign {
	Mat I, tmp, out;
	vector<Point2f> selection;
	string inPath, outPath;
};

// main interactive function for selecting the input region to be warped
// inPath should be without extension, outPath should
void interactivePerspTransform(const Mat& input, string inPath, string outPath, DataAlign & D);

// function to restore the initial perspective from the saved warp matrix in .xml file
Mat restorePerspective(const Mat& input, string xmlPath);

int mainAlign();