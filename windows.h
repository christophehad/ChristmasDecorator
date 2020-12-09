#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ChristmasDecorator.h"
#include "corners.h"

using namespace cv;
using namespace std;

const string mistletoePath = CD::srcDir + "/data/mistletoe.jpg";
const string mistletoeAlphaPath = CD::srcDir + "/data/mistletoe-alpha.jpg";

// main function for decorating the windows with mistletoes
void decorateWindows(const Mat& input, const Mat& labels, Mat& output);

struct Data {
	Mat I1, M1;
	Mat W1;
	Point prev;
};