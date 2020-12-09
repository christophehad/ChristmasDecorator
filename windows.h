#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

void insertMistleToe(const Mat& input, Mat& output, const Mat& mistletoe, const Mat& mistletoe_alpha, Point corner, int height, int width);

struct Data {
	Mat I1, M1;
	Mat W1;
	Point prev;
};