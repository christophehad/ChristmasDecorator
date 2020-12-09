#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include "windows.h"

using namespace cv;
using namespace std;

void insertMistleToe(const Mat& input, Mat& output, const Mat& mistletoe, const Mat& mistletoe_alpha, Point corner, int height, int width) {
	// Region to edit in input
	Mat mistleRegion, outputRegion, resizedMistle, resizedAlpha;

	// Fit the mistletoe in the region
	int newDim = min(height, width);
	Point newCorner(corner.x+(width-newDim)/2,corner.y+(height-newDim)/2);

	// Copy the input region inside the mistletoe boundaries to mistleRegion
	input(Rect(newCorner.x,newCorner.y, newDim, newDim)).copyTo(mistleRegion);
	input(Rect(newCorner.x, newCorner.y, newDim, newDim)).copyTo(outputRegion);

	resize(mistletoe, resizedMistle, Size(newDim, newDim));
	resize(mistletoe_alpha, resizedAlpha, Size(newDim, newDim));
	resizedMistle.convertTo(resizedMistle, CV_32FC3);
	mistleRegion.convertTo(mistleRegion, CV_32FC3);
	resizedAlpha.convertTo(resizedAlpha, CV_32FC3, 1.0 / 255);

	cout << resizedAlpha.type() << " " << resizedMistle.type() << endl;
	multiply(resizedAlpha, resizedMistle, resizedMistle);
	multiply(Scalar::all(1.0) - resizedAlpha, mistleRegion, mistleRegion);

	// mistleRegion now contains the mistletoe added to the square window of the input
	add(mistleRegion, resizedMistle, mistleRegion);
	mistleRegion.convertTo(mistleRegion, input.type());

	// Optional blend so that it's more realistic
	double alpha = 0.9;
	addWeighted(mistleRegion, alpha, outputRegion, 1 - alpha, 0, outputRegion);

	input.copyTo(output);
	outputRegion.convertTo(outputRegion, input.type());
	outputRegion.copyTo(output(Rect(newCorner.x, newCorner.y, newDim, newDim)));
}

void decorateWindows(const Mat& input, const Mat& labels, Mat& output) {
	vector <RectRegion> windows = getRegions(labels, CD::windowColor);

	//vector <RectRegion> windows = getRegions(labels, Vec3b(0,255,0)); // trying to decorate from masks
	
	Mat mistletoe = imread(mistletoePath);
	Mat mistletoeAlpha = imread(mistletoeAlphaPath);

	output = input;

	for (auto& region : windows) {
		insertMistleToe(output, output, mistletoe, mistletoeAlpha, region.corner, region.height, region.width);
	}
}

void onMouse1(int event, int x, int y, int foo, void* p)
{
	if (event == EVENT_LBUTTONDOWN) {
		Point m1(x, y);

		Data* D = (Data*)p;
		Mat tmp;
		D->W1.copyTo(tmp);
		circle(tmp, m1, 2, Scalar(0, 255, 0), 2);
		imshow("Window", tmp);

		D->prev = m1;
	}
	else if (event == EVENT_RBUTTONDOWN) {
		Point m1(x, y);

		Data* D = (Data*)p;
		Point prev = D->prev;

		int width = x - prev.x, height = y - prev.y;

		Mat output;
		Mat tmp;
		D->W1.copyTo(tmp);
		circle(tmp, m1, 2, Scalar(0, 0, 255), 2);
		imshow("Window", tmp);

		insertMistleToe(D->W1, output,D->I1, D->M1, D->prev, height, width);
		imshow("Added Mistletoe", output);

		D->W1 = output;
	}

}

int debugWindows(int argc, char** argv)
{
	Data D;
	D.I1 = imread(CD::srcDir + "/data/mistletoe.jpg");
	D.M1 = imread(CD::srcDir + "/data/mistletoe-alpha.jpg");
	D.W1 = imread(CD::srcDir + "/data/windows.jpg");

	imshow("Window", D.W1);
	//imshow("Mistletoe", D.I1);
	//imshow("Mistletoe Alpha", D.M1);

	//cvtColor(D.I1, G1, COLOR_BGR2GRAY);

	setMouseCallback("Window", onMouse1, &D);

	waitKey(0);
	return 0;
}
