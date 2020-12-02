#include "align.h"

Mat transformWithTemplate(const Mat& input, const Mat& temp) {
	Mat input_gray, template_gray;
	cvtColor(input, input_gray, COLOR_RGB2GRAY);
	cvtColor(temp, template_gray, COLOR_RGB2GRAY);

	int warpMode = MOTION_HOMOGRAPHY;
	int num_iterations = 500;
	double termination_eps = 1e-10;

	Mat warp;
	if (warpMode == MOTION_HOMOGRAPHY)
		warp = Mat::eye(3, 3, CV_32F);
	else
		warp = Mat::eye(2, 3, CV_32F);

	TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, num_iterations, termination_eps);

	findTransformECC(template_gray, input_gray, warp, warpMode, criteria);

	Mat ret;

	if (warpMode != MOTION_HOMOGRAPHY)
		warpAffine(input, ret, warp, temp.size(), INTER_LINEAR + WARP_INVERSE_MAP);
	else
		warpPerspective(input, ret, warp, input.size(), INTER_LINEAR + WARP_INVERSE_MAP);
	
	return ret;
}

Mat perspTransform(const Mat& input, vector<Point2f> & selection, string xmlPath) {
	int offset = WARP_OFFSET; // for the returned matrix
	
	Point2f S1 = selection[0], S2 = selection[1], S3 = selection[2], S4 = selection[3];
	double leftH = sqrt((S4.x - S1.x) * (S4.x - S1.x) + (S4.y - S1.y) * (S4.y - S1.y));
	double rightH = sqrt((S3.x - S2.x) * (S3.x - S2.x) + (S3.y - S2.y) * (S3.y - S2.y));

	Rect R;
	if (leftH > rightH)
		R = Rect(S1.x, S1.y, S2.x - S1.x, leftH);
	else
		R = Rect(S1.x, S2.y, S2.x - S1.x, rightH);

	Point2f D1 = Point2f(R.x, R.y);
	Point2f D2 = Point2f(R.x+R.width, R.y);
	Point2f D3 = Point2f(R.x+R.width, R.y+R.height);
	Point2f D4 = Point2f(R.x, R.y+R.height);

	vector<Point2f> source = selection;
	vector<Point2f> destination;
	
	destination.push_back(D1); destination.push_back(D2); destination.push_back(D3); destination.push_back(D4);

	Mat warpM = getPerspectiveTransform(source, destination);
	cout << "Warp Matrix\n" << warpM << endl;
	FileStorage file(xmlPath, FileStorage::WRITE);
	file << "dim" << Size(input.rows, input.cols);
	file << "warp" << warpM;
	
	Mat transformedM = Mat::zeros(input.rows + offset, input.cols + offset,input.type());
	warpPerspective(input, transformedM, warpM, transformedM.size());

	return transformedM;
}

Mat perspTransformWithWarp(const Mat& input, const Mat& warp) {
	Mat transformedM = Mat::zeros(input.rows + WARP_OFFSET, input.cols + WARP_OFFSET, input.type());
	warpPerspective(input, transformedM, warp, transformedM.size());
	return transformedM;
}

void perspOnMouse(int event, int x, int y, int foo, void* p)
{
	DataAlign * D = (DataAlign*) p;

	// select the bounding points
	if (event == EVENT_LBUTTONDOWN) {
		Point m1(x, y);

		circle(D->tmp, m1, 2, Scalar(0, 255, 0), 2);
		imshow("Interactive input", D->tmp);

		D->selection.push_back(Point2f(x,y));
	}
	// retrieve a stored warp
	else if (event == EVENT_LBUTTONDBLCLK) {
		string xmlPath = D->inPath + ".xml";
		FileStorage file(xmlPath, FileStorage::READ);
		if (!file.isOpened())
		{
			cerr << "failed to open " << xmlPath << endl;
		}
		else {
			Mat warp;  file["warp"] >> warp;
			Mat alignedImg = perspTransformWithWarp(D->I, warp);
			imshow("Aligned image", alignedImg);
			imwrite(D->outPath, alignedImg);
			D->out = alignedImg;
		}

	}
	// execute the warp
	else if (event == EVENT_RBUTTONDOWN) {
		Mat alignedImg = perspTransform(D->I, D->selection, D->inPath + ".xml");
		imshow("Aligned image", alignedImg);
		imwrite(D->outPath, alignedImg);
		D->out = alignedImg;
	}

}

// Use the left mouse button to add the four corner points (starting from upper-left and going clockwise)
// Use the right mouse button to generate the warped image using the selection
// Use the left mouse button and double-click to retrieve the previous pre-computed warp
void interactivePerspTransform(const Mat& input, string inPath, string outPath, DataAlign & D) {
	vector<Point2f> selection;
	D.selection = selection; D.inPath = inPath; D.outPath = outPath;
	input.copyTo(D.I); input.copyTo(D.tmp);
	imshow("Interactive input", input);
	//resizeWindow("Interactive input", 2 * input.rows, 2 * input.cols);
	setWindowProperty("Interactive input", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
	setMouseCallback("Interactive input", perspOnMouse, &D);
	waitKey();
}

Mat restorePerspective(const Mat & input, string xmlPath) {
	Mat restoredImage;
	FileStorage file(xmlPath, FileStorage::READ);
	if (!file.isOpened())
	{
		cerr << "failed to open " << xmlPath << endl;
	}
	else {
		Mat warp;  file["warp"] >> warp;
		Size origSize; file["dim"] >> origSize;

		restoredImage = Mat::zeros(origSize, input.type());
		warpPerspective(input, restoredImage, warp, origSize, INTER_LINEAR + WARP_INVERSE_MAP, BORDER_TRANSPARENT);
		imshow("Restored image", restoredImage);
		
	}
	return restoredImage;
}

int mainAlign() {
	string inputPath1 = "/data/cmp_b0377"; //Zurich cropped bldg
	string inputPath2 = "/data/cmp_b0001"; //vertical tall bldg
	string inputPath3 = "/data/facade_to_align_template"; //flipped image using iPhone editing
	string inputPath = inputPath3;
	string toAlignPath = "/data/facade_to_align";

	string outputPath = CD::srcDir + toAlignPath + "_aligned" + ".jpg";
	
	Mat inputTemplate = imread(CD::srcDir + inputPath + ".jpg");
	Mat inputToAlign = imread(CD::srcDir + toAlignPath + ".jpg");

	imshow("Image to align", inputToAlign);
	//imshow("Template image", inputTemplate);

	//Mat alignedImg = transformWithTemplate(inputToAlign, inputTemplate);
	//imshow("Aligned image", alignedImg);

	DataAlign D;
	interactivePerspTransform(inputToAlign,CD::srcDir + inputPath,outputPath,D);
	restorePerspective(D.out, D.inPath + ".xml");
	waitKey();

	return 0;
}
