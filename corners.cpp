#include "corners.h"


#define PI 3.14159265

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

// The function takes care of the bounds
float convol(const Mat& input, vector<vector<float>> & mask, int k, int i, int j, float weight) {
	int m = input.rows, n = input.cols;
	float res = 0.0;
	for (int u = -k; u <= k; u++) {
		for (int v = -k; v <= k; v++) {
			int row_idx = i - u, col_idx = j - v;
			float entry = 0.0;
			if ((0 <= row_idx && row_idx < m) && (0 <= col_idx && col_idx < n))
				entry = input.at<uchar>(row_idx, col_idx);
			res += entry * weight * mask[u+k][v+k] ;
		}
	}
	return res;
}

// Raw gradient. No denoising
void gradient(const Mat&Ic, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, COLOR_BGR2GRAY);

	int m = I.rows, n = I.cols;
	G2 = Mat(m, n, CV_32F);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			// Compute squared gradient (except on borders)
			float ix = 0.0, iy = 0.0;
			if (i != m - 1)
				iy = (float(I.at<uchar>(i + 1, j))) - (float(I.at<uchar>(i, j)));
			if (j != n - 1)
				ix = (float(I.at<uchar>(i, j + 1))) - (float(I.at<uchar>(i, j)));

			G2.at<float>(i, j) = ix * ix + iy * ix;
		}
	}
}

// Gradient (and derivatives), Sobel denoising
void sobel(const Mat&Ic, Mat& Ix, Mat& Iy, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, COLOR_BGR2GRAY);

	int m = I.rows, n = I.cols;
	Ix = Mat(m, n, CV_32F);
	Iy = Mat(m, n, CV_32F);
	G2 = Mat(m, n, CV_32F);

	vector<vector<float>> mask_sobel_x {
		{-1,0,1},
		{-2,0,2},
		{-1,0,1}};

	vector<vector<float>> mask_sobel_y {
		{1,2,1},
		{0,0,0},
		{-1,-2,-1} };

	float mask_sobel_x2 [9] = {
	-1,0,1,
	-2,0,2,
	-1,0,1 };

	float mask_sobel_y2 [9] = {
		1,2,1,
		0,0,0,
		-1,-2,-1 };

	float weight = 1.0 / 8;

	Mat Ifloat = Mat(m, n, CV_32F);
	I.convertTo(Ifloat, CV_32F);
	Mat maskSobelX = Mat(3, 3, CV_32F, mask_sobel_x2);
	Mat maskSobelY = Mat(3, 3, CV_32F, mask_sobel_y2);
	maskSobelX = maskSobelX * weight;
	maskSobelY = maskSobelY * weight;

	filter2D(Ifloat, Ix, -1, maskSobelX);
	filter2D(Ifloat, Iy, -1, maskSobelY);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float ix = Ix.at<float>(i, j);
			float iy = Iy.at<float>(i, j);
			G2.at<float>(i, j) = ix * ix + iy * iy;
		}
	}
}

// Gradient thresholding, default = do not denoise
Mat threshold(const Mat& Ic, float s, bool denoise = false)
{
	Mat Ix, Iy, G2;
	if (denoise)
		sobel(Ic, Ix, Iy, G2);
	else
		gradient(Ic, G2);
	int m = Ic.rows, n = Ic.cols;
	Mat C(m, n, CV_8U);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			C.at<uchar>(i, j) = (G2.at<float>(i,j)>s*s)*255;
	return C;
}

bool isMaxAlongGradient(const Mat& G2, int i, int j, pair<int, int>& p) {
	int m = G2.rows, n = G2.cols;
	float init_value = G2.at<float>(i, j);
	vector<float> toCompare{ init_value };

	for (int k = -1; k <= 1; k = k + 2) {
		int row_idx = i + k*p.first, col_idx = j + k*p.second;
		if ((0 <= row_idx && row_idx < m) && (0 <= col_idx && col_idx < n))
			toCompare.push_back(G2.at<float>(row_idx, col_idx));
	}

	auto max_value = max_element(toCompare.begin(), toCompare.end());
	return *max_value == init_value;
}

void checkNeighbors(Mat& C, const Mat& Max, int i, int j) {

	int m = C.rows, n = C.cols;

	for (int k = -1; k <= 1; k++) {
		for (int l = -1; l <= 1; l++) {
			if (k != 0 || l != 0) {
				int row_idx = i + k, col_idx = j + l;
				if ((0 <= row_idx && row_idx < m) && (0 <= col_idx && col_idx < n)) {
					uchar& c_entry = C.at<uchar>(row_idx, col_idx);
					uchar max_entry = Max.at<uchar>(row_idx, col_idx);

					// If we reach a strong edge (initially strong but not in C yet, or weak that's connected to our current strong)
					if (max_entry && !c_entry) {
						c_entry = 255;
						checkNeighbors(C, Max, row_idx, col_idx);
					}
				}
			}
		}
	}
}

// Canny edge detector, with thresholds s1<s2
Mat canny(const Mat& Ic, float s1, float s2)
{
	Mat Ix, Iy, G2;
	sobel(Ic, Ix, Iy, G2);
	vector <pair<int, int>> directions{ {1,0}, {1,1}, {0,1}, {-1,1} };
	int m = Ic.rows, n = Ic.cols;
	Mat Max(m, n, CV_8U);	// Binary black&white image with white pixels when ( G2 > s1 && max in the direction of the gradient )
	// http://www.cplusplus.com/reference/queue/queue/
	queue<Point> Q;			// Enqueue seeds ( Max pixels for which G2 > s2 )
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float dx = Ix.at<float>(i, j), dy = Iy.at<float>(i, j), g2 = G2.at<float>(i,j);

			// angle between -90 and +90
			float direction_angle = dx == 0? 0: atan(dy / dx) * 180.0 / PI;
			int direction_tmp = (int) round((direction_angle + 90.0) / 22.5);
			int direction_idx = ((direction_tmp + 1) % 8) / 2;

			Max.at<uchar>(i, j) = 0;

			if (isMaxAlongGradient(G2, i, j, directions[direction_idx])) {
				if (g2 > s1*s1)
					Max.at<uchar>(i, j) = 255;
				if (g2 > s2*s2)
					Q.push(Point(j, i));
				
			}
		}
	}
	Max;
	// Propagate seeds

	// Max contains pixels that are set if they are weak or strong edges
	// C contains pixels that are set if they are strong edges (initially or by chain)
	// Q contains the initial strong edges
	Mat C(m, n, CV_8U);
	C.setTo(0);
	int idx = 0;
	while (!Q.empty()) {
		int i = Q.front().y, j = Q.front().x;
		Q.pop();
		C.at<uchar>(i, j) = 255;
		checkNeighbors(C, Max, i, j);
	}

	return C;
}

vector<Point> harris(const Mat& Ic) {
	// To Tweak According to the Image
	int thresh = HARRIS_THRESH;

	vector<Point> corners;

	int blockSize = 10;
	int apertureSize = 11;
	double k = 0.04;
	Mat dst = Mat::zeros(Ic.size(), CV_32FC1);
	cornerHarris(Ic, dst, blockSize, apertureSize, k);
	Mat dst_norm, dst_norm_scaled;
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);
	for (int i = 0; i < dst_norm.rows; i++)
	{
		for (int j = 0; j < dst_norm.cols; j++)
		{
			if ((int)dst_norm.at<float>(i, j) > thresh)
			{
				circle(dst_norm_scaled, Point(j, i), 4, Scalar(200), 1, 8, 0);
				corners.push_back(Point(j, i));
			}
		}
	}
	//imshow("Harris Corners", dst_norm_scaled);
	return corners;
}


Mat selectColor(const Mat& Ic, Vec3b colorLower, Vec3b colorUpper) {
	Mat mask, goodMask, out; 

	inRange(Ic, colorLower, colorUpper, mask);
	vector<Mat> tmp = { mask,mask,mask };
	merge(tmp, goodMask);

	cvtColor(mask,goodMask, COLOR_GRAY2RGB);
	//cout << type2str(goodMask.type()) << endl << type2str(Ic.type()) << endl;
	multiply(Ic, goodMask, out);

	//imshow("Mask", goodMask);
	//imshow("Modified I", Ic);
	return out;
}

bool isLessByX(const Point& p1, const Point& p2) {
	if (p1.x < p2.x)
		return true;
	else {
		if (p1.x == p2.x)
			return p1.y < p2.y;
		else
			return false;
	}
}

bool isLessByY(const Point& p1, const Point& p2) {
	if (p1.y < p2.y)
		return true;
	else {
		if (p1.y == p2.y)
			return p1.x < p2.x;
		else
			return false;
	}
}

bool isInAnyRegion(Point p, const vector <RectRegion>& regions) {
	bool ret = false;

	if (WITH_INTERS) {
		for (auto& region : regions) {
			if (region.corner.x < p.x && p.x < (region.corner.x + region.width) && region.corner.y < p.y && p.y < (region.corner.y + region.height))
				return true;
		}
	}
	else {
		for (auto& region : regions) {
			if (region.corner.x <= p.x && p.x <= (region.corner.x + region.width) && region.corner.y <= p.y && p.y <= (region.corner.y + region.height))
				return true;
		}
	}
	return ret;
}

void filterPoints(vector <Point>& v, int threshold) {
	auto cur = v.begin();

	// remove close points
	while (cur != v.end()) {
		auto next = cur + 1;
		while (next != v.end()) {
			if (sqrt((cur->x - next->x) * (cur->x - next->x) + (cur->y - next->y) * (cur->y - next->y)) <= threshold)
				next = v.erase(next);
			else
				next++;
		}
		cur++;
	}

	cur = v.begin();
	//align points on same horizontal line
	while (cur != v.end()) {
		auto next = cur + 1;
		while (next != v.end()) {
			if (next->y-cur->y <= threshold)
				next->y = cur->y;
			next++;
		}
		cur++;
	}
}

void findRectRegions(vector <Point> pByY, vector <RectRegion>& toFill, const Mat & I) {
	int minDim = I.cols < I.rows ? I.cols : I.rows;
	int minCorner = minDimRatio * minDim;

	int cumMin = 0;
	// the vectors pBy.. are the points sorted by ..
	Mat tmp = I;

	filterPoints(pByY, 0.5 * minCorner); cout << "Threshold:" << 0.5 * minCorner << " Min corner:" << minCorner << endl;
	sort(pByY.begin(), pByY.end(), isLessByY);
	//cout << "Filtered of length: " << pByY.size() << " \n" << pByY << endl;
	for (auto u : pByY) { circle(I, u, 8, { 0,0,255 }); } imshow("Filtered corners", tmp);
	vector <Point> pByX(pByY);
	int n = pByY.size();
	sort(pByX.begin(), pByX.end(), isLessByX);

	//int xMargin = (int) (cornersMarginThresh * I.cols);
	//int yMargin = (int) (cornersMarginThresh * I.rows);
	int xMargin = (int) (cornersMarginThresh * minDim);
	int yMargin = (int) (cornersMarginThresh * minDim);

	for (int i = 0; i < n; i++) {
		auto& p = pByY[i];
		if (!isInAnyRegion(p, toFill)) {
			int startX = p.x, startY = p.y;
			Point leftCorner(0, 0), rightCorner(0, 0);
			int height = 0, width = 0;

			// assume it is an upper left corner => need to find width and height if applicable

			//going along Y
			for (int j = i + 1; j < n; j++) {
				auto& candidate = pByY[j];
				if (!isInAnyRegion(candidate, toFill)) {
					if (abs(candidate.x - startX) <= xMargin && (candidate.y - startY)>=minCorner) {
						int minX = candidate.x < startX ? candidate.x : startX;
						startX = minX; leftCorner = Point(minX, candidate.y);
						break;
					}
				}
			}

			//going along X
			int xIdx = lower_bound(pByX.begin(), pByX.end(), p, isLessByX) - pByX.begin();
			for (int j = xIdx + 1; j < n; j++) {
				auto& candidate = pByX[j];
				if (!isInAnyRegion(candidate, toFill)) {
					if (abs(candidate.y - startY) <= yMargin && (candidate.x - startX) >= minCorner) {
						int minY = candidate.y < startY ? candidate.y : startY;
						startY = minY; rightCorner = Point(candidate.x, minY);
						width = candidate.x - startX;
						break;
					}
				}
			}

			if (leftCorner != Point(0, 0) && rightCorner != Point(0, 0)) {
				height = leftCorner.y - startY;
				width = rightCorner.x - startX;
			}

			if (height != 0 && width != 0) {
				int curMinDim = height < width ? height : width;
				if (cumMin != 0) {
					double average = 1.0 * cumMin / toFill.size();
					if ((curMinDim / average - 1) >= 0.5) // if the current minDim is at least 2 more than the average skip
						continue;
				}
				cumMin += curMinDim;
				RectRegion region; region.corner = Point(startX,startY); region.height = height; region.width = width;
				toFill.push_back(region);
			}
		}
	}
}

void scaleRegions(vector <RectRegion>& ret, double scale, const Mat& I) {
	if (scale == 0) { return; }
	int rows = I.rows, cols = I.cols;
	for (auto& region : ret) {
		Point prevCorner = region.corner;
		int prevWidth = region.width, prevHeight = region.height;
		int addedW = scale * prevWidth, addedH = scale * prevHeight;
		int newX = prevCorner.x - addedW / 2; newX = newX < 0 ? 0 : newX;
		int newY = prevCorner.y - addedH / 2; newY = newY < 0 ? 0 : newY;
		int newW = newX + prevWidth + addedW > I.cols ? I.cols - newX : prevWidth + addedW;
		int newH = newY + prevHeight + addedH > I.rows ? I.rows - newY : prevHeight + addedH;
		region.corner = Point(newX, newY);
		region.height = newH;
		region.width = newW;
	}
}

vector<RectRegion> getRegions(const Mat& I, Vec3b colorLower, Vec3b colorUpper) {
	Mat selectedRegion = selectColor(I, colorLower, colorUpper); imshow("Selected Region", selectedRegion);

	// Harris requires the input to be in grayscale
	Mat forHarris; cvtColor(selectedRegion, forHarris, COLOR_RGB2GRAY);
	vector <Point> corners = harris(forHarris);
	vector <RectRegion> ret;
	findRectRegions(corners, ret, I);
	scaleRegions(ret, regionScale,I);
	return ret;
}

int debugCorners()
{
	Mat I = imread(CD::srcDir + "/data/cmp_b0377.png");
	Mat out = Mat::zeros(I.size(), CV_8UC3);

	imshow("Input", I);

	Vec3b color = { 255, 85, 0 };
	vector<RectRegion> regions = getRegions(I, color,color);

	for (auto& r : regions) {
		rectangle(out, Rect(r.corner.x, r.corner.y, r.width, r.height), color);
	}

	imshow("Regions, hopefully", out);

	waitKey();

	return 0;
}