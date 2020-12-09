#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include "ChristmasDecorator.h"
#include "windows.h"
#include "align.h"

using namespace cv;
using namespace std;

const int WINDOWS_IDX = 9;

struct lessVec3b
{
    bool operator()(const Vec3b& lhs, const Vec3b& rhs) const {
        return (lhs[0] != rhs[0]) ? (lhs[0] < rhs[0]) : ((lhs[1] != rhs[1]) ? (lhs[1] < rhs[1]) : (lhs[2] < rhs[2]));
    }
};

void quantizeImageWithKmeans(const Mat& image, Mat quantized_image, int nmb_clusters);

Mat increaseColor(const Mat& image, double scale, int channel);

Mat changeHSVchannel(const Mat& image, double scale, int channel);
void getMaskAsGirlandes(const Mat3b& mask, Mat& image_decorated, Mat& lights, vector<Vec3b> lights_color, bool crop_to_mask);

void getMaskAsLights(const Mat3b& mask, Mat& image_decorated, Mat& lights, vector<Vec3b> lights_color, bool crop_to_mask, bool window_glow);

map<Vec3b, int, lessVec3b> getLabels(const Mat3b& src);

vector<Mat> getColorsAsColoredMasks(Mat labels, map<Vec3b, int, lessVec3b> label_map, Vec3b background_color, Vec3b label_color);