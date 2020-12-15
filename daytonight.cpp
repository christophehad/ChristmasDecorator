#include <vector>
#include <opencv2/core.hpp>
#include <map>
#include "decorate.h"
#include "daytonight.h"


void calculateGradientOfImage(const Mat& image, Mat Ix, Mat Iy, Mat G)
{
    int m = image.rows, n = image.cols;
    // Image Gradient computation
	// Browse the rows and columns of the image and compute gradients using finite differences
    Mat image_grey;
    cvtColor(image, image_grey, COLOR_BGR2GRAY);
    Sobel(image_grey, Ix, CV_32FC1, 1, 0);
    Sobel(image_grey, Iy, CV_32FC1, 0, 1);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float ix, iy;
			ix = Ix.at<float>(i, j);
			iy = Iy.at<float>(i, j);
			G.at<float>(i, j) = sqrt(ix*ix + iy * iy);
		}
	}
}

/** Quantize image with kmeans
 * adapted from https://stackoverflow.com/questions/9575652/opencv-using-k-means-to-posterize-an-image
 * and https://stackoverflow.com/questions/49710006/fast-color-quantization-in-opencv
 *
 */
void quantizeImageWithKmeans(const Mat& image, Mat quantized_image, int nmb_clusters)
{
    Mat img;
    image.copyTo(img);
    img.convertTo(img, CV_32F);

    // reshape into featurevector to apply kmeans
    int origRows = img.rows;
    Mat colVec = img.reshape(1, img.rows*img.cols); // change to a Nx3 column vector
    Mat colVecD, bestLabels, centers, clustered;
    int attempts = 5;
    double eps = 0.001;
    colVec.convertTo(colVecD, CV_32FC3); // convert to floating point

    double compactness = kmeans(colVecD, nmb_clusters, bestLabels,
            TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, attempts, eps),
            attempts, KMEANS_PP_CENTERS, centers);

    Mat labelsImg = bestLabels.reshape(1, origRows); // single channel image of labels

    labelsImg.convertTo(labelsImg, CV_8U);
    centers.convertTo(centers, CV_8U);

    // assign each pixel to its corrresponding centroid
    for (int r = 0; r < labelsImg.rows; ++r)
    {
        for (int c = 0; c < labelsImg.cols; ++c)
        {
            int currlabel = labelsImg.at<uchar>(r,c);
            Vec3b center_color = Vec3b(centers.at<uchar>(currlabel, 0), centers.at<uchar>(currlabel, 1), centers.at<uchar>(currlabel, 2));
            quantized_image.at<Vec3b>(r,c) = center_color;
        }
    }
    // if you would denoise here, all process of quantizing is lost
}

/** convert Mat image to HSV and scale V of every pixel by scale
 *  causes change of brightness of image
 */
Mat changeHSVchannel(const Mat& image, double scale, int channel)
{
    imshow("pre hsv", image);
    int m = image.rows, n = image.cols;

    Mat out_bgr(m, n, CV_32FC3);
    Mat out_bgr8(m, n, CV_8U);

	Mat hsv(m, n, CV_32FC3);

    image.convertTo(hsv, CV_32FC3);
    cvtColor(hsv, hsv, COLOR_BGR2HSV, 3);

    Mat out(m, n, CV_32FC3);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			// retrieve the pixel value at i,j as a Vec3f
			// overwrite the given channel value x with x*scale
			// write it the modified Vec3f to out.
			Vec3f c = 0;
			c = hsv.at<Vec3f>(i, j);
			c[channel] = c[channel] * scale;
			out.at<Vec3f>(i,j) = c;
		}
	}

    cvtColor(out, out_bgr, COLOR_HSV2BGR, 3);

	out_bgr.convertTo(out_bgr8, CV_8UC3);
    imshow("post hsv", out_bgr8);
    return out_bgr8;
}
/** scale a single color channel of Mat image
 * e.g. for creating a blue shift
 * @params: image
 *          scale
 *          channel determines which channel of RGB image is scaled
 */
Mat increaseColor(const Mat& image, double scale, int channel)
{
    Mat new_image;
    image.copyTo(new_image);

    for (int r = 0; r < image.rows; ++r)
    {
        for (int c = 0; c < image.cols; ++c)
        {
            // do scaling in double
            unsigned char pixel = (unsigned char) std::min((double)image.at<Vec3b>(r, c)[channel] * scale, 255.0);
            new_image.at<Vec3b>(r, c)[channel] = pixel;
        }
    }
    return new_image;
}

Mat darkenSkyOfImage(const Mat& image)
{
    // quantize labels
    Mat quantized_image;
    image.copyTo(quantized_image);
    int nmb_of_clusters = 12;

    quantizeImageWithKmeans(image, quantized_image, 12);
    // replace sky
    vector<Mat> img_patches;
    vector<Mat> masks;
    Mat image_changed;

    float old_percentage = 0.0;
    float percentage;



    // calculate gradient of image
    int m = image.rows, n = image.cols;
	Mat Ix(m, n, CV_32F), Iy(m, n, CV_32F), G(m, n, CV_32F);
    calculateGradientOfImage(image, Ix, Iy, G);

    Mat gradient_char, gradient_mask;

    // smooth gradient
    G.convertTo(gradient_char, CV_8U);
    // imwrite("../report/gradient.png", gradient_char);
    blur(G, G, Size(5,5));
    G.convertTo(gradient_char, CV_8U);
    // imwrite("../report/gradient_smoothed.png", gradient_char);

    // get mask where gradient is smaller than threshhold
    threshold(gradient_char, gradient_mask, 5, 255, THRESH_BINARY_INV);
    // imwrite("../report/gradient_mask.png", gradient_mask);
    // only consider part of image where gradient is really smooth and now find color of sky

    Mat image_with_small_gradient = Mat::zeros(image.rows, image.cols, CV_8UC3);

    // extract each color as mask with colors black and white
    Vec3b black = Vec3b(0,0,0);
    Vec3b white = Vec3b(255,255,255);

    quantized_image.copyTo(image_with_small_gradient, gradient_mask);
    map<Vec3b, int, lessVec3b> image_map = getLabels(image_with_small_gradient);
    // imwrite("../report/quantized_image.png", quantized_image);
    // imwrite("../report/quantized_image_with_small_gradient.png", image_with_small_gradient);

    img_patches = getColorsAsColoredMasks(quantized_image, image_map, black, white);
    Mat image_darksky;
    Mat sky_patch;
    Mat patch_mask;
    for (auto patch : img_patches)
    {
        image.copyTo(image_changed);
        image_changed.convertTo(image_changed, CV_32FC3);
        // convert color to mask
        Mat patch_mask;
        patch.copyTo(patch_mask);

        cvtColor(patch, patch, COLOR_BGR2GRAY, 1);
        blur(patch, patch, Size(2,2));
        GaussianBlur(patch, patch, Size(5,5),1);

        //calculate percentage in top n rows
        Scalar total_val = sum(patch_mask);
        Scalar row_val = sum(patch_mask(Range(1,50), Range::all()));
        percentage = row_val[0]/total_val[0];

        if (percentage > old_percentage)
        {
            old_percentage = percentage;
            //cout << percentage << endl;
            patch.copyTo(sky_patch);
        }

    }
    // imwrite("../report/sky_patch.png", sky_patch);

    sky_patch.convertTo(patch_mask, CV_32F, 1.0 / 255, 0);
    patch_mask = patch_mask * 0.6;

    Vec3f background_color = {0,0,0};
    for (int i = 0; i < patch_mask.rows; i++)
    {
        for (int j = 0; j < patch_mask.cols; j++)
        {
            float b = ((1-patch_mask.at<float>(i,j)) * float(image.at<Vec3b>(i,j)[0]) + patch_mask.at<float>(i,j) * background_color[0]);
            float g = ((1-patch_mask.at<float>(i,j)) * float(image.at<Vec3b>(i,j)[1]) + patch_mask.at<float>(i,j) * background_color[1]);
            float r = ((1-patch_mask.at<float>(i,j)) * float(image.at<Vec3b>(i,j)[2]) + patch_mask.at<float>(i,j) * background_color[2]);

            Vec3f v = {b,g,r};
            image_changed.at<Vec3f>(i,j) = v;
        }
    }

    image_changed.convertTo(image_darksky, CV_8UC3);
    // imwrite("../report/image_darksky.png", image_darksky);
    return image_darksky;
}

Mat dayToNightTransfer(const Mat& image)
{
        Mat image_blueshifted, image_blueshifted_gamma_corrected, image_to_decorate;
        // do blueshift and gammacorrection
        image_blueshifted = increaseColor(image, 1.2, 0);
        // imwrite("../report/image_blueshift.png",image_blueshifted);
        // decrease Value
        image_blueshifted = changeHSVchannel(image_blueshifted, 0.8, 2);
        // imwrite("../report/image_brightness_decreased.png",image_blueshifted);

        // decrease saturation
        image_blueshifted = changeHSVchannel(image_blueshifted, 0.8, 1);
        // imwrite("../report/image_saturation_decreased.png",image_blueshifted);

        intensity_transform::gammaCorrection(image_blueshifted, image_blueshifted_gamma_corrected, 2);
        // imwrite("../report/image_gamma_corrected.png",image_blueshifted_gamma_corrected);
        return image_blueshifted_gamma_corrected;
}