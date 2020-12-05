#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/intensity_transform.hpp>
#include <opencv2/core.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "decorate.h"

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

void getMaskAsGirlandes(const Mat3b& mask, Mat& image_decorated, Mat& lights, vector<Vec3b> lights_color, bool crop_to_mask){
    Vec3b black = Vec3b(0,0,0);

    // extract edges from a single mask
    Mat3b mask_boundary;
    int m = mask.rows, n = mask.cols;

    // Image Gradient computation in x direction
    
    Laplacian(mask, mask_boundary, 0);
    threshold(mask_boundary, mask_boundary, 10, 255, THRESH_BINARY);
    Sobel(mask_boundary, mask_boundary,0, 0, 1);
    imshow("mask_boundary", mask_boundary);waitKey(0);

    lights = Mat::zeros(mask_boundary.rows, mask_boundary.cols, CV_8UC3);

    // hang guirlands from horizontal edges
    int color_ind = 0;
    int maximal_guirland_length = 8;

    // do not put guirlandes on the floor
    int limit_down = 100;
    for (int r = 0; r < mask_boundary.rows-limit_down; r++)
    {
        for (int c = 0; c < mask_boundary.cols; c++)
        {
            Vec3b pixel_color = mask_boundary(r, c);
            
            if (pixel_color != black)
            {   
                if ((c % 6 == 0))
                {
                    // you don't want to many lights
                    // handles case when edge is lighted s.t. not too many lights are shown
                    
                    // create lights
                    Vec3b current_color = lights_color[color_ind%lights_color.size()];
                    int guirland_length = rand( ) % maximal_guirland_length + 4;
                    for(int ii = 0; ii < guirland_length; ii++)
                    {
                        Point location_light = Point(c+rand()%3-1, r + (ii * 3));
                        circle(lights, location_light, 1, current_color);
                    }
                    
                    color_ind++;   
                }
            }
        }
    }

    Mat grey_mask;
    Mat lights_mask;
    Mat bright_lights;
    Mat bright_lights_cropped;
    Mat grey_mask_smoothed;

    cvtColor(mask, grey_mask, COLOR_BGR2GRAY, 1);
    cvtColor(lights, lights_mask, COLOR_BGR2GRAY, 1);

    // blur lights with filter of different sizes to create a light effect
    // additionally scale the glow to make it visible
    Mat lights_glow, lights_glow1, lights_glow2, lights_glow3;
    blur(lights, lights, Size(3,3));
    blur(lights,lights_glow, Size(5,5));
    // blur(lights,lights_glow1, Size(6,6));
    // blur(lights,lights_glow2, Size(10,10));
    blur(lights,lights_glow3, Size(30,30));
    lights = lights;
    lights_glow = lights_glow*2;
    // lights_glow1 = lights_glow1 * 5;
    // lights_glow2 = lights_glow2 * 7;
    lights_glow3 = lights_glow3*2;

    // superpose lights and glow
    // max(lights, lights_glow1, lights);
    // max(lights, lights_glow2, lights);
    max(lights, lights_glow3, lights);
    max(lights, lights_glow, bright_lights);
    blur(bright_lights, bright_lights, Size(1,1));
    // imshow("bright_lights", bright_lights);
    // waitKey(0);

    // copy the lights separately to image to make them more bright
    lights.copyTo(image_decorated, lights_mask);

    double alpha = 0.8;
    double beta = 0.7;
    double gamma = 0;

    // smooth the mask to create smoother boundaries when lights and glow are cropped

    blur(grey_mask, grey_mask_smoothed, Size(10,10));

    bright_lights.copyTo(bright_lights_cropped, grey_mask_smoothed);

    if (crop_to_mask)
    {
        addWeighted(image_decorated, alpha, bright_lights_cropped, beta, gamma, image_decorated);
        bright_lights_cropped.copyTo(lights);
    }
    else
    {
        addWeighted(image_decorated, alpha, bright_lights, beta, gamma, image_decorated);
        bright_lights.copyTo(lights);
    }
}

/** Decorates image with lights according to boundaries given in mask
 * @params: mask: where on boundaries of mask, the lights are applied
 *          image_decorated: image to be decorated
 *          lights: returns the lights only if they are needed separately
 *          lights_color: vector with lights color that are randomly picked
 *          crop_to_mask: indicates wheter light and there mask shoul be cropped to mask, e.g. glow and lights only inside of windows
 */ 
void getMaskAsLights(const Mat3b& mask, Mat& image_decorated, Mat& lights, vector<Vec3b> lights_color, bool crop_to_mask){
    Vec3b black = Vec3b(0,0,0);

    // extract edges from a single mask
    Mat3b mask_boundary;
    Laplacian(mask, mask_boundary, 0);
    threshold(mask_boundary, mask_boundary, 10, 255, THRESH_BINARY);
    imshow("mask_boundary", mask_boundary);waitKey(0);

    lights = Mat::zeros(mask_boundary.rows, mask_boundary.cols, CV_8UC3);
    // imshow("maskbondary", mask_boundary);

    // paint circels on mask boundary by choosing a random color from lights_color and shifting them randomly
    // away from the boundary, skip some pixels to create individual light bulb effects
    int color_ind = 0;
    for (int r = 0; r < mask_boundary.rows; r++)
    {
        for (int c = 0; c < mask_boundary.cols; c++)
        {
            Vec3b pixel_color = mask_boundary(r, c);
            
            if (pixel_color != black)
            {   
                if ((c % 10 == 0) ^ (r % 10 == 0))
                {
                    // you don't want to many lights
                    // handles case when edge is lighted s.t. not two many lights are coming
                    Vec3b current_color = lights_color[color_ind%lights_color.size()];
                    Point location_light = Point(c + (rand()%2),r + (rand()%2));
                    // create lights
                    circle(lights, location_light, 1, current_color);
                    color_ind++;   
                }
            }
        }
    }

    Mat grey_mask;
    Mat lights_mask;
    Mat bright_lights;
    Mat bright_lights_cropped;
    Mat grey_mask_smoothed;

    cvtColor(mask, grey_mask, COLOR_BGR2GRAY, 1);
    cvtColor(lights, lights_mask, COLOR_BGR2GRAY, 1);

    // blur lights with filter of different sizes to create a light effect
    // additionally scale the glow to make it visible
    Mat lights_glow, lights_glow1, lights_glow2, lights_glow3;
    blur(lights, lights, Size(3,3));
    blur(lights,lights_glow, Size(5,5));
    blur(lights,lights_glow1, Size(6,6));
    blur(lights,lights_glow2, Size(10,10));
    blur(lights,lights_glow3, Size(30,30));
    lights = lights * 3;
    lights_glow = lights_glow * 4;
    lights_glow1 = lights_glow1 * 5;
    lights_glow2 = lights_glow2 * 7;
    lights_glow3 = lights_glow3 * 10;

    // superpose lights and glow
    max(lights, lights_glow1, lights);
    max(lights, lights_glow2, lights);
    max(lights, lights_glow3, lights);
    max(lights, lights_glow, bright_lights);
    blur(bright_lights, bright_lights, Size(1,1));
    // imshow("bright_lights", bright_lights);
    // waitKey(0);

    // copy the lights separately to image to make them more bright
    lights.copyTo(image_decorated, lights_mask);

    double alpha = 0.8;
    double beta = 0.7;
    double gamma = 0;

    // smooth the mask to create smoother boundaries when lights and glow are cropped

    blur(grey_mask, grey_mask_smoothed, Size(10,10));

    bright_lights.copyTo(bright_lights_cropped, grey_mask_smoothed);

    if (crop_to_mask)
    {
        addWeighted(image_decorated, alpha, bright_lights_cropped, beta, gamma, image_decorated);
        bright_lights_cropped.copyTo(lights);
    }
    else
    {
        addWeighted(image_decorated, alpha, bright_lights, beta, gamma, image_decorated);
        bright_lights.copyTo(lights);
    }
}

/**
 * get individual colors of src as map
 * credit goes to: taken from: https://stackoverflow.com/questions/35479344/how-to-get-a-color-palette-from-an-image-using-opencv
 */
map<Vec3b, int, lessVec3b> getLabels(const Mat3b& src)
{
    map<Vec3b, int, lessVec3b> palette;
    for (int r = 0; r < src.rows; ++r)
    {
        for (int c = 0; c < src.cols; ++c)
        {
            Vec3b color = src(r, c);
            if (palette.count(color) == 0)
            {

                palette[color] = 1;
            }
            else
            {
                palette[color] = palette[color] + 1;
            }
        }
    }
    return palette;
}


/** Get a color as a mask with label_color and background_color
 * 
 */
vector<Mat> getColorsAsColoredMasks(Mat labels, map<Vec3b, int, lessVec3b> label_map, Vec3b background_color, Vec3b label_color)
{
    vector<Mat> masks;
    for (auto color : label_map)
    {
        masks.push_back(labels.clone());
        for (int i = 0; i < labels.rows; i++)
        {
            for (int j = 0; j < labels.cols; j++)
            {
                if (labels.at<Vec3b>(i,j) == color.first)
                {
                    masks.back().at<Vec3b>(i,j) = label_color;
                }
                else
                {
                    masks.back().at<Vec3b>(i,j) = background_color;
                }  
            }
        }
    }
    
    return masks;
}


static void usage(char *s, int ntests){
    fprintf(stderr, "Usage: %s <test_number in 1..%d>\n", s, ntests);
}

int main(int argc, char *argv[]){
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <IMAGEPATH> " << " <LABELPATH> " << std::endl;
        return 1;
    }
    string path_image = argv[1];
    string path_label = argv[2];

    // Path to old images
    // "../data/cmp_b0377.jpg"
    // "../data/cmp_b0377.png"

    Mat image = imread(path_image);
    Mat labels = imread(path_label);

    if( image.empty() )
    {
        cout << "Couldn't load " << path_image << endl;
        return EXIT_FAILURE;
    }
    if( labels.empty() )
    {
        cout << "Couldn't load " << path_label << endl;
        return EXIT_FAILURE;
    }

    //imshow("labels", labels);
    //imshow("image", image);

    // quantize labels
    Mat quantized_labels, quantized_image;
    image.copyTo(quantized_image);
    labels.copyTo(quantized_labels);
    int nmb_of_clusters = 12;

    quantizeImageWithKmeans(labels, quantized_labels, nmb_of_clusters);
    quantizeImageWithKmeans(image, quantized_image, 12);

    
    // int area = image.rows * image.cols;
    // for (auto color : image_map)
    // {
    //     cout << "Color: " << color.first << " \t - Area: " << 100.f * float(color.second) / float(area) << "%" << endl;
    // }

    
    // Get palette
    // taken from: https://stackoverflow.com/questions/35479344/how-to-get-a-color-palette-from-an-image-using-opencv
    map<Vec3b, int, lessVec3b> labels_map = getLabels(quantized_labels);

    // extract each color as mask with colors black and white
    Vec3b black = Vec3b(0,0,0);
    Vec3b white = Vec3b(255,255,255);

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
    blur(G, G, Size(5,5));
    G.convertTo(gradient_char, CV_8U);
    imshow("gradient val", gradient_char);waitKey(0);

    // get mask where gradient is smaller than threshhold
    threshold(gradient_char, gradient_mask, 5, 255, THRESH_BINARY_INV);
    imshow("gradient mask", gradient_mask); waitKey(0);
    // only consider part of image where gradient is really smooth and now find color of sky

    Mat image_with_small_gradient = Mat::zeros(image.rows, image.cols, CV_8UC3);

    quantized_image.copyTo(image_with_small_gradient, gradient_mask);
    map<Vec3b, int, lessVec3b> image_map = getLabels(image_with_small_gradient);
    imshow("quantized image", quantized_image);waitKey(0);

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
            cout << percentage << endl;
            patch.copyTo(sky_patch);      
        }

    }
    imshow("patch", sky_patch);waitKey(0);
    
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
    imshow("im changed", image_darksky);waitKey(0);


    // TODO: change to grey masks
    masks = getColorsAsColoredMasks(quantized_labels, labels_map, black, white);

    // define the colors of the lights to be added
    // orange, red, yellow
    // vector<Vec3b> lights_colors = {Vec3b(0,69,255), Vec3b(0,165,255), Vec3b(51,255,255)};
    // red, yellow, blue, green
    vector<Vec3b> lights_colors = {Vec3b(0,255,0), Vec3b(255,0,0), Vec3b(0,0,255), Vec3b(0,255,255)};
    Vec3b gold = Vec3b(0,255,240);
    vector<Vec3b> guirland_colors = {gold};

    int imcount = 0;
    time_t timer;
    bool crop_lights_to_labels = true;

    Mat lights, image_blueshifted, image_blueshifted_gamma_corrected, image_to_decorate;

    for (auto single_mask : masks)
    {
        // blur the individual mask
        Mat single_mask_deblurred;
        blur( single_mask, single_mask, Size(5,5));
        fastNlMeansDenoisingColored(single_mask,single_mask_deblurred, 10, 10);
        // get mask where gradient is smaller than threshhold
        threshold(single_mask_deblurred, single_mask_deblurred, 5, 255, THRESH_BINARY);
        imshow("mask deblurred", single_mask_deblurred);waitKey(0);

        // do blueshift and gammacorrection
        image_blueshifted = increaseColor(image_darksky, 1.2, 0);
        intensity_transform::gammaCorrection(image_blueshifted, image_blueshifted_gamma_corrected, 2);
        
        // replace edges from mask by lights with lights_colors
        //getMaskAsLights(single_mask, image_blueshifted_gamma_corrected, lights, lights_colors, crop_lights_to_labels);
        getMaskAsGirlandes(single_mask, image_blueshifted_gamma_corrected, lights, guirland_colors, false);

        imshow("Decorated image", image_blueshifted_gamma_corrected); waitKey(0);

        // save image to file with unique name
        time_t t = time(0);   // get time now
        struct tm * now = localtime( & t );
        char buffer [80];
        strftime (buffer,80,"%Y-%m-%d-%H-%M-%S",now);
        string result_path = "../results/dec";
        result_path.append(buffer);
        result_path.append(to_string(imcount));
        result_path.append(".png");
        //imwrite(result_path, image_to_decorate);

        imcount++;
    }

    

    return 0;
}
