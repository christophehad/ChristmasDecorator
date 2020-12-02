#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/intensity_transform.hpp>
#include <opencv2/core.hpp>
#include <opencv2/photo.hpp>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <stdlib.h>
#include <time.h>

#include "decorate.h"


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
    // cout << "Compactness = " << compactness << endl;
    imshow("not quantized", image);
    imshow("quantized", quantized_image);

    // if you would denoise here, all process of quantizin is lost
}


/** convert Mat image to HSV and scale V of every pixel by scale
 *  causes change of brightness of image
 */ 
Mat changeBrightness(const Mat& image, double scale)
{  

    int m = image.rows, n = image.cols;

    Mat out_bgr(m, n, CV_32FC3);
    Mat out_bgr8(m, n, CV_8U);

	Mat hsv(m, n, CV_32FC3);

    image.convertTo(hsv, CV_32FC3);
    cvtColor(hsv, hsv, COLOR_BGR2HSV, 3);

    int channel = 0;
    Mat out(m, n, CV_32FC3);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			// retrieve the pixel value at i,j as a Vec3f
			// overwrite the given channel value x with x*scale
			// write it the modified Vec3f to out.
			Vec3f c = 0;
			c = hsv.at<Vec3f>(i, j);
			c[channel] = c[channel];
			out.at<Vec3f>(i,j) = c * scale;
		}
	}
    
    cvtColor(out, out_bgr, COLOR_HSV2BGR, 3);

	out_bgr.convertTo(out_bgr8, CV_8U);

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

    lights = Mat::zeros(mask_boundary.rows, mask_boundary.cols, CV_8UC3);
    Mat lights_glow = Mat::zeros(mask_boundary.rows, mask_boundary.cols, CV_8UC3);


    // paint circels on mask boundary by choosing a random color from lights_color and shifting them randomly
    // away from the boundary, skip some pixels to create individual light bulb effects
    for (int r = 0; r < mask_boundary.rows; ++r)
    {
        for (int c = 0; c < mask_boundary.cols; ++c)
        {
            Vec3b pixel_color = mask_boundary(r, c);
            if (pixel_color != black)
            {   
                
                if ((c % 4== 0) || (r % 4== 0))
                {
                    Vec3b current_color = lights_color[rand()%lights_color.size()];
                    Point location_light = Point(c + (rand()%2),r + (rand()%2));

                    // create lights
                    circle(lights, location_light, 2, current_color, FILLED, LINE_4);     
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
    blur(lights, lights_glow, Size(30,30));
    lights_glow = lights_glow * 3;
    blur(lights, lights, Size(3,3));
    lights = lights * 0.7;

    // superpose lights and glow
    max(lights, lights_glow, bright_lights);

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
    Mat quantized_labels;
    labels.copyTo(quantized_labels);
    int nmb_of_clusters = 12;
    quantizeImageWithKmeans(labels, quantized_labels, nmb_of_clusters);

    // Get palette
    // taken from: https://stackoverflow.com/questions/35479344/how-to-get-a-color-palette-from-an-image-using-opencv
    map<Vec3b, int, lessVec3b> labels_map = getLabels(quantized_labels);

    // Print palette
    int area = image.rows * image.cols;

    for (auto color : labels_map)
    {
        cout << "Color: " << color.first << " \t - Area: " << 100.f * float(color.second) / float(area) << "%" << endl;
    }

    Vec3b background_color = Vec3b(170,0,0);

    // extract each color as mask with colors black and white
    Vec3b black = Vec3b(0,0,0);
    Vec3b white = Vec3b(255,255,255);

    vector<Mat> masks;

    // TODO: change to grey masks
    masks = getColorsAsColoredMasks(quantized_labels, labels_map, black, white);

    //  ASSUMPTION: background is biggest label
    map<Vec3b, int, lessVec3b> background_map;
    background_map[background_color] = 0;
    vector<Mat> background_mask;
    background_mask = getColorsAsColoredMasks(quantized_labels, background_map, white, black);


    imshow("background mask", background_mask[0]);
    waitKey(0);
    
    
    Mat decImage;
    Mat G;

    Mat lights;
    Mat bright_image;
    Mat bright_blue;
    Mat frame;


    // define the colors of the lights to be added
    // orange, red, yellow
    // vector<Vec3b> lights_colors = {Vec3b(0,69,255), Vec3b(0,165,255), Vec3b(51,255,255)};
    // red, yellow, blue, green
    vector<Vec3b> lights_colors = {Vec3b(0,255,0), Vec3b(255,0,0), Vec3b(0,0,255), Vec3b(0,255,255)};


    int imcount = 0;
    time_t timer;

    Mat image_blue;
    image.copyTo(image_blue);

    image_blue = increaseColor(image, 1.4, 0);
    image_blue = increaseColor(image, 3.0, 0);

    bool crop_lights_to_labels = true;

    for (auto single_mask : masks)
    {

        imshow("single mask", single_mask);

        Mat single_mask_deblurred;
        blur( single_mask, single_mask, Size(5,5) );
        fastNlMeansDenoisingColored(single_mask,single_mask_deblurred, 10, 10);
        //Canny( single_mask, single_mask_deblurred, 254, 254 ,7,0);
        
        imshow("single mask denoised", single_mask_deblurred);
        image_blue = increaseColor(image, 1.4, 0);
        intensity_transform::gammaCorrection(image_blue, bright_blue, 4);
        
        imshow("gammacorrection", bright_blue);
        Mat bright_blue_bg;
        addWeighted(bright_blue, 0.9, background_mask[0], 0.5, 0, bright_blue_bg);
        imshow("background darkened", bright_blue_bg);
        //imwrite("../results/image_neutral.png", image);
        //imwrite("../results/image_dark.png", bright_blue);
        

        // replace edges from mask by lights with lights_colors
        getMaskAsLights(single_mask, bright_blue, lights, lights_colors, crop_lights_to_labels);

        imshow("Decorated image", bright_blue);
        //imwrite("../results/image_dark_decorated.png", bright_blue);
        //imshow("dec", decImage);
        waitKey(0);

        // sharpen image
        //GaussianBlur(decImage,frame, cv::Size(0, 0), 3);
        //addWeighted(decImage, 1.5, frame, -0.5, 0, frame);
        //imshow("sharpened image", frame);

        // save image to file with unique name
        time_t t = time(0);   // get time now
        struct tm * now = localtime( & t );
        char buffer [80];
        strftime (buffer,80,"%Y-%m-%d-%H-%M-%S",now);
        string result_path = "../results/dec";
        result_path.append(buffer);
        result_path.append(to_string(imcount));
        result_path.append(".png");
        //imwrite(result_path, bright_blue);

        imcount++;
    }

    

    return 0;
}
