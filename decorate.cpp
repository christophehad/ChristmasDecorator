#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/intensity_transform.hpp>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <stdlib.h>
#include <time.h>

#include "decorate.h"


// convert Mat image to HSV and scale V of every pixel by scale
// changes brightness of image
Mat changeBrightness(const Mat& image, double scale)
{  
    int m = image.rows, n = image.cols;

    Mat out_bgr(m, n, CV_32FC3);
    Mat out_bgr8(m, n, CV_8U);

	Mat hsv(m, n, CV_32FC3);

    image.convertTo(hsv, CV_32FC3);
    cvtColor(hsv, hsv, COLOR_BGR2HSV, 3);

    cout << "type" << hsv.type() << endl;

    int channel = 0;
    Mat out(m, n, CV_32FC3);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			// retrieve the pixel value at i,j as a Vec3f
			// overwrite the given channel value x with x+amount, modulo 1 (using fmod)
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

// scale a single color channel of Mat image
// e.g. for creating a blue shift
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

void getMaskAsLights(const Mat3b& mask, Mat& image_decorated, Mat& lights, vector<Vec3b> lights_color){
    Vec3b black = Vec3b(0,0,0);


    // extract edges from a single mask
    Mat3b mask_boundary;
    Laplacian(mask, mask_boundary, 0);

    lights = Mat::zeros(mask_boundary.rows, mask_boundary.cols, CV_8UC3);
    Mat lights_glow = Mat::zeros(mask_boundary.rows, mask_boundary.cols, CV_8UC3);
    //imshow("mask_boundary to decorate", mask_boundary);


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



    Mat lights_glow_masked;
    Mat image_glowing;
    Mat grey_mask;
    Mat lights_mask;
    Mat lights_glow_mask;

    cvtColor(mask, grey_mask, COLOR_BGR2GRAY, 1);
    cvtColor(lights, lights_mask, COLOR_BGR2GRAY, 1);

    // blur lights with filter of different sizes to create a ligth effect
    // additionally scale the glow to make it visible
    blur(lights, lights_glow, Size(30,30));
    lights_glow = lights_glow * 3;
    blur(lights, lights, Size(3,3));
    lights = lights * 0.7;


    // trying out different combination methods
    // from mask copy to addWeighted
    cvtColor(lights_glow, lights_glow_mask, COLOR_BGR2GRAY, 1);

    lights_glow.copyTo(lights_glow_masked, grey_mask);

    Mat bright_lights;

    max(lights, lights_glow, bright_lights);

    image_decorated.copyTo(image_glowing);
    lights_glow.copyTo(image_glowing, lights_glow_mask);
    lights.copyTo(image_decorated, lights_mask);

    double alpha = 0.8;
    double beta = 0.7;
    double gamma = 0;

    imshow("image decorated", image_decorated);
    imshow("image_glowing", image_glowing);

    Mat bright_lights_cropped;
    bright_lights.copyTo(bright_lights_cropped, grey_mask);

    //addWeighted(image_decorated, alpha, bright_lights_cropped, beta, gamma, image_decorated);
    //imshow("result cropped", image_decorated);

    addWeighted(image_decorated, alpha, bright_lights, beta, gamma, image_decorated);
    imshow("result", image_decorated);
    bright_lights.copyTo(lights);

}

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


// TODO: add colors as vector
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

    imshow("labels", labels);
    imshow("image", image);

    // Get palette
    // taken from: https://stackoverflow.com/questions/35479344/how-to-get-a-color-palette-from-an-image-using-opencv
    map<Vec3b, int, lessVec3b> labels_map = getLabels(labels);

    // Print palette
    int area = image.rows * image.cols;

    // for (auto color : labels_map)
    // {
    //     cout << "Color: " << color.first << " \t - Area: " << 100.f * float(color.second) / float(area) << "%" << endl;
    // }

    Vec3b black = Vec3b(0,0,0);
    Vec3b white = Vec3b(0,255,0);

    vector<Mat> masks;
    masks = getColorsAsColoredMasks(labels, labels_map, black, white);
    
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
    

    for (auto single_mask : masks)
    {
        image_blue = increaseColor(image, 1.4, 0);
        intensity_transform::gammaCorrection(image_blue, bright_blue, 3);
        imshow("gammacorrection", bright_blue);
        

        // replace edges from mask by lights with lights_colors
        getMaskAsLights(single_mask, bright_blue, lights, lights_colors);

        imshow("Bounding box", bright_blue);
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
        //imwrite(result_path, frame);

        imcount++;
    }

    

    return 0;
}
