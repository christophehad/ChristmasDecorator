#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "decorate.h"
#include "daytonight.h"


Mat getMaskAsGuirlandes(const Mat3b& mask, const Mat& image, Mat& lights, vector<Vec3b> lights_color, bool crop_to_mask){
    Vec3b black = Vec3b(0,0,0);

    // extract edges from a single mask
    Mat3b mask_boundary;
    int m = mask.rows, n = mask.cols;

    // Image Gradient computation in x direction

    Laplacian(mask, mask_boundary, 0);
    threshold(mask_boundary, mask_boundary, 10, 255, THRESH_BINARY);
    Sobel(mask_boundary, mask_boundary,0, 0, 1);
    // imshow("mask_boundary", mask_boundary);waitKey(0);

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
    Mat image_decorated;
    image.copyTo(image_decorated);
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
    return image_decorated;
}

/** Decorates image with lights according to boundaries given in mask
 * @params: mask: where on boundaries of mask, the lights are applied
 *          image_decorated: image to be decorated
 *          lights: returns the lights only if they are needed separately
 *          lights_color: vector with lights color that are randomly picked
 *          crop_to_mask: indicates wheter light and there mask shoul be cropped to mask, e.g. glow and lights only inside of windows
 */
Mat getMaskAsLights(const Mat3b& mask, const Mat& image, Mat& lights, vector<Vec3b> lights_color, bool crop_to_mask, bool window_glow){
    Vec3b black = Vec3b(0,0,0);
    Vec3b white = Vec3b(255,255,255);

    // extract edges from a single mask
    Mat3b mask_boundary, mask_boundary_x, mask_boundary_y;
    // imwrite("../report/mask.png",mask);

    Laplacian(mask, mask_boundary, 0);
    // imwrite("../report/mask_laplacian.png",mask_boundary);
    // remove weird artifacts from masks
    Sobel(mask_boundary, mask_boundary_y,0, 0, 1);
    // imshow("mask_boundary_y", mask_boundary_y);waitKey(0);
    threshold(mask_boundary_y, mask_boundary_y, 100, 255, THRESH_BINARY);
    Sobel(mask_boundary, mask_boundary_x,0, 1, 0);
    // imshow("mask_boundary_x", mask_boundary_x);waitKey(0);
    threshold(mask_boundary_x, mask_boundary_x, 100, 255, THRESH_BINARY);

    mask_boundary = mask_boundary_x + mask_boundary_y;
    threshold(mask_boundary, mask_boundary, 10, 255, THRESH_BINARY);
    // imwrite("../report/mask_boundary_thresholded.png",mask_boundary);

    // imshow("mask_boundary", mask_boundary);waitKey(0);
    // imshow("image to decorate", image_decorated); waitKey(0);

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

            if (pixel_color == white)
            {
                if ((r % 10 == 0)||(c % 10 == 0))
                {
                    bool light_in_proximity = false;
                    for (int i = 0; i < 7; i++){
                        for (int j = 0; j < 7; j++)
                        {
                            if (lights.at<Vec3b>(max(r-i,0), max(c-j,0)) != Vec3b(0,0,0))
                            {
                                //cout << lights.at<Vec3b>(max(r-i,0), max(c-j,0)) << endl;
                                light_in_proximity = true;
                            }
                        }
                    }
                    if (light_in_proximity == false){
                        Vec3b current_color = lights_color[color_ind%lights_color.size()];
                        Point location_light = Point(c + (rand()%2),r + (rand()%2));
                        // create lights
                        circle(lights, location_light, 1, current_color);
                        color_ind++;
                    }
                }
            }
        }
    }

    Mat grey_mask;
    Mat lights_mask;
    Mat bright_lights;
    Mat bright_lights_cropped;
    Mat grey_mask_smoothed;

    // imwrite("../report/decorated_mask.png", lights);

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
    // imwrite("../report/decorated_mask_glow.png", bright_lights);
    // imshow("bright_lights", bright_lights);
    // waitKey(0);

    // copy the lights separately to image to make them more bright
    Mat image_decorated;
    image.copyTo(image_decorated);

    lights.copyTo(image_decorated, lights_mask);

    double alpha = 0.8;
    double beta = 0.7;
    double gamma = 0;

    // create yellow window mask

    Mat window_mask = Mat::zeros(mask.rows, mask.cols, mask.type());
     for (int r = 0; r < mask.rows; r++)
    {
        for (int c = 0; c < mask.cols; c++)
        {
            Vec3b pixel_color = mask(r, c);
            if(pixel_color == white){
                window_mask.at<Vec3b>(r,c) = Vec3b(0,255,255);
            }
        }
    }
    blur(window_mask, window_mask, Size(3,3));
    // imwrite("../report/windows_glow.png", window_mask);

    if (window_glow)
    {
        addWeighted(bright_lights, 0.9, window_mask, 0.2, 0, bright_lights);
    }
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
    // imwrite("../report/decorated_image.png", image_decorated);
    return image_decorated;
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
    labels.copyTo(quantized_labels);
    int nmb_of_clusters = 12;

    quantizeImageWithKmeans(labels, quantized_labels, nmb_of_clusters);
    Mat image_darksky = darkenSkyOfImage(image);
    Mat image_night = dayToNightTransfer(image_darksky);

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
    vector<Mat> masks;

    masks = getColorsAsColoredMasks(quantized_labels, labels_map, black, white);

    // define the colors of the lights to be added
    // orange, red, yellow
    // vector<Vec3b> lights_colors = {Vec3b(0,69,255), Vec3b(0,165,255), Vec3b(51,255,255)};
    // red, yellow, blue, green
    Vec3b red = Vec3b(0,0,255);
    Vec3b yellow = Vec3b(0,255,255);
    Vec3b blue = Vec3b(255,0,0);
    Vec3b green = Vec3b(0,255,0);
    vector<Vec3b> lights_colors = {red, white};
    Vec3b gold = Vec3b(0,255,240);
    vector<Vec3b> guirland_colors = {white};

    int imcount = 0;
    time_t timer;
    bool crop_lights_to_labels = true;
    bool window_glow = false;

    Mat lights, image_blueshifted, image_blueshifted_gamma_corrected, image_to_decorate;
    Mat windows_labels;

    for (auto single_mask : masks)
    {
        if (imcount == WINDOWS_IDX) {
            // do stuff for windows mask
        }
        // blur the individual mask
        Mat single_mask_deblurred;
        // imwrite("../report/single_mask.png",single_mask);
        blur( single_mask, single_mask, Size(5,5));
        fastNlMeansDenoisingColored(single_mask,single_mask_deblurred, 10, 10);
        // imwrite("../report/single_mask_deblurred.png",single_mask_deblurred);

        // get mask where gradient is smaller than threshhold
        threshold(single_mask_deblurred, single_mask_deblurred, 5, 255, THRESH_BINARY);
        // imshow("mask deblurred", single_mask_deblurred);waitKey(0);

        // replace edges from mask by lights with lights_colors
        Mat image_decorated;
        image_decorated = getMaskAsLights(single_mask, image_night, lights, lights_colors, crop_lights_to_labels, window_glow);
        //image_decorated = getMaskAsGuirlandes(single_mask, image_night, lights, guirland_colors, false);

        imshow("Decorated image", image_decorated); waitKey(0);

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

    // decorate the windows :)
    Mat decorated_image;

    decorateWindows(image_night, labels, decorated_image);

    imshow("Decorated Image", decorated_image);
    imwrite(CD::srcDir + "/out/decorated-image.jpg", decorated_image);
    waitKey(0);

    return 0;
}
