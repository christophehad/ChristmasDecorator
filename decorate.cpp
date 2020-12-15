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
    // For aligning facades using the interactive selector, to comment/uncomment
    //mainAlign(); waitKey(); return 0;
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
    Mat image_with_mistletoes; decorateWindows(image, quantized_labels, image_with_mistletoes);
    Mat image_darksky = darkenSkyOfImage(image_with_mistletoes);
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
    Mat image_decorated;

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
        image_decorated.release();
        image_decorated = getMaskAsLights(single_mask, image_night, lights, lights_colors, crop_lights_to_labels, window_glow);
        //image_decorated = getMaskAsGuirlandes(single_mask, image_night, lights, guirland_colors, false);

        imshow("Decorated image", image_decorated); // waitKey(0);

        // save image to file with unique name
        time_t t = time(0);   // get time now
        struct tm * now = localtime( & t );
        char buffer [80];
        strftime (buffer,80,"%Y-%m-%d-%H-%M-%S",now);
        string result_path = "../results/dec";
        result_path.append(buffer);
        result_path.append(to_string(imcount));
        result_path.append(".png");
        imwrite(result_path, image_decorated);

        imcount++;
    }

    // decorate the windows :)
    Mat decorated_image = image_decorated;


    // imshow("Decorated Image", decorated_image);
    imwrite(CD::srcDir + "/out/decorated-image.jpg", decorated_image);
    waitKey(0);

    return 0;
}
