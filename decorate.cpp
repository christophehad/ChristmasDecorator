#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <stdlib.h>
#include <time.h>

#include "decorate.h"

Mat increaseBrightness(const Mat& image, double alpha, int beta)
{  
    Mat new_image = Mat::zeros( image.size(), image.type() );
    image.convertTo(new_image, -1, alpha, beta);
    return new_image;
}

Mat getMaskAsLights(const Mat3b& mask, Mat& lights, vector<Vec3b> lights_color){
    Vec3b black = Vec3b(0,0,0);

    lights = Mat::zeros(mask.rows, mask.cols, CV_8UC3);
    imshow("mask to decorate", mask);

    for (int r = 0; r < mask.rows; ++r)
    {
        for (int c = 0; c < mask.cols; ++c)
        {
            Vec3b pixel_color = mask(r, c);
            if (pixel_color != black)
            {   
                
                if ((c % 6== 0) || (r % 6== 0))
                {
                    circle(lights, Point(c + (rand()%2),r + (rand()%2)), 2, lights_color[rand()%lights_color.size()], FILLED, LINE_4);
                    
                }
            }
        }
    }
    blur(lights, lights, Size(3,3));

    Mat bright_lights;
    bright_lights = increaseBrightness(lights, 1.0, 50);
    
    imshow("lights alone", lights);
    imshow("bright lights alone", bright_lights);
    //waitKey(0);
    return bright_lights;
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
        // cout << "showing: " << color.first << endl;
        //imshow("Mask",masks);
        //waitKey(0);
    }
    
    return masks;
}


static void usage(char *s, int ntests){
    fprintf(stderr, "Usage: %s <test_number in 1..%d>\n", s, ntests);
}

int main(int argc, char *argv[]){
    if(argc < 2){
        // Tell the user how to run the program
        std::cerr << "Usage: " << argv[0] << " <IMAGEPATH> " << " <LABELPATH> " << std::endl;
        return 1;
    }
    string path_image = argv[1];
    string path_label = argv[2];

    // "../data/cmp_b0377.jpg"
    // ""../data/cmp_b0377.png"

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
    double alpha = 0.6; double beta; double input;
    image.copyTo(decImage);
    beta = ( 1.0 - alpha );

    Mat lights;
    Mat bright_image;
    Mat frame;
    vector<Vec3b> lights_colors = {Vec3b(0,255,0), Vec3b(255,0,0), Vec3b(0,0,255), Vec3b(0,255,255)};

    int imcount = 0;
    time_t timer;
    for (auto single_mask : masks)
    {
        Laplacian(single_mask, G, 0);
        getMaskAsLights(G, lights, lights_colors);
        addWeighted(image, alpha, lights, beta, 0.0, decImage);
        bright_image = increaseBrightness(decImage, 1.0, 30);
        imshow("Bounding box", decImage);

        // sharpen image
        GaussianBlur(decImage,frame, cv::Size(0, 0), 3);
        addWeighted(decImage, 1.5, frame, -0.5, 0, frame);
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
        imwrite(result_path, frame);

        imcount++;
    }

    

    return 0;
}
