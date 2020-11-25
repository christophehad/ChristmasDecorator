#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include "decorate.h"

const string srcDir = "../../..";

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
        cout << "showing: " << color.first << endl;
        //imshow("Mask",masks);
        //waitKey(0);
    }
    
    return masks;
}


int main(int argc, char** argv)
{
    string inputPath1 = "/data/cmp_b0377";
    string inputPath2 = "/data/cmp_b0001";
    string inputPath = inputPath1;
    Mat image = imread(CD::srcDir + inputPath + ".jpg");
    Mat labels = imread(CD::srcDir + inputPath + ".png");
    imshow("labels", labels);
    imshow("image", image);

    // Get palette
    // taken from: https://stackoverflow.com/questions/35479344/how-to-get-a-color-palette-from-an-image-using-opencv
    map<Vec3b, int, lessVec3b> labels_map = getLabels(labels);

    // Print palette
    int area = image.rows * image.cols;

    for (auto color : labels_map)
    {
        cout << "Color: " << color.first << " \t - Area: " << 100.f * float(color.second) / float(area) << "%" << endl;
    }

    Vec3b black = Vec3b(0,0,0);
    Vec3b white = Vec3b(0,255,0);

    vector<Mat> masks;
    masks = getColorsAsColoredMasks(labels, labels_map, black, white);
    
    Mat decImage, windowsLabels;
    Mat G;
    double alpha = 0.5; double beta; double input;
    image.copyTo(decImage);
    beta = ( 1.0 - alpha );

    int index = 0;
    for (auto single_mask : masks)
    {
        Laplacian(single_mask, G, 0);
        addWeighted(image, alpha, G, beta, 0.0, decImage);
        imshow("Bounding box", decImage);
        if (index == WINDOWS_IDX)
            decImage.copyTo(windowsLabels);
        //waitKey(0);
        index++;
    }

    // decorate the windows :)
    decorateWindows(image, labels, image);

    imshow("Decorated Image", image);
    imwrite(CD::srcDir + "/out/decorated-image.jpg", image);
    waitKey();

    return 0;
}
