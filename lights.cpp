#include "lights.h"
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