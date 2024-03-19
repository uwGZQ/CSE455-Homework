#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>


#include "image.h"
#define TWOPI 6.2831853

/******************************** Resizing *****************************
  To resize we'll need some interpolation methods and a function to create
  a new image and fill it in with our interpolation methods.
************************************************************************/

float nn_interpolate(image im, float x, float y, int c)
{
    // TODO
    /***********************************************************************
      This function performs nearest-neighbor interpolation on image "im"
      given a floating column value "x", row value "y" and integer channel "c",
      and returns the interpolated value.
    ************************************************************************/
    x = round(x);
    y = round(y);
    return get_pixel(im, x, y, c);
}

image nn_resize(image im, int w, int h)
{
    // TODO Fill in (also fix the return line)
    /***********************************************************************
      This function uses nearest-neighbor interpolation on image "im" to a new
      image of size "w x h"
    ************************************************************************/
    int c, x, y;
    float x_r = (float)im.w / (float)w;
    float y_r = (float)im.h / (float)h;
    image new_img = make_image(w, h, im.c);
    for (c = 0; c < im.c; c++) {
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                set_pixel(new_img, x, y, c, nn_interpolate(im, (x + 0.5) * x_r - 0.5, (y + 0.5) * y_r - 0.5, c));
            }
        }
    }
    return new_img;

}

float bilinear_interpolate(image im, float x, float y, int c)
{
    // TODO
    /***********************************************************************
      This function performs bilinear interpolation on image "im" given
      a floating column value "x", row value "y" and integer channel "c".
      It interpolates and returns the interpolated value.
    ************************************************************************/
    int x_low = floor(x);
    int x_high = ceil(x);
    int y_low = floor(y);
    int y_high = ceil(y);
    float v1 = get_pixel(im, x_low, y_low, c);
    float v2 = get_pixel(im, x_high, y_low, c);
    float v3 = get_pixel(im, x_low, y_high, c);
    float v4 = get_pixel(im, x_high, y_high, c);
    float d1 = x - x_low;
    float d2 = x_high - x;
    float d3 = y - y_low;
    float d4 = y_high - y;
    float A1 = d2 * d4;
    float A2 = d1 * d4;
    float A3 = d2 * d3;
    float A4 = d1 * d3;
    return A1 * v1 + A2 * v2 + A3 * v3 + A4 * v4;
}

image bilinear_resize(image im, int w, int h)
{
    // TODO
    /***********************************************************************
      This function uses bilinear interpolation on image "im" to a new image
      of size "w x h". Algorithm is same as nearest-neighbor interpolation.
    ************************************************************************/
    int c, x, y;
    float x_r = (float)im.w / (float)w;
    float y_r = (float)im.h / (float)h;
    image new_img = make_image(w, h, im.c);
    for (c = 0; c < im.c; c++) {
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                set_pixel(new_img, x, y, c, bilinear_interpolate(im, (x + 0.5) * x_r - 0.5, (y + 0.5) * y_r - 0.5, c));
            }
        }
    }
    return new_img;
}


/********************** Filtering: Box filter ***************************
  We want to create a box filter. We will only use square box filters.
************************************************************************/

void l1_normalize(image im)
{
    // TODO
    /***********************************************************************
      This function divides each value in image "im" by the sum of all the
      values in the image and modifies the image in place.
    ************************************************************************/
    int i;
    float sum = 0;
    for (i = 0; i < im.w * im.h * im.c; i++) {
        sum += im.data[i];
    }
    for (i = 0; i < im.w * im.h * im.c; i++) {
        im.data[i] /= sum;
    }
}

image make_box_filter(int w)
{
    // TODO
    /***********************************************************************
      This function makes a square filter of size "w x w". Make an image of
      width = height = w and number of channels = 1, with all entries equal
      to 1. Then use "l1_normalize" to normalize your filter.
    ************************************************************************/
    image filter = make_image(w, w, 1);
    int i;
    for (i = 0; i < w * w; i++) {
        filter.data[i] = 1;
    }
    l1_normalize(filter);
    return filter;
}

image convolve_image(image im, image filter, int preserve)
{
    // TODO
    /***********************************************************************
      This function convolves the image "im" with the "filter". The value
      of preserve is 1 if the number of input image channels need to be 
      preserved. Check the detailed algorithm given in the README.  
    ************************************************************************/
    assert(filter.c == im.c || filter.c == 1);
    image out_img;
    if (preserve == 0)
        out_img = make_image(im.w, im.h, 1);
    else
        out_img = make_image(im.w, im.h, im.c);
    if(preserve == 0 && filter.c == im.c){
        for(int i = 0; i < im.w; i++){
            for(int j = 0; j < im.h; j++){
                float sum = 0;
                for(int k = 0; k < im.c; k++){
                    for(int l = 0; l < filter.w; l++){
                        for(int m = 0; m < filter.h; m++){
                            sum += get_pixel(im, i + l - filter.w / 2, j + m - filter.h / 2, k) * get_pixel(filter, l, m, k);
                        }
                    }
                }
                set_pixel(out_img, i, j, 0, sum);
            }
        }
    }
    else if(preserve == 1 && filter.c == im.c){
        for(int i = 0; i < im.w; i++){
            for(int j = 0; j < im.h; j++){
                for(int k = 0; k < im.c; k++){
                    float sum = 0;
                    for(int l = 0; l < filter.w; l++){
                        for(int m = 0; m < filter.h; m++){
                            sum += get_pixel(im, i + l - filter.w / 2, j + m - filter.h / 2, k) * get_pixel(filter, l, m, k);
                        }
                    }
                    set_pixel(out_img, i, j, k, sum);
                }
            }
        }
    }
    else if(preserve == 0 && filter.c != im.c){
        for(int i = 0; i < im.w; i++){
            for(int j = 0; j < im.h; j++){
                float sum = 0;
                for(int k = 0; k < im.c; k++){
                    
                    for(int l = 0; l < filter.w; l++){
                        for(int m = 0; m < filter.h; m++){
                            sum += get_pixel(im, i + l - filter.w / 2, j + m - filter.h / 2, k) * get_pixel(filter, l, m, 0);
                        }
                    }
                    
                }
                set_pixel(out_img, i, j, 0, sum);
            }
        }
    }
    else if(preserve == 1 && filter.c != im.c){
        for(int i = 0; i < im.w; i++){
            for(int j = 0; j < im.h; j++){
                for(int k = 0; k < im.c; k++){
                    float sum = 0;
                    for(int l = 0; l < filter.w; l++){
                        for(int m = 0; m < filter.h; m++){
                            sum += get_pixel(im, i + l - filter.w / 2, j + m - filter.h / 2, k) * get_pixel(filter, l, m, 0);
                        }
                    }
                    set_pixel(out_img, i, j, k, sum);
                }
            }
        }
    }
    return out_img;

}

image make_highpass_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 filter with highpass filter values using image.data[]
    ************************************************************************/
    image filter = make_image(3, 3, 1);
    filter.data[0] = 0;
    filter.data[1] = -1;
    filter.data[2] = 0;
    filter.data[3] = -1;
    filter.data[4] = 4;
    filter.data[5] = -1;
    filter.data[6] = 0;
    filter.data[7] = -1;
    filter.data[8] = 0;
    return filter;
}

image make_sharpen_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 filter with sharpen filter values using image.data[]
    ************************************************************************/
    image filter = make_image(3, 3, 1);
    filter.data[0] = 0;
    filter.data[1] = -1;
    filter.data[2] = 0;
    filter.data[3] = -1;
    filter.data[4] = 5;
    filter.data[5] = -1;
    filter.data[6] = 0;
    filter.data[7] = -1;
    filter.data[8] = 0;


    return filter;
}

image make_emboss_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 filter with emboss filter values using image.data[]
    ************************************************************************/
    
    image filter = make_image(3, 3, 1);
    filter.data[0] = -2;
    filter.data[1] = -1;
    filter.data[2] = 0;
    filter.data[3] = -1;
    filter.data[4] = 1;
    filter.data[5] = 1;
    filter.data[6] = 0;
    filter.data[7] = 1;
    filter.data[8] = 2;
    return filter;

}

// Question 2.3.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Answer: We should use preserve for sharpen and emboss filters because they are applied to all three bands. 
//         We user emboss kernel to create a textured, three-dimensional effect in images, and use sharpen kernel to enhance the clarity and detail in an image.
//         We should not use preserve for highpass filter because it is used to find edges and applied to the graytone image.

// Question 2.3.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: We have to do post-processing for sharpen and emboss filters because they have negative values. We need to clamp the values.

image make_gaussian_filter(float sigma)
{
    // TODO
    /***********************************************************************
      sigma: a float number for the Gaussian.
      Create a Gaussian filter with the given sigma. Note that the kernel size 
      is the next highest odd integer from 6 x sigma. Return the Gaussian filter.
    ************************************************************************/
   /*Fill in image make_gaussian_filter(float sigma) which will take a standard deviation value sigma and return a filter that smooths using a gaussian with that sigma. How big should the filter be? 99% of the probability mass for a gaussian is within +/- 3 standard deviations, so make the kernel be 6 times the size of sigma. But also we want an odd number, so make it be the next highest odd integer from 6 x sigma. We need to fill in our kernel with some values (take care of the 0.5 offset for the pixel co-ordinates). Use the probability density function for a 2D gaussian:
   Technically this isn't perfect, what we would really want to do is integrate over the area covered by each cell in the filter. But that's much more complicated and this is a decent estimate. Remember though, this is a blurring filter so we want all the weights to sum to 1 (i.e. normalize the filter). Now you should be able to try out your new blurring function:*/
    int size = ceil(6 * sigma);
    if (size % 2 == 0) size++;
    image filter = make_image(size, size, 1);
    int i, j;
    float sum = 0;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            float x = i - size / 2;
            float y = j - size / 2;
            float v = 1 / (TWOPI * sigma * sigma) * exp(-(x * x + y * y) / (2 * sigma * sigma));
            filter.data[i * size + j] = v;
            sum += v;
        }
    }
    for (i = 0; i < size * size; i++) {
        filter.data[i] /= sum;
    }
    return filter;
}

image add_image(image a, image b)
{
    // TODO
    /***********************************************************************
      The input images a and image b have the same height, width, and channels.
      Sum the given two images and return the result, which should also have
      the same height, width, and channels as the inputs. Do necessary checks.
    ************************************************************************/
    /*For this task you'll have to extract the high frequency and low frequency from some images. You already know how to get low frequency, using your gaussian filter. To get high frequency you just subtract the low frequency data from the original image. You will probably need different values for each image to get it to look good.

TO DO

Fill in image add_image(image a, image b) to add two images a and b and image sub_image(image a, image b) to subtract image b from image a, so that we can perform our transformations of + and - like this:  The functions MUST include some checks that the images are the same size using assert(). Now we should be able to get these results:*/
    assert(a.w == b.w && a.h == b.h && a.c == b.c);
    image out_img = make_image(a.w, a.h, a.c);
    for (int i = 0; i < a.w * a.h * a.c; i++) {
        out_img.data[i] = a.data[i] + b.data[i];
    }
    return out_img;
}

image sub_image(image a, image b)
{
    // TODO
    /***********************************************************************
      The input image a and image b have the same height, width, and channels.
      Subtract the given two images and return the result, which should have
      the same height, width, and channels as the inputs. Do necessary checks.
    ************************************************************************/
    assert(a.w == b.w && a.h == b.h && a.c == b.c);
    image out_img = make_image(a.w, a.h, a.c);
    for (int i = 0; i < a.w * a.h * a.c; i++) {
        out_img.data[i] = a.data[i] - b.data[i];
    }
    return out_img;
}

image make_gx_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 Sobel Gx filter and return it
    ************************************************************************/
    image filter = make_image(3, 3, 1);
    filter.data[0] = -1;
    filter.data[1] = 0;
    filter.data[2] = 1;
    filter.data[3] = -2;
    filter.data[4] = 0;
    filter.data[5] = 2;
    filter.data[6] = -1;
    filter.data[7] = 0;
    filter.data[8] = 1;
    return filter;
}

image make_gy_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 Sobel Gy filter and return it
    ************************************************************************/
    image filter = make_image(3, 3, 1);
    filter.data[0] = -1;
    filter.data[1] = -2;
    filter.data[2] = -1;
    filter.data[3] = 0;
    filter.data[4] = 0;
    filter.data[5] = 0;
    filter.data[6] = 1;
    filter.data[7] = 2;
    filter.data[8] = 1;
    return filter;
}

void feature_normalize(image im)
{
    // TODO
    /***********************************************************************
      Calculate minimum and maximum pixel values. Normalize the image by
      subtracting the minimum and dividing by the max-min difference.
    ************************************************************************/
    float min = im.data[0];
    float max = im.data[0];
    for (int i = 0; i < im.w * im.h * im.c; i++) {
        if (im.data[i] < min) min = im.data[i];
        if (im.data[i] > max) max = im.data[i];
    }
    float range = max - min;
    if (range == 0) {
        for (int i = 0; i < im.w * im.h * im.c; i++) {
            im.data[i] = 0;
        }
    }
    else {
        for (int i = 0; i < im.w * im.h * im.c; i++) {
            im.data[i] = (im.data[i] - min) / range;
        }
    }
}

image *sobel_image(image im)
{
    // TODO
    /***********************************************************************
      Apply Sobel filter to the input image "im", get the magnitude as sobelimg[0]
      and gradient as sobelimg[1], and return the result.
    ************************************************************************/
    image *sobelimg = calloc(2, sizeof(image));
    sobelimg[0] = make_image(im.w, im.h, 1);
    sobelimg[1] = make_image(im.w, im.h, 1);
    image gx_filter = make_gx_filter();
    image gy_filter = make_gy_filter();
    image gx_img = convolve_image(im, gx_filter, 0);
    image gy_img = convolve_image(im, gy_filter, 0);
    for (int i = 0; i < im.w * im.h; i++) {
        sobelimg[0].data[i] = sqrt(gx_img.data[i] * gx_img.data[i] + gy_img.data[i] * gy_img.data[i]);
        sobelimg[1].data[i] = atan2(gy_img.data[i], gx_img.data[i]);
    }
    feature_normalize(sobelimg[0]);
    feature_normalize(sobelimg[1]);

    return sobelimg;
}

image colorize_sobel(image im)
{
  // TODO
  /***********************************************************************
    Create a colorized version of the edges in image "im" using the 
    algorithm described in the README.
  ************************************************************************/
    image *sobelimg = sobel_image(im);
    image out_img = make_image(im.w, im.h, im.c);
    for (int i = 0; i < im.w * im.h; i++) {
        out_img.data[i] = sobelimg[0].data[i];
        out_img.data[i + im.w * im.h] = sobelimg[1].data[i];
        out_img.data[i + 2 * im.w * im.h] = sobelimg[1].data[i];
    }
    hsv_to_rgb(out_img);
    return out_img;
  
}

// EXTRA CREDIT: Median filter

int compare(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}
image apply_median_filter(image im, int kernel_size) {
    int edge = kernel_size / 2;
    image out_image = make_image(im.w, im.h, im.c); 

    float *window = malloc(kernel_size * kernel_size * sizeof(float));

    for (int k = 0; k < im.c; ++k) {
        for (int j = 0; j < im.h; ++j) {
            for (int i = 0; i < im.w; ++i) {
                int count = 0;
                for (int y = -edge; y <= edge; ++y) {
                    for (int x = -edge; x <= edge; ++x) {
                        int curX = i + x;
                        int curY = j + y;
                        if (curX < 0) {
                            curX = 0;
                        } else if (curX >= im.w) {
                            curX = im.w - 1;
                        }

                        if (curY < 0) {
                            curY = 0;
                        } else if (curY >= im.h) {
                            curY = im.h - 1;
                        }
                        window[count++] = get_pixel(im, curX, curY, k); 
                    }
                }
                qsort(window, count, sizeof(float), compare);
                set_pixel(out_image, i, j, k, window[count / 2]); 
            }
        }
    }
    free(window);
    return out_image;
}




