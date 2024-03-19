#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#include "matrix.h"
#include <time.h>

// Frees an array of descriptors.
// descriptor *d: the array.
// int n: number of elements in array.
void free_descriptors(descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(d[i].data);
    }
    free(d);
}

// Create a feature descriptor for an index in an image.
// image im: source image.
// int i: index in image for the pixel we want to describe.
// returns: descriptor for that index.
descriptor describe_index(image im, int i)
{
    int w = 5;
    descriptor d;
    d.p.x = i%im.w;
    d.p.y = i/im.w;
    d.data = calloc(w*w*im.c, sizeof(float));
    d.n = w*w*im.c;
    int c, dx, dy;
    int count = 0;
    // If you want you can experiment with other descriptors
    // This subtracts the central value from neighbors
    // to compensate some for exposure/lighting changes.
    for(c = 0; c < im.c; ++c){
        float cval = im.data[c*im.w*im.h + i];
        for(dx = -w/2; dx < (w+1)/2; ++dx){
            for(dy = -w/2; dy < (w+1)/2; ++dy){
                float val = get_pixel(im, i%im.w+dx, i/im.w+dy, c);
                d.data[count++] = cval - val;
            }
        }
    }
    return d;
}

// Marks the spot of a point in an image.
// image im: image to mark.
// ponit p: spot to mark in the image.
void mark_spot(image im, point p)
{
    int x = p.x;
    int y = p.y;
    int i;
    for(i = -9; i < 10; ++i){
        set_pixel(im, x+i, y, 0, 1);
        set_pixel(im, x, y+i, 0, 1);
        set_pixel(im, x+i, y, 1, 0);
        set_pixel(im, x, y+i, 1, 0);
        set_pixel(im, x+i, y, 2, 1);
        set_pixel(im, x, y+i, 2, 1);
    }
}

// Marks corners denoted by an array of descriptors.
// image im: image to mark.
// descriptor *d: corners in the image.
// int n: number of descriptors to mark.
void mark_corners(image im, descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        mark_spot(im, d[i].p);
    }
}

// Creates a 1d Gaussian filter.
// float sigma: standard deviation of Gaussian.
// returns: single row image of the filter.
/*You want a fast corner detector! You can decompose the Gaussian blur from one large 2D convolution to 2 1D convolutions. Instead of using an N x N filter you should convolve with a 1 x N filter followed by the same filter flipped to be N x 1. The size of the 2D Gaussian filter should be the same as described in HW2. The formula is given in the slides.*/
image make_1d_gaussian(float sigma) {
    int size = (int)roundf(6*sigma) | 1; 
    image filter = make_image(size, 1, 1); 
    float sum = 0.0; 
    int center = size / 2;
    for (int i = 0; i < size; i++) {
        float x = (float)(i - center);
        float value = expf(-(x*x) / (2*sigma*sigma)) / (sqrtf(2*M_PI) * sigma);
        set_pixel(filter, i, 0, 0, value);
        sum += value;
    }

    for (int i = 0; i < size; i++) {
        float normalized_value = get_pixel(filter, i, 0, 0) / sum;
        set_pixel(filter, i, 0, 0, normalized_value);
    }
    return filter;
}


// Smooths an image using separable Gaussian filter.
// image im: image to smooth.
// float sigma: std dev. for Gaussian.
// returns: smoothed image.
image smooth_image(image im, float sigma) {
    image filter = make_1d_gaussian(sigma);
    image horizontal_blur = convolve_image(im, filter, 0);

    image filter_transpose = make_image(1, filter.w, 1); // Make sure to adjust dimensions for the vertical filter
    for (int i = 0; i < filter.w; ++i) {
        float val = get_pixel(filter, i, 0, 0);
        set_pixel(filter_transpose, 0, i, 0, val);
    }

    image smoothed_image = convolve_image(horizontal_blur, filter_transpose, 0);
    free_image(filter);
    free_image(filter_transpose);
    free_image(horizontal_blur);

    return smoothed_image;
}


// Calculate the structure matrix of an image.
// image im: the input image.
// float sigma: std dev. to use for weighted sum.
// returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
//          third channel is IxIy.

image structure_matrix(image im, float sigma) {
    image gx = make_gx_filter();
    image gy = make_gy_filter();
    image Ix = convolve_image(im, gx, 0); 
    image Iy = convolve_image(im, gy, 0); 

    image Ixx = make_image(im.w, im.h, 1);
    image Iyy = make_image(im.w, im.h, 1);
    image Ixy = make_image(im.w, im.h, 1);
    for (int i = 0; i < im.w; ++i) {
        for (int j = 0; j < im.h; ++j) {
            float ix = get_pixel(Ix, i, j, 0);
            float iy = get_pixel(Iy, i, j, 0);
            set_pixel(Ixx, i, j, 0, ix * ix);
            set_pixel(Iyy, i, j, 0, iy * iy);
            set_pixel(Ixy, i, j, 0, ix * iy);
        }
    }
    image Sxx = smooth_image(Ixx, sigma);
    image Syy = smooth_image(Iyy, sigma);
    image Sxy = smooth_image(Ixy, sigma);

    image S = make_image(im.w, im.h, 3);
    for (int k = 0; k < im.w * im.h; ++k) {
        S.data[k] = Sxx.data[k];         // Ixx
        S.data[k + im.w * im.h] = Syy.data[k]; // Iyy
        S.data[k + 2 * im.w * im.h] = Sxy.data[k]; // Ixy
    }
    free_image(gx);
    free_image(gy);
    free_image(Ix);
    free_image(Iy);
    free_image(Ixx);
    free_image(Iyy);
    free_image(Ixy);
    free_image(Sxx);
    free_image(Syy);
    free_image(Sxy);

    return S;
}

// Estimate the cornerness of each pixel given a structure matrix S.
// image S: structure matrix for an image.
// returns: a response map of cornerness calculations.
image cornerness_response(image S) {
    image R = make_image(S.w, S.h, 1);
    float alpha = 0.06; 
    for (int y = 0; y < S.h; y++) {
        for (int x = 0; x < S.w; x++) {
            float Sxx = get_pixel(S, x, y, 0);
            float Syy = get_pixel(S, x, y, 1);
            float Sxy = get_pixel(S, x, y, 2);
            float det = Sxx * Syy - Sxy * Sxy; 
            float trace = Sxx + Syy; 
            float Rvalue = det - alpha * trace * trace;
            set_pixel(R, x, y, 0, Rvalue);
        }
    }
    return R;
}


// Perform non-max supression on an image of feature responses.
// image im: 1-channel image of feature responses.
// int w: distance to look for larger responses.
// returns: image with only local-maxima responses within w pixels.
image nms_image(image im, int w) {
    image r = copy_image(im);
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            float current_value = get_pixel(im, x, y, 0);
            int is_local_max = 1; 
            for (int dy = -w; dy <= w; dy++) {
                for (int dx = -w; dx <= w; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < im.w && ny >= 0 && ny < im.h) {
                        if (get_pixel(im, nx, ny, 0) > current_value) {
                            is_local_max = 0;
                            break;
                        }
                    }
                }
                if (!is_local_max) break; 
            }
            if (!is_local_max) {
                set_pixel(r, x, y, 0, -1000000); 
            }
        }
    }
    return r;
}

// Perform harris corner detection and extract features from the corners.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
// int *n: pointer to number of corners detected, should fill in.
// returns: array of descriptors of the corners in the image.
descriptor *harris_corner_detector(image im, float sigma, float thresh, int nms, int *n) {
    // Step 1: Calculate structure matrix
    image S = structure_matrix(im, sigma);

    // Step 2: Estimate cornerness
    image R = cornerness_response(S);

    // Step 3: Run NMS on the responses
    image Rnms = nms_image(R, nms);

    int count = 0;
    for (int i = 0; i < Rnms.w * Rnms.h; ++i) {
        if (Rnms.data[i] > thresh) count++;
    }
    
    *n = count; 
    descriptor *d = calloc(count, sizeof(descriptor));
    
    if (count == 0) return d;
    
    int idx = 0;
    for (int i = 0; i < Rnms.w; ++i) {
        for (int j = 0; j < Rnms.h; ++j) {
            if (get_pixel(Rnms, i, j, 0) > thresh) {
                int offset = j * Rnms.w + i;
                d[idx++] = describe_index(im, offset);
            }
        }
    }

    free_image(S);
    free_image(R);
    free_image(Rnms);

    return d;
}


// Find and draw corners on an image.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
void detect_and_draw_corners(image im, float sigma, float thresh, int nms)
{
    int n = 0;
    descriptor *d = harris_corner_detector(im, sigma, thresh, nms, &n);
    mark_corners(im, d, n);
}
