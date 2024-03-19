
/***********************************************************************
 * EXTRA CREDIT ATTEMPTED
 * This file contains image processing functions
 * including scale_image, rgb_to_lch and lch_to_rgb.
 ***********************************************************************/

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

float get_pixel(image im, int x, int y, int c)
{
    if (x < 0) x = 0;
    if (x >= im.w) x = im.w - 1;
    if (y < 0) y = 0;
    if (y >= im.h) y = im.h - 1;
    if (c < 0) c = 0;
    if (c >= im.c) c = im.c - 1;
    return im.data[c * im.w * im.h + y * im.w + x];

}

void set_pixel(image im, int x, int y, int c, float v)
{
    if (x < 0 || x >= im.w || y < 0 || y >= im.h || c < 0 || c >= im.c) return;
    im.data[c * im.w * im.h + y * im.w + x] = v;
    return;
}


image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    memcpy(copy.data, im.data, sizeof(float) * im.w * im.h * im.c);
    return copy;
}

//Y = 0.299 R + 0.587 G + .114 B
image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
    
    for (int i = 0; i < im.w; i++)
    {
        for (int j = 0; j < im.h; j++)
        {
            float r = get_pixel(im, i, j, 0);
            float g = get_pixel(im, i, j, 1);
            float b = get_pixel(im, i, j, 2);
            float y = 0.299 * r + 0.587 * g + 0.114 * b;
            set_pixel(gray, i, j, 0, y);
        }
    }
    return gray;
}

void shift_image(image im, int c, float v)
{
    // TODO Fill this in
    for (int i = 0; i < im.w; i++)
    {
        for (int j = 0; j < im.h; j++)
        {
            float p = get_pixel(im, i, j, c);
            set_pixel(im, i, j, c, p + v);
        }
    }
}

void clamp_image(image im)
{
    for (int i = 0; i < im.w; i++)
    {
        for (int j = 0; j < im.h; j++)
        {
            for (int k = 0; k < im.c; k++)
            {
                float p = get_pixel(im, i, j, k);
                if (p < 0) p = 0;
                if (p > 1) p = 1;
                set_pixel(im, i, j, k, p);
            }
        }
    }
}


// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    float r, g, b, H, S, V;
    for (int i = 0; i < im.w; i++)
    {
        for (int j = 0; j < im.h; j++)
        {
            r = get_pixel(im, i, j, 0);
            g = get_pixel(im, i, j, 1);
            b = get_pixel(im, i, j, 2);
            
            V = three_way_max(r, g, b);
            float m = three_way_min(r, g, b);
            float C = V - m;
            if (V == 0) S = 0;
            else S = C / V;
            if (C == 0) H = 0;
            else if (V == r) H = (g - b) / C;
            else if (V == g) H = (b - r) / C + 2;
            else if (V == b) H = (r - g) / C + 4;

            if (H < 0) H = H / 6 + 1;
            else H /= 6;
            set_pixel(im, i, j, 0, H);
            set_pixel(im, i, j, 1, S);
            set_pixel(im, i, j, 2, V);
        }
    }

}

void hsv_to_rgb(image im) {
    float H, S, V, r, g, b, P, Q, T, F;
    for (int i = 0; i < im.w; i++) {
        for (int j = 0; j < im.h; j++) {
            H = get_pixel(im, i, j, 0) * 6;
            S = get_pixel(im, i, j, 1);
            V = get_pixel(im, i, j, 2);

            F = fmodf(H, 1.0);
            P = V * (1 - S);
            Q = V * (1 - F * S);
            T = V * (1 - (1 - F) * S);

            int sector = floor(H);

            switch (sector) {
                case 0: r = V; g = T; b = P; break;
                case 1: r = Q; g = V; b = P; break;
                case 2: r = P; g = V; b = T; break;
                case 3: r = P; g = Q; b = V; break;
                case 4: r = T; g = P; b = V; break;
                case 5: default: r = V; g = P; b = Q; break;
            }

            set_pixel(im, i, j, 0, r);
            set_pixel(im, i, j, 1, g);
            set_pixel(im, i, j, 2, b);
        }
    }
}

/*
Implement void scale_image(image im, int c, float v) to 
scale a channel by a certain amount. 
This will give us better saturation results. 
Note, you will have to add the necessary lines to the header file(s) to include the function, 
it should be very similar to what's already there for shift_image. Now if we multiply (scale) saturation by 2 instead of just shifting it all up we get much better results:

*/
void scale_image(image im, int c, float v)
{
    for (int i = 0; i < im.w; i++)
    {
        for (int j = 0; j < im.h; j++)
        {
            float p = get_pixel(im, i, j, c);
            set_pixel(im, i, j, c, p * v);
        }
    }
}




/* 
Implement RGB to Hue, Chroma, Lightness, 
a perceptually more accurate version of Hue, Saturation, Value. Note, 
this will involve gamma decompression, 
converting to CIEXYZ, 
converting to CIELUV, 
converting to HCL, 
and the reverse transformations.
The upside is a similar colorspace to HSV but with better perceptual properties!
*/

/*
Reference: 
[1] http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
[2] https://www.wikiwand.com/en/CIELUV#XYZ_→_CIELUV_and_CIELUV_→_XYZ_conversions
[3] https://www.w3.org/TR/css-color-4/#color-conversion-code
*/
 
/* D65 */
// RGB to sRGB
float gamma_decompress(float c) {
    if (c <= 0.04045) {
        return c / 12.92;
    } else {
        return pow((c + 0.055) / 1.055, 2.4);
    }
}

//sRGB to XYZ conversion
void rgb_to_xyz(image im, int i, int j, float *X, float *Y, float *Z) {
    // Assuming sRGB color space with D65 white point
    float r = gamma_decompress(get_pixel(im, i, j, 0));
    float g = gamma_decompress(get_pixel(im, i, j, 1));
    float b = gamma_decompress(get_pixel(im, i, j, 2));

    // sRGB to XYZ conversion matrix
    *X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    *Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    *Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
}
//XYZ to LUV conversion
void xyz_to_luv(image im, int i, int j, float X, float Y, float Z, float *L, float *u, float *v) {
    // D65 white point
    float Xr = 0.95047, Yr = 1.0, Zr = 1.08883;

    float yr = Y / Yr;
    *L = (yr > 0.008856) ? (116.0 * cbrt(yr) - 16.0) : (903.3 * yr);

    float denom = X + 15.0 * Y + 3.0 * Z;
    float ur = (denom != 0) ? (4.0 * X / denom) : 0;
    float vr = (denom != 0) ? (9.0 * Y / denom) : 0;

    denom = Xr + 15.0 * Yr + 3.0 * Zr;
    float ur_prime = 4.0 * Xr / denom;
    float vr_prime = 9.0 * Yr / denom;

    *u = 13.0 * (*L) * (ur - ur_prime);
    *v = 13.0 * (*L) * (vr - vr_prime);
}
// LUV to HCL conversion
void luv_to_hcl(image im, int i, int j, float L, float u, float v, float *H, float *C, float *L_out) {
    *H = atan2(v, u);
    *C = sqrt(u * u + v * v);
    *L_out = L;
}

// RGB to HCL conversion
void rgb_to_hcl(image im) {
    for (int i = 0; i < im.w; i++) {
        for (int j = 0; j < im.h; j++) {
            float X, Y, Z, L, u, v, H, C, L_out;
            rgb_to_xyz(im, i, j, &X, &Y, &Z);
            xyz_to_luv(im, i, j, X, Y, Z, &L, &u, &v);
            luv_to_hcl(im, i, j, L, u, v, &H, &C, &L_out);
            set_pixel(im, i, j, 0, H);
            set_pixel(im, i, j, 1, C);
            set_pixel(im, i, j, 2, L_out);
        }
    }
}


// sRGB to RGB
float gamma_compress(float c) {
    if (c <= 0.0031308) {
        return 12.92 * c;
    } else {
        return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
    }
}

// HCL to LUV conversion
void hcl_to_luv(image im, int i, int j, float *L, float *u, float *v) {
    float H = get_pixel(im, i, j, 0);
    float C = get_pixel(im, i, j, 1);
    *L = get_pixel(im, i, j, 2);
    *u = cos(H) * C;
    *v = sin(H) * C;
}

// LUV to XYZ conversion
void luv_to_xyz(image im, int i, int j, float L, float u, float v, float *X, float *Y, float *Z) {
    // D65 white point
    float Xr = 0.95047, Yr = 1.0, Zr = 1.08883;

    float u_prime = u / (13 * L) + (4 * Xr) / (Xr + 15 * Yr + 3 * Zr);
    float v_prime = v / (13 * L) + (9 * Yr) / (Xr + 15 * Yr + 3 * Zr);

    *Y = (L > 7.999592) ? pow((L + 16) / 116, 3) : L / 903.3;
    *X = *Y * 9 * u_prime / (4 * v_prime);
    *Z = *Y * (12 - 3 * u_prime - 20 * v_prime) / (4 * v_prime);
}

// XYZ to RGB conversion
void xyz_to_rgb(image im, int i, int j, float X, float Y, float Z) {
    // XYZ to sRGB conversion
    float r = X * 3.2406 + Y * -1.5372 + Z * -0.4986;
    float g = X * -0.9689 + Y * 1.8758 + Z * 0.0415;
    float b = X * 0.0557 + Y * -0.2040 + Z * 1.0570;

    // sRGB to RGB conversion
    r = gamma_compress(r);
    g = gamma_compress(g);
    b = gamma_compress(b);

    set_pixel(im, i, j, 0, r);
    set_pixel(im, i, j, 1, g);
    set_pixel(im, i, j, 2, b);
}

void hcl_to_rgb(image im) {
    for (int i = 0; i < im.w; i++) {
        for (int j = 0; j < im.h; j++) {
            float L, u, v, X, Y, Z;
            hcl_to_luv(im, i, j, &L, &u, &v);
            luv_to_xyz(im, i, j, L, u, v, &X, &Y, &Z);
            xyz_to_rgb(im, i, j, X, Y, Z);
        }
    }
}
