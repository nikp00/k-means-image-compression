#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../lib/FreeImage.h"
#include "./utils.h"

#define MAX_SOURCE_SIZE 16384

void read_image(struct image *image, char *source)
{
    FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, source, 0);

    FIBITMAP *imageBitmap32 = FreeImage_ConvertTo32Bits(imageBitmap);

    // if not RGB convert to RGBA
    if (FreeImage_GetBPP(imageBitmap) != 32)
    {
        FIBITMAP *tempImage = imageBitmap;
        imageBitmap = FreeImage_ConvertTo32Bits(tempImage);
    }

    // //Get image dimensions
    image->width = FreeImage_GetWidth(imageBitmap32);
    image->height = FreeImage_GetHeight(imageBitmap32);
    image->pitch = FreeImage_GetPitch(imageBitmap32);
    image->offset = image->pitch / image->width;

    // //Preapare room for a raw data copy of the image
    image->in = (unsigned char *)malloc(image->height * image->pitch * sizeof(unsigned char));
    FreeImage_ConvertToRawBits(image->in, imageBitmap32, image->pitch,
                               32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

    image->out = (unsigned char *)malloc(image->height * image->pitch * sizeof(unsigned char));

    free(imageBitmap);
    free(imageBitmap32);
}

void save_image(struct image *image, char *filename)
{
    FIBITMAP *dst = FreeImage_ConvertFromRawBits(image->out, image->width, image->height, image->pitch,
                                                 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

    FreeImage_Save(FIF_PNG, dst, filename, 0);
    free(dst);
}

char *read_kernel_source(char *filename)
{
    FILE *fp;
    size_t source_size;
    char *source_str;
    fp = fopen(filename, "r");
    if (!fp)
    {
        fprintf(stderr, ":-(#\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose(fp);
    return source_str;
}

void parse_argv(int argc, char **argv, int *k, int *n, int *t, char *mode, char *run_mode, char *img_path)
{
    if (argc >= 7)
    {
        strncpy(run_mode, argv[6], 10);
        if (!(strcmp(run_mode, "normal") == 0 || strcmp(run_mode, "benchmark") == 0))
        {
            fprintf(stderr, "Error: fifth arg (run mode) must be \"normal\" or \"benchmark\"!\n");
            strncpy(run_mode, "normal", 10);
        }
    }
    else
    {
        strncpy(run_mode, "normal", 10);
    }

    if (argc >= 6)
    {
        strncpy(mode, argv[5], 10);
        if (!(strcmp(mode, "gpu") == 0 || strcmp(mode, "cpu") == 0 || strcmp(mode, "cpup") == 0 || strcmp(mode, "all") == 0))
        {
            fprintf(stderr, "Error: fourth arg (mode) must be \"gpu\", \"cpu\" or \"cpup\"!\n");
            strncpy(mode, "all", 10);
        }
    }
    else
    {
        strncpy(mode, "all", 10);
    }

    if (argc < 5)
    {
        fprintf(stderr, "Error: not enough arguments!\n");
        exit(1);
    }
    if (sscanf(argv[1], "%i", k) != 1)
    {
        fprintf(stderr, "Error: first arg (number of colors after compression) must be an int!\n");
        exit(1);
    }
    if (sscanf(argv[2], "%i", n) != 1)
    {
        fprintf(stderr, "Error: second arg (number of iterations) must be an int!\n");
        exit(1);
    }
    if (sscanf(argv[3], "%i", t) != 1)
    {
        fprintf(stderr, "Error: third arg (number of threads used) must be an int!\n");
        exit(1);
    }

    strncpy(img_path, argv[4], 60);
}