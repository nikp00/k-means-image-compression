#ifndef UTILS_H_
#define UTILS_H_

#ifdef DEBUG
#define DEBUG_PRINT(x) printf x
#else
#define DEBUG_PRINT(x) \
    do                 \
    {                  \
    } while (0)
#endif

struct image
{
    unsigned char *in;
    unsigned char *out;
    int width;
    int height;
    int pitch;
    int offset;
};

void read_image(struct image *image, char *source);

void save_image(struct image *image, char *filename);

char *read_kernel_source(char *filename);

void parse_argv(int argc, char **argv, int *k, int *n, int *t, char *mode, char *run_mode, char *img_path);

#endif