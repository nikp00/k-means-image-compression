#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <omp.h>
#include <math.h>
#include <string.h>

#include "./utils/utils.h"

#define WORKGROUP_SIZE 16

void k_means_cpu(struct image *image, int k, int iter);
void k_means_cpu_parallel(struct image *image, int k, int iter, int num_threads);
double k_means_gpu(struct image *image, int k, int iter);

int euclidean_dist(int db, int dg, int dr)
{
    return (db * db) + (dg * dg) + (dr * dr);
}

int main(int argc, char **argv)
{
    char buff[60];
    int k, n, t;
    char *mode = (char *)malloc(10 * sizeof(char));
    char *run_mode = (char *)malloc(10 * sizeof(char));
    char img_path[60];
    double start, end;

    struct image *image = (struct image *)malloc(sizeof(struct image));

    // srand(time(NULL));

    parse_argv(argc, argv, &k, &n, &t, mode, run_mode, img_path);
    read_image(image, img_path);

    if (strcmp(run_mode, "normal") == 0)
    {
        printf("K: %d  |  N: %d  |  T: %d  | mode: %s  |  img: %s\n", k, n, t, mode, img_path);
        printf("width: %d  |  height: %d  |  pitch: %d  |  bpp: %d  |  K: %d  |  n: %d\n",
               image->width, image->height, image->pitch, image->pitch / image->width, k, n);
    }

    if (strcmp(mode, "cpu") == 0 || strcmp(mode, "all") == 0)
    {
        start = omp_get_wtime();
        k_means_cpu(image, k, n);
        end = omp_get_wtime();
        snprintf(buff, 60, "./output/out_cpu_%dx%d.png", image->width, image->height);
        save_image(image, buff);

        if (strcmp(run_mode, "normal") == 0)
            printf("CPU Elapsed: %fs \n", end - start);
        else
            printf("%d,%d,%d,%d,%f\n", k, n, image->width, image->height, end - start);
    }

    if (strcmp(mode, "cpup") == 0 || strcmp(mode, "all") == 0)
    {
        start = omp_get_wtime();
        k_means_cpu_parallel(image, k, n, t);
        end = omp_get_wtime();
        snprintf(buff, 60, "./output/out_cpu_parallel_%dx%d.png", image->width, image->height);
        save_image(image, buff);

        if (strcmp(run_mode, "normal") == 0)
            printf("CPU parallel Elapsed: %fs \n", end - start);
        else
            printf("%d,%d,%d,%d,%d,%f\n", k, n, t, image->width, image->height, end - start);
    }

    if (strcmp(mode, "gpu") == 0 || strcmp(mode, "all") == 0)
    {
        double elapsed = k_means_gpu(image, k, n);
        snprintf(buff, 60, "./output/out_gpu_%dx%d.png", image->width, image->height);
        save_image(image, buff);

        if (strcmp(run_mode, "normal") == 0)
            printf("GPU Elapsed: %fs \n", elapsed);
        else
            printf("%d,%d,%d,%d,%f\n", k, n, image->width, image->height, elapsed);
    }

    free(image->in);
    free(image->out);
    free(image);
}

void k_means_cpu(struct image *image, int k, int iter)
{

    int width = image->width;
    int height = image->height;

    unsigned char *image_in = image->in;
    unsigned char *image_out = image->out;

    int img_size = width * height;

    int *centroid_index = (int *)calloc(img_size, sizeof(int));
    int *centroid_count = (int *)calloc(k, sizeof(int));
    int *centroid_color = (int *)calloc(k * 3, sizeof(int));
    int *centroids = (int *)calloc(k * 3, sizeof(int));

    // Init centroids
    for (int i = 0; i < k; i++)
    {
        int randomPx = rand() % img_size;
        centroids[i * 3] = (int)image_in[randomPx * 4];
        centroids[i * 3 + 1] = (int)image_in[randomPx * 4 + 1];
        centroids[i * 3 + 2] = (int)image_in[randomPx * 4 + 2];
    }

    while (iter-- > 0)
    {
        // Find min dist
        for (int i = 0; i < img_size; i++)
        {
            int minDist = euclidean_dist(image_in[i * 4] - centroids[0],
                                         image_in[i * 4 + 1] - centroids[1],
                                         image_in[i * 4 + 2] - centroids[2]);
            int minIndex = 0;
            for (int j = 1; j < k; j++)
            {

                int dist = euclidean_dist(image_in[i * 4] - centroids[j * 3],
                                          image_in[i * 4 + 1] - centroids[j * 3 + 1],
                                          image_in[i * 4 + 2] - centroids[j * 3 + 2]);

                if (dist < minDist)
                {
                    minIndex = j;
                    minDist = dist;
                }
            }

            centroid_index[i] = minIndex;
            centroid_count[minIndex]++;

            centroid_color[minIndex * 3] += image_in[i * 4];
            centroid_color[minIndex * 3 + 1] += image_in[i * 4 + 1];
            centroid_color[minIndex * 3 + 2] += image_in[i * 4 + 2];
        }

        // Calc mean color for centroid
        for (int j = 0; j < k; j++)
        {
            int count = centroid_count[j];
            if (count > 0)
            {
                centroids[j * 3] = centroid_color[j * 3] / count;
                centroids[j * 3 + 1] = centroid_color[j * 3 + 1] / count;
                centroids[j * 3 + 2] = centroid_color[j * 3 + 2] / count;
            }
            else
            {
                int randomPx = rand() % img_size;
                centroids[j * 3] = image_in[randomPx * 4];
                centroids[j * 3 + 1] = image_in[randomPx * 4 + 1];
                centroids[j * 3 + 2] = image_in[randomPx * 4 + 2];
            }

            centroid_count[j] = 0;
            centroid_color[j * 3] = 0;
            centroid_color[j * 3 + 1] = 0;
            centroid_color[j * 3 + 2] = 0;
        }
    }

    // Applay new colors to image
    for (int i = 0; i < img_size; i++)
    {
        image_out[i * 4] = centroids[centroid_index[i] * 3];
        image_out[i * 4 + 1] = centroids[centroid_index[i] * 3 + 1];
        image_out[i * 4 + 2] = centroids[centroid_index[i] * 3 + 2];
        image_out[i * 4 + 3] = image_in[i * 4 + 3];
    }

    free(centroid_color);
    free(centroid_index);
    free(centroid_count);
    free(centroids);
}

void k_means_cpu_parallel(struct image *image, int k, int iter, int num_threads)
{
    int width = image->width;
    int height = image->height;

    unsigned char *image_in = image->in;
    unsigned char *image_out = image->out;

    int img_size = width * height;

    int *centroid_index = (int *)calloc(img_size, sizeof(int));
    int *centroid_count = (int *)calloc(k, sizeof(int));
    int *centroid_color = (int *)calloc(k * 3, sizeof(int));
    int *centroids = (int *)calloc(k * 3, sizeof(int));

    omp_set_num_threads(num_threads);

// Init centroids
#pragma omp parallel for
    for (int i = 0; i < k; i++)
    {
        int randomPx = rand() % img_size;
        centroids[i * 3] = image_in[randomPx * 4];
        centroids[i * 3 + 1] = image_in[randomPx * 4 + 1];
        centroids[i * 3 + 2] = image_in[randomPx * 4 + 2];
    }

    while (iter-- > 0)
    {
        // Find min dist
#pragma omp parallel
        {
            int *centroid_color_private = (int *)calloc(k * 3, sizeof(int));
            int *centroid_count_private = (int *)calloc(k, sizeof(int));
            int dist, minDist, minIndex;
#pragma omp for
            for (int i = 1; i < img_size; i++)
            {
                minDist = euclidean_dist(image_in[i * 4] - centroids[0],
                                         image_in[i * 4 + 1] - centroids[1],
                                         image_in[i * 4 + 2] - centroids[2]);
                minIndex = 0;
                for (int j = 1; j < k; j++)
                {
                    dist = euclidean_dist(image_in[i * 4] - centroids[j * 3],
                                          image_in[i * 4 + 1] - centroids[j * 3 + 1],
                                          image_in[i * 4 + 2] - centroids[j * 3 + 2]);
                    if (dist < minDist)
                    {
                        minIndex = j;
                        minDist = dist;
                    }
                }

                centroid_index[i] = minIndex;
                centroid_count_private[minIndex]++;
                centroid_color_private[minIndex * 3] += image_in[i * 4];
                centroid_color_private[minIndex * 3 + 1] += image_in[i * 4 + 1];
                centroid_color_private[minIndex * 3 + 2] += image_in[i * 4 + 2];
            }

#pragma omp critical
            {
                for (int n = 0; n < k; ++n)
                {
                    centroid_color[n * 3] += centroid_color_private[n * 3];
                    centroid_color[n * 3 + 1] += centroid_color_private[n * 3 + 1];
                    centroid_color[n * 3 + 2] += centroid_color_private[n * 3 + 2];
                    centroid_count[n] += centroid_count_private[n];
                }
            }
        }
// Calc mean color for centroid
#pragma omp parallel for
        for (int j = 0; j < k; j++)
        {
            int count = centroid_count[j];
            if (count > 0)
            {
                centroids[j * 3] = centroid_color[j * 3] / count;
                centroids[j * 3 + 1] = centroid_color[j * 3 + 1] / count;
                centroids[j * 3 + 2] = centroid_color[j * 3 + 2] / count;
            }
            else
            {
                int randomPx = rand() % img_size;
                centroids[j * 3] = image_in[randomPx * 4];
                centroids[j * 3 + 1] = image_in[randomPx * 4 + 1];
                centroids[j * 3 + 2] = image_in[randomPx * 4 + 2];
            }

            centroid_count[j] = 0;
            centroid_color[j * 3] = 0;
            centroid_color[j * 3 + 1] = 0;
            centroid_color[j * 3 + 2] = 0;
        }
    }

// Applay new colors to image
#pragma omp parallel for
    for (int i = 0; i < img_size; i++)
    {
        image_out[i * 4] = centroids[centroid_index[i] * 3];
        image_out[i * 4 + 1] = centroids[centroid_index[i] * 3 + 1];
        image_out[i * 4 + 2] = centroids[centroid_index[i] * 3 + 2];
        image_out[i * 4 + 3] = image_in[i * 4 + 3];
    }

    free(centroid_color);
    free(centroid_index);
    free(centroid_count);
    free(centroids);
}

double k_means_gpu(struct image *image, int k, int iter)
{
    int width = image->width;
    int height = image->height;
    int offset = image->offset;

    unsigned char *image_in = image->in;
    unsigned char *image_out = image->out;

    int img_size = width * height;

    unsigned int *centroid_index = (unsigned int *)calloc(img_size, sizeof(unsigned int));
    unsigned int *centroid_count = (unsigned int *)calloc(k, sizeof(unsigned int));
    unsigned int *centroid_color = (unsigned int *)calloc(k * 3, sizeof(unsigned int));
    unsigned char *centroids = (unsigned char *)calloc(k * 3, sizeof(unsigned char));

#pragma omp parallel for
    for (int i = 0; i < k; i++)
    {
        int randomPx = rand() % img_size;
        centroids[i * 3] = image_in[randomPx * offset];
        centroids[i * 3 + 1] = image_in[randomPx * offset + 1];
        centroids[i * 3 + 2] = image_in[randomPx * offset + 2];
    }

    int ws = WORKGROUP_SIZE;

    char ch;
    int i;
    cl_int ret;

    // Read kernel source
    char *source_str = read_kernel_source("./kernels/kernel.cl");

    // Platform info
    cl_platform_id platform_id[10];
    cl_uint ret_num_platforms;
    char *buf;
    size_t buf_len;
    ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);

    // Device info
    cl_device_id device_id[10];
    cl_uint ret_num_devices;
    ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10,
                         device_id, &ret_num_devices);

    // Context
    cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);

    // Comand Queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);

    // Work distribution
    size_t *global_item_size_k1 = (size_t *)malloc(sizeof(size_t) * 2);
    global_item_size_k1[0] = (size_t)((height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) * WORKGROUP_SIZE;
    global_item_size_k1[1] = (size_t)((width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) * WORKGROUP_SIZE;

    size_t *local_item_size_k1 = (size_t *)malloc(sizeof(size_t) * 2);
    local_item_size_k1[0] = (size_t)WORKGROUP_SIZE;
    local_item_size_k1[1] = (size_t)WORKGROUP_SIZE;

    // Work distribution
    size_t global_item_size_k2 = (size_t)((k + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) * WORKGROUP_SIZE;
    size_t local_item_size_k2 = (size_t)WORKGROUP_SIZE;

    // Memory allocation on target device
    cl_mem image_in_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             img_size * offset * sizeof(unsigned char), image_in, &ret);

    cl_mem image_out_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                                              img_size * offset * sizeof(unsigned char), image_out, &ret);

    cl_mem centroid_index_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                   img_size * sizeof(unsigned int), centroid_index, &ret);

    cl_mem centroid_count_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                   k * sizeof(unsigned int), centroid_count, &ret);

    cl_mem centroid_color_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                   k * 3 * sizeof(unsigned int), centroid_color, &ret);

    cl_mem centroids_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                              k * 3 * sizeof(unsigned char), centroids, &ret);

    // Create program from source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                                   NULL, &ret);

    // Build program
    ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);
    DEBUG_PRINT(("Program build status: %d\n", ret));

    // Log
    size_t build_log_len;
    char *build_log;
    ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
                                0, NULL, &build_log_len);
    build_log = (char *)malloc(sizeof(char) * (build_log_len + 1));
    ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
                                build_log_len, build_log, NULL);
    free(build_log);

    // Create kernel
    cl_kernel kernel_1 = clCreateKernel(program, "compute_centroids", &ret);
    cl_kernel kernel_2 = clCreateKernel(program, "compute_centroids_color", &ret);
    cl_kernel kernel_3 = clCreateKernel(program, "save_img", &ret);

    // Set kernel args
    ret = clSetKernelArg(kernel_1, 0, sizeof(cl_int), (void *)&width);
    ret |= clSetKernelArg(kernel_1, 1, sizeof(cl_int), (void *)&height);
    ret |= clSetKernelArg(kernel_1, 2, sizeof(cl_int), (void *)&offset);
    ret |= clSetKernelArg(kernel_1, 3, sizeof(cl_int), (void *)&k);
    ret |= clSetKernelArg(kernel_1, 4, sizeof(cl_mem), (void *)&image_in_mem_obj);
    ret |= clSetKernelArg(kernel_1, 5, sizeof(cl_mem), (void *)&centroid_index_mem_obj);
    ret |= clSetKernelArg(kernel_1, 6, sizeof(cl_mem), (void *)&centroid_count_mem_obj);
    ret |= clSetKernelArg(kernel_1, 7, sizeof(cl_mem), (void *)&centroid_color_mem_obj);
    ret |= clSetKernelArg(kernel_1, 8, sizeof(cl_mem), (void *)&centroids_mem_obj);
    ret |= clSetKernelArg(kernel_1, 9, WORKGROUP_SIZE * WORKGROUP_SIZE * offset * sizeof(unsigned char), NULL);
    ret |= clSetKernelArg(kernel_1, 10, k * 3 * sizeof(unsigned char), NULL);

    ret = clSetKernelArg(kernel_2, 0, sizeof(cl_int), (void *)&width);
    ret |= clSetKernelArg(kernel_2, 1, sizeof(cl_int), (void *)&height);
    ret |= clSetKernelArg(kernel_2, 2, sizeof(cl_int), (void *)&offset);
    ret |= clSetKernelArg(kernel_2, 3, sizeof(cl_int), (void *)&k);
    ret |= clSetKernelArg(kernel_2, 4, sizeof(cl_mem), (void *)&image_in_mem_obj);
    ret |= clSetKernelArg(kernel_2, 5, sizeof(cl_mem), (void *)&centroid_index_mem_obj);
    ret |= clSetKernelArg(kernel_2, 6, sizeof(cl_mem), (void *)&centroid_count_mem_obj);
    ret |= clSetKernelArg(kernel_2, 7, sizeof(cl_mem), (void *)&centroid_color_mem_obj);
    ret |= clSetKernelArg(kernel_2, 8, sizeof(cl_mem), (void *)&centroids_mem_obj);
    ret |= clSetKernelArg(kernel_2, 9, local_item_size_k2 * 3 * sizeof(unsigned int), NULL);

    ret = clSetKernelArg(kernel_3, 0, sizeof(cl_int), (void *)&width);
    ret |= clSetKernelArg(kernel_3, 1, sizeof(cl_int), (void *)&height);
    ret |= clSetKernelArg(kernel_3, 2, sizeof(cl_int), (void *)&offset);
    ret |= clSetKernelArg(kernel_3, 3, sizeof(cl_mem), (void *)&image_out_mem_obj);
    ret |= clSetKernelArg(kernel_3, 4, sizeof(cl_mem), (void *)&centroid_index_mem_obj);
    ret |= clSetKernelArg(kernel_3, 5, sizeof(cl_mem), (void *)&centroids_mem_obj);

    double start = omp_get_wtime();

    // Run kernel
    while (iter-- > 0)
    {
        ret = clEnqueueNDRangeKernel(command_queue, kernel_1, 2, NULL,
                                     global_item_size_k1, local_item_size_k1, 0, NULL, NULL);
        ret = clEnqueueNDRangeKernel(command_queue, kernel_2, 1, NULL,
                                     &global_item_size_k2, &local_item_size_k2, 0, NULL, NULL);
    }
    ret = clEnqueueNDRangeKernel(command_queue, kernel_3, 2, NULL,
                                 global_item_size_k1, local_item_size_k1, 0, NULL, NULL);

    // Copy data from device
    ret = clEnqueueReadBuffer(command_queue, image_out_mem_obj, CL_TRUE, 0,
                              img_size * offset * sizeof(unsigned char), image_out, 0, NULL, NULL);

    double end = omp_get_wtime();

    // Free memory
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel_1);
    ret = clReleaseKernel(kernel_2);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(image_in_mem_obj);
    ret = clReleaseMemObject(image_out_mem_obj);
    ret = clReleaseMemObject(centroid_index_mem_obj);
    ret = clReleaseMemObject(centroid_count_mem_obj);
    ret = clReleaseMemObject(centroid_color_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    return end - start;
}