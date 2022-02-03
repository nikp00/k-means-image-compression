#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

int euclidean_dist(__local unsigned char *s1, int i1, int offset1,
                   __local unsigned char *s2, int i2, int offset2) {
  return (((s1[i1 * offset1] - s2[i2 * offset2]) *
           (s1[i1 * offset1] - s2[i2 * offset2])) +

          ((s1[i1 * offset1 + 1] - s2[i2 * offset2 + 1]) *
           (s1[i1 * offset1 + 1] - s2[i2 * offset2 + 1])) +

          ((s1[i1 * offset1 + 2] - s2[i2 * offset2 + 2]) *
           (s1[i1 * offset1 + 2] - s2[i2 * offset2 + 2])));
}

unsigned int rand(int global_index) {
  uint2 randoms;
  randoms.x = 10;
  randoms.y = 10;

  unsigned int seed = randoms.x + global_index;
  unsigned int t = seed ^ (seed << 11);
  return randoms.y ^ (randoms.y >> 19) ^ (t ^ (t >> 8));
}

void copy_to_local_char(__global unsigned char *buff_glob, int global_index,
                        __local unsigned char *buff_loc, int local_index,
                        int offset) {

  buff_loc[local_index * offset] = buff_glob[global_index * offset];
  buff_loc[local_index * offset + 1] = buff_glob[global_index * offset + 1];
  buff_loc[local_index * offset + 2] = buff_glob[global_index * offset + 2];
}

void copy_to_local_int(__global unsigned int *buff_glob, int global_index,
                       __local unsigned int *buff_loc, int local_index,
                       int offset) {

  buff_loc[local_index * offset] = buff_glob[global_index * offset];
  buff_loc[local_index * offset + 1] = buff_glob[global_index * offset + 1];
  buff_loc[local_index * offset + 2] = buff_glob[global_index * offset + 2];
}

__kernel void compute_centroids(int width, int height, int offset, int k,
                                __global unsigned char *image_in,
                                __global unsigned int *centroid_index,
                                __global unsigned int *centroid_count,
                                __global unsigned int *centroid_color,
                                __global unsigned char *centroids,
                                __local unsigned char *image_in_loc,
                                __local unsigned char *centroids_loc

) {

  const int gi = get_global_id(0);
  const int gj = get_global_id(1);
  const int li = get_local_id(0);
  const int lj = get_local_id(1);
  const int lw = get_local_size(0);
  const int gw = get_global_size(1);

  const int local_index = li * lw + lj;
  const int global_index = gi * gw + gj;
  const int img_size = width * height;
  const int to_copy = (k + (lw * lw) - 1) / (lw * lw);

  if (global_index < img_size) {
    copy_to_local_char(image_in, global_index, image_in_loc, local_index,
                       offset);

    if (local_index < k) {
      if ((lw * lw) < k) {
        for (int i = 0; i < to_copy; i++) {
          if (local_index * to_copy + i >= k)
            break;
          copy_to_local_char(centroids, local_index * to_copy + i,
                             centroids_loc, local_index * to_copy + i, 3);
        }
      } else
        copy_to_local_char(centroids, local_index, centroids_loc, local_index,
                           3);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float minDist = 2147483647;
    int minIndex = 0;

    for (int j = 0; j < k; j++) {

      float dist = euclidean_dist(centroids_loc, j, 3, image_in_loc,
                                  local_index, offset);
      if (dist < minDist) {
        minIndex = j;
        minDist = dist;
      }
    }

    centroid_index[global_index] = minIndex;

    atomic_inc((centroid_count + minIndex));

    atomic_add((centroid_color + minIndex * 3),
               image_in[global_index * offset]);

    atomic_add((centroid_color + minIndex * 3 + 1),
               image_in[global_index * offset + 1]);

    atomic_add((centroid_color + minIndex * 3 + 2),
               image_in[global_index * offset + 2]);
  }
}

__kernel void compute_centroids_color(
    int width, int height, int offset, int k, __global unsigned char *image_in,
    __global unsigned int *centroid_index,
    __global unsigned int *centroid_count,
    __global unsigned int *centroid_color, __global unsigned char *centroids,
    __local unsigned int *centroid_color_loc) {

  const int global_index = get_global_id(0);
  const int local_index = get_local_id(0);
  const int img_size = width * height;

  if (global_index < k) {
    copy_to_local_int(centroid_color, global_index, centroid_color_loc,
                      local_index, 3);

    barrier(CLK_LOCAL_MEM_FENCE);

    int count = centroid_count[global_index];
    if (count > 0) {

      centroids[global_index * 3] = centroid_color_loc[local_index * 3] / count;

      centroids[global_index * 3 + 1] =
          centroid_color_loc[local_index * 3 + 1] / count;

      centroids[global_index * 3 + 2] =
          centroid_color_loc[local_index * 3 + 2] / count;

    } else {
      int randomPx = rand(global_index) % img_size;
      centroids[global_index * 3] = image_in[randomPx * offset];
      centroids[global_index * 3 + 1] = image_in[randomPx * offset + 1];
      centroids[global_index * 3 + 2] = image_in[randomPx * offset + 2];
    }

    centroid_count[global_index] = 0;
    centroid_color[global_index * 3] = 0;
    centroid_color[global_index * 3 + 1] = 0;
    centroid_color[global_index * 3 + 2] = 0;
  }
}

__kernel void save_img(int width, int height, int offset,
                       __global unsigned char *image_out,
                       __global unsigned int *centroid_index,
                       __global unsigned char *centroids) {

  const int gi = get_global_id(0);
  const int gj = get_global_id(1);
  const int gw = get_global_size(1);

  const int global_index = gi * gw + gj;
  const int img_size = width * height;

  if (global_index < img_size) {
    image_out[global_index * offset] =
        centroids[centroid_index[global_index] * 3];

    image_out[global_index * offset + 1] =
        centroids[centroid_index[global_index] * 3 + 1];

    image_out[global_index * offset + 2] =
        centroids[centroid_index[global_index] * 3 + 2];

    image_out[global_index * offset + 3] = 255;
  }
}
