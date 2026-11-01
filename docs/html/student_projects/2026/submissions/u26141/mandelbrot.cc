// mandelbrot.cc : serial baseline
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
  int width        = argc > 1 ? atoi(argv[1]) : 1024;
  int height       = argc > 2 ? atoi(argv[2]) : 1024;
  int max_iter     = argc > 3 ? atoi(argv[3]) : 1000;
  const char *out  = argc > 4 ? argv[4] : "mandelbrot.ppm";

  const double x_min = -2.0, x_max = 1.0;
  const double y_min = -1.5, y_max = 1.5;

  int *iters = (int *)malloc(sizeof(int) * (size_t)width * height);
  if (!iters) { fprintf(stderr, "malloc failed\n"); return 1; }

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      double cr = x_min + (x_max - x_min) * (x + 0.5) / width;
      double ci = y_min + (y_max - y_min) * (y + 0.5) / height;
      double zr = 0.0, zi = 0.0;
      int k = 0;
      while (k < max_iter && zr * zr + zi * zi <= 4.0) {
        double t = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = t;
        k++;
      }
      iters[(size_t)y * width + x] = k;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

  /* 検証用: 集合に属する画素数から、面積を推定する */
  long long inside = 0, total_iters = 0;
  for (long long i = 0; i < (long long)width * height; i++) {
    if (iters[i] == max_iter) inside++;
    total_iters += iters[i];
  }
  double area = (x_max - x_min) * (y_max - y_min) * inside / ((double)width * height);

  /* PPM(P6)で画像を出力 */
  FILE *fp = fopen(out, "wb");
  if (!fp) { perror("fopen"); return 1; }
  fprintf(fp, "P6\n%d %d\n255\n", width, height);
  for (long long i = 0; i < (long long)width * height; i++) {
    int k = iters[i];
    unsigned char rgb[3];
    if (k == max_iter) {
      rgb[0] = rgb[1] = rgb[2] = 0;
    } else {
      rgb[0] = (unsigned char)((k * 9) % 256);
      rgb[1] = (unsigned char)((k * 5) % 256);
      rgb[2] = (unsigned char)((k * 13) % 256);
    }
    fwrite(rgb, 1, 3, fp);
  }
  fclose(fp);

  printf("size=%dx%d max_iter=%d\n", width, height, max_iter);
  printf("inside=%lld total_iters=%lld estimated_area=%.4f (known: about 1.5066)\n",
         inside, total_iters, area);
  printf("time=%.3f sec\n", sec);
  free(iters);
  return 0;
}