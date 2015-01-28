#ifndef __KERNELS__
#define __KERNELS__
#include <cuda.h>

__global__ void initToZero(float *array, int size);
__global__ void computeSpeedIncrement(float *positions, float *speed, float *speedIncrement, int size, float rs, float ra, float rc, float ws, float wa, float wc);
__global__ void updateSpeedPosition(float *positions, float *speed, float *speedIncrement, int size, float vmax);
#endif
