#ifndef __KERNELS__
#define __KERNELS__
#include <cuda.h>

__global__ void initToZero(float *array, int size);
__global__ void initCells(	int *cellFirst, int *cellLast, int *cellNeighbors, int *cellCount, 
							float *cellDimension, float cellSize, int gridSize, 
							float *boidPosition, int *boidNext, int *boidPrevious, 
							int *boidCell, int nbBoids);
__global__ void computeSpeedIncrement(	float *positions, float *speed, float *speedIncrement, int *boidNext, int *boidCell, int nbBoids, float rs, float ra, float rc, float ws, float wa, float wc, int *cellFirst, int *cellNeighbors);
__global__ void updateSpeedPosition(float *positions, float *speed, float *speedIncrement, 
									int size);
__global__ void updateLists(int *cellFirst, int *cellLast, int *cellNeighbors, int *cellCount, 
							float *cellDimension, float cellSize, int gridSize, 
							float *boidPosition, int *boidNext, int *boidPrevious, 
							int *boidCell, int nbBoids);
#endif
