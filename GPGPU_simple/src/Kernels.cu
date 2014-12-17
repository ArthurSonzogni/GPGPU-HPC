#include "Kernels.hpp"

__global__ void initToZero(float *array, int size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = x*blockDim.y*gridDim.y + y;
	while(index < size)
	{
		array[index] = 0.0;
		index += blockDim.x*gridDim.x*blockDim.y*gridDim.y;
	}
}
