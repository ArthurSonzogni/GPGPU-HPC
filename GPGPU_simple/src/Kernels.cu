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

__global__ void computeSpeedIncrement(float *positions, float *speed, float *speedIncrement, int size, float rs, float ra, float rc, float ws, float wa, float wc)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = x*blockDim.y*gridDim.y + y;
	while(index < size)
	{
		float countS = 0;
		float countA = 0;
		float countC = 0;
		float speedSX = 0, speedSY = 0, speedSZ = 0;
		float speedAX = 0, speedAY = 0, speedAZ = 0;
		float speedCX = 0, speedCY = 0, speedCZ = 0;
		for(int i = 0 ; i < size ; i++)
		{
			if(i == index) continue;
			//glm::vec3 direction = position[j] - position[i];
			float directionX = positions[3*i]-positions[3*index];
			float directionY = positions[3*i+1]-positions[3*index+1];
			float directionZ = positions[3*i+2]-positions[3*index+2];
			float dist = sqrt(directionX*directionX+directionY*directionY+directionZ*directionZ);

			// separation/alignment/cohesion
			if (dist < rs )
			{
				speedSX -= directionX * ws;
				speedSY -= directionY * ws;
				speedSZ -= directionZ * ws;
				countS++;
			}
			if (dist < ra )
			{
				speedAX += speed[3*i]  * wa;
				speedAY += speed[3*i+1]  * wa;
				speedAZ += speed[3*i+2]  * wa;
				countA++;
			}
			if (dist < rc )
			{
				speedCX += directionX * wc;
				speedCY += directionY * wc;
				speedCZ += directionZ * wc;
				countC++;
			}
			speedSX = countS>0?speedSX/countS:speedSX;
			speedSY = countS>0?speedSY/countS:speedSY;
			speedSZ = countS>0?speedSZ/countS:speedSZ;

			speedAX = countA>0?speedAX/countA:speedAX;
			speedAY = countA>0?speedAY/countA:speedAY;
			speedAZ = countA>0?speedAZ/countA:speedAZ;

			speedCX = countC>0?speedCX/countC:speedCX;
			speedCY = countC>0?speedCY/countC:speedCY;
			speedCZ = countC>0?speedCZ/countC:speedCZ;

			speedIncrement[3*i] = speedCX+speedAX+speedSX;
			speedIncrement[3*i+1] = speedCY+speedAY+speedSY;
			speedIncrement[3*i+2] = speedCZ+speedAZ+speedSZ;
		}
		index += blockDim.x*gridDim.x*blockDim.y*gridDim.y;
	}
}

__global__ void updateSpeedPosition(float *positions, float *speed, float *speedIncrement, int size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = x*blockDim.y*gridDim.y + y;
	while(index < size)
	{
		speed[index] += speedIncrement[index];
		positions[index] += speed[index];
		positions[index] = positions[index]-floor(positions[index]);
		index += blockDim.x*gridDim.x*blockDim.y*gridDim.y;
	}
}
