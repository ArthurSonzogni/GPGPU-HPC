#include "Kernels.hpp"

__global__ void initToZero(float *array, int size)
{
    /*array[0] = 0.f;*/
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = x*blockDim.y*gridDim.y + y;
    while (index < size)
    {
        array[index] = 0.0;
        index += blockDim.x*gridDim.x*blockDim.y*gridDim.y;
    }
}

__device__ void getGridPosition(int rank, int gridSize, int *position)
{
    position[2] = (rank/gridSize)/gridSize;
    position[1] = (rank/gridSize)%gridSize;
    position[0] = (rank%gridSize);
}
__device__ int getRank(int x, int y, int z, int gridSize)
{
    return (x + gridSize)%gridSize + gridSize*(
           (y + gridSize)%gridSize + gridSize*(
           (z + gridSize)%gridSize  ));
}

__global__ void initCells(	int *cellFirst, int *cellLast, int *cellNeighbors, int *cellCount, 
							float *cellDimension, float cellSize, int gridSize, 
							float *boidPosition, int *boidNext, int *boidPrevious, 
							int *boidCell, int nbBoids)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int rank = x*blockDim.y*gridDim.y + y;
	if(rank >= gridSize*gridSize*gridSize) return;

	int cellPosition[3];
	getGridPosition(rank, gridSize, cellPosition);
	// Build neighbors table
	int i = 0;
	for(int x = -1 ; x <= 1 ; x++) {
		for(int y = -1 ; y <= 1 ; y++) {
			for(int z = -1 ; z <= 1 ; z++) {
				cellNeighbors[27*rank+i] = getRank(cellPosition[0]+x, cellPosition[1]+y, cellPosition[2]+z, gridSize);
				i++;
			}
		}
	}
	// Compute dimensions
	cellDimension[6*rank+0] = cellPosition[0]*cellSize;
	cellDimension[6*rank+1] = cellPosition[1]*cellSize;
	cellDimension[6*rank+2] = cellPosition[2]*cellSize;
	cellDimension[6*rank+3] = cellDimension[6*rank+0] + cellSize;
	cellDimension[6*rank+4] = cellDimension[6*rank+1] + cellSize;
	cellDimension[6*rank+5] = cellDimension[6*rank+2] + cellSize;
	float minX = cellDimension[6*rank+0];
	float minY = cellDimension[6*rank+1];
	float minZ = cellDimension[6*rank+2];
	float maxX = cellDimension[6*rank+3];
	float maxY = cellDimension[6*rank+4];
	float maxZ = cellDimension[6*rank+5];

	// Build list
	cellFirst[rank] = -1;
	cellLast[rank] = -1;
	cellCount[rank] = 0;
	for(int n = 0 ; n < nbBoids ; n++)
	{
		if(
				boidPosition[3*n+0] >= minX && boidPosition[3*n+0] <= maxX &&
				boidPosition[3*n+1] >= minY && boidPosition[3*n+1] <= maxY &&
				boidPosition[3*n+2] >= minZ && boidPosition[3*n+2] <= maxZ
		  )
		{
			boidCell[n] = rank;
			if(cellLast[rank] == -1) {
				cellLast[rank] = n;
				boidNext[n] = -1;
			}
			int next = cellFirst[rank];
			cellFirst[rank] = n;
			boidPrevious[n] = -1;
			boidNext[n] = next;
			if(next != -1)
			{
				boidPrevious[next] = n;
			}
			boidCell[n] = rank;
			cellCount[rank]++;
		}
	}
}

__global__ void computeSpeedIncrement(	float *positions, float *speed, float *speedIncrement, int *boidNext, int *boidCell, int nbBoids, float rs, float ra, float rc, float ws, float wa, float wc, int *cellFirst, int *cellNeighbors)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = x*blockDim.y*gridDim.y + y;
	while(index < nbBoids)
	{
		float countS = 0;
		float countA = 0;
		float countC = 0;
		float speedSX = 0, speedSY = 0, speedSZ = 0;
		float speedAX = 0, speedAY = 0, speedAZ = 0;
		float speedCX = 0, speedCY = 0, speedCZ = 0;
		for(int k = 0 ; k < 27 ; k++)
		{
			int currentRank = cellNeighbors[27*boidCell[index]+k];
			int i = cellFirst[currentRank];
			while(i != -1)
			{
				if(i != index)
				{

					//glm::vec3 direction = position[j] - position[i];
					float directionX = positions[3*i]-positions[3*index];
					float directionY = positions[3*i+1]-positions[3*index+1];
					float directionZ = positions[3*i+2]-positions[3*index+2];
					float dist = sqrt(directionX*directionX + directionY*directionY + directionZ*directionZ);
					
					// separation/alignment/cohesion
					if (dist < rs )
					{
						speedSX -= directionX;
						speedSY -= directionY;
						speedSZ -= directionZ;
						countS++;
					
					}
					if (dist < ra )
					{
						speedAX += speed[3*i];
						speedAY += speed[3*i+1];
						speedAZ += speed[3*i+2];
						countA++;
					}
					if (dist < rc )
					{
						speedCX += directionX;
						speedCY += directionY;
						speedCZ += directionZ;
						countC++;
					}
				}
				i = boidNext[i];
			}
		}

		if(countS > 0)
		{
			speedSX = speedSX*ws/countS;
			speedSY = speedSY*ws/countS;
			speedSZ = speedSZ*ws/countS;
		}

		if(countA > 0)
		{
			speedAX = speedAX*wa/countA;
			speedAY = speedAY*wa/countA;
			speedAZ = speedAZ*wa/countA;
		}

		if(countC > 0)
		{
			speedCX = speedCX*wc/countC;
			speedCY = speedCY*wc/countC;
			speedCZ = speedCZ*wc/countC;
		}

		speedIncrement[3*index] = speedCX+speedAX+speedSX;
		speedIncrement[3*index+1] = speedCY+speedAY+speedSY;
		speedIncrement[3*index+2] = speedCZ+speedAZ+speedSZ;
		index += blockDim.x*gridDim.x*blockDim.y*gridDim.y;
	}
}

__global__ void updateSpeedPosition(float *positions, float *speed, float *speedIncrement, 
									int nbBoids)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = x*blockDim.y*gridDim.y + y;
	while(index < nbBoids)
	{
		speed[3*index] += speedIncrement[3*index];
		speed[3*index+1] += speedIncrement[3*index+1];
		speed[3*index+2] += speedIncrement[3*index+2];
		float maxSpeed = 0.08;
		float s = sqrt(speed[3*index]*speed[3*index] + speed[3*index+1]*speed[3*index+1] + speed[3*index+2]*speed[3*index+2]);
		if(s > maxSpeed)
		{
			speed[3*index] *= maxSpeed/s;
			speed[3*index+1] *= maxSpeed/s;
			speed[3*index+2] *= maxSpeed/s;
		}
		positions[3*index] += speed[3*index];
		positions[3*index+1] += speed[3*index+1];
		positions[3*index+2] += speed[3*index+2];

		positions[3*index] = positions[3*index]-floor(positions[3*index]);
		positions[3*index+1] = positions[3*index+1]-floor(positions[3*index+1]);
		positions[3*index+2] = positions[3*index+2]-floor(positions[3*index+2]);

		index += blockDim.x*gridDim.x*blockDim.y*gridDim.y;
	}
}

__global__ void updateLists(int *cellFirst, int *cellLast, int *cellNeighbors, int *cellCount, 
							float *cellDimension, float cellSize, int gridSize, 
							float *boidPosition, int *boidNext, int *boidPrevious, 
							int *boidCell, int nbBoids)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int rank = x*blockDim.y*gridDim.y + y;
	if(rank >= gridSize*gridSize*gridSize) return;

	int currentIndex = cellFirst[rank];
	while(currentIndex != -1)
	{
		float x = boidPosition[3*currentIndex+0];
		float y = boidPosition[3*currentIndex+1];
		float z = boidPosition[3*currentIndex+2];

		// Prepare move
		int newRankIndex = 0;
		if(x > cellDimension[6*rank+0])
			newRankIndex += 9;
		if(x > cellDimension[6*rank+3])
			newRankIndex += 9;
		if(y > cellDimension[6*rank+1])
			newRankIndex += 3;
		if(y > cellDimension[6*rank+4])
			newRankIndex += 3;
		if(z > cellDimension[6*rank+2])
			newRankIndex += 1;
		if(z > cellDimension[6*rank+5])
			newRankIndex += 1;
		int newRank = cellNeighbors[27*rank+newRankIndex];
		if(newRank != rank)
		{
			// If currentIndex is the head of the list
			if(cellFirst[rank] == currentIndex)
			{
				// Change the head
				cellFirst[rank] = boidNext[currentIndex];
				// If the head is not null, update its previous
				if(cellFirst[rank] != -1)
					boidPrevious[cellFirst[rank]] = -1;
			}
			// If currentIndex is the tail of the list
			if(cellLast[rank] == currentIndex)
			{
				// Change the tail
				cellLast[rank] = boidPrevious[currentIndex];
				// If the tail is not null, update its next
				if(cellLast[rank] != -1)
					boidNext[cellLast[rank]] = -1;
			}
			// In every case
			// If there is a previous, change its next
			if(boidPrevious[currentIndex] != -1)
				boidNext[boidPrevious[currentIndex]] = boidNext[currentIndex];
			// If there is a next, change its previous
			if(boidNext[currentIndex] != -1)
				boidPrevious[boidNext[currentIndex]] = boidPrevious[currentIndex];
		}
		__syncthreads();

		// Move, one direction at a time
		for(int i = 0 ; i < 27 ; i++)
		{
			if(newRankIndex == i && newRank != rank)
			{
				// If the list of the new cell is empty
				if(cellFirst[newRank] == -1)
				{
					// Set the head of the list
					cellFirst[newRank] = currentIndex;
					// Set the next of the current
					boidNext[currentIndex] = -1;
					// Set the previous of the current
					boidPrevious[currentIndex] = -1;
					// Set the tail of the list
					cellLast[newRank] = currentIndex;
				}
				// Else add in tail
				else
				{
					// Update the next of the old tail
					boidNext[cellLast[newRank]] = currentIndex;
					// Set the previous of the current
					boidPrevious[currentIndex] = cellLast[newRank];
					// Set the next of the current
					boidNext[currentIndex] = -1;
					// Set the new tail
					cellLast[newRank] = currentIndex;
				}
				boidCell[currentIndex] = newRank;
			}
			__syncthreads();
		}

		currentIndex = boidNext[currentIndex];
	}
}
