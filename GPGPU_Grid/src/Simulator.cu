#include "Simulator.hpp"
#include <cstdlib>
#include <iostream>
//#include "ProgressBar.hpp"
#include <fstream>
#include <sstream>
#include "Kernels.hpp"

#include "../../Common/src/ProgressBar.hpp"

#define gpuCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

float randDouble()
{ 
	return rand() / float(RAND_MAX);
}

Simulator::Simulator(
		int agent,
		int step,
		float wc, float wa, float ws,
		float rc, float ra, float rs,
        bool write):
	agent(agent),step(step),
	wc(wc),wa(wa),ws(ws),
	rc(rc),ra(ra),rs(rs),
    write(write)
{
	init();
}


void Simulator::init()
{
	float maxRadius = std::max(std::max(ra,rs),rc);
	gridSize = 1;
	cellSize = 1.0;
	// TODO Modify the kernel to allow more than 512 cells
	while(cellSize > maxRadius && gridSize <= 8)
	{
		gridSize += 1;
		cellSize = (float)1/gridSize;
	}
	gridSize -= 1;
	cellSize = (float)1/gridSize;

	// initialization initial position
	position.reserve(3*agent);
	for(int i = 0; i < agent; ++i)
	{
		position.push_back(randDouble()); // x
		position.push_back(randDouble()); // y
		position.push_back(randDouble()); // z
	}

	gpuCheck(cudaGetLastError());

	// create the cuda data
	gpuCheck(cudaMalloc((void**)&position_cuda,position.size()*sizeof(float)));
	gpuCheck(cudaMalloc((void**)&speed_cuda,position.size()*sizeof(float)));
	gpuCheck(cudaMalloc((void**)&speedIncrement_cuda,position.size()*sizeof(float)));

	gpuCheck(cudaMalloc((void**)&cellFirst_cuda,gridSize*gridSize*gridSize*sizeof(int)));
	gpuCheck(cudaMalloc((void**)&cellLast_cuda,gridSize*gridSize*gridSize*sizeof(int)));
	gpuCheck(cudaMalloc((void**)&cellCount_cuda,gridSize*gridSize*gridSize*sizeof(int)));
	gpuCheck(cudaMalloc((void**)&cellNeighbors_cuda,27*gridSize*gridSize*gridSize*sizeof(int)));
	gpuCheck(cudaMalloc((void**)&cellDimension_cuda,6*gridSize*gridSize*gridSize*sizeof(float)));

	gpuCheck(cudaMalloc((void**)&boidNext_cuda,agent*sizeof(int)));
	gpuCheck(cudaMalloc((void**)&boidPrevious_cuda,agent*sizeof(int)));
	gpuCheck(cudaMalloc((void**)&boidCell_cuda,agent*sizeof(int)));

	// copy the position to cuda
	gpuCheck(cudaMemcpy(position_cuda, &(position[0]), 3*agent*sizeof(float), cudaMemcpyHostToDevice));

	// init speed to zero
    int blockDim;
    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockDim, initToZero, 0, 3*agent);
    int gridDim = (3*agent + blockDim - 1) / blockDim; 
    initToZero<<<gridDim,blockDim>>>(speed_cuda, 3*agent);
    gpuCheck(cudaGetLastError());

	// init lists & neighbors
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockDim, initCells, 0, gridSize*gridSize*gridSize);
    gridDim = (gridSize*gridSize*gridSize + blockDim - 1) / blockDim; 
	initCells<<<gridDim,blockDim>>>(cellFirst_cuda, cellLast_cuda, cellNeighbors_cuda, cellCount_cuda, cellDimension_cuda, cellSize, gridSize, position_cuda, boidNext_cuda, boidPrevious_cuda, boidCell_cuda, agent);
}


void Simulator::run()
{
    ProgressBar progressBar;
	for(int i = 0; i < step; ++i)
	{
		oneStep();
		progressBar.update(i/float(step));
		gpuCheck(cudaMemcpy(&(position[0]), position_cuda, 3*agent*sizeof(float), cudaMemcpyDeviceToHost));

		// print the result
		std::stringstream filename;
		filename << "./output/boids_" << i << ".xyz";
		if (write)  save(filename.str()); 
	}
}

void Simulator::oneStep()
{

    int blockDim,minGridSize,gridDim,dataSize;

    // computeSpeedIncrement
    dataSize = agent;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockDim, initToZero, 0, dataSize);
    gridDim = (dataSize + blockDim - 1) / blockDim; 
	computeSpeedIncrement<<<blockDim,gridDim>>>(position_cuda, speed_cuda, speedIncrement_cuda, boidNext_cuda, boidCell_cuda, dataSize, rs,ra,rc, ws,wa,wc, cellFirst_cuda, cellNeighbors_cuda);
	cudaThreadSynchronize();

    gpuCheck(cudaGetLastError());

    // updatePosition
    dataSize = agent;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockDim, initToZero, 0, dataSize);
    gridDim = (dataSize + blockDim - 1) / blockDim; 
	updateSpeedPosition<<<blockDim,gridDim>>>(position_cuda, speed_cuda, speedIncrement_cuda, dataSize);
    gpuCheck(cudaGetLastError());

	// updateList
	gridDim = 1;
	blockDim = gridSize*gridSize*gridSize;
	updateLists<<<gridDim,blockDim>>>(cellFirst_cuda, cellLast_cuda, cellNeighbors_cuda, cellCount_cuda, cellDimension_cuda, cellSize, gridSize, position_cuda, boidNext_cuda, boidPrevious_cuda, boidCell_cuda, agent);
}

void Simulator::save(const std::string& filename)
{
	std::ofstream file;
	file.open(filename.c_str());

	for(int i = 0; i < agent; ++i)
	{
		file
			<< position[3*i] << " "
			<< position[3*i+1] << " "
			<< position[3*i+2]
			<< std::endl;
	}

	file.close();
}
