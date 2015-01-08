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

	// copy the position to cuda
	gpuCheck(cudaMemcpy(position_cuda, &(position[0]), 3*agent*sizeof(float), cudaMemcpyHostToDevice));

	// init speed to zero
    int blockSize;
    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initToZero, 0, 3*agent);
    int gridSize = (3*agent + blockSize - 1) / blockSize; 
    initToZero<<<gridSize,blockSize>>>(speed_cuda, 3*agent);
    gpuCheck(cudaGetLastError());
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

    int blockSize,minGridSize,gridSize,dataSize;

    // computeSpeedIncrement
    dataSize = agent;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initToZero, 0, dataSize);
    gridSize = (dataSize + blockSize - 1) / blockSize; 
	computeSpeedIncrement<<<blockSize,gridSize>>>(position_cuda, speed_cuda, speedIncrement_cuda, dataSize, rs,ra,rc, ws,wa,wc);
    gpuCheck(cudaGetLastError());

    // updatePosition
    dataSize = agent;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initToZero, 0, dataSize);
    gridSize = (dataSize + blockSize - 1) / blockSize; 
	updateSpeedPosition<<<blockSize,gridSize>>>(position_cuda, speed_cuda, speedIncrement_cuda, dataSize);
    gpuCheck(cudaGetLastError());
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
