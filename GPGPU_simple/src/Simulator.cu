#include "Simulator.hpp"
#include <cstdlib>
#include <iostream>
//#include "ProgressBar.hpp"
#include <fstream>
#include <sstream>
#include "Kernels.hpp"

#define gpuCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

double randDouble()
{ 
	return rand() / double(RAND_MAX);
}

Simulator::Simulator(
		int agent,
		int step,
		float wc, float wa, float ws,
		float rc, float ra, float rs):
	agent(agent),step(step),
	wc(wc),wa(wa),ws(ws),
	rc(rc),ra(ra),rs(rs)
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

	// create the cuda data
	gpuCheck(cudaMalloc((void**)&position_cuda,position.size()*sizeof(float)));
	gpuCheck(cudaMalloc((void**)&speed_cuda,position.size()*sizeof(float)));
	gpuCheck(cudaMalloc((void**)&speedIncrement_cuda,position.size()*sizeof(float)));

	// copy the position to cuda
	gpuCheck(cudaMemcpy(position_cuda, &(position[0]), 3*agent*sizeof(float), cudaMemcpyHostToDevice));

	// init speed to zero
	dim3 gridSize(1,1,1);
	dim3 blockSize(32,32,1);
	// The kernel doesn't seem to be working
	initToZero<<<blockSize,gridSize>>>(speed_cuda, 3*agent);
	gpuCheck(cudaGetLastError());
}


void Simulator::run()
{
	//	ProgressBar progressBar;
	for(int i = 0; i < step; ++i)
	{
		oneStep();
		//progressBar.update(i/float(step));
		gpuCheck(cudaMemcpy(&(position[0]), position_cuda, 3*agent*sizeof(float), cudaMemcpyDeviceToHost));

		// print the result
		std::stringstream filename;
		filename << "./output/boids_" << i << ".xyz";
		save(filename.str()); 
	}
}

void Simulator::oneStep()
{
	dim3 gridSize(1,1,1);
	dim3 blockSize(32,32,1);

	gpuCheck(cudaGetLastError());
	computeSpeedIncrement<<<blockSize,gridSize>>>(position_cuda, speed_cuda, speedIncrement_cuda, agent, rs,ra,rc, ws,wa,wc);
	gpuCheck(cudaGetLastError());
	updateSpeedPosition<<<blockSize,gridSize>>>(position_cuda, speed_cuda, speedIncrement_cuda, 3*agent);
	gpuCheck(cudaGetLastError());
	//	// compute the speedIncrement
	//	for(int i = 0; i < agent; ++i)
	//	{
	//		glm::vec3 speedA(0.f),speedS(0.f),speedC(0.f);
	//		float countA=0,countS=0,countC=0;
	//		for(int j = 0; j < agent; ++j)
	//		{
	//			if(i == j) continue;
	//			glm::vec3 direction = position[j] - position[i];
	//			float dist = glm::length(direction);
	//
	//			// separation/alignment/cohesion
	//			if (dist < rs )
	//			{
	//				speedS -= direction * ws;
	//				countS++;
	//			}
	//			if (dist < ra )
	//			{
	//				speedA += speed[j]  * wa;
	//				countA++;
	//			}
	//			if (dist < rc )
	//			{
	//				speedC += direction * wc;
	//				countC++;
	//			}
	//		}
	//		speedC = countC>0?speedC/countC:speedC;
	//		speedA = countA>0?speedA/countA:speedA;
	//		speedS = countS>0?speedS/countS:speedS;
	//		speedIncrement[i] = speedC+speedA+speedS;
	//	}
	//
	//	// sum the speedIncrement to the speed
	//	for(int i = 0; i < agent; ++i)
	//	{
	//		speed[i] += speedIncrement[i];
	//
	//		// limit the speed;
	//		const float maxSpeed = 0.3;
	//		float s = glm::length(speed[i]);
	//		if (s>maxSpeed)
	//			speed[i] *= maxSpeed/s;
	//	}
	//
	//	// sum the speed to the position (Euler intégration)
	//	for(int i = 0; i < agent; ++i)
	//	{
	//		position[i] += speed[i];
	////		position[i] = glm::modf(position[i], bounds);
	//		position[i] = glm::fract(position[i]);
	//	}
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
