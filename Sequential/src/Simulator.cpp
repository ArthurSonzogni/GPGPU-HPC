#include "Simulator.hpp"
#include <cstdlib>
#include <iostream>
#include "ProgressBar.hpp"
#include <fstream>
#include <sstream>

glm::vec3 Simulator::bounds = glm::vec3(1.f,1.f,1.f);

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
	position.reserve(agent);
	speed.reserve(agent);
	speedIncrement.reserve(agent);
	for(int i = 0; i < agent; ++i)
	{
		position.push_back(glm::vec3(randDouble(),randDouble(),randDouble()));
		speed.push_back(glm::vec3(0.0,0.0,0.0));
		speedIncrement.push_back(glm::vec3(0.0,0.0,0.0));
	}
}

void Simulator::run()
{
	ProgressBar progressBar;
	for(int i = 0; i < step; ++i)
	{
		oneStep();
		progressBar.update(i/float(step));

		// print the result
		std::stringstream filename;
		filename << "./output/boids_" << i << ".xyz";
		save(filename.str()); 
	}
}

void Simulator::oneStep()
{
	// compute the speedIncrement
	for(int i = 0; i < agent; ++i)
	{
		glm::vec3 speedA(0.f),speedS(0.f),speedC(0.f);
		float countA=0,countS=0,countC=0;
		for(int j = 0; j < agent; ++j)
		{
			if(i == j) continue;
			glm::vec3 direction = position[j] - position[i];
			float dist = glm::length(direction);

			// separation/alignment/cohesion
			if (dist < rs )
			{
				speedS -= direction * ws;
				countS++;
			}
			if (dist < ra )
			{
				speedA += speed[j]  * wa;
				countA++;
			}
			if (dist < rc )
			{
				speedC += direction * wc;
				countC++;
			}
		}
		speedC = countC>0?speedC/countC:speedC;
		speedA = countA>0?speedA/countA:speedA;
		speedS = countS>0?speedS/countS:speedS;

        {
            float l = glm::length(position[i]);
            speedIncrement[i] += 0.01f*position[i]/(l*l);
        }

		speedIncrement[i] = speedC+speedA+speedS;
	}

	// sum the speedIncrement to the speed
	for(int i = 0; i < agent; ++i)
	{
		speed[i] += speedIncrement[i];

		// limit the speed;
		const float maxSpeed = 0.2;
		float s = glm::length(speed[i]);
		if (s>maxSpeed)
			speed[i] *= maxSpeed/s;
	}

	// sum the speed to the position (Euler intégration)
	for(int i = 0; i < agent; ++i)
	{
		position[i] += speed[i];
//		position[i] = glm::modf(position[i], bounds);
		position[i] = glm::fract(position[i]);
	}
}

void Simulator::save(const std::string& filename)
{
	std::ofstream file;
	file.open(filename.c_str());

	for(int i = 0; i < agent; ++i)
	{
		file
			<< position[i].x << " "
			<< position[i].y << " "
			<< position[i].z
			<< std::endl;
	}

	file.close();
}
