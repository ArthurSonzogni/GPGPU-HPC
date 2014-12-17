#include "Simulator.hpp"
#include <cstdlib>
#include <iostream>
#include "ProgressBar.hpp"
#include <fstream>
#include <sstream>

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
    position.resize(agent);
    speed.resize(agent);
    speedIncrement.resize(agent);
    #pragma omp parallel
	{
        #pragma omp for
		for(int i = 0; i < agent; ++i)
		{
			position[i] = glm::vec3(randDouble(),randDouble(),randDouble());
			speed[i] = glm::vec3(0.0,0.0,0.0);
			speedIncrement[i] = glm::vec3(0.0,0.0,0.0);
		}
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
    #pragma omp parallel
	{
		// compute the speedIncrement
        #pragma omp for
		for(int i = 0; i < agent; ++i)
		{
			glm::vec3 speedInc(0.0);
			for(int j = 0; j < agent; ++j)
			{
				glm::vec3 direction = position[j] - position[i];
				float dist = glm::length(direction);

				// separation/alignment/cohesion
				if (dist < rs ) speedInc -= direction * ws;
				if (dist < ra ) speedInc += speed[j]  * wa;
				if (dist < rc ) speedInc += direction * wc;
			}
			speedIncrement[i] = speedInc * 0.01f;
		}

		// sum the speedIncrement to the speed
        #pragma omp for
		for(int i = 0; i < agent; ++i)
		{
			speed[i] += speedIncrement[i];

			// limit the speed;
			const float maxSpeed = 0.3;
			float s = glm::length(speed[i]);
			if (s>maxSpeed)
				speed[i] *= maxSpeed/s;
		}

		// sum the speed to the position (Euler intégration)
        #pragma omp for
		for(int i = 0; i < agent; ++i)
		{
			position[i] += speed[i];
            position[i] = glm::fract(position[i]);
		}
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
