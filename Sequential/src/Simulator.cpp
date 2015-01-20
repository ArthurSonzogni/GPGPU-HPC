#include "Simulator.hpp"
#include <cstdlib>
#include <iostream>
#include "ProgressBar.hpp"
#include <fstream>
#include <sstream>

glm::dvec3 Simulator::bounds = glm::dvec3(1.0,1.0,1.0);

double randDouble()
{ 
	return rand() / double(RAND_MAX);
}

Simulator::Simulator(
		int agent,
		int step,
		double wc, double wa, double ws,
		double rc, double ra, double rs,
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
    position.resize(agent);
    speed.resize(agent);
    speedIncrement.resize(agent);
    for(int i = 0; i < agent; ++i)
    {
        double x = randDouble();
        double y = randDouble();
        double z = randDouble();
        position[i] = glm::dvec3(x,y,z);
        speed[i] = glm::dvec3(0.0,0.0,0.0);
        speedIncrement[i] = glm::dvec3(0.0,0.0,0.0);
    }
}

void Simulator::run()
{
	ProgressBar progressBar;
	for(int i = 0; i < step; ++i)
	{
		oneStep();
		progressBar.update(i/double(step));

		// print the result
		std::stringstream filename;
		filename << "./output/boids_" << i << ".xyz";
        if (write)  save(filename.str()); 
	}
}

void Simulator::oneStep()
{
	// compute the speedIncrement
	for(int i = 0; i < agent; ++i)
	{
		glm::dvec3 speedA(0.0),speedS(0.0),speedC(0.0);
		double countA=0,countS=0,countC=0;
		for(int j = 0; j < agent; ++j)
		{
			if(i == j) continue;
			glm::dvec3 direction = position[j] - position[i];
			double dist = glm::length(direction);

			// separation/alignment/cohesion
			if (dist < rs ) { speedS -= direction; countS++; }
			if (dist < ra ) { speedA += speed[j] ; countA++; }
			if (dist < rc ) { speedC += direction; countC++; }
		}
		speedC = countC>0?speedC/countC:speedC;
		speedA = countA>0?speedA/countA:speedA;
		speedS = countS>0?speedS/countS:speedS;

		speedIncrement[i] =
            wc*speedC+
            wa*speedA+
            ws*speedS;
	}

	// sum the speedIncrement to the speed
	for(int i = 0; i < agent; ++i)
	{
		speed[i] += speedIncrement[i];

		// limit the speed;
		const double maxSpeed = 0.08;
		double s = glm::length(speed[i]);
		if (s>maxSpeed)
			speed[i] *= maxSpeed/s;
	}

	// sum the speed to the position (Euler intégration)
	for(int i = 0; i < agent; ++i)
	{
		position[i] += speed[i];
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
