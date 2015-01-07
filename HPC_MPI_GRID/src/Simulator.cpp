#include "Simulator.hpp"
#include <cstdlib>
#include <iostream>
#include "ProgressBar.hpp"
#include <fstream>
#include <sstream>

#include <mpi.h>

double randDouble()
{
    return rand() / double(RAND_MAX);
}

Simulator::Simulator(
    int mpi_rank,
    int mpi_size,
    int agent,
    int step,
    double wc, double wa, double ws,
    double rc, double ra, double rs,
    bool write):
    mpi_rank(mpi_rank),
    mpi_size(mpi_size),
    agent(agent),step(step),
    wc(wc),wa(wa),ws(ws),
    rc(rc),ra(ra),rs(rs),
    write(write)
{
    init();
}


glm::ivec3 Simulator::getGridPosition(int rank)
{
    glm::ivec3 pos;
    pos.z = rank/grid_size/grid_size;
    pos.y = rank/grid_size%grid_size;
    pos.x = rank%grid_size;
    return pos;
}
int Simulator::getRank(const glm::ivec3 position)
{
    return position.x + grid_size*(
           position.y + grid_size*(
           position.z  ));
}

void Simulator::init()
{
    // compute grid size
    grid_size = 0;
    for(grid_size = 0; grid_size*grid_size*grid_size<mpi_size; ++grid_size);

    // compute grid position
    grid_position = getGridPosition(mpi_rank);

    // spawn agents
    int nbAgent = agent/grid_size/grid_size/grid_size;
    for(int i = 0; i < nbAgent; ++i)
    {
        double x = randDouble();
        double y = randDouble();
        double z = randDouble();
        glm::dvec3 pos = grid_min + (grid_max-grid_min)*(glm::dvec3(x,y,z));

        position.push_back(pos);
        speed.push_back(glm::dvec3(0.0));
        speedIncrement.push_back(glm::dvec3(0.0));
    }
}

void Simulator::run()
{
    ProgressBar progressBar;
    for(int i = 0; i < step; ++i)
    {
        oneStep();
        if (mpi_rank == 0)  progressBar.update(i/double(step));
        
        // print the result
        std::stringstream filename;
        filename << "./output/boids_" << i << ".xyz";
        save(filename.str()); 
    }
}

void Simulator::oneStep()
{
    // compute the speedIncrement
    for(int i = mpi_offset; i < mpi_offset + mpi_subsize; ++i)
    {
		glm::dvec3 speedA(0.0),speedS(0.0),speedC(0.0);
		double countA=0,countS=0,countC=0;
		for(int j = 0; j < agent; ++j)
		{
			if(i == j) continue;
			glm::dvec3 direction = position[j] - position[i];
			double dist = glm::length(direction);

			// separation/alignment/cohesion
			if (dist < rs ) { speedS -= direction * ws; countS++; }
			if (dist < ra ) { speedA += speed[j]  * wa; countA++; }
			if (dist < rc ) { speedC += direction * wc; countC++; }
		}
		speedC = countC>0?speedC/countC:speedC;
		speedA = countA>0?speedA/countA:speedA;
		speedS = countS>0?speedS/countS:speedS;


		speedIncrement[i] = speedC+speedA+speedS;
    }

    // apply the speed increment
    for(int i = mpi_offset; i < mpi_offset + mpi_subsize; ++i)
    {

		speed[i] += speedIncrement[i];

		// limit the speed;
		const double maxSpeed = 0.2;
		double s = glm::length(speed[i]);
		if (s>maxSpeed)
			speed[i] *= maxSpeed/s;

		position[i] += speed[i];
		position[i] = glm::fract(position[i]);
    }

    // share the results
    for(int i = 0; i<mpi_size; ++i)
    {
        int offset,subsize;
        computeGroupDimension(i,offset,subsize);
        MPI_Bcast(&position[offset],subsize*3,MPI_DOUBLE,i,MPI_COMM_WORLD);
        MPI_Bcast(&speed[offset],subsize*3,MPI_DOUBLE,i,MPI_COMM_WORLD);
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
