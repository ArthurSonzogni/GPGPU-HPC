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

void Simulator::computeGroupDimension(int i, int& offset, int& size)
{
    int min_mpi_size = agent/mpi_size;
    int remainder_mpi_size = agent % min_mpi_size;
    if (i < remainder_mpi_size)
    {
        offset = i * (min_mpi_size + 1);
        size = min_mpi_size + 1;
    }
    else
    {
        offset = remainder_mpi_size * (min_mpi_size + 1) + (i - remainder_mpi_size) * min_mpi_size;
        size = min_mpi_size;
    }
}

Simulator::Simulator(
    int mpi_rank,
    int mpi_size,
    int agent,
    int step,
    double wc, double wa, double ws,
    double rc, double ra, double rs,
    double vmax,
    bool write):
    mpi_rank(mpi_rank),
    mpi_size(mpi_size),
    agent(agent),step(step),
    wc(wc),wa(wa),ws(ws),
    rc(rc),ra(ra),rs(rs),
    vmax(vmax),
    write(write)
{
    // compute the mpi_subsize
    computeGroupDimension(mpi_rank, mpi_offset, mpi_subsize);
    init();
}


void Simulator::init()
{

    position.resize(agent);
    speed.resize(agent);
    speedIncrement.resize(agent);

	if (mpi_rank == 0)
	{
        for(int i = 0; i < agent; ++i)
		{
            double x = randDouble();
            double y = randDouble();
            double z = randDouble();
            position[i] = glm::dvec3(x,y,z);
		}
	}

    // share the data
    MPI_Bcast(&position[0],position.size()*3,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&speed[0],speed.size()*3,MPI_DOUBLE,0,MPI_COMM_WORLD);
}

void Simulator::run()
{
    ProgressBar *progressBar = NULL;
	if(mpi_rank == 0) progressBar = new ProgressBar;
    for(int i = 0; i < step; ++i)
    {
        oneStep();
        if (mpi_rank == 0)  progressBar->update(i/double(step));
        
        // print the result
        std::stringstream filename;
        filename << "./output/boids_" << i << ".xyz";
        save(filename.str()); 
    }
	if(progressBar) delete progressBar;
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

    // apply the speed increment
    for(int i = mpi_offset; i < mpi_offset + mpi_subsize; ++i)
    {

		speed[i] += speedIncrement[i];

		// limit the speed;
		double s = glm::length(speed[i]);
		if (s>vmax)
			speed[i] *= vmax/s;

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
