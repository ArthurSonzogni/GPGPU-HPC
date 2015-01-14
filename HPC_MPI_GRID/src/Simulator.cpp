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
    pos.z = (rank/grid_size)/grid_size;
    pos.y = (rank/grid_size)%grid_size;
    pos.x = (rank%grid_size);
    return pos;
}
int Simulator::getRank(const glm::ivec3 position)
{
    return (position.x + grid_size)%grid_size + grid_size*(
           (position.y + grid_size)%grid_size + grid_size*(
           (position.z + grid_size)%grid_size  ));
}

void Simulator::init()
{
    // compute grid size
    grid_size = 0;
    for(grid_size = 0; grid_size*grid_size*grid_size<mpi_size; ++grid_size);

    // compute grid position
    grid_position = getGridPosition(mpi_rank);

    // compute grid_max and grid_min
    grid_min = glm::dvec3(grid_position ) / double(grid_size);
    grid_max = glm::dvec3(grid_position + glm::ivec3(1,1,1)) / double(grid_size);

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

void Simulator::computeVirtual()
{
    meanPosition = glm::dvec3(0.0);
    meanSpeed = glm::dvec3(0.0);
    double nbAgent = position.size();

    if (nbAgent <= 0) return;

    for(std::list<glm::dvec3>::iterator it = position.begin(); it != position.end(); ++it)
    {
       meanPosition +=  *it;
    }

    for(std::list<glm::dvec3>::iterator it = speed.begin(); it != speed.end(); ++it)
    {
       meanSpeed +=  *it;
    }

    meanPosition /= nbAgent;
    meanSpeed /= nbAgent;
}

void Simulator::virtualTransmission()
{
    glm::dvec3 buffer[2*3*3*3];
    MPI_Request sendReq[3*3*3];
    MPI_Status status;

    // reception
    {
        int bufferPosition = 0;
        for(int x = -1 ; x <= 1 ; ++x )
        for(int y = -1 ; y <= 1 ; ++y )
        for(int z = -1 ; z <= 1 ; ++z )
        {
            int rank = getRank(grid_position + glm::ivec3(x,y,z)); 
            if ( rank != mpi_rank )
            {
                MPI_Irecv(&buffer[2*bufferPosition], 6 , MPI_DOUBLE, rank, 0 , MPI_COMM_WORLD, &sendReq[bufferPosition]);
            }
            ++bufferPosition;
        }
    }
    // envoie
    {
        int bufferPosition = 0;
        for(int x = -1 ; x <= 1 ; ++x )
        for(int y = -1 ; y <= 1 ; ++y )
        for(int z = -1 ; z <= 1 ; ++z )
        {
            int rank = getRank(grid_position + glm::ivec3(x,y,z)); 
            if ( rank != mpi_rank )
            {
                double buffer[6];
                
                buffer[0] = meanPosition.x;
                buffer[1] = meanPosition.y;
                buffer[2] = meanPosition.z;
                buffer[3] = meanSpeed.x;
                buffer[4] = meanSpeed.y;
                buffer[5] = meanSpeed.z;
                MPI_Send(buffer, 6, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
            }
            ++bufferPosition;
        }
    }

    // attente de reception
    {
        virtualPosition.clear();
        virtualSpeed.clear();
        
        int bufferPosition = 0;
        for(int x = -1 ; x <= 1 ; ++x )
        for(int y = -1 ; y <= 1 ; ++y )
        for(int z = -1 ; z <= 1 ; ++z )
        {
            int rank = getRank(grid_position + glm::ivec3(x,y,z)); 
            if ( rank != mpi_rank )
            {
                MPI_Wait(&sendReq[bufferPosition],&status);
                virtualPosition.push_back(glm::dvec3(
                    buffer[0],buffer[1],buffer[2]
                ));
                virtualSpeed.push_back(glm::dvec3(
                    buffer[3],buffer[4],buffer[5]
                ));
            }
            ++bufferPosition;
        }
    }
}

void Simulator::run()
{
    ProgressBar progressBar;
    for(int i = 0; i < step; ++i)
    {
        std::cout << mpi_rank << std::endl;
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
    computeVirtual();
    virtualTransmission();
    compute();
}

void Simulator::compute()
{
    // compute the speedIncrement
    {
        std::list<glm::dvec3>::iterator my_position,other_position, other_speed, my_speedIncrement;

        for(my_position = position.begin(), my_speedIncrement = speedIncrement.begin();
            my_position != position.end();
            ++my_position,++my_speedIncrement)
        {
            glm::dvec3 speedA(0.0),speedS(0.0),speedC(0.0);
            double countA=0,countS=0,countC=0;

            for(other_position = position.begin(), other_speed = speed.begin();
                other_position != position.end();
                ++other_position,++other_speed)
            {
                if ( other_position != my_position )
                {
                    glm::dvec3 direction = (*other_position) - (*my_position);
                    double dist = glm::length(direction);

                    // separation/alignment/cohesion
                    if (dist < rs ) { speedS -= direction * ws; countS++; }
                    if (dist < ra ) { speedA += (*other_speed)  * wa; countA++; }
                    if (dist < rc ) { speedC += direction * wc; countC++; }

                    speedC = countC>0?speedC/countC:speedC;
                    speedA = countA>0?speedA/countA:speedA;
                    speedS = countS>0?speedS/countS:speedS;
                    
                }

            }

            *my_speedIncrement = speedC+speedA+speedS;
        }
    }


    // apply the speed increment
    {
        std::list<glm::dvec3>::iterator my_position, my_speed, my_speedIncrement;

        for(my_position = position.begin(), my_speed = speed.begin(), my_speedIncrement = speedIncrement.begin() ;
            my_position != position.end() ;
            ++my_position, ++my_speed, ++my_speedIncrement)
        {
            // increment the speed
            *my_speed += *my_speedIncrement;

            // limit the speed
            const double maxSpeed = 0.2;
            double s = glm::length(*my_speed);
            if (s>maxSpeed)
                *my_speed *= maxSpeed/s;

            *my_position += *my_speed;
            *my_position = glm::fract(*my_position);
        }
    }
}

void Simulator::outInTransmission()
{
    
}

void Simulator::save(const std::string& filename)
{
    //std::ofstream file;
    //file.open(filename.c_str());

    //for(int i = 0; i < agent; ++i)
    //{
        //file
            //<< position[i].x << " "
            //<< position[i].y << " "
            //<< position[i].z
            //<< std::endl;
    //}

    //file.close();
}
