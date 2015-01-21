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

    // compute Neighbour Rank
    computeNeibourRank();
}

void Simulator::computeVirtual()
{
    mean.position = glm::dvec3(0.0);
    mean.speed = glm::dvec3(0.0);
    double nbAgent = position.size();

    if (nbAgent <= 0) return;

    for(std::list<Boid>::iterator it = boids.begin(); it != boids.end(); ++it)
    {
        mean.position += it->position;
        mean.speed += it->speed;
    }

    mean.position /= nbAgent;
    mean.speed /= nbAgent;
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
            int rank = neighbourRank[x+1][y+1][z+1];
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
            int rank = neighbourRank[x+1][y+1][z+1];
            if ( rank != mpi_rank )
            {
                double buffer[6];
                
                buffer[0] = mean.position.x;
                buffer[1] = mean.position.y;
                buffer[2] = mean.position.z;
                buffer[3] = mean.speed.x;
                buffer[4] = mean.speed.y;
                buffer[5] = mean.speed.z;
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
            int rank = neighbourRank[x+1][y+1][z+1];
            if ( rank != mpi_rank )
            {
                MPI_Wait(&sendReq[bufferPosition],&status);
                Boid newBoid;
                newBoid.position = buffer[2*bufferPosition];
                newBoid.speed = buffer[2*bufferPosition + 1];
                virtualBoids.push_back(newBoid);
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
    outInTransmission();
}

void Simulator::compute()
{
    std::list<Boid>::iterator my_boid,other_boid;
    std::list<glm::dvec3> my_speed_increment
    // compute the speedIncrement
    {
        for(my_boid = boid.begin(), my_speed_increment = speedIncrement.begin();
            my_boid = boid.end(), my_speed_increment = speedIncrement.end();
            ++my_boid, ++my_speed_increment)
        {
            glm::dvec3 speedA(0.0),speedS(0.0),speedC(0.0);
            double countA=0,countS=0,countC=0;

            for(other_boid = boid.begin(); other_boid = boid.end(); ++other_boid)
            {
                if ( other_boid != my_boid )
                {
                    glm::dvec3 direction = other_boid->position - my_boid->position;
                    double dist = glm::length(direction);

                    // separation/alignment/cohesion
                    if (dist < rs ) { speedS -= direction; countS++; }
                    if (dist < ra ) { speedA += other_boid->speed; countA++; }
                    if (dist < rc ) { speedC += direction; countC++; }
                }

            }
            speedC = countC>0?speedC/countC:speedC;
            speedA = countA>0?speedA/countA:speedA;
            speedS = countS>0?speedS/countS:speedS;

            *my_speedIncrement =
                wc*speedC+
                wa*speedA+
                ws*speedS;
            // TODO from HERE (TODO)
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
    // extraction des sorties
    std::vector<glm::dvec3> outPosition[3][3][3];
    std::vector<glm::dvec3> outSpeed[3][3][3];

    std::list<glm::dvec3>::iterator my_position, my_speed;

    // detect outter
    for(my_position = position.begin(), my_speed = speed.begin() ;
        my_position != position.end() ;)
    {
        glm::ivec3 ipos = glm::ivec3((*my_position) * double(grid_size));
        if ( ipos != grid_position )
        {
            glm::ivec3 d = ipos - grid_position + glm::ivec3(1,1,1);
            d = glm::min(d,glm::ivec3(0,0,0));
            d = glm::max(d,glm::ivec3(2,2,2));

            outPosition[d.x][d.y][d.z].push_back( *my_position );
            outPosition[d.x][d.y][d.z].push_back( *my_speed );

            my_position = position.erase(my_position);
            my_speed = speed.erase(my_speed);
        }
        else
        {
            ++my_position;
            ++my_speed;
        }
    }

    // reservation de buffer pour la reception
    int inDimension[3][3][3];
    std::vector<glm::dvec3> inPosition[3][3][3];
    std::vector<glm::dvec3> inSpeed[3][3][3];

    // MPI request/status
    MPI_Request sendReq[3][3][3];
    MPI_Status status;

    // reception de la dimension
    {
        for(int x = 0 ; x <= 2 ; ++x )
        for(int y = 0 ; y <= 2 ; ++y )
        for(int z = 0 ; z <= 2 ; ++z )
        {
            int rank = neighbourRank[x][y][z];
            if ( rank != mpi_rank )
            {
                MPI_Irecv(&inDimension[x][y][z], 1 , MPI_INT, rank, 0 , MPI_COMM_WORLD, &sendReq[x][y][z]);
            }
        }
    }

    // envoie de la dimension
    {
        for(int x = 0 ; x <= 2 ; ++x )
        for(int y = 0 ; y <= 2 ; ++y )
        for(int z = 0 ; z <= 2 ; ++z )
        {
            int rank = neighbourRank[x][y][z];
            if ( rank != mpi_rank )
            {
                int sendBuffer = outPosition[x][y][z].size();
                MPI_Send(&sendBuffer, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
                std::cout << mpi_rank << " send " << sendBuffer << " boids to " << rank << std::endl;
            }
        }
    }

    // attente de reception
    {
        virtualPosition.clear();
        virtualSpeed.clear();
        
        for(int x = 0 ; x <= 2 ; ++x )
        for(int y = 0 ; y <= 2 ; ++y )
        for(int z = 0 ; z <= 2 ; ++z )
        {
            int rank = neighbourRank[x][y][z];
            if ( rank != mpi_rank )
            {
                MPI_Wait(&sendReq[x][y][z],&status);
                std::cout << mpi_rank << " receive " << inDimension[x][y][z] << " boids  from " << rank << std::endl;
            }
        }
    }

    // reception des boids
    //{
        //int bufferPosition = 0;
        //for(int x = 0 ; x <= 2 ; ++x )
        //for(int y = 0 ; y <= 2 ; ++y )
        //for(int z = 0 ; z <= 2 ; ++z )
        //{
            //int rank = neighbourRank[x][y][z];
            //if ( rank != mpi_rank )
            //{
                //in
                //MPI_Irecv(&inDimension[x][y][z], 1 , MPI_INT, rank, 0 , MPI_COMM_WORLD, &sendReq[x][y][z]);
            //}
        //}
    //}

    //// envoie de la dimension
    //{
        //for(int x = 0 ; x <= 2 ; ++x )
        //for(int y = 0 ; y <= 2 ; ++y )
        //for(int z = 0 ; z <= 2 ; ++z )
        //{
            //int rank = getRank(grid_position + glm::ivec3(x-1,y-1,z-1)); 
            //if ( rank != mpi_rank )
            //{
                //int sendBuffer = outPosition[x][y][z].size();
                //MPI_Send(&sendBuffer, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
                //std::cout << mpi_rank << " send " << sendBuffer << " boids to " << rank << std::endl;
            //}
        //}
    //}

    //// attente de reception
    //{
        //virtualPosition.clear();
        //virtualSpeed.clear();
        
        //for(int x = 0 ; x <= 2 ; ++x )
        //for(int y = 0 ; y <= 2 ; ++y )
        //for(int z = 0 ; z <= 2 ; ++z )
        //{
            //int rank = getRank(grid_position + glm::ivec3(x-1,y-1,z-1)); 
            //if ( rank != mpi_rank )
            //{
                //MPI_Wait(&sendReq[x][y][z],&status);
                //std::cout << mpi_rank << " receive " << inDimension[x][y][z] << " boids  from " << rank << std::endl;
            //}
        //}
    //}

    std::cout << "This is the end for me : " << mpi_rank << std::endl;
    exit(0);
}

void Simulator::save(const std::string& filename)
{
    if (mpi_rank == 0 and write)
    {
        std::ofstream file;
        file.open(filename.c_str());

        for(std::list<glm::dvec3>::iterator p = position.begin(); p != position.end(); ++p)
        {
            file
                << (*p).x << " "
                << (*p).y << " "
                << (*p).z
                << std::endl;
        }

        file.close();
    }
}

void Simulator::computeNeibourRank()
{
    for(int x = 0 ; x <= 2 ; ++x )
    for(int y = 0 ; y <= 2 ; ++y )
    for(int z = 0 ; z <= 2 ; ++z )
    {
        neighbourRank[x][y][z] = getRank(grid_position + glm::ivec3(x-1,y-1,z-1)); 
    }
}
