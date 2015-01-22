#include "Simulator.hpp"
#include <cstdlib>
#include <iostream>
#include "ProgressBar.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>

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
    for(int i = 0; i < agent; ++i)
    {
        double x = randDouble();
        double y = randDouble();
        double z = randDouble();

        if (
                x>grid_min.x and x<grid_max.x  and
                y>grid_min.y and y<grid_max.y  and
                z>grid_min.z and z<grid_max.z
        ){
            Boid b;
            b.position = glm::dvec3(x,y,z);
            b.speed = glm::dvec3(0,0,0);
            boids.push_back(b);
        }

    }

    // compute Neighbour Rank
    computeNeibourRank();
}

void Simulator::computeVirtual()
{
    mean.boid.position = glm::dvec3(0.0);
    mean.boid.speed = glm::dvec3(0.0);
    double nbAgent = boids.size();
    mean.weight = nbAgent;

    if (nbAgent <= 0) return;

    for(std::list<Boid>::iterator it = boids.begin(); it != boids.end(); ++it)
    {
        mean.boid.position += it->position;
        mean.boid.speed += it->speed;
    }

    mean.boid.position /= nbAgent;
    mean.boid.speed /= nbAgent;
}

void Simulator::virtualTransmission()
{
    BoidWeight in[3][3][3];
    MPI_Request sendReq[3][3][3];
    MPI_Status status;

    // reception
    {
        for(int x = 0 ; x <= 2 ; ++x )
        for(int y = 0 ; y <= 2 ; ++y )
        for(int z = 0 ; z <= 2 ; ++z )
        {
            int rank = neighbourRank[x][y][z];
            if ( rank != mpi_rank )
            {
                MPI_Irecv(&in[x][y][z], 7 , MPI_DOUBLE, rank, 0 , MPI_COMM_WORLD, &sendReq[x][y][z]);
            }
        }
    }
    // envoie
    {
        for(int x = 0 ; x <= 2 ; ++x )
        for(int y = 0 ; y <= 2 ; ++y )
        for(int z = 0 ; z <= 2 ; ++z )
        {
            int rank = neighbourRank[x][y][z];
            if ( rank != mpi_rank )
            {
                MPI_Send(&mean, 7, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
            }
        }
    }

    // attente de reception
    {
        virtualBoids.clear();
        
        for(int x = 0 ; x <= 2 ; ++x )
        for(int y = 0 ; y <= 2 ; ++y )
        for(int z = 0 ; z <= 2 ; ++z )
        {
            int rank = neighbourRank[x][y][z];
            if ( rank != mpi_rank )
            {
                MPI_Wait(&sendReq[x][y][z],&status);
                virtualBoids.push_back(in[x][y][z]);
            }
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
        filename << "./output/boids_" << i << "_" << mpi_rank << ".xyz";
        save(filename.str()); 
    }
}

void Simulator::oneStep()
{
    //wa = 10.0; // TODO (REMOVE THIS IF YOU CAN)
    computeVirtual();
    virtualTransmission();
    compute();
    outInTransmission();
}

void Simulator::compute()
{
    std::list<Boid>::iterator my_boid,other_boid;
    std::list<BoidWeight>::iterator cell_boids;
    std::list<glm::dvec3>::iterator my_speed_increment;

    // compute the speedIncrement
    {
        for(my_boid = boids.begin(), my_speed_increment = speedIncrement.begin();
            my_boid != boids.end();
            ++my_boid, ++my_speed_increment)
        {
            glm::dvec3 speedA(0.0),speedS(0.0),speedC(0.0);
            double countA=0,countS=0,countC=0;

            // interation avec les boids de la cellule
            for(other_boid = boids.begin(); other_boid != boids.end(); ++other_boid)
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

            // interaction avec les boids des cellules voisines
            for(cell_boids = virtualBoids.begin(); cell_boids != virtualBoids.end(); ++cell_boids)
            {
                glm::dvec3 direction = cell_boids->boid.position - my_boid->position;
                double dist = glm::length(direction);

                double w = cell_boids->weight;
                // separation/alignment/cohesion
                if (dist < rs ) { speedS -= w*direction; countS+=w; }
                if (dist < ra ) { speedA += w*cell_boids->boid.speed; countA+=w; }
                if (dist < rc ) { speedC += w*direction; countC+=w; }

            }
            speedC = countC>0?speedC/countC:speedC;
            speedA = countA>0?speedA/countA:speedA;
            speedS = countS>0?speedS/countS:speedS;

            *my_speed_increment =
                wc*speedC+
                wa*speedA+
                ws*speedS;
        }
    }


    // apply the speed increment
    {
        for(my_boid = boids.begin(), my_speed_increment = speedIncrement.begin();
            my_boid != boids.end();
            ++my_boid, ++my_speed_increment)
        {
            // increment the speed
            my_boid->speed += *my_speed_increment;

            // limit the speed
            const double maxSpeed = 0.08;
            double s = glm::length(my_boid->speed);
            if (s>maxSpeed)
                my_boid->speed *= maxSpeed/s;

            my_boid->position += my_boid->speed;
            my_boid->position = glm::fract(my_boid->position);
        }
    }
}

void Simulator::outInTransmission()
{
    //return;

    // extraction des sorties
    std::vector<Boid> out[3][3][3];

    std::list<Boid>::iterator my_boid;

    // detect outter
    for(my_boid = boids.begin(); my_boid != boids.end(); /* no increment */)
    {
        glm::ivec3 ipos = glm::ivec3((my_boid->position) * double(grid_size));
        if ( ipos != grid_position )
        {
            glm::ivec3 d = ipos - grid_position + glm::ivec3(1,1,1);
            d = glm::min(d,glm::ivec3(2,2,2));
            d = glm::max(d,glm::ivec3(0,0,0));
            //std::cout << d.x << " " << d.y << " " << d.z << "----" << std::endl;

            out[d.x][d.y][d.z].push_back( *my_boid );
            my_boid = boids.erase(my_boid);
        }
        else
        {
            ++my_boid;
        }
    }

    //return;

    // reservation de buffer pour la reception
    int inDimension[3][3][3];
    std::vector<Boid> in[3][3][3];

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
                int sendBuffer = out[x][y][z].size();
                MPI_Send(&sendBuffer, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
                //std::cout << mpi_rank << " send " << sendBuffer << " boids to " << rank << std::endl;
            }
        }
    }

    // attente de reception
    {
        for(int x = 0 ; x <= 2 ; ++x )
        for(int y = 0 ; y <= 2 ; ++y )
        for(int z = 0 ; z <= 2 ; ++z )
        {
            int rank = neighbourRank[x][y][z];
            if ( rank != mpi_rank )
            {
                MPI_Wait(&sendReq[x][y][z],&status);
                //std::cout << mpi_rank << " receive " << inDimension[x][y][z] << " boids  from " << rank << std::endl;
                in[x][y][z].resize(inDimension[x][y][z]);
            }
        }
    }

    
    //reception des boids
    {
        for(int x = 0 ; x <= 2 ; ++x )
        for(int y = 0 ; y <= 2 ; ++y )
        for(int z = 0 ; z <= 2 ; ++z )
        {
            int rank = neighbourRank[x][y][z];
            if ( rank != mpi_rank and inDimension[x][y][z] )
            {
                //std::cout << mpi_rank << " is receiving from " << rank << " : " << inDimension[x][y][z] << std::endl;
                MPI_Irecv(&(in[x][y][z][0]), 3*2*inDimension[x][y][z] , MPI_DOUBLE, rank, 0 , MPI_COMM_WORLD, &sendReq[x][y][z]);
                //std::cout << inDimension[x][y][z] << std::endl;
            }
        }
    }

    // envoie des boids
    {
        for(int x = 0 ; x <= 2 ; ++x )
        for(int y = 0 ; y <= 2 ; ++y )
        for(int z = 0 ; z <= 2 ; ++z )
        {
            int rank = neighbourRank[x][y][z];
            if ( rank != mpi_rank and out[x][y][z].size())
            {
                //std::cout << mpi_rank << " is sending to " << rank << " :  " << out[x][y][z].size() << std::endl;
                MPI_Send(&(out[x][y][z][0]), 3*2*out[x][y][z].size(), MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
            }
        }
    }

    // attente de reception
    {
        for(int x = 0 ; x <= 2 ; ++x )
        for(int y = 0 ; y <= 2 ; ++y )
        for(int z = 0 ; z <= 2 ; ++z )
        {
            int rank = neighbourRank[x][y][z];
            if ( rank != mpi_rank and inDimension[x][y][z])
            {
                //std::cout << mpi_rank << " is waiting : " << rank << std::endl;
                MPI_Wait(&sendReq[x][y][z],&status);
                boids.insert(boids.end(),in[x][y][z].begin(),in[x][y][z].end());
            }
        }
    }


}

void Simulator::save(const std::string& filename)
{
    if (write)
    {
        std::ofstream file;
        file.open(filename.c_str());

        for(std::list<Boid>::iterator p = boids.begin(); p !=  boids.end(); ++p)
        {
            file
                << p->position.x << " "
                << p->position.y << " "
                << p->position.z
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
