#include "CommandArguments.hpp"
#include <iostream>
#include "Simulator.hpp"
#include <mpi.h>
#include <cstdlib>
#include <Timer.hpp>

int main(int argc, const char *argv[])
{
	Timer chrono;
	srand(0);
    char ** argvv = const_cast<char**>(argv);
    MPI_Init(&argc,&argvv);

    int mpi_rank;
    int mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);

    CommandArguments arguments;

    // set arguments defaults values
    arguments.loadDefault();

    // parse the argument
    arguments.parse(argc,argv);

    // print the arguments
    if (mpi_rank == 0)
    {
        std::cout << "arguments : " << std::endl;
        std::cout << arguments.print() << std::endl;
    }

    // extract the arguments
    int agents = arguments.get<int>("agents");
    int steps  = arguments.get<int>("steps");
    double wc =  arguments.get<double>("wc");
    double wa =  arguments.get<double>("wa");
    double ws =  arguments.get<double>("ws");
    double rc =  arguments.get<double>("rc");
    double ra =  arguments.get<double>("ra");
    double rs =  arguments.get<double>("rs");
    double vmax =  arguments.get<double>("vmax");
    bool write = arguments.get<bool>("write");
    
	if(mpi_rank == 0) chrono.start();
    Simulator simulator(mpi_rank,mpi_size,agents,steps,wc,wa,ws,rc,ra,rs,vmax,write);
	if(mpi_rank == 0) chrono.display("Initialization");

	if(mpi_rank == 0) chrono.start();
    simulator.run();
	if(mpi_rank == 0) chrono.display("Simulation time");
    MPI_Finalize();
    
    return 0;
}
