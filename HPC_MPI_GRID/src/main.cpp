#include "CommandArguments.hpp"
#include <iostream>
#include "Simulator.hpp"
#include <mpi.h>
#include <cstdlib>

int main(int argc, const char *argv[])
{
	srand(0);
    char ** argvv = const_cast<char**>(argv);
    MPI_Init(&argc,&argvv);

    int mpi_rank;
    int mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);


    CommandArguments arguments;

    // set arguments defaults values
    arguments.setDefault("agents", "640");
    arguments.setDefault("steps", "500");
    arguments.setDefault("wc", "12");
    arguments.setDefault("wa", "15");
    arguments.setDefault("ws", "35");
    arguments.setDefault("rc", "0.11");
    arguments.setDefault("ra", "0.15");
    arguments.setDefault("rs", "0.01");
    arguments.setDefault("write", "0");

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
    bool write = arguments.get<bool>("write");
    
    Simulator simulator(mpi_rank,mpi_size,agents,steps,wc,wa,ws,rc,ra,rs,write);
    simulator.run();
    MPI_Finalize();
    
    return 0;
}