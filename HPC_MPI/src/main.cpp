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
    float wc =  arguments.get<float>("wc");
    float wa =  arguments.get<float>("wa");
    float ws =  arguments.get<float>("ws");
    float rc =  arguments.get<float>("rc");
    float ra =  arguments.get<float>("ra");
    float rs =  arguments.get<float>("rs");
    
    Simulator simulator(mpi_rank,mpi_size,agents,steps,wc,wa,ws,rc,ra,rs);
    simulator.run();
    MPI_Finalize();
    
    return 0;
}
