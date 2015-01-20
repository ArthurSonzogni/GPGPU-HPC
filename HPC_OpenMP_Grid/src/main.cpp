#include "CommandArguments.hpp"
#include <iostream>
#include "Simulator.hpp"
#include <cstdlib>

int main(int argc, const char *argv[])
{
	srand(0);
    CommandArguments arguments;

    // set arguments defaults values
    arguments.loadDefault();

    // parse the argument
    arguments.parse(argc,argv);

    // print the arguments
    std::cout << "arguments : " << std::endl;
    std::cout << arguments.print() << std::endl;

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

    Simulator simulator(agents,steps,wc,wa,ws,rc,ra,rs,write);
    simulator.run();
    
    return 0;
}
