#include "CommandArguments.hpp"
#include <iostream>
#include "Simulator.hpp"

int main(int argc, const char *argv[])
{
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

    Simulator simulator(agents,steps,wc,wa,ws,rc,ra,rs);
    simulator.run();
    
    return 0;
}