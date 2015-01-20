#include "CommandArguments.hpp"
#include <iostream>
#include "Simulator.hpp"

int main(int argc, const char *argv[])
{
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
    float wc =  arguments.get<float>("wc");
    float wa =  arguments.get<float>("wa");
    float ws =  arguments.get<float>("ws");
    float rc =  arguments.get<float>("rc");
    float ra =  arguments.get<float>("ra");
    float rs =  arguments.get<float>("rs");
    bool write = arguments.get<bool>("write");

    Simulator simulator(agents,steps,wc,wa,ws,rc,ra,rs,write);
    simulator.run();
    
    return 0;
}
