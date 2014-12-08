#include "CommandArguments.hpp"
#include <iostream>

int main(int argc, const char *argv[])
{
    CommandArguments arguments;
    arguments.setDefault("agents", "640");
    arguments.setDefault("steps", "500");
    arguments.setDefault("wc", "12");
    arguments.setDefault("wa", "15");
    arguments.setDefault("ws", "35");

    arguments.setDefault("rc", "0.11");
    arguments.setDefault("ra", "0.15");
    arguments.setDefault("rs", "0.01");
    arguments.parse(argc,argv);

    std::cout << arguments.print() << std::endl;
    
    return 0;
}
