#include "CommandArguments.hpp"

void CommandArguments::parse(int argc, const char *argv[])
{
    for(int i = 1; i<argc; ++i)
    {
        if (argv[i][0] == '-' && (i+1)<argc)
        {
            arguments[argv[i]+1]=argv[i+1];
            ++i;
        }
    }
}

void CommandArguments::setDefault(const std::string& name, const std::string& value)
{
    arguments[name] = value;
}

std::string CommandArguments::print()
{
    std::string value = "";
    for(ArgMap::iterator it = arguments.begin(); it!= arguments.end(); ++it)
    {
        value += "[ -" + (it->first) + " = " + (it->second) + " ] ";
    }
    return value;
}
