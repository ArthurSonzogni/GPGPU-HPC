#ifndef COMMANDARGS_3QGQZ3OP
#define COMMANDARGS_3QGQZ3OP

#include <string>
#include <iostream>
#include <sstream>
#include <map>


// Parse the command line argument
// The format should be : ./application ( -argname[i] argvalue[i] )*
class CommandArguments
{
    public:
        void parse(int argc, const char *argv[]);
        void setDefault(const std::string& name, const std::string& value);
        std::string print();
        
        // argument access
        template<class T>
        T get(const std::string arg)
        {
            T value;
            std::stringstream ss;
            ss << arguments[arg];
            ss >> value;
            return value;
        }

    private:
        typedef std::map<std::string,std::string> ArgMap;
        ArgMap arguments;
};

#endif /* end of include guard: COMMANDARGS_3QGQZ3OP */
