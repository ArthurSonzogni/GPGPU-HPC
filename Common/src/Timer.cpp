#include "Timer.hpp"
#include <sys/time.h>
#include <iostream>
using namespace std;

long getSeconds() { 
    timeval t;    
    gettimeofday(&t, NULL);
    return t.tv_sec * 1000 + t.tv_usec /1000.0;
}


void Timer::start()
{
    startTime = getSeconds();
}

void Timer::display(const char *message)
{
	if(message)
		std::cerr << "    " << message << " : ";
	else
		std::cerr << "Timer : ";
	std::cerr << float( getSeconds() - startTime )/1000.f << "s" << std::endl;
}
