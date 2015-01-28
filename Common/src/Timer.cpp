#include "Timer.hpp"
#include <iostream>

void Timer::start()
{
	startTime = clock();
}

void Timer::display(const char *message)
{
	clock_t dt = clock() - startTime;
	if(message)
		std::cout << message << " : ";
	else
		std::cout << "Timer : ";
	std::cout << ((float)dt)/CLOCKS_PER_SEC << "s" << std::endl;
}
