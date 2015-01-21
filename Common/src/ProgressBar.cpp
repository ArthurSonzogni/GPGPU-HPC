#include "ProgressBar.hpp"
#include <iostream>


ProgressBar::ProgressBar():
    previousPercent(-1)
{
    
}

ProgressBar::~ProgressBar()
{
	std::cout << std::endl;
}

void ProgressBar::update(float ratio)
{
    if (int(ratio*100)>previousPercent)
    {
        previousPercent = int(ratio*100);
        std::cout << "\r";
        std::cout << "[ " << previousPercent << " % ] ";
        std::cout << "[";
        for(int i = 0; i<previousPercent; ++i)
        std::cout << "=";
        std::cout << "]";
        std::cout << std::flush;
    }
}
