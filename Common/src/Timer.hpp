#ifndef __TIMER__
#define __TIMER__
#include <time.h>

class Timer
{
	public:
		void start();
		void display(const char *message = NULL);
	private:
		clock_t startTime;
};

#endif
