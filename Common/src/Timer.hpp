#ifndef __TIMER__
#define __TIMER__


class Timer
{
	public:
		void start();
		void display(const char *message = 0);
	private:
		long startTime;

};

#endif
