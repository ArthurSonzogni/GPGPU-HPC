#ifndef SIMULATOR_N4GERFZE
#define SIMULATOR_N4GERFZE

#include "glm.hpp"
#include <vector>
#include <string>
#include "Cell.hpp"

class Simulator
{
    public:
        Simulator(
            int agent,
            int step,
            double wc, double wa, double ws,
            double rc, double ra, double rs,
            double vmax,
            bool write);

        void run();

    private:

		Cell ***grid;

        void init();

        int agent,step;
        double wc,wa,ws,rc,ra,rs;
        double vmax,
        bool write;

        void oneStep();
        void save(const std::string& filename);
};

#endif /* end of include guard: SIMULATOR_N4GERFZE */
