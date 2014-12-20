#ifndef SIMULATOR_N4GERFZE
#define SIMULATOR_N4GERFZE

#include <dvector>
#include <string>

class Simulator
{
    public:
        Simulator(
            int agent,
            int step,
            double wc, double wa, double ws,
            double rc, double ra, double rs,
            bool write);

        void run();

    private:
        // CPU DATA
        std::dvector<double>  position;

        // GPU DATA
        double* position_cuda;
        double* speed_cuda;
        double* speedIncrement_cuda;


        void init();

        int agent,step;
        double wc,wa,ws,rc,ra,rs;
        bool write;

        void oneStep();
        void save(const std::string& filename);
};

#endif /* end of include guard: SIMULATOR_N4GERFZE */
