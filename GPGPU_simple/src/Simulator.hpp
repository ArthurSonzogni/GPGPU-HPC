#ifndef SIMULATOR_N4GERFZE
#define SIMULATOR_N4GERFZE

#include <vector>
#include <string>

class Simulator
{
    public:
        Simulator(
            int agent,
            int step,
            float wc, float wa, float ws,
            float rc, float ra, float rs,
            bool write);

        void run();

    private:
        // CPU DATA
        std::vector<float>  position;

        // GPU DATA
        float* position_cuda;
        float* speed_cuda;
        float* speedIncrement_cuda;


        void init();

        int agent,step;
        float wc,wa,ws,rc,ra,rs;
        bool write;

        void oneStep();
        void save(const std::string& filename);
};

#endif /* end of include guard: SIMULATOR_N4GERFZE */
