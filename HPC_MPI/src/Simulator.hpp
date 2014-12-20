#ifndef SIMULATOR_N4GERFZE
#define SIMULATOR_N4GERFZE

#include "glm.hpp"
#include <vector>
#include <string>

class Simulator
{
    public:
        Simulator(
            int mpi_rank,
            int mpi_size,
            int agent,
            int step,
            double wc, double wa, double ws,
            double rc, double ra, double rs,
            bool write);

        void run();

    private:
        std::vector<glm::dvec3> position;
        std::vector<glm::dvec3> speed;
        std::vector<glm::dvec3> speedIncrement;


        int mpi_rank;
        int mpi_size;

        int mpi_offset;
        int mpi_subsize;

        void init();

        int agent,step;
        double wc,wa,ws,rc,ra,rs;
        bool write;

        void oneStep();
        void save(const std::string& filename);

        void computeGroupDimension(int i, int& offset, int& size);
};

#endif /* end of include guard: SIMULATOR_N4GERFZE */
