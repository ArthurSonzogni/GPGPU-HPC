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
            float wc, float wa, float ws,
            float rc, float ra, float rs);

        void run();

    private:
        std::vector<glm::vec3> position;
        std::vector<glm::vec3> speed;
        std::vector<glm::vec3> speedIncrement;


        int mpi_rank;
        int mpi_size;

        int mpi_offset;
        int mpi_subsize;

        void init();

        int agent,step;
        float wc,wa,ws,rc,ra,rs;

        void oneStep();
        void save(const std::string& filename);

        void computeGroupDimension(int i, int& offset, int& size);
};

#endif /* end of include guard: SIMULATOR_N4GERFZE */
