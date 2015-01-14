#ifndef SIMULATOR_N4GERFZE
#define SIMULATOR_N4GERFZE

#include "glm.hpp"
#include <vector>
#include <string>
#include <list>

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


        // boids
        std::list<glm::dvec3> position;
        std::list<glm::dvec3> speed;
        std::list<glm::dvec3> speedIncrement;

        // virtual data from this block
        glm::dvec3 meanPosition;
        glm::dvec3 meanSpeed;
        
        // virtual boids from neighbours blocks
        std::list<glm::dvec3> virtualPosition;
        std::list<glm::dvec3> virtualSpeed;

        // programme phase
        void init();
        void computeVirtual();
        void virtualTransmission();
        void compute();
        void outInTransmission();

        // mpi data
        int mpi_rank;
        int mpi_size;

        glm::ivec3 grid_position;
        glm::dvec3 grid_min;
        glm::dvec3 grid_max;

        int grid_size;

        glm::ivec3 getGridPosition(int rank);
        int getRank(const glm::ivec3 position);

        int agent,step;
        double wc,wa,ws,rc,ra,rs;
        bool write;

        void oneStep();
        void save(const std::string& filename);

};

#endif /* end of include guard: SIMULATOR_N4GERFZE */
