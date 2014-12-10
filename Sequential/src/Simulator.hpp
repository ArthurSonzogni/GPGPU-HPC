#ifndef SIMULATOR_N4GERFZE
#define SIMULATOR_N4GERFZE

#include "glm.hpp"
#include <vector>
#include <string>

class Simulator
{
    public:
        Simulator(
            int agent,
            int step,
            float wc, float wa, float ws,
            float rc, float ra, float rs);

        void run();

    private:
		static glm::vec3 bounds;
        std::vector<glm::vec3> position;
        std::vector<glm::vec3> speed;
        std::vector<glm::vec3> speedIncrement;

        void init();

        int agent,step;
        float wc,wa,ws,rc,ra,rs;

        void oneStep();
        void save(const std::string& filename);
};

#endif /* end of include guard: SIMULATOR_N4GERFZE */
