#include "Simulator.hpp"
#include <cstdlib>
#include <iostream>

double randDouble()
{
    return rand() / double(RAND_MAX);
}

Simulator::Simulator(
    int agent,
    int step,
    float wc, float wa, float ws,
    float rc, float ra, float rs):
    agent(agent),step(step),
    wc(wc),wa(wa),ws(ws),
    rc(rc),ra(ra),rs(rs)
{
    init();
}


void Simulator::init()
{
    position.reserve(agent);
    speed.reserve(agent);
    speedIncrement.reserve(agent);
    for(int i = 0; i < agent; ++i)
    {
        position.push_back(glm::vec3(randDouble(),randDouble(),randDouble()));
        speed.push_back(glm::vec3(0.0,0.0,0.0));
        speedIncrement.push_back(glm::vec3(0.0,0.0,0.0));
    }
}

void Simulator::run()
{
    for(int i = 0; i < step; ++i)
    {
        oneStep();
        std::cout << i << "/" << step << std::endl;
    }
}

void Simulator::oneStep()
{
    // compute the speedIncrement
    for(int i = 0; i < agent; ++i)
    {
        glm::vec3 speedInc;
        for(int j = 0; j < agent; ++j)
        {
            glm::vec3 direction = position[j] - position[i];
            float dist = glm::length(direction);

            // separation/alignment/cohesion
            if (dist < rs ) speedInc -= direction * ws;
            if (dist < ra ) speedInc += speed[j]  * wa;
            if (dist < rc ) speedInc += direction * wc;
        }
        speedIncrement[i] = speedInc;
    }

    // sum the speedIncrement to the speed
    for(int i = 0; i < agent; ++i)
    {
        speed[i] += speedIncrement[i];
    }

    // sum the speed to the position (Euler intégration)
    for(int i = 0; i < agent; ++i)
    {
        position[i] += speed[i];
    }
}
