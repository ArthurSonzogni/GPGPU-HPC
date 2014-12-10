#include "Simulator.hpp"
#include <cstdlib>
#include <iostream>
#include "ProgressBar.hpp"
#include <fstream>
#include <sstream>

double randDouble()
{ return rand() / double(RAND_MAX);
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
    ProgressBar progressBar;
    for(int i = 0; i < step; ++i)
    {
        oneStep();
        progressBar.update(i/float(step));
        
        // print the result
        std::stringstream filename;
        filename << "./output/boids_" << i << ".xyz";
        save(filename.str()); 
    }
}

void Simulator::oneStep()
{
    // compute the speedIncrement
    for(int i = 0; i < agent; ++i)
    {
        glm::vec3 speedInc(0.0);
        for(int j = 0; j < agent; ++j)
        {
            glm::vec3 direction = position[j] - position[i];
            float dist = glm::length(direction);

            // separation/alignment/cohesion
            if (dist < rs ) speedInc -= direction * ws;
            if (dist < ra ) speedInc += speed[j]  * wa;
            if (dist < rc ) speedInc += direction * wc;

            //speedInc -= position[i] * 0.0000001f;
        }
        speedIncrement[i] = speedInc * 0.1f;
    }

    // sum the speedIncrement to the speed
    for(int i = 0; i < agent; ++i)
    {
        speed[i] += speedIncrement[i];

        // limit the speed;
        const float maxSpeed = 100.f/500.f;
        float s = glm::length(speed[i]);
        if (s>maxSpeed)
            speed[i] *= maxSpeed/s;
    }

    // sum the speed to the position (Euler intégration)
    for(int i = 0; i < agent; ++i)
    {
        position[i] += speed[i];
    }
}

void Simulator::save(const std::string& filename)
{
    std::ofstream file;
    file.open(filename.c_str());

    for(int i = 0; i < agent; ++i)
    {
        file
            << position[i].x << " "
            << position[i].y << " "
            << position[i].z
            << std::endl;
    }

    file.close();
}
