#ifndef __CELL__
#define __CELL__

class Simulator;
class Cell;
Cell *Simulator::getCell(int x, int y, int z);
void Simulator::dispatchBoid(glm::dvec3 pos, glm::dvec3 speed);

class Cell
{
	private:
        std::vector<glm::dvec3> position;
        std::vector<glm::dvec3> speed;
        std::vector<glm::dvec3> speedIncrement;

        std::vector<glm::dvec3> position;
        std::vector<glm::dvec3> speed;

		Simulator *sim;
		glm::ivec3 position;
		glm::dvec3 centerPosition;
	public:
		Cell(Simulator *sim, int x, int y, int z);
		void addBoid(glm::dvec3 pos);
		std::vector<glm::dvec3>::iterator getPositionIterator();
		std::vector<glm::dvec3>::iterator getSpeedIterator();

		void computeSpeedIncrement();
		void updateSpeed();
		void updatePosition();
		void dispatchBoids();
}

#endif
