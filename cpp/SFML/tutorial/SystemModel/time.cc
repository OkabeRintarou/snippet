#include <unistd.h>
#include <iostream>
#include <SFML/System.hpp>

int main() {
	{
		sf::Time t1 = sf::microseconds(10000);
		sf::Time t2 = sf::milliseconds(10);
		sf::Time t3 = sf::seconds(0.01);

		sf::Int64 usec = t1.asMicroseconds();
		sf::Int32 msec = t2.asMilliseconds();
		float sec = t3.asSeconds();
		
		// Playing with time values
		sf::Time t4 = t1 * sf::Int64(2);
		sf::Time t5 = t1 + t2;
		sf::Time t6 = -t3;

		// Measuring time
		sf::Clock clock;
		std::cout << "sleeping..." << std::endl;
		sleep(2); // sleep 2 seconds
		sf::Time elapsed1 = clock.getElapsedTime();
		std::cout << elapsed1.asSeconds() << std::endl;
		clock.restart();
		std::cout << "sleeping..." << std::endl;
		sleep(1);
		sf::Time elapsed2 = clock.getElapsedTime();
		std::cout << elapsed2.asSeconds() << std::endl;
				
	}
	return 0;
}
