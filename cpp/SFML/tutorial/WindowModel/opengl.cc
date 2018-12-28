#include <iostream>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>

int main() {
	sf::ContextSettings settings;
	settings.depthBits = 24;
	settings.stencilBits = 8;
	settings.antialiasingLevel = 4;
	settings.majorVersion = 3;
	settings.minorVersion = 0;
	sf::Window window(sf::VideoMode(800,600), "OpenGL", sf::Style::Default, settings);


	settings = window.getSettings();

	std::cout << "depth bits: " << settings.depthBits << std::endl;
	std::cout << "stencil bits: " << settings.stencilBits << std::endl;
	std::cout << "antialiasing level: " << settings.antialiasingLevel << std::endl;
	std::cout << "major version: " << settings.majorVersion << std::endl;
	std::cout << "minor version: " << settings.minorVersion << std::endl;

	window.setVerticalSyncEnabled(true);

	// active the window
	window.setActive(true);

	bool running = true;
	while (running) {
		// handle events
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				running = false;
			} else if (event.type == sf::Event::Resized) {
				glViewport(0, 0, event.size.width, event.size.height);
			}
		}

		// clear the buffers
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// draw
		// ...

		// end the current frame (internally swaps the front and back buffers)
		window.display();

	}
	return 0;
}
