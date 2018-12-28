#include <iostream>
#include <SFML/Window.hpp>

int main() {
	sf::Window window(sf::VideoMode(800,600), "My window");

	window.setPosition(sf::Vector2i(10,50));
	window.setSize(sf::Vector2u(640,480));
	window.setTitle("SFML Window");
	sf::Vector2u size = window.getSize();
	unsigned width = size.x, height = size.y;
	std::cout << "width = " << width << ", height = " << height << std::endl;

	window.setVerticalSyncEnabled(true);
	window.setFramerateLimit(60);

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				window.close();
			} else if (event.type == sf::Event::Resized) {
				std::cout << "new width = " << event.size.width << ", new height = " << event.size.height << std::endl;
			} else if (event.type == sf::Event::LostFocus) {
				std::cout << "window lost focus" << std::endl;
			} else if (event.type == sf::Event::GainedFocus) {
				std::cout << "window gained focus" << std::endl;
			} else if (event.type == sf::Event::TextEntered) {
				if (event.text.unicode < 128) {
					std::cout << "ASCII character typed: " << static_cast<char>(event.text.unicode) << std::endl;
				}
			} else if (event.type == sf::Event::KeyPressed) {
				if (event.key.code == sf::Keyboard::Escape) {
					std::cout << "the escape key was pressed" << std::endl;
					std::cout << "control:" << event.key.control << std::endl;
					std::cout << "alt:" << event.key.alt << std::endl;
					std::cout << "shift:" << event.key.shift << std::endl;
					std::cout << "system:" << event.key.system << std::endl;
				}
			} else if (event.type == sf::Event::MouseWheelScrolled) {
				if (event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel) {
					std::cout << "wheel type: vertical" << std::endl;
				} else if (event.mouseWheelScroll.wheel == sf::Mouse::HorizontalWheel) {
					std::cout << "wheel type: horizontal" << std::endl;
				} else {
					std::cout << "wheel type: unknow" << std::endl;
				}

				std::cout << "wheel movement: " << event.mouseWheelScroll.delta << std::endl;
				std::cout << "mouse.x: " << event.mouseWheelScroll.x << std::endl;
				std::cout << "mouse.y: " << event.mouseWheelScroll.y << std::endl;
			} else if (event.type == sf::Event::MouseButtonPressed) {
				if (event.mouseButton.button == sf::Mouse::Right) {
					std::cout << "the right button was pressed" << std::endl;
					std::cout << "mouse x: " << event.mouseButton.x << std::endl;
					std::cout << "mouse y: " << event.mouseButton.y << std::endl;
				}
			} else if (event.type == sf::Event::MouseMoved) {
				std::cout << "new mouse x: " << event.mouseMove.x << std::endl;
				std::cout << "new mouse y: " << event.mouseMove.y << std::endl;
			} else if (event.type == sf::Event::MouseEntered) {
				std::cout << "the mouse cursor has entered the window" << std::endl;
			} else if (event.type == sf::Event::MouseLeft) {
				std::cout << "the mouse cursor has left the window" << std::endl;
			}
		}
	}
	return 0;
}
