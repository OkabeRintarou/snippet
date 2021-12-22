#!/usr/bin/env bash

if [ "$(id -u)" != "0" ]; then
	echo "Run the script as root"
	exit 1
fi

function install_glfw()
{
	echo "install glfw..."
	rm -rf ./glfw
	git clone "https://github.com/glfw/glfw"
	cd ./glfw
	git checkout tags/3.3.6 -b v3.3.6
	apt -y install libwayland-dev libxkbcommon-dev wayland-protocols extra-cmake-modules xorg-dev
	mkdir build && cd build
	cmake -DCMAKE_BUILD_TYPE=Release ..
	make -j16
	make install
	rm -rf ./glfw
	echo "Done"
}

if [ ! -d "/usr/local/include/GLFW" ]; then
	install_glfw
fi

function install_glew()
{
	echo "install glew..."
	rm -rf ./glew-2.1.0
	wget https://sourceforge.net/projects/glew/files/glew/2.1.0/glew-2.1.0.zip/download -O glew-2.1.0.zip
	unzip glew-2.1.0.zip && cd glew-2.1.0
	apt install -y libxmu-dev libxi-dev libgl-dev
	cd glew-2.1.0 && make -j16
	make install
	cd ..
	rm -rf glew-2.1.0.zip glew-2.1.0
	echo "Done"
}

if [ ! -d "/usr/local/include/GLEW" ]; then
	install_glew
fi
