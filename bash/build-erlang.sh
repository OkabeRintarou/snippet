#!/bin/bash

# install build tools
# sudo apt -y install libwxgtk3.0-gtk3-dev libwxgtk-webview3.0-gtk3-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev autoconf m4 libncurses5-dev libssh-dev unixodbc-dev
# git clone https://github.com/erlang/otp
# cd erlang-src-dir
./configure
make -j16
make install

# git clone https://github.com/elixir-lang/elixir
# cd elixir
# make clean test
