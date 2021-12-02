all:server client

server:server.cpp
	g++ -std=c++17 -O2 -o server server.cpp -lboost_system -lpthread

client:client.cpp
	g++ -std=c++17 -O2 -o client client.cpp -lboost_system -lpthread

.PHONY:clean
clean:
	@rm server client
