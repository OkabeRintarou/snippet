ts=arguments exception test_thread hardware_concurrency accumulate
all: $(ts)

exception:exception.cc
	g++ -std=c++11 -o $@ $< -lpthread

arguments:arguments.cc
	g++ -std=c++11 -o $@ $< -lpthread

test_thread: test_thread.cc
	g++ -std=c++11 -o $@ $< -lpthread

hardware_concurrency:hardware_concurrency.cc
	g++ -std=c++11 -o $@ $< -lpthread

accumulate:accumulate.cc
	g++ -std=c++11 -o $@ $< -lpthread

.PHONY:clean
clean:
	rm -rf $(ts)
