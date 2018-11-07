TARS=stack1 stack2 stackpartspec stack3 

all:$(TARS)

stack1:stack1.cpp
	g++ -std=c++11 -o $@ $<
stack2:stack2.cpp stack2.hpp
	g++ -std=c++11 -o $@ $<
stackpartspec:stackpartspec.cpp
	g++ -std=c++11 -o $@ $<
stack3:stack3.cpp
	g++ -std=c++11 -o $@ $<

.PHONY:clean
clean:
	-rm $(TARS)
