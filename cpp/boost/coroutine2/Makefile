OBJS=push_co2 pull_co2
all: $(OBJS)

push_co2:push_co2.cc
	g++ -std=c++20 -o $@ $< -lboost_context

pull_co2:pull_co2.cc
	g++ -std=c++20 -o $@ $< -lboost_context

.PHONY:clean
clean:
	@rm $(OBJS)
