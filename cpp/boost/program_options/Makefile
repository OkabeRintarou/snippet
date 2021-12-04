BINS=first_example
all:$(BINS)

first_example:first_example.cpp
	g++ -std=c++17 -o $@ $< -lboost_program_options

.PHONY:clean
clean:
	@rm $(BINS)
