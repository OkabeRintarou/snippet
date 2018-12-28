BINS := time
CXX := clang++
CXXFLAGS := -O3
LDFLAGS := -lsfml-system -lsfml-graphics

time:time.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

.PHONY:clean
clean:$(BINS)
	@rm -rf $(BINS)
