CXXFLAGS = -O0 -g -Wall
LDFLAGS = -lboost_system

$(BINARIES):
	g++ $(CXXFLAGS) -o $@ $(filter %.cc,$^) $(LDFLAGS)
clean:
	-rm -f $(BINARIES)
	
