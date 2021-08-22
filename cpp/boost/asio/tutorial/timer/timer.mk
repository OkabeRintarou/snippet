CXXFLAGS = -O0 -g -Wall
LDFLAGS = -lboost_system -lpthread -lboost_thread

$(BINARIES):
	g++ $(CXXFLAGS) -o $@ $(filter %.cc,$^) $(LDFLAGS)
clean:
	-rm -f $(BINARIES)
	
