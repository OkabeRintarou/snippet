BINS := window opengl
CXX := clang++
CXXFLAGS := -O3
LDFLAGS := -lsfml-system -lsfml-graphics -lsfml-window -lGL

all:$(BINS)
window:window.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

opengl:opengl.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

.PHONY:clean
clean:$(BINS)
	@rm -rf $(BINS)
