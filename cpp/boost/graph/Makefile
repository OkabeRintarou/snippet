CXX = g++
CXXFLAGS = -Wall 

BINS = create_graph scan_graph undirected_graph in_edges visit_graph weighted_graph \
spanning_tree dfs

.PHONY:all

all:$(BINS)
create_graph:create_graph.cpp
	$(CXX) -o $@ $< $(CXXFLAGS)
scan_graph:scan_graph.cpp
	$(CXX) -o $@ $< $(CXXFLAGS)
undirected_graph:undirected_graph.cpp
	$(CXX) -o $@ $< $(CXXFLAGS)	
in_edges:in_edges.cpp
	$(CXX) -o $@ $< $(CXXFLAGS)
visit_graph:visit_graph.cpp
	$(CXX) -o $@ $< $(CXXFLAGS)
weighted_graph:weighted_graph.cpp
	$(CXX) -o $@ $< $(CXXFLAGS)
spanning_tree:spanning_tree.cpp
	$(CXX) -o $@ $< $(CXXFLAGS)
dfs:dfs.cpp
	$(CXX) -o $@ $< $(CXXFLAGS)
.PHONY:clean
clean:
	-rm $(BINS)
