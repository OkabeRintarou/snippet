#include <iostream>
#include <boost/graph/adjacency_list.hpp>

using namespace std;
using namespace boost;

typedef boost::adjacency_list<listS,vecS,bidirectionalS> mygraph;

int main()
{
	mygraph g;
	add_edge(0,1,g);
	add_edge(0,3,g);
	add_edge(1,2,g);
	add_edge(2,3,g);
	add_edge(3,1,g);
	add_edge(1,3,g);
	cout << "Number of edges: " << num_edges(g) << endl;
	cout << "Number of vertices: " << num_vertices(g) << endl;
	mygraph::vertex_iterator vertexIt,vertexEnd;
	tie(vertexIt,vertexEnd) = vertices(g);
	for(;vertexIt != vertexEnd;++vertexIt){
		cout << "in-degree for " << *vertexIt << ": " << in_degree(*vertexIt,g) << '\n';
		cout << "out-degree for " << *vertexIt << ": " << out_degree(*vertexIt,g) << '\n';
	}
	mygraph::edge_iterator edgeIt,edgeEnd;
	tie(edgeIt,edgeEnd) = edges(g);
	for(;edgeIt != edgeEnd;++edgeIt){
		cout << "edge " << source(*edgeIt,g) << " --> " << target(*edgeIt,g) << '\n';
	}
	return 0;
}
