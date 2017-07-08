#include <iostream>
#include <boost/graph/adjacency_list.hpp>

using namespace std;
using namespace boost;

typedef boost::adjacency_list<listS,vecS,undirectedS> mygraph;

int main()
{
	mygraph g;
	add_edge(0,1,g);
	add_edge(0,3,g);
	add_edge(1,2,g);
	add_edge(2,3,g);

	mygraph::vertex_iterator vertexIt,vertexEnd;
	mygraph::adjacency_iterator neighborIt,neighborEnd;
	tie(vertexIt,vertexEnd) = vertices(g);
	for(;vertexIt != vertexEnd;++vertexIt){
		cout << *vertexIt << " is connected to ";
		tie(neighborIt,neighborEnd) = adjacent_vertices(*vertexIt,g);
		for(;neighborIt != neighborEnd;++neighborIt){
			cout << *neighborIt << " ";
		}
		cout << endl;
	}
	return 0;
}

