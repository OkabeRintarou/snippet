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
	add_edge(3,2,g);
	mygraph::vertex_iterator vertexIt,vertexEnd;
	mygraph::in_edge_iterator inedgeIt,inedgeEnd;
	mygraph::out_edge_iterator outedgeIt,outedgeEnd;
	tie(vertexIt,vertexEnd) = vertices(g);
	for(;vertexIt != vertexEnd;++vertexIt){
		cout << "incoming edges for " << *vertexIt << ": ";
		tie(inedgeIt,inedgeEnd) = in_edges(*vertexIt,g);
		for(;inedgeIt != inedgeEnd;++inedgeIt){
			cout << *inedgeIt << "  ";
		}
		cout << '\n';
	}
	tie(vertexIt,vertexEnd) = vertices(g);
	for(;vertexIt != vertexEnd;++vertexIt){
		cout << "outcoming edges for " << *vertexIt << ": ";
		tie(outedgeIt,outedgeEnd) = out_edges(*vertexIt,g);
		for(;outedgeIt != outedgeEnd;++outedgeIt){
			cout << *outedgeIt << "  ";
		}
		cout << '\n';
	}
	
	return 0;
}
