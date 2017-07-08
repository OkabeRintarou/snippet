#include <iostream>
#include <list>
#include <iterator>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

using namespace std;
using namespace boost;

typedef property<edge_weight_t,int> EdgeWeightProperty;
typedef adjacency_list<listS,vecS,directedS,no_property,EdgeWeightProperty> mygraph;
typedef mygraph::edge_descriptor Edge;
	
int main()
{
	mygraph g;
	add_edge (0, 1, 8, g);
  	add_edge (0, 3, 18, g);
  	add_edge (1, 2, 20, g);
  	add_edge (2, 3, 2, g);
  	add_edge (3, 1, 1, g);
	list<Edge> spanning_tree;
	kruskal_minimum_spanning_tree(g,std::back_inserter(spanning_tree));
	for(list<Edge>::iterator it = spanning_tree.begin();
		it != spanning_tree.end();++it){
		cout << *it << "  ";
	}
	cout << '\n';
	return 0;
}
