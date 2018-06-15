#include <vector>
#include <iostream>
#include <algorithm>

int main() {
	std::vector<int> vec = {1,2,3,4};
	if (const std::vector<int>::iterator itr = std::find(vec.begin(),vec.end(),2);
			itr != vec.end()) {
		*itr = 3;
	}
	return 0;
}
