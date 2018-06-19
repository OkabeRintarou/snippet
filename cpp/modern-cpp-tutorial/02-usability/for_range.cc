#include <iostream>
#include <vector>
#include <algorithm>

int main() {
	std::vector<int> vec {1,2,3,4};
	if (auto it = find(vec.begin(),vec.end(),3);it != vec.end()) {
		*it = 4;
	}
	for (auto &element : vec) {
		element += 1;
	}
	for (auto element : vec) {
		std::cout << element << " ";
	}
	return 0;
}
