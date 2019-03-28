#include <iostream>
#include <thread>

using namespace std;

void update(int& data) {
	data = 100;
}

int main() {
	int data = 0;
	thread t(update,std::ref(data));
	t.join();
	cout << data << endl;
	return 0;
}
