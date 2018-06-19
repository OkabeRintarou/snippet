#include <iostream>
#include <mutex>
#include <thread>

std::mutex mtx;

void block_area() {
  std::unique_lock<std::mutex> lock(mtx);
  std::cout << "Only one thread can enter here at the same time" << std::endl;
  lock.unlock();
}

int main() {
  std::thread t1(block_area);
  std::thread t2(block_area);
  t1.join();
  t2.join();
  return 0;
}
