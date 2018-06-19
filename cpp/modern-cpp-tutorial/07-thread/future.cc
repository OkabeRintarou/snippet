#include <future>
#include <iostream>
#include <thread>

int main() {
  std::packaged_task<int()> task([] { return 7; });
  std::future<int> result = task.get_future();
  std::thread(std::move(task)).detach();
  std::cout << "Waiting..." << std::endl;
  result.wait();
  std::cout << "Done!" << std::endl
            << "Result is " << result.get() << std::endl;
  return 0;
}
