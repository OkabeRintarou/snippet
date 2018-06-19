#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

int main() {
  std::queue<int> produced_nums;
  std::mutex mtx;
  std::condition_variable cond_var;
  bool done = false;
  bool notified = false;

  std::thread producer([&]() {
    for (int i = 0; i < 5; i++) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      std::unique_lock<std::mutex> lock(mtx);
      std::cout << "producing " << i << std::endl;
      produced_nums.push(i);
      notified = true;
      cond_var.notify_one();
    }
    done = true;
    notified = true;
    cond_var.notify_one();
  });

  std::thread consumer([&]() {
    std::unique_lock<std::mutex> lock(mtx);
    while (!done) {
      while (!notified) {
        cond_var.wait(lock);
      }
      while (!produced_nums.empty()) {
        std::cout << "consuming " << produced_nums.front() << std::endl;
        produced_nums.pop();
      }
      notified = false;
    }
  });

  producer.join();
  consumer.join();
  return 0;
}
