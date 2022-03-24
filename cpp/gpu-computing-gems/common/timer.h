#pragma once

#include <chrono>

class Timer {
public:
  void reset();

  [[nodiscard]] double elapsed_nanoseconds() const;
  [[nodiscard]] double elapsed_microseconds() const;
  [[nodiscard]] double elapsed_milliseconds() const;
  [[nodiscard]] double elapsed_seconds() const;
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};
