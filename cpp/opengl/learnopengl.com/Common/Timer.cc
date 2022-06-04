#include "Timer.h"

using namespace std::chrono;

void Timer::reset() { start_ = high_resolution_clock::now(); }

double Timer::elapsed_nanoseconds() const {
  const auto l =
      duration_cast<nanoseconds>(high_resolution_clock::now() - start_).count();
  return static_cast<double>(l);
}

double Timer::elapsed_microseconds() const {
  const auto l =
      duration_cast<microseconds>(high_resolution_clock::now() - start_)
          .count();
  return static_cast<double>(l);
}

double Timer::elapsed_milliseconds() const {
  const auto l =
      duration_cast<milliseconds>(high_resolution_clock::now() - start_)
          .count();
  return static_cast<double>(l);
}

double Timer::elapsed_seconds() const {
  const auto l =
      duration_cast<seconds>(high_resolution_clock::now() - start_).count();
  return static_cast<double>(l);
}
