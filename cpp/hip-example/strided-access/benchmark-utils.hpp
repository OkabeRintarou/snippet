#pragma once

#include <sys/time.h>

class Timer {
public:
	Timer() : ts(0) {}

	void start() {
		struct timeval tv;
		gettimeofday(&tv, nullptr);
		ts = static_cast<double>(
				tv.tv_sec * 1000000 + tv.tv_usec);
	}

	double get() const {
		struct timeval tv;
		gettimeofday(&tv, nullptr);

		double end_time = static_cast<double>(
				tv.tv_sec * 1000000 + tv.tv_usec);
		return static_cast<double>(end_time - ts) / 1000000.0;
	}
private:
	double ts;
};
