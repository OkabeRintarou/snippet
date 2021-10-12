#pragma once

#include <vector>

namespace amdgpu {


class Devices : private std::vector<int> {
private:
	using self = std::vector<int>;
public:
	using self::empty;
	using self::size;
	using self::operator[];

	void add(int fd) {
		push_back(fd);
	}

	Devices() = default;
	Devices(const Devices&) = delete;
	Devices(Devices &&) = delete;
	Devices& operator=(const Devices&) = delete;
	Devices& operator=(Devices &&) = delete;

	~Devices();
};

bool open_devices(Devices &devices, bool open_render_node);

}
